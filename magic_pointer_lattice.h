/*
 * magic_pointer_lattice.h — Unified Magic Pointer for MPS + PEPS 2D–6D
 *
 * ┌──────────────────────────────────────────────────────────────┐
 * │  Any lattice topology → MagicPointer → O(N+E) amplitudes    │
 * │                                                              │
 * │  1D chain  (MPS replacement)     N sites, N-1 edges         │
 * │  2D grid   (PEPS 2D)            Lx×Ly sites, 2D adjacency   │
 * │  3D cubic  (TNS 3D)             Lx×Ly×Lz, 3D adjacency      │
 * │  4D hyper  (TNS 4D)             4D adjacency                 │
 * │  5D        (TNS 5D)             5D adjacency                 │
 * │  6D        (TNS 6D)             6D adjacency — D=6 in 6D    │
 * └──────────────────────────────────────────────────────────────┘
 *
 * The lattice constructor auto-generates the CZ edge graph from
 * the grid topology. Then mp_amplitude() extrapolates any amplitude
 * of the full D^N state in O(N+E).
 */

#ifndef MAGIC_POINTER_LATTICE_H
#define MAGIC_POINTER_LATTICE_H

#include "magic_pointer.h"

/* ═══════════════════════════════════════════════════════════════════════
 * TOPOLOGY
 * ═══════════════════════════════════════════════════════════════════════ */

typedef enum {
    MPL_1D = 1,
    MPL_2D = 2,
    MPL_3D = 3,
    MPL_4D = 4,
    MPL_5D = 5,
    MPL_6D = 6
} MPLTopology;

/* ═══════════════════════════════════════════════════════════════════════
 * LATTICE DESCRIPTOR
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    MPLTopology topology;
    int         dims[6];       /* Lx, Ly, Lz, Lw, Lv, Lu (0 if unused) */
    int         n_dims;        /* 1..6                                  */
    uint64_t    total_sites;   /* Product of all dims                   */
    uint64_t    total_edges;   /* Number of nearest-neighbor edges      */
    MagicPointer *mp;          /* The underlying magic pointer          */
} MagicPointerLattice;

/* ═══════════════════════════════════════════════════════════════════════
 * COORDINATE ↔ FLAT INDEX
 *
 * Row-major: flat = x + Lx*(y + Ly*(z + Lz*(w + Lw*(v + Lv*u))))
 * ═══════════════════════════════════════════════════════════════════════ */

static inline uint64_t mpl_flat(const MagicPointerLattice *mpl,
                                int x, int y, int z, int w, int v, int u)
{
    uint64_t idx = x;
    if (mpl->n_dims >= 2) idx += (uint64_t)mpl->dims[0] * y;
    if (mpl->n_dims >= 3) idx += (uint64_t)mpl->dims[0] * mpl->dims[1] * z;
    if (mpl->n_dims >= 4) idx += (uint64_t)mpl->dims[0] * mpl->dims[1] * mpl->dims[2] * w;
    if (mpl->n_dims >= 5) idx += (uint64_t)mpl->dims[0] * mpl->dims[1] * mpl->dims[2] * mpl->dims[3] * v;
    if (mpl->n_dims >= 6) idx += (uint64_t)mpl->dims[0] * mpl->dims[1] * mpl->dims[2] * mpl->dims[3] * mpl->dims[4] * u;
    return idx;
}

static inline void mpl_unflat(const MagicPointerLattice *mpl, uint64_t flat,
                               int *x, int *y, int *z, int *w, int *v, int *u)
{
    *x = *y = *z = *w = *v = *u = 0;
    *x = flat % mpl->dims[0]; flat /= mpl->dims[0];
    if (mpl->n_dims >= 2) { *y = flat % mpl->dims[1]; flat /= mpl->dims[1]; }
    if (mpl->n_dims >= 3) { *z = flat % mpl->dims[2]; flat /= mpl->dims[2]; }
    if (mpl->n_dims >= 4) { *w = flat % mpl->dims[3]; flat /= mpl->dims[3]; }
    if (mpl->n_dims >= 5) { *v = flat % mpl->dims[4]; flat /= mpl->dims[4]; }
    if (mpl->n_dims >= 6) { *u = flat % mpl->dims[5]; }
}

/* ═══════════════════════════════════════════════════════════════════════
 * INTERNAL: Generate adjacency edges for an n-dimensional grid
 *
 * For each dimension d, connect site at coord[d]=i to coord[d]=i+1
 * along axis d (nearest-neighbor).
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void mpl_generate_edges(MagicPointerLattice *mpl)
{
    int L[6];
    for (int d = 0; d < 6; d++) L[d] = (d < mpl->n_dims) ? mpl->dims[d] : 1;

    /* For each axis, connect adjacent sites */
    for (int axis = 0; axis < mpl->n_dims; axis++) {
        /* Iterate over all sites */
        for (int u = 0; u < L[5]; u++)
        for (int v = 0; v < L[4]; v++)
        for (int w = 0; w < L[3]; w++)
        for (int z = 0; z < L[2]; z++)
        for (int y = 0; y < L[1]; y++)
        for (int x = 0; x < L[0]; x++) {
            int coords[6] = {x, y, z, w, v, u};
            /* Only connect if there's a neighbor in the +axis direction */
            if (coords[axis] + 1 < L[axis]) {
                uint64_t flat_a = mpl_flat(mpl, x, y, z, w, v, u);
                int coords_b[6] = {x, y, z, w, v, u};
                coords_b[axis]++;
                uint64_t flat_b = mpl_flat(mpl, coords_b[0], coords_b[1],
                                           coords_b[2], coords_b[3],
                                           coords_b[4], coords_b[5]);
                mp_cz(mpl->mp, flat_a, flat_b);
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * INTERNAL: Compute expected edge count for validation
 * ═══════════════════════════════════════════════════════════════════════ */

static inline uint64_t mpl_expected_edges(const int *dims, int n_dims)
{
    /* For each axis d: (L_d - 1) × Π_{other axes} L_i */
    uint64_t total = 0;
    for (int d = 0; d < n_dims; d++) {
        uint64_t edges_this_axis = dims[d] - 1;
        for (int i = 0; i < n_dims; i++) {
            if (i != d) edges_this_axis *= dims[i];
        }
        total += edges_this_axis;
    }
    return total;
}

/* ═══════════════════════════════════════════════════════════════════════
 * GENERIC CONSTRUCTOR
 * ═══════════════════════════════════════════════════════════════════════ */

static inline MagicPointerLattice *mpl_create(int n_dims, const int *dims)
{
    MagicPointerLattice *mpl = (MagicPointerLattice *)calloc(1, sizeof(MagicPointerLattice));
    if (!mpl) return NULL;

    mpl->n_dims = n_dims;
    mpl->topology = (MPLTopology)n_dims;
    mpl->total_sites = 1;
    for (int d = 0; d < 6; d++) {
        mpl->dims[d] = (d < n_dims) ? dims[d] : 1;
        if (d < n_dims) mpl->total_sites *= dims[d];
    }

    mpl->total_edges = mpl_expected_edges(mpl->dims, n_dims);
    mpl->mp = mp_create(mpl->total_sites);
    if (!mpl->mp) { free(mpl); return NULL; }

    /* Auto-generate CZ edges from grid adjacency */
    mpl_generate_edges(mpl);

    return mpl;
}

/* ═══════════════════════════════════════════════════════════════════════
 * DIMENSION-SPECIFIC CONSTRUCTORS
 * ═══════════════════════════════════════════════════════════════════════ */

/* 1D chain — replaces MPS */
static inline MagicPointerLattice *mpl_create_1d(int Lx)
{
    int dims[1] = {Lx};
    return mpl_create(1, dims);
}

/* 2D grid — replaces PEPS */
static inline MagicPointerLattice *mpl_create_2d(int Lx, int Ly)
{
    int dims[2] = {Lx, Ly};
    return mpl_create(2, dims);
}

/* 3D cubic */
static inline MagicPointerLattice *mpl_create_3d(int Lx, int Ly, int Lz)
{
    int dims[3] = {Lx, Ly, Lz};
    return mpl_create(3, dims);
}

/* 4D hypercubic */
static inline MagicPointerLattice *mpl_create_4d(int Lx, int Ly, int Lz, int Lw)
{
    int dims[4] = {Lx, Ly, Lz, Lw};
    return mpl_create(4, dims);
}

/* 5D */
static inline MagicPointerLattice *mpl_create_5d(int Lx, int Ly, int Lz,
                                                  int Lw, int Lv)
{
    int dims[5] = {Lx, Ly, Lz, Lw, Lv};
    return mpl_create(5, dims);
}

/* 6D — D=6 in 6 spatial dimensions */
static inline MagicPointerLattice *mpl_create_6d(int Lx, int Ly, int Lz,
                                                  int Lw, int Lv, int Lu)
{
    int dims[6] = {Lx, Ly, Lz, Lw, Lv, Lu};
    return mpl_create(6, dims);
}

/* ═══════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void mpl_destroy(MagicPointerLattice *mpl)
{
    if (!mpl) return;
    mp_destroy(mpl->mp);
    free(mpl);
}

/* ═══════════════════════════════════════════════════════════════════════
 * STATE INITIALIZATION
 * ═══════════════════════════════════════════════════════════════════════ */

/* Set site at coordinates to a specific state */
static inline void mpl_set_local(MagicPointerLattice *mpl,
                                  int x, int y, int z, int w, int v, int u,
                                  const double re[6], const double im[6])
{
    uint64_t flat = mpl_flat(mpl, x, y, z, w, v, u);
    mp_set_local(mpl->mp, flat, re, im);
}

/* Set ALL sites to |+⟩ = (1/√6) Σ|k⟩ */
static inline void mpl_set_all_plus(MagicPointerLattice *mpl)
{
    double plus_re[6], plus_im[6];
    double inv_sqrt6 = 1.0 / sqrt(6.0);
    for (int i = 0; i < 6; i++) { plus_re[i] = inv_sqrt6; plus_im[i] = 0; }
    for (uint64_t s = 0; s < mpl->total_sites; s++)
        mp_set_local(mpl->mp, s, plus_re, plus_im);
}

/* Set ALL sites to basis state |k⟩ */
static inline void mpl_set_all_basis(MagicPointerLattice *mpl, int k)
{
    double re[6] = {0}, im[6] = {0};
    re[k] = 1.0;
    for (uint64_t s = 0; s < mpl->total_sites; s++)
        mp_set_local(mpl->mp, s, re, im);
}

/* ═══════════════════════════════════════════════════════════════════════
 * GATE APPLICATION
 * ═══════════════════════════════════════════════════════════════════════ */

/* DFT on one site */
static inline void mpl_dft(MagicPointerLattice *mpl,
                            int x, int y, int z, int w, int v, int u)
{
    mp_dft(mpl->mp, mpl_flat(mpl, x, y, z, w, v, u));
}

/* DFT on ALL sites */
static inline void mpl_dft_all(MagicPointerLattice *mpl)
{
    for (uint64_t s = 0; s < mpl->total_sites; s++)
        mp_dft(mpl->mp, s);
}

/* Additional CZ between two coordinate-addressed sites */
static inline void mpl_cz(MagicPointerLattice *mpl,
                           int x1, int y1, int z1, int w1, int v1, int u1,
                           int x2, int y2, int z2, int w2, int v2, int u2)
{
    mp_cz(mpl->mp, mpl_flat(mpl, x1, y1, z1, w1, v1, u1),
                    mpl_flat(mpl, x2, y2, z2, w2, v2, u2));
}

/* ═══════════════════════════════════════════════════════════════════════
 * TROTTER STEP — Additional CZ sweep across all lattice edges
 *
 * Adds another layer of CZ entanglement across all nearest-neighbor
 * pairs. Each call adds total_edges more edges to the phase graph.
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void mpl_trotter_cz_sweep(MagicPointerLattice *mpl)
{
    mpl_generate_edges(mpl);
}

/* ═══════════════════════════════════════════════════════════════════════
 * THE MAGIC: LATTICE AMPLITUDE — delegates to mp_amplitude
 *
 * indices[total_sites] — one value per lattice site, flat-indexed
 * Returns ψ(i₁,...,iₙ) in O(N+E)
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void mpl_amplitude(const MagicPointerLattice *mpl,
                                  const uint32_t *indices,
                                  double *out_re, double *out_im)
{
    mp_amplitude(mpl->mp, indices, out_re, out_im);
}

static inline double mpl_probability(const MagicPointerLattice *mpl,
                                      const uint32_t *indices)
{
    return mp_probability(mpl->mp, indices);
}

/* ═══════════════════════════════════════════════════════════════════════
 * MEASUREMENT
 * ═══════════════════════════════════════════════════════════════════════ */

static inline uint32_t mpl_measure(MagicPointerLattice *mpl,
                                    int x, int y, int z, int w, int v, int u,
                                    double random_01)
{
    return mp_measure(mpl->mp, mpl_flat(mpl, x, y, z, w, v, u), random_01);
}

/* ═══════════════════════════════════════════════════════════════════════
 * ENTROPY — across an axis cut
 *
 * Cut along axis d at position p: sites with coord[d] ≤ p are "left"
 * ═══════════════════════════════════════════════════════════════════════ */

static inline double mpl_entropy_cut_axis(const MagicPointerLattice *mpl,
                                           int axis, int cut_pos)
{
    /* Count CZ edges crossing this cut */
    uint64_t crossing = 0;
    int L[6];
    for (int d = 0; d < 6; d++) L[d] = (d < mpl->n_dims) ? mpl->dims[d] : 1;

    for (uint64_t e = 0; e < mpl->mp->n_edges; e++) {
        uint64_t sa = mpl->mp->edges[e].site_a;
        uint64_t sb = mpl->mp->edges[e].site_b;

        /* Extract coordinate along the cut axis */
        int xa, ya, za, wa, va, ua, xb, yb, zb, wb, vb, ub;
        mpl_unflat(mpl, sa, &xa, &ya, &za, &wa, &va, &ua);
        mpl_unflat(mpl, sb, &xb, &yb, &zb, &wb, &vb, &ub);

        int coords_a[6] = {xa, ya, za, wa, va, ua};
        int coords_b[6] = {xb, yb, zb, wb, vb, ub};

        int ca = coords_a[axis], cb = coords_b[axis];
        if ((ca <= cut_pos && cb > cut_pos) ||
            (cb <= cut_pos && ca > cut_pos)) {
            crossing++;
        }
    }

    return crossing * log2(6.0);
}

/* ═══════════════════════════════════════════════════════════════════════
 * EXOTIC INVARIANT — average Δ across all lattice sites
 * ═══════════════════════════════════════════════════════════════════════ */

static inline double mpl_exotic_invariant(MagicPointerLattice *mpl)
{
    return mp_exotic_invariant(mpl->mp);
}

/* ═══════════════════════════════════════════════════════════════════════
 * NORMALIZATION (brute force — small lattices only)
 * ═══════════════════════════════════════════════════════════════════════ */

static inline double mpl_norm_sq(const MagicPointerLattice *mpl)
{
    return mp_norm_sq(mpl->mp);
}

/* ═══════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void mpl_print_info(const MagicPointerLattice *mpl)
{
    printf("╔════════════════════════════════════════════════════╗\n");
    printf("║  Magic Pointer Lattice — %dD                       ║\n", mpl->n_dims);
    printf("╠════════════════════════════════════════════════════╣\n");
    printf("║  Topology:    %dD", mpl->n_dims);
    for (int d = 0; d < mpl->n_dims; d++) printf(" L%c=%d", "xyzwvu"[d], mpl->dims[d]);
    printf("\n");
    printf("║  Sites:       %lu\n", mpl->total_sites);
    printf("║  CZ edges:    %lu (expected: %lu)\n",
           mpl->mp->n_edges, mpl->total_edges);
    printf("║  Full SV:     6^%lu ≈ 10^%.0f entries\n",
           mpl->total_sites, mpl->total_sites * log10(6.0));

    uint64_t mem = mpl->total_sites * sizeof(TrialityQuhit) +
                   mpl->mp->n_edges * sizeof(MPEdge) +
                   sizeof(MagicPointerLattice) + sizeof(MagicPointer);
    if (mem < 1024*1024)
        printf("║  Memory:      %.1f KB\n", mem / 1024.0);
    else
        printf("║  Memory:      %.1f MB\n", mem / (1024.0*1024.0));
    printf("╚════════════════════════════════════════════════════╝\n");
}

#endif /* MAGIC_POINTER_LATTICE_H */
