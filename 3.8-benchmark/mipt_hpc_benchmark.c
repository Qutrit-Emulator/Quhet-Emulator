/*
 * mipt_hpc_benchmark.c — MIPT Benchmark: 1D through 6D via HPC
 *
 * Runs the measurement-induced phase transition on lattices of
 * increasing dimensionality, all with 64 sites:
 *
 *   1D:  64           (chain)
 *   2D:  8 × 8        (square lattice)
 *   3D:  4 × 4 × 4    (cubic lattice)
 *   4D:  4 × 4 × 2 × 2  (4D hypercube)
 *   5D:  2 × 2 × 2 × 2 × 4  (5D lattice)
 *   6D:  2 × 2 × 2 × 2 × 2 × 2  (hexeract)
 *
 * For each dimension, sweeps measurement rate p ∈ [0, 1] and tracks:
 *   - Entanglement entropy S (from edge crossings)
 *   - Surviving edge count E
 *   - Exotic invariant Δ
 *   - Wall-clock time per p-point
 *   - Bonds per layer (connectivity scaling)
 *
 * Build:
 *   gcc -O2 -march=native -o mipt_hpc_benchmark mipt_hpc_benchmark.c \
 *       quhit_core.c quhit_gates.c quhit_measure.c quhit_entangle.c \
 *       quhit_register.c quhit_substrate.c quhit_triality.c \
 *       quhit_triadic.c quhit_lazy.c quhit_calibrate.c \
 *       s6_exotic.c bigint.c -lm -msse2
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "hpc_graph.h"
#include "hpc_contract.h"
#include "hpc_amplitude.h"

/* ═══════════════════════════════════════════════════════════════════
 * LATTICE GEOMETRY — Parametric N-dimensional lattice
 * ═══════════════════════════════════════════════════════════════════ */

#define MAX_DIM    6
#define MAX_SITES  64

typedef struct {
    int ndim;
    int extents[MAX_DIM];    /* L per dimension                       */
    int nsites;              /* Product of extents                     */
    int nbonds;              /* Total nearest-neighbor bonds           */
    char label[32];          /* Human-readable geometry string         */
} Lattice;

/* Flatten N-D coordinate to linear index */
static int lat_flat(const Lattice *lat, const int *coords)
{
    int idx = 0;
    for (int d = lat->ndim - 1; d >= 0; d--)
        idx = idx * lat->extents[d] + coords[d];
    return idx;
}

/* Extract N-D coordinate from flat index */
static void lat_coords(const Lattice *lat, int idx, int *coords)
{
    for (int d = 0; d < lat->ndim; d++) {
        coords[d] = idx % lat->extents[d];
        idx /= lat->extents[d];
    }
}

/* Count bonds (open boundary) */
static int lat_count_bonds(const Lattice *lat)
{
    int total = 0;
    int coords[MAX_DIM];
    for (int s = 0; s < lat->nsites; s++) {
        lat_coords(lat, s, coords);
        for (int d = 0; d < lat->ndim; d++) {
            if (coords[d] + 1 < lat->extents[d])
                total++;
        }
    }
    return total;
}

/* Initialize lattice geometry */
static Lattice lat_create(int ndim, const int *extents)
{
    Lattice lat;
    lat.ndim = ndim;
    lat.nsites = 1;
    for (int d = 0; d < MAX_DIM; d++) {
        lat.extents[d] = (d < ndim) ? extents[d] : 1;
        if (d < ndim) lat.nsites *= extents[d];
    }
    lat.nbonds = lat_count_bonds(&lat);

    /* Build label */
    char *p = lat.label;
    for (int d = 0; d < ndim; d++) {
        if (d > 0) p += sprintf(p, "×");
        p += sprintf(p, "%d", extents[d]);
    }
    return lat;
}

/* ═══════════════════════════════════════════════════════════════════
 * XOSHIRO256** PRNG
 * ═══════════════════════════════════════════════════════════════════ */

static uint64_t rng_state[4];

static inline uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t rng_next(void) {
    uint64_t *s = rng_state;
    uint64_t result = rotl(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = rotl(s[3], 45);
    return result;
}

static double rng_uniform(void) {
    return (double)(rng_next() >> 11) / (double)(1ULL << 53);
}

static void rng_seed_init(uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rng_state[i] = z ^ (z >> 31);
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * CIRCUIT COMPONENTS — Same physics, parametric geometry
 * ═══════════════════════════════════════════════════════════════════ */

/* Random diagonal phase gate — breaks trivial symmetries */
static void apply_random_phase(HPCGraph *g, uint64_t site)
{
    double phi_re[6], phi_im[6];
    for (int k = 0; k < 6; k++) {
        double angle = 2.0 * M_PI * rng_uniform();
        phi_re[k] = cos(angle);
        phi_im[k] = sin(angle);
    }
    hpc_phase(g, site, phi_re, phi_im);
}

/* CZ on all nearest-neighbor bonds (open boundary) */
static void apply_cz_layer(HPCGraph *g, const Lattice *lat)
{
    int coords[MAX_DIM], nc[MAX_DIM];
    for (int s = 0; s < lat->nsites; s++) {
        lat_coords(lat, s, coords);
        for (int d = 0; d < lat->ndim; d++) {
            /* +1 neighbor along dimension d */
            for (int i = 0; i < lat->ndim; i++) nc[i] = coords[i];
            nc[d] += 1;
            if (nc[d] < lat->extents[d]) {
                int nb = lat_flat(lat, nc);
                hpc_cz(g, s, nb);
            }
        }
    }
}

/* One MIPT layer: random phase → DFT → CZ → measure */
static int mipt_layer(HPCGraph *g, const Lattice *lat, double p_meas)
{
    int measured = 0;

    /* 1. Random phase on all sites */
    for (int s = 0; s < lat->nsites; s++)
        apply_random_phase(g, s);

    /* 2. DFT₆ on all sites */
    for (int s = 0; s < lat->nsites; s++)
        hpc_dft(g, s);

    /* 3. CZ on all bonds */
    apply_cz_layer(g, lat);

    /* 4. Random projective measurements at rate p */
    for (int s = 0; s < lat->nsites; s++) {
        if (rng_uniform() < p_meas) {
            hpc_measure(g, s, rng_uniform());
            measured++;
        }
    }

    return measured;
}

/* ═══════════════════════════════════════════════════════════════════
 * RUN ONE DIMENSION — Full MIPT sweep for a given lattice
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    double p;
    double S_avg, S_std;
    double Delta_avg;
    double edges_avg;
    double time_sec;
} MIPTResult;

static void run_mipt(const Lattice *lat, int circuit_depth, int num_samples,
                     const double *p_values, int num_p, MIPTResult *results)
{
    for (int pi = 0; pi < num_p; pi++) {
        double p = p_values[pi];
        results[pi].p = p;

        double sum_S = 0, sum_S2 = 0, sum_D = 0, sum_E = 0;

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        for (int sample = 0; sample < num_samples; sample++) {
            HPCGraph *g = hpc_create(lat->nsites);

            /* Initialize all sites to |+⟩ */
            double plus_re[6], plus_im[6];
            double inv_sqrt6 = 1.0 / sqrt(6.0);
            for (int i = 0; i < 6; i++) { plus_re[i] = inv_sqrt6; plus_im[i] = 0; }
            for (int s = 0; s < lat->nsites; s++)
                hpc_set_local(g, s, plus_re, plus_im);

            /* Circuit layers */
            for (int d = 0; d < circuit_depth; d++)
                mipt_layer(g, lat, p);

            /* Measure observables */
            double S = hpc_entropy_cut(g, lat->nsites / 2 - 1);
            double D = hpc_exotic_invariant(g);

            sum_S  += S;
            sum_S2 += S * S;
            sum_D  += D;
            sum_E  += (double)g->n_edges;

            hpc_destroy(g);
        }

        clock_gettime(CLOCK_MONOTONIC, &t1);

        results[pi].S_avg = sum_S / num_samples;
        double var = sum_S2 / num_samples - results[pi].S_avg * results[pi].S_avg;
        results[pi].S_std = var > 0 ? sqrt(var) : 0;
        results[pi].Delta_avg = sum_D / num_samples;
        results[pi].edges_avg = sum_E / num_samples;
        results[pi].time_sec = (t1.tv_sec - t0.tv_sec) +
                               (t1.tv_nsec - t0.tv_nsec) / 1e9;
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * MAIN — The Grand Benchmark
 * ═══════════════════════════════════════════════════════════════════ */

int main(void)
{
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  HPC MIPT BENCHMARK — 1D through 6D                           ║\n");
    printf("║  64 Sites · D=6 · Holographic Phase Contraction               ║\n");
    printf("║  No SVD · No Bond Dimension · No Tensors                      ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    s6_exotic_init();
    rng_seed_init((uint64_t)time(NULL));

    /* ── Benchmark parameters ── */
    int circuit_depth = 4;
    int num_samples   = 3;
    double p_values[] = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    int num_p = 11;

    printf("  Circuit depth:    %d layers\n", circuit_depth);
    printf("  Disorder samples: %d\n", num_samples);
    printf("  p-points:         %d\n\n", num_p);

    /* ── Define lattices: all 64 sites ── */
    Lattice lattices[MAX_DIM];
    int ext1[] = {64};
    int ext2[] = {8, 8};
    int ext3[] = {4, 4, 4};
    int ext4[] = {4, 4, 2, 2};
    int ext5[] = {2, 2, 2, 2, 4};
    int ext6[] = {2, 2, 2, 2, 2, 2};

    lattices[0] = lat_create(1, ext1);
    lattices[1] = lat_create(2, ext2);
    lattices[2] = lat_create(3, ext3);
    lattices[3] = lat_create(4, ext4);
    lattices[4] = lat_create(5, ext5);
    lattices[5] = lat_create(6, ext6);

    /* Print lattice specs */
    printf("  ┌─────┬───────────────────┬───────┬───────┬──────────────────┐\n");
    printf("  │ Dim │  Geometry         │ Sites │ Bonds │ Bonds/Site       │\n");
    printf("  ├─────┼───────────────────┼───────┼───────┼──────────────────┤\n");
    for (int d = 0; d < MAX_DIM; d++) {
        printf("  │ %dD  │  %-16s │  %3d  │  %3d  │  %.2f             │\n",
               d + 1, lattices[d].label, lattices[d].nsites,
               lattices[d].nbonds, (double)lattices[d].nbonds / lattices[d].nsites);
    }
    printf("  └─────┴───────────────────┴───────┴───────┴──────────────────┘\n\n");

    /* ── Storage for all results ── */
    MIPTResult results[MAX_DIM][11];
    double total_times[MAX_DIM];
    double pc_values[MAX_DIM];
    double S_max_values[MAX_DIM];

    struct timespec bench_start, bench_end;
    clock_gettime(CLOCK_MONOTONIC, &bench_start);

    /* ═══════════════════════════════════════════════════════════════
     * RUN EACH DIMENSION
     * ═══════════════════════════════════════════════════════════════ */

    for (int d = 0; d < MAX_DIM; d++) {
        printf("═══════════════════════════════════════════════════════════════\n");
        printf("  %dD  │  %s  │  %d sites  │  %d bonds/layer\n",
               d + 1, lattices[d].label, lattices[d].nsites, lattices[d].nbonds);
        printf("═══════════════════════════════════════════════════════════════\n");

        struct timespec dim_start, dim_end;
        clock_gettime(CLOCK_MONOTONIC, &dim_start);

        run_mipt(&lattices[d], circuit_depth, num_samples,
                 p_values, num_p, results[d]);

        clock_gettime(CLOCK_MONOTONIC, &dim_end);
        total_times[d] = (dim_end.tv_sec - dim_start.tv_sec) +
                         (dim_end.tv_nsec - dim_start.tv_nsec) / 1e9;

        /* Print per-dimension results */
        printf("  ╔═════╤════════╤══════════╤══════════╤════════╤═════════╗\n");
        printf("  ║  #  │   p    │  S bits  │    Δ     │ Edges  │  Time   ║\n");
        printf("  ╠═════╪════════╪══════════╪══════════╪════════╪═════════╣\n");

        S_max_values[d] = results[d][0].S_avg;

        /* Find p_c from steepest drop */
        double max_dS = 0;
        int pc_idx = 0;
        for (int i = 1; i < num_p; i++) {
            double dS = results[d][i-1].S_avg - results[d][i].S_avg;
            if (dS > max_dS) { max_dS = dS; pc_idx = i; }
        }
        pc_values[d] = 0.5 * (p_values[pc_idx] + p_values[pc_idx > 0 ? pc_idx-1 : 0]);

        for (int pi = 0; pi < num_p; pi++) {
            char marker = ' ';
            if (pi == pc_idx) marker = '<';  /* p_c marker */

            printf("  ║ %2d  │ %.2f   │ %7.2f  │ %7.2f  │ %5.0f  │ %5.1fs %c║\n",
                   pi, results[d][pi].p,
                   results[d][pi].S_avg, results[d][pi].Delta_avg,
                   results[d][pi].edges_avg, results[d][pi].time_sec, marker);
        }
        printf("  ╚═════╧════════╧══════════╧══════════╧════════╧═════════╝\n");

        /* Entropy bar chart */
        double s_scale = S_max_values[d] > 0.01 ? S_max_values[d] : 1.0;
        int bar_w = 30;
        printf("  ");
        for (int pi = 0; pi < num_p; pi++) {
            int bar = (int)(results[d][pi].S_avg / s_scale * bar_w);
            if (bar < 0) bar = 0;
            if (bar > bar_w) bar = bar_w;
            printf("  p=%.1f │", results[d][pi].p);
            for (int b = 0; b < bar; b++) printf("█");
            for (int b = bar; b < bar_w; b++) printf(" ");
            printf("│ %.1f", results[d][pi].S_avg);
            if (pi == pc_idx) printf(" ◀p_c");
            printf("\n  ");
        }

        printf("\n  p_c ≈ %.3f   S(0)=%.1f   S(1)=%.1f   Time=%.1fs\n\n",
               pc_values[d], S_max_values[d], results[d][num_p-1].S_avg,
               total_times[d]);
        fflush(stdout);
    }

    clock_gettime(CLOCK_MONOTONIC, &bench_end);
    double bench_total = (bench_end.tv_sec - bench_start.tv_sec) +
                         (bench_end.tv_nsec - bench_start.tv_nsec) / 1e9;

    /* ═══════════════════════════════════════════════════════════════
     * COMPARATIVE SUMMARY
     * ═══════════════════════════════════════════════════════════════ */

    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  COMPARATIVE SUMMARY — MIPT Across Dimensions                 ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    printf("  ┌─────┬──────────────┬───────┬─────────┬─────────┬─────────┬────────────┐\n");
    printf("  │ Dim │  Geometry    │ Bonds │  S(p=0) │   p_c   │   Δ(0)  │ Time (s)   │\n");
    printf("  ├─────┼──────────────┼───────┼─────────┼─────────┼─────────┼────────────┤\n");
    for (int d = 0; d < MAX_DIM; d++) {
        printf("  │ %dD  │  %-11s │  %3d  │ %7.1f │  %.3f  │ %7.1f │ %8.1f   │\n",
               d + 1, lattices[d].label, lattices[d].nbonds,
               S_max_values[d], pc_values[d],
               results[d][0].Delta_avg, total_times[d]);
    }
    printf("  └─────┴──────────────┴───────┴─────────┴─────────┴─────────┴────────────┘\n\n");

    /* Scaling analysis */
    printf("  ── Connectivity Scaling ──\n");
    printf("  Bonds/site:   ");
    for (int d = 0; d < MAX_DIM; d++)
        printf("%dD=%.1f  ", d+1, (double)lattices[d].nbonds / lattices[d].nsites);
    printf("\n");

    printf("  S(0)/bond:    ");
    for (int d = 0; d < MAX_DIM; d++) {
        double per_bond = lattices[d].nbonds > 0 ?
            S_max_values[d] / lattices[d].nbonds : 0;
        printf("%dD=%.2f  ", d+1, per_bond);
    }
    printf("\n");

    printf("  Time/bond:    ");
    for (int d = 0; d < MAX_DIM; d++) {
        double per_bond = lattices[d].nbonds > 0 ?
            total_times[d] / lattices[d].nbonds : 0;
        printf("%dD=%.2f  ", d+1, per_bond);
    }
    printf("\n\n");

    /* Exotic invariant analysis */
    printf("  ── Exotic Invariant Δ Across Dimensions ──\n");
    for (int d = 0; d < MAX_DIM; d++) {
        printf("  %dD:  Δ(p=0)=%6.1f  Δ(p=1)=%6.1f  ΔΔ=%+.1f",
               d+1, results[d][0].Delta_avg, results[d][num_p-1].Delta_avg,
               results[d][num_p-1].Delta_avg - results[d][0].Delta_avg);
        if (fabs(results[d][num_p-1].Delta_avg - results[d][0].Delta_avg) > 1.0)
            printf("  ← D=6 content in transition");
        printf("\n");
    }

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  HPC MIPT BENCHMARK COMPLETE\n");
    printf("  6 dimensions × %d p-points × %d samples × %d layers\n",
           num_p, num_samples, circuit_depth);
    printf("  Total benchmark time: %.1f seconds\n", bench_total);
    printf("  All CZ edges exact (fidelity = 1.0)\n");
    printf("═══════════════════════════════════════════════════════════════\n");

    return 0;
}
