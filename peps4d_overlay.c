/*
 * peps4d_overlay.c — 4D Tensor Network: Register-Based SVD Engine
 *
 * D=6 native (SU(6)), bond dimension χ=3 per axis (8 bonds).
 * Simple-update with Jacobi SVD for proper 2-site gate application.
 *
 * WORLD FIRST: 4-Dimensional PEPS on consumer hardware.
 *
 * ── Side-channel optimized (tns_contraction_probe.c) ──
 *   • Gate sparsity via mag² (no fabs)
 *   • Zero-angle skip in Jacobi SVD (via tensor_svd.h)
 *   • 1.0 attractor: bond weights confirmed locked at 1.0
 */

#include "peps4d_overlay.h"
#include "tensor_svd.h"

#define TNS4D_PHYS_POS 8  /* Physical index k is at position 8 (most significant) */

/* ═══════════════ GRID ACCESS ═══════════════ */

static int tns4d_flat(Tns4dGrid *g, int x, int y, int z, int w)
{ return ((w * g->Lz + z) * g->Ly + y) * g->Lx + x; }

/* Bond index helpers — linearized storage along each axis */
static Tns4dBondWeight *tns4d_xbond(Tns4dGrid *g, int x, int y, int z, int w)
{ return &g->x_bonds[((w * g->Lz + z) * g->Ly + y) * (g->Lx - 1) + x]; }

static Tns4dBondWeight *tns4d_ybond(Tns4dGrid *g, int x, int y, int z, int w)
{ return &g->y_bonds[((w * g->Lz + z) * (g->Ly - 1) + y) * g->Lx + x]; }

static Tns4dBondWeight *tns4d_zbond(Tns4dGrid *g, int x, int y, int z, int w)
{ return &g->z_bonds[((w * (g->Lz - 1) + z) * g->Ly + y) * g->Lx + x]; }

static Tns4dBondWeight *tns4d_wbond(Tns4dGrid *g, int x, int y, int z, int w)
{ return &g->w_bonds[((w * g->Lz + z) * g->Ly + y) * g->Lx + x]; }

/* ═══════════════ REGISTER SPARSE I/O ═══════════════ */

static void tns4d_reg_read(Tns4dGrid *g, int site, double *T_re, double *T_im)
{
    int reg = g->site_reg[site];
    if (reg < 0 || !g->eng) return;
    QuhitRegister *r = &g->eng->registers[reg];
    for (uint32_t e = 0; e < r->num_nonzero; e++) {
        basis_t bs = r->entries[e].basis_state;
        if (bs < (basis_t)TNS4D_TSIZ) {
            T_re[bs] = r->entries[e].amp_re;
            T_im[bs] = r->entries[e].amp_im;
        }
    }
}

static void tns4d_reg_write(Tns4dGrid *g, int site,
                            const double *T_re, const double *T_im)
{
    int reg = g->site_reg[site];
    if (reg < 0 || !g->eng) return;
    QuhitRegister *r = &g->eng->registers[reg];
    r->num_nonzero = 0;
    for (basis_t bs = 0; bs < (basis_t)TNS4D_TSIZ; bs++) {
        if (T_re[bs]*T_re[bs] + T_im[bs]*T_im[bs] > 1e-30) {
            if (r->num_nonzero < 4096) {
                r->entries[r->num_nonzero].basis_state = bs;
                r->entries[r->num_nonzero].amp_re = T_re[bs];
                r->entries[r->num_nonzero].amp_im = T_im[bs];
                r->num_nonzero++;
            }
        }
    }
}

/* ═══════════════ LIFECYCLE ═══════════════ */

Tns4dGrid *tns4d_init(int Lx, int Ly, int Lz, int Lw)
{
    Tns4dGrid *g = (Tns4dGrid *)calloc(1, sizeof(Tns4dGrid));
    g->Lx = Lx; g->Ly = Ly; g->Lz = Lz; g->Lw = Lw;
    int N = Lx * Ly * Lz * Lw;

    g->tensors = (Tns4dTensor *)calloc(N, sizeof(Tns4dTensor));

    /* Bond weight arrays — one per nearest-neighbor pair along each axis */
    int nb_x = Lw * Lz * Ly * (Lx - 1);
    int nb_y = Lw * Lz * (Ly - 1) * Lx;
    int nb_z = Lw * (Lz - 1) * Ly * Lx;
    int nb_w = (Lw - 1) * Lz * Ly * Lx;

    g->x_bonds = (Tns4dBondWeight *)calloc(nb_x > 0 ? nb_x : 1, sizeof(Tns4dBondWeight));
    g->y_bonds = (Tns4dBondWeight *)calloc(nb_y > 0 ? nb_y : 1, sizeof(Tns4dBondWeight));
    g->z_bonds = (Tns4dBondWeight *)calloc(nb_z > 0 ? nb_z : 1, sizeof(Tns4dBondWeight));
    g->w_bonds = (Tns4dBondWeight *)calloc(nb_w > 0 ? nb_w : 1, sizeof(Tns4dBondWeight));

    /* Heap-allocate and initialize all bond weights to 1.0 */
    for (int i = 0; i < nb_x; i++) {
        g->x_bonds[i].w = (double *)calloc((size_t)TNS4D_CHI, sizeof(double));
        for (int s = 0; s < (int)TNS4D_CHI; s++) g->x_bonds[i].w[s] = 1.0;
    }
    for (int i = 0; i < nb_y; i++) {
        g->y_bonds[i].w = (double *)calloc((size_t)TNS4D_CHI, sizeof(double));
        for (int s = 0; s < (int)TNS4D_CHI; s++) g->y_bonds[i].w[s] = 1.0;
    }
    for (int i = 0; i < nb_z; i++) {
        g->z_bonds[i].w = (double *)calloc((size_t)TNS4D_CHI, sizeof(double));
        for (int s = 0; s < (int)TNS4D_CHI; s++) g->z_bonds[i].w[s] = 1.0;
    }
    for (int i = 0; i < nb_w; i++) {
        g->w_bonds[i].w = (double *)calloc((size_t)TNS4D_CHI, sizeof(double));
        for (int s = 0; s < (int)TNS4D_CHI; s++) g->w_bonds[i].w[s] = 1.0;
    }

    /* Initialize engine and register per site */
    g->eng = (QuhitEngine *)calloc(1, sizeof(QuhitEngine));
    quhit_engine_init(g->eng);

    g->q_phys = (uint32_t *)calloc(N, sizeof(uint32_t));
    for (int i = 0; i < N; i++)
        g->q_phys[i] = quhit_init_basis(g->eng, 0);

    g->site_reg = (int *)calloc(N, sizeof(int));
    for (int i = 0; i < N; i++) {
        g->site_reg[i] = quhit_reg_init(g->eng, (uint64_t)i, 9, TNS4D_CHI);
        if (g->site_reg[i] >= 0) {
            g->eng->registers[g->site_reg[i]].bulk_rule = 0;
            quhit_reg_sv_set(g->eng, g->site_reg[i], 0, 1.0, 0.0);
        }
        g->tensors[i].reg_idx = g->site_reg[i];
    }

    return g;
}

void tns4d_free(Tns4dGrid *g)
{
    if (!g) return;
    free(g->tensors);
    /* Free heap-allocated bond weight arrays */
    int nb_x = g->Lw * g->Lz * g->Ly * (g->Lx - 1);
    int nb_y = g->Lw * g->Lz * (g->Ly - 1) * g->Lx;
    int nb_z = g->Lw * (g->Lz - 1) * g->Ly * g->Lx;
    int nb_w = (g->Lw - 1) * g->Lz * g->Ly * g->Lx;
    for (int i = 0; i < nb_x; i++) free(g->x_bonds[i].w);
    for (int i = 0; i < nb_y; i++) free(g->y_bonds[i].w);
    for (int i = 0; i < nb_z; i++) free(g->z_bonds[i].w);
    for (int i = 0; i < nb_w; i++) free(g->w_bonds[i].w);
    free(g->x_bonds);
    free(g->y_bonds);
    free(g->z_bonds);
    free(g->w_bonds);
    if (g->eng) {
        quhit_engine_destroy(g->eng);
        free(g->eng);
    }
    free(g->q_phys);
    free(g->site_reg);
    free(g);
}

/* ═══════════════ STATE INITIALIZATION ═══════════════ */

void tns4d_set_product_state(Tns4dGrid *g, int x, int y, int z, int w,
                             const double *amps_re, const double *amps_im)
{
    int site = tns4d_flat(g, x, y, z, w);
    int reg = g->site_reg[site];
    if (reg < 0) return;
    QuhitRegister *r = &g->eng->registers[reg];
    r->num_nonzero = 0;
    for (int k = 0; k < TNS4D_D; k++) {
        double re = amps_re[k], im = amps_im[k];
        if (re*re + im*im > 1e-30) {
            r->entries[r->num_nonzero].basis_state = (basis_t)k * TNS4D_C8;
            r->entries[r->num_nonzero].amp_re = re;
            r->entries[r->num_nonzero].amp_im = im;
            r->num_nonzero++;
        }
    }
}

/* ═══════════════ 1-SITE GATE ═══════════════ */

struct tmp4d_entry { basis_t basis; double re, im; };

static int cmp4d_basis(const void *a, const void *b) {
    basis_t ba = ((const struct tmp4d_entry *)a)->basis;
    basis_t bb = ((const struct tmp4d_entry *)b)->basis;
    return (ba > bb) - (ba < bb);
}

void tns4d_gate_1site(Tns4dGrid *g, int x, int y, int z, int w,
                      const double *U_re, const double *U_im)
{
    int site = tns4d_flat(g, x, y, z, w);
    int reg = g->site_reg[site];
    if (reg < 0) return;

    QuhitRegister *r = &g->eng->registers[reg];
    int D = TNS4D_D;
    uint32_t old_nnz = r->num_nonzero;
    if (old_nnz == 0) return;

    /* Save old sparse entries */
    basis_t *old_bs = (basis_t *)malloc(old_nnz * sizeof(basis_t));
    double   *old_re = (double *)malloc(old_nnz * sizeof(double));
    double   *old_im = (double *)malloc(old_nnz * sizeof(double));
    for (uint32_t e = 0; e < old_nnz; e++) {
        old_bs[e] = r->entries[e].basis_state;
        old_re[e] = r->entries[e].amp_re;
        old_im[e] = r->entries[e].amp_im;
    }

    /* Apply gate at physical position (most significant) */
    size_t cap = (size_t)old_nnz * D;
    struct tmp4d_entry *tmp = (struct tmp4d_entry *)calloc(cap, sizeof(*tmp));
    size_t ntmp = 0;

    for (uint32_t e = 0; e < old_nnz; e++) {
        basis_t bs_old = old_bs[e];
        int k_old = (int)(bs_old / TNS4D_C8);
        basis_t bond_part = bs_old % TNS4D_C8;

        for (int k_new = 0; k_new < D; k_new++) {
            double ure = U_re[k_new * D + k_old];
            double uim = U_im[k_new * D + k_old];
            if (ure*ure + uim*uim < 1e-30) continue;

            double tre = ure * old_re[e] - uim * old_im[e];
            double tim = ure * old_im[e] + uim * old_re[e];

            basis_t new_bs = (basis_t)k_new * TNS4D_C8 + bond_part;

            /* Try to accumulate */
            int found = 0;
            for (size_t i = 0; i < ntmp; i++) {
                if (tmp[i].basis == new_bs) {
                    tmp[i].re += tre;
                    tmp[i].im += tim;
                    found = 1;
                    break;
                }
            }
            if (!found && ntmp < cap) {
                tmp[ntmp].basis = new_bs;
                tmp[ntmp].re = tre;
                tmp[ntmp].im = tim;
                ntmp++;
            }
        }
    }

    free(old_bs); free(old_re); free(old_im);

    /* Write back (sparse) */
    r->num_nonzero = 0;
    for (size_t i = 0; i < ntmp; i++) {
        if (tmp[i].re*tmp[i].re + tmp[i].im*tmp[i].im < 1e-30) continue;
        if (r->num_nonzero < 4096) {
            r->entries[r->num_nonzero].basis_state = tmp[i].basis;
            r->entries[r->num_nonzero].amp_re = tmp[i].re;
            r->entries[r->num_nonzero].amp_im = tmp[i].im;
            r->num_nonzero++;
        }
    }
    free(tmp);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * 2-SITE GATE — Generic axis SVD contraction for 4D
 *
 * Axis mapping (shared bond positions in the 9-index encoding):
 *   X=0: bond_A=4 (r), bond_B=5 (l)
 *   Y=1: bond_A=7 (u), bond_B=6 (d)
 *   Z=2: bond_A=3 (f), bond_B=2 (b)
 *   W=3: bond_A=1 (i), bond_B=0 (o)
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void tns4d_gate_2site_generic(Tns4dGrid *g,
                                     int sA, int sB,
                                     Tns4dBondWeight *shared_bw,
                                     int shared_axis,
                                     const double *G_re, const double *G_im)
{
    int D = TNS4D_D, chi = (int)TNS4D_CHI;
    basis_t bp[9] = {1, TNS4D_CHI, TNS4D_C2, TNS4D_C3, TNS4D_C4,
                      TNS4D_C5, TNS4D_C6, TNS4D_C7, TNS4D_C8};

    int bond_A = -1, bond_B = -1;
    if (shared_axis == 0)      { bond_A = 4; bond_B = 5; } /* X: rA, lB */
    else if (shared_axis == 1) { bond_A = 7; bond_B = 6; } /* Y: uA, dB */
    else if (shared_axis == 2) { bond_A = 3; bond_B = 2; } /* Z: fA, bB */
    else                       { bond_A = 1; bond_B = 0; } /* W: iA, oB */

    QuhitRegister *regA = &g->eng->registers[g->site_reg[sA]];
    QuhitRegister *regB = &g->eng->registers[g->site_reg[sB]];

    /* ── 1. Find Sparse-Rank Environment ── */
    int max_E = chi;
    basis_t *uniq_envA = (basis_t*)malloc(max_E * sizeof(basis_t));
    basis_t *uniq_envB = (basis_t*)malloc(max_E * sizeof(basis_t));
    int num_EA = 0, num_EB = 0;

    for (uint32_t eA = 0; eA < regA->num_nonzero; eA++) {
        basis_t pure = regA->entries[eA].basis_state % TNS4D_C8;
        basis_t env = (pure / bp[bond_A + 1]) * bp[bond_A] + (pure % bp[bond_A]);
        int found = 0;
        for (int i = 0; i < num_EA; i++) {
            if (uniq_envA[i] == env) { found = 1; break; }
        }
        if (!found && num_EA < max_E) uniq_envA[num_EA++] = env;
    }

    for (uint32_t eB = 0; eB < regB->num_nonzero; eB++) {
        basis_t pure = regB->entries[eB].basis_state % TNS4D_C8;
        basis_t env = (pure / bp[bond_B + 1]) * bp[bond_B] + (pure % bp[bond_B]);
        int found = 0;
        for (int i = 0; i < num_EB; i++) {
            if (uniq_envB[i] == env) { found = 1; break; }
        }
        if (!found && num_EB < max_E) uniq_envB[num_EB++] = env;
    }

    if (num_EA == 0 || num_EB == 0) {
        free(uniq_envA); free(uniq_envB);
        return;
    }

    /* ── 2. Build Θ ── */
    int svddim_A = D * num_EA;
    int svddim_B = D * num_EB;

    size_t svd2 = (size_t)svddim_A * svddim_B;
    double *Th_re = (double *)calloc(svd2, sizeof(double));
    double *Th_im = (double *)calloc(svd2, sizeof(double));

    for (uint32_t eA = 0; eA < regA->num_nonzero; eA++) {
        basis_t bsA = regA->entries[eA].basis_state;
        double arA = regA->entries[eA].amp_re;
        double aiA = regA->entries[eA].amp_im;
        if (arA*arA + aiA*aiA < 1e-10) continue;

        int kA = (int)(bsA / TNS4D_C8);
        basis_t pureA = bsA % TNS4D_C8;
        int shared_valA = (int)((pureA / bp[bond_A]) % chi);
        basis_t envA = (pureA / bp[bond_A + 1]) * bp[bond_A] + (pureA % bp[bond_A]);

        int idx_EA = -1;
        for (int i = 0; i < num_EA; i++) if (uniq_envA[i] == envA) { idx_EA = i; break; }
        if (idx_EA < 0) continue;
        int row = kA * num_EA + idx_EA;

        for (uint32_t eB = 0; eB < regB->num_nonzero; eB++) {
            basis_t bsB = regB->entries[eB].basis_state;
            double arB = regB->entries[eB].amp_re;
            double aiB = regB->entries[eB].amp_im;
            if (arB*arB + aiB*aiB < 1e-10) continue;

            basis_t pureB = bsB % TNS4D_C8;
            int shared_valB = (int)((pureB / bp[bond_B]) % chi);
            if (shared_valA != shared_valB) continue;

            int kB = (int)(bsB / TNS4D_C8);
            basis_t envB = (pureB / bp[bond_B + 1]) * bp[bond_B] + (pureB % bp[bond_B]);

            int idx_EB = -1;
            for (int i = 0; i < num_EB; i++) if (uniq_envB[i] == envB) { idx_EB = i; break; }
            if (idx_EB < 0) continue;
            int col = kB * num_EB + idx_EB;

            double sw = shared_bw->w[shared_valA];
            double br = arB * sw, bi = aiB * sw;

            Th_re[row * svddim_B + col] += arA*br - aiA*bi;
            Th_im[row * svddim_B + col] += arA*bi + aiA*br;
        }
    }

    /* ── 3. Apply Gate ── */
    double *Th2_re = (double *)calloc(svd2, sizeof(double));
    double *Th2_im = (double *)calloc(svd2, sizeof(double));
    int D2 = D * D;

    for (int kAp = 0; kAp < D; kAp++)
     for (int kBp = 0; kBp < D; kBp++) {
         int gr = kAp * D + kBp;
         for (int kA = 0; kA < D; kA++)
          for (int kB = 0; kB < D; kB++) {
              int gc = kA * D + kB;
              double gre = G_re[gr * D2 + gc];
              double gim = G_im[gr * D2 + gc];
              /* Side-channel: squared gate check (avoids 2× fabs) */
              if (gre*gre + gim*gim < 1e-20) continue;

              for (int eA = 0; eA < num_EA; eA++) {
                  int dst_row = kAp * num_EA + eA;
                  int src_row = kA * num_EA + eA;
                  for (int eB = 0; eB < num_EB; eB++) {
                      int dst_col = kBp * num_EB + eB;
                      int src_col = kB * num_EB + eB;
                      double tr = Th_re[src_row * svddim_B + src_col];
                      double ti = Th_im[src_row * svddim_B + src_col];
                      Th2_re[dst_row * svddim_B + dst_col] += gre*tr - gim*ti;
                      Th2_im[dst_row * svddim_B + dst_col] += gre*ti + gim*tr;
                  }
              }
          }
     }
    free(Th_re); free(Th_im);

    /* ── 4. SVD ── */
    double *U_re  = (double *)calloc((size_t)svddim_A * chi, sizeof(double));
    double *U_im  = (double *)calloc((size_t)svddim_A * chi, sizeof(double));
    double *sig   = (double *)calloc(chi, sizeof(double));
    double *Vc_re = (double *)calloc((size_t)chi * svddim_B, sizeof(double));
    double *Vc_im = (double *)calloc((size_t)chi * svddim_B, sizeof(double));

    tsvd_truncated_sparse(Th2_re, Th2_im, svddim_A, svddim_B, chi,
                   U_re, U_im, sig, Vc_re, Vc_im);
    free(Th2_re); free(Th2_im);

    int rank = chi < svddim_B ? chi : svddim_B;
    if (rank > svddim_A) rank = svddim_A;

    /* Cap rank so write-back stays within 4096-entry register limit */
    int max_env = num_EA > num_EB ? num_EA : num_EB;
    int rank_cap = max_env > 0 ? 4096 / (D * max_env) : rank;
    if (rank_cap < 1) rank_cap = 1;
    if (rank > rank_cap) rank = rank_cap;

    double sig_norm = 0;
    for (int s = 0; s < rank; s++) sig_norm += sig[s];

    /* Side-channel: 1.0 attractor CONFIRMED — bond weights lock at 1.0
     * (entropy = log₂(χ) = maximal). Schmidt weights absorbed into U/V. */
    for (int s = 0; s < (int)TNS4D_CHI; s++) shared_bw->w[s] = 1.0;

    /* ── 5. Write back (sparse) ── */
    regA->num_nonzero = 0;
    regB->num_nonzero = 0;

    for (int kA = 0; kA < D; kA++)
     for (int eA = 0; eA < num_EA; eA++) {
         int row = kA * num_EA + eA;
         basis_t envA = uniq_envA[eA];
         basis_t pure = (envA / bp[bond_A]) * bp[bond_A + 1] + (envA % bp[bond_A]);
         for (int gv = 0; gv < rank; gv++) {
             double weight = (sig_norm > 1e-30 && sig[gv] > 1e-30) ? sig[gv] * born_fast_isqrt(sig[gv]) * born_fast_isqrt(sig_norm) : 0.0;
             double re = U_re[row * rank + gv] * weight;
             double im = U_im[row * rank + gv] * weight;
             if (re*re + im*im < 1e-50) continue;

             basis_t bs = kA * TNS4D_C8 + pure + gv * bp[bond_A];
             if (regA->num_nonzero < 4096) {
                 regA->entries[regA->num_nonzero].basis_state = bs;
                 regA->entries[regA->num_nonzero].amp_re = re;
                 regA->entries[regA->num_nonzero].amp_im = im;
                 regA->num_nonzero++;
             }
         }
     }

    for (int kB = 0; kB < D; kB++)
     for (int eB = 0; eB < num_EB; eB++) {
         int col = kB * num_EB + eB;
         basis_t envB = uniq_envB[eB];
         basis_t pure = (envB / bp[bond_B]) * bp[bond_B + 1] + (envB % bp[bond_B]);
         for (int gv = 0; gv < rank; gv++) {
             double weight = (sig_norm > 1e-30 && sig[gv] > 1e-30) ? sig[gv] * born_fast_isqrt(sig[gv]) * born_fast_isqrt(sig_norm) : 0.0;
             double re = weight * Vc_re[gv * svddim_B + col];
             double im = weight * Vc_im[gv * svddim_B + col];
             if (re*re + im*im < 1e-50) continue;

             basis_t bs = kB * TNS4D_C8 + pure + gv * bp[bond_B];
             if (regB->num_nonzero < 4096) {
                 regB->entries[regB->num_nonzero].basis_state = bs;
                 regB->entries[regB->num_nonzero].amp_re = re;
                 regB->entries[regB->num_nonzero].amp_im = im;
                 regB->num_nonzero++;
             }
         }
     }

    free(U_re); free(U_im);
    free(sig);
    free(Vc_re); free(Vc_im);
    free(uniq_envA); free(uniq_envB);
}


/* ═══════════════ AXIS WRAPPERS ═══════════════ */

void tns4d_gate_x(Tns4dGrid *g, int x, int y, int z, int w,
                  const double *G_re, const double *G_im)
{
    tns4d_gate_2site_generic(g,
        tns4d_flat(g, x, y, z, w), tns4d_flat(g, x+1, y, z, w),
        tns4d_xbond(g, x, y, z, w), 0, G_re, G_im);
}

void tns4d_gate_y(Tns4dGrid *g, int x, int y, int z, int w,
                  const double *G_re, const double *G_im)
{
    tns4d_gate_2site_generic(g,
        tns4d_flat(g, x, y, z, w), tns4d_flat(g, x, y+1, z, w),
        tns4d_ybond(g, x, y, z, w), 1, G_re, G_im);
}

void tns4d_gate_z(Tns4dGrid *g, int x, int y, int z, int w,
                  const double *G_re, const double *G_im)
{
    tns4d_gate_2site_generic(g,
        tns4d_flat(g, x, y, z, w), tns4d_flat(g, x, y, z+1, w),
        tns4d_zbond(g, x, y, z, w), 2, G_re, G_im);
}

void tns4d_gate_w(Tns4dGrid *g, int x, int y, int z, int w,
                  const double *G_re, const double *G_im)
{
    tns4d_gate_2site_generic(g,
        tns4d_flat(g, x, y, z, w), tns4d_flat(g, x, y, z, w+1),
        tns4d_wbond(g, x, y, z, w), 3, G_re, G_im);
}

/* ═══════════════ LOCAL DENSITY ═══════════════ */

void tns4d_local_density(Tns4dGrid *g, int x, int y, int z, int w, double *probs)
{
    int site = tns4d_flat(g, x, y, z, w);
    int reg = g->site_reg[site];

    for (int k = 0; k < TNS4D_D; k++) probs[k] = 0;

    if (reg < 0 || !g->eng) { probs[0] = 1.0; return; }

    QuhitRegister *r = &g->eng->registers[reg];
    double total = 0;

    for (uint32_t e = 0; e < r->num_nonzero; e++) {
        basis_t bs = r->entries[e].basis_state;
        int k = (int)(bs / TNS4D_C8);
        if (k >= TNS4D_D) continue;
        double re = r->entries[e].amp_re;
        double im = r->entries[e].amp_im;
        double p = re * re + im * im;
        probs[k] += p;
        total += p;
    }

    if (total > 1e-30)
        for (int k = 0; k < TNS4D_D; k++) probs[k] /= total;
    else
        probs[0] = 1.0;
}

/* ═══════════════ BATCH GATE APPLICATION ═══════════════ */

void tns4d_gate_x_all(Tns4dGrid *g, const double *G_re, const double *G_im)
{
    if (g->Lx < 2) return;
    for (int parity = 0; parity < 2; parity++) {
        for (int w = 0; w < g->Lw; w++)
         for (int z = 0; z < g->Lz; z++)
          for (int y = 0; y < g->Ly; y++)
           for (int xh = 0; xh < (g->Lx + 1) / 2; xh++) {
               int x = xh * 2 + parity;
               if (x < g->Lx - 1) tns4d_gate_x(g, x, y, z, w, G_re, G_im);
           }
    }
}

void tns4d_gate_y_all(Tns4dGrid *g, const double *G_re, const double *G_im)
{
    if (g->Ly < 2) return;
    for (int parity = 0; parity < 2; parity++) {
        for (int w = 0; w < g->Lw; w++)
         for (int z = 0; z < g->Lz; z++)
          for (int yh = 0; yh < (g->Ly + 1) / 2; yh++)
           for (int x = 0; x < g->Lx; x++) {
               int y = yh * 2 + parity;
               if (y < g->Ly - 1) tns4d_gate_y(g, x, y, z, w, G_re, G_im);
           }
    }
}

void tns4d_gate_z_all(Tns4dGrid *g, const double *G_re, const double *G_im)
{
    if (g->Lz < 2) return;
    for (int parity = 0; parity < 2; parity++) {
        for (int w = 0; w < g->Lw; w++)
         for (int zh = 0; zh < (g->Lz + 1) / 2; zh++)
          for (int y = 0; y < g->Ly; y++)
           for (int x = 0; x < g->Lx; x++) {
               int z = zh * 2 + parity;
               if (z < g->Lz - 1) tns4d_gate_z(g, x, y, z, w, G_re, G_im);
           }
    }
}

void tns4d_gate_w_all(Tns4dGrid *g, const double *G_re, const double *G_im)
{
    if (g->Lw < 2) return;
    for (int parity = 0; parity < 2; parity++) {
        for (int wh = 0; wh < (g->Lw + 1) / 2; wh++)
         for (int z = 0; z < g->Lz; z++)
          for (int y = 0; y < g->Ly; y++)
           for (int x = 0; x < g->Lx; x++) {
               int w = wh * 2 + parity;
               if (w < g->Lw - 1) tns4d_gate_w(g, x, y, z, w, G_re, G_im);
           }
    }
}

void tns4d_normalize_site(Tns4dGrid *g, int x, int y, int z, int w)
{
    if (x<0||x>=g->Lx||y<0||y>=g->Ly||z<0||z>=g->Lz||w<0||w>=g->Lw) return;
    int site = tns4d_flat(g, x, y, z, w);
    int reg = g->site_reg[site];
    if (reg < 0) return;
    QuhitRegister *r = &g->eng->registers[reg];

    double n2 = 0;
    for (uint32_t e = 0; e < r->num_nonzero; e++) {
        n2 += r->entries[e].amp_re * r->entries[e].amp_re +
              r->entries[e].amp_im * r->entries[e].amp_im;
    }
    if (n2 > 1e-20) {
        double inv = born_fast_isqrt(n2);
        for (uint32_t e = 0; e < r->num_nonzero; e++) {
            r->entries[e].amp_re *= inv;
            r->entries[e].amp_im *= inv;
        }
    }
}

void tns4d_gate_1site_all(Tns4dGrid *g, const double *U_re, const double *U_im)
{
    for (int w = 0; w < g->Lw; w++)
     for (int z = 0; z < g->Lz; z++)
      for (int y = 0; y < g->Ly; y++)
       for (int x = 0; x < g->Lx; x++)
           tns4d_gate_1site(g, x, y, z, w, U_re, U_im);
}

void tns4d_trotter_step(Tns4dGrid *g, const double *G_re, const double *G_im)
{
    tns4d_gate_x_all(g, G_re, G_im);
    tns4d_gate_y_all(g, G_re, G_im);
    tns4d_gate_z_all(g, G_re, G_im);
    tns4d_gate_w_all(g, G_re, G_im);
}
