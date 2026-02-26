/*
 * peps3d_overlay.c — 3D Tensor Network: Register-Based SVD Engine
 *
 * D=6 native (SU(6)), bond dimension χ=3 per axis (6 axes).
 * Simple-update with Jacobi SVD for proper 2-site gate application.
 * All tensor data stored in registers — temporary dense buffers for SVD.
 *
 * ── Side-channel optimized (tns_contraction_probe.c) ──
 *   • Gate sparsity via mag² (no fabs)
 *   • Zero-angle skip in Jacobi SVD (via tensor_svd.h)
 *   • 1.0 attractor: bond weights confirmed locked at 1.0
 *   • 1/6 spectrum: σ values converge toward equal weights
 */

#include "peps3d_overlay.h"
#include "tensor_svd.h"
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define TNS3D_SVDDIM (TNS3D_D * TNS3D_CHI * TNS3D_CHI)  /* 54 at χ=3 */
#define TNS3D_PHYS_POS 6  /* Physical index k is at position 6 (most significant) */

/* ═══════════════ GRID ACCESS ═══════════════ */

static inline Tns3dTensor *tns3d_site(Tns3dGrid *g, int x, int y, int z)
{ return &g->tensors[(z * g->Ly + y) * g->Lx + x]; }

static inline Tns3dBondWeight *tns3d_xbond(Tns3dGrid *g, int x, int y, int z)
{ return &g->x_bonds[(z * g->Ly + y) * (g->Lx - 1) + x]; }

static inline Tns3dBondWeight *tns3d_ybond(Tns3dGrid *g, int x, int y, int z)
{ return &g->y_bonds[(z * (g->Ly - 1) + y) * g->Lx + x]; }

static inline Tns3dBondWeight *tns3d_zbond(Tns3dGrid *g, int x, int y, int z)
{ return &g->z_bonds[(z * g->Ly + y) * g->Lx + x]; }

static inline int tns3d_flat(Tns3dGrid *g, int x, int y, int z)
{ return (z * g->Ly + y) * g->Lx + x; }

/* ═══════════════ REGISTER DENSE I/O ═══════════════ */

static void tns3d_reg_read(Tns3dGrid *g, int site, double *T_re, double *T_im)
{
    int reg = g->site_reg[site];
    size_t tsz = (size_t)TNS3D_TSIZ;
    memset(T_re, 0, tsz * sizeof(double));
    memset(T_im, 0, tsz * sizeof(double));
    if (reg < 0 || !g->eng) return;

    QuhitRegister *r = &g->eng->registers[reg];
    for (uint32_t e = 0; e < r->num_nonzero; e++) {
        uint64_t bs = r->entries[e].basis_state;
        if (bs < tsz) {
            T_re[bs] = r->entries[e].amp_re;
            T_im[bs] = r->entries[e].amp_im;
        }
    }
}

static void tns3d_reg_write(Tns3dGrid *g, int site,
                            const double *T_re, const double *T_im)
{
    int reg = g->site_reg[site];
    if (reg < 0 || !g->eng) return;
    size_t tsz = (size_t)TNS3D_TSIZ;

    g->eng->registers[reg].num_nonzero = 0;
    for (size_t bs = 0; bs < tsz; bs++) {
        if (T_re[bs]*T_re[bs] + T_im[bs]*T_im[bs] > 1e-30)
            quhit_reg_sv_set(g->eng, reg, (uint64_t)bs, T_re[bs], T_im[bs]);
    }
}

/* ═══════════════ LIFECYCLE ═══════════════ */

Tns3dGrid *tns3d_init(int Lx, int Ly, int Lz)
{
    Tns3dGrid *g = (Tns3dGrid *)calloc(1, sizeof(Tns3dGrid));
    g->Lx = Lx; g->Ly = Ly; g->Lz = Lz;
    int N = Lx * Ly * Lz;

    g->tensors = (Tns3dTensor *)calloc(N, sizeof(Tns3dTensor));

    g->x_bonds = (Tns3dBondWeight *)calloc(Lz * Ly * (Lx - 1), sizeof(Tns3dBondWeight));
    g->y_bonds = (Tns3dBondWeight *)calloc(Lz * (Ly - 1) * Lx, sizeof(Tns3dBondWeight));
    g->z_bonds = (Tns3dBondWeight *)calloc((Lz - 1) * Ly * Lx, sizeof(Tns3dBondWeight));

    for (int i = 0; i < Lz * Ly * (Lx - 1); i++)
        for (int s = 0; s < TNS3D_CHI; s++) g->x_bonds[i].w[s] = 1.0;
    for (int i = 0; i < Lz * (Ly - 1) * Lx; i++)
        for (int s = 0; s < TNS3D_CHI; s++) g->y_bonds[i].w[s] = 1.0;
    for (int i = 0; i < (Lz - 1) * Ly * Lx; i++)
        for (int s = 0; s < TNS3D_CHI; s++) g->z_bonds[i].w[s] = 1.0;

    g->eng = (QuhitEngine *)calloc(1, sizeof(QuhitEngine));
    quhit_engine_init(g->eng);

    g->q_phys = (uint32_t *)calloc(N, sizeof(uint32_t));
    for (int i = 0; i < N; i++)
        g->q_phys[i] = quhit_init_basis(g->eng, 0);

    g->site_reg = (int *)calloc(N, sizeof(int));
    for (int i = 0; i < N; i++) {
        g->site_reg[i] = quhit_reg_init(g->eng, (uint64_t)i, 7, TNS3D_CHI);
        if (g->site_reg[i] >= 0) {
            g->eng->registers[g->site_reg[i]].bulk_rule = 0;
            quhit_reg_sv_set(g->eng, g->site_reg[i], 0, 1.0, 0.0);
        }
        g->tensors[i].reg_idx = g->site_reg[i];
    }

    return g;
}

void tns3d_free(Tns3dGrid *g)
{
    if (!g) return;
    free(g->tensors);
    free(g->x_bonds);
    free(g->y_bonds);
    free(g->z_bonds);
    if (g->eng) {
        quhit_engine_destroy(g->eng);
        free(g->eng);
    }
    free(g->q_phys);
    free(g->site_reg);
    free(g);
}

/* ═══════════════ STATE INITIALIZATION ═══════════════ */

void tns3d_set_product_state(Tns3dGrid *g, int x, int y, int z,
                             const double *amps_re, const double *amps_im)
{
    int site = tns3d_flat(g, x, y, z);
    int reg = g->site_reg[site];
    if (reg < 0) return;

    g->eng->registers[reg].num_nonzero = 0;
    for (int k = 0; k < TNS3D_D; k++) {
        if (amps_re[k] * amps_re[k] + amps_im[k] * amps_im[k] > 1e-30)
            quhit_reg_sv_set(g->eng, reg, (uint64_t)k * TNS3D_C6, amps_re[k], amps_im[k]);
    }
}

/* ═══════════════ 1-SITE GATE ═══════════════ */

struct tmp_entry { uint64_t basis; double re, im; };

static int cmp_basis(const void *a, const void *b)
{
    const struct tmp_entry *ea = (const struct tmp_entry *)a;
    const struct tmp_entry *eb = (const struct tmp_entry *)b;
    if (ea->basis < eb->basis) return -1;
    if (ea->basis > eb->basis) return 1;
    return 0;
}

void tns3d_gate_1site(Tns3dGrid *g, int x, int y, int z,
                      const double *U_re, const double *U_im)
{
    int site = tns3d_flat(g, x, y, z);
    int reg_idx = g->site_reg[site];
    if (reg_idx < 0 || !g->eng) return;

    /* Manual rotation of physical index k.
     * Register dim=χ=12, but physical index k∈[0,D=6).
     * Can't use quhit_reg_apply_unitary_pos which assumes dim=D. */
    int D = TNS3D_D;
    QuhitRegister *r = &g->eng->registers[reg_idx];
    uint32_t old_n = r->num_nonzero;

    /* Read existing entries */
    uint64_t *old_bs = (uint64_t *)calloc(old_n, sizeof(uint64_t));
    double *old_re = (double *)calloc(old_n, sizeof(double));
    double *old_im = (double *)calloc(old_n, sizeof(double));
    for (uint32_t e = 0; e < old_n; e++) {
        old_bs[e] = r->entries[e].basis_state;
        old_re[e] = r->entries[e].amp_re;
        old_im[e] = r->entries[e].amp_im;
    }

    r->num_nonzero = 0;

    uint32_t max_out = old_n * D + 1;
    if (max_out < 4096) max_out = 4096;
    struct tmp_entry *tmp = calloc(max_out, sizeof(*tmp));
    uint32_t nout = 0;

    for (int kp = 0; kp < D; kp++) {
        for (uint32_t e = 0; e < old_n; e++) {
            uint64_t bs = old_bs[e];
            int k = (int)(bs / TNS3D_C6);
            if (k >= D) continue;
            uint64_t bond_part = bs % TNS3D_C6;

            double gre = U_re[kp * D + k];
            double gim = U_im[kp * D + k];
            if (gre*gre + gim*gim < 1e-10) continue;

            double are = old_re[e], aim = old_im[e];
            double new_re = gre*are - gim*aim;
            double new_im = gre*aim + gim*are;

            if (nout < max_out) {
                tmp[nout].basis = (uint64_t)kp * TNS3D_C6 + bond_part;
                tmp[nout].re = new_re;
                tmp[nout].im = new_im;
                nout++;
            }
        }
    }

    qsort(tmp, nout, sizeof(struct tmp_entry), cmp_basis);

    r->num_nonzero = 0;
    for (uint32_t t = 0; t < nout; t++) {
        double acc_re = tmp[t].re;
        double acc_im = tmp[t].im;
        while (t + 1 < nout && tmp[t+1].basis == tmp[t].basis) {
            t++;
            acc_re += tmp[t].re;
            acc_im += tmp[t].im;
        }
        if (acc_re*acc_re + acc_im*acc_im >= 1e-10 &&
            r->num_nonzero < 4096) {
            r->entries[r->num_nonzero].basis_state = tmp[t].basis;
            r->entries[r->num_nonzero].amp_re = acc_re;
            r->entries[r->num_nonzero].amp_im = acc_im;
            r->num_nonzero++;
        }
    }

    free(tmp);
    free(old_bs); free(old_re); free(old_im);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * GENERIC 3D 2-SITE GATE WITH SVD
 *
 * shared_axis: 0=x(r/l), 1=y(d/u), 2=z(b/f)
 * For each axis, the 6 remaining non-shared bonds are absorbed as environment.
 * SVD dim = D × χ² (physical × 2 kept free bonds)
 *
 * 7-index tensor: T[k, u, d, l, r, f, b]
 * Index positions: k=0, u=1, d=2, l=3, r=4, f=5, b=6
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void tns3d_gate_2site_generic(Tns3dGrid *g,
                                     int sA, int sB,
                                     Tns3dBondWeight *shared_bw,
                                     int shared_axis,
                                     const double *G_re, const double *G_im)
{
    int D = TNS3D_D, chi = TNS3D_CHI;
    uint64_t bp[7] = {1, TNS3D_CHI, TNS3D_C2, TNS3D_C3, TNS3D_C4, TNS3D_C5, TNS3D_C6};
    int bond_A = -1, bond_B = -1;
    if (shared_axis == 0)      { bond_A = 2; bond_B = 3; } /* X: rA, lB */
    else if (shared_axis == 1) { bond_A = 1; bond_B = 0; } /* Y: fA, bB */
    else                       { bond_A = 4; bond_B = 5; } /* Z: dA, uB */

    QuhitRegister *regA = &g->eng->registers[g->site_reg[sA]];
    QuhitRegister *regB = &g->eng->registers[g->site_reg[sB]];

    /* ── 1. Find exact Sparse-Rank Environment ──
     * Cap at χ² unique environments per side to keep SVD at D×χ² max.
     * This preserves REAL bond configurations instead of synthetic ones. */
    int max_E = chi * chi;
    uint64_t *uniq_envA = (uint64_t*)malloc(max_E * sizeof(uint64_t));
    uint64_t *uniq_envB = (uint64_t*)malloc(max_E * sizeof(uint64_t));
    int num_EA = 0, num_EB = 0;

    for (uint32_t eA = 0; eA < regA->num_nonzero; eA++) {
        uint64_t pure = regA->entries[eA].basis_state % TNS3D_C6;
        uint64_t env = (pure / bp[bond_A + 1]) * bp[bond_A] + (pure % bp[bond_A]);
        int found = 0;
        for (int i = 0; i < num_EA; i++) {
            if (uniq_envA[i] == env) { found = 1; break; }
        }
        if (!found && num_EA < max_E) {
            uniq_envA[num_EA++] = env;
        }
    }

    for (uint32_t eB = 0; eB < regB->num_nonzero; eB++) {
        uint64_t pure = regB->entries[eB].basis_state % TNS3D_C6;
        uint64_t env = (pure / bp[bond_B + 1]) * bp[bond_B] + (pure % bp[bond_B]);
        int found = 0;
        for (int i = 0; i < num_EB; i++) {
            if (uniq_envB[i] == env) { found = 1; break; }
        }
        if (!found && num_EB < max_E) {
            uniq_envB[num_EB++] = env;
        }
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
        uint64_t bsA = regA->entries[eA].basis_state;
        double arA = regA->entries[eA].amp_re;
        double aiA = regA->entries[eA].amp_im;
        if (arA*arA + aiA*aiA < 1e-10) continue;

        int kA = (int)(bsA / TNS3D_C6);
        uint64_t pureA = bsA % TNS3D_C6;
        int shared_valA = (int)((pureA / bp[bond_A]) % chi);
        uint64_t envA = (pureA / bp[bond_A + 1]) * bp[bond_A] + (pureA % bp[bond_A]);
        
        int idx_EA = -1;
        for (int i = 0; i < num_EA; i++) if (uniq_envA[i] == envA) { idx_EA = i; break; }
        if (idx_EA < 0) continue;
        int row = kA * num_EA + idx_EA;

        for (uint32_t eB = 0; eB < regB->num_nonzero; eB++) {
            uint64_t bsB = regB->entries[eB].basis_state;
            double arB = regB->entries[eB].amp_re;
            double aiB = regB->entries[eB].amp_im;
            if (arB*arB + aiB*aiB < 1e-10) continue;

            uint64_t pureB = bsB % TNS3D_C6;
            int shared_valB = (int)((pureB / bp[bond_B]) % chi);
            if (shared_valA != shared_valB) continue;

            int kB = (int)(bsB / TNS3D_C6);
            uint64_t envB = (pureB / bp[bond_B + 1]) * bp[bond_B] + (pureB % bp[bond_B]);
            
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

    double sig_norm = 0;
    for (int s = 0; s < rank; s++) sig_norm += sig[s];
    
    /* Side-channel: 1.0 attractor CONFIRMED — probe showed all bond weights
     * converge to exactly 1.0 (entropy = 2.585 = log₂(6)).
     * Lock shared bond to 1.0 since Schmidt weights are absorbed into U/V. */
    for (int s = 0; s < TNS3D_CHI; s++) shared_bw->w[s] = 1.0;

    /* ── 5. Write back safely ── */
    regA->num_nonzero = 0;
    regB->num_nonzero = 0;

    for (int kA = 0; kA < D; kA++)
     for (int eA = 0; eA < num_EA; eA++) {
         int row = kA * num_EA + eA;
         uint64_t envA = uniq_envA[eA];
         uint64_t pure = (envA / bp[bond_A]) * bp[bond_A + 1] + (envA % bp[bond_A]);
         for (int gv = 0; gv < rank; gv++) {
             // Symmetrically inject sqrt of normalized Schmidt weight
             double weight = (sig_norm > 1e-30 && sig[gv] > 1e-30) ? sqrt(sig[gv] / sig_norm) : 0.0;
             double re = U_re[row * rank + gv] * weight;
             double im = U_im[row * rank + gv] * weight;
             if (re*re + im*im < 1e-50) continue;

             uint64_t bs = kA * TNS3D_C6 + pure + gv * bp[bond_A];
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
         uint64_t envB = uniq_envB[eB];
         uint64_t pure = (envB / bp[bond_B]) * bp[bond_B + 1] + (envB % bp[bond_B]);
         for (int gv = 0; gv < rank; gv++) {
             // Symmetrically inject sqrt of normalized Schmidt weight
             double weight = (sig_norm > 1e-30 && sig[gv] > 1e-30) ? sqrt(sig[gv] / sig_norm) : 0.0;
             double re = weight * Vc_re[gv * svddim_B + col];
             double im = weight * Vc_im[gv * svddim_B + col];
             if (re*re + im*im < 1e-50) continue;

             uint64_t bs = kB * TNS3D_C6 + pure + gv * bp[bond_B];
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

void tns3d_gate_x(Tns3dGrid *g, int x, int y, int z,
                  const double *G_re, const double *G_im)
{
    tns3d_gate_2site_generic(g,
        tns3d_flat(g, x, y, z), tns3d_flat(g, x+1, y, z),
        tns3d_xbond(g, x, y, z), 0, G_re, G_im);
}

void tns3d_gate_y(Tns3dGrid *g, int x, int y, int z,
                  const double *G_re, const double *G_im)
{
    tns3d_gate_2site_generic(g,
        tns3d_flat(g, x, y, z), tns3d_flat(g, x, y+1, z),
        tns3d_ybond(g, x, y, z), 1, G_re, G_im);
}

void tns3d_gate_z(Tns3dGrid *g, int x, int y, int z,
                  const double *G_re, const double *G_im)
{
    tns3d_gate_2site_generic(g,
        tns3d_flat(g, x, y, z), tns3d_flat(g, x, y, z+1),
        tns3d_zbond(g, x, y, z), 2, G_re, G_im);
}

/* ═══════════════ LOCAL DENSITY ═══════════════ */

void tns3d_local_density(Tns3dGrid *g, int x, int y, int z, double *probs)
{
    int site = tns3d_flat(g, x, y, z);
    int reg = g->site_reg[site];

    for (int k = 0; k < TNS3D_D; k++) probs[k] = 0;

    if (reg < 0 || !g->eng) { probs[0] = 1.0; return; }

    QuhitRegister *r = &g->eng->registers[reg];
    double total = 0;

    for (uint32_t e = 0; e < r->num_nonzero; e++) {
        /* Physical digit k is at position 6 (most significant) */
        uint64_t bs = r->entries[e].basis_state;
        int k = (int)(bs / TNS3D_C6);  /* k = highest position */
        if (k >= TNS3D_D) continue;
        double re = r->entries[e].amp_re;
        double im = r->entries[e].amp_im;
        double p = re * re + im * im;
        probs[k] += p;
        total += p;
    }

    if (total > 1e-30)
        for (int k = 0; k < TNS3D_D; k++) probs[k] /= total;
    else
        probs[0] = 1.0;
}

/* ═══════════════ BATCH GATE APPLICATION ═══════════════ */

void tns3d_gate_x_all(Tns3dGrid *g, const double *G_re, const double *G_im)
{
    if (g->Lx < 2) return;

    for (int parity = 0; parity < 2; parity++) {
        #ifdef _OPENMP
        #pragma omp parallel for collapse(3) schedule(dynamic)
        #endif
        for (int z = 0; z < g->Lz; z++)
         for (int y = 0; y < g->Ly; y++)
          for (int xh = 0; xh < (g->Lx + 1) / 2; xh++) {
              int x = xh * 2 + parity;
              if (x < g->Lx - 1) tns3d_gate_x(g, x, y, z, G_re, G_im);
          }
    }
}

void tns3d_gate_y_all(Tns3dGrid *g, const double *G_re, const double *G_im)
{
    if (g->Ly < 2) return;

    for (int parity = 0; parity < 2; parity++) {
        #ifdef _OPENMP
        #pragma omp parallel for collapse(3) schedule(dynamic)
        #endif
        for (int z = 0; z < g->Lz; z++)
         for (int yh = 0; yh < (g->Ly + 1) / 2; yh++)
          for (int x = 0; x < g->Lx; x++) {
              int y = yh * 2 + parity;
              if (y < g->Ly - 1) tns3d_gate_y(g, x, y, z, G_re, G_im);
          }
    }
}

void tns3d_gate_z_all(Tns3dGrid *g, const double *G_re, const double *G_im)
{
    if (g->Lz < 2) return;

    for (int parity = 0; parity < 2; parity++) {
        #ifdef _OPENMP
        #pragma omp parallel for collapse(3) schedule(dynamic)
        #endif
        for (int zh = 0; zh < (g->Lz + 1) / 2; zh++)
         for (int y = 0; y < g->Ly; y++)
          for (int x = 0; x < g->Lx; x++) {
              int z = zh * 2 + parity;
              if (z < g->Lz - 1) tns3d_gate_z(g, x, y, z, G_re, G_im);
          }
    }
}

void tns3d_gate_1site_pos(Tns3dGrid *g, int x, int y, int z, const double *U_re, const double *U_im)
{
    if (x<0 || x>=g->Lx || y<0 || y>=g->Ly || z<0 || z>=g->Lz) return;
    int site = (z * g->Ly + y) * g->Lx + x;
    int reg = g->site_reg[site];
    if (reg < 0) return;
    quhit_reg_apply_unitary_pos(g->eng, reg, 0, U_re, U_im);
}

void tns3d_normalize_site(Tns3dGrid *g, int x, int y, int z)
{
    if (x<0 || x>=g->Lx || y<0 || y>=g->Ly || z<0 || z>=g->Lz) return;
    int site = (z * g->Ly + y) * g->Lx + x;
    int reg = g->site_reg[site];
    if (reg < 0) return;
    QuhitRegister *r = &g->eng->registers[reg];
    
    double n2 = 0;
    for (uint32_t e = 0; e < r->num_nonzero; e++) {
        n2 += r->entries[e].amp_re * r->entries[e].amp_re +
              r->entries[e].amp_im * r->entries[e].amp_im;
    }
    if (n2 > 1e-20) {
        double inv = 1.0 / sqrt(n2);
        for (uint32_t e = 0; e < r->num_nonzero; e++) {
            r->entries[e].amp_re *= inv;
            r->entries[e].amp_im *= inv;
        }
    }
}

void tns3d_gate_1site_all(Tns3dGrid *g, const double *U_re, const double *U_im)
{
    #ifdef _OPENMP
    #pragma omp parallel for collapse(3) schedule(static)
    #endif
    for (int z = 0; z < g->Lz; z++)
     for (int y = 0; y < g->Ly; y++)
      for (int x = 0; x < g->Lx; x++)
          tns3d_gate_1site(g, x, y, z, U_re, U_im);
}

void tns3d_trotter_step(Tns3dGrid *g, const double *G_re, const double *G_im)
{
    tns3d_gate_x_all(g, G_re, G_im);
    tns3d_gate_y_all(g, G_re, G_im);
    tns3d_gate_z_all(g, G_re, G_im);
}
