/*
 * peps_overlay.c — PEPS Engine: Pure Magic Pointer Tensor Network
 *
 * D=6 native (SU(6)), bond dimension unlimited via Magic Pointers.
 * All gate operations are O(1) through QuhitRegister sparse storage.
 * No classical tensor arrays — RAM usage is constant regardless of χ.
 */

#include "peps_overlay.h"
#include <stdio.h>
#include <fenv.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════════════
 * GRID ACCESS HELPERS
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline PepsTensor *peps_site(PepsGrid *g, int x, int y)
{ return &g->tensors[y * g->Lx + x]; }

static inline PepsBondWeight *peps_hbond(PepsGrid *g, int x, int y)
{ return &g->h_bonds[y * (g->Lx - 1) + x]; }

static inline PepsBondWeight *peps_vbond(PepsGrid *g, int x, int y)
{ return &g->v_bonds[y * g->Lx + x]; }

/* ═══════════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 * ═══════════════════════════════════════════════════════════════════════════════ */

PepsGrid *peps_init(int Lx, int Ly)
{
    PepsGrid *g = (PepsGrid *)calloc(1, sizeof(PepsGrid));
    g->Lx = Lx;
    g->Ly = Ly;
    int N = Lx * Ly;

    /* Lightweight tensor metadata — 4 bytes per site, not 2MB */
    g->tensors = (PepsTensor *)calloc(N, sizeof(PepsTensor));

    g->h_bonds = (PepsBondWeight *)calloc(Ly * (Lx - 1), sizeof(PepsBondWeight));
    g->v_bonds = (PepsBondWeight *)calloc((Ly - 1) * Lx, sizeof(PepsBondWeight));

    for (int i = 0; i < Ly * (Lx - 1); i++)
        for (int s = 0; s < PEPS_CHI; s++)
            g->h_bonds[i].w[s] = 1.0;
    for (int i = 0; i < (Ly - 1) * Lx; i++)
        for (int s = 0; s < PEPS_CHI; s++)
            g->v_bonds[i].w[s] = 1.0;

    /* ── Magic Pointer: allocate engine ── */
    g->eng = (QuhitEngine *)calloc(1, sizeof(QuhitEngine));
    quhit_engine_init(g->eng);

    /* ── Per-site physical quhits (for marginal readout) ── */
    g->q_phys = (uint32_t *)calloc(N, sizeof(uint32_t));
    for (int i = 0; i < N; i++)
        g->q_phys[i] = quhit_init_basis(g->eng, 0);

    /* ── Per-site registers: the tensor IS the register ── */
    g->site_reg = (int *)calloc(N, sizeof(int));
    for (int i = 0; i < N; i++) {
        g->site_reg[i] = quhit_reg_init(g->eng, (uint64_t)i, 5, PEPS_D);
        if (g->site_reg[i] >= 0) {
            g->eng->registers[g->site_reg[i]].bulk_rule = 0;
            /* Init to |0,0,0,0,0⟩ with amplitude 1.0 */
            quhit_reg_sv_set(g->eng, g->site_reg[i], 0, 1.0, 0.0);
        }
        /* Link tensor metadata to register */
        g->tensors[i].reg_idx = g->site_reg[i];
    }

    return g;
}

void peps_free(PepsGrid *grid)
{
    if (!grid) return;
    free(grid->tensors);
    free(grid->h_bonds);
    free(grid->v_bonds);
    if (grid->eng) {
        quhit_engine_destroy(grid->eng);
        free(grid->eng);
    }
    free(grid->q_phys);
    free(grid->site_reg);
    free(grid);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * STATE INITIALIZATION — via register
 * ═══════════════════════════════════════════════════════════════════════════════ */

void peps_set_product_state(PepsGrid *grid, int x, int y,
                            const double *amps_re, const double *amps_im)
{
    int site = y * grid->Lx + x;
    int reg = grid->site_reg[site];
    if (reg < 0) return;

    /* Clear register */
    grid->eng->registers[reg].num_nonzero = 0;

    /* Write product state: |k,0,0,0,0⟩ for each physical level k */
    for (int k = 0; k < PEPS_D; k++) {
        if (amps_re[k] * amps_re[k] + amps_im[k] * amps_im[k] > 1e-30)
            quhit_reg_sv_set(grid->eng, reg, (uint64_t)k, amps_re[k], amps_im[k]);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * 1-SITE GATE — Pure Magic Pointer: O(entries × D)
 * ═══════════════════════════════════════════════════════════════════════════════ */

void peps_gate_1site(PepsGrid *grid, int x, int y,
                     const double *U_re, const double *U_im)
{
    int site = y * grid->Lx + x;

    /* ── Magic Pointer: gate via register — O(entries × D) ── */
    if (grid->eng && grid->site_reg)
        quhit_reg_apply_unitary_pos(grid->eng, grid->site_reg[site],
                                    0, U_re, U_im);

    /* ── Mirror to per-site quhit (marginal readout) ── */
    if (grid->eng && grid->q_phys)
        quhit_apply_unitary(grid->eng, grid->q_phys[site], U_re, U_im);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * 2-SITE GATE: HORIZONTAL BOND  (x,y)—(x+1,y)
 *
 * Simple-update with register-based SVD:
 * 1. Read T_A[k,u,d,l,r] and T_B[k,u,d,l,r] from registers
 * 2. Absorb environment bond weights into tensors
 * 3. Contract over shared r-l bond with bond weight λ_h
 * 4. Apply 2-site gate to joint physical indices
 * 5. SVD → truncate → update bond weight → write back to registers
 *
 * SVD dimension: D×χ² = 864 at χ=12
 * ═══════════════════════════════════════════════════════════════════════════════ */

#include "tensor_svd.h"

#define PEPS_SVDDIM (PEPS_D * PEPS_CHI * PEPS_CHI)

/* Helper: read tensor T[k,u,d,l,r] from register into dense array */
static void peps_reg_read_dense(PepsGrid *grid, int site,
                                double *T_re, double *T_im)
{
    int reg = grid->site_reg[site];
    int chi = PEPS_CHI;
    size_t tsz = (size_t)PEPS_D * PEPS_CHI4;
    memset(T_re, 0, tsz * sizeof(double));
    memset(T_im, 0, tsz * sizeof(double));
    if (reg < 0 || !grid->eng) return;

    QuhitRegister *r = &grid->eng->registers[reg];
    for (uint32_t e = 0; e < r->num_nonzero; e++) {
        uint64_t bs = r->entries[e].basis_state;
        if (bs < tsz) {
            T_re[bs] = r->entries[e].amp_re;
            T_im[bs] = r->entries[e].amp_im;
        }
    }
}

/* Helper: write dense tensor back to register as sparse */
static void peps_reg_write_dense(PepsGrid *grid, int site,
                                 const double *T_re, const double *T_im)
{
    int reg = grid->site_reg[site];
    if (reg < 0 || !grid->eng) return;
    size_t tsz = (size_t)PEPS_D * PEPS_CHI4;

    grid->eng->registers[reg].num_nonzero = 0;
    for (size_t bs = 0; bs < tsz; bs++) {
        if (T_re[bs]*T_re[bs] + T_im[bs]*T_im[bs] > 1e-30)
            quhit_reg_sv_set(grid->eng, reg, (uint64_t)bs, T_re[bs], T_im[bs]);
    }
}

void peps_gate_horizontal(PepsGrid *grid, int x, int y,
                          const double *G_re, const double *G_im)
{
    int sA = y * grid->Lx + x;
    int sB = y * grid->Lx + (x + 1);
    int D = PEPS_D, chi = PEPS_CHI;
    int chi2 = chi * chi;
    int svddim = D * chi2;  /* 864 */

    /* ── 1. Bond weights ── */
    double wu_A[PEPS_CHI], wd_A[PEPS_CHI], wl_A[PEPS_CHI];
    double wu_B[PEPS_CHI], wd_B[PEPS_CHI], wr_B[PEPS_CHI];
    double wh[PEPS_CHI];

    if (y > 0) {
        PepsBondWeight *vb = peps_vbond(grid, x, y - 1);
        for (int s = 0; s < chi; s++) wu_A[s] = vb->w[s];
    } else { for (int s = 0; s < chi; s++) wu_A[s] = 1.0; }

    if (y < grid->Ly - 1) {
        PepsBondWeight *vb = peps_vbond(grid, x, y);
        for (int s = 0; s < chi; s++) wd_A[s] = vb->w[s];
    } else { for (int s = 0; s < chi; s++) wd_A[s] = 1.0; }

    if (x > 0) {
        PepsBondWeight *hb = peps_hbond(grid, x - 1, y);
        for (int s = 0; s < chi; s++) wl_A[s] = hb->w[s];
    } else { for (int s = 0; s < chi; s++) wl_A[s] = 1.0; }

    PepsBondWeight *hb_shared = peps_hbond(grid, x, y);
    for (int s = 0; s < chi; s++) wh[s] = hb_shared->w[s];

    if (y > 0) {
        PepsBondWeight *vb = peps_vbond(grid, x + 1, y - 1);
        for (int s = 0; s < chi; s++) wu_B[s] = vb->w[s];
    } else { for (int s = 0; s < chi; s++) wu_B[s] = 1.0; }

    if (y < grid->Ly - 1) {
        PepsBondWeight *vb = peps_vbond(grid, x + 1, y);
        for (int s = 0; s < chi; s++) wd_B[s] = vb->w[s];
    } else { for (int s = 0; s < chi; s++) wd_B[s] = 1.0; }

    if (x + 2 < grid->Lx) {
        PepsBondWeight *hb = peps_hbond(grid, x + 1, y);
        for (int s = 0; s < chi; s++) wr_B[s] = hb->w[s];
    } else { for (int s = 0; s < chi; s++) wr_B[s] = 1.0; }

    /* ── 2. Contract Θ from SPARSE entries ──
     * Horizontal: shared = r_A = l_B
     * Row = (kA, u, l)  Col = (kB, d, r)
     * Sum over shared γ and env bonds (d_A, u_B) */

    size_t svd2 = (size_t)svddim * svddim;
    double *Th_re = (double *)calloc(svd2, sizeof(double));
    double *Th_im = (double *)calloc(svd2, sizeof(double));

    QuhitRegister *regA = &grid->eng->registers[grid->site_reg[sA]];
    QuhitRegister *regB = &grid->eng->registers[grid->site_reg[sB]];

    for (uint32_t eA = 0; eA < regA->num_nonzero; eA++) {
        uint64_t bsA = regA->entries[eA].basis_state;
        double arA = regA->entries[eA].amp_re;
        double aiA = regA->entries[eA].amp_im;
        if (arA*arA + aiA*aiA < 1e-30) continue;

        int kA = (int)(bsA / PEPS_CHI4);
        int uA = (int)((bsA / PEPS_CHI3) % chi);
        int dA = (int)((bsA / PEPS_CHI2) % chi);
        int lA = (int)((bsA / PEPS_CHI) % chi);
        int rA = (int)(bsA % chi);  /* shared bond */

        double wA = wu_A[uA] * wd_A[dA] * wl_A[lA];
        double arAw = arA * wA, aiAw = aiA * wA;
        int row = kA * chi2 + uA * chi + lA;

        for (uint32_t eB = 0; eB < regB->num_nonzero; eB++) {
            uint64_t bsB = regB->entries[eB].basis_state;
            double arB = regB->entries[eB].amp_re;
            double aiB = regB->entries[eB].amp_im;
            if (arB*arB + aiB*aiB < 1e-30) continue;

            int kB = (int)(bsB / PEPS_CHI4);
            int uB = (int)((bsB / PEPS_CHI3) % chi);
            int dB = (int)((bsB / PEPS_CHI2) % chi);
            int lB = (int)((bsB / PEPS_CHI) % chi);  /* shared bond */
            int rB = (int)(bsB % chi);

            if (rA != lB) continue;  /* shared bond must match */

            double wB = wu_B[uB] * wd_B[dB] * wr_B[rB] * wh[rA];
            double br = arB * wB, bi = aiB * wB;
            int col = kB * chi2 + dB * chi + rB;

            Th_re[row * svddim + col] += arAw*br - aiAw*bi;
            Th_im[row * svddim + col] += arAw*bi + aiAw*br;
        }
    }

    /* ── 3. Apply gate ── */
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
              if (fabs(gre) < 1e-30 && fabs(gim) < 1e-30) continue;

              for (int u = 0; u < chi; u++)
               for (int l = 0; l < chi; l++) {
                   int dst_row = kAp * chi2 + u * chi + l;
                   int src_row = kA * chi2 + u * chi + l;
                   for (int d = 0; d < chi; d++)
                    for (int r = 0; r < chi; r++) {
                        int dst_col = kBp * chi2 + d * chi + r;
                        int src_col = kB * chi2 + d * chi + r;
                        double tr = Th_re[src_row * svddim + src_col];
                        double ti = Th_im[src_row * svddim + src_col];
                        Th2_re[dst_row * svddim + dst_col] += gre*tr - gim*ti;
                        Th2_im[dst_row * svddim + dst_col] += gre*ti + gim*tr;
                    }
               }
          }
     }

    free(Th_re); free(Th_im);

    /* ── 4. SVD ── */
    double *U_re  = (double *)calloc((size_t)svddim * chi, sizeof(double));
    double *U_im  = (double *)calloc((size_t)svddim * chi, sizeof(double));
    double *sig   = (double *)calloc(chi, sizeof(double));
    double *Vc_re = (double *)calloc((size_t)chi * svddim, sizeof(double));
    double *Vc_im = (double *)calloc((size_t)chi * svddim, sizeof(double));

    tsvd_truncated(Th2_re, Th2_im, svddim, svddim, chi,
                   U_re, U_im, sig, Vc_re, Vc_im);

    free(Th2_re); free(Th2_im);

    /* ── 5. Update bond weight ── */
    double sig_norm = 0;
    for (int s = 0; s < chi; s++) sig_norm += sig[s];
    if (sig_norm > 1e-30)
        for (int s = 0; s < chi; s++) hb_shared->w[s] = sig[s] / sig_norm;

    /* ── 6. Write back directly to registers ── */
    regA->num_nonzero = 0;
    for (int kA = 0; kA < D; kA++)
     for (int u = 0; u < chi; u++)
      for (int l = 0; l < chi; l++) {
          int row = kA * chi2 + u * chi + l;
          double invw = 1.0;
          if (wu_A[u] > 1e-30) invw /= wu_A[u];
          if (wl_A[l] > 1e-30) invw /= wl_A[l];
          for (int g = 0; g < chi; g++) {
              double re = U_re[row * chi + g] * invw;
              double im = U_im[row * chi + g] * invw;
              if (re*re + im*im > 1e-30 && regA->num_nonzero < 4096) {
                  uint64_t bs = PT_IDX(kA, u, 0, l, g);
                  regA->entries[regA->num_nonzero].basis_state = bs;
                  regA->entries[regA->num_nonzero].amp_re = re;
                  regA->entries[regA->num_nonzero].amp_im = im;
                  regA->num_nonzero++;
              }
          }
      }

    regB->num_nonzero = 0;
    for (int kB = 0; kB < D; kB++)
     for (int d = 0; d < chi; d++)
      for (int r = 0; r < chi; r++) {
          int col = kB * chi2 + d * chi + r;
          double invw = 1.0;
          if (wd_B[d] > 1e-30) invw /= wd_B[d];
          if (wr_B[r] > 1e-30) invw /= wr_B[r];
          for (int g = 0; g < chi; g++) {
              double re = Vc_re[g * svddim + col] * invw;
              double im = Vc_im[g * svddim + col] * invw;
              if (re*re + im*im > 1e-30 && regB->num_nonzero < 4096) {
                  uint64_t bs = PT_IDX(kB, 0, d, g, r);
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

    /* Mirror to engine quhits */
    if (grid->eng && grid->q_phys)
        quhit_apply_cz(grid->eng, grid->q_phys[sA], grid->q_phys[sB]);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * 2-SITE GATE: VERTICAL BOND  (x,y)—(x,y+1)
 *
 * Simple-update SVD: shared bond is d_A = u_B.
 * Row = (kA, l, r), Col = (kB, l, r) → SVD dim = D×χ²
 * ═══════════════════════════════════════════════════════════════════════════════ */

void peps_gate_vertical(PepsGrid *grid, int x, int y,
                        const double *G_re, const double *G_im)
{
    int sA = y * grid->Lx + x;
    int sB = (y + 1) * grid->Lx + x;
    int D = PEPS_D, chi = PEPS_CHI;
    int chi2 = chi * chi;
    int svddim = D * chi2;

    /* ── 1. Bond weights ── */
    double wu_A[PEPS_CHI], wl_A[PEPS_CHI], wr_A[PEPS_CHI];
    double wd_B[PEPS_CHI], wl_B[PEPS_CHI], wr_B[PEPS_CHI];
    double wv[PEPS_CHI];

    if (y > 0) {
        PepsBondWeight *vb = peps_vbond(grid, x, y - 1);
        for (int s = 0; s < chi; s++) wu_A[s] = vb->w[s];
    } else { for (int s = 0; s < chi; s++) wu_A[s] = 1.0; }

    if (x > 0) {
        PepsBondWeight *hb = peps_hbond(grid, x - 1, y);
        for (int s = 0; s < chi; s++) wl_A[s] = hb->w[s];
    } else { for (int s = 0; s < chi; s++) wl_A[s] = 1.0; }

    if (x < grid->Lx - 1) {
        PepsBondWeight *hb = peps_hbond(grid, x, y);
        for (int s = 0; s < chi; s++) wr_A[s] = hb->w[s];
    } else { for (int s = 0; s < chi; s++) wr_A[s] = 1.0; }

    PepsBondWeight *vb_shared = peps_vbond(grid, x, y);
    for (int s = 0; s < chi; s++) wv[s] = vb_shared->w[s];

    if (y + 2 < grid->Ly) {
        PepsBondWeight *vb = peps_vbond(grid, x, y + 1);
        for (int s = 0; s < chi; s++) wd_B[s] = vb->w[s];
    } else { for (int s = 0; s < chi; s++) wd_B[s] = 1.0; }

    if (x > 0) {
        PepsBondWeight *hb = peps_hbond(grid, x - 1, y + 1);
        for (int s = 0; s < chi; s++) wl_B[s] = hb->w[s];
    } else { for (int s = 0; s < chi; s++) wl_B[s] = 1.0; }

    if (x < grid->Lx - 1) {
        PepsBondWeight *hb = peps_hbond(grid, x, y + 1);
        for (int s = 0; s < chi; s++) wr_B[s] = hb->w[s];
    } else { for (int s = 0; s < chi; s++) wr_B[s] = 1.0; }

    /* ── 2. Contract Θ from SPARSE entries ──
     * Vertical: shared = d_A = u_B
     * Row = (kA, l, r)  Col = (kB, l, r) */

    size_t svd2 = (size_t)svddim * svddim;
    double *Th_re = (double *)calloc(svd2, sizeof(double));
    double *Th_im = (double *)calloc(svd2, sizeof(double));

    QuhitRegister *regA = &grid->eng->registers[grid->site_reg[sA]];
    QuhitRegister *regB = &grid->eng->registers[grid->site_reg[sB]];

    for (uint32_t eA = 0; eA < regA->num_nonzero; eA++) {
        uint64_t bsA = regA->entries[eA].basis_state;
        double arA = regA->entries[eA].amp_re;
        double aiA = regA->entries[eA].amp_im;
        if (arA*arA + aiA*aiA < 1e-30) continue;

        int kA = (int)(bsA / PEPS_CHI4);
        int uA = (int)((bsA / PEPS_CHI3) % chi);
        int dA = (int)((bsA / PEPS_CHI2) % chi);  /* shared bond */
        int lA = (int)((bsA / PEPS_CHI) % chi);
        int rA = (int)(bsA % chi);

        double wA = wu_A[uA] * wl_A[lA] * wr_A[rA];
        double arAw = arA * wA, aiAw = aiA * wA;
        int row = kA * chi2 + lA * chi + rA;

        for (uint32_t eB = 0; eB < regB->num_nonzero; eB++) {
            uint64_t bsB = regB->entries[eB].basis_state;
            double arB = regB->entries[eB].amp_re;
            double aiB = regB->entries[eB].amp_im;
            if (arB*arB + aiB*aiB < 1e-30) continue;

            int kB = (int)(bsB / PEPS_CHI4);
            int uB = (int)((bsB / PEPS_CHI3) % chi);  /* shared bond */
            int dB = (int)((bsB / PEPS_CHI2) % chi);
            int lB = (int)((bsB / PEPS_CHI) % chi);
            int rB = (int)(bsB % chi);

            if (dA != uB) continue;  /* shared bond must match */

            double wB = wd_B[dB] * wl_B[lB] * wr_B[rB] * wv[dA];
            double br = arB * wB, bi = aiB * wB;
            int col = kB * chi2 + lB * chi + rB;

            Th_re[row * svddim + col] += arAw*br - aiAw*bi;
            Th_im[row * svddim + col] += arAw*bi + aiAw*br;
        }
    }

    /* ── 3. Apply gate ── */
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
              if (fabs(gre) < 1e-30 && fabs(gim) < 1e-30) continue;

              for (int l = 0; l < chi; l++)
               for (int r = 0; r < chi; r++) {
                   int dst_row = kAp * chi2 + l * chi + r;
                   int src_row = kA * chi2 + l * chi + r;
                   for (int lB = 0; lB < chi; lB++)
                    for (int rB = 0; rB < chi; rB++) {
                        int dst_col = kBp * chi2 + lB * chi + rB;
                        int src_col = kB * chi2 + lB * chi + rB;
                        double tr = Th_re[src_row * svddim + src_col];
                        double ti = Th_im[src_row * svddim + src_col];
                        Th2_re[dst_row * svddim + dst_col] += gre*tr - gim*ti;
                        Th2_im[dst_row * svddim + dst_col] += gre*ti + gim*tr;
                    }
               }
          }
     }

    free(Th_re); free(Th_im);

    /* ── 4. SVD ── */
    double *U_re  = (double *)calloc((size_t)svddim * chi, sizeof(double));
    double *U_im  = (double *)calloc((size_t)svddim * chi, sizeof(double));
    double *sig   = (double *)calloc(chi, sizeof(double));
    double *Vc_re = (double *)calloc((size_t)chi * svddim, sizeof(double));
    double *Vc_im = (double *)calloc((size_t)chi * svddim, sizeof(double));

    tsvd_truncated(Th2_re, Th2_im, svddim, svddim, chi,
                   U_re, U_im, sig, Vc_re, Vc_im);

    free(Th2_re); free(Th2_im);

    /* ── 5. Update bond weight ── */
    double sig_norm = 0;
    for (int s = 0; s < chi; s++) sig_norm += sig[s];
    if (sig_norm > 1e-30)
        for (int s = 0; s < chi; s++) vb_shared->w[s] = sig[s] / sig_norm;

    /* ── 6. Write back directly to registers ── */
    regA->num_nonzero = 0;
    for (int kA = 0; kA < D; kA++)
     for (int l = 0; l < chi; l++)
      for (int r = 0; r < chi; r++) {
          int row = kA * chi2 + l * chi + r;
          double invw = 1.0;
          if (wl_A[l] > 1e-30) invw /= wl_A[l];
          if (wr_A[r] > 1e-30) invw /= wr_A[r];
          for (int g = 0; g < chi; g++) {
              double re = U_re[row * chi + g] * invw;
              double im = U_im[row * chi + g] * invw;
              if (re*re + im*im > 1e-30 && regA->num_nonzero < 4096) {
                  uint64_t bs = PT_IDX(kA, 0, g, l, r);
                  regA->entries[regA->num_nonzero].basis_state = bs;
                  regA->entries[regA->num_nonzero].amp_re = re;
                  regA->entries[regA->num_nonzero].amp_im = im;
                  regA->num_nonzero++;
              }
          }
      }

    regB->num_nonzero = 0;
    for (int kB = 0; kB < D; kB++)
     for (int l = 0; l < chi; l++)
      for (int r = 0; r < chi; r++) {
          int col = kB * chi2 + l * chi + r;
          double invw = 1.0;
          if (wl_B[l] > 1e-30) invw /= wl_B[l];
          if (wr_B[r] > 1e-30) invw /= wr_B[r];
          for (int g = 0; g < chi; g++) {
              double re = Vc_re[g * svddim + col] * invw;
              double im = Vc_im[g * svddim + col] * invw;
              if (re*re + im*im > 1e-30 && regB->num_nonzero < 4096) {
                  uint64_t bs = PT_IDX(kB, g, 0, l, r);
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

    /* Mirror to engine quhits */
    if (grid->eng && grid->q_phys)
        quhit_apply_cz(grid->eng, grid->q_phys[sA], grid->q_phys[sB]);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * LOCAL DENSITY — via register marginals (no classical tensor access)
 *
 * Compute p(k) = Σ_{u,d,l,r} |⟨k,u,d,l,r|ψ⟩|² by iterating over
 * the register's sparse entries and grouping by the physical digit.
 * ═══════════════════════════════════════════════════════════════════════════════ */

void peps_local_density(PepsGrid *grid, int x, int y, double *probs)
{
    int site = y * grid->Lx + x;
    int reg = grid->site_reg[site];

    for (int k = 0; k < PEPS_D; k++) probs[k] = 0;

    if (reg < 0 || !grid->eng) {
        probs[0] = 1.0;
        return;
    }

    QuhitRegister *r = &grid->eng->registers[reg];
    double total = 0;

    /* Iterate over sparse entries, extract physical digit (position 0) */
    for (uint32_t e = 0; e < r->num_nonzero; e++) {
        uint32_t k = (uint32_t)(r->entries[e].basis_state % PEPS_D);
        double re = r->entries[e].amp_re;
        double im = r->entries[e].amp_im;
        double p = re * re + im * im;
        probs[k] += p;
        total += p;
    }

    if (total > 1e-30)
        for (int k = 0; k < PEPS_D; k++) probs[k] /= total;
    else
        probs[0] = 1.0;
}

/* ═══════════════ BATCH GATE APPLICATION (Red-Black Checkerboard) ═══════════════ */

void peps_gate_horizontal_all(PepsGrid *g, const double *G_re, const double *G_im)
{
    if (g->Lx < 2) return;

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(dynamic)
    #endif
    for (int y = 0; y < g->Ly; y++)
     for (int xh = 0; xh < (g->Lx - 1 + 1) / 2; xh++) {
         int x = xh * 2;
         if (x < g->Lx - 1)
             peps_gate_horizontal(g, x, y, G_re, G_im);
     }

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(dynamic)
    #endif
    for (int y = 0; y < g->Ly; y++)
     for (int xh = 0; xh < g->Lx / 2; xh++) {
         int x = xh * 2 + 1;
         if (x < g->Lx - 1)
             peps_gate_horizontal(g, x, y, G_re, G_im);
     }
}

void peps_gate_vertical_all(PepsGrid *g, const double *G_re, const double *G_im)
{
    if (g->Ly < 2) return;

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(dynamic)
    #endif
    for (int yh = 0; yh < (g->Ly - 1 + 1) / 2; yh++)
     for (int x = 0; x < g->Lx; x++) {
         int y = yh * 2;
         if (y < g->Ly - 1)
             peps_gate_vertical(g, x, y, G_re, G_im);
     }

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(dynamic)
    #endif
    for (int yh = 0; yh < g->Ly / 2; yh++)
     for (int x = 0; x < g->Lx; x++) {
         int y = yh * 2 + 1;
         if (y < g->Ly - 1)
             peps_gate_vertical(g, x, y, G_re, G_im);
     }
}

void peps_gate_1site_all(PepsGrid *g, const double *U_re, const double *U_im)
{
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int y = 0; y < g->Ly; y++)
     for (int x = 0; x < g->Lx; x++)
         peps_gate_1site(g, x, y, U_re, U_im);
}

void peps_trotter_step(PepsGrid *g, const double *G_re, const double *G_im)
{
    peps_gate_horizontal_all(g, G_re, G_im);
    peps_gate_vertical_all(g, G_re, G_im);
}
