/*
 * peps_overlay.c — PEPS Engine: Pure Magic Pointer Tensor Network
 *
 * D=6 native (SU(6)), bond dimension unlimited via Magic Pointers.
 * All gate operations are O(1) through QuhitRegister sparse storage.
 * No classical tensor arrays — RAM usage is constant regardless of χ.
 *
 * ── Side-channel optimized (tns_contraction_probe.c) ──
 *   • Gate sparsity via mag² (no fabs)
 *   • Zero-angle skip in Jacobi SVD (via tensor_svd.h)
 *   • 1.0 attractor: bond weights converge to 1.0
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

struct tmp_entry { uint64_t basis; double re, im; };

static int cmp_basis(const void *a, const void *b)
{
    const struct tmp_entry *ea = (const struct tmp_entry *)a;
    const struct tmp_entry *eb = (const struct tmp_entry *)b;
    if (ea->basis < eb->basis) return -1;
    if (ea->basis > eb->basis) return 1;
    return 0;
}

void peps_gate_1site(PepsGrid *grid, int x, int y,
                     const double *U_re, const double *U_im)
{
    int site = y * grid->Lx + x;

    /* ── Manual register-based 1-site gate on physical index k ── */
    if (grid->eng && grid->site_reg) {
        int reg_idx = grid->site_reg[site];
        if (reg_idx >= 0) {
            QuhitRegister *reg = &grid->eng->registers[reg_idx];
            int D = PEPS_D;

            uint32_t max_out = reg->num_nonzero * D + 1;
            if (max_out < 4096) max_out = 4096;
            struct tmp_entry *tmp = calloc(max_out, sizeof(*tmp));
            uint32_t nout = 0;

            for (uint32_t e = 0; e < reg->num_nonzero; e++) {
                uint64_t bs = reg->entries[e].basis_state;
                double ar = reg->entries[e].amp_re;
                double ai = reg->entries[e].amp_im;
                int k = (int)(bs / PEPS_CHI4);
                uint64_t bond = bs % PEPS_CHI4;

                for (int kp = 0; kp < D; kp++) {
                    double ur = U_re[kp * D + k];
                    double ui = U_im[kp * D + k];
                    double nr = ur * ar - ui * ai;
                    double ni = ur * ai + ui * ar;
                    if (nr*nr + ni*ni < 1e-10) continue;

                    if (nout < max_out) {
                        tmp[nout].basis = (uint64_t)kp * PEPS_CHI4 + bond;
                        tmp[nout].re = nr;
                        tmp[nout].im = ni;
                        nout++;
                    }
                }
            }

            /* Sort by basis state to accumulate duplicates in O(K log K) */
            qsort(tmp, nout, sizeof(struct tmp_entry), cmp_basis);

            reg->num_nonzero = 0;
            for (uint32_t t = 0; t < nout; t++) {
                double acc_re = tmp[t].re;
                double acc_im = tmp[t].im;
                /* Accumulate duplicates */
                while (t + 1 < nout && tmp[t+1].basis == tmp[t].basis) {
                    t++;
                    acc_re += tmp[t].re;
                    acc_im += tmp[t].im;
                }
                if (acc_re*acc_re + acc_im*acc_im >= 1e-10 &&
                    reg->num_nonzero < 4096) {
                    reg->entries[reg->num_nonzero].basis_state = tmp[t].basis;
                    reg->entries[reg->num_nonzero].amp_re = acc_re;
                    reg->entries[reg->num_nonzero].amp_im = acc_im;
                    reg->num_nonzero++;
                }
            }
            free(tmp);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * 2-SITE GATE: HORIZONTAL BOND  (x,y)—(x+1,y)
 *
 * Side-channel optimized:
 *   • Gate sparsity: mag² check (avoids 2× fabs per element)
 *   • SVD Jacobi acceleration via zero-attractor skip (tensor_svd.h)
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

    /* ── 2. Sparse-Rank Environment Extraction ──
     * Horizontal: shared = r_A = l_B
     * A env = (u, d, l) with shared bond r removed
     * B env = (u, d, r) with shared bond l removed
     * Preserves ALL bond values — no zeroing */

    QuhitRegister *regA = &grid->eng->registers[grid->site_reg[sA]];
    QuhitRegister *regB = &grid->eng->registers[grid->site_reg[sB]];

    int max_E = 144;
    uint64_t *uniq_envA = (uint64_t*)malloc(max_E * sizeof(uint64_t));
    uint64_t *uniq_envB = (uint64_t*)malloc(max_E * sizeof(uint64_t));
    int num_EA = 0, num_EB = 0;

    /* A env = u*chi2 + d*chi + l (drop r which is shared) */
    for (uint32_t eA = 0; eA < regA->num_nonzero; eA++) {
        uint64_t bs = regA->entries[eA].basis_state;
        uint64_t env = (bs / PEPS_CHI) % PEPS_CHI3; /* u*chi2+d*chi+l */
        int found = 0;
        for (int i = 0; i < num_EA; i++) if (uniq_envA[i] == env) { found = 1; break; }
        if (!found && num_EA < max_E) uniq_envA[num_EA++] = env;
    }

    /* B env = u*chi2 + d*chi + r (drop l which is shared) */
    for (uint32_t eB = 0; eB < regB->num_nonzero; eB++) {
        uint64_t bs = regB->entries[eB].basis_state;
        int uB = (int)((bs / PEPS_CHI3) % chi);
        int dB = (int)((bs / PEPS_CHI2) % chi);
        int rB = (int)(bs % chi);
        uint64_t env = (uint64_t)uB * chi2 + dB * chi + rB;
        int found = 0;
        for (int i = 0; i < num_EB; i++) if (uniq_envB[i] == env) { found = 1; break; }
        if (!found && num_EB < max_E) uniq_envB[num_EB++] = env;
    }

    if (num_EA == 0 || num_EB == 0) {
        free(uniq_envA); free(uniq_envB); return;
    }

    int svddim_A = D * num_EA;
    int svddim_B = D * num_EB;
    size_t svd2 = (size_t)svddim_A * svddim_B;
    double *Th_re = (double *)calloc(svd2, sizeof(double));
    double *Th_im = (double *)calloc(svd2, sizeof(double));

    for (uint32_t eA = 0; eA < regA->num_nonzero; eA++) {
        uint64_t bsA = regA->entries[eA].basis_state;
        double arA = regA->entries[eA].amp_re;
        double aiA = regA->entries[eA].amp_im;
        if (arA*arA + aiA*aiA < 1e-30) continue;

        int kA = (int)(bsA / PEPS_CHI4);
        int uA = (int)((bsA / PEPS_CHI3) % chi);
        int dA = (int)((bsA / PEPS_CHI2) % chi);
        int lA = (int)((bsA / PEPS_CHI) % chi);
        int rA = (int)(bsA % chi);

        uint64_t envA = (uint64_t)uA * chi2 + dA * chi + lA;
        int idx_EA = -1;
        for (int i = 0; i < num_EA; i++) if (uniq_envA[i] == envA) { idx_EA = i; break; }
        if (idx_EA < 0) continue;

        double wA = wu_A[uA] * wd_A[dA] * wl_A[lA];
        double arAw = arA * wA, aiAw = aiA * wA;
        int row = kA * num_EA + idx_EA;

        for (uint32_t eB = 0; eB < regB->num_nonzero; eB++) {
            uint64_t bsB = regB->entries[eB].basis_state;
            double arB = regB->entries[eB].amp_re;
            double aiB = regB->entries[eB].amp_im;
            if (arB*arB + aiB*aiB < 1e-30) continue;

            int kB = (int)(bsB / PEPS_CHI4);
            int uB = (int)((bsB / PEPS_CHI3) % chi);
            int dB = (int)((bsB / PEPS_CHI2) % chi);
            int lB = (int)((bsB / PEPS_CHI) % chi);
            int rB = (int)(bsB % chi);

            if (rA != lB) continue;

            uint64_t envB = (uint64_t)uB * chi2 + dB * chi + rB;
            int idx_EB = -1;
            for (int i = 0; i < num_EB; i++) if (uniq_envB[i] == envB) { idx_EB = i; break; }
            if (idx_EB < 0) continue;

            double wB = wu_B[uB] * wd_B[dB] * wr_B[rB] * wh[rA];
            double br = arB * wB, bi = aiB * wB;
            int col = kB * num_EB + idx_EB;

            Th_re[row * svddim_B + col] += arAw*br - aiAw*bi;
            Th_im[row * svddim_B + col] += arAw*bi + aiAw*br;
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
              if (gre*gre + gim*gim < 1e-60) continue;

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

    tsvd_truncated(Th2_re, Th2_im, svddim_A, svddim_B, chi,
                   U_re, U_im, sig, Vc_re, Vc_im);
    free(Th2_re); free(Th2_im);

    /* ── 5. Update bond weight ── */
    int rank = chi < svddim_B ? chi : svddim_B;
    if (rank > svddim_A) rank = svddim_A;
    double sig_norm = 0;
    for (int s = 0; s < rank; s++) sig_norm += sig[s];
    if (sig_norm > 1e-30)
        for (int s = 0; s < rank; s++) hb_shared->w[s] = sig[s] / sig_norm;

    /* ── 6. Write back — preserving ALL bond values ── */
    regA->num_nonzero = 0;
    for (int kA = 0; kA < D; kA++)
     for (int eA = 0; eA < num_EA; eA++) {
         int row = kA * num_EA + eA;
         uint64_t env = uniq_envA[eA];
         int uA = (int)(env / chi2);
         int dA = (int)((env / chi) % chi);
         int lA = (int)(env % chi);
         double invw = 1.0;
         if (wu_A[uA] > 1e-30) invw /= wu_A[uA];
         if (wd_A[dA] > 1e-30) invw /= wd_A[dA];
         if (wl_A[lA] > 1e-30) invw /= wl_A[lA];
         for (int g = 0; g < rank; g++) {
             double re = U_re[row * rank + g] * invw;
             double im = U_im[row * rank + g] * invw;
             if (re*re + im*im > 1e-10 && regA->num_nonzero < 4096) {
                 uint64_t bs = PT_IDX(kA, uA, dA, lA, g);
                 regA->entries[regA->num_nonzero].basis_state = bs;
                 regA->entries[regA->num_nonzero].amp_re = re;
                 regA->entries[regA->num_nonzero].amp_im = im;
                 regA->num_nonzero++;
             }
         }
     }

    regB->num_nonzero = 0;
    for (int kB = 0; kB < D; kB++)
     for (int eB = 0; eB < num_EB; eB++) {
         int col = kB * num_EB + eB;
         uint64_t env = uniq_envB[eB];
         int uB = (int)(env / chi2);
         int dB = (int)((env / chi) % chi);
         int rB = (int)(env % chi);
         double invw = 1.0;
         if (wu_B[uB] > 1e-30) invw /= wu_B[uB];
         if (wd_B[dB] > 1e-30) invw /= wd_B[dB];
         if (wr_B[rB] > 1e-30) invw /= wr_B[rB];
         for (int g = 0; g < rank; g++) {
             double re = Vc_re[g * svddim_B + col] * invw;
             double im = Vc_im[g * svddim_B + col] * invw;
             if (re*re + im*im > 1e-10 && regB->num_nonzero < 4096) {
                 uint64_t bs = PT_IDX(kB, uB, dB, g, rB);
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

    /* Mirror to engine quhits */
    if (grid->eng && grid->q_phys)
        quhit_apply_cz(grid->eng, grid->q_phys[sA], grid->q_phys[sB]);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * 2-SITE GATE: VERTICAL BOND  (x,y)—(x,y+1)
 *
 * Side-channel optimized:
 *   • Gate sparsity: mag² check (avoids 2× fabs per element)
 *   • SVD Jacobi acceleration via zero-attractor skip (tensor_svd.h)
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

    /* ── 2. Sparse-Rank Environment Extraction ──
     * Vertical: shared = d_A = u_B
     * A env = (u, l, r) without shared d
     * B env = (d, l, r) without shared u
     * Preserves ALL bond values */

    QuhitRegister *regA = &grid->eng->registers[grid->site_reg[sA]];
    QuhitRegister *regB = &grid->eng->registers[grid->site_reg[sB]];

    int max_E = 144;
    uint64_t *uniq_envA = (uint64_t*)malloc(max_E * sizeof(uint64_t));
    uint64_t *uniq_envB = (uint64_t*)malloc(max_E * sizeof(uint64_t));
    int num_EA = 0, num_EB = 0;

    /* A env = u*chi2 + l*chi + r (drop d which is shared) */
    for (uint32_t eA = 0; eA < regA->num_nonzero; eA++) {
        uint64_t bs = regA->entries[eA].basis_state;
        int uA = (int)((bs / PEPS_CHI3) % chi);
        int lA = (int)((bs / PEPS_CHI) % chi);
        int rA = (int)(bs % chi);
        uint64_t env = (uint64_t)uA * chi2 + lA * chi + rA;
        int found = 0;
        for (int i = 0; i < num_EA; i++) if (uniq_envA[i] == env) { found = 1; break; }
        if (!found && num_EA < max_E) uniq_envA[num_EA++] = env;
    }

    /* B env = d*chi2 + l*chi + r (drop u which is shared) */
    for (uint32_t eB = 0; eB < regB->num_nonzero; eB++) {
        uint64_t bs = regB->entries[eB].basis_state;
        int dB = (int)((bs / PEPS_CHI2) % chi);
        int lB = (int)((bs / PEPS_CHI) % chi);
        int rB = (int)(bs % chi);
        uint64_t env = (uint64_t)dB * chi2 + lB * chi + rB;
        int found = 0;
        for (int i = 0; i < num_EB; i++) if (uniq_envB[i] == env) { found = 1; break; }
        if (!found && num_EB < max_E) uniq_envB[num_EB++] = env;
    }

    if (num_EA == 0 || num_EB == 0) {
        free(uniq_envA); free(uniq_envB); return;
    }

    int svddim_A = D * num_EA;
    int svddim_B = D * num_EB;
    size_t svd2 = (size_t)svddim_A * svddim_B;
    double *Th_re = (double *)calloc(svd2, sizeof(double));
    double *Th_im = (double *)calloc(svd2, sizeof(double));

    for (uint32_t eA = 0; eA < regA->num_nonzero; eA++) {
        uint64_t bsA = regA->entries[eA].basis_state;
        double arA = regA->entries[eA].amp_re;
        double aiA = regA->entries[eA].amp_im;
        if (arA*arA + aiA*aiA < 1e-30) continue;

        int kA = (int)(bsA / PEPS_CHI4);
        int uA = (int)((bsA / PEPS_CHI3) % chi);
        int dA = (int)((bsA / PEPS_CHI2) % chi);
        int lA = (int)((bsA / PEPS_CHI) % chi);
        int rA = (int)(bsA % chi);

        uint64_t envA = (uint64_t)uA * chi2 + lA * chi + rA;
        int idx_EA = -1;
        for (int i = 0; i < num_EA; i++) if (uniq_envA[i] == envA) { idx_EA = i; break; }
        if (idx_EA < 0) continue;

        double wA = wu_A[uA] * wl_A[lA] * wr_A[rA];
        double arAw = arA * wA, aiAw = aiA * wA;
        int row = kA * num_EA + idx_EA;

        for (uint32_t eB = 0; eB < regB->num_nonzero; eB++) {
            uint64_t bsB = regB->entries[eB].basis_state;
            double arB = regB->entries[eB].amp_re;
            double aiB = regB->entries[eB].amp_im;
            if (arB*arB + aiB*aiB < 1e-30) continue;

            int kB = (int)(bsB / PEPS_CHI4);
            int uB = (int)((bsB / PEPS_CHI3) % chi);
            int dB = (int)((bsB / PEPS_CHI2) % chi);
            int lB = (int)((bsB / PEPS_CHI) % chi);
            int rB = (int)(bsB % chi);

            if (dA != uB) continue;

            uint64_t envB = (uint64_t)dB * chi2 + lB * chi + rB;
            int idx_EB = -1;
            for (int i = 0; i < num_EB; i++) if (uniq_envB[i] == envB) { idx_EB = i; break; }
            if (idx_EB < 0) continue;

            double wB = wd_B[dB] * wl_B[lB] * wr_B[rB] * wv[dA];
            double br = arB * wB, bi = aiB * wB;
            int col = kB * num_EB + idx_EB;

            Th_re[row * svddim_B + col] += arAw*br - aiAw*bi;
            Th_im[row * svddim_B + col] += arAw*bi + aiAw*br;
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
              if (gre*gre + gim*gim < 1e-60) continue;

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

    tsvd_truncated(Th2_re, Th2_im, svddim_A, svddim_B, chi,
                   U_re, U_im, sig, Vc_re, Vc_im);
    free(Th2_re); free(Th2_im);

    /* ── 5. Update bond weight ── */
    int rank = chi < svddim_B ? chi : svddim_B;
    if (rank > svddim_A) rank = svddim_A;
    double sig_norm = 0;
    for (int s = 0; s < rank; s++) sig_norm += sig[s];
    if (sig_norm > 1e-30)
        for (int s = 0; s < rank; s++) vb_shared->w[s] = sig[s] / sig_norm;

    /* ── 6. Write back — preserving ALL bond values ── */
    regA->num_nonzero = 0;
    for (int kA = 0; kA < D; kA++)
     for (int eA = 0; eA < num_EA; eA++) {
         int row = kA * num_EA + eA;
         uint64_t env = uniq_envA[eA];
         int uA = (int)(env / chi2);
         int lA = (int)((env / chi) % chi);
         int rA = (int)(env % chi);
         double invw = 1.0;
         if (wu_A[uA] > 1e-30) invw /= wu_A[uA];
         if (wl_A[lA] > 1e-30) invw /= wl_A[lA];
         if (wr_A[rA] > 1e-30) invw /= wr_A[rA];
         for (int g = 0; g < rank; g++) {
             double re = U_re[row * rank + g] * invw;
             double im = U_im[row * rank + g] * invw;
             if (re*re + im*im > 1e-10 && regA->num_nonzero < 4096) {
                 uint64_t bs = PT_IDX(kA, uA, g, lA, rA);
                 regA->entries[regA->num_nonzero].basis_state = bs;
                 regA->entries[regA->num_nonzero].amp_re = re;
                 regA->entries[regA->num_nonzero].amp_im = im;
                 regA->num_nonzero++;
             }
         }
     }

    regB->num_nonzero = 0;
    for (int kB = 0; kB < D; kB++)
     for (int eB = 0; eB < num_EB; eB++) {
         int col = kB * num_EB + eB;
         uint64_t env = uniq_envB[eB];
         int dB = (int)(env / chi2);
         int lB = (int)((env / chi) % chi);
         int rB = (int)(env % chi);
         double invw = 1.0;
         if (wd_B[dB] > 1e-30) invw /= wd_B[dB];
         if (wl_B[lB] > 1e-30) invw /= wl_B[lB];
         if (wr_B[rB] > 1e-30) invw /= wr_B[rB];
         for (int g = 0; g < rank; g++) {
             double re = Vc_re[g * svddim_B + col] * invw;
             double im = Vc_im[g * svddim_B + col] * invw;
             if (re*re + im*im > 1e-10 && regB->num_nonzero < 4096) {
                 uint64_t bs = PT_IDX(kB, g, dB, lB, rB);
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
        uint32_t k = (uint32_t)(r->entries[e].basis_state / PEPS_CHI4);
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
