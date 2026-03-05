/*
 * mps_overlay.c — 1D Matrix Product State: Register-Based SVD Engine
 *
 * D=6 native (SU(6)), bond dimension χ per bond (2 bonds per site).
 * Simple-update with Jacobi SVD for proper 2-site gate application.
 * All tensor data stored in registers — temporary dense buffers for SVD.
 *
 * Adapted from peps3d_overlay.c — simplified to 1D chain topology.
 *
 * 3-index tensor: T[k, α, β]
 *   k ∈ [0,D)   — physical index
 *   α ∈ [0,χ)   — left bond
 *   β ∈ [0,χ)   — right bond
 *
 * Register encoding: β + α*χ + k*χ²
 *   Position 0 = β (least significant)
 *   Position 1 = α
 *   Position 2 = k (most significant)
 */

#include "mps_overlay.h"
#include "tensor_svd.h"
#include "svd_truncate.h"

static SvdTmpBuf mps_svd_buf;
#include <stdio.h>
#include <stdint.h>
#include <math.h>

/* ═══════════════ LIFECYCLE ═══════════════ */

MpsChain *mps_init(int L)
{
    MpsChain *c = (MpsChain *)calloc(1, sizeof(MpsChain));
    c->L = L;

    c->tensors = (MpsTensor *)calloc(L, sizeof(MpsTensor));

    /* L-1 bonds between adjacent sites */
    c->bonds = (MpsBondWeight *)calloc(L - 1, sizeof(MpsBondWeight));
    for (int i = 0; i < L - 1; i++) {
        c->bonds[i].w = (double *)calloc((size_t)MPS_CHI, sizeof(double));
        for (int s = 0; s < (int)MPS_CHI; s++) c->bonds[i].w[s] = 1.0;
    }

    c->eng = (QuhitEngine *)calloc(1, sizeof(QuhitEngine));
    quhit_engine_init(c->eng);

    c->q_phys = (uint32_t *)calloc(L, sizeof(uint32_t));
    for (int i = 0; i < L; i++)
        c->q_phys[i] = quhit_init_basis(c->eng, 0);

    c->site_reg = (int *)calloc(L, sizeof(int));
    for (int i = 0; i < L; i++) {
        c->site_reg[i] = quhit_reg_init(c->eng, (uint64_t)i, 3, MPS_CHI);
        if (c->site_reg[i] >= 0) {
            c->eng->registers[c->site_reg[i]].bulk_rule = 0;
            quhit_reg_sv_set(c->eng, c->site_reg[i], 0, 1.0, 0.0);
        }
        c->tensors[i].reg_idx = c->site_reg[i];
    }

    /* Per-site Triality state */
    c->tri_sites = (TriOverlaySite *)calloc(L, sizeof(TriOverlaySite));
    for (int i = 0; i < L; i++)
        tri_site_init(&c->tri_sites[i]);

    return c;
}

void mps_free(MpsChain *c)
{
    if (!c) return;
    free(c->tensors);
    for (int i = 0; i < c->L - 1; i++) free(c->bonds[i].w);
    free(c->bonds);
    if (c->eng) {
        quhit_engine_destroy(c->eng);
        free(c->eng);
    }
    free(c->q_phys);
    free(c->site_reg);
    free(c->tri_sites);
    free(c);
}

/* ═══════════════ STATE INITIALIZATION ═══════════════ */

void mps_set_product_state(MpsChain *c, int site,
                           const double *amps_re, const double *amps_im)
{
    int reg = c->site_reg[site];
    if (reg < 0) return;

    c->eng->registers[reg].num_nonzero = 0;
    for (int k = 0; k < MPS_D; k++) {
        if (amps_re[k] * amps_re[k] + amps_im[k] * amps_im[k] > 1e-30)
            quhit_reg_sv_set(c->eng, reg, (basis_t)k * MPS_C2, amps_re[k], amps_im[k]);
    }
}

/* ═══════════════ 1-SITE GATE ═══════════════ */

void mps_gate_1site(MpsChain *c, int site,
                    const double *U_re, const double *U_im)
{
    int reg_idx = c->site_reg[site];
    if (reg_idx < 0 || !c->eng) return;

    QuhitRegister *r = &c->eng->registers[reg_idx];
    uint8_t mask = c->tri_sites ? c->tri_sites[site].active_mask : 0x3F;
    unsigned __int128 chi_power = (unsigned __int128)MPS_C2;
    tri_reg_gate_1site_masked(r, U_re, U_im, mask, chi_power);

    /* Mirror to triality site */
    if (c->tri_sites)
        tri_site_apply_gate(&c->tri_sites[site], U_re, U_im);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * 2-SITE BOND GATE WITH SVD
 *
 * For MPS, the shared bond between site `i` and site `i+1` is:
 *   Site A's β (position 0) — right bond of left site
 *   Site B's α (position 1) — left bond of right site
 *
 * 3-index tensor: T[k, α, β]
 * Index positions: β=0 (least sig), α=1, k=2 (most sig)
 *
 * bond_A = 0 (position of shared β in A's bond indices)
 * bond_B = 1 (position of shared α in B's bond indices)
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_gate_bond(MpsChain *c, int site,
                   const double *G_re, const double *G_im)
{
    int sA = site, sB = site + 1;
    int D = MPS_D, chi = (int)MPS_CHI;

    /* Bond index powers: position 0=β, position 1=α, position 2=k */
    basis_t bp[3] = {1, MPS_CHI, MPS_C2};

    /* Shared bond: A's β (position 0), B's α (position 1) */
    int bond_A = 0;  /* β position in A */
    int bond_B = 1;  /* α position in B */

    MpsBondWeight *shared_bw = &c->bonds[site];

    QuhitRegister *regA = &c->eng->registers[c->site_reg[sA]];
    QuhitRegister *regB = &c->eng->registers[c->site_reg[sB]];

    /* ── 1. Find exact Sparse-Rank Environment ── */
    int max_E = chi;
    basis_t *uniq_envA = (basis_t *)malloc(max_E * sizeof(basis_t));
    basis_t *uniq_envB = (basis_t *)malloc(max_E * sizeof(basis_t));
    int num_EA = 0, num_EB = 0;

    /* A env: strip shared bond (position 0) from bond part
     * pure = basis % χ² (strip k), env = α (position 1 only) */
    for (uint32_t eA = 0; eA < regA->num_nonzero; eA++) {
        basis_t pure = regA->entries[eA].basis_state % MPS_C2;
        basis_t env = (pure / bp[bond_A + 1]) * bp[bond_A] + (pure % bp[bond_A]);
        int found = 0;
        for (int i = 0; i < num_EA; i++) {
            if (uniq_envA[i] == env) { found = 1; break; }
        }
        if (!found && num_EA < max_E) {
            uniq_envA[num_EA++] = env;
        }
    }

    /* B env: strip shared bond (position 1) from bond part
     * pure = basis % χ², env = β (position 0 only) */
    for (uint32_t eB = 0; eB < regB->num_nonzero; eB++) {
        basis_t pure = regB->entries[eB].basis_state % MPS_C2;
        basis_t env = (pure / bp[bond_B + 1]) * bp[bond_B] + (pure % bp[bond_B]);
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
        basis_t bsA = regA->entries[eA].basis_state;
        double arA = regA->entries[eA].amp_re;
        double aiA = regA->entries[eA].amp_im;
        if (arA*arA + aiA*aiA < 1e-10) continue;

        int kA = (int)(bsA / MPS_C2);
        basis_t pureA = bsA % MPS_C2;
        int shared_valA = (int)((pureA / bp[bond_A]) % chi);  /* β_A */
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

            basis_t pureB = bsB % MPS_C2;
            int shared_valB = (int)((pureB / bp[bond_B]) % chi);  /* α_B */
            if (shared_valA != shared_valB) continue;

            int kB = (int)(bsB / MPS_C2);
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

    /* Store σ on bonds — Θ contraction absorbs via sw = bonds.w[s].
     * U and V are written unscaled. 1-site gates between bond gates
     * are compatible because they don't touch the bond index. */
    for (int s = 0; s < (int)MPS_CHI; s++)
        shared_bw->w[s] = (s < rank) ? sig[s] : 0.0;

    /* ── 5. Write back safely ── */
    svd_buf_reset(&mps_svd_buf);
    for (int kA = 0; kA < D; kA++)
     for (int eA = 0; eA < num_EA; eA++) {
         int row_idx = kA * num_EA + eA;
         basis_t envA = uniq_envA[eA];
         basis_t pure = (envA / bp[bond_A]) * bp[bond_A + 1] + (envA % bp[bond_A]);
         for (int gv = 0; gv < rank; gv++) {
             double re = U_re[row_idx * rank + gv];
             double im = U_im[row_idx * rank + gv];
             if (re == 0.0 && im == 0.0) continue;
             svd_buf_push(&mps_svd_buf, kA * MPS_C2 + pure + gv * bp[bond_A], re, im);
         }
     }
    svd_buf_flush(&mps_svd_buf, regA);

    svd_buf_reset(&mps_svd_buf);
    for (int kB = 0; kB < D; kB++)
     for (int eB = 0; eB < num_EB; eB++) {
         int col_idx = kB * num_EB + eB;
         basis_t envB = uniq_envB[eB];
         basis_t pure = (envB / bp[bond_B]) * bp[bond_B + 1] + (envB % bp[bond_B]);
         for (int gv = 0; gv < rank; gv++) {
             double re = Vc_re[gv * svddim_B + col_idx];
             double im = Vc_im[gv * svddim_B + col_idx];
             if (re == 0.0 && im == 0.0) continue;
             svd_buf_push(&mps_svd_buf, kB * MPS_C2 + pure + gv * bp[bond_B], re, im);
         }
     }
    svd_buf_flush(&mps_svd_buf, regB);

    free(U_re); free(U_im);
    free(sig);
    free(Vc_re); free(Vc_im);
    free(uniq_envA); free(uniq_envB);
}

/* ═══════════════ LOCAL DENSITY ═══════════════ */

void mps_local_density(MpsChain *c, int site, double *probs)
{
    int reg = c->site_reg[site];

    for (int k = 0; k < MPS_D; k++) probs[k] = 0;

    if (reg < 0 || !c->eng) { probs[0] = 1.0; return; }

    QuhitRegister *r = &c->eng->registers[reg];
    double total = 0;

    for (uint32_t e = 0; e < r->num_nonzero; e++) {
        /* Physical digit k is at position 2 (most significant) */
        basis_t bs = r->entries[e].basis_state;
        int k = (int)(bs / MPS_C2);  /* k = highest position */
        if (k >= MPS_D) continue;
        double re = r->entries[e].amp_re;
        double im = r->entries[e].amp_im;
        double p = re * re + im * im;
        probs[k] += p;
        total += p;
    }

    if (total > 1e-30)
        for (int k = 0; k < MPS_D; k++) probs[k] /= total;
    else
        probs[0] = 1.0;
}

/* ═══════════════ BATCH GATE APPLICATION ═══════════════ */

void mps_gate_bond_all(MpsChain *c, const double *G_re, const double *G_im)
{
    /* Sequential left-to-right sweep for 1D chain */
    for (int site = 0; site < c->L - 1; site++)
        mps_gate_bond(c, site, G_re, G_im);
}

void mps_gate_1site_all(MpsChain *c, const double *U_re, const double *U_im)
{
    for (int site = 0; site < c->L; site++)
        mps_gate_1site(c, site, U_re, U_im);
}

void mps_normalize_site(MpsChain *c, int site)
{
    if (site < 0 || site >= c->L) return;
    int reg = c->site_reg[site];
    if (reg < 0) return;
    QuhitRegister *r = &c->eng->registers[reg];

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

void mps_trotter_step(MpsChain *c, const double *G_re, const double *G_im)
{
    mps_gate_bond_all(c, G_re, G_im);
}

/* ═══════════════ GATE CONSTRUCTORS ═══════════════ */

void mps_build_dft6(double *U_re, double *U_im)
{
    int D = MPS_D;
    double inv = 1.0 / sqrt((double)D);
    for (int j = 0; j < D; j++)
        for (int k = 0; k < D; k++) {
            double angle = 2.0 * M_PI * j * k / (double)D;
            U_re[j * D + k] = inv * cos(angle);
            U_im[j * D + k] = inv * sin(angle);
        }
}

void mps_build_cz(double *G_re, double *G_im)
{
    int D = MPS_D, D2 = D * D, D4 = D2 * D2;
    memset(G_re, 0, D4 * sizeof(double));
    memset(G_im, 0, D4 * sizeof(double));
    for (int j = 0; j < D; j++)
        for (int k = 0; k < D; k++) {
            int idx = (j * D + k) * D2 + (j * D + k);
            double angle = 2.0 * M_PI * j * k / (double)D;
            G_re[idx] = cos(angle);
            G_im[idx] = sin(angle);
        }
}
