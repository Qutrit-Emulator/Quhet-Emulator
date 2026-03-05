/*
 * mps_overlay.c — 1D Tensor Network: Register-Based SVD Engine
 *
 * D=6 native (SU(6)), bond dimension χ=256 per axis (2 axes: left, right).
 * Simple-update with Jacobi SVD for proper 2-site gate application.
 * All tensor data stored in registers — temporary dense buffers for SVD.
 *
 * Modeled directly on peps3d_overlay.c, adapted for 1D chains.
 *
 * 3-index tensor: T[k, α, β]
 *   k = physical index (position 2, most significant)
 *   α = left bond  (position 1)
 *   β = right bond (position 0, least significant)
 * basis = k*χ² + α*χ + β
 *
 * Shared bond between site i and i+1: β_i = α_{i+1}
 *   A's shared = β (position 0)
 *   B's shared = α (position 1)
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

    /* Bond weights: L-1 bonds between consecutive sites */
    c->bonds = (MpsBondWeight *)calloc(L - 1, sizeof(MpsBondWeight));
    for (int i = 0; i < L - 1; i++) {
        c->bonds[i].w = (double *)calloc((size_t)MPS_CHI, sizeof(double));
        c->bonds[i].w[0] = 1.0;  /* Product state: only shared bond 0 active */
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
            quhit_reg_sv_set(c->eng, reg, (basis_t)k * MPS_C2,
                             amps_re[k], amps_im[k]);
    }

    /* Sync triality sidecar */
    if (c->tri_sites) {
        triality_ensure_view(&c->tri_sites[site].tri, VIEW_EDGE);
        for (int k = 0; k < MPS_D; k++) {
            c->tri_sites[site].tri.edge_re[k] = amps_re[k];
            c->tri_sites[site].tri.edge_im[k] = amps_im[k];
        }
        c->tri_sites[site].tri.dirty = DIRTY_VERTEX | DIRTY_DIAGONAL |
                                        DIRTY_FOLDED | DIRTY_EXOTIC;
        tri_site_sync(&c->tri_sites[site]);
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
 * 2-SITE GATE: BOND (site, site+1) WITH SVD
 *
 * 3-index tensor: T[k, α, β]
 * Index positions: k=2 (most sig), α=1, β=0 (least sig)
 * bp[3] = {1, χ, χ²}
 *
 * Shared bond: β_A (position 0) = α_B (position 1)
 * A environment: α_A (position 1) — just the left bond
 * B environment: β_B (position 0) — just the right bond
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_gate_bond(MpsChain *c, int site,
                   const double *G_re, const double *G_im)
{
    int sA = site, sB = site + 1;
    int D = MPS_D, chi = (int)MPS_CHI;

    /* Bond index powers: position 0=β, 1=α, 2=k */
    basis_t bp[3] = {1, MPS_CHI, MPS_C2};

    /* Shared bond: A's β (position 0), B's α (position 1)
     * bond_A = 0 (position of shared in A's bond indices)
     * bond_B = 1 (position of shared in B's bond indices) */
    int bond_A = 0;  /* β position */
    int bond_B = 1;  /* α position */

    MpsBondWeight *shared_bw = &c->bonds[site];

    QuhitRegister *regA = &c->eng->registers[c->site_reg[sA]];
    QuhitRegister *regB = &c->eng->registers[c->site_reg[sB]];

    /* ── 1. Find exact Sparse-Rank Environment ── */
    int max_E = chi;
    basis_t *uniq_envA = (basis_t *)malloc(max_E * sizeof(basis_t));
    basis_t *uniq_envB = (basis_t *)malloc(max_E * sizeof(basis_t));
    int num_EA = 0, num_EB = 0;

    /* A env: strip shared bond (position 0) from bond part
     * pure = basis % χ² (strip k), env = pure / bp[1] = α */
    for (uint32_t eA = 0; eA < regA->num_nonzero; eA++) {
        basis_t pure = regA->entries[eA].basis_state % MPS_C2;
        basis_t env = (pure / bp[bond_A + 1]) * bp[bond_A] + (pure % bp[bond_A]);
        int found = 0;
        for (int i = 0; i < num_EA; i++)
            if (uniq_envA[i] == env) { found = 1; break; }
        if (!found && num_EA < max_E) uniq_envA[num_EA++] = env;
    }

    /* B env: strip shared bond (position 1) from bond part
     * pure = basis % χ², env = pure / bp[2] * bp[1] + pure % bp[1] = β */
    for (uint32_t eB = 0; eB < regB->num_nonzero; eB++) {
        basis_t pure = regB->entries[eB].basis_state % MPS_C2;
        basis_t env = (pure / bp[bond_B + 1]) * bp[bond_B] + (pure % bp[bond_B]);
        int found = 0;
        for (int i = 0; i < num_EB; i++)
            if (uniq_envB[i] == env) { found = 1; break; }
        if (!found && num_EB < max_E) uniq_envB[num_EB++] = env;
    }

    if (num_EA == 0 || num_EB == 0) {
        free(uniq_envA); free(uniq_envB);
        return;
    }

    /* Adjacent bond weights for Γ-Λ simple-update:
     * left  = bonds[site-1] for site A's α (environment)
     * right = bonds[site+1] for site B's β (environment) */
    MpsBondWeight *bw_left  = (sA > 0) ? &c->bonds[sA - 1] : NULL;
    MpsBondWeight *bw_right = (sB < c->L - 1) ? &c->bonds[sB] : NULL;

    /* ── 2. Build Θ (Γ-Λ form) ──
     * Θ = √Λ_left · ΓA · Λ_shared · ΓB · √Λ_right
     * Absorb all bond weights to create properly normalized Θ. */
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
        for (int i = 0; i < num_EA; i++)
            if (uniq_envA[i] == envA) { idx_EA = i; break; }
        if (idx_EA < 0) continue;
        int row = kA * num_EA + idx_EA;

        /* Absorb √Λ_left: weight A amplitude by √σ_left[α_A] */
        double wA = 1.0;
        if (bw_left) {
            int alpha_A = (int)((pureA / bp[1]) % chi);
            wA = bw_left->w[alpha_A];
            if (wA > 0) wA = sqrt(wA); else wA = 0;
        }
        double warA = arA * wA, waiA = aiA * wA;

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
            for (int i = 0; i < num_EB; i++)
                if (uniq_envB[i] == envB) { idx_EB = i; break; }
            if (idx_EB < 0) continue;
            int col = kB * num_EB + idx_EB;

            /* Absorb √Λ_right: weight B amplitude by √σ_right[β_B] */
            double wB = 1.0;
            if (bw_right) {
                int beta_B = (int)(pureB % chi);
                wB = bw_right->w[beta_B];
                if (wB > 0) wB = sqrt(wB); else wB = 0;
            }

            /* Absorb Λ_shared: multiply by σ_shared[s] */
            double sw = shared_bw->w[shared_valA];
            double br = arB * sw * wB, bi = aiB * sw * wB;

            Th_re[row * svddim_B + col] += warA*br - waiA*bi;
            Th_im[row * svddim_B + col] += warA*bi + waiA*br;
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

    tsvd_vesica_truncated_sparse(Th2_re, Th2_im, svddim_A, svddim_B,
                   D, num_EA, num_EB, chi,
                   U_re, U_im, sig, Vc_re, Vc_im);
    free(Th2_re); free(Th2_im);

    /* Determine actual SVD rank: min(rows, cols, chi), trimmed at zero sigmas */
    int svd_cols = svddim_A < svddim_B ? svddim_A : svddim_B;
    if (svd_cols > chi) svd_cols = chi;
    int rank = svd_cols;


    /* Trim trailing zero singular values */
    while (rank > 0 && sig[rank - 1] < 1e-14) rank--;
    if (rank == 0) rank = 1;

    /* Cap rank for 4096-entry register limit */
    int max_env = num_EA > num_EB ? num_EA : num_EB;
    int rank_cap = max_env > 0 ? 4096 / (D * max_env) : rank;
    if (rank_cap < 1) rank_cap = 1;
    if (rank > rank_cap) rank = rank_cap;

    /* Store σ as bond weights for the shared bond */
    for (int s = 0; s < (int)MPS_CHI; s++)
        shared_bw->w[s] = (s < rank) ? sig[s] : 0.0;

    /* ── 5. Write back (Γ form) ──
     * Divide out √Λ_left from U to get ΓA, and √Λ_right from V† to get ΓB.
     * ΓA[kA, envA, gv] = U[row, gv] / √σ_left[α_A]
     * ΓB[kB, gv, envB] = V†[gv, col] / √σ_right[β_B] */
    svd_buf_reset(&mps_svd_buf);
    for (int kA = 0; kA < D; kA++)
     for (int eA = 0; eA < num_EA; eA++) {
         int row_idx = kA * num_EA + eA;
         basis_t envA = uniq_envA[eA];
         basis_t pure = (envA / bp[bond_A]) * bp[bond_A + 1] + (envA % bp[bond_A]);

         /* Compute √Λ_left inverse for this environment index */
         double inv_wL = 1.0;
         if (bw_left) {
             int alpha_A = (int)(envA % chi);  /* envA is the α value */
             double wL = bw_left->w[alpha_A];
             inv_wL = (wL > 1e-15) ? 1.0 / sqrt(wL) : 0.0;
         }

         for (int gv = 0; gv < rank; gv++) {
             double re = U_re[row_idx * svd_cols + gv] * inv_wL;
             double im = U_im[row_idx * svd_cols + gv] * inv_wL;
             if (re*re + im*im < 1e-50) continue;
             svd_buf_push(&mps_svd_buf,
                          kA * MPS_C2 + pure + gv * bp[bond_A], re, im);
         }
     }
    svd_buf_flush(&mps_svd_buf, regA);

    svd_buf_reset(&mps_svd_buf);
    for (int kB = 0; kB < D; kB++)
     for (int eB = 0; eB < num_EB; eB++) {
         int col_idx = kB * num_EB + eB;
         basis_t envB = uniq_envB[eB];
         basis_t pure = (envB / bp[bond_B]) * bp[bond_B + 1] + (envB % bp[bond_B]);

         /* Compute √Λ_right inverse for this environment index */
         double inv_wR = 1.0;
         if (bw_right) {
             int beta_B = (int)(envB % chi);  /* envB is the β value */
             double wR = bw_right->w[beta_B];
             inv_wR = (wR > 1e-15) ? 1.0 / sqrt(wR) : 0.0;
         }

         for (int gv = 0; gv < rank; gv++) {
             double re = Vc_re[gv * svddim_B + col_idx] * inv_wR;
             double im = Vc_im[gv * svddim_B + col_idx] * inv_wR;
             if (re*re + im*im < 1e-50) continue;
             svd_buf_push(&mps_svd_buf,
                          kB * MPS_C2 + pure + gv * bp[bond_B], re, im);
         }
     }
    svd_buf_flush(&mps_svd_buf, regB);

    free(U_re); free(U_im);
    free(sig);
    free(Vc_re); free(Vc_im);
    free(uniq_envA); free(uniq_envB);

    /* ── 6. Mirror to triality ── */
    if (c->tri_sites)
        tri_site_apply_cz(&c->tri_sites[sA], &c->tri_sites[sB]);
}

/* ═══════════════ LOCAL DENSITY ═══════════════ */

void mps_local_density(MpsChain *c, int site, double *probs)
{
    int reg = c->site_reg[site];

    for (int k = 0; k < MPS_D; k++) probs[k] = 0;

    if (reg < 0 || !c->eng) { probs[0] = 1.0; return; }

    QuhitRegister *r = &c->eng->registers[reg];
    int chi = (int)MPS_CHI;
    double total = 0;

    /* Bond weight pointers: left bond (α, position 1) and right bond (β, position 0) */
    MpsBondWeight *bw_left  = (site > 0) ? &c->bonds[site - 1] : NULL;
    MpsBondWeight *bw_right = (site < c->L - 1) ? &c->bonds[site] : NULL;

    for (uint32_t e = 0; e < r->num_nonzero; e++) {
        basis_t bs = r->entries[e].basis_state;
        int k = (int)(bs / MPS_C2);
        if (k >= MPS_D) continue;
        basis_t pure = bs % MPS_C2;
        int beta  = (int)(pure % chi);         /* position 0: right bond index */
        int alpha = (int)((pure / chi) % chi); /* position 1: left bond index */

        double re = r->entries[e].amp_re;
        double im = r->entries[e].amp_im;
        double p = re * re + im * im;

        /* Weight by σ² for each adjacent bond (σ_left[α]² × σ_right[β]²)
         * The density ρ_k = Σ_{α,β} |U[k,α,β]|² × σ_L[α]² × σ_R[β]²
         * where σ² are the squared singular values (= eigenvalues of reduced ρ) */
        if (bw_left && alpha < chi) {
            double wL = bw_left->w[alpha];
            p *= wL * wL;
        }
        if (bw_right && beta < chi) {
            double wR = bw_right->w[beta];
            p *= wR * wR;
        }

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
    if (c->L < 2) return;

    /* Sequential left-to-right sweep.
     * MPS simple-update requires sequential ordering: each bond gate
     * modifies both adjacent tensors, and the next bond must see the
     * updated tensor from the previous step. */
    for (int i = 0; i < c->L - 1; i++)
        mps_gate_bond(c, i, G_re, G_im);
}

void mps_gate_1site_all(MpsChain *c, const double *U_re, const double *U_im)
{
    for (int i = 0; i < c->L; i++)
        mps_gate_1site(c, i, U_re, U_im);
}

void mps_normalize_site(MpsChain *c, int site)
{
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
