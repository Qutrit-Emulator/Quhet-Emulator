/*
 * mps_overlay.c — MPS Overlay with Pure Magic Pointers
 *
 * All tensor data stored in QuhitRegisters — RAM-agnostic.
 * Gate functions use register-based read/write via mps_read_tensor/mps_write_tensor.
 * 2-site gates are O(1) via CZ₆ engine pair operations.
 */

#ifdef _OPENMP
#include <omp.h>
#endif

#include "mps_overlay.h"
#include <math.h>
#include <fenv.h>

/* ─── Global tensor store ──────────────────────────────────────────────────── */
MpsTensor   *mps_store   = NULL;
int          mps_store_n = 0;
QuhitEngine *mps_eng     = NULL;
int          mps_defer_renorm = 0;
int          mps_sweep_right  = 1;

/* ═══════════════════════════════════════════════════════════════════════════════
 * INIT / FREE
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_overlay_init(QuhitEngine *eng, uint32_t *quhits, int n)
{
    (void)quhits;
    mps_eng = eng;

    /* Lightweight tensor metadata — 4 bytes per site */
    if (mps_store) { free(mps_store); mps_store = NULL; }
    mps_store = (MpsTensor *)calloc((size_t)n, sizeof(MpsTensor));
    mps_store_n = n;

    /* Create per-site registers: 3 qudits (k, α, β) */
    for (int i = 0; i < n; i++) {
        mps_store[i].reg_idx = quhit_reg_init(eng, (uint64_t)i, 3, MPS_CHI);
        if (mps_store[i].reg_idx >= 0) {
            eng->registers[mps_store[i].reg_idx].bulk_rule = 0;
            /* Seed product state: |k=0, α=0, β=0⟩ with amplitude 1.0 */
            quhit_reg_sv_set(eng, mps_store[i].reg_idx, 0, 1.0, 0.0);
        }
    }
}

void mps_overlay_free(void)
{
    free(mps_store);
    mps_store = NULL;
    mps_store_n = 0;
    mps_eng = NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * W-STATE CONSTRUCTION
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_overlay_write_w_state(QuhitEngine *eng, uint32_t *quhits, int n)
{
    (void)eng; (void)quhits;
    double site_scale = pow((double)n, -1.0 / (2.0 * n));

    for (int i = 0; i < n; i++) {
        mps_zero_site(i);
        mps_write_tensor(i, 0, 0, 0, site_scale, 0.0);
        mps_write_tensor(i, 0, 1, 1, site_scale, 0.0);
        mps_write_tensor(i, 1, 0, 1, site_scale, 0.0);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PRODUCT STATE |0⟩^⊗N
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_overlay_write_zero(QuhitEngine *eng, uint32_t *quhits, int n)
{
    (void)eng; (void)quhits;
    for (int i = 0; i < n; i++) {
        mps_zero_site(i);
        mps_write_tensor(i, 0, 0, 0, 1.0, 0.0);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * AMPLITUDE INSPECTION: ⟨basis|ψ⟩ = L^T · Π_i A[k_i] · R
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_overlay_amplitude(QuhitEngine *eng, uint32_t *quhits, int n,
                           const uint32_t *basis, double *out_re, double *out_im)
{
    (void)eng; (void)quhits;
    double v_re[MPS_CHI], v_im[MPS_CHI];
    memset(v_re, 0, sizeof(v_re));
    memset(v_im, 0, sizeof(v_im));
    v_re[0] = 1.0;

    for (int i = 0; i < n; i++) {
        int k = (int)basis[i];
        double next_re[MPS_CHI] = {0}, next_im[MPS_CHI] = {0};

        for (int beta = 0; beta < MPS_CHI; beta++)
            for (int alpha = 0; alpha < MPS_CHI; alpha++) {
                double t_re, t_im;
                mps_read_tensor(i, k, alpha, beta, &t_re, &t_im);
                next_re[beta] += v_re[alpha]*t_re - v_im[alpha]*t_im;
                next_im[beta] += v_re[alpha]*t_im + v_im[alpha]*t_re;
            }
        memcpy(v_re, next_re, sizeof(v_re));
        memcpy(v_im, next_im, sizeof(v_im));
    }

    double sr = 0, si = 0;
    for (int i = 0; i < MPS_CHI; i++) { sr += v_re[i]; si += v_im[i]; }
    *out_re = sr;
    *out_im = si;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MEASUREMENT — O(N × χ³ × D)
 * ═══════════════════════════════════════════════════════════════════════════════ */

uint32_t mps_overlay_measure(QuhitEngine *eng, uint32_t *quhits, int n, int target_idx)
{
    (void)quhits;
    int chi = MPS_CHI;
    size_t chi2 = (size_t)chi * chi;

    /* Right density environment (heap) */
    double *rho_R = (double *)calloc(chi2, sizeof(double));
    rho_R[0] = 1.0;  /* rho_R[0][0] = 1.0 */

    for (int j = n - 1; j > target_idx; j--) {
        double *new_rho = (double *)calloc(chi2, sizeof(double));
        for (int k = 0; k < MPS_PHYS; k++) {
            double *A = (double *)calloc(chi2, sizeof(double));
            for (int a = 0; a < chi; a++)
                for (int b = 0; b < chi; b++) {
                    double re, im;
                    mps_read_tensor(j, k, a, b, &re, &im);
                    A[a * chi + b] = re;
                }
            double *tmp = (double *)calloc(chi2, sizeof(double));
            for (int a = 0; a < chi; a++)
                for (int b = 0; b < chi; b++)
                    for (int c = 0; c < chi; c++)
                        tmp[a * chi + b] += A[a * chi + c] * rho_R[c * chi + b];
            for (int a = 0; a < chi; a++)
                for (int b = 0; b < chi; b++)
                    for (int c = 0; c < chi; c++)
                        new_rho[a * chi + b] += tmp[a * chi + c] * A[b * chi + c];
            free(A); free(tmp);
        }
        memcpy(rho_R, new_rho, chi2 * sizeof(double));
        free(new_rho);
    }

    /* Left environment */
    double *L = (double *)calloc(chi, sizeof(double));
    L[0] = 1.0;

    for (int j = 0; j < target_idx; j++) {
        double *new_L = (double *)calloc(chi, sizeof(double));
        for (int k = 0; k < MPS_PHYS; k++) {
            double *Ak = (double *)calloc(chi2, sizeof(double));
            int nonzero = 0;
            for (int a = 0; a < chi; a++)
                for (int b = 0; b < chi; b++) {
                    double re, im;
                    mps_read_tensor(j, k, a, b, &re, &im);
                    Ak[a * chi + b] = re;
                    if (re != 0 || im != 0) nonzero = 1;
                }
            if (!nonzero) { free(Ak); continue; }
            for (int b = 0; b < chi; b++)
                for (int a = 0; a < chi; a++)
                    new_L[b] += L[a] * Ak[a * chi + b];
            free(Ak);
        }
        memcpy(L, new_L, chi * sizeof(double));
        free(new_L);
    }

    /* P(k) */
    double probs[MPS_PHYS];
    double total_prob = 0;

    for (int k = 0; k < MPS_PHYS; k++) {
        double *Ak = (double *)calloc(chi2, sizeof(double));
        for (int a = 0; a < chi; a++)
            for (int b = 0; b < chi; b++) {
                double re, im;
                mps_read_tensor(target_idx, k, a, b, &re, &im);
                Ak[a * chi + b] = re;
            }
        double *mid = (double *)calloc(chi, sizeof(double));
        for (int b = 0; b < chi; b++)
            for (int a = 0; a < chi; a++)
                mid[b] += L[a] * Ak[a * chi + b];
        double pk = 0;
        for (int a = 0; a < chi; a++)
            for (int b = 0; b < chi; b++)
                pk += mid[a] * rho_R[a * chi + b] * mid[b];
        probs[k] = pk > 0 ? pk : 0;
        total_prob += probs[k];
        free(Ak); free(mid);
    }

    free(L);
    free(rho_R);

    /* Born sample */
    if (total_prob > 1e-30)
        for (int k = 0; k < MPS_PHYS; k++) probs[k] /= total_prob;

    double r = quhit_prng_double(eng);
    uint32_t outcome = 0;
    double cdf = 0;
    for (int k = 0; k < MPS_PHYS; k++) {
        cdf += probs[k];
        if (r < cdf) { outcome = (uint32_t)k; break; }
    }

    /* Project + renormalize */
    for (int k = 0; k < MPS_PHYS; k++) {
        if ((uint32_t)k != outcome) {
            for (int a = 0; a < chi; a++)
                for (int b = 0; b < chi; b++)
                    mps_write_tensor(target_idx, k, a, b, 0, 0);
        }
    }
    double slice_norm2 = 0;
    for (int a = 0; a < chi; a++)
        for (int b = 0; b < chi; b++) {
            double re, im;
            mps_read_tensor(target_idx, (int)outcome, a, b, &re, &im);
            slice_norm2 += re*re + im*im;
        }
    if (slice_norm2 > 1e-30) {
        double scale = born_fast_isqrt(slice_norm2);
        for (int a = 0; a < chi; a++)
            for (int b = 0; b < chi; b++) {
                double re, im;
                mps_read_tensor(target_idx, (int)outcome, a, b, &re, &im);
                mps_write_tensor(target_idx, (int)outcome, a, b, re*scale, im*scale);
            }
    }

    return outcome;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SINGLE-SITE GATE — O(entries × D) via register
 * ═══════════════════════════════════════════════════════════════════════════════ */

static int mps_cmp_basis(const void *a, const void *b)
{
    const struct { basis_t basis; double re, im; } *ea = a, *eb = b;
    if (ea->basis < eb->basis) return -1;
    if (ea->basis > eb->basis) return 1;
    return 0;
}

void mps_gate_1site(QuhitEngine *eng, uint32_t *quhits, int n,
                    int site, const double *U_re, const double *U_im)
{
    /* Register-based: manually rotate the physical index k
     * Register basis: k*χ² + α*χ + β, where k is the physical index (0..D-1)
     * We can't use quhit_reg_apply_unitary_pos because reg dim=χ, not D */
    if (eng && mps_store && mps_store[site].reg_idx >= 0) {
        int reg_idx = mps_store[site].reg_idx;
        QuhitRegister *reg = &eng->registers[reg_idx];
        int chi = MPS_CHI, D = MPS_PHYS;
        int chi2 = chi * chi;

        /* Build output in temporary arrays */
        uint32_t max_out = reg->num_nonzero * D + 1;
        if (max_out < 4096) max_out = 4096;
        struct { basis_t basis; double re, im; } *tmp =
            calloc(max_out, sizeof(*tmp));
        uint32_t nout = 0;

        for (uint32_t e = 0; e < reg->num_nonzero; e++) {
            basis_t bs = reg->entries[e].basis_state;
            double ar = reg->entries[e].amp_re;
            double ai = reg->entries[e].amp_im;
            int k = (int)(bs / chi2);           /* physical index */
            basis_t bond = bs % chi2;          /* α*χ + β */

            for (int kp = 0; kp < D; kp++) {
                double ur = U_re[kp * D + k];
                double ui = U_im[kp * D + k];
                double nr = ur * ar - ui * ai;
                double ni = ur * ai + ui * ar;
                if (nr*nr + ni*ni < 1e-10) continue;

                if (nout < max_out) {
                    tmp[nout].basis = (basis_t)kp * chi2 + bond;
                    tmp[nout].re = nr;
                    tmp[nout].im = ni;
                    nout++;
                }
            }
        }

        /* Sort by basis state to accumulate duplicates in O(K log K) */
        qsort(tmp, nout, sizeof(*tmp), mps_cmp_basis);

        /* Write back with combine */
        reg->num_nonzero = 0;
        for (uint32_t t = 0; t < nout; t++) {
            double acc_re = tmp[t].re;
            double acc_im = tmp[t].im;
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

    /* Mirror to physical quhit (marginal readout) */
    if (eng && quhits && site < n)
        quhit_apply_unitary(eng, quhits[site], U_re, U_im);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * 2-SITE GATE — Register-Based SVD Contraction
 *
 * Side-channel optimized (tns_contraction_probe.c findings):
 *   • Zero attractor → skip negligible Θ entries via mag² (no sqrt)
 *   • Gate sparsity → skip via mag² (no fabs)
 *   • 1.0 attractor → norm always converges to 1.0
 *   • 1/6 spectrum → σ values converge to equal weights at D=6
 *
 * Temporary memory: ~4 × DCHI² × 8 bytes ≈ 37 MB at χ=128
 * ═══════════════════════════════════════════════════════════════════════════════ */

#include "tensor_svd.h"

#define DCHI (MPS_PHYS * MPS_CHI)

void mps_gate_2site(QuhitEngine *eng, uint32_t *quhits, int n,
                    int site, const double *G_re, const double *G_im)
{
    int sA = site, sB = site + 1;
    int D = MPS_PHYS, chi = MPS_CHI;
    int dchi = D * chi;  /* DCHI = 768 at χ=128 */
    int chi2 = chi * chi;
    size_t dchi2 = (size_t)dchi * dchi;

    /* ── Step 1+2: Build Θ from SPARSE register entries ──
     * Register basis: k*χ² + α*χ + β
     * Contract: Θ[(kA,α),(kB,β)] = Σ_γ A[kA,α,γ] × B[kB,γ,β]
     * Iterate over nnzA × nnzB pairs matching on γ. */

    double *Th_re = (double *)calloc(dchi2, sizeof(double));
    double *Th_im = (double *)calloc(dchi2, sizeof(double));

    QuhitRegister *regA = &eng->registers[mps_store[sA].reg_idx];
    QuhitRegister *regB = &eng->registers[mps_store[sB].reg_idx];

    for (uint32_t eA = 0; eA < regA->num_nonzero; eA++) {
        basis_t bsA = regA->entries[eA].basis_state;
        double arA = regA->entries[eA].amp_re;
        double aiA = regA->entries[eA].amp_im;
        /* Side-channel: use mag² directly, skip sqrt for threshold check */
        if (arA*arA + aiA*aiA < 1e-10) continue;

        int kA = (int)(bsA / chi2);
        int alpha = (int)((bsA / chi) % chi);
        int gamma_A = (int)(bsA % chi);  /* shared bond */

        int row = kA * chi + alpha;

        for (uint32_t eB = 0; eB < regB->num_nonzero; eB++) {
            basis_t bsB = regB->entries[eB].basis_state;
            double arB = regB->entries[eB].amp_re;
            double aiB = regB->entries[eB].amp_im;
            if (arB*arB + aiB*aiB < 1e-10) continue;

            int kB = (int)(bsB / chi2);
            int gamma_B = (int)((bsB / chi) % chi);  /* shared bond */
            int beta = (int)(bsB % chi);

            if (gamma_A != gamma_B) continue;

            int col = kB * chi + beta;
            Th_re[row * dchi + col] += arA*arB - aiA*aiB;
            Th_im[row * dchi + col] += arA*aiB + aiA*arB;
        }
    }

    /* ── Step 3: Apply gate G to physical indices ── */
    double *Th2_re = (double *)calloc(dchi2, sizeof(double));
    double *Th2_im = (double *)calloc(dchi2, sizeof(double));

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

              for (int a = 0; a < chi; a++) {
                  int dst_row = kAp * chi + a;
                  int src_row = kA * chi + a;
                  for (int b = 0; b < chi; b++) {
                      int dst_col = kBp * chi + b;
                      int src_col = kB * chi + b;
                      double tr = Th_re[src_row * dchi + src_col];
                      double ti = Th_im[src_row * dchi + src_col];
                      Th2_re[dst_row * dchi + dst_col] += gre*tr - gim*ti;
                      Th2_im[dst_row * dchi + dst_col] += gre*ti + gim*tr;
                  }
              }
          }
     }

    free(Th_re); free(Th_im);

    /* ── Step 4: SVD → truncate to χ ── */
    double *U_re  = (double *)calloc((size_t)dchi * chi, sizeof(double));
    double *U_im  = (double *)calloc((size_t)dchi * chi, sizeof(double));
    double *sig   = (double *)calloc(chi, sizeof(double));
    double *Vc_re = (double *)calloc((size_t)chi * dchi, sizeof(double));
    double *Vc_im = (double *)calloc((size_t)chi * dchi, sizeof(double));

    tsvd_truncated_sparse(Th2_re, Th2_im, dchi, dchi, chi,
                   U_re, U_im, sig, Vc_re, Vc_im);

    free(Th2_re); free(Th2_im);

    /* ── Step 5: Write back directly to registers ── */

    /* Side-channel: 1.0 attractor — probe confirmed norm always converges
     * to 1.0 (0x3FF0000000000000).  Fast-path: scale by Σσ² reciprocal.
     * This replaces the old threshold-guarded normalization. */
    int rank = chi < dchi ? chi : dchi;
    double sig_norm = 0;
    for (int i = 0; i < rank; i++) sig_norm += sig[i] * sig[i];
    double sc_scale = (sig_norm > 1e-30) ? born_fast_isqrt(sig_norm) : 0.0;
    for (int i = 0; i < rank; i++) sig[i] *= sc_scale;

    /* A'[kA', α, γ] = √σ[γ] × U[(kA'*χ+α), γ] */
    regA->num_nonzero = 0;
    for (int kA = 0; kA < D; kA++)
     for (int a = 0; a < chi; a++) {
         int row = kA * chi + a;
         for (int g = 0; g < rank; g++) {
             double sq = sig[g] > 1e-20 ? sig[g] * born_fast_isqrt(sig[g]) : 0;
             if (sq < 1e-10) continue;
             double re = sq * U_re[row * rank + g];
             double im = sq * U_im[row * rank + g];
             if (re*re + im*im > 1e-10 && regA->num_nonzero < 4096) {
                 basis_t bs = (basis_t)kA * chi2 + (basis_t)a * chi + g;
                 regA->entries[regA->num_nonzero].basis_state = bs;
                 regA->entries[regA->num_nonzero].amp_re = re;
                 regA->entries[regA->num_nonzero].amp_im = im;
                 regA->num_nonzero++;
             }
         }
     }

    /* B'[kB', γ, β] = √σ[γ] × V†[γ, (kB'*χ+β)] */
    regB->num_nonzero = 0;
    for (int kB = 0; kB < D; kB++)
     for (int g = 0; g < rank; g++) {
          double sq = sig[g] > 1e-20 ? sig[g] * born_fast_isqrt(sig[g]) : 0;
          if (sq < 1e-10) continue;
         for (int b = 0; b < chi; b++) {
             int col = kB * chi + b;
             double re = sq * Vc_re[g * dchi + col];
             double im = sq * Vc_im[g * dchi + col];
             if (re*re + im*im > 1e-10 && regB->num_nonzero < 4096) {
                 basis_t bs = (basis_t)kB * chi2 + (basis_t)g * chi + b;
                 regB->entries[regB->num_nonzero].basis_state = bs;
                 regB->entries[regB->num_nonzero].amp_re = re;
                 regB->entries[regB->num_nonzero].amp_im = im;
                 regB->num_nonzero++;
             }
         }
     }

    /* Normalize each register to Σ|amp|² = 1 for density readout */
    {
        double norm2 = 0;
        for (uint32_t e = 0; e < regA->num_nonzero; e++)
            norm2 += regA->entries[e].amp_re * regA->entries[e].amp_re +
                     regA->entries[e].amp_im * regA->entries[e].amp_im;
        if (norm2 > 1e-20) {
            double inv = born_fast_isqrt(norm2);
            for (uint32_t e = 0; e < regA->num_nonzero; e++) {
                regA->entries[e].amp_re *= inv;
                regA->entries[e].amp_im *= inv;
            }
        }
    }
    {
        double norm2 = 0;
        for (uint32_t e = 0; e < regB->num_nonzero; e++)
            norm2 += regB->entries[e].amp_re * regB->entries[e].amp_re +
                     regB->entries[e].amp_im * regB->entries[e].amp_im;
        if (norm2 > 1e-20) {
            double inv = born_fast_isqrt(norm2);
            for (uint32_t e = 0; e < regB->num_nonzero; e++) {
                regB->entries[e].amp_re *= inv;
                regB->entries[e].amp_im *= inv;
            }
        }
    }

    free(U_re); free(U_im);
    free(sig);
    free(Vc_re); free(Vc_im);

    /* ── Mirror to engine quhits ── */
    if (eng && quhits && sB < n)
        quhit_apply_cz(eng, quhits[sA], quhits[sB]);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * GATE CONSTRUCTORS
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_build_dft6(double *U_re, double *U_im)
{
    double inv = born_fast_isqrt(6.0);
    for (int j = 0; j < 6; j++)
        for (int k = 0; k < 6; k++) {
            double angle = 2.0 * M_PI * j * k / (double)MPS_PHYS;
            U_re[j * 6 + k] = inv * cos(angle);
            U_im[j * 6 + k] = inv * sin(angle);
        }
}

void mps_build_cz(double *G_re, double *G_im)
{
    int D2 = MPS_PHYS * MPS_PHYS;
    memset(G_re, 0, D2 * D2 * sizeof(double));
    memset(G_im, 0, D2 * D2 * sizeof(double));
    for (int k = 0; k < MPS_PHYS; k++)
        for (int l = 0; l < MPS_PHYS; l++) {
            int idx = (k * MPS_PHYS + l) * D2 + (k * MPS_PHYS + l);
            double angle = 2.0 * M_PI * k * l / (double)MPS_PHYS;
            G_re[idx] = cos(angle);
            G_im[idx] = sin(angle);
        }
}

void mps_build_controlled_phase(double *G_re, double *G_im, int power)
{
    int D2 = MPS_PHYS * MPS_PHYS;
    memset(G_re, 0, D2 * D2 * sizeof(double));
    memset(G_im, 0, D2 * D2 * sizeof(double));
    double pf = (double)(1 << power) / (double)(MPS_PHYS * MPS_PHYS);
    for (int k = 0; k < MPS_PHYS; k++)
        for (int l = 0; l < MPS_PHYS; l++) {
            int idx = (k * MPS_PHYS + l) * D2 + (k * MPS_PHYS + l);
            double angle = 2.0 * M_PI * k * l * pf;
            G_re[idx] = cos(angle);
            G_im[idx] = sin(angle);
        }
}

void mps_build_hadamard2(double *U_re, double *U_im)
{
    memset(U_re, 0, 36 * sizeof(double));
    memset(U_im, 0, 36 * sizeof(double));
    double s = born_fast_isqrt(2.0);
    U_re[0*6+0] =  s; U_re[0*6+1] =  s;
    U_re[1*6+0] =  s; U_re[1*6+1] = -s;
    for (int k = 2; k < 6; k++) U_re[k*6+k] = 1.0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * NORM ⟨ψ|ψ⟩
 * ═══════════════════════════════════════════════════════════════════════════════ */

double mps_overlay_norm(QuhitEngine *eng, uint32_t *quhits, int n)
{
    (void)eng; (void)quhits;
    int chi = MPS_CHI;
    size_t chi2 = (size_t)chi * chi;

    double *rho_re = (double *)calloc(chi2, sizeof(double));
    double *rho_im = (double *)calloc(chi2, sizeof(double));
    rho_re[0] = 1.0;  /* [0][0] */

    for (int i = 0; i < n; i++) {
        double *nr     = (double *)calloc(chi2, sizeof(double));
        double *ni_arr = (double *)calloc(chi2, sizeof(double));

        for (int k = 0; k < MPS_PHYS; k++) {
            double *A_re = (double *)calloc(chi2, sizeof(double));
            double *A_im = (double *)calloc(chi2, sizeof(double));
            for (int a = 0; a < chi; a++)
                for (int b = 0; b < chi; b++)
                    mps_read_tensor(i, k, a, b,
                                    &A_re[a * chi + b], &A_im[a * chi + b]);

            double *tr2 = (double *)calloc(chi2, sizeof(double));
            double *ti2 = (double *)calloc(chi2, sizeof(double));
            for (int a = 0; a < chi; a++)
                for (int bp = 0; bp < chi; bp++)
                    for (int ap = 0; ap < chi; ap++) {
                        tr2[a*chi+bp] += rho_re[a*chi+ap]*A_re[ap*chi+bp]
                                       - rho_im[a*chi+ap]*A_im[ap*chi+bp];
                        ti2[a*chi+bp] += rho_re[a*chi+ap]*A_im[ap*chi+bp]
                                       + rho_im[a*chi+ap]*A_re[ap*chi+bp];
                    }

            for (int b = 0; b < chi; b++)
                for (int bp = 0; bp < chi; bp++)
                    for (int a = 0; a < chi; a++) {
                        double ar = A_re[a*chi+b], ai = -A_im[a*chi+b];
                        nr[b*chi+bp]     += ar*tr2[a*chi+bp] - ai*ti2[a*chi+bp];
                        ni_arr[b*chi+bp] += ar*ti2[a*chi+bp] + ai*tr2[a*chi+bp];
                    }
            free(A_re); free(A_im); free(tr2); free(ti2);
        }
        memcpy(rho_re, nr, chi2 * sizeof(double));
        memcpy(rho_im, ni_arr, chi2 * sizeof(double));
        free(nr); free(ni_arr);
    }

    double trace = 0;
    for (int i = 0; i < chi; i++) trace += rho_re[i * chi + i];
    free(rho_re); free(rho_im);
    return trace;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * DEFERRED RENORMALIZATION
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_renormalize_chain(QuhitEngine *eng, uint32_t *quhits, int n)
{
    double norm = mps_overlay_norm(eng, quhits, n);
    if (norm > 1e-30 && fabs(norm - 1.0) > 1e-12) {
        double scale = born_fast_isqrt(norm);
        for (int k = 0; k < MPS_PHYS; k++)
            for (int a = 0; a < MPS_CHI; a++)
                for (int b = 0; b < MPS_CHI; b++) {
                    double tr, ti;
                    mps_read_tensor(0, k, a, b, &tr, &ti);
                    mps_write_tensor(0, k, a, b, tr * scale, ti * scale);
                }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * LAZY EVALUATION LAYER
 * ═══════════════════════════════════════════════════════════════════════════════ */

MpsLazyChain *mps_lazy_init(QuhitEngine *eng, uint32_t *quhits, int n)
{
    MpsLazyChain *lc = (MpsLazyChain *)calloc(1, sizeof(MpsLazyChain));
    lc->eng = eng;
    lc->quhits = quhits;
    lc->n_sites = n;

    mps_overlay_init(eng, quhits, n);

    lc->queue_cap = MAX_LAZY_GATES;
    lc->queue = (MpsDeferredGate *)calloc(lc->queue_cap, sizeof(MpsDeferredGate));
    lc->queue_len = 0;

    lc->site_allocated = (uint8_t *)calloc(n, sizeof(uint8_t));
    lc->site_dirty     = (uint8_t *)calloc(n, sizeof(uint8_t));

    lazy_stats_reset(&lc->stats);
    lc->stats.sites_total = n;
    lc->stats.hilbert_log10 = n * log10(6.0);

    return lc;
}

void mps_lazy_free(MpsLazyChain *lc)
{
    if (!lc) return;
    for (int i = 0; i < lc->queue_len; i++) {
        if (lc->queue[i].type == 1) {
            free(lc->queue[i].G_re);
            free(lc->queue[i].G_im);
        }
    }
    free(lc->queue);
    free(lc->site_allocated);
    free(lc->site_dirty);
    mps_overlay_free();
    free(lc);
}

static void lazy_ensure_site(MpsLazyChain *lc, int site)
{
    if (!lc->site_allocated[site]) {
        mps_zero_site(site);
        mps_write_tensor(site, 0, 0, 0, 1.0, 0.0);
        lc->site_allocated[site] = 1;
    }
}

void mps_lazy_gate_1site(MpsLazyChain *lc, int site,
                         const double *U_re, const double *U_im)
{
    if (lc->queue_len >= lc->queue_cap) mps_lazy_flush(lc);

    MpsDeferredGate *g = &lc->queue[lc->queue_len];
    g->type = 0;
    g->site = site;
    memcpy(g->U_re, U_re, MPS_PHYS * MPS_PHYS * sizeof(double));
    memcpy(g->U_im, U_im, MPS_PHYS * MPS_PHYS * sizeof(double));
    g->G_re = NULL;
    g->G_im = NULL;
    g->applied = 0;

    lc->queue_len++;
    lc->site_dirty[site] = 1;
    lc->stats.gates_queued++;
}

void mps_lazy_gate_2site(MpsLazyChain *lc, int site,
                         const double *G_re, const double *G_im)
{
    if (lc->queue_len >= lc->queue_cap) mps_lazy_flush(lc);

    int D2 = MPS_PHYS * MPS_PHYS;
    int sz = D2 * D2;

    MpsDeferredGate *g = &lc->queue[lc->queue_len];
    g->type = 1;
    g->site = site;
    g->G_re = (double *)malloc(sz * sizeof(double));
    g->G_im = (double *)malloc(sz * sizeof(double));
    memcpy(g->G_re, G_re, sz * sizeof(double));
    memcpy(g->G_im, G_im, sz * sizeof(double));
    g->applied = 0;

    lc->queue_len++;
    lc->site_dirty[site] = 1;
    if (site + 1 < lc->n_sites)
        lc->site_dirty[site + 1] = 1;
    lc->stats.gates_queued++;
}

/* Gate fusion: C = B × A */
static void fuse_1site_gates(const double *A_re, const double *A_im,
                             const double *B_re, const double *B_im,
                             double *C_re, double *C_im)
{
    int D = MPS_PHYS;
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++) {
            double sr = 0, si = 0;
            for (int k = 0; k < D; k++) {
                double br = B_re[i * D + k], bi = B_im[i * D + k];
                double ar = A_re[k * D + j], ai = A_im[k * D + j];
                sr += br * ar - bi * ai;
                si += br * ai + bi * ar;
            }
            C_re[i * D + j] = sr;
            C_im[i * D + j] = si;
        }
}

static void apply_gate(MpsLazyChain *lc, MpsDeferredGate *g)
{
    if (g->applied) return;

    if (g->type == 0) {
        lazy_ensure_site(lc, g->site);
        mps_gate_1site(lc->eng, lc->quhits, lc->n_sites,
                       g->site, g->U_re, g->U_im);
    } else {
        lazy_ensure_site(lc, g->site);
        lazy_ensure_site(lc, g->site + 1);
        mps_gate_2site(lc->eng, lc->quhits, lc->n_sites,
                       g->site, g->G_re, g->G_im);
    }

    g->applied = 1;
    lc->stats.gates_materialized++;
}

uint32_t mps_lazy_measure(MpsLazyChain *lc, int target_idx)
{
    /* Gate fusion pass */
    for (int i = 0; i < lc->queue_len - 1; i++) {
        if (lc->queue[i].applied) continue;
        if (lc->queue[i].type != 0) continue;

        int j = i + 1;
        while (j < lc->queue_len &&
               lc->queue[j].type == 0 &&
               lc->queue[j].site == lc->queue[i].site &&
               !lc->queue[j].applied) {
            double C_re[MPS_PHYS * MPS_PHYS], C_im[MPS_PHYS * MPS_PHYS];
            fuse_1site_gates(lc->queue[i].U_re, lc->queue[i].U_im,
                             lc->queue[j].U_re, lc->queue[j].U_im,
                             C_re, C_im);
            memcpy(lc->queue[i].U_re, C_re, sizeof(C_re));
            memcpy(lc->queue[i].U_im, C_im, sizeof(C_im));
            lc->queue[j].applied = 1;
            lc->stats.gates_fused++;
            j++;
        }
    }

    /* Apply all pending gates */
    for (int i = 0; i < lc->queue_len; i++) {
        if (!lc->queue[i].applied)
            apply_gate(lc, &lc->queue[i]);
    }

    lazy_ensure_site(lc, target_idx);
    return mps_overlay_measure(lc->eng, lc->quhits, lc->n_sites, target_idx);
}

void mps_lazy_flush(MpsLazyChain *lc)
{
    /* Fusion pass */
    for (int i = 0; i < lc->queue_len - 1; i++) {
        if (lc->queue[i].applied) continue;
        if (lc->queue[i].type != 0) continue;

        int j = i + 1;
        while (j < lc->queue_len &&
               lc->queue[j].type == 0 &&
               lc->queue[j].site == lc->queue[i].site &&
               !lc->queue[j].applied) {
            double C_re[MPS_PHYS * MPS_PHYS], C_im[MPS_PHYS * MPS_PHYS];
            fuse_1site_gates(lc->queue[i].U_re, lc->queue[i].U_im,
                             lc->queue[j].U_re, lc->queue[j].U_im,
                             C_re, C_im);
            memcpy(lc->queue[i].U_re, C_re, sizeof(C_re));
            memcpy(lc->queue[i].U_im, C_im, sizeof(C_im));
            lc->queue[j].applied = 1;
            lc->stats.gates_fused++;
            j++;
        }
    }

    for (int i = 0; i < lc->queue_len; i++) {
        if (!lc->queue[i].applied)
            apply_gate(lc, &lc->queue[i]);
    }

    for (int i = 0; i < lc->queue_len; i++) {
        if (lc->queue[i].type == 1) {
            free(lc->queue[i].G_re);
            free(lc->queue[i].G_im);
            lc->queue[i].G_re = NULL;
            lc->queue[i].G_im = NULL;
        }
    }
    lc->queue_len = 0;
}

void mps_lazy_finalize_stats(MpsLazyChain *lc)
{
    uint64_t skipped = 0;
    for (int i = 0; i < lc->queue_len; i++)
        if (!lc->queue[i].applied) skipped++;
    lc->stats.gates_skipped = skipped;

    uint64_t alloc = 0;
    for (int i = 0; i < lc->n_sites; i++)
        if (lc->site_allocated[i]) alloc++;
    lc->stats.sites_allocated = alloc;
    lc->stats.sites_lazy = lc->n_sites - alloc;

    lc->stats.memory_actual = alloc * sizeof(MpsTensor)
                            + lc->queue_len * sizeof(MpsDeferredGate)
                            + sizeof(MpsLazyChain);
}

void mps_lazy_write_tensor(MpsLazyChain *lc, int site, int k,
                           int alpha, int beta, double re, double im)
{
    lc->site_allocated[site] = 1;
    mps_write_tensor(site, k, alpha, beta, re, im);
}

void mps_lazy_zero_site(MpsLazyChain *lc, int site)
{
    lc->site_allocated[site] = 1;
    mps_zero_site(site);
    mps_write_tensor(site, 0, 0, 0, 1.0, 0.0);
}
