/*
 * mps_overlay.c — Implementation of the Side-Channel Overlay
 *
 * This implementation provides the logic to treat the engine's
 * pairwise memory blocks (QuhitPair) as nodes in a Matrix Product State.
 */

#include "mps_overlay.h"
#include <string.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════════════
 * INIT
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_overlay_init(QuhitEngine *eng, uint32_t *quhits, int n)
{
    /* For each quhit, create a self-pair (or dummy pair) to allocate storage. */
    /* We use 'product entangle' with a dummy to get a fresh pair struct. */
    
    for (int i = 0; i < n; i++) {
        uint32_t dummy = quhit_init(eng);
        quhit_entangle_product(eng, quhits[i], dummy);
        
        /* Zero out the storage to prepare for MPS tensor use */
        int pid = eng->quhits[quhits[i]].pair_id;
        if (pid >= 0) {
            memset(&eng->pairs[pid].joint, 0, 576);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * W-STATE CONSTRUCTION
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_overlay_write_w_state(QuhitEngine *eng, uint32_t *quhits, int n)
{
    /* Distribute normalization evenly: each site gets N^{-1/(2N)} */
    /* This ensures the right density transfer matrix is well-conditioned */
    double site_scale = pow((double)n, -1.0 / (2.0 * n));

    for (int i = 0; i < n; i++) {
        int pid = eng->quhits[quhits[i]].pair_id;
        if (pid < 0) continue;
        QuhitPair *p = &eng->pairs[pid];

        /* A[0] = Identity on bond (scaled) */
        mps_write_tensor(p, 0, 0, 0, site_scale, 0.0);
        mps_write_tensor(p, 0, 1, 1, site_scale, 0.0);

        /* A[1] = Transition 0→1 (scaled) */
        mps_write_tensor(p, 1, 0, 1, site_scale, 0.0);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * AMPLITUDE INSPECTION
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_overlay_amplitude(QuhitEngine *eng, uint32_t *quhits, int n,
                           const uint32_t *basis, double *out_re, double *out_im)
{
    /* Contract L * A[k0] * ... * A[kn-1] * R */
    /* L = [1, 0], R = [0, 1]^T */
    
    double v_re[2] = {1.0, 0.0};
    double v_im[2] = {0.0, 0.0};

    for (int i = 0; i < n; i++) {
        int pid = eng->quhits[quhits[i]].pair_id;
        QuhitPair *p = &eng->pairs[pid];
        int k = (int)basis[i];

        double next_re[2] = {0, 0};
        double next_im[2] = {0, 0};

        /* v_next[beta] = Σ_alpha v[alpha] * A[k][alpha][beta] */
        for (int beta = 0; beta < 2; beta++) {
            for (int alpha = 0; alpha < 2; alpha++) {
                double t_re, t_im;
                mps_read_tensor(p, k, alpha, beta, &t_re, &t_im);
                
                next_re[beta] += v_re[alpha]*t_re - v_im[alpha]*t_im;
                next_im[beta] += v_re[alpha]*t_im + v_im[alpha]*t_re;
            }
        }
        v_re[0] = next_re[0]; v_re[1] = next_re[1];
        v_im[0] = next_im[0]; v_im[1] = next_im[1];
    }

    *out_re = v_re[1]; /* Project onto R=[0,1] */
    *out_im = v_im[1];
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MEASUREMENT
 * ═══════════════════════════════════════════════════════════════════════════════ */

uint32_t mps_overlay_measure(QuhitEngine *eng, uint32_t *quhits, int n, int target_idx)
{
    /*
     * Full left-right density environment contraction for exact P(k).
     * Cost: O(N × χ³ × D) = O(N × 8 × 6) = O(48N). Trivial for χ=2.
     *
     * Step 1: Build right density environment ρ_R from site N-1 down to target+1.
     *         ρ_R[j] = Σ_k A[k]_j† × ρ_R[j+1] × A[k]_j
     *         Boundary: ρ_R[N] = |R⟩⟨R| where R=[0,...,0,1] (last bond index).
     *
     * Step 2: Build left environment vector L from site 0 up to target-1.
     *         For previously-measured sites, use the projected tensor.
     *         L starts as [1, 0, ..., 0].
     *
     * Step 3: For each physical index k at target:
     *         P(k) = L† × A[k]† × ρ_R[target] × A[k] × L
     *
     * Step 4: Born sample from {P(k)}.
     * Step 5: Project tensor at target, renormalize.
     */

    /* ── Step 1: Right density environment ── */
    /* ρ_R is a χ×χ = 2×2 Hermitian matrix at each site */
    double rho_R[MPS_CHI][MPS_CHI]; /* current right env */

    /* Boundary: |R⟩⟨R| where R = [0, 1] */
    rho_R[0][0] = 0; rho_R[0][1] = 0;
    rho_R[1][0] = 0; rho_R[1][1] = 1;

    /* Sweep from site N-1 down to target+1 */
    for (int j = n - 1; j > target_idx; j--) {
        int pid = eng->quhits[quhits[j]].pair_id;
        if (pid < 0) continue;
        QuhitPair *p = &eng->pairs[pid];

        double new_rho[MPS_CHI][MPS_CHI] = {{0}};

        for (int k = 0; k < MPS_PHYS; k++) {
            /* Extract A[k] (χ×χ real matrix for typical states) */
            double A[MPS_CHI][MPS_CHI];
            for (int a = 0; a < MPS_CHI; a++)
                for (int b = 0; b < MPS_CHI; b++) {
                    double re, im;
                    mps_read_tensor(p, k, a, b, &re, &im);
                    A[a][b] = re; /* general case would track im separately */
                }

            /* tmp = A[k] * ρ_R  (correct: multiply from left) */
            double tmp[MPS_CHI][MPS_CHI] = {{0}};
            for (int a = 0; a < MPS_CHI; a++)
                for (int b = 0; b < MPS_CHI; b++)
                    for (int c = 0; c < MPS_CHI; c++)
                        tmp[a][b] += A[a][c] * rho_R[c][b];

            /* new_rho += tmp * A[k]^T  (= A ρ A^T, correct transfer) */
            for (int a = 0; a < MPS_CHI; a++)
                for (int b = 0; b < MPS_CHI; b++)
                    for (int c = 0; c < MPS_CHI; c++)
                        new_rho[a][b] += tmp[a][c] * A[b][c];
        }

        memcpy(rho_R, new_rho, sizeof(rho_R));
    }

    /* ── Step 2: Left environment vector ── */
    double L[MPS_CHI] = {1.0, 0.0}; /* Boundary: L = [1, 0] */

    for (int j = 0; j < target_idx; j++) {
        int pid = eng->quhits[quhits[j]].pair_id;
        if (pid < 0) continue;
        QuhitPair *p = &eng->pairs[pid];

        /* For previously measured sites, the tensor has been projected.
         * We use whatever physical slice remains (non-zero). */
        double new_L[MPS_CHI] = {0};
        for (int k = 0; k < MPS_PHYS; k++) {
            double Ak[MPS_CHI][MPS_CHI];
            int nonzero = 0;
            for (int a = 0; a < MPS_CHI; a++)
                for (int b = 0; b < MPS_CHI; b++) {
                    double re, im;
                    mps_read_tensor(p, k, a, b, &re, &im);
                    Ak[a][b] = re;
                    if (re != 0 || im != 0) nonzero = 1;
                }
            if (!nonzero) continue; /* This slice was zeroed by projection */

            /* new_L[b] += Σ_a L[a] × A[k][a][b] */
            for (int b = 0; b < MPS_CHI; b++)
                for (int a = 0; a < MPS_CHI; a++)
                    new_L[b] += L[a] * Ak[a][b];
        }
        L[0] = new_L[0]; L[1] = new_L[1];
    }

    /* ── Step 3: Compute P(k) for each physical index ── */
    double probs[MPS_PHYS];
    double total_prob = 0;

    int target_pid = eng->quhits[quhits[target_idx]].pair_id;
    QuhitPair *pt = &eng->pairs[target_pid];

    for (int k = 0; k < MPS_PHYS; k++) {
        double Ak[MPS_CHI][MPS_CHI];
        for (int a = 0; a < MPS_CHI; a++)
            for (int b = 0; b < MPS_CHI; b++) {
                double re, im;
                mps_read_tensor(pt, k, a, b, &re, &im);
                Ak[a][b] = re;
            }

        /* mid[b] = Σ_a L[a] × A[k][a][b] */
        double mid[MPS_CHI] = {0};
        for (int b = 0; b < MPS_CHI; b++)
            for (int a = 0; a < MPS_CHI; a++)
                mid[b] += L[a] * Ak[a][b];

        /* P(k) = mid^T × ρ_R × mid */
        double pk = 0;
        for (int a = 0; a < MPS_CHI; a++)
            for (int b = 0; b < MPS_CHI; b++)
                pk += mid[a] * rho_R[a][b] * mid[b];

        probs[k] = pk > 0 ? pk : 0;
        total_prob += probs[k];
    }

    /* ── Step 4: Born sample ── */
    if (total_prob > 1e-30)
        for (int k = 0; k < MPS_PHYS; k++) probs[k] /= total_prob;

    double r = quhit_prng_double(eng);
    uint32_t outcome = 0;
    double cdf = 0;
    for (int k = 0; k < MPS_PHYS; k++) {
        cdf += probs[k];
        if (r < cdf) { outcome = (uint32_t)k; break; }
    }

    /* ── Step 5: Project + renormalize tensor at target ── */
    for (int k = 0; k < MPS_PHYS; k++) {
        if ((uint32_t)k != outcome) {
            for (int a = 0; a < MPS_CHI; a++)
                for (int b = 0; b < MPS_CHI; b++)
                    mps_write_tensor(pt, k, a, b, 0, 0);
        }
    }

    /* Renormalize the surviving slice */
    double slice_norm2 = 0;
    for (int a = 0; a < MPS_CHI; a++)
        for (int b = 0; b < MPS_CHI; b++) {
            double re, im;
            mps_read_tensor(pt, (int)outcome, a, b, &re, &im);
            slice_norm2 += re * re + im * im;
        }
    if (slice_norm2 > 1e-30) {
        double scale = 1.0 / sqrt(slice_norm2);
        for (int a = 0; a < MPS_CHI; a++)
            for (int b = 0; b < MPS_CHI; b++) {
                double re, im;
                mps_read_tensor(pt, (int)outcome, a, b, &re, &im);
                mps_write_tensor(pt, (int)outcome, a, b, re * scale, im * scale);
            }
    }

    return outcome;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PRODUCT STATE |0⟩^⊗N
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_overlay_write_zero(QuhitEngine *eng, uint32_t *quhits, int n)
{
    for (int i = 0; i < n; i++) {
        int pid = eng->quhits[quhits[i]].pair_id;
        if (pid < 0) continue;
        QuhitPair *p = &eng->pairs[pid];

        /* A[0] = I (only k=0 has content, identity on bond) */
        /* A[k>0] = 0 */
        mps_write_tensor(p, 0, 0, 0, 1.0, 0.0);
        mps_write_tensor(p, 0, 1, 1, 1.0, 0.0);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SINGLE-SITE GATE
 *
 * A'[k'][α][β] = Σ_k U[k'][k] × A[k][α][β]
 *
 * This is a rotation of the physical index.
 * Cost: O(D² × χ²) = O(144). Trivial.
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_gate_1site(QuhitEngine *eng, uint32_t *quhits, int n,
                    int site, const double *U_re, const double *U_im)
{
    (void)n;
    int pid = eng->quhits[quhits[site]].pair_id;
    if (pid < 0) return;
    QuhitPair *p = &eng->pairs[pid];

    /* Read current tensors into buffer */
    double old_re[MPS_PHYS][MPS_CHI][MPS_CHI];
    double old_im[MPS_PHYS][MPS_CHI][MPS_CHI];

    for (int k = 0; k < MPS_PHYS; k++)
        for (int a = 0; a < MPS_CHI; a++)
            for (int b = 0; b < MPS_CHI; b++)
                mps_read_tensor(p, k, a, b, &old_re[k][a][b], &old_im[k][a][b]);

    /* Apply: A'[k'] = Σ_k U[k'][k] × A[k] */
    for (int kp = 0; kp < MPS_PHYS; kp++) {
        for (int a = 0; a < MPS_CHI; a++) {
            for (int b = 0; b < MPS_CHI; b++) {
                double sum_re = 0, sum_im = 0;
                for (int k = 0; k < MPS_PHYS; k++) {
                    double u_re = U_re[kp * MPS_PHYS + k];
                    double u_im = U_im[kp * MPS_PHYS + k];
                    sum_re += u_re * old_re[k][a][b] - u_im * old_im[k][a][b];
                    sum_im += u_re * old_im[k][a][b] + u_im * old_re[k][a][b];
                }
                mps_write_tensor(p, kp, a, b, sum_re, sum_im);
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * TWO-SITE GATE WITH SVD TRUNCATION
 *
 * Steps:
 *  1. Contract adjacent tensors into Θ[k,l][α][γ]
 *  2. Apply gate: Θ'[k',l'][α][γ] = Σ_{k,l} G[k'l',kl] × Θ[k,l][α][γ]
 *  3. Reshape: M[(k',α)][(l',γ)] — a (Dχ) × (Dχ) = 12×12 matrix
 *  4. SVD: M = U Σ V†, keep top χ singular values
 *  5. New A_i[k'][α][β'] = U[k'χ+α, β'], A_{i+1}[l'][β'][γ] = σ_β' V†[β', l'χ+γ]
 *
 * For D=6, χ=2: M is 12×12. SVD is done via Jacobi eigendecomposition
 * of M†M (12×12 Hermitian → trivial for this size).
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Internal: 2×2 SVD via analytic formulas (for our truncated output) */
/* Full SVD for m×n complex matrix: use Jacobi one-sided for small matrices */

/*
 * For Dχ=12, we need a 12×12 SVD. We compute M†M (12×12 Hermitian),
 * find its eigenvalues/eigenvectors, then derive U and V.
 * Since χ=2, we only need the TOP 2 singular values.
 *
 * Simplified approach: compute the 12×12 M†M, extract top-2 eigenvectors
 * via power iteration (2 iterations suffice for well-separated eigenvalues).
 */

#define DCHI (MPS_PHYS * MPS_CHI)  /* 12 */

void mps_gate_2site(QuhitEngine *eng, uint32_t *quhits, int n,
                    int site, const double *G_re, const double *G_im)
{
    (void)n;
    int pid_i = eng->quhits[quhits[site]].pair_id;
    int pid_j = eng->quhits[quhits[site + 1]].pair_id;
    if (pid_i < 0 || pid_j < 0) return;
    QuhitPair *pi = &eng->pairs[pid_i];
    QuhitPair *pj = &eng->pairs[pid_j];

    /* Step 1: Read tensors */
    double Ai_re[MPS_PHYS][MPS_CHI][MPS_CHI], Ai_im[MPS_PHYS][MPS_CHI][MPS_CHI];
    double Aj_re[MPS_PHYS][MPS_CHI][MPS_CHI], Aj_im[MPS_PHYS][MPS_CHI][MPS_CHI];

    for (int k = 0; k < MPS_PHYS; k++)
        for (int a = 0; a < MPS_CHI; a++)
            for (int b = 0; b < MPS_CHI; b++) {
                mps_read_tensor(pi, k, a, b, &Ai_re[k][a][b], &Ai_im[k][a][b]);
                mps_read_tensor(pj, k, a, b, &Aj_re[k][a][b], &Aj_im[k][a][b]);
            }

    /* Step 2: Contract Θ[k,l,α,γ] = Σ_β Ai[k][α][β] × Aj[l][β][γ] */
    double Th_re[MPS_PHYS][MPS_PHYS][MPS_CHI][MPS_CHI];
    double Th_im[MPS_PHYS][MPS_PHYS][MPS_CHI][MPS_CHI];
    memset(Th_re, 0, sizeof(Th_re));
    memset(Th_im, 0, sizeof(Th_im));

    for (int k = 0; k < MPS_PHYS; k++)
        for (int l = 0; l < MPS_PHYS; l++)
            for (int a = 0; a < MPS_CHI; a++)
                for (int g = 0; g < MPS_CHI; g++)
                    for (int b = 0; b < MPS_CHI; b++) {
                        double ar = Ai_re[k][a][b], ai = Ai_im[k][a][b];
                        double br = Aj_re[l][b][g], bi = Aj_im[l][b][g];
                        Th_re[k][l][a][g] += ar*br - ai*bi;
                        Th_im[k][l][a][g] += ar*bi + ai*br;
                    }

    /* Step 3: Apply gate: Θ'[k',l',α,γ] = Σ_{k,l} G[k'D+l', kD+l] × Θ[k,l,α,γ] */
    double Tp_re[MPS_PHYS][MPS_PHYS][MPS_CHI][MPS_CHI];
    double Tp_im[MPS_PHYS][MPS_PHYS][MPS_CHI][MPS_CHI];
    memset(Tp_re, 0, sizeof(Tp_re));
    memset(Tp_im, 0, sizeof(Tp_im));

    int D2 = MPS_PHYS * MPS_PHYS;
    for (int kp = 0; kp < MPS_PHYS; kp++)
        for (int lp = 0; lp < MPS_PHYS; lp++) {
            int row = kp * MPS_PHYS + lp;
            for (int k = 0; k < MPS_PHYS; k++)
                for (int l = 0; l < MPS_PHYS; l++) {
                    int col = k * MPS_PHYS + l;
                    double gr = G_re[row * D2 + col];
                    double gi = G_im[row * D2 + col];
                    for (int a = 0; a < MPS_CHI; a++)
                        for (int g = 0; g < MPS_CHI; g++) {
                            double tr = Th_re[k][l][a][g];
                            double ti = Th_im[k][l][a][g];
                            Tp_re[kp][lp][a][g] += gr*tr - gi*ti;
                            Tp_im[kp][lp][a][g] += gr*ti + gi*tr;
                        }
                }
        }

    /* Step 4: Reshape to M[(k',α)][(l',γ)] — 12×12 matrix */
    /* Row index: r = k' * χ + α  (0..11) */
    /* Col index: c = l' * χ + γ  (0..11) */
    double M_re[DCHI][DCHI], M_im[DCHI][DCHI];

    for (int kp = 0; kp < MPS_PHYS; kp++)
        for (int a = 0; a < MPS_CHI; a++) {
            int r = kp * MPS_CHI + a;
            for (int lp = 0; lp < MPS_PHYS; lp++)
                for (int g = 0; g < MPS_CHI; g++) {
                    int c = lp * MPS_CHI + g;
                    M_re[r][c] = Tp_re[kp][lp][a][g];
                    M_im[r][c] = Tp_im[kp][lp][a][g];
                }
        }

    /* Step 5: SVD of 12×12 M, keep top χ=2 singular values.
     *
     * Strategy: Compute H = M†M (12×12 Hermitian positive semi-definite).
     * Find top-2 eigenvectors of H via power iteration + deflation.
     * σ_i = √(λ_i), V columns = eigenvectors of H, U = M V Σ⁻¹.
     */

    /* Compute H = M†M */
    double H_re[DCHI][DCHI], H_im[DCHI][DCHI];
    for (int i = 0; i < DCHI; i++)
        for (int j = 0; j < DCHI; j++) {
            double sr = 0, si = 0;
            for (int r = 0; r < DCHI; r++) {
                /* M†[i][r] = conj(M[r][i]) */
                double mr = M_re[r][i], mi = -M_im[r][i]; /* conj */
                double nr = M_re[r][j], ni = M_im[r][j];
                sr += mr*nr - mi*ni;
                si += mr*ni + mi*nr;
            }
            H_re[i][j] = sr;
            H_im[i][j] = si;
        }

    /* Power iteration for top eigenvector v1 */
    double v1_re[DCHI], v1_im[DCHI];
    /* Initialize to random-ish unit vector */
    for (int i = 0; i < DCHI; i++) {
        v1_re[i] = (i == 0) ? 1.0 : 0.0;
        v1_im[i] = 0;
    }

    for (int iter = 0; iter < 40; iter++) {
        /* w = H × v */
        double w_re[DCHI] = {0}, w_im[DCHI] = {0};
        for (int i = 0; i < DCHI; i++)
            for (int j = 0; j < DCHI; j++) {
                w_re[i] += H_re[i][j]*v1_re[j] - H_im[i][j]*v1_im[j];
                w_im[i] += H_re[i][j]*v1_im[j] + H_im[i][j]*v1_re[j];
            }
        /* Normalize */
        double norm = 0;
        for (int i = 0; i < DCHI; i++)
            norm += w_re[i]*w_re[i] + w_im[i]*w_im[i];
        norm = sqrt(norm);
        if (norm < 1e-30) break;
        for (int i = 0; i < DCHI; i++) {
            v1_re[i] = w_re[i] / norm;
            v1_im[i] = w_im[i] / norm;
        }
    }

    /* Eigenvalue λ1 = v1† H v1 */
    double lam1 = 0;
    for (int i = 0; i < DCHI; i++) {
        double hr = 0, hi = 0;
        for (int j = 0; j < DCHI; j++) {
            hr += H_re[i][j]*v1_re[j] - H_im[i][j]*v1_im[j];
            hi += H_re[i][j]*v1_im[j] + H_im[i][j]*v1_re[j];
        }
        lam1 += v1_re[i]*hr + v1_im[i]*hi; /* real part of v1† H v1 */
    }
    double sig1 = sqrt(fabs(lam1));

    /* Deflate: H' = H - λ1 × |v1⟩⟨v1| */
    for (int i = 0; i < DCHI; i++)
        for (int j = 0; j < DCHI; j++) {
            /* |v1⟩⟨v1|[i][j] = v1[i] × conj(v1[j]) */
            double vv_re = v1_re[i]*v1_re[j] + v1_im[i]*v1_im[j];
            double vv_im = v1_im[i]*v1_re[j] - v1_re[i]*v1_im[j];
            H_re[i][j] -= lam1 * vv_re;
            H_im[i][j] -= lam1 * vv_im;
        }

    /* Power iteration for second eigenvector v2 */
    double v2_re[DCHI], v2_im[DCHI];
    for (int i = 0; i < DCHI; i++) {
        v2_re[i] = (i == 1) ? 1.0 : 0.0;
        v2_im[i] = 0;
    }

    for (int iter = 0; iter < 40; iter++) {
        double w_re[DCHI] = {0}, w_im[DCHI] = {0};
        for (int i = 0; i < DCHI; i++)
            for (int j = 0; j < DCHI; j++) {
                w_re[i] += H_re[i][j]*v2_re[j] - H_im[i][j]*v2_im[j];
                w_im[i] += H_re[i][j]*v2_im[j] + H_im[i][j]*v2_re[j];
            }
        double norm = 0;
        for (int i = 0; i < DCHI; i++)
            norm += w_re[i]*w_re[i] + w_im[i]*w_im[i];
        norm = sqrt(norm);
        if (norm < 1e-30) break;
        for (int i = 0; i < DCHI; i++) {
            v2_re[i] = w_re[i] / norm;
            v2_im[i] = w_im[i] / norm;
        }
    }

    double lam2 = 0;
    for (int i = 0; i < DCHI; i++) {
        double hr = 0, hi = 0;
        for (int j = 0; j < DCHI; j++) {
            hr += H_re[i][j]*v2_re[j] - H_im[i][j]*v2_im[j];
            hi += H_re[i][j]*v2_im[j] + H_im[i][j]*v2_re[j];
        }
        lam2 += v2_re[i]*hr + v2_im[i]*hi;
    }
    double sig2 = sqrt(fabs(lam2));

    /* U columns: u_k = M × v_k / σ_k */
    double u1_re[DCHI] = {0}, u1_im[DCHI] = {0};
    double u2_re[DCHI] = {0}, u2_im[DCHI] = {0};

    if (sig1 > 1e-30) {
        for (int i = 0; i < DCHI; i++) {
            for (int j = 0; j < DCHI; j++) {
                u1_re[i] += M_re[i][j]*v1_re[j] - M_im[i][j]*v1_im[j];
                u1_im[i] += M_re[i][j]*v1_im[j] + M_im[i][j]*v1_re[j];
            }
            u1_re[i] /= sig1; u1_im[i] /= sig1;
        }
    }
    if (sig2 > 1e-30) {
        for (int i = 0; i < DCHI; i++) {
            for (int j = 0; j < DCHI; j++) {
                u2_re[i] += M_re[i][j]*v2_re[j] - M_im[i][j]*v2_im[j];
                u2_im[i] += M_re[i][j]*v2_im[j] + M_im[i][j]*v2_re[j];
            }
            u2_re[i] /= sig2; u2_im[i] /= sig2;
        }
    }

    /* Step 6: Write back new tensors.
     *
     * A'_i[k'][α][β'] where row = k'χ+α, col = β' (0 or 1).
     * A'_i[k'][α][0] = u1[k'χ+α], A'_i[k'][α][1] = u2[k'χ+α]
     *
     * A'_j[l'][β'][γ] where Σ⁻¹V†[β'][l'χ+γ]
     * (Σ V†)[0][c] = σ1 × conj(v1[c]), (Σ V†)[1][c] = σ2 × conj(v2[c])
     * A'_j[l'][0][γ] = sig1 × conj(v1[l'χ+γ])
     * A'_j[l'][1][γ] = sig2 × conj(v2[l'χ+γ])
     */

    /* Clear old tensors */
    for (int k = 0; k < MPS_PHYS; k++)
        for (int a = 0; a < MPS_CHI; a++)
            for (int b = 0; b < MPS_CHI; b++) {
                mps_write_tensor(pi, k, a, b, 0, 0);
                mps_write_tensor(pj, k, a, b, 0, 0);
            }

    /* Write A_i: columns of U */
    for (int kp = 0; kp < MPS_PHYS; kp++)
        for (int a = 0; a < MPS_CHI; a++) {
            int r = kp * MPS_CHI + a;
            mps_write_tensor(pi, kp, a, 0, u1_re[r], u1_im[r]);
            mps_write_tensor(pi, kp, a, 1, u2_re[r], u2_im[r]);
        }

    /* Write A_j: rows of Σ V† */
    for (int lp = 0; lp < MPS_PHYS; lp++)
        for (int g = 0; g < MPS_CHI; g++) {
            int c = lp * MPS_CHI + g;
            /* (Σ V†)[0][c] = σ1 × conj(v1[c]) */
            mps_write_tensor(pj, lp, 0, g, sig1 * v1_re[c],  sig1 * (-v1_im[c]));
            mps_write_tensor(pj, lp, 1, g, sig2 * v2_re[c],  sig2 * (-v2_im[c]));
        }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * GATE CONSTRUCTORS
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_build_dft6(double *U_re, double *U_im)
{
    /* DFT₆: U[j][k] = (1/√6) × ω^{jk}, ω = e^{2πi/6} */
    double inv_sqrt6 = 1.0 / sqrt(6.0);
    for (int j = 0; j < 6; j++)
        for (int k = 0; k < 6; k++) {
            double angle = 2.0 * M_PI * j * k / 6.0;
            U_re[j * 6 + k] = inv_sqrt6 * cos(angle);
            U_im[j * 6 + k] = inv_sqrt6 * sin(angle);
        }
}

void mps_build_cz(double *G_re, double *G_im)
{
    /* CZ: diagonal, phase ω^{k·l} where ω = e^{2πi/6} */
    int D2 = MPS_PHYS * MPS_PHYS; /* 36 */
    memset(G_re, 0, D2 * D2 * sizeof(double));
    memset(G_im, 0, D2 * D2 * sizeof(double));

    for (int k = 0; k < MPS_PHYS; k++)
        for (int l = 0; l < MPS_PHYS; l++) {
            int idx = (k * MPS_PHYS + l) * D2 + (k * MPS_PHYS + l);
            double angle = 2.0 * M_PI * k * l / 6.0;
            G_re[idx] = cos(angle);
            G_im[idx] = sin(angle);
        }
}

void mps_build_controlled_phase(double *G_re, double *G_im, int power)
{
    /* Controlled rotation: phase = e^{2πi × k × l × 2^power / D²}
     * Used in QFT decomposition for phase kickback. */
    int D2 = MPS_PHYS * MPS_PHYS;
    memset(G_re, 0, D2 * D2 * sizeof(double));
    memset(G_im, 0, D2 * D2 * sizeof(double));

    double phase_factor = (double)(1 << power) / (double)(MPS_PHYS * MPS_PHYS);

    for (int k = 0; k < MPS_PHYS; k++)
        for (int l = 0; l < MPS_PHYS; l++) {
            int idx = (k * MPS_PHYS + l) * D2 + (k * MPS_PHYS + l);
            double angle = 2.0 * M_PI * k * l * phase_factor;
            G_re[idx] = cos(angle);
            G_im[idx] = sin(angle);
        }
}

void mps_build_hadamard2(double *U_re, double *U_im)
{
    /* Hadamard on qubit subspace {|0⟩, |1⟩}, identity on {|2⟩..|5⟩} */
    memset(U_re, 0, 36 * sizeof(double));
    memset(U_im, 0, 36 * sizeof(double));

    double s = 1.0 / sqrt(2.0);

    /* |0'⟩ = (|0⟩ + |1⟩)/√2,  |1'⟩ = (|0⟩ - |1⟩)/√2 */
    U_re[0 * 6 + 0] =  s;  /* H[0][0] */
    U_re[0 * 6 + 1] =  s;  /* H[0][1] */
    U_re[1 * 6 + 0] =  s;  /* H[1][0] */
    U_re[1 * 6 + 1] = -s;  /* H[1][1] */

    /* Identity on 2..5 */
    for (int k = 2; k < 6; k++)
        U_re[k * 6 + k] = 1.0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * NORM COMPUTATION
 *
 * ⟨ψ|ψ⟩ via transfer matrix: T[α,α'][β,β'] = Σ_k A[k][α][β] × conj(A[k][α'][β'])
 * ═══════════════════════════════════════════════════════════════════════════════ */

double mps_overlay_norm(QuhitEngine *eng, uint32_t *quhits, int n)
{
    /* Left boundary: ρ_L = |L⟩⟨L| = [[1,0],[0,0]] */
    double rho_re[MPS_CHI][MPS_CHI] = {{1,0},{0,0}};
    double rho_im[MPS_CHI][MPS_CHI] = {{0}};

    for (int i = 0; i < n; i++) {
        int pid = eng->quhits[quhits[i]].pair_id;
        if (pid < 0) continue;
        QuhitPair *p = &eng->pairs[pid];

        double nr[MPS_CHI][MPS_CHI] = {{0}}, ni[MPS_CHI][MPS_CHI] = {{0}};

        for (int k = 0; k < MPS_PHYS; k++) {
            /* Read A[k] */
            double A_re[MPS_CHI][MPS_CHI], A_im[MPS_CHI][MPS_CHI];
            for (int a = 0; a < MPS_CHI; a++)
                for (int b = 0; b < MPS_CHI; b++)
                    mps_read_tensor(p, k, a, b, &A_re[a][b], &A_im[a][b]);

            /* Transfer: ρ' = Σ_k A[k]† × ρ × A[k]
             * ρ'[β][β'] = Σ_{α,α'} conj(A[k][α][β]) × ρ[α][α'] × A[k][α'][β']
             *
             * Actually for norm: ρ'[β][β'] += Σ_{α,α'} A†[β][α] ρ[α][α'] A[α'][β']
             */

            /* tmp = ρ × A[k] */
            double tmp_re[MPS_CHI][MPS_CHI] = {{0}}, tmp_im[MPS_CHI][MPS_CHI] = {{0}};
            for (int a = 0; a < MPS_CHI; a++)
                for (int bp = 0; bp < MPS_CHI; bp++)
                    for (int ap = 0; ap < MPS_CHI; ap++) {
                        tmp_re[a][bp] += rho_re[a][ap]*A_re[ap][bp] - rho_im[a][ap]*A_im[ap][bp];
                        tmp_im[a][bp] += rho_re[a][ap]*A_im[ap][bp] + rho_im[a][ap]*A_re[ap][bp];
                    }

            /* nr += A[k]† × tmp = conj(A[k])^T × tmp */
            for (int b = 0; b < MPS_CHI; b++)
                for (int bp = 0; bp < MPS_CHI; bp++)
                    for (int a = 0; a < MPS_CHI; a++) {
                        /* A†[b][a] = conj(A[a][b]) */
                        double ar = A_re[a][b], ai = -A_im[a][b]; /* conj */
                        nr[b][bp] += ar*tmp_re[a][bp] - ai*tmp_im[a][bp];
                        ni[b][bp] += ar*tmp_im[a][bp] + ai*tmp_re[a][bp];
                    }
        }

        memcpy(rho_re, nr, sizeof(rho_re));
        memcpy(rho_im, ni, sizeof(rho_im));
    }

    /* Project onto right boundary R=[0,1]: norm = ρ[1][1] */
    return rho_re[1][1]; /* Should be real for a proper state */
}
