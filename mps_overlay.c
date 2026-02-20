/*
 * mps_overlay.c — MPS Overlay with χ=6 (exact, no truncation)
 *
 * All tensor data lives in the dynamically-allocated mps_store.
 * Functions use site index (0..N-1) for tensor access.
 * DCHI = D×χ = 36; Jacobi SVD on 36×36 is exact.
 */

#include "mps_overlay.h"
#include <math.h>

/* ─── Global tensor store ──────────────────────────────────────────────────── */
MpsTensor *mps_store   = NULL;
int        mps_store_n = 0;

/* ═══════════════════════════════════════════════════════════════════════════════
 * INIT / FREE
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_overlay_init(QuhitEngine *eng, uint32_t *quhits, int n)
{
    (void)eng; (void)quhits;
    /* Allocate per-site tensor store (independent of engine pairs) */
    if (mps_store) { free(mps_store); mps_store = NULL; }
    mps_store = (MpsTensor *)calloc((size_t)n, sizeof(MpsTensor));
    mps_store_n = n;
}

void mps_overlay_free(void)
{
    free(mps_store);
    mps_store = NULL;
    mps_store_n = 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * W-STATE CONSTRUCTION
 *
 * Bond semantics (χ=6, but only bond indices 0,1 used for W-state):
 *   bond-0 = "no excitation yet", bond-1 = "excitation placed"
 *   A[0] = scale × I₂ ⊕ 0₄  (pass through)
 *   A[1] = scale × |1⟩⟨0|   (transition 0→1)
 * Boundary: L=[1,0,...,0], R=[0,1,0,...,0]
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_overlay_write_w_state(QuhitEngine *eng, uint32_t *quhits, int n)
{
    (void)eng; (void)quhits;
    double site_scale = pow((double)n, -1.0 / (2.0 * n));

    for (int i = 0; i < n; i++) {
        mps_zero_site(i);
        /* A[0]: pass bond-0→0 and bond-1→1 */
        mps_write_tensor(i, 0, 0, 0, site_scale, 0.0);
        mps_write_tensor(i, 0, 1, 1, site_scale, 0.0);
        /* A[1]: transition bond-0→1 */
        mps_write_tensor(i, 1, 0, 1, site_scale, 0.0);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PRODUCT STATE |0⟩^⊗N
 *
 * Boundary: L = R = e_0 = [1,0,...,0]
 * All sites: A[0][0][0] = 1 (rank-1, uses only bond channel 0)
 * This leaves bond channels 1..χ-1 free for entanglement.
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
 * AMPLITUDE INSPECTION
 *
 * ⟨basis|ψ⟩ = L^T · Π_i A[k_i] · R
 * L = R = e_0 = [1,0,...,0]
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

    /* Project onto R = Σ e_i (sum all boundary channels) */
    double sr = 0, si = 0;
    for (int i = 0; i < MPS_CHI; i++) { sr += v_re[i]; si += v_im[i]; }
    *out_re = sr;
    *out_im = si;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MEASUREMENT
 *
 * Full left-right density environment contraction for exact P(k).
 * Cost: O(N × χ³ × D) per measurement.
 * ═══════════════════════════════════════════════════════════════════════════════ */

uint32_t mps_overlay_measure(QuhitEngine *eng, uint32_t *quhits, int n, int target_idx)
{
    (void)quhits;

    /* ── Step 1: Right density environment ρ_R ── */
    double rho_R[MPS_CHI][MPS_CHI];
    memset(rho_R, 0, sizeof(rho_R));
    rho_R[0][0] = 1.0; /* |R⟩⟨R| where R = e_0 */

    for (int j = n - 1; j > target_idx; j--) {
        double new_rho[MPS_CHI][MPS_CHI] = {{0}};
        for (int k = 0; k < MPS_PHYS; k++) {
            double A[MPS_CHI][MPS_CHI];
            for (int a = 0; a < MPS_CHI; a++)
                for (int b = 0; b < MPS_CHI; b++) {
                    double re, im;
                    mps_read_tensor(j, k, a, b, &re, &im);
                    A[a][b] = re; /* real case */
                }
            /* tmp = A * ρ_R */
            double tmp[MPS_CHI][MPS_CHI] = {{0}};
            for (int a = 0; a < MPS_CHI; a++)
                for (int b = 0; b < MPS_CHI; b++)
                    for (int c = 0; c < MPS_CHI; c++)
                        tmp[a][b] += A[a][c] * rho_R[c][b];
            /* new_rho += tmp * A^T = A ρ A^T */
            for (int a = 0; a < MPS_CHI; a++)
                for (int b = 0; b < MPS_CHI; b++)
                    for (int c = 0; c < MPS_CHI; c++)
                        new_rho[a][b] += tmp[a][c] * A[b][c];
        }
        memcpy(rho_R, new_rho, sizeof(rho_R));
    }

    /* ── Step 2: Left environment vector L ── */
    double L[MPS_CHI];
    memset(L, 0, sizeof(L));
    L[0] = 1.0; /* e_0 */

    for (int j = 0; j < target_idx; j++) {
        double new_L[MPS_CHI] = {0};
        for (int k = 0; k < MPS_PHYS; k++) {
            double Ak[MPS_CHI][MPS_CHI];
            int nonzero = 0;
            for (int a = 0; a < MPS_CHI; a++)
                for (int b = 0; b < MPS_CHI; b++) {
                    double re, im;
                    mps_read_tensor(j, k, a, b, &re, &im);
                    Ak[a][b] = re;
                    if (re != 0 || im != 0) nonzero = 1;
                }
            if (!nonzero) continue;
            for (int b = 0; b < MPS_CHI; b++)
                for (int a = 0; a < MPS_CHI; a++)
                    new_L[b] += L[a] * Ak[a][b];
        }
        memcpy(L, new_L, sizeof(L));
    }

    /* ── Step 3: Compute P(k) ── */
    double probs[MPS_PHYS];
    double total_prob = 0;

    for (int k = 0; k < MPS_PHYS; k++) {
        double Ak[MPS_CHI][MPS_CHI];
        for (int a = 0; a < MPS_CHI; a++)
            for (int b = 0; b < MPS_CHI; b++) {
                double re, im;
                mps_read_tensor(target_idx, k, a, b, &re, &im);
                Ak[a][b] = re;
            }
        double mid[MPS_CHI] = {0};
        for (int b = 0; b < MPS_CHI; b++)
            for (int a = 0; a < MPS_CHI; a++)
                mid[b] += L[a] * Ak[a][b];
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

    /* ── Step 5: Project + renormalize ── */
    for (int k = 0; k < MPS_PHYS; k++) {
        if ((uint32_t)k != outcome) {
            for (int a = 0; a < MPS_CHI; a++)
                for (int b = 0; b < MPS_CHI; b++)
                    mps_write_tensor(target_idx, k, a, b, 0, 0);
        }
    }
    double slice_norm2 = 0;
    for (int a = 0; a < MPS_CHI; a++)
        for (int b = 0; b < MPS_CHI; b++) {
            double re, im;
            mps_read_tensor(target_idx, (int)outcome, a, b, &re, &im);
            slice_norm2 += re*re + im*im;
        }
    if (slice_norm2 > 1e-30) {
        double scale = 1.0 / sqrt(slice_norm2);
        for (int a = 0; a < MPS_CHI; a++)
            for (int b = 0; b < MPS_CHI; b++) {
                double re, im;
                mps_read_tensor(target_idx, (int)outcome, a, b, &re, &im);
                mps_write_tensor(target_idx, (int)outcome, a, b, re*scale, im*scale);
            }
    }

    return outcome;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SINGLE-SITE GATE
 *
 * A'[k'][α][β] = Σ_k U[k'][k] × A[k][α][β]
 * Cost: O(D² × χ²) = O(36 × 36) = O(1296)
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_gate_1site(QuhitEngine *eng, uint32_t *quhits, int n,
                    int site, const double *U_re, const double *U_im)
{
    (void)eng; (void)quhits; (void)n;

    double old_re[MPS_PHYS][MPS_CHI][MPS_CHI];
    double old_im[MPS_PHYS][MPS_CHI][MPS_CHI];

    for (int k = 0; k < MPS_PHYS; k++)
        for (int a = 0; a < MPS_CHI; a++)
            for (int b = 0; b < MPS_CHI; b++)
                mps_read_tensor(site, k, a, b, &old_re[k][a][b], &old_im[k][a][b]);

    for (int kp = 0; kp < MPS_PHYS; kp++)
        for (int a = 0; a < MPS_CHI; a++)
            for (int b = 0; b < MPS_CHI; b++) {
                double sr = 0, si = 0;
                for (int k = 0; k < MPS_PHYS; k++) {
                    double ur = U_re[kp * MPS_PHYS + k];
                    double ui = U_im[kp * MPS_PHYS + k];
                    sr += ur * old_re[k][a][b] - ui * old_im[k][a][b];
                    si += ur * old_im[k][a][b] + ui * old_re[k][a][b];
                }
                mps_write_tensor(site, kp, a, b, sr, si);
            }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * TWO-SITE GATE WITH SVD
 *
 * With χ=D=6, DCHI=36.  M is 36×36. Jacobi SVD keeps top-χ=6 singular values.
 * Since χ=D, this is EXACT — no truncation loss.
 * ═══════════════════════════════════════════════════════════════════════════════ */

#define DCHI (MPS_PHYS * MPS_CHI) /* 36 */

void mps_gate_2site(QuhitEngine *eng, uint32_t *quhits, int n,
                    int site, const double *G_re, const double *G_im)
{
    (void)eng; (void)quhits; (void)n;
    int si = site, sj = site + 1;

    /* Step 1: Read tensors */
    double Ai_re[MPS_PHYS][MPS_CHI][MPS_CHI], Ai_im[MPS_PHYS][MPS_CHI][MPS_CHI];
    double Aj_re[MPS_PHYS][MPS_CHI][MPS_CHI], Aj_im[MPS_PHYS][MPS_CHI][MPS_CHI];

    for (int k = 0; k < MPS_PHYS; k++)
        for (int a = 0; a < MPS_CHI; a++)
            for (int b = 0; b < MPS_CHI; b++) {
                mps_read_tensor(si, k, a, b, &Ai_re[k][a][b], &Ai_im[k][a][b]);
                mps_read_tensor(sj, k, a, b, &Aj_re[k][a][b], &Aj_im[k][a][b]);
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

    /* Step 3: Apply gate */
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

    /* Step 4: Reshape to M[DCHI][DCHI] (36×36) */
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

    /* Step 5: SVD via Jacobi eigendecomposition of H = M^T M (36×36)
     * Keep top χ=6 singular values → EXACT since rank ≤ min(DCHI, DCHI) = 36 ≥ χ.
     */

    /* Heap-allocate H and V (36×36 each = ~10KB) to avoid stack overflow */
    double (*H)[DCHI] = (double (*)[DCHI])calloc(DCHI, sizeof(double[DCHI]));
    double (*V)[DCHI] = (double (*)[DCHI])calloc(DCHI, sizeof(double[DCHI]));

    /* H = M^T M + M_im^T M_im */
    for (int i = 0; i < DCHI; i++)
        for (int j = 0; j < DCHI; j++) {
            double s = 0;
            for (int r = 0; r < DCHI; r++)
                s += M_re[r][i] * M_re[r][j] + M_im[r][i] * M_im[r][j];
            H[i][j] = s;
        }

    /* V = I */
    for (int i = 0; i < DCHI; i++) V[i][i] = 1.0;

    /* Jacobi sweeps */
    for (int sweep = 0; sweep < 200; sweep++) {
        double off = 0;
        for (int i = 0; i < DCHI; i++)
            for (int j = i + 1; j < DCHI; j++)
                off += H[i][j] * H[i][j];
        if (off < 1e-28) break;

        for (int p = 0; p < DCHI; p++)
            for (int q = p + 1; q < DCHI; q++) {
                if (fabs(H[p][q]) < 1e-15) continue;
                double tau = (H[q][q] - H[p][p]) / (2.0 * H[p][q]);
                double t;
                if (fabs(tau) > 1e15)
                    t = 1.0 / (2.0 * tau);
                else
                    t = (tau >= 0 ? 1.0 : -1.0) / (fabs(tau) + sqrt(1.0 + tau*tau));
                double c = 1.0 / sqrt(1.0 + t*t);
                double s = t * c;

                /* Rotate rows p, q of H */
                double Hp[DCHI], Hq[DCHI];
                for (int i = 0; i < DCHI; i++) {
                    Hp[i] = c*H[p][i] - s*H[q][i];
                    Hq[i] = s*H[p][i] + c*H[q][i];
                }
                for (int i = 0; i < DCHI; i++) { H[p][i] = Hp[i]; H[q][i] = Hq[i]; }
                /* Rotate columns p, q of H */
                for (int i = 0; i < DCHI; i++) {
                    double hip = H[i][p], hiq = H[i][q];
                    H[i][p] = c*hip - s*hiq;
                    H[i][q] = s*hip + c*hiq;
                }
                /* Accumulate in V */
                for (int i = 0; i < DCHI; i++) {
                    double vip = V[i][p], viq = V[i][q];
                    V[i][p] = c*vip - s*viq;
                    V[i][q] = s*vip + c*viq;
                }
            }
    }

    /* Find top-χ eigenvalue indices (selection sort by H[i][i] descending) */
    int top[MPS_CHI];
    {
        int used[DCHI];
        memset(used, 0, sizeof(used));
        for (int t = 0; t < MPS_CHI; t++) {
            int best = -1;
            double best_val = -1e30;
            for (int i = 0; i < DCHI; i++) {
                if (!used[i] && H[i][i] > best_val) {
                    best_val = H[i][i]; best = i;
                }
            }
            top[t] = best;
            if (best >= 0) used[best] = 1;
        }
    }

    /* Extract singular values and right singular vectors */
    double sig[MPS_CHI];
    double v_cols[MPS_CHI][DCHI]; /* v_cols[t][i] = V[i][top[t]] */
    for (int t = 0; t < MPS_CHI; t++) {
        sig[t] = (top[t] >= 0) ? sqrt(fabs(H[top[t]][top[t]])) : 0;
        for (int i = 0; i < DCHI; i++)
            v_cols[t][i] = (top[t] >= 0) ? V[i][top[t]] : 0;
    }

    /* ── RENORMALIZE after truncation ──────────────────────────────
     * The engine's quhit_disentangle preserves norm by renormalizing
     * marginals. We do the same: rescale kept σ's so their total
     * matches the full Frobenius norm (= Σ all eigenvalues of H).
     *
     * Without this, truncated σ's (σ₇..σ₃₆) leak norm away,
     * causing exponential decay in multi-site random circuits.
     * ─────────────────────────────────────────────────────────── */
    {
        double full_norm_sq = 0;
        for (int i = 0; i < DCHI; i++) full_norm_sq += fabs(H[i][i]);

        double kept_norm_sq = 0;
        for (int t = 0; t < MPS_CHI; t++) kept_norm_sq += sig[t] * sig[t];

        if (kept_norm_sq > 1e-30 && full_norm_sq > kept_norm_sq * 1.0000001) {
            double scale = sqrt(full_norm_sq / kept_norm_sq);
            for (int t = 0; t < MPS_CHI; t++) sig[t] *= scale;
        }
    }

    /* U columns: u_t = M × v_t / σ_t */
    double u_re[MPS_CHI][DCHI], u_im[MPS_CHI][DCHI];
    memset(u_re, 0, sizeof(u_re));
    memset(u_im, 0, sizeof(u_im));
    for (int t = 0; t < MPS_CHI; t++) {
        if (sig[t] > 1e-30) {
            for (int i = 0; i < DCHI; i++) {
                double sr = 0, si2 = 0;
                for (int j = 0; j < DCHI; j++) {
                    sr += M_re[i][j] * v_cols[t][j];
                    si2 += M_im[i][j] * v_cols[t][j];
                }
                u_re[t][i] = sr / sig[t];
                u_im[t][i] = si2 / sig[t];
            }
        }
    }

    /* Step 6: Write back */
    /* A'_i[k'][α][β'] = U_col_β'[k'χ+α]  (β' = 0..χ-1) */
    mps_zero_site(si);
    for (int kp = 0; kp < MPS_PHYS; kp++)
        for (int a = 0; a < MPS_CHI; a++) {
            int r = kp * MPS_CHI + a;
            for (int bp = 0; bp < MPS_CHI; bp++)
                mps_write_tensor(si, kp, a, bp, u_re[bp][r], u_im[bp][r]);
        }

    /* A'_j[l'][β'][γ] = σ_{β'} × v_{β'}[l'χ+γ]  (real V) */
    mps_zero_site(sj);
    for (int lp = 0; lp < MPS_PHYS; lp++)
        for (int g = 0; g < MPS_CHI; g++) {
            int c = lp * MPS_CHI + g;
            for (int bp = 0; bp < MPS_CHI; bp++)
                mps_write_tensor(sj, lp, bp, g, sig[bp] * v_cols[bp][c], 0);
        }

    free(H);
    free(V);

    /* ── GLOBAL RENORMALIZATION ────────────────────────────────────
     * After SVD truncation, compute ||ψ||² = Tr(transfer matrix)
     * and rescale site si to restore unitarity.
     * This mirrors the engine's quhit_disentangle → renormalize.
     * Cost: O(n·χ²·D) per gate, standard in DMRG/TEBD.
     * ──────────────────────────────────────────────────────────── */
    {
        /* Compute global norm via transfer matrix contraction */
        double rho_re[MPS_CHI][MPS_CHI] = {{0}};
        double rho_im[MPS_CHI][MPS_CHI] = {{0}};
        rho_re[0][0] = 1.0;

        for (int site2 = 0; site2 < n; site2++) {
            double nr[MPS_CHI][MPS_CHI] = {{0}};
            double ni2[MPS_CHI][MPS_CHI] = {{0}};
            for (int k = 0; k < MPS_PHYS; k++) {
                double Ak_re[MPS_CHI][MPS_CHI], Ak_im[MPS_CHI][MPS_CHI];
                for (int a = 0; a < MPS_CHI; a++)
                    for (int b = 0; b < MPS_CHI; b++)
                        mps_read_tensor(site2, k, a, b, &Ak_re[a][b], &Ak_im[a][b]);
                double tr2[MPS_CHI][MPS_CHI] = {{0}};
                double ti2[MPS_CHI][MPS_CHI] = {{0}};
                for (int a = 0; a < MPS_CHI; a++)
                    for (int bp = 0; bp < MPS_CHI; bp++)
                        for (int ap = 0; ap < MPS_CHI; ap++) {
                            tr2[a][bp] += rho_re[a][ap]*Ak_re[ap][bp] - rho_im[a][ap]*Ak_im[ap][bp];
                            ti2[a][bp] += rho_re[a][ap]*Ak_im[ap][bp] + rho_im[a][ap]*Ak_re[ap][bp];
                        }
                for (int b = 0; b < MPS_CHI; b++)
                    for (int bp = 0; bp < MPS_CHI; bp++)
                        for (int a = 0; a < MPS_CHI; a++) {
                            double ar2 = Ak_re[a][b], ai2 = -Ak_im[a][b];
                            nr[b][bp] += ar2*tr2[a][bp] - ai2*ti2[a][bp];
                            ni2[b][bp] += ar2*ti2[a][bp] + ai2*tr2[a][bp];
                        }
            }
            memcpy(rho_re, nr, sizeof(rho_re));
            memcpy(rho_im, ni2, sizeof(rho_im));
        }

        double norm = 0;
        for (int i = 0; i < MPS_CHI; i++) norm += rho_re[i][i];

        /* Rescale site si if norm significantly differs from 1 */
        if (norm > 1e-30 && fabs(norm - 1.0) > 1e-12) {
            double scale = 1.0 / sqrt(norm);
            for (int k = 0; k < MPS_PHYS; k++)
                for (int a = 0; a < MPS_CHI; a++)
                    for (int b = 0; b < MPS_CHI; b++) {
                        double tr, ti;
                        mps_read_tensor(si, k, a, b, &tr, &ti);
                        mps_write_tensor(si, k, a, b, tr * scale, ti * scale);
                    }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * GATE CONSTRUCTORS
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_build_dft6(double *U_re, double *U_im)
{
    double inv = 1.0 / sqrt(6.0);
    for (int j = 0; j < 6; j++)
        for (int k = 0; k < 6; k++) {
            double angle = 2.0 * M_PI * j * k / 6.0;
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
            double angle = 2.0 * M_PI * k * l / 6.0;
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
    double s = 1.0 / sqrt(2.0);
    U_re[0*6+0] =  s; U_re[0*6+1] =  s;
    U_re[1*6+0] =  s; U_re[1*6+1] = -s;
    for (int k = 2; k < 6; k++) U_re[k*6+k] = 1.0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * NORM  ⟨ψ|ψ⟩
 *
 * Transfer: ρ' = Σ_k A†[k] ρ A[k]
 * Boundary: ρ_L = |e_0⟩⟨e_0|, project onto R = e_{χ-1}
 * ═══════════════════════════════════════════════════════════════════════════════ */

double mps_overlay_norm(QuhitEngine *eng, uint32_t *quhits, int n)
{
    (void)eng; (void)quhits;

    double rho_re[MPS_CHI][MPS_CHI] = {{0}};
    double rho_im[MPS_CHI][MPS_CHI] = {{0}};
    rho_re[0][0] = 1.0; /* |L⟩⟨L| = |e_0⟩⟨e_0| */

    for (int i = 0; i < n; i++) {
        double nr[MPS_CHI][MPS_CHI] = {{0}};
        double ni_arr[MPS_CHI][MPS_CHI] = {{0}};

        for (int k = 0; k < MPS_PHYS; k++) {
            double A_re[MPS_CHI][MPS_CHI], A_im[MPS_CHI][MPS_CHI];
            for (int a = 0; a < MPS_CHI; a++)
                for (int b = 0; b < MPS_CHI; b++)
                    mps_read_tensor(i, k, a, b, &A_re[a][b], &A_im[a][b]);

            /* tmp = ρ × A[k] */
            double tr2[MPS_CHI][MPS_CHI] = {{0}};
            double ti2[MPS_CHI][MPS_CHI] = {{0}};
            for (int a = 0; a < MPS_CHI; a++)
                for (int bp = 0; bp < MPS_CHI; bp++)
                    for (int ap = 0; ap < MPS_CHI; ap++) {
                        tr2[a][bp] += rho_re[a][ap]*A_re[ap][bp] - rho_im[a][ap]*A_im[ap][bp];
                        ti2[a][bp] += rho_re[a][ap]*A_im[ap][bp] + rho_im[a][ap]*A_re[ap][bp];
                    }

            /* nr += A†[k] × tmp */
            for (int b = 0; b < MPS_CHI; b++)
                for (int bp = 0; bp < MPS_CHI; bp++)
                    for (int a = 0; a < MPS_CHI; a++) {
                        double ar = A_re[a][b], ai = -A_im[a][b];
                        nr[b][bp]     += ar*tr2[a][bp] - ai*ti2[a][bp];
                        ni_arr[b][bp] += ar*ti2[a][bp] + ai*tr2[a][bp];
                    }
        }
        memcpy(rho_re, nr, sizeof(rho_re));
        memcpy(rho_im, ni_arr, sizeof(rho_im));
    }

    /* Full trace: Tr(ρ) sums all boundary channels */
    double trace = 0;
    for (int i = 0; i < MPS_CHI; i++) trace += rho_re[i][i];
    return trace;
}
