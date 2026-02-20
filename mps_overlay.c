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
int        mps_defer_renorm = 0;
int        mps_sweep_right  = 1;  /* 1 = L→R (default), 0 = R→L */

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

    /* Heap-allocate: at χ=128, each is 6×128×128 × 8 = 768 KB */
    size_t tsz = (size_t)MPS_PHYS * MPS_CHI * MPS_CHI;
    double *old_re = (double *)malloc(tsz * sizeof(double));
    double *old_im = (double *)malloc(tsz * sizeof(double));

    for (int k = 0; k < MPS_PHYS; k++)
        for (int a = 0; a < MPS_CHI; a++)
            for (int b = 0; b < MPS_CHI; b++) {
                int idx = k * MPS_CHI * MPS_CHI + a * MPS_CHI + b;
                mps_read_tensor(site, k, a, b, &old_re[idx], &old_im[idx]);
            }

    for (int kp = 0; kp < MPS_PHYS; kp++)
        for (int a = 0; a < MPS_CHI; a++)
            for (int b = 0; b < MPS_CHI; b++) {
                double sr = 0, si = 0;
                for (int k = 0; k < MPS_PHYS; k++) {
                    double ur = U_re[kp * MPS_PHYS + k];
                    double ui = U_im[kp * MPS_PHYS + k];
                    int idx = k * MPS_CHI * MPS_CHI + a * MPS_CHI + b;
                    sr += ur * old_re[idx] - ui * old_im[idx];
                    si += ur * old_im[idx] + ui * old_re[idx];
                }
                mps_write_tensor(site, kp, a, b, sr, si);
            }

    free(old_re);
    free(old_im);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * TWO-SITE GATE WITH SVD
 *
 * With χ=128, D=6, DCHI=768. M is 768×768. Jacobi SVD keeps top-χ=128
 * singular values. All large arrays are heap-allocated.
 * ═══════════════════════════════════════════════════════════════════════════════ */

#define DCHI (MPS_PHYS * MPS_CHI) /* D × χ */

/* Helper macros for flat 4D array: Th[k][l][a][g] */
#define TH_IDX(k,l,a,g) ((k)*MPS_PHYS*MPS_CHI*MPS_CHI + (l)*MPS_CHI*MPS_CHI + (a)*MPS_CHI + (g))
#define AI_IDX(k,a,b)   ((k)*MPS_CHI*MPS_CHI + (a)*MPS_CHI + (b))

void mps_gate_2site(QuhitEngine *eng, uint32_t *quhits, int n,
                    int site, const double *G_re, const double *G_im)
{
    (void)eng; (void)quhits; (void)n;
    int si = site, sj = site + 1;

    size_t ai_sz = (size_t)MPS_PHYS * MPS_CHI * MPS_CHI;
    size_t th_sz = (size_t)MPS_PHYS * MPS_PHYS * MPS_CHI * MPS_CHI;
    size_t m_sz  = (size_t)DCHI * DCHI;

    /* Step 1: Read tensors (heap) */
    double *Ai_re = (double *)malloc(ai_sz * sizeof(double));
    double *Ai_im = (double *)malloc(ai_sz * sizeof(double));
    double *Aj_re = (double *)malloc(ai_sz * sizeof(double));
    double *Aj_im = (double *)malloc(ai_sz * sizeof(double));

    for (int k = 0; k < MPS_PHYS; k++)
        for (int a = 0; a < MPS_CHI; a++)
            for (int b = 0; b < MPS_CHI; b++) {
                mps_read_tensor(si, k, a, b,
                    &Ai_re[AI_IDX(k,a,b)], &Ai_im[AI_IDX(k,a,b)]);
                mps_read_tensor(sj, k, a, b,
                    &Aj_re[AI_IDX(k,a,b)], &Aj_im[AI_IDX(k,a,b)]);
            }

    /* Step 2: Contract Θ[k,l,α,γ] = Σ_β Ai[k][α][β] × Aj[l][β][γ] */
    double *Th_re = (double *)calloc(th_sz, sizeof(double));
    double *Th_im = (double *)calloc(th_sz, sizeof(double));

    for (int k = 0; k < MPS_PHYS; k++)
        for (int l = 0; l < MPS_PHYS; l++)
            for (int a = 0; a < MPS_CHI; a++)
                for (int g = 0; g < MPS_CHI; g++)
                    for (int b = 0; b < MPS_CHI; b++) {
                        double ar = Ai_re[AI_IDX(k,a,b)];
                        double ai = Ai_im[AI_IDX(k,a,b)];
                        double br = Aj_re[AI_IDX(l,b,g)];
                        double bi = Aj_im[AI_IDX(l,b,g)];
                        Th_re[TH_IDX(k,l,a,g)] += ar*br - ai*bi;
                        Th_im[TH_IDX(k,l,a,g)] += ar*bi + ai*br;
                    }

    /* Free Ai, Aj — no longer needed */
    free(Ai_re); free(Ai_im);
    free(Aj_re); free(Aj_im);

    /* Step 3: Apply gate */
    double *Tp_re = (double *)calloc(th_sz, sizeof(double));
    double *Tp_im = (double *)calloc(th_sz, sizeof(double));

    int D2 = MPS_PHYS * MPS_PHYS;
    for (int kp = 0; kp < MPS_PHYS; kp++)
        for (int lp = 0; lp < MPS_PHYS; lp++) {
            int row = kp * MPS_PHYS + lp;
            for (int k = 0; k < MPS_PHYS; k++)
                for (int l = 0; l < MPS_PHYS; l++) {
                    int col = k * MPS_PHYS + l;
                    double gr = G_re[row * D2 + col];
                    double gi = G_im[row * D2 + col];
                    if (fabs(gr) < 1e-30 && fabs(gi) < 1e-30) continue;
                    for (int a = 0; a < MPS_CHI; a++)
                        for (int g = 0; g < MPS_CHI; g++) {
                            double tr = Th_re[TH_IDX(k,l,a,g)];
                            double ti = Th_im[TH_IDX(k,l,a,g)];
                            Tp_re[TH_IDX(kp,lp,a,g)] += gr*tr - gi*ti;
                            Tp_im[TH_IDX(kp,lp,a,g)] += gr*ti + gi*tr;
                        }
                }
        }

    free(Th_re); free(Th_im);

    /* Step 4: Reshape to M[DCHI][DCHI] */
    double *M_re = (double *)malloc(m_sz * sizeof(double));
    double *M_im = (double *)malloc(m_sz * sizeof(double));

    for (int kp = 0; kp < MPS_PHYS; kp++)
        for (int a = 0; a < MPS_CHI; a++) {
            int r = kp * MPS_CHI + a;
            for (int lp = 0; lp < MPS_PHYS; lp++)
                for (int g = 0; g < MPS_CHI; g++) {
                    int c = lp * MPS_CHI + g;
                    M_re[r * DCHI + c] = Tp_re[TH_IDX(kp,lp,a,g)];
                    M_im[r * DCHI + c] = Tp_im[TH_IDX(kp,lp,a,g)];
                }
        }

    free(Tp_re); free(Tp_im);

    /* Step 5: SVD via eigendecomposition of M†M (DCHI×DCHI complex Hermitian)
     *
     * M†M is complex Hermitian, NOT real symmetric.
     *   (M†M)_re[j][k] = Σ_r M_re[r][j]*M_re[r][k] + M_im[r][j]*M_im[r][k]
     *   (M†M)_im[j][k] = Σ_r M_re[r][j]*M_im[r][k] - M_im[r][j]*M_re[r][k]
     *
     * We diagonalize the FULL Hermitian matrix with complex Jacobi rotations:
     *   1. Phase-rotate column q by e^{-iφ} to make H[p][q] real
     *   2. Apply real Givens rotation to zero the (now-real) off-diagonal
     *   3. Accumulate complex eigenvectors in V = V_re + i·V_im
     */

    /* Heap-allocate SVD work arrays */
    double *Hr = (double *)calloc(m_sz, sizeof(double));
    double *Hi_a = (double *)calloc(m_sz, sizeof(double));
    double *Vr = (double *)calloc(m_sz, sizeof(double));
    double *Vi_a = (double *)calloc(m_sz, sizeof(double));

    /* H = M†M (complex Hermitian) */
    for (int i = 0; i < DCHI; i++)
        for (int j = 0; j < DCHI; j++) {
            double sr = 0, si2 = 0;
            for (int r = 0; r < DCHI; r++) {
                sr  += M_re[r*DCHI+i]*M_re[r*DCHI+j] + M_im[r*DCHI+i]*M_im[r*DCHI+j];
                si2 += M_re[r*DCHI+i]*M_im[r*DCHI+j] - M_im[r*DCHI+i]*M_re[r*DCHI+j];
            }
            Hr[i*DCHI+j] = sr;
            Hi_a[i*DCHI+j] = si2;
        }

    /* V = I (complex) */
    for (int i = 0; i < DCHI; i++) Vr[i*DCHI+i] = 1.0;

    /* Complex Hermitian Jacobi sweeps */
    for (int sweep = 0; sweep < 200; sweep++) {
        double off = 0;
        for (int i = 0; i < DCHI; i++)
            for (int j = i + 1; j < DCHI; j++)
                off += Hr[i*DCHI+j]*Hr[i*DCHI+j] + Hi_a[i*DCHI+j]*Hi_a[i*DCHI+j];
        if (off < 1e-28) break;

        for (int p = 0; p < DCHI; p++)
            for (int q = p + 1; q < DCHI; q++) {
                double hpq_r = Hr[p*DCHI+q], hpq_i = Hi_a[p*DCHI+q];
                double mag = sqrt(hpq_r*hpq_r + hpq_i*hpq_i);
                if (mag < 1e-15) continue;

                double eR = hpq_r / mag, eI = -hpq_i / mag;

                for (int i = 0; i < DCHI; i++) {
                    double xr = Hr[i*DCHI+q], xi = Hi_a[i*DCHI+q];
                    Hr[i*DCHI+q] = xr*eR - xi*eI;
                    Hi_a[i*DCHI+q] = xr*eI + xi*eR;
                }
                for (int j = 0; j < DCHI; j++) {
                    double xr = Hr[q*DCHI+j], xi = Hi_a[q*DCHI+j];
                    Hr[q*DCHI+j] =  xr*eR + xi*eI;
                    Hi_a[q*DCHI+j] = -xr*eI + xi*eR;
                }
                for (int i = 0; i < DCHI; i++) {
                    double xr = Vr[i*DCHI+q], xi = Vi_a[i*DCHI+q];
                    Vr[i*DCHI+q] = xr*eR - xi*eI;
                    Vi_a[i*DCHI+q] = xr*eI + xi*eR;
                }

                double hpp = Hr[p*DCHI+p], hqq = Hr[q*DCHI+q];
                double hpq_real = Hr[p*DCHI+q];

                double tau = (hqq - hpp) / (2.0 * hpq_real);
                double t;
                if (fabs(tau) > 1e15)
                    t = 1.0 / (2.0 * tau);
                else
                    t = (tau >= 0 ? 1.0 : -1.0) /
                        (fabs(tau) + sqrt(1.0 + tau*tau));
                double c = 1.0 / sqrt(1.0 + t*t);
                double s = t * c;

                for (int j = 0; j < DCHI; j++) {
                    double rp = Hr[p*DCHI+j], ip = Hi_a[p*DCHI+j];
                    double rq = Hr[q*DCHI+j], iq = Hi_a[q*DCHI+j];
                    Hr[p*DCHI+j] = c*rp - s*rq;  Hi_a[p*DCHI+j] = c*ip - s*iq;
                    Hr[q*DCHI+j] = s*rp + c*rq;  Hi_a[q*DCHI+j] = s*ip + c*iq;
                }
                for (int i = 0; i < DCHI; i++) {
                    double rp = Hr[i*DCHI+p], ip = Hi_a[i*DCHI+p];
                    double rq = Hr[i*DCHI+q], iq = Hi_a[i*DCHI+q];
                    Hr[i*DCHI+p] = c*rp - s*rq;  Hi_a[i*DCHI+p] = c*ip - s*iq;
                    Hr[i*DCHI+q] = s*rp + c*rq;  Hi_a[i*DCHI+q] = s*ip + c*iq;
                }
                for (int i = 0; i < DCHI; i++) {
                    double rp = Vr[i*DCHI+p], ip = Vi_a[i*DCHI+p];
                    double rq = Vr[i*DCHI+q], iq = Vi_a[i*DCHI+q];
                    Vr[i*DCHI+p] = c*rp - s*rq;  Vi_a[i*DCHI+p] = c*ip - s*iq;
                    Vr[i*DCHI+q] = s*rp + c*rq;  Vi_a[i*DCHI+q] = s*ip + c*iq;
                }
            }
    }

    /* Find top-χ eigenvalue indices (selection sort by Hr[i][i] descending) */
    int *top = (int *)malloc(MPS_CHI * sizeof(int));
    {
        int *used = (int *)calloc(DCHI, sizeof(int));
        for (int t = 0; t < MPS_CHI; t++) {
            int best = -1;
            double best_val = -1e30;
            for (int i = 0; i < DCHI; i++) {
                if (!used[i] && Hr[i*DCHI+i] > best_val) {
                    best_val = Hr[i*DCHI+i]; best = i;
                }
            }
            top[t] = best;
            if (best >= 0) used[best] = 1;
        }
        free(used);
    }

    /* Extract singular values and complex right singular vectors */
    double *sig = (double *)malloc(MPS_CHI * sizeof(double));
    size_t vc_sz = (size_t)MPS_CHI * DCHI;
    double *vc_re = (double *)calloc(vc_sz, sizeof(double));
    double *vc_im = (double *)calloc(vc_sz, sizeof(double));

    for (int t = 0; t < MPS_CHI; t++) {
        sig[t] = (top[t] >= 0) ? sqrt(fabs(Hr[top[t]*DCHI+top[t]])) : 0;
        for (int i = 0; i < DCHI; i++) {
            vc_re[t*DCHI+i] = (top[t] >= 0) ? Vr[i*DCHI+top[t]] : 0;
            vc_im[t*DCHI+i] = (top[t] >= 0) ? Vi_a[i*DCHI+top[t]] : 0;
        }
    }

    free(Hr); free(Hi_a);
    free(Vr); free(Vi_a);
    free(top);

    /* U columns: u_t = M × v_t / σ_t  (complex M × complex v) */
    double *u_re = (double *)calloc(vc_sz, sizeof(double));
    double *u_im = (double *)calloc(vc_sz, sizeof(double));
    for (int t = 0; t < MPS_CHI; t++) {
        if (sig[t] > 1e-30) {
            for (int i = 0; i < DCHI; i++) {
                double sr = 0, si2 = 0;
                for (int j = 0; j < DCHI; j++) {
                    sr  += M_re[i*DCHI+j]*vc_re[t*DCHI+j] - M_im[i*DCHI+j]*vc_im[t*DCHI+j];
                    si2 += M_re[i*DCHI+j]*vc_im[t*DCHI+j] + M_im[i*DCHI+j]*vc_re[t*DCHI+j];
                }
                u_re[t*DCHI+i] = sr / sig[t];
                u_im[t*DCHI+i] = si2 / sig[t];
            }
        }
    }

    free(M_re); free(M_im);

    /* ── LOCAL O(1) RENORMALIZATION ──────────────────────────────
     * In mixed-canonical form, kept_norm_sq = Σ σ_t² IS the exact
     * global norm ||ψ||². We rescale σ to restore ||ψ|| = 1.0
     * without touching the rest of the chain.  Cost: O(χ) = O(1).
     * U columns were computed above using original σ so they remain
     * orthonormal (left-canonical).
     * ─────────────────────────────────────────────────────────── */
    double kept_norm_sq = 0;
    for (int t = 0; t < MPS_CHI; t++) kept_norm_sq += sig[t] * sig[t];

    if (kept_norm_sq > 1e-30) {
        double scale = 1.0 / sqrt(kept_norm_sq);
        for (int t = 0; t < MPS_CHI; t++) sig[t] *= scale;
    }

    /* Step 6: Write back — direction depends on sweep
     *
     * L→R (mps_sweep_right=1):
     *   Left site  = U             (left-canonical)
     *   Right site = σ·V           (gauge center, moves right)
     *
     * R→L (mps_sweep_right=0):
     *   Left site  = U·σ           (gauge center, moves left)
     *   Right site = V             (right-canonical)
     *
     * V is now complex — both re and im parts stored.
     */
    mps_zero_site(si);
    mps_zero_site(sj);

    if (mps_sweep_right) {
        for (int kp = 0; kp < MPS_PHYS; kp++)
            for (int a = 0; a < MPS_CHI; a++) {
                int r = kp * MPS_CHI + a;
                for (int bp = 0; bp < MPS_CHI; bp++)
                    mps_write_tensor(si, kp, a, bp,
                                     u_re[bp*DCHI+r], u_im[bp*DCHI+r]);
            }
        for (int lp = 0; lp < MPS_PHYS; lp++)
            for (int g = 0; g < MPS_CHI; g++) {
                int cc = lp * MPS_CHI + g;
                for (int bp = 0; bp < MPS_CHI; bp++)
                    mps_write_tensor(sj, lp, bp, g,
                                     sig[bp]*vc_re[bp*DCHI+cc],
                                     sig[bp]*vc_im[bp*DCHI+cc]);
            }
    } else {
        for (int kp = 0; kp < MPS_PHYS; kp++)
            for (int a = 0; a < MPS_CHI; a++) {
                int r = kp * MPS_CHI + a;
                for (int bp = 0; bp < MPS_CHI; bp++)
                    mps_write_tensor(si, kp, a, bp,
                                     u_re[bp*DCHI+r] * sig[bp],
                                     u_im[bp*DCHI+r] * sig[bp]);
            }
        for (int lp = 0; lp < MPS_PHYS; lp++)
            for (int g = 0; g < MPS_CHI; g++) {
                int cc = lp * MPS_CHI + g;
                for (int bp = 0; bp < MPS_CHI; bp++)
                    mps_write_tensor(sj, lp, bp, g,
                                     vc_re[bp*DCHI+cc], vc_im[bp*DCHI+cc]);
            }
    }

    free(sig);
    free(vc_re); free(vc_im);
    free(u_re); free(u_im);
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

/* ═══════════════════════════════════════════════════════════════════════════════
 * DEFERRED RENORMALIZATION
 *
 * Call after a batch of 2-site gates (e.g. one full CNOT layer).
 * Computes global norm O(n) and rescales site 0 to restore ||ψ||=1.
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_renormalize_chain(QuhitEngine *eng, uint32_t *quhits, int n)
{
    double norm = mps_overlay_norm(eng, quhits, n);
    if (norm > 1e-30 && fabs(norm - 1.0) > 1e-12) {
        double scale = 1.0 / sqrt(norm);
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
 * LAZY EVALUATION LAYER IMPLEMENTATION
 *
 * "Reality computes on demand."
 *
 * The core idea: gates are appended to a queue. Nothing happens until
 * you measure. Measurement identifies which gates affect the target
 * site's causal cone and materializes ONLY those. Gates outside the
 * cone are marked as skipped and never applied.
 *
 * Gate fusion: if two consecutive queue entries are both 1-site gates
 * on the same site, they are multiplied into a single 6×6 matrix.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* ── Lifecycle ── */

MpsLazyChain *mps_lazy_init(QuhitEngine *eng, uint32_t *quhits, int n)
{
    MpsLazyChain *lc = (MpsLazyChain *)calloc(1, sizeof(MpsLazyChain));
    lc->eng = eng;
    lc->quhits = quhits;
    lc->n_sites = n;

    /* Initialize underlying MPS store */
    mps_overlay_init(eng, quhits, n);

    /* Gate queue */
    lc->queue_cap = MAX_LAZY_GATES;
    lc->queue = (MpsDeferredGate *)calloc(lc->queue_cap, sizeof(MpsDeferredGate));
    lc->queue_len = 0;

    /* Per-site flags */
    lc->site_allocated = (uint8_t *)calloc(n, sizeof(uint8_t));
    lc->site_dirty     = (uint8_t *)calloc(n, sizeof(uint8_t));

    /* Stats */
    lazy_stats_reset(&lc->stats);
    lc->stats.sites_total = n;
    lc->stats.hilbert_log10 = n * log10(6.0);

    return lc;
}

void mps_lazy_free(MpsLazyChain *lc)
{
    if (!lc) return;

    /* Free any heap-allocated 2-site gate matrices */
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

/* ── Ensure site has an allocated tensor ── */
static void lazy_ensure_site(MpsLazyChain *lc, int site)
{
    if (!lc->site_allocated[site]) {
        /* Implicit |0⟩: write identity-like tensor for |0⟩ product state */
        mps_zero_site(site);
        mps_write_tensor(site, 0, 0, 0, 1.0, 0.0);
        lc->site_allocated[site] = 1;
    }
}

/* ── Gate queuing ── */

void mps_lazy_gate_1site(MpsLazyChain *lc, int site,
                         const double *U_re, const double *U_im)
{
    if (lc->queue_len >= lc->queue_cap) {
        /* Queue full — flush everything and reset */
        mps_lazy_flush(lc);
    }

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
    if (lc->queue_len >= lc->queue_cap) {
        mps_lazy_flush(lc);
    }

    int D2 = MPS_PHYS * MPS_PHYS; /* 36 */
    int sz = D2 * D2;              /* 1296 */

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

/* ── Gate fusion: multiply two 6×6 complex matrices ── */
static void fuse_1site_gates(const double *A_re, const double *A_im,
                             const double *B_re, const double *B_im,
                             double *C_re, double *C_im)
{
    /* C = B × A (B applied after A) */
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

/* ── Determine causal cone: which sites are needed for measuring target ── */
static void compute_causal_cone(MpsLazyChain *lc, int target,
                                uint8_t *needed)
{
    /* A gate on site s affects the measurement of target t if:
     *   - 1-site gate on site s, and s is in [0, n-1] (always needed
     *     because left/right environments depend on all sites)
     *   - 2-site gate on site s, affecting sites s and s+1
     *
     * For MPS measurement of target t:
     *   - Left environment: sites 0..t-1
     *   - Right environment: sites t+1..n-1
     *   - Target: site t
     * ALL sites contribute to the measurement probability.
     * Therefore, ALL pending gates must be materialized.
     *
     * BUT: we can still skip gates whose sites have been fully
     * measured/collapsed already. For the first measurement, all
     * gates are needed. For subsequent measurements, only gates
     * on un-collapsed sites matter.
     *
     * For now: mark all sites from 0 to n-1 as needed.
     * This is correct. The optimization for partial measurement
     * comes from not measuring all sites.
     */
    for (int i = 0; i < lc->n_sites; i++)
        needed[i] = 1;
}

/* ── Apply a single gate from the queue ── */
static void apply_gate(MpsLazyChain *lc, MpsDeferredGate *g)
{
    if (g->applied) return;

    if (g->type == 0) {
        /* 1-site gate */
        lazy_ensure_site(lc, g->site);
        mps_gate_1site(lc->eng, lc->quhits, lc->n_sites,
                       g->site, g->U_re, g->U_im);
    } else {
        /* 2-site gate */
        lazy_ensure_site(lc, g->site);
        lazy_ensure_site(lc, g->site + 1);
        mps_gate_2site(lc->eng, lc->quhits, lc->n_sites,
                       g->site, g->G_re, g->G_im);
    }

    g->applied = 1;
    lc->stats.gates_materialized++;
}

/* ── Measurement-driven materialization ── */

uint32_t mps_lazy_measure(MpsLazyChain *lc, int target_idx)
{
    /* Step 1: Compute causal cone */
    uint8_t *needed = (uint8_t *)calloc(lc->n_sites, sizeof(uint8_t));
    compute_causal_cone(lc, target_idx, needed);

    /* Step 2: Gate fusion pass — fuse consecutive 1-site gates on same site */
    for (int i = 0; i < lc->queue_len - 1; i++) {
        if (lc->queue[i].applied) continue;
        if (lc->queue[i].type != 0) continue; /* only fuse 1-site */

        int j = i + 1;
        while (j < lc->queue_len &&
               lc->queue[j].type == 0 &&
               lc->queue[j].site == lc->queue[i].site &&
               !lc->queue[j].applied) {
            /* Fuse gate j into gate i: C = B × A */
            double C_re[MPS_PHYS * MPS_PHYS], C_im[MPS_PHYS * MPS_PHYS];
            fuse_1site_gates(lc->queue[i].U_re, lc->queue[i].U_im,
                             lc->queue[j].U_re, lc->queue[j].U_im,
                             C_re, C_im);
            memcpy(lc->queue[i].U_re, C_re, sizeof(C_re));
            memcpy(lc->queue[i].U_im, C_im, sizeof(C_im));
            lc->queue[j].applied = 1; /* consumed by fusion */
            lc->stats.gates_fused++;
            j++;
        }
    }

    /* Step 3: Apply all un-applied gates whose sites are in the causal cone */
    for (int i = 0; i < lc->queue_len; i++) {
        if (lc->queue[i].applied) continue;

        int s = lc->queue[i].site;
        int s2 = (lc->queue[i].type == 1) ? s + 1 : s;

        if (needed[s] || needed[s2]) {
            apply_gate(lc, &lc->queue[i]);
        }
        /* Gates outside the cone stay un-applied (will be counted as skipped) */
    }

    free(needed);

    /* Step 4: Ensure target site is allocated */
    lazy_ensure_site(lc, target_idx);

    /* Step 5: Delegate to existing measurement */
    return mps_overlay_measure(lc->eng, lc->quhits, lc->n_sites, target_idx);
}

/* ── Flush: force-apply all pending gates ── */

void mps_lazy_flush(MpsLazyChain *lc)
{
    /* Fusion pass first */
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

    /* Apply all remaining */
    for (int i = 0; i < lc->queue_len; i++) {
        if (!lc->queue[i].applied)
            apply_gate(lc, &lc->queue[i]);
    }

    /* Free 2-site heap data + reset queue */
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

/* ── Finalize stats ── */

void mps_lazy_finalize_stats(MpsLazyChain *lc)
{
    /* Count skipped gates */
    uint64_t skipped = 0;
    for (int i = 0; i < lc->queue_len; i++)
        if (!lc->queue[i].applied) skipped++;
    lc->stats.gates_skipped = skipped;

    /* Count allocated vs lazy sites */
    uint64_t alloc = 0;
    for (int i = 0; i < lc->n_sites; i++)
        if (lc->site_allocated[i]) alloc++;
    lc->stats.sites_allocated = alloc;
    lc->stats.sites_lazy = lc->n_sites - alloc;

    /* Memory */
    lc->stats.memory_actual = alloc * sizeof(MpsTensor)
                            + lc->queue_len * sizeof(MpsDeferredGate)
                            + sizeof(MpsLazyChain);
}

/* ── Lazy tensor write (marks site allocated) ── */

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
    mps_write_tensor(site, 0, 0, 0, 1.0, 0.0);  /* |0⟩ product state */
}

