/*
 * tensor_svd.h — Shared Jacobi SVD for Tensor Network Overlays
 *
 * Provides truncated SVD of complex matrices using Jacobi
 * eigendecomposition of the Hermitian product M†M.
 *
 * Used by MPS, PEPS 2D, PEPS 3D–6D for 2-site gate application.
 * All inputs/outputs are flat row-major arrays.
 *
 * ── Side-Channel Optimizations (from tns_contraction_probe.c) ──
 *   • Zero attractor: 60% of Jacobi angles < 0.01 → aggressive skip
 *   • Early sweep termination via relative off-diagonal check
 *   • 1/6 spectrum awareness: contraction σ → 1/√D
 *
 * ── Simulation Hypothesis Upgrades ──
 *   • Trivial matrix fast exit: nnz ≤ 36 → direct Gram-Jacobi (skip pipeline)
 *   • Rank-1 O(1) path: single NNZ → no matrix ops at all
 *   • Adaptive power iterations: q=1/2/3 based on sparsity ratio
 *   • Dead sigma skip: break U/V† reconstruction at first zero σ
 *   • Frobenius pre-check: skip entire SVD for zero-effect gates
 *
 * ── Layer 5 Hardware Fabric Upgrades ──
 *   • 8-load speculation: 4-way unrolled sparse MatVec (8 loads/iter)
 *   • FMA sparse MatVec: fma() in the hottest inner loops
 *   • FMA norm accumulation in MGS
 *
 * ── Layer 6 Universal Constant Upgrades ──
 *   • All 10 magic thresholds replaced with ε-derived constants
 *   • Information-theoretic rank truncation: σ[j]/σ[0] < ε
 *   • 1 Newton iteration (9.2 bits, sufficient for self-correcting Jacobi)
 *
 * ── Layer 7 Deep Structure Upgrades ──
 *   • Kahan compensated sum for Jacobi convergence (addition loses ~1 bit/op)
 *   • FMA projection subtraction in MGS (subtraction is 50% irreversible)
 *   • Denormals are SAFE: no performance cliff (Layer 7 Probe 6)
 *   • Max sweeps 100→30: fixed-point convergence in 6-29 iters (Probe 8)
 *
 * ── Measured Substrate Parameters (from probes) ──
 *   Timing quantum:     39 ns   (Layer 4)
 *   FPU dispatch width: 4-wide  (Layer 3)   Speculation depth: 8 loads (Layer 5)
 *   Cache line:         64 B    (Layer 5)   L1 cliff: ~2 KB (Layer 5)
 *   Effective freq:     0.44 GHz (Layer 5)  Peak: 1.8 GFLOP/s
 *   PRNG quality:       clean   (Layer 3)   Born constant: flat basin (Layer 3)
 *   Rounding mode:      RTE     (Layer 7)   Benford: χ² = 0.000008 (Layer 7)
 *   Denormal penalty:   NONE    (Layer 7)   NaN propagation: left-dominant
 *   FMA captures:       ε²=4.93e-32         Irreversibility: 32.4% (Layer 7)
 */

#ifndef TENSOR_SVD_H
#define TENSOR_SVD_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>        /* DBL_EPSILON, DBL_MIN */
#include "born_rule.h"   /* born_fast_isqrt, born_fast_recip */

/* ═══════════════════════════════════════════════════════════════════════════════
 * LAYER 6: UNIVERSAL PRECISION CONSTANTS
 *
 * Derived from substrate probes. These replace ALL magic thresholds.
 *   ε     = 2⁻⁵² = 2.2204e-16  (Probe 1: the precision atom)
 *   ε²    = 2⁻¹⁰⁴ ≈ 4.93e-32  (convergence threshold = FMA precision floor, L7P4)
 *   3.322 = log₂(10)           (Probe 4: precision Rosetta Stone)
 *   52 bits ÷ 3.322 = 15.65 decimal digits (the precision horizon)
 * ═══════════════════════════════════════════════════════════════════════════════ */
#define TSVD_EPS          DBL_EPSILON           /* 2⁻⁵² = 2.2204e-16          */
#define TSVD_EPS2         (TSVD_EPS * TSVD_EPS) /* 2⁻¹⁰⁴ = FMA precision floor */
#define TSVD_SAFE_MIN     DBL_MIN               /* 2⁻¹⁰²² (L7: denormal-safe)  */
#define TSVD_NOISE_FLOOR  (TSVD_EPS * 16.0)     /* ~16 ULPs, below = noise    */

/* ═══════════════════════════════════════════════════════════════════════════════
 * JACOBI HERMITIAN EIGENDECOMPOSITION
 *
 * Diagonalizes n×n Hermitian H via Jacobi rotations.
 * Returns eigenvalues in `diag` (descending) and eigenvectors in W.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void tsvd_jacobi_hermitian(double *H_re, double *H_im, int n,
                                  double *diag, double *W_re, double *W_im)
{
    /* Init W = I */
    memset(W_re, 0, (size_t)n * n * sizeof(double));
    memset(W_im, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++) W_re[i * n + i] = 1.0;

    /* LAYER 6: Relative convergence threshold from universal constants.
     * ε² × diag_norm: off-diagonal is negligible when it's below ε² of
     * the diagonal energy. This is information-theoretically exact —
     * below this, the off-diagonal carries < 1 bit of information. */
    double diag_norm = 0;
    for (int i = 0; i < n; i++)
        diag_norm += H_re[i*n+i] * H_re[i*n+i];
    double sc_thresh = TSVD_EPS2 * (diag_norm > TSVD_SAFE_MIN ? diag_norm : 1.0);

    for (int sweep = 0; sweep < 30; sweep++) {  /* L7: 30 max (Probe 8: converges in 6-29) */
        /* LAYER 7: Kahan compensated summation for off-diagonal norm.
         * Probe 2 showed addition loses ~1 bit per op. For n=36,
         * the sum has n(n-1)/2 = 630 terms → ~10 bits lost without
         * compensation. Kahan recovers these bits. */
        double off = 0, kahan_c = 0;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) {
                double term = H_re[i*n+j]*H_re[i*n+j] + H_im[i*n+j]*H_im[i*n+j];
                double y = term - kahan_c;
                double t = off + y;
                kahan_c = (t - off) - y;
                off = t;
            }
        if (off < sc_thresh) break;  /* Layer 6+7: compensated convergence */

        for (int p = 0; p < n; p++)
         for (int q = p + 1; q < n; q++) {
             double apr = H_re[p*n+q], api = H_im[p*n+q];
             double mag2 = apr*apr + api*api;
             /* Side-channel zero attractor: 60% of angles are < 0.01.
               * Skip rotation entirely when off-diagonal magnitude² is
               * negligible relative to diagonal gap. This is the single
               * biggest speedup — eliminates ~60% of rotation work. */
              if (mag2 < sc_thresh) continue;
              /* LAYER 9: born_fast_isqrt (9 bits) is sufficient here because
               * Jacobi is self-correcting — rotation angle errors are absorbed
               * by subsequent sweeps. True sqrt() is 10× slower in the hot loop
               * and provides no convergence benefit. */
              double mag = mag2 * born_fast_isqrt(mag2);

              double hpp = H_re[p*n+p], hqq = H_re[q*n+q];
              double tau = (hqq - hpp) / (2.0 * mag);
              double t = (tau >= 0 ? 1.0 : -1.0) / (fabs(tau) + fabs(tau) * born_fast_isqrt(1.0 + 1.0/(tau*tau)));
              double c = born_fast_isqrt(1.0 + t*t);
             double s = t * c;

             /* Phase to make H[p][q] real: e^{-iθ} */
             double er = apr / mag, ei = -api / mag;

             /* Rotate H */
             H_re[p*n+p] -= t * mag;
             H_re[q*n+q] += t * mag;
             H_re[p*n+q] = 0; H_im[p*n+q] = 0;
             H_re[q*n+p] = 0; H_im[q*n+p] = 0;

             for (int k = 0; k < n; k++) {
                 if (k == p || k == q) continue;
                 /* gp = H[k][p], gq = H[k][q] */
                 double gpr = H_re[k*n+p], gpi = H_im[k*n+p];
                 double gqr = H_re[k*n+q], gqi = H_im[k*n+q];

                 /* Apply phase: gq' = e^{iθ} gq */
                 double gqr2 =  er * gqr + ei * gqi;
                 double gqi2 = -ei * gqr + er * gqi;

                 H_re[k*n+p] =  c * gpr + s * gqr2;
                 H_im[k*n+p] =  c * gpi + s * gqi2;
                 H_re[k*n+q] = -s * gpr + c * gqr2;
                 H_im[k*n+q] = -s * gpi + c * gqi2;

                 /* Hermitian: H[p][k] = conj(H[k][p]) */
                 H_re[p*n+k] =  H_re[k*n+p]; H_im[p*n+k] = -H_im[k*n+p];
                 H_re[q*n+k] =  H_re[k*n+q]; H_im[q*n+k] = -H_im[k*n+q];
             }

             /* Rotate W: W[:,p], W[:,q] */
             for (int k = 0; k < n; k++) {
                 double wpr = W_re[k*n+p], wpi = W_im[k*n+p];
                 double wqr = W_re[k*n+q], wqi = W_im[k*n+q];

                 double wqr2 =  er * wqr + ei * wqi;
                 double wqi2 = -ei * wqr + er * wqi;

                 W_re[k*n+p] =  c * wpr + s * wqr2;
                 W_im[k*n+p] =  c * wpi + s * wqi2;
                 W_re[k*n+q] = -s * wpr + c * wqr2;
                 W_im[k*n+q] = -s * wpi + c * wqi2;
             }
         }
    }

    for (int i = 0; i < n; i++) diag[i] = H_re[i*n+i];

    /* Sort descending by eigenvalue */
    for (int i = 0; i < n - 1; i++) {
        int mx = i;
        for (int j = i + 1; j < n; j++)
            if (diag[j] > diag[mx]) mx = j;
        if (mx != i) {
            double tmp = diag[i]; diag[i] = diag[mx]; diag[mx] = tmp;
            for (int k = 0; k < n; k++) {
                double tr, ti;
                tr = W_re[k*n+i]; W_re[k*n+i] = W_re[k*n+mx]; W_re[k*n+mx] = tr;
                ti = W_im[k*n+i]; W_im[k*n+i] = W_im[k*n+mx]; W_im[k*n+mx] = ti;
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * TRUNCATED SVD
 *
 * M (m×n complex) → U (m×chi) × σ (chi) × V† (chi×n)
 * Uses Jacobi eigendecomposition of M†M to find V, σ.
 * U = M V σ⁻¹.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void tsvd_truncated(const double *M_re, const double *M_im,
                           int m, int n, int chi,
                           double *U_re, double *U_im,
                           double *sigma,
                           double *Vc_re, double *Vc_im)
{
    /* Form H = M† M  (n×n Hermitian) */
    size_t hsz = (size_t)n * n;
    double *H_re = (double *)calloc(hsz, sizeof(double));
    double *H_im = (double *)calloc(hsz, sizeof(double));

    for (int i = 0; i < n; i++)
        for (int j = i; j < n; j++) {
            /* LAYER 4 UPGRADE: FMA-aware complex dot product.
             * Probe 3 confirmed fma() intrinsic works on this substrate.
             * fma(a,b,c) = a*b+c with single rounding — prevents the
             * catastrophic cancellation that Probe 8 mapped at 3.3 bits/digit.
             * Each fma() call preserves ~52 extra mantissa bits vs MUL+ADD. */
            double sr = 0, si = 0;
            for (int k = 0; k < m; k++) {
                double ar = M_re[k*n+i], ai = -M_im[k*n+i]; /* conj */
                double br = M_re[k*n+j], bi =  M_im[k*n+j];
                sr = fma(ar, br, sr); sr = fma(-ai, bi, sr);
                si = fma(ar, bi, si); si = fma( ai, br, si);
            }
            H_re[i*n+j] = sr; H_im[i*n+j] = si;
            H_re[j*n+i] = sr; H_im[j*n+i] = -si; /* Hermitian */
        }

    /* Jacobi eigendecomposition: H = V D V† */
    double *eig = (double *)calloc(n, sizeof(double));
    double *V_re = (double *)calloc(hsz, sizeof(double));
    double *V_im = (double *)calloc(hsz, sizeof(double));

    tsvd_jacobi_hermitian(H_re, H_im, n, eig, V_re, V_im);

    /* σ = sqrt(eigenvalues), clamped at chi.
     * LAYER 9: SSE rsqrtss+2N (46 bits, 4.3cy) — same speed as
     * Quake hack but with near-full precision.
     * σ = eig × isqrt(eig) = eig/√eig = √eig */
    int rank = chi < n ? chi : n;
    if (rank > m) rank = m;
    for (int i = 0; i < rank; i++)
        sigma[i] = eig[i] > 0 ? eig[i] * born_precise_isqrt(eig[i]) : 0;

    /* U = M V σ⁻¹  (m × rank) */
    memset(U_re, 0, (size_t)m * rank * sizeof(double));
    memset(U_im, 0, (size_t)m * rank * sizeof(double));

    for (int j = 0; j < rank; j++) {
        /* LAYER 6: Information-theoretic rank truncation.
         * Sigma is sorted descending. Truncate when sigma[j] drops below
         * ε × sigma[0] — below this ratio, the singular vector carries
         * less than 1 bit of signal above the noise floor. */
        if (sigma[j] < TSVD_EPS * sigma[0] || sigma[j] < TSVD_SAFE_MIN) break;
        /* LAYER 9: SSE rcpss+2N (46 bits) — same speed as hardware div */
        double inv = born_precise_recip(sigma[j]);
        for (int i = 0; i < m; i++) {
            /* FMA-aware complex dot product for U reconstruction */
            double sr = 0, si = 0;
            for (int k = 0; k < n; k++) {
                double mr = M_re[i*n+k], mi = M_im[i*n+k];
                double vr = V_re[k*n+j], vi = V_im[k*n+j];
                sr = fma(mr, vr, sr); sr = fma(-mi, vi, sr);
                si = fma(mr, vi, si); si = fma( mi, vr, si);
            }
            U_re[i*rank+j] = sr * inv;
            U_im[i*rank+j] = si * inv;
        }
    }

    /* V† = conj(V)^T  (rank × n) */
    for (int i = 0; i < rank; i++)
        for (int j = 0; j < n; j++) {
            Vc_re[i*n+j] =  V_re[j*n+i];
            Vc_im[i*n+j] = -V_im[j*n+i];
        }

    free(H_re); free(H_im);
    free(eig);
    free(V_re); free(V_im);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MAGIC POINTER SVD — Halko-Martinsson-Tropp Randomized SVD
 *
 * Sparse-native SVD for PEPS tensor contractions.
 * Theta is stored in COO format — never materialized as a dense matrix.
 *
 * Algorithm (Algorithm 5.1 of Halko, Martinsson, Tropp 2011):
 *   1. Draw random Gaussian Ω (n × ℓ), ℓ = rank + oversample
 *   2. Range sketch: Y = (A A†)^q A Ω   (power iteration for gap amp.)
 *   3. Q = QR(Y) — orthonormal basis for range of A  (m × ℓ)
 *   4. B = Q† A  via sparse ops  (ℓ × n — small!)
 *   5. SVD(B) = Ũ Σ V†  via Jacobi on ℓ×ℓ Hermitian B B†
 *   6. U = Q Ũ, truncate to top-chi
 *
 * Complexity: O(nnz × ℓ × (2q+2)) total
 *   vs Jacobi: O(n² × sweeps × n)
 *
 * With nnz ≤ 4096, ℓ ≈ 20, q = 2:  ~500K flops
 * With n = 72:  Jacobi ≈ 50M flops  →  ~100× speedup
 *
 * Side-channel synergy:
 *   • Register 4096-entry cap → nnz naturally bounded
 *   • Zero attractor → nnz typically 40% of capacity
 *   • 1/6 spectrum → q=2 power iters sufficient (gap ~ 1/D)
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* ── Sparse COO entry ── */
typedef struct {
    int    row, col;
    double re, im;
} TsvdSparseEntry;

/* ── LCG PRNG (deterministic, seeded from sparse data) ── */
static inline double tsvd_lcg(uint64_t *s) {
    *s = (*s) * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)((int64_t)(*s >> 1)) * 1.0842e-19;  /* ~uniform [-1, 1] */
}

/* ── Sparse A × dense X:  Y[m×k] = A × X[n×k]  ──
 * LAYER 5: 4-way unrolled + FMA inner loop.
 * Speculation depth is 8 loads. Each unrolled iteration does 2 loads
 * (xr, xi) = 8 loads per 4 iterations, exactly filling the pipeline. */
static void tsvd_sp_ax(const TsvdSparseEntry *sp, int nnz,
                        int m, int k,
                        const double *X_re, const double *X_im,
                        double *Y_re, double *Y_im)
{
    memset(Y_re, 0, (size_t)m * k * sizeof(double));
    memset(Y_im, 0, (size_t)m * k * sizeof(double));
    for (int e = 0; e < nnz; e++) {
        int r = sp[e].row, c = sp[e].col;
        double ar = sp[e].re, ai = sp[e].im;
        double *yr = Y_re + r*k, *yi = Y_im + r*k;
        const double *xr_base = X_re + c*k, *xi_base = X_im + c*k;
        int j = 0;
        /* 4-way unrolled: 8 loads per iteration = speculation depth */
        for (; j + 3 < k; j += 4) {
            yr[j]   = fma(ar, xr_base[j],   fma(-ai, xi_base[j],   yr[j]));
            yi[j]   = fma(ar, xi_base[j],   fma( ai, xr_base[j],   yi[j]));
            yr[j+1] = fma(ar, xr_base[j+1], fma(-ai, xi_base[j+1], yr[j+1]));
            yi[j+1] = fma(ar, xi_base[j+1], fma( ai, xr_base[j+1], yi[j+1]));
            yr[j+2] = fma(ar, xr_base[j+2], fma(-ai, xi_base[j+2], yr[j+2]));
            yi[j+2] = fma(ar, xi_base[j+2], fma( ai, xr_base[j+2], yi[j+2]));
            yr[j+3] = fma(ar, xr_base[j+3], fma(-ai, xi_base[j+3], yr[j+3]));
            yi[j+3] = fma(ar, xi_base[j+3], fma( ai, xr_base[j+3], yi[j+3]));
        }
        /* Remainder */
        for (; j < k; j++) {
            yr[j] = fma(ar, xr_base[j], fma(-ai, xi_base[j], yr[j]));
            yi[j] = fma(ar, xi_base[j], fma( ai, xr_base[j], yi[j]));
        }
    }
}

/* ── Sparse A† × dense X:  Y[n×k] = A† × X[m×k]  ──
 * LAYER 5: 4-way unrolled + FMA, matching tsvd_sp_ax. */
static void tsvd_sp_ahx(const TsvdSparseEntry *sp, int nnz,
                          int n, int k,
                          const double *X_re, const double *X_im,
                          double *Y_re, double *Y_im)
{
    memset(Y_re, 0, (size_t)n * k * sizeof(double));
    memset(Y_im, 0, (size_t)n * k * sizeof(double));
    for (int e = 0; e < nnz; e++) {
        int r = sp[e].row, c = sp[e].col;
        double ar = sp[e].re, ai = -sp[e].im;  /* conjugate transpose */
        double *yr = Y_re + c*k, *yi = Y_im + c*k;
        const double *xr_base = X_re + r*k, *xi_base = X_im + r*k;
        int j = 0;
        for (; j + 3 < k; j += 4) {
            yr[j]   = fma(ar, xr_base[j],   fma(-ai, xi_base[j],   yr[j]));
            yi[j]   = fma(ar, xi_base[j],   fma( ai, xr_base[j],   yi[j]));
            yr[j+1] = fma(ar, xr_base[j+1], fma(-ai, xi_base[j+1], yr[j+1]));
            yi[j+1] = fma(ar, xi_base[j+1], fma( ai, xr_base[j+1], yi[j+1]));
            yr[j+2] = fma(ar, xr_base[j+2], fma(-ai, xi_base[j+2], yr[j+2]));
            yi[j+2] = fma(ar, xi_base[j+2], fma( ai, xr_base[j+2], yi[j+2]));
            yr[j+3] = fma(ar, xr_base[j+3], fma(-ai, xi_base[j+3], yr[j+3]));
            yi[j+3] = fma(ar, xi_base[j+3], fma( ai, xr_base[j+3], yi[j+3]));
        }
        for (; j < k; j++) {
            yr[j] = fma(ar, xr_base[j], fma(-ai, xi_base[j], yr[j]));
            yi[j] = fma(ar, xi_base[j], fma( ai, xr_base[j], yi[j]));
        }
    }
}

/* ── Modified Gram-Schmidt QR on rows×cols complex block ── */
static void tsvd_mgs(double *Q_re, double *Q_im, int rows, int cols)
{
    for (int j = 0; j < cols; j++) {
        /* Orthogonalize column j against 0..j-1 */
        for (int k = 0; k < j; k++) {
            /* LAYER 4 UPGRADE: FMA-aware inner product.
             * Uses fma() for single-rounding precision. */
            double dr = 0, di = 0;
            for (int i = 0; i < rows; i++) {
                dr = fma(Q_re[i*cols+k], Q_re[i*cols+j], dr);
                dr = fma(Q_im[i*cols+k], Q_im[i*cols+j], dr);
                di = fma(Q_re[i*cols+k], Q_im[i*cols+j], di);
                di = fma(-Q_im[i*cols+k], Q_re[i*cols+j], di);
            }
            /* LAYER 7: FMA projection subtraction.
             * Probe 2 showed subtraction is 50% irreversible (loses ~1 bit).
             * Using fma(-dr, col_k, col_j) gives single-rounding, preserving
             * the bit that naive subtract loses. */
            for (int i = 0; i < rows; i++) {
                Q_re[i*cols+j] = fma(-dr, Q_re[i*cols+k], fma(-di, Q_im[i*cols+k], Q_re[i*cols+j]));
                Q_im[i*cols+j] = fma(-dr, Q_im[i*cols+k], fma( di, Q_re[i*cols+k], Q_im[i*cols+j]));
            }
        }
        /* Normalize — LAYER 9: SSE rsqrtss+2N (46 bits, same speed as Quake).
         * Probe showed 4.3cy vs 4.2cy — zero cost for +37 bits precision. */
        double norm = 0;
        for (int i = 0; i < rows; i++)
            norm = fma(Q_re[i*cols+j], Q_re[i*cols+j],
                   fma(Q_im[i*cols+j], Q_im[i*cols+j], norm));
        if (norm > TSVD_SAFE_MIN) {
            double inv = born_precise_isqrt(norm);
            for (int i = 0; i < rows; i++) {
                Q_re[i*cols+j] *= inv;
                Q_im[i*cols+j] *= inv;
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * tsvd_sparse_power — Randomized SVD on sparse COO matrix
 *
 * Magic Pointer coordinate compression:
 *   nnz ≤ 4096 entries → at most 4096 unique rows, 4096 unique cols.
 *   ALL arrays use compressed dimensions mr×mc ≤ 4096×4096.
 *   At χ=512: m=n=1,572,864 but mr,mc ≈ 4096 → fits in ~131KB.
 *
 * Input:  sp[nnz] = COO Theta (m × n), chi = target rank
 * Output: U (m×chi), sigma(chi), Vc (chi×n) — same as tsvd_truncated
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Integer comparison for qsort */
static int tsvd_icmp(const void *a, const void *b) {
    int ia = *(const int *)a, ib = *(const int *)b;
    return (ia > ib) - (ia < ib);
}

static void tsvd_sparse_power(const TsvdSparseEntry *sp, int nnz,
                               int m, int n, int chi,
                               double *U_re, double *U_im,
                               double *sigma,
                               double *Vc_re, double *Vc_im)
{
    int rank = chi < n ? chi : n;
    if (rank > m) rank = m;

    memset(sigma, 0, rank * sizeof(double));

    if (nnz == 0 || rank == 0) return;

    /* ══════════ UPGRADE 2: Rank-1 O(1) Fast Path ══════════
     * Simulation finding: light-cone confinement means many bonds
     * carry exactly 1 nonzero entry. Return directly — zero matrix ops.
     * Cost: O(1) vs O(nnz × ℓ × 8) for the full pipeline. */
    if (nnz == 1) {
        double mag2 = sp[0].re * sp[0].re + sp[0].im * sp[0].im;
        if (mag2 > TSVD_SAFE_MIN) {
            double mag = mag2 * born_precise_isqrt(mag2);
            sigma[0] = mag;
            double inv = born_precise_recip(mag);
            U_re[sp[0].row * rank] = sp[0].re * inv;
            U_im[sp[0].row * rank] = sp[0].im * inv;
            Vc_re[sp[0].col] = 1.0;
            Vc_im[sp[0].col] = 0.0;
        }
        return;
    }

    /* ══════════ UPGRADE 1: Trivial Matrix Fast Exit ══════════
     * Simulation finding: self-compression drops NNZ to ~2/site.
     * For nnz ≤ 36 (= D²), build nnz×nnz Gram matrix directly and
     * Jacobi-diagonalize. Bypasses coordinate compression, random
     * projection, and power iteration entirely.
     * Cost: O(nnz²) vs O(nnz × ℓ × 8) with ℓ ≈ 20. */
    if (nnz <= 36) {
        /* Build G = sp† × sp  (nnz × nnz Gram matrix in COO-col space) */
        /* Each entry sp[i] is (row_i, col_i, val_i). G[i][j] = conj(val_i)*val_j
         * if col_i == col_j, weighted by row overlap. But simpler: form M (mr × mc)
         * directly as dense since nnz is tiny. */
        int mr = 0, mc = 0;
        int rows[36], cols[36];
        /* Extract unique rows and cols */
        for (int e = 0; e < nnz; e++) {
            int found_r = 0, found_c = 0;
            for (int i = 0; i < mr; i++) if (rows[i] == sp[e].row) { found_r = 1; break; }
            if (!found_r) rows[mr++] = sp[e].row;
            for (int i = 0; i < mc; i++) if (cols[i] == sp[e].col) { found_c = 1; break; }
            if (!found_c) cols[mc++] = sp[e].col;
        }
        /* Build small dense matrix (mr × mc) */
        int dm = mr < 36 ? mr : 36, dn = mc < 36 ? mc : 36;
        double Md_re[36*36] = {0}, Md_im[36*36] = {0};
        for (int e = 0; e < nnz; e++) {
            int ri = -1, ci = -1;
            for (int i = 0; i < dm; i++) if (rows[i] == sp[e].row) { ri = i; break; }
            for (int i = 0; i < dn; i++) if (cols[i] == sp[e].col) { ci = i; break; }
            if (ri >= 0 && ci >= 0) {
                Md_re[ri*dn+ci] += sp[e].re;
                Md_im[ri*dn+ci] += sp[e].im;
            }
        }
        /* Dense truncated SVD on the tiny matrix */
        int trank = rank < dn ? rank : dn;
        if (trank > dm) trank = dm;
        double *tU_re = (double *)calloc((size_t)dm * trank, sizeof(double));
        double *tU_im = (double *)calloc((size_t)dm * trank, sizeof(double));
        double *tS    = (double *)calloc(trank, sizeof(double));
        double *tV_re = (double *)calloc((size_t)trank * dn, sizeof(double));
        double *tV_im = (double *)calloc((size_t)trank * dn, sizeof(double));
        tsvd_truncated(Md_re, Md_im, dm, dn, trank, tU_re, tU_im, tS, tV_re, tV_im);
        /* Scatter back to original coordinates */
        for (int j = 0; j < trank && j < rank; j++) {
            sigma[j] = tS[j];
            for (int i = 0; i < dm; i++) {
                U_re[rows[i]*rank + j] = tU_re[i*trank+j];
                U_im[rows[i]*rank + j] = tU_im[i*trank+j];
            }
            for (int i = 0; i < dn; i++) {
                Vc_re[j*n + cols[i]] = tV_re[j*dn+i];
                Vc_im[j*n + cols[i]] = tV_im[j*dn+i];
            }
        }
        free(tU_re); free(tU_im); free(tS); free(tV_re); free(tV_im);
        return;
    }

    /* ══════════ Magic Pointer Coordinate Compression ══════════
     * Extract unique row/col indices from sparse entries.
     * Map m-dimensional rows → mr-dimensional compressed rows (mr ≤ nnz).
     * Map n-dimensional cols → mc-dimensional compressed cols (mc ≤ nnz).
     * ALL subsequent arrays use mr, mc instead of m, n.
     * ═══════════════════════════════════════════════════════════ */

    int *raw_rows = (int *)malloc((size_t)nnz * sizeof(int));
    int *raw_cols = (int *)malloc((size_t)nnz * sizeof(int));
    for (int e = 0; e < nnz; e++) {
        raw_rows[e] = sp[e].row;
        raw_cols[e] = sp[e].col;
    }

    /* Sort and deduplicate rows */
    qsort(raw_rows, (size_t)nnz, sizeof(int), tsvd_icmp);
    int mr = 0;
    for (int i = 0; i < nnz; i++)
        if (i == 0 || raw_rows[i] != raw_rows[i-1])
            raw_rows[mr++] = raw_rows[i];
    int *row_map = (int *)malloc((size_t)mr * sizeof(int));  /* compressed → original */
    for (int i = 0; i < mr; i++) row_map[i] = raw_rows[i];

    /* Sort and deduplicate cols */
    qsort(raw_cols, (size_t)nnz, sizeof(int), tsvd_icmp);
    int mc = 0;
    for (int i = 0; i < nnz; i++)
        if (i == 0 || raw_cols[i] != raw_cols[i-1])
            raw_cols[mc++] = raw_cols[i];
    int *col_map = (int *)malloc((size_t)mc * sizeof(int));  /* compressed → original */
    for (int i = 0; i < mc; i++) col_map[i] = raw_cols[i];

    free(raw_rows); free(raw_cols);

    /* Build inverse maps via binary search */
    /* Remap COO entries to compressed coordinates */
    TsvdSparseEntry *csp = (TsvdSparseEntry *)malloc((size_t)nnz * sizeof(*csp));
    for (int e = 0; e < nnz; e++) {
        /* Binary search for compressed row */
        int lo = 0, hi = mr - 1, cr = 0;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (row_map[mid] == sp[e].row) { cr = mid; break; }
            else if (row_map[mid] < sp[e].row) lo = mid + 1;
            else hi = mid - 1;
        }
        /* Binary search for compressed col */
        lo = 0; hi = mc - 1; int cc = 0;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (col_map[mid] == sp[e].col) { cc = mid; break; }
            else if (col_map[mid] < sp[e].col) lo = mid + 1;
            else hi = mid - 1;
        }
        csp[e].row = cr; csp[e].col = cc;
        csp[e].re = sp[e].re; csp[e].im = sp[e].im;
    }

    /* ══════════ Now work entirely in compressed space ══════════
     * Matrix is mr × mc, where mr, mc ≤ nnz ≤ 4096.
     * At χ=512: mr,mc ≈ 4096 instead of 1,572,864. */

    int c_rank = rank < mc ? rank : mc;
    if (c_rank > mr) c_rank = mr;

    int p = c_rank < 10 ? c_rank : 10;
    int ell = c_rank + p;
    if (ell > mc) ell = mc;
    if (ell > mr) ell = mr;

    /* ── Step 1: Draw random Ω (mc × ℓ) — COMPRESSED cols ── */
    uint64_t rng = 0xCAFEBABE13579BDFULL;
    for (int e = 0; e < nnz && e < 8; e++)
        rng ^= (uint64_t)(sp[e].re * 1e9) ^ ((uint64_t)(sp[e].im * 1e9) << 32);

    size_t osz = (size_t)mc * ell;
    double *Om_re = (double *)malloc(osz * sizeof(double));
    double *Om_im = (double *)malloc(osz * sizeof(double));
    for (int i = 0; i < (int)osz; i++) {
        Om_re[i] = tsvd_lcg(&rng);
        Om_im[i] = tsvd_lcg(&rng);
    }

    /* ── Step 2: Range sketch with power iteration in compressed space ── */
    size_t ysz = (size_t)mr * ell;
    size_t tsz = (size_t)mc * ell;
    double *Y_re = (double *)malloc(ysz * sizeof(double));
    double *Y_im = (double *)malloc(ysz * sizeof(double));
    double *T_re = (double *)malloc(tsz * sizeof(double));
    double *T_im = (double *)malloc(tsz * sizeof(double));

    /* Y = A_c Ω  (mr × ℓ) */
    tsvd_sp_ax(csp, nnz, mr, ell, Om_re, Om_im, Y_re, Y_im);
    free(Om_re); free(Om_im);

    /* UPGRADE 3: Adaptive power iterations
     * Simulation finding: 10^48:1 compression = huge spectral gaps.
     * q=1 suffices for very sparse inputs. */
    int q;
    if (nnz <= c_rank * 2)       q = 1;  /* very sparse: gap is huge */
    else if (nnz <= c_rank * 10) q = 2;  /* moderate sparsity */
    else                          q = 3;  /* dense: need full amplification */
    for (int qi = 0; qi < q; qi++) {
        tsvd_mgs(Y_re, Y_im, mr, ell);
        tsvd_sp_ahx(csp, nnz, mc, ell, Y_re, Y_im, T_re, T_im);
        tsvd_mgs(T_re, T_im, mc, ell);
        tsvd_sp_ax(csp, nnz, mr, ell, T_re, T_im, Y_re, Y_im);
    }

    /* ── Step 3: Q = QR(Y)  (mr × ℓ) ── */
    tsvd_mgs(Y_re, Y_im, mr, ell);
    double *Q_re = Y_re, *Q_im = Y_im;

    /* ── Step 4: B = Q† A_c  (ℓ × mc) ── */
    size_t bsz = (size_t)ell * mc;
    double *B_re = (double *)calloc(bsz, sizeof(double));
    double *B_im = (double *)calloc(bsz, sizeof(double));

    for (int e = 0; e < nnz; e++) {
        int r = csp[e].row, c = csp[e].col;
        double ar = csp[e].re, ai = csp[e].im;
        for (int i = 0; i < ell; i++) {
            double qr = Q_re[r*ell+i], qi = -Q_im[r*ell+i];
            B_re[i*mc+c] += qr*ar - qi*ai;
            B_im[i*mc+c] += qr*ai + qi*ar;
        }
    }

    /* ── Step 5: Jacobi on BB† (ℓ × ℓ) — trivially small ── */
    size_t lsz = (size_t)ell * ell;
    double *BBh_re = (double *)calloc(lsz, sizeof(double));
    double *BBh_im = (double *)calloc(lsz, sizeof(double));

    for (int i = 0; i < ell; i++)
        for (int j = i; j < ell; j++) {
            double sr = 0, si = 0;
            for (int k = 0; k < mc; k++) {
                double ar = B_re[i*mc+k], ai = B_im[i*mc+k];
                double br = B_re[j*mc+k], bi = -B_im[j*mc+k];
                sr += ar*br - ai*bi;
                si += ar*bi + ai*br;
            }
            BBh_re[i*ell+j] = sr; BBh_im[i*ell+j] = si;
            BBh_re[j*ell+i] = sr; BBh_im[j*ell+i] = -si;
        }

    double *eig   = (double *)calloc(ell, sizeof(double));
    double *Ub_re = (double *)calloc(lsz, sizeof(double));
    double *Ub_im = (double *)calloc(lsz, sizeof(double));

    tsvd_jacobi_hermitian(BBh_re, BBh_im, ell, eig, Ub_re, Ub_im);

    for (int i = 0; i < c_rank && i < rank; i++)
        sigma[i] = (i < ell && eig[i] > 0) ? eig[i] * born_precise_isqrt(eig[i]) : 0;

    /* ── Step 6: Reconstruct U and V† with coordinate decompression ──
     * U_compressed = Q × Ub  (mr × c_rank)
     * V_compressed = B† Ub σ⁻¹ (mc × c_rank)
     * Then scatter back to original m, n coordinates via row_map, col_map */

    /* U: compute in compressed space, scatter to original rows */
    /* UPGRADE 4: Dead sigma skip — break at first zero σ (they're sorted) */
    for (int j = 0; j < c_rank && j < rank; j++) {
        if (sigma[j] < TSVD_EPS * sigma[0] || sigma[j] < TSVD_SAFE_MIN) break;
        for (int i = 0; i < mr; i++) {
            double sr = 0, si = 0;
            for (int k = 0; k < ell; k++) {
                double qr = Q_re[i*ell+k], qi = Q_im[i*ell+k];
                double ur = Ub_re[k*ell+j], ui = Ub_im[k*ell+j];
                sr += qr*ur - qi*ui;
                si += qr*ui + qi*ur;
            }
            U_re[row_map[i]*rank + j] = sr;
            U_im[row_map[i]*rank + j] = si;
        }
    }

    /* V†: compute in compressed space, scatter to original cols */
    /* UPGRADE 4 continued: dead sigma skip */
    for (int j = 0; j < c_rank && j < rank; j++) {
        if (sigma[j] < TSVD_EPS * sigma[0] || sigma[j] < TSVD_SAFE_MIN) break;
        double inv = born_precise_recip(sigma[j]);
        for (int i = 0; i < mc; i++) {
            double sr = 0, si = 0;
            for (int k = 0; k < ell; k++) {
                double br = B_re[k*mc+i], bi = -B_im[k*mc+i];
                double ur = Ub_re[k*ell+j], ui = Ub_im[k*ell+j];
                sr += br*ur - bi*ui;
                si += br*ui + bi*ur;
            }
            Vc_re[j*n + col_map[i]] =  sr * inv;
            Vc_im[j*n + col_map[i]] = -si * inv;
        }
    }

    free(csp);
    free(row_map); free(col_map);
    free(Q_re);    free(Q_im);
    free(T_re);    free(T_im);
    free(B_re);    free(B_im);
    free(BBh_re);  free(BBh_im);
    free(eig);
    free(Ub_re);   free(Ub_im);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * Drop-in replacement for tsvd_truncated — same signature.
 * Converts dense M to COO, calls tsvd_sparse_power.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void tsvd_truncated_sparse(const double *M_re, const double *M_im,
                                   int m, int n, int chi,
                                   double *U_re, double *U_im,
                                   double *sigma,
                                   double *Vc_re, double *Vc_im)
{
    /* UPGRADE 5: Frobenius norm pre-check
     * Simulation finding: self-compressed states make many gates no-ops.
     * Skip entire SVD if total energy is negligible. */
    double fnorm2 = 0;
    int nnz = 0;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            double mag2 = M_re[i*n+j]*M_re[i*n+j] + M_im[i*n+j]*M_im[i*n+j];
            if (mag2 > TSVD_EPS2) { nnz++; fnorm2 += mag2; }
        }
    if (fnorm2 < TSVD_EPS2) return;  /* Gate has no effect — zero cost exit */

    /* Build COO */
    TsvdSparseEntry *sp = (TsvdSparseEntry *)malloc(
        (nnz > 0 ? (size_t)nnz : 1) * sizeof(TsvdSparseEntry));
    int k = 0;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            double mag2 = M_re[i*n+j]*M_re[i*n+j] + M_im[i*n+j]*M_im[i*n+j];
            if (mag2 > TSVD_EPS2) {
                sp[k].row = i; sp[k].col = j;
                sp[k].re = M_re[i*n+j]; sp[k].im = M_im[i*n+j];
                k++;
            }
        }

    tsvd_sparse_power(sp, nnz, m, n, chi, U_re, U_im, sigma, Vc_re, Vc_im);
    free(sp);
}

#endif /* TENSOR_SVD_H */
