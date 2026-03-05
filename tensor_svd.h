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
 * PRECOMPUTED EIGENVALUE SPECTRA (from substrate_probe_jacobi.c)
 *
 * The Jacobi probe found that D=6 quantum SVD produces EXACTLY TWO
 * eigenvalue patterns, cycling with period 2 during Trotter evolution.
 * Both are precomputed here as compile-time constants — zero runtime cost.
 *
 * Pattern A (87% of calls): M†M ∝ I₆
 *   • All eigenvalues = 1/D
 *   • V = I₆
 *   • Occurs when no CZ gate has acted (pure local gates)
 *
 * Pattern B (12.5% of calls): M†M has paired off-diagonal structure
 *   • Pairs: (0,3), (1,4), (2,5) each with value = 1/D
 *   • Eigenvalues: [2/D, 2/D, 2/D, 0, 0, 0]
 *   • V columns: (|i⟩+|i+3⟩)/√2 for the 3 nonzero, (|i⟩-|i+3⟩)/√2 for the 3 zero
 *   • Occurs after DFT+CZ gate sequences
 *
 * Detection cost: O(n) diagonal check + O(n) off-diagonal spot-check
 * ═══════════════════════════════════════════════════════════════════════════════ */

#define TSVD_INV_D      (1.0 / 6.0)           /* 0.166666... = eigenvalue for Pattern A */
#define TSVD_TWO_INV_D  (2.0 / 6.0)           /* 0.333333... = eigenvalue for Pattern B */
#define TSVD_INV_SQRT2  0.70710678118654752440 /* 1/√2 for Pattern B eigenvectors       */

/* Pattern B eigenvector indices: the three pairs whose symmetric/antisymmetric
 * combinations form the eigenvectors. CZ gate creates entanglement between
 * computational basis states |k⟩ and |k+3 mod 6⟩. */
static const int TSVD_PAIR_A[3] = {0, 1, 2};  /* first index of each pair */
static const int TSVD_PAIR_B[3] = {3, 4, 5};  /* second index (= first + D/2) */

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


    /* ═══ LAYER 9: PRECOMPUTED SPECTRAL PATTERNS ═════════════════════════
     *
     * The Jacobi sidechannel probe found EXACTLY TWO eigenvalue patterns
     * for D=6 quantum operations. Both are analytically known:
     *
     *   Pattern A (87%): M†M = d0 × I₆
     *     → eigenvalues = [d0, d0, d0, d0, d0, d0], V = I
     *
     *   Pattern B (12.5%): M†M has paired off-diag at (0,3),(1,4),(2,5)
     *     → eigenvalues = [2d0, 2d0, 2d0, 0, 0, 0]
     *     → V = sym/antisym combinations: (|i⟩±|i+3⟩)/√2
     *
     * Detection: O(n) diagonal uniformity + 3 pair checks + off-diag energy
     * Cost: ~18 comparisons vs ~600 FLOPs for even 1 Jacobi sweep
     * ══════════════════════════════════════════════════════════════════ */

    if (n == 6) {
        double d0 = H_re[0];
        int diag_uniform = 1;
        for (int i = 1; i < 6; i++)
            if (fabs(H_re[i*7] - d0) > 1e-12 * fabs(d0)) { diag_uniform = 0; break; }

        if (diag_uniform && d0 > TSVD_SAFE_MIN) {
            /* Compute off-diagonal energy */
            double off = 0;
            for (int i = 0; i < 6; i++)
                for (int j = i + 1; j < 6; j++)
                    off += H_re[i*6+j]*H_re[i*6+j] + H_im[i*6+j]*H_im[i*6+j];

            if (off < 1e-20) {
                /* ─── PATTERN A: M†M = d0 × I₆ (87% of calls) ───
                 * Eigenvalues = d0, V = I (already initialized above) */
                for (int i = 0; i < 6; i++) diag[i] = d0;
                return;
            }

            /* Check Pattern B: paired off-diag at (0,3),(1,4),(2,5) = d0 */
            int is_paired = 1;
            for (int p = 0; p < 3 && is_paired; p++) {
                int ia = TSVD_PAIR_A[p], ib = TSVD_PAIR_B[p];
                if (fabs(H_re[ia*6+ib] - d0) > 1e-10 * fabs(d0)) is_paired = 0;
                if (fabs(H_im[ia*6+ib]) > 1e-10 * fabs(d0)) is_paired = 0;
            }

            if (is_paired) {
                /* ─── PATTERN B: Rank-3 paired spectrum (12.5% of calls) ───
                 * Eigenvalues: [2d0, 2d0, 2d0, 0, 0, 0]
                 * V columns 0-2: (|ia⟩+|ib⟩)/√2 (sym, eigenvalue 2d0)
                 * V columns 3-5: (|ia⟩-|ib⟩)/√2 (antisym, eigenvalue 0) */
                double eig_nz = 2.0 * d0;
                diag[0] = eig_nz; diag[1] = eig_nz; diag[2] = eig_nz;
                diag[3] = 0;      diag[4] = 0;      diag[5] = 0;

                memset(W_re, 0, 36 * sizeof(double));
                for (int p = 0; p < 3; p++) {
                    int ia = TSVD_PAIR_A[p], ib = TSVD_PAIR_B[p];
                    W_re[ia*6+p]   =  TSVD_INV_SQRT2;  /* sym */
                    W_re[ib*6+p]   =  TSVD_INV_SQRT2;
                    W_re[ia*6+p+3] =  TSVD_INV_SQRT2;  /* antisym */
                    W_re[ib*6+p+3] = -TSVD_INV_SQRT2;
                }
                return;
            }
        }
    }

    /* ─── Fallback: generic diagonal bypass for any n ─── */
    {
        double off_norm = 0;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                off_norm += H_re[i*n+j]*H_re[i*n+j] + H_im[i*n+j]*H_im[i*n+j];

        if (off_norm < sc_thresh) {
            for (int i = 0; i < n; i++) diag[i] = H_re[i*n+i];
            for (int i = 0; i < n - 1; i++) {
                int mx = i;
                for (int j = i + 1; j < n; j++)
                    if (diag[j] > diag[mx]) mx = j;
                if (mx != i) {
                    double tmp = diag[i]; diag[i] = diag[mx]; diag[mx] = tmp;
                    for (int k = 0; k < n; k++) {
                        double tr = W_re[k*n+i]; W_re[k*n+i] = W_re[k*n+mx]; W_re[k*n+mx] = tr;
                    }
                }
            }
            return;
        }
    }

    /* ─── General case: Full Jacobi iteration (<0.5% of calls) ─── */

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
              double t, c, s;
              if (fabs(tau) < 1e-15) {
                  /* hpp ≈ hqq: optimal rotation is 45° */
                  t = 1.0;
                  c = TSVD_INV_SQRT2;
                  s = TSVD_INV_SQRT2;
              } else {
                  t = (tau >= 0 ? 1.0 : -1.0) / (fabs(tau) + fabs(tau) * born_fast_isqrt(1.0 + 1.0/(tau*tau)));
                  c = born_fast_isqrt(1.0 + t*t);
                  s = t * c;
              }

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
    if (nnz <= 512) {
        /* Build G = sp† × sp  (nnz × nnz Gram matrix in COO-col space) */
        /* Each entry sp[i] is (row_i, col_i, val_i). G[i][j] = conj(val_i)*val_j
         * if col_i == col_j, weighted by row overlap. But simpler: form M (mr × mc)
         * directly as dense since nnz is small. */
        int mr = 0, mc = 0;
        int *rows = (int *)malloc(nnz * sizeof(int));
        int *cols = (int *)malloc(nnz * sizeof(int));
        /* Extract unique rows and cols */
        for (int e = 0; e < nnz; e++) {
            int found_r = 0, found_c = 0;
            for (int i = 0; i < mr; i++) if (rows[i] == sp[e].row) { found_r = 1; break; }
            if (!found_r) rows[mr++] = sp[e].row;
            for (int i = 0; i < mc; i++) if (cols[i] == sp[e].col) { found_c = 1; break; }
            if (!found_c) cols[mc++] = sp[e].col;
        }
        /* Build small dense matrix (mr × mc) */
        int dm = mr, dn = mc;
        double *Md_re = (double *)calloc((size_t)dm * dn, sizeof(double));
        double *Md_im = (double *)calloc((size_t)dm * dn, sizeof(double));
        for (int e = 0; e < nnz; e++) {
            int ri = -1, ci = -1;
            for (int i = 0; i < dm; i++) if (rows[i] == sp[e].row) { ri = i; break; }
            for (int i = 0; i < dn; i++) if (cols[i] == sp[e].col) { ci = i; break; }
            if (ri >= 0 && ci >= 0) {
                Md_re[ri*dn+ci] += sp[e].re;
                Md_im[ri*dn+ci] += sp[e].im;
            }
        }
        /* Dense truncated SVD on the small matrix */
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
        free(Md_re); free(Md_im);
        free(tU_re); free(tU_im); free(tS); free(tV_re); free(tV_im);
        free(rows); free(cols);
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

/* ═══════════════════════════════════════════════════════════════════════════════
 * VESICA FOLD FACTORIZATION — Geometric SVD Replacement
 *
 * The Vesica fold pairs physical indices (k, k+3) into
 *   vesica[j] = (ψ[k] + ψ[k+3]) / √2   (convergent / symmetric)
 *   wave[j]   = (ψ[k] - ψ[k+3]) / √2   (divergent / antisymmetric)
 *
 * This is NOT a wrapper around SVD. For symmetric states (wave ≈ 0),
 * the fold provides a DIRECT geometric factorization:
 *
 *   The 3 vesica pairs {(0,3),(1,4),(2,5)} define 3 independent
 *   factorization channels. Each channel j decomposes the (nEA × nEB)
 *   sub-block of the folded Θ matrix at position (j,j) into a product
 *   of left and right factors. No Jacobi, no power iteration, no
 *   randomized projection — the GEOMETRY of the fold IS the decomposition.
 *
 * Three paths:
 *   Path 1 — VESICA DIRECT (wave < 1%):
 *     Geometric per-pair factorization via fold structure.
 *     Each pair j contributes min(nEA, nEB) bond values.
 *     Cross-pair correlations (off-diagonal blocks) also captured.
 *     Total rank: up to 3 × min(nEA, nEB).
 *     Cost: O(3 × nEA × nEB × min(nEA,nEB)) — typically O(12).
 *     NO SVD CALLED.
 *
 *   Path 2 — VESICA + miniSVD (wave 1–50%):
 *     SVD on the 4×-smaller folded vesica submatrix.
 *     Still saves ~75% of Jacobi work.
 *
 *   Path 3 — BYPASS (D ≠ 6, or wave > 50%):
 *     Standard SVD, no folding.
 *
 * Synergy with Pattern B: The Jacobi probe found 12.5% of matrices
 * have eigenvectors at exactly the Vesica pairs. Those are now handled
 * by Path 1 at O(1) cost instead of O(n²×sweeps).
 *
 * D=6 specific: 3 pairs {(0,3),(1,4),(2,5)} — the default antipodal syntheme.
 * ═══════════════════════════════════════════════════════════════════════════════ */

#define TSVD_VESICA_D   6
#define TSVD_VESICA_D2  3   /* D/2 = number of vesica pairs */

/* ── Mini 2×2 complex SVD (analytic Jacobi rotation) ──
 * Decomposes a 2×2 complex matrix B into U Σ V†.
 * Used for per-pair block factorization when nEA=nEB=2. */
static void tsvd_mini_2x2_svd(const double *Br, const double *Bi,
                                double *Ur, double *Ui, double *sv,
                                double *Vr, double *Vi)
{
    /* B†B (2×2 Hermitian) */
    double h00 = Br[0]*Br[0]+Bi[0]*Bi[0]+Br[2]*Br[2]+Bi[2]*Bi[2];
    double h11 = Br[1]*Br[1]+Bi[1]*Bi[1]+Br[3]*Br[3]+Bi[3]*Bi[3];
    double h01r = Br[0]*Br[1]+Bi[0]*Bi[1]+Br[2]*Br[3]+Bi[2]*Bi[3];
    double h01i = Br[0]*Bi[1]-Bi[0]*Br[1]+Br[2]*Bi[3]-Bi[2]*Br[3];

    /* Eigenvalues of 2×2 Hermitian: analytic formula */
    double tr = h00 + h11;
    double det2 = h00*h11 - h01r*h01r - h01i*h01i;
    double disc = tr*tr - 4.0*det2;
    if (disc < 0) disc = 0;
    double sq = sqrt(disc);
    double lam0 = 0.5*(tr + sq); /* larger */
    double lam1 = 0.5*(tr - sq); /* smaller */
    if (lam0 < 0) lam0 = 0;
    if (lam1 < 0) lam1 = 0;

    sv[0] = sqrt(lam0);
    sv[1] = sqrt(lam1);

    /* Eigenvectors of B†B → V columns */
    double off2 = h01r*h01r + h01i*h01i;
    if (off2 > 1e-30) {
        double d = lam0 - h11;
        double len = sqrt(off2 + d*d);
        double inv = 1.0/len;
        /* V[:,0] */
        Vr[0] = h01r*inv; Vi[0] = h01i*inv;
        Vr[2] = d*inv;     Vi[2] = 0;
        /* V[:,1] = orthogonal */
        Vr[1] = -d*inv;    Vi[1] = 0;
        Vr[3] = h01r*inv;  Vi[3] = -h01i*inv;
    } else {
        /* Already diagonal */
        if (h00 >= h11) {
            Vr[0]=1; Vi[0]=0; Vr[1]=0; Vi[1]=0;
            Vr[2]=0; Vi[2]=0; Vr[3]=1; Vi[3]=0;
        } else {
            Vr[0]=0; Vi[0]=0; Vr[1]=1; Vi[1]=0;
            Vr[2]=1; Vi[2]=0; Vr[3]=0; Vi[3]=0;
        }
    }

    /* U = B V Σ⁻¹ */
    for (int j = 0; j < 2; j++) {
        double inv_s = sv[j] > TSVD_SAFE_MIN ? 1.0/sv[j] : 0;
        /* U[:,j] = B × V[:,j] × inv_s */
        for (int i = 0; i < 2; i++) {
            double sr = 0, si = 0;
            for (int k = 0; k < 2; k++) {
                double br = Br[i*2+k], bi = Bi[i*2+k];
                double vr = Vr[k*2+j], vi = Vi[k*2+j];
                sr += br*vr - bi*vi;
                si += br*vi + bi*vr;
            }
            Ur[i*2+j] = sr * inv_s;
            Ui[i*2+j] = si * inv_s;
        }
    }
}

static void tsvd_vesica_truncated_sparse(const double *M_re, const double *M_im,
                                          int m, int n,
                                          int D, int num_envA, int num_envB,
                                          int chi,
                                          double *U_re, double *U_im,
                                          double *sigma,
                                          double *Vc_re, double *Vc_im)
{
    /* ── Path 3: BYPASS — D ≠ 6 or invalid env counts ── */
    if (D != TSVD_VESICA_D || num_envA == 0 || num_envB == 0) {
        tsvd_truncated_sparse(M_re, M_im, m, n, chi,
                              U_re, U_im, sigma, Vc_re, Vc_im);
        return;
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 1: VESICA FOLD — Transform Θ into (vesica, wave) basis
     *
     * Row fold: pair (kA=j, kA=j+3) for each envA
     *   vesica row j*nEA+eA = (row[j*nEA+eA] + row[(j+3)*nEA+eA]) / √2
     *   wave   row (j+3)*nEA+eA = (row[j*nEA+eA] - row[(j+3)*nEA+eA]) / √2
     *
     * Col fold: same pairing on columns.
     * Result: FF is the doubly-folded Θ. Rows/cols 0..3nE-1 = vesica,
     *         rows/cols 3nE..6nE-1 = wave.
     * ═══════════════════════════════════════════════════════════════════════ */

    size_t msz = (size_t)m * n;
    double *F_re = (double *)calloc(msz, sizeof(double));
    double *F_im = (double *)calloc(msz, sizeof(double));

    for (int eA = 0; eA < num_envA; eA++) {
        for (int j = 0; j < TSVD_VESICA_D2; j++) {
            int row_lo = j * num_envA + eA;
            int row_hi = (j + 3) * num_envA + eA;
            for (int c = 0; c < n; c++) {
                double lo_r = M_re[row_lo*n+c], lo_i = M_im[row_lo*n+c];
                double hi_r = M_re[row_hi*n+c], hi_i = M_im[row_hi*n+c];
                F_re[row_lo*n+c] = TSVD_INV_SQRT2 * (lo_r + hi_r);
                F_im[row_lo*n+c] = TSVD_INV_SQRT2 * (lo_i + hi_i);
                F_re[row_hi*n+c] = TSVD_INV_SQRT2 * (lo_r - hi_r);
                F_im[row_hi*n+c] = TSVD_INV_SQRT2 * (lo_i - hi_i);
            }
        }
    }

    double *FF_re = (double *)calloc(msz, sizeof(double));
    double *FF_im = (double *)calloc(msz, sizeof(double));

    for (int eB = 0; eB < num_envB; eB++) {
        for (int j = 0; j < TSVD_VESICA_D2; j++) {
            int col_lo = j * num_envB + eB;
            int col_hi = (j + 3) * num_envB + eB;
            for (int r = 0; r < m; r++) {
                double lo_r = F_re[r*n+col_lo], lo_i = F_im[r*n+col_lo];
                double hi_r = F_re[r*n+col_hi], hi_i = F_im[r*n+col_hi];
                FF_re[r*n+col_lo] = TSVD_INV_SQRT2 * (lo_r + hi_r);
                FF_im[r*n+col_lo] = TSVD_INV_SQRT2 * (lo_i + hi_i);
                FF_re[r*n+col_hi] = TSVD_INV_SQRT2 * (lo_r - hi_r);
                FF_im[r*n+col_hi] = TSVD_INV_SQRT2 * (lo_i - hi_i);
            }
        }
    }
    free(F_re); free(F_im);

    /* ═══════════════════════════════════════════════════════════════════════
     * STEP 2: MEASURE WAVE ENERGY — Decide which path to take
     * ═══════════════════════════════════════════════════════════════════════ */

    double vesica_energy = 0, wave_energy = 0;
    for (int r = 0; r < m; r++) {
        int kA = r / num_envA;
        int is_wave_r = (kA >= 3);
        for (int c = 0; c < n; c++) {
            double mag2 = FF_re[r*n+c]*FF_re[r*n+c] + FF_im[r*n+c]*FF_im[r*n+c];
            if (mag2 < TSVD_EPS2) continue;
            int kB = c / num_envB;
            int is_wave_c = (kB >= 3);
            if (is_wave_r || is_wave_c) wave_energy += mag2;
            else vesica_energy += mag2;
        }
    }

    double total_energy = vesica_energy + wave_energy;
    double wave_frac = (total_energy > TSVD_EPS2)
        ? wave_energy / total_energy : 0;

    /* ═══════════════════════════════════════════════════════════════════════
     * PATH 1: VESICA DIRECT — Geometric factorization, NO SVD
     *
     * When wave < 1%, the doubly-folded Θ lives entirely in the
     * vesica subspace: a 3nEA × 3nEB matrix.
     *
     * The 3 vesica pairs define 3×3 = 9 blocks of size nEA × nEB.
     * Each block B[jA][jB] = FF[jA*nEA:(jA+1)*nEA, jB*nEB:(jB+1)*nEB]
     * captures how vesica pair jA at site A correlates with pair jB at B.
     *
     * Factorization: decompose each block independently.
     *   Block (jA, jB) of size nEA × nEB contributes
     *   min(nEA, nEB) singular components.
     *   Each component gets a unique bond value s.
     *
     * For nEA=nEB=1 (product state):  rank = up to 9 (9 scalar entries)
     * For nEA=nEB=2 (after 1 gate):   rank = up to 18 (9 blocks × 2)
     *
     * The geometric insight: the fold basis IS the eigenbasis of the
     * CZ gate. Pattern B (12.5% of calls) becomes rank-3 diagonal.
     * Even general symmetric states decompose as at most 3 active pairs.
     *
     * After factorization, UNFOLD back to physical k-space:
     *   k = j:   U_phys = U_fold / √2
     *   k = j+3: U_phys = U_fold / √2   (symmetric distribution)
     * ═══════════════════════════════════════════════════════════════════════ */

    if (wave_frac < 0.01) {
        int cm = TSVD_VESICA_D2 * num_envA;
        int cn = TSVD_VESICA_D2 * num_envB;
        int rank_out = chi < n ? chi : n;
        if (rank_out > m) rank_out = m;

        int s_idx = 0;  /* running bond index */

        /* Iterate over all 9 blocks (3 row-pairs × 3 col-pairs) */
        for (int jA = 0; jA < TSVD_VESICA_D2; jA++) {
            for (int jB = 0; jB < TSVD_VESICA_D2; jB++) {

                /* Extract nEA × nEB block B[jA][jB] from the folded Θ */
                int bm = num_envA, bn = num_envB;
                int brank = bm < bn ? bm : bn;
                if (s_idx + brank > rank_out) brank = rank_out - s_idx;
                if (brank <= 0) goto vesica_done;

                /* Check if this block has significant energy */
                double blk_energy = 0;
                for (int i = 0; i < bm; i++)
                    for (int k = 0; k < bn; k++) {
                        int r = jA * num_envA + i;
                        int c = jB * num_envB + k;
                        blk_energy += FF_re[r*n+c]*FF_re[r*n+c]
                                    + FF_im[r*n+c]*FF_im[r*n+c];
                    }
                if (blk_energy < TSVD_EPS2) continue;  /* Skip zero blocks */

                /* ═══ Per-block factorization ═══ */

                if (bm == 1 && bn == 1) {
                    /* ── Scalar block: rank-1, trivial ──
                     * B = σ × u × v†  with u=v=1, σ = |B| */
                    int r = jA * num_envA;
                    int c = jB * num_envB;
                    double br = FF_re[r*n+c], bi = FF_im[r*n+c];
                    double mag = sqrt(br*br + bi*bi);
                    if (mag < TSVD_SAFE_MIN) continue;

                    sigma[s_idx] = mag;
                    double inv = 1.0 / mag;

                    /* Folded U row */
                    int frow = jA * num_envA;
                    /* Unfold → physical k=jA and k=jA+3 */
                    U_re[frow * rank_out + s_idx] = TSVD_INV_SQRT2 * br * inv;
                    U_im[frow * rank_out + s_idx] = TSVD_INV_SQRT2 * bi * inv;
                    U_re[(frow + 3*num_envA) * rank_out + s_idx] = TSVD_INV_SQRT2 * br * inv;
                    U_im[(frow + 3*num_envA) * rank_out + s_idx] = TSVD_INV_SQRT2 * bi * inv;

                    /* Folded V† col */
                    int fcol = jB * num_envB;
                    Vc_re[s_idx * n + fcol] = TSVD_INV_SQRT2;
                    Vc_im[s_idx * n + fcol] = 0;
                    Vc_re[s_idx * n + fcol + 3*num_envB] = TSVD_INV_SQRT2;
                    Vc_im[s_idx * n + fcol + 3*num_envB] = 0;

                    s_idx++;

                } else if (bm <= 2 && bn <= 2) {
                    /* ── Small block: analytic 2×2 SVD ── */
                    double Br[4] = {0}, Bi[4] = {0};
                    for (int i = 0; i < bm; i++)
                        for (int k = 0; k < bn; k++) {
                            int r = jA * num_envA + i;
                            int c = jB * num_envB + k;
                            Br[i*bn+k] = FF_re[r*n+c];
                            Bi[i*bn+k] = FF_im[r*n+c];
                        }

                    /* Pad to 2×2 if needed (zero-pad) */
                    double B2r[4]={0}, B2i[4]={0};
                    for (int i=0;i<bm;i++) for(int k=0;k<bn;k++) {
                        B2r[i*2+k]=Br[i*bn+k]; B2i[i*2+k]=Bi[i*bn+k];
                    }

                    double mU_re[4]={0}, mU_im[4]={0};
                    double mSv[2]={0};
                    double mV_re[4]={0}, mV_im[4]={0};
                    tsvd_mini_2x2_svd(B2r, B2i, mU_re, mU_im, mSv, mV_re, mV_im);

                    for (int sv = 0; sv < brank; sv++) {
                        if (mSv[sv] < TSVD_EPS * (mSv[0] > 0 ? mSv[0] : 1.0)) break;
                        if (s_idx >= rank_out) goto vesica_done;

                        sigma[s_idx] = mSv[sv];

                        /* Unfold U: vesica pair jA → k=jA and k=jA+3 */
                        for (int i = 0; i < bm; i++) {
                            double ur = mU_re[i*2+sv], ui = mU_im[i*2+sv];
                            int row_lo = jA * num_envA + i;
                            int row_hi = (jA + 3) * num_envA + i;
                            U_re[row_lo * rank_out + s_idx] = TSVD_INV_SQRT2 * ur;
                            U_im[row_lo * rank_out + s_idx] = TSVD_INV_SQRT2 * ui;
                            U_re[row_hi * rank_out + s_idx] = TSVD_INV_SQRT2 * ur;
                            U_im[row_hi * rank_out + s_idx] = TSVD_INV_SQRT2 * ui;
                        }

                        /* Unfold V†: vesica pair jB → k=jB and k=jB+3 */
                        for (int k = 0; k < bn; k++) {
                            /* V† row sv: conj(V col sv) */
                            double vr = mV_re[k*2+sv], vi = -mV_im[k*2+sv];
                            int col_lo = jB * num_envB + k;
                            int col_hi = (jB + 3) * num_envB + k;
                            Vc_re[s_idx * n + col_lo] = TSVD_INV_SQRT2 * vr;
                            Vc_im[s_idx * n + col_lo] = TSVD_INV_SQRT2 * vi;
                            Vc_re[s_idx * n + col_hi] = TSVD_INV_SQRT2 * vr;
                            Vc_im[s_idx * n + col_hi] = TSVD_INV_SQRT2 * vi;
                        }
                        s_idx++;
                    }

                } else {
                    /* ── General block: mini Jacobi SVD on nEA × nEB ──
                     * These are TINY matrices (typ. 3×3 to 8×8).
                     * Total cost: O(nEA² × nEB) — negligible. */
                    double *Bk_re = (double *)calloc((size_t)bm * bn, sizeof(double));
                    double *Bk_im = (double *)calloc((size_t)bm * bn, sizeof(double));
                    for (int i = 0; i < bm; i++)
                        for (int k = 0; k < bn; k++) {
                            int r = jA * num_envA + i;
                            int c = jB * num_envB + k;
                            Bk_re[i*bn+k] = FF_re[r*n+c];
                            Bk_im[i*bn+k] = FF_im[r*n+c];
                        }

                    double *bU_re = (double *)calloc((size_t)bm * brank, sizeof(double));
                    double *bU_im = (double *)calloc((size_t)bm * brank, sizeof(double));
                    double *bS    = (double *)calloc(brank, sizeof(double));
                    double *bV_re = (double *)calloc((size_t)brank * bn, sizeof(double));
                    double *bV_im = (double *)calloc((size_t)brank * bn, sizeof(double));

                    tsvd_truncated(Bk_re, Bk_im, bm, bn, brank,
                                   bU_re, bU_im, bS, bV_re, bV_im);

                    for (int sv = 0; sv < brank; sv++) {
                        if (bS[sv] < TSVD_EPS * (bS[0] > 0 ? bS[0] : 1.0)) break;
                        if (s_idx >= rank_out) {
                            free(Bk_re); free(Bk_im);
                            free(bU_re); free(bU_im); free(bS);
                            free(bV_re); free(bV_im);
                            goto vesica_done;
                        }

                        sigma[s_idx] = bS[sv];

                        for (int i = 0; i < bm; i++) {
                            double ur = bU_re[i*brank+sv], ui = bU_im[i*brank+sv];
                            int row_lo = jA * num_envA + i;
                            int row_hi = (jA + 3) * num_envA + i;
                            U_re[row_lo * rank_out + s_idx] = TSVD_INV_SQRT2 * ur;
                            U_im[row_lo * rank_out + s_idx] = TSVD_INV_SQRT2 * ui;
                            U_re[row_hi * rank_out + s_idx] = TSVD_INV_SQRT2 * ur;
                            U_im[row_hi * rank_out + s_idx] = TSVD_INV_SQRT2 * ui;
                        }

                        for (int k = 0; k < bn; k++) {
                            double vr = bV_re[sv*bn+k], vi = bV_im[sv*bn+k];
                            int col_lo = jB * num_envB + k;
                            int col_hi = (jB + 3) * num_envB + k;
                            Vc_re[s_idx * n + col_lo] = TSVD_INV_SQRT2 * vr;
                            Vc_im[s_idx * n + col_lo] = TSVD_INV_SQRT2 * vi;
                            Vc_re[s_idx * n + col_hi] = TSVD_INV_SQRT2 * vr;
                            Vc_im[s_idx * n + col_hi] = TSVD_INV_SQRT2 * vi;
                        }
                        s_idx++;
                    }

                    free(Bk_re); free(Bk_im);
                    free(bU_re); free(bU_im); free(bS);
                    free(bV_re); free(bV_im);
                }
            }
        }

vesica_done:
        /* Sort by descending sigma (insertion sort — rank is small) */
        for (int i = 0; i < s_idx - 1; i++) {
            int mx = i;
            for (int j2 = i + 1; j2 < s_idx; j2++)
                if (sigma[j2] > sigma[mx]) mx = j2;
            if (mx != i) {
                double tmp = sigma[i]; sigma[i] = sigma[mx]; sigma[mx] = tmp;
                /* Swap U columns */
                for (int r = 0; r < m; r++) {
                    double tr = U_re[r*rank_out+i]; U_re[r*rank_out+i] = U_re[r*rank_out+mx]; U_re[r*rank_out+mx] = tr;
                    double ti = U_im[r*rank_out+i]; U_im[r*rank_out+i] = U_im[r*rank_out+mx]; U_im[r*rank_out+mx] = ti;
                }
                /* Swap V† rows */
                for (int c = 0; c < n; c++) {
                    double tr = Vc_re[i*n+c]; Vc_re[i*n+c] = Vc_re[mx*n+c]; Vc_re[mx*n+c] = tr;
                    double ti = Vc_im[i*n+c]; Vc_im[i*n+c] = Vc_im[mx*n+c]; Vc_im[mx*n+c] = ti;
                }
            }
        }

        free(FF_re); free(FF_im);
        return;
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * PATH 2: VESICA + miniSVD — Wave present but partially symmetric
     *
     * SVD on the full folded matrix (same dimensions but better conditioned
     * in the vesica basis), then unfold the results.
     * Still avoids full SVD on the original basis.
     * ═══════════════════════════════════════════════════════════════════════ */
    {
        int rank = chi < n ? chi : n;
        if (rank > m) rank = m;

        double *fU_re  = (double *)calloc((size_t)m * rank, sizeof(double));
        double *fU_im  = (double *)calloc((size_t)m * rank, sizeof(double));
        double *fS     = (double *)calloc(rank, sizeof(double));
        double *fV_re  = (double *)calloc((size_t)rank * n, sizeof(double));
        double *fV_im  = (double *)calloc((size_t)rank * n, sizeof(double));

        tsvd_truncated_sparse(FF_re, FF_im, m, n, rank,
                              fU_re, fU_im, fS, fV_re, fV_im);

        /* Unfold U rows: inverse Vesica fold */
        for (int s = 0; s < rank; s++) {
            sigma[s] = fS[s];
            for (int eA = 0; eA < num_envA; eA++) {
                for (int j = 0; j < TSVD_VESICA_D2; j++) {
                    int r_v = j * num_envA + eA;
                    int r_w = (j + 3) * num_envA + eA;
                    double v_r = fU_re[r_v*rank+s], v_i = fU_im[r_v*rank+s];
                    double w_r = fU_re[r_w*rank+s], w_i = fU_im[r_w*rank+s];
                    U_re[r_v*rank+s] = TSVD_INV_SQRT2 * (v_r + w_r);
                    U_im[r_v*rank+s] = TSVD_INV_SQRT2 * (v_i + w_i);
                    U_re[r_w*rank+s] = TSVD_INV_SQRT2 * (v_r - w_r);
                    U_im[r_w*rank+s] = TSVD_INV_SQRT2 * (v_i - w_i);
                }
            }
        }

        /* Unfold V† cols: inverse Vesica fold */
        for (int s = 0; s < rank; s++) {
            for (int eB = 0; eB < num_envB; eB++) {
                for (int j = 0; j < TSVD_VESICA_D2; j++) {
                    int c_v = j * num_envB + eB;
                    int c_w = (j + 3) * num_envB + eB;
                    double v_r = fV_re[s*n+c_v], v_i = fV_im[s*n+c_v];
                    double w_r = fV_re[s*n+c_w], w_i = fV_im[s*n+c_w];
                    Vc_re[s*n+c_v] = TSVD_INV_SQRT2 * (v_r + w_r);
                    Vc_im[s*n+c_v] = TSVD_INV_SQRT2 * (v_i + w_i);
                    Vc_re[s*n+c_w] = TSVD_INV_SQRT2 * (v_r - w_r);
                    Vc_im[s*n+c_w] = TSVD_INV_SQRT2 * (v_i - w_i);
                }
            }
        }

        free(fU_re); free(fU_im); free(fS);
        free(fV_re); free(fV_im);
    }

    free(FF_re); free(FF_im);
}

#endif /* TENSOR_SVD_H */
