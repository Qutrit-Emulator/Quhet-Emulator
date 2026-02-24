/*
 * tensor_svd.h — Shared Jacobi SVD for Tensor Network Overlays
 *
 * Provides truncated SVD of complex matrices using Jacobi
 * eigendecomposition of the Hermitian product M†M.
 *
 * Used by MPS, PEPS 2D, and PEPS3D for 2-site gate application.
 * All inputs/outputs are flat row-major arrays.
 */

#ifndef TENSOR_SVD_H
#define TENSOR_SVD_H

#include <stdlib.h>
#include <string.h>
#include <math.h>

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

    for (int sweep = 0; sweep < 100; sweep++) {
        double off = 0;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                off += H_re[i*n+j]*H_re[i*n+j] + H_im[i*n+j]*H_im[i*n+j];
        if (off < 1e-100 * n * n) break;

        for (int p = 0; p < n; p++)
         for (int q = p + 1; q < n; q++) {
             double apr = H_re[p*n+q], api = H_im[p*n+q];
             double mag = sqrt(apr*apr + api*api);
             if (mag < 1e-100) continue;

             double hpp = H_re[p*n+p], hqq = H_re[q*n+q];
             double tau = (hqq - hpp) / (2.0 * mag);
             double t = (tau >= 0 ? 1.0 : -1.0) / (fabs(tau) + sqrt(1.0 + tau*tau));
             double c = 1.0 / sqrt(1.0 + t*t);
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
            double sr = 0, si = 0;
            for (int k = 0; k < m; k++) {
                double ar = M_re[k*n+i], ai = -M_im[k*n+i]; /* conj */
                double br = M_re[k*n+j], bi =  M_im[k*n+j];
                sr += ar*br - ai*bi;
                si += ar*bi + ai*br;
            }
            H_re[i*n+j] = sr; H_im[i*n+j] = si;
            H_re[j*n+i] = sr; H_im[j*n+i] = -si; /* Hermitian */
        }

    /* Jacobi eigendecomposition: H = V D V† */
    double *eig = (double *)calloc(n, sizeof(double));
    double *V_re = (double *)calloc(hsz, sizeof(double));
    double *V_im = (double *)calloc(hsz, sizeof(double));

    tsvd_jacobi_hermitian(H_re, H_im, n, eig, V_re, V_im);

    /* σ = sqrt(eigenvalues), clamped at chi */
    int rank = chi < n ? chi : n;
    if (rank > m) rank = m;
    for (int i = 0; i < rank; i++)
        sigma[i] = eig[i] > 0 ? sqrt(eig[i]) : 0;

    /* U = M V σ⁻¹  (m × rank) */
    memset(U_re, 0, (size_t)m * rank * sizeof(double));
    memset(U_im, 0, (size_t)m * rank * sizeof(double));

    for (int j = 0; j < rank; j++) {
        if (sigma[j] < 1e-100) continue;
        double inv = 1.0 / sigma[j];
        for (int i = 0; i < m; i++) {
            double sr = 0, si = 0;
            for (int k = 0; k < n; k++) {
                double mr = M_re[i*n+k], mi = M_im[i*n+k];
                double vr = V_re[k*n+j], vi = V_im[k*n+j];
                sr += mr*vr - mi*vi;
                si += mr*vi + mi*vr;
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

#endif /* TENSOR_SVD_H */
