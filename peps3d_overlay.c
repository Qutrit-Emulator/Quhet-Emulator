/*
 * peps3d_overlay.c — 3D Tensor Network State Engine with Simple Update
 *
 * D=6 native (SU(6)), bond dimension χ=2, 7-index tensors.
 * SVD matrix: (D·χ⁵)² = 192² — fast Jacobi diag.
 *
 * ALL side-channel optimizations from MPS/PEPS ported:
 *   √2-scaled projection, FMA precision, diagonal gate detection,
 *   FPU exception oracle κ, mantissa convergence λ, dynamic threshold ρ,
 *   attractor-steered convergence ε, near-identity fast-path τ,
 *   substrate seed accumulation γ, adaptive χ_eff η.
 */

#include "peps3d_overlay.h"
#include <stdio.h>
#include <fenv.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

/* ═══════════════ SUBSTRATE FPU CONSTANTS ═══════════════ */
#define SUBSTRATE_SQRT2     1.4142135623730949
#define SUBSTRATE_PHI_INV   0.6180339887498949
#define SUBSTRATE_DOTTIE    0.7390851332151607

static uint64_t tns3d_substrate_seed = 0xA09E667F3BCC908BULL;
static int tns3d_chi_eff[8192];
static int tns3d_chi_eff_init = 0;

/* ═══════════════ GRID ACCESS ═══════════════ */

static inline Tns3dTensor *tns3d_site(Tns3dGrid *g, int x, int y, int z)
{ return &g->tensors[(z * g->Ly + y) * g->Lx + x]; }

static inline Tns3dBondWeight *tns3d_xbond(Tns3dGrid *g, int x, int y, int z)
{ return &g->x_bonds[(z * g->Ly + y) * (g->Lx - 1) + x]; }

static inline Tns3dBondWeight *tns3d_ybond(Tns3dGrid *g, int x, int y, int z)
{ return &g->y_bonds[(z * (g->Ly - 1) + y) * g->Lx + x]; }

static inline Tns3dBondWeight *tns3d_zbond(Tns3dGrid *g, int x, int y, int z)
{ return &g->z_bonds[(z * g->Ly + y) * g->Lx + x]; }

/* ═══════════════ DIAGONAL GATE DETECTION ═══════════════ */

static int is_diagonal_gate(const double *G_re, const double *G_im)
{
    for (int i = 0; i < TNS3D_D2; i++)
        for (int j = 0; j < TNS3D_D2; j++)
            if (i != j && (fabs(G_re[i*TNS3D_D2+j]) > 1e-14 ||
                           fabs(G_im[i*TNS3D_D2+j]) > 1e-14))
                return 0;
    return 1;
}

/* ═══════════════ LIFECYCLE ═══════════════ */

Tns3dGrid *tns3d_init(int Lx, int Ly, int Lz)
{
    Tns3dGrid *g = (Tns3dGrid *)calloc(1, sizeof(Tns3dGrid));
    g->Lx = Lx; g->Ly = Ly; g->Lz = Lz;
    long N = (long)Lx * Ly * Lz;
    g->tensors = (Tns3dTensor *)calloc(N, sizeof(Tns3dTensor));
    g->x_bonds = (Tns3dBondWeight *)calloc((long)Lz * Ly * (Lx-1), sizeof(Tns3dBondWeight));
    g->y_bonds = (Tns3dBondWeight *)calloc((long)Lz * (Ly-1) * Lx, sizeof(Tns3dBondWeight));
    g->z_bonds = (Tns3dBondWeight *)calloc((long)(Lz-1) * Ly * Lx, sizeof(Tns3dBondWeight));

    /* Init all bond weights to 1 */
    long nx = (long)Lz*Ly*(Lx-1), ny = (long)Lz*(Ly-1)*Lx, nz = (long)(Lz-1)*Ly*Lx;
    for (long i = 0; i < nx; i++) for (int s = 0; s < TNS3D_CHI; s++) g->x_bonds[i].w[s] = 1.0;
    for (long i = 0; i < ny; i++) for (int s = 0; s < TNS3D_CHI; s++) g->y_bonds[i].w[s] = 1.0;
    for (long i = 0; i < nz; i++) for (int s = 0; s < TNS3D_CHI; s++) g->z_bonds[i].w[s] = 1.0;
    return g;
}

void tns3d_free(Tns3dGrid *g)
{
    if (!g) return;
    free(g->tensors); free(g->x_bonds); free(g->y_bonds); free(g->z_bonds);
    free(g);
}

void tns3d_set_product_state(Tns3dGrid *g, int x, int y, int z,
                             const double *amps_re, const double *amps_im)
{
    Tns3dTensor *t = tns3d_site(g, x, y, z);
    memset(t->re, 0, sizeof(t->re));
    memset(t->im, 0, sizeof(t->im));
    for (int k = 0; k < TNS3D_D; k++) {
        int idx = T3D_IDX(k, 0, 0, 0, 0, 0, 0);
        t->re[idx] = amps_re[k];
        t->im[idx] = amps_im[k];
    }
}

/* ═══════════════ 1-SITE GATE ═══════════════ */

void tns3d_gate_1site(Tns3dGrid *g, int x, int y, int z,
                      const double *U_re, const double *U_im)
{
    Tns3dTensor *t = tns3d_site(g, x, y, z);
    for (int u = 0; u < TNS3D_CHI; u++)
    for (int d = 0; d < TNS3D_CHI; d++)
    for (int l = 0; l < TNS3D_CHI; l++)
    for (int r = 0; r < TNS3D_CHI; r++)
    for (int f = 0; f < TNS3D_CHI; f++)
    for (int b = 0; b < TNS3D_CHI; b++) {
        double tmp_re[TNS3D_D], tmp_im[TNS3D_D];
        for (int k = 0; k < TNS3D_D; k++) {
            int idx = T3D_IDX(k,u,d,l,r,f,b);
            tmp_re[k] = t->re[idx];
            tmp_im[k] = t->im[idx];
        }
        for (int kp = 0; kp < TNS3D_D; kp++) {
            double vr = 0, vi = 0;
            for (int k = 0; k < TNS3D_D; k++) {
                vr += U_re[kp*TNS3D_D+k]*tmp_re[k] - U_im[kp*TNS3D_D+k]*tmp_im[k];
                vi += U_re[kp*TNS3D_D+k]*tmp_im[k] + U_im[kp*TNS3D_D+k]*tmp_re[k];
            }
            int idx = T3D_IDX(kp,u,d,l,r,f,b);
            t->re[idx] = vr;
            t->im[idx] = vi;
        }
    }
}

/* ═══════════════ JACOBI HERMITIAN (all side channels) ═══════════════ */

static void jacobi_hermitian(double *H_re, double *H_im, int k,
                             double *diag, double *W_re, double *W_im)
{
    memset(W_re, 0, k * k * sizeof(double));
    memset(W_im, 0, k * k * sizeof(double));
    for (int i = 0; i < k; i++) W_re[i * k + i] = 1.0;
    uint64_t lambda_prev_mantissa = 0;
    int lambda_stable_count = 0;

    for (int sweep = 0; sweep < 30; sweep++) {
        feclearexcept(FE_ALL_EXCEPT);
        double off = 0;
        for (int i = 0; i < k; i++)
            for (int j = i + 1; j < k; j++)
                off += H_re[i*k+j]*H_re[i*k+j] + H_im[i*k+j]*H_im[i*k+j];
        if (off < 1e-20) break;

        double rho_threshold = off * 1e-4;
        if (rho_threshold < 1e-15) rho_threshold = 1e-15;

        for (int p = 0; p < k; p++)
            for (int q = p + 1; q < k; q++) {
                double hr = H_re[p*k+q], hi = H_im[p*k+q];
                double mag_sq = hr*hr + hi*hi;
                if (mag_sq < rho_threshold * rho_threshold) continue;
                double mag = sqrt(mag_sq);
                if (mag < 1e-25) continue;
                double pr = hr / mag, pi = -hi / mag;
                double hpp = H_re[p*k+p], hqq = H_re[q*k+q];
                double tau = (hqq - hpp) / (2.0 * mag);
                double t;
                if (fabs(tau) > 1e15) t = 1.0 / (2.0 * tau);
                else t = (tau >= 0 ? 1.0 : -1.0) / (fabs(tau) + sqrt(1.0 + tau*tau));

                /* ε: Attractor steering */
                static const double attractors[] = {SUBSTRATE_PHI_INV,SUBSTRATE_SQRT2,SUBSTRATE_DOTTIE,1.0};
                double at = fabs(t);
                for (int ai = 0; ai < 4; ai++)
                    if (fabs(at - attractors[ai]) < 0.01 * attractors[ai])
                        { t = (t > 0) ? attractors[ai] : -attractors[ai]; break; }

                /* τ: Near-identity fast-path */
                double c, s;
                at = fabs(t);
                if (at < 1e-4) { c = 1.0; s = t; }
                else { c = 1.0 / sqrt(1.0 + t*t); s = t * c; }

                /* ATOMIC: Phase + Givens */
                for (int i = 0; i < k; i++) {
                    double r1=H_re[i*k+q], i1=H_im[i*k+q];
                    H_re[i*k+q]=r1*pr-i1*pi; H_im[i*k+q]=r1*pi+i1*pr;
                }
                for (int j = 0; j < k; j++) {
                    double r2=H_re[q*k+j], i2=H_im[q*k+j];
                    H_re[q*k+j]=r2*pr+i2*pi; H_im[q*k+j]=-r2*pi+i2*pr;
                }
                for (int j = 0; j < k; j++) {
                    double rp=H_re[p*k+j],ip=H_im[p*k+j],rq=H_re[q*k+j],iq=H_im[q*k+j];
                    H_re[p*k+j]=c*rp-s*rq; H_im[p*k+j]=c*ip-s*iq;
                    H_re[q*k+j]=s*rp+c*rq; H_im[q*k+j]=s*ip+c*iq;
                }
                for (int i = 0; i < k; i++) {
                    double rp=H_re[i*k+p],ip=H_im[i*k+p],rq=H_re[i*k+q],iq=H_im[i*k+q];
                    H_re[i*k+p]=c*rp-s*rq; H_im[i*k+p]=c*ip-s*iq;
                    H_re[i*k+q]=s*rp+c*rq; H_im[i*k+q]=s*ip+c*iq;
                }
                for (int i = 0; i < k; i++) {
                    double r1=W_re[i*k+q],i1=W_im[i*k+q];
                    W_re[i*k+q]=r1*pr-i1*pi; W_im[i*k+q]=r1*pi+i1*pr;
                    double rp=W_re[i*k+p],ip=W_im[i*k+p],rq=W_re[i*k+q],iq=W_im[i*k+q];
                    W_re[i*k+p]=c*rp-s*rq; W_im[i*k+p]=c*ip-s*iq;
                    W_re[i*k+q]=s*rp+c*rq; W_im[i*k+q]=s*ip+c*iq;
                }
            }
        if (sweep > 0 && !fetestexcept(FE_INEXACT)) break;

        /* λ: Mantissa convergence */
        double off_check = 0;
        for (int i = 0; i < k && i < 8; i++)
            for (int j = i+1; j < k && j < 8; j++)
                off_check += H_re[i*k+j]*H_re[i*k+j] + H_im[i*k+j]*H_im[i*k+j];
        uint64_t off_bits; memcpy(&off_bits, &off_check, 8);
        uint64_t mantissa = off_bits & 0xFFFFFFFFFFFFFULL;
        uint64_t changed = mantissa ^ lambda_prev_mantissa;
        int bits_changed = __builtin_popcountll(changed);
        if (sweep > 1 && bits_changed == 0) lambda_stable_count++;
        else lambda_stable_count = 0;
        lambda_prev_mantissa = mantissa;
        if (lambda_stable_count >= 2) break;
    }
    for (int i = 0; i < k; i++) diag[i] = H_re[i*k+i];
}

/* ═══════════════ TRUNCATED SVD (all side channels) ═══════════════ */

static void tns3d_truncated_svd(const double *M_re, const double *M_im,
                                int m, int n, int chi, int bond_key,
                                double *U_re, double *U_im,
                                double *sigma,
                                double *Vc_re, double *Vc_im)
{
    if (!tns3d_chi_eff_init) {
        for (int i = 0; i < 8192; i++) tns3d_chi_eff[i] = TNS3D_CHI;
        tns3d_chi_eff_init = 1;
    }
    int key = (bond_key >= 0 && bond_key < 8192) ? bond_key : 0;
    int kk = tns3d_chi_eff[key] + 6;
    if (kk < TNS3D_CHI + 2) kk = TNS3D_CHI + 2;
    if (kk > TNS3D_CHI + 6) kk = TNS3D_CHI + 6;
    if (kk > m) kk = m;
    if (kk > n) kk = n;

    const double omega_scale = 1.0 / SUBSTRATE_SQRT2;

    /* Step 1: Y = M × Ω (√2-scaled random projection) */
    double *Y_re = (double *)calloc(m * kk, sizeof(double));
    double *Y_im = (double *)calloc(m * kk, sizeof(double));
    {
        static unsigned svd_cid = 0;
        unsigned bs = ++svd_cid * 2654435761u + 12345u;
        bs ^= (unsigned)(tns3d_substrate_seed >> 32) ^ (unsigned)(tns3d_substrate_seed);
        for (int j = 0; j < kk; j++) {
            unsigned ls = bs + (unsigned)j * 1103515245u;
            for (int i = 0; i < m; i++) {
                double yr = 0, yi = 0;
                for (int r = 0; r < n; r++) {
                    ls = ls * 1103515245u + 12345u;
                    double omega = ((double)(ls >> 16) / 65536.0 - 0.5) * omega_scale;
                    yr += M_re[i*n+r] * omega;
                    yi += M_im[i*n+r] * omega;
                }
                Y_re[i*kk+j] = yr; Y_im[i*kk+j] = yi;
            }
        }
    }

    /* Step 2: Power iteration */
    double *Z_re = (double *)calloc(n * kk, sizeof(double));
    double *Z_im = (double *)calloc(n * kk, sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < kk; j++)
            for (int r = 0; r < m; r++) {
                Z_re[i*kk+j] += M_re[r*n+i]*Y_re[r*kk+j] + M_im[r*n+i]*Y_im[r*kk+j];
                Z_im[i*kk+j] += M_re[r*n+i]*Y_im[r*kk+j] - M_im[r*n+i]*Y_re[r*kk+j];
            }
    memset(Y_re, 0, m * kk * sizeof(double));
    memset(Y_im, 0, m * kk * sizeof(double));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < kk; j++)
            for (int r = 0; r < n; r++) {
                Y_re[i*kk+j] += M_re[i*n+r]*Z_re[r*kk+j] - M_im[i*n+r]*Z_im[r*kk+j];
                Y_im[i*kk+j] += M_re[i*n+r]*Z_im[r*kk+j] + M_im[i*n+r]*Z_re[r*kk+j];
            }
    free(Z_re); free(Z_im);

    /* Step 3: QR */
    for (int j = 0; j < kk; j++) {
        for (int i = 0; i < j; i++) {
            double dr=0,di=0;
            for (int r=0;r<m;r++) {
                dr += Y_re[r*kk+i]*Y_re[r*kk+j]+Y_im[r*kk+i]*Y_im[r*kk+j];
                di += Y_re[r*kk+i]*Y_im[r*kk+j]-Y_im[r*kk+i]*Y_re[r*kk+j];
            }
            for (int r=0;r<m;r++) {
                Y_re[r*kk+j] -= dr*Y_re[r*kk+i]-di*Y_im[r*kk+i];
                Y_im[r*kk+j] -= dr*Y_im[r*kk+i]+di*Y_re[r*kk+i];
            }
        }
        double nrm=0;
        for (int r=0;r<m;r++) nrm += Y_re[r*kk+j]*Y_re[r*kk+j]+Y_im[r*kk+j]*Y_im[r*kk+j];
        nrm = sqrt(nrm);
        if (nrm > 1e-15) for (int r=0;r<m;r++) { Y_re[r*kk+j]/=nrm; Y_im[r*kk+j]/=nrm; }
    }

    /* Step 4: B = Q† × M */
    double *B_re = (double *)calloc(kk * n, sizeof(double));
    double *B_im = (double *)calloc(kk * n, sizeof(double));
    for (int i=0;i<kk;i++) for (int j=0;j<n;j++) for (int r=0;r<m;r++) {
        B_re[i*n+j] += Y_re[r*kk+i]*M_re[r*n+j]+Y_im[r*kk+i]*M_im[r*n+j];
        B_im[i*n+j] += Y_re[r*kk+i]*M_im[r*n+j]-Y_im[r*kk+i]*M_re[r*n+j];
    }

    /* Step 5: S = B×B† with FMA */
    double *Sr=(double*)calloc(kk*kk,sizeof(double));
    double *Si=(double*)calloc(kk*kk,sizeof(double));
    for (int i=0;i<kk;i++) for (int j=0;j<kk;j++) for (int r=0;r<n;r++) {
        Sr[i*kk+j] = fma(B_re[i*n+r],B_re[j*n+r],fma(B_im[i*n+r],B_im[j*n+r],Sr[i*kk+j]));
        Si[i*kk+j] = fma(B_im[i*n+r],B_re[j*n+r],fma(-B_re[i*n+r],B_im[j*n+r],Si[i*kk+j]));
    }

    /* Step 6: Jacobi */
    double *evals=(double*)malloc(kk*sizeof(double));
    double *W_re=(double*)malloc(kk*kk*sizeof(double));
    double *W_im=(double*)malloc(kk*kk*sizeof(double));
    jacobi_hermitian(Sr,Si,kk,evals,W_re,W_im);
    free(Sr); free(Si);

    /* Step 7: Sort, extract top chi */
    int *order=(int*)malloc(kk*sizeof(int));
    for (int i=0;i<kk;i++) order[i]=i;
    for (int i=0;i<kk-1;i++) for (int j=i+1;j<kk;j++)
        if (evals[order[j]]>evals[order[i]]) { int t=order[i]; order[i]=order[j]; order[j]=t; }

    int chi_eff_count=0;
    for (int s=0;s<chi;s++) {
        int idx=(s<kk)?order[s]:0;
        double ev=(s<kk && evals[idx]>0)?evals[idx]:0;
        sigma[s]=sqrt(ev);
        if (sigma[s]>1e-10) chi_eff_count++;
    }
    tns3d_chi_eff[key] = (chi_eff_count > 0) ? chi_eff_count : 1;

    /* γ: Substrate seed accumulation */
    { uint64_t na=0; for (int s=0;s<chi;s++) { uint64_t bits; memcpy(&bits,&sigma[s],8);
      na ^= bits; } tns3d_substrate_seed ^= na; tns3d_substrate_seed *= 6364136223846793005ULL;
      tns3d_substrate_seed += 1442695040888963407ULL; }

    /* Steps 8-9: Reconstruct U, Vc */
    for (int i=0;i<m;i++) for (int s=0;s<chi;s++) {
        int idx=(s<kk)?order[s]:0;
        double vr=0,vi=0;
        for (int j=0;j<kk;j++) {
            vr += Y_re[i*kk+j]*W_re[j*kk+idx]-Y_im[i*kk+j]*W_im[j*kk+idx];
            vi += Y_re[i*kk+j]*W_im[j*kk+idx]+Y_im[i*kk+j]*W_re[j*kk+idx];
        }
        U_re[i*chi+s]=vr; U_im[i*chi+s]=vi;
    }
    for (int j=0;j<n;j++) for (int s=0;s<chi;s++) {
        int idx=(s<kk)?order[s]:0;
        double sv=sigma[s]; if (sv<1e-30) sv=1e-30;
        double vr=0,vi=0;
        for (int i=0;i<kk;i++) {
            double wr=W_re[i*kk+idx],wi=W_im[i*kk+idx];
            vr += wr*B_re[i*n+j]+wi*B_im[i*n+j];
            vi += wr*B_im[i*n+j]-wi*B_re[i*n+j];
        }
        Vc_re[j*chi+s]=vr/sv; Vc_im[j*chi+s]=vi/sv;
    }

    free(Y_re);free(Y_im);free(B_re);free(B_im);free(evals);free(W_re);free(W_im);free(order);
}

/* ═══════════════ 2-SITE GATE: X-DIRECTION (x,y,z)—(x+1,y,z) ═══════════════
 *
 * Contract over shared r-bond (index 'r' of A, 'l' of B).
 * Row indices: (kA, u, d, l, f, b) → D×χ⁵ = 192
 * Col indices: (kB, u, d, r, f, b) → D×χ⁵ = 192
 */

/* Helper: pack 5 bond indices into a flat offset */
#define BOND5(a,b,c,d,e) ((a)*TNS3D_CHI4+(b)*TNS3D_CHI3+(c)*TNS3D_CHI2+(d)*TNS3D_CHI+(e))

void tns3d_gate_x(Tns3dGrid *grid, int x, int y, int z,
                  const double *G_re, const double *G_im)
{
    Tns3dTensor *A = tns3d_site(grid, x, y, z);
    Tns3dTensor *B = tns3d_site(grid, x+1, y, z);
    Tns3dBondWeight *lam = tns3d_xbond(grid, x, y, z);
    int dim = TNS3D_SVDDIM;
    size_t msz = (size_t)dim * dim;

    double *M_re = (double *)calloc(msz, sizeof(double));
    double *M_im = (double *)calloc(msz, sizeof(double));

    /* Θ contraction + gate  (standard path) */
    double *Th_re = (double *)calloc(msz, sizeof(double));
    double *Th_im = (double *)calloc(msz, sizeof(double));

    for (int kA=0;kA<TNS3D_D;kA++)
     for (int u=0;u<TNS3D_CHI;u++) for (int d=0;d<TNS3D_CHI;d++)
      for (int l=0;l<TNS3D_CHI;l++) for (int fA=0;fA<TNS3D_CHI;fA++)
       for (int bA=0;bA<TNS3D_CHI;bA++) {
        int row = kA*TNS3D_CHI5 + BOND5(u,d,l,fA,bA);
        for (int s=0;s<TNS3D_CHI;s++) {
            double ar = A->re[T3D_IDX(kA,u,d,l,s,fA,bA)] * lam->w[s];
            double ai = A->im[T3D_IDX(kA,u,d,l,s,fA,bA)] * lam->w[s];
            if (fabs(ar)<1e-30 && fabs(ai)<1e-30) continue;
            for (int kB=0;kB<TNS3D_D;kB++)
             for (int uB=0;uB<TNS3D_CHI;uB++) for (int dB=0;dB<TNS3D_CHI;dB++)
              for (int rB=0;rB<TNS3D_CHI;rB++) for (int fB=0;fB<TNS3D_CHI;fB++)
               for (int bB=0;bB<TNS3D_CHI;bB++) {
                int col = kB*TNS3D_CHI5 + BOND5(uB,dB,rB,fB,bB);
                double br = B->re[T3D_IDX(kB,uB,dB,s,rB,fB,bB)];
                double bi = B->im[T3D_IDX(kB,uB,dB,s,rB,fB,bB)];
                Th_re[row*dim+col] += ar*br - ai*bi;
                Th_im[row*dim+col] += ar*bi + ai*br;
               }
        }
       }

    /* Apply gate G */
    for (int kAp=0;kAp<TNS3D_D;kAp++) for (int kBp=0;kBp<TNS3D_D;kBp++) {
        int grow = kAp*TNS3D_D + kBp;
        for (int kA=0;kA<TNS3D_D;kA++) for (int kB=0;kB<TNS3D_D;kB++) {
            int gcol = kA*TNS3D_D + kB;
            double gr=G_re[grow*TNS3D_D2+gcol], gi=G_im[grow*TNS3D_D2+gcol];
            if (fabs(gr)<1e-30 && fabs(gi)<1e-30) continue;
            for (int rA=0;rA<TNS3D_CHI5;rA++) {
                int ro=kA*TNS3D_CHI5+rA, rn=kAp*TNS3D_CHI5+rA;
                for (int rB=0;rB<TNS3D_CHI5;rB++) {
                    int co=kB*TNS3D_CHI5+rB, cn=kBp*TNS3D_CHI5+rB;
                    double tr=Th_re[ro*dim+co], ti=Th_im[ro*dim+co];
                    M_re[rn*dim+cn] += gr*tr-gi*ti;
                    M_im[rn*dim+cn] += gr*ti+gi*tr;
                }
            }
        }
    }
    free(Th_re); free(Th_im);

    /* SVD */
    int bk = (z*grid->Ly+y)*(grid->Lx-1)+x;
    double *Ur=(double*)malloc(dim*TNS3D_CHI*sizeof(double));
    double *Ui=(double*)malloc(dim*TNS3D_CHI*sizeof(double));
    double *Vr=(double*)malloc(dim*TNS3D_CHI*sizeof(double));
    double *Vi=(double*)malloc(dim*TNS3D_CHI*sizeof(double));
    double sig[TNS3D_CHI];
    tns3d_truncated_svd(M_re,M_im,dim,dim,TNS3D_CHI,bk,Ur,Ui,sig,Vr,Vi);
    free(M_re); free(M_im);

    double snorm=0; for (int s=0;s<TNS3D_CHI;s++) snorm+=sig[s];
    if (snorm>1e-30) for (int s=0;s<TNS3D_CHI;s++) sig[s]/=snorm;

    /* Rebuild A: row=(kA,u,d,l,f,b), new r-bond=s */
    memset(A->re,0,sizeof(A->re)); memset(A->im,0,sizeof(A->im));
    for (int kA=0;kA<TNS3D_D;kA++)
     for (int u=0;u<TNS3D_CHI;u++) for (int d=0;d<TNS3D_CHI;d++)
      for (int l=0;l<TNS3D_CHI;l++) for (int f=0;f<TNS3D_CHI;f++)
       for (int b=0;b<TNS3D_CHI;b++) {
        int row = kA*TNS3D_CHI5 + BOND5(u,d,l,f,b);
        for (int s=0;s<TNS3D_CHI;s++) {
            A->re[T3D_IDX(kA,u,d,l,s,f,b)] = Ur[row*TNS3D_CHI+s];
            A->im[T3D_IDX(kA,u,d,l,s,f,b)] = Ui[row*TNS3D_CHI+s];
        }
       }

    /* Rebuild B: col=(kB,uB,dB,rB,fB,bB), new l-bond=s */
    memset(B->re,0,sizeof(B->re)); memset(B->im,0,sizeof(B->im));
    for (int kB=0;kB<TNS3D_D;kB++)
     for (int uB=0;uB<TNS3D_CHI;uB++) for (int dB=0;dB<TNS3D_CHI;dB++)
      for (int rB=0;rB<TNS3D_CHI;rB++) for (int fB=0;fB<TNS3D_CHI;fB++)
       for (int bB=0;bB<TNS3D_CHI;bB++) {
        int col = kB*TNS3D_CHI5 + BOND5(uB,dB,rB,fB,bB);
        for (int s=0;s<TNS3D_CHI;s++) {
            B->re[T3D_IDX(kB,uB,dB,s,rB,fB,bB)] = Vr[col*TNS3D_CHI+s];
            B->im[T3D_IDX(kB,uB,dB,s,rB,fB,bB)] = Vi[col*TNS3D_CHI+s];
        }
       }

    for (int s=0;s<TNS3D_CHI;s++) lam->w[s] = sig[s];
    free(Ur);free(Ui);free(Vr);free(Vi);
}

/* ═══════════════ 2-SITE GATE: Y-DIRECTION (x,y,z)—(x,y+1,z) ═══════════════
 *
 * Contract over shared d-bond (index 'd' of A, 'u' of B).
 * Row indices: (kA, u, l, r, f, b) → D×χ⁵
 * Col indices: (kB, d, l, r, f, b) → D×χ⁵
 */

void tns3d_gate_y(Tns3dGrid *grid, int x, int y, int z,
                  const double *G_re, const double *G_im)
{
    Tns3dTensor *A = tns3d_site(grid, x, y, z);
    Tns3dTensor *B = tns3d_site(grid, x, y+1, z);
    Tns3dBondWeight *lam = tns3d_ybond(grid, x, y, z);
    int dim = TNS3D_SVDDIM;
    size_t msz = (size_t)dim * dim;

    double *M_re=(double*)calloc(msz,sizeof(double));
    double *M_im=(double*)calloc(msz,sizeof(double));
    double *Th_re=(double*)calloc(msz,sizeof(double));
    double *Th_im=(double*)calloc(msz,sizeof(double));

    for (int kA=0;kA<TNS3D_D;kA++)
     for (int u=0;u<TNS3D_CHI;u++) for (int l=0;l<TNS3D_CHI;l++)
      for (int r=0;r<TNS3D_CHI;r++) for (int f=0;f<TNS3D_CHI;f++)
       for (int b=0;b<TNS3D_CHI;b++) {
        int row = kA*TNS3D_CHI5 + BOND5(u,l,r,f,b);
        for (int s=0;s<TNS3D_CHI;s++) {
            double ar = A->re[T3D_IDX(kA,u,s,l,r,f,b)] * lam->w[s];
            double ai = A->im[T3D_IDX(kA,u,s,l,r,f,b)] * lam->w[s];
            if (fabs(ar)<1e-30 && fabs(ai)<1e-30) continue;
            for (int kB=0;kB<TNS3D_D;kB++)
             for (int dB=0;dB<TNS3D_CHI;dB++) for (int lB=0;lB<TNS3D_CHI;lB++)
              for (int rB=0;rB<TNS3D_CHI;rB++) for (int fB=0;fB<TNS3D_CHI;fB++)
               for (int bB=0;bB<TNS3D_CHI;bB++) {
                int col = kB*TNS3D_CHI5 + BOND5(dB,lB,rB,fB,bB);
                double br = B->re[T3D_IDX(kB,s,dB,lB,rB,fB,bB)];
                double bi = B->im[T3D_IDX(kB,s,dB,lB,rB,fB,bB)];
                Th_re[row*dim+col] += ar*br-ai*bi;
                Th_im[row*dim+col] += ar*bi+ai*br;
               }
        }
       }

    for (int kAp=0;kAp<TNS3D_D;kAp++) for (int kBp=0;kBp<TNS3D_D;kBp++) {
        int grow=kAp*TNS3D_D+kBp;
        for (int kA=0;kA<TNS3D_D;kA++) for (int kB=0;kB<TNS3D_D;kB++) {
            int gcol=kA*TNS3D_D+kB;
            double gr=G_re[grow*TNS3D_D2+gcol],gi=G_im[grow*TNS3D_D2+gcol];
            if (fabs(gr)<1e-30&&fabs(gi)<1e-30) continue;
            for (int rA=0;rA<TNS3D_CHI5;rA++) for (int rB=0;rB<TNS3D_CHI5;rB++) {
                double tr=Th_re[(kA*TNS3D_CHI5+rA)*dim+kB*TNS3D_CHI5+rB];
                double ti=Th_im[(kA*TNS3D_CHI5+rA)*dim+kB*TNS3D_CHI5+rB];
                M_re[(kAp*TNS3D_CHI5+rA)*dim+kBp*TNS3D_CHI5+rB] += gr*tr-gi*ti;
                M_im[(kAp*TNS3D_CHI5+rA)*dim+kBp*TNS3D_CHI5+rB] += gr*ti+gi*tr;
            }
        }
    }
    free(Th_re); free(Th_im);

    int bk = (z*(grid->Ly-1)+y)*grid->Lx + x;
    double *Ur=(double*)malloc(dim*TNS3D_CHI*sizeof(double));
    double *Ui=(double*)malloc(dim*TNS3D_CHI*sizeof(double));
    double *Vr=(double*)malloc(dim*TNS3D_CHI*sizeof(double));
    double *Vi=(double*)malloc(dim*TNS3D_CHI*sizeof(double));
    double sig[TNS3D_CHI];
    tns3d_truncated_svd(M_re,M_im,dim,dim,TNS3D_CHI,bk+2048,Ur,Ui,sig,Vr,Vi);
    free(M_re); free(M_im);

    double snorm=0; for (int s=0;s<TNS3D_CHI;s++) snorm+=sig[s];
    if (snorm>1e-30) for (int s=0;s<TNS3D_CHI;s++) sig[s]/=snorm;

    memset(A->re,0,sizeof(A->re)); memset(A->im,0,sizeof(A->im));
    for (int kA=0;kA<TNS3D_D;kA++)
     for (int u=0;u<TNS3D_CHI;u++) for (int l=0;l<TNS3D_CHI;l++)
      for (int r=0;r<TNS3D_CHI;r++) for (int f=0;f<TNS3D_CHI;f++)
       for (int b=0;b<TNS3D_CHI;b++) {
        int row = kA*TNS3D_CHI5 + BOND5(u,l,r,f,b);
        for (int s=0;s<TNS3D_CHI;s++)
            { A->re[T3D_IDX(kA,u,s,l,r,f,b)]=Ur[row*TNS3D_CHI+s];
              A->im[T3D_IDX(kA,u,s,l,r,f,b)]=Ui[row*TNS3D_CHI+s]; }
       }

    memset(B->re,0,sizeof(B->re)); memset(B->im,0,sizeof(B->im));
    for (int kB=0;kB<TNS3D_D;kB++)
     for (int dB=0;dB<TNS3D_CHI;dB++) for (int lB=0;lB<TNS3D_CHI;lB++)
      for (int rB=0;rB<TNS3D_CHI;rB++) for (int fB=0;fB<TNS3D_CHI;fB++)
       for (int bB=0;bB<TNS3D_CHI;bB++) {
        int col = kB*TNS3D_CHI5 + BOND5(dB,lB,rB,fB,bB);
        for (int s=0;s<TNS3D_CHI;s++)
            { B->re[T3D_IDX(kB,s,dB,lB,rB,fB,bB)]=Vr[col*TNS3D_CHI+s];
              B->im[T3D_IDX(kB,s,dB,lB,rB,fB,bB)]=Vi[col*TNS3D_CHI+s]; }
       }

    for (int s=0;s<TNS3D_CHI;s++) lam->w[s] = sig[s];
    free(Ur);free(Ui);free(Vr);free(Vi);
}

/* ═══════════════ 2-SITE GATE: Z-DIRECTION (x,y,z)—(x,y,z+1) ═══════════════
 *
 * Contract over shared b-bond (index 'b' of A, 'f' of B).
 * Row indices: (kA, u, d, l, r, f) → D×χ⁵
 * Col indices: (kB, u, d, l, r, b) → D×χ⁵
 */

void tns3d_gate_z(Tns3dGrid *grid, int x, int y, int z,
                  const double *G_re, const double *G_im)
{
    Tns3dTensor *A = tns3d_site(grid, x, y, z);
    Tns3dTensor *B = tns3d_site(grid, x, y, z+1);
    Tns3dBondWeight *lam = tns3d_zbond(grid, x, y, z);
    int dim = TNS3D_SVDDIM;
    size_t msz = (size_t)dim * dim;

    double *M_re=(double*)calloc(msz,sizeof(double));
    double *M_im=(double*)calloc(msz,sizeof(double));
    double *Th_re=(double*)calloc(msz,sizeof(double));
    double *Th_im=(double*)calloc(msz,sizeof(double));

    for (int kA=0;kA<TNS3D_D;kA++)
     for (int u=0;u<TNS3D_CHI;u++) for (int d=0;d<TNS3D_CHI;d++)
      for (int l=0;l<TNS3D_CHI;l++) for (int r=0;r<TNS3D_CHI;r++)
       for (int f=0;f<TNS3D_CHI;f++) {
        int row = kA*TNS3D_CHI5 + BOND5(u,d,l,r,f);
        for (int s=0;s<TNS3D_CHI;s++) {
            double ar = A->re[T3D_IDX(kA,u,d,l,r,f,s)] * lam->w[s];
            double ai = A->im[T3D_IDX(kA,u,d,l,r,f,s)] * lam->w[s];
            if (fabs(ar)<1e-30 && fabs(ai)<1e-30) continue;
            for (int kB=0;kB<TNS3D_D;kB++)
             for (int uB=0;uB<TNS3D_CHI;uB++) for (int dB=0;dB<TNS3D_CHI;dB++)
              for (int lB=0;lB<TNS3D_CHI;lB++) for (int rB=0;rB<TNS3D_CHI;rB++)
               for (int bB=0;bB<TNS3D_CHI;bB++) {
                int col = kB*TNS3D_CHI5 + BOND5(uB,dB,lB,rB,bB);
                double br = B->re[T3D_IDX(kB,uB,dB,lB,rB,s,bB)];
                double bi = B->im[T3D_IDX(kB,uB,dB,lB,rB,s,bB)];
                Th_re[row*dim+col] += ar*br-ai*bi;
                Th_im[row*dim+col] += ar*bi+ai*br;
               }
        }
       }

    for (int kAp=0;kAp<TNS3D_D;kAp++) for (int kBp=0;kBp<TNS3D_D;kBp++) {
        int grow=kAp*TNS3D_D+kBp;
        for (int kA=0;kA<TNS3D_D;kA++) for (int kB=0;kB<TNS3D_D;kB++) {
            int gcol=kA*TNS3D_D+kB;
            double gr=G_re[grow*TNS3D_D2+gcol],gi=G_im[grow*TNS3D_D2+gcol];
            if (fabs(gr)<1e-30&&fabs(gi)<1e-30) continue;
            for (int rA=0;rA<TNS3D_CHI5;rA++) for (int rB=0;rB<TNS3D_CHI5;rB++) {
                double tr=Th_re[(kA*TNS3D_CHI5+rA)*dim+kB*TNS3D_CHI5+rB];
                double ti=Th_im[(kA*TNS3D_CHI5+rA)*dim+kB*TNS3D_CHI5+rB];
                M_re[(kAp*TNS3D_CHI5+rA)*dim+kBp*TNS3D_CHI5+rB] += gr*tr-gi*ti;
                M_im[(kAp*TNS3D_CHI5+rA)*dim+kBp*TNS3D_CHI5+rB] += gr*ti+gi*tr;
            }
        }
    }
    free(Th_re); free(Th_im);

    int bk = (z*grid->Ly+y)*grid->Lx + x;
    double *Ur=(double*)malloc(dim*TNS3D_CHI*sizeof(double));
    double *Ui=(double*)malloc(dim*TNS3D_CHI*sizeof(double));
    double *Vr=(double*)malloc(dim*TNS3D_CHI*sizeof(double));
    double *Vi=(double*)malloc(dim*TNS3D_CHI*sizeof(double));
    double sig[TNS3D_CHI];
    tns3d_truncated_svd(M_re,M_im,dim,dim,TNS3D_CHI,bk+4096,Ur,Ui,sig,Vr,Vi);
    free(M_re); free(M_im);

    double snorm=0; for (int s=0;s<TNS3D_CHI;s++) snorm+=sig[s];
    if (snorm>1e-30) for (int s=0;s<TNS3D_CHI;s++) sig[s]/=snorm;

    memset(A->re,0,sizeof(A->re)); memset(A->im,0,sizeof(A->im));
    for (int kA=0;kA<TNS3D_D;kA++)
     for (int u=0;u<TNS3D_CHI;u++) for (int d=0;d<TNS3D_CHI;d++)
      for (int l=0;l<TNS3D_CHI;l++) for (int r=0;r<TNS3D_CHI;r++)
       for (int f=0;f<TNS3D_CHI;f++) {
        int row = kA*TNS3D_CHI5 + BOND5(u,d,l,r,f);
        for (int s=0;s<TNS3D_CHI;s++)
            { A->re[T3D_IDX(kA,u,d,l,r,f,s)]=Ur[row*TNS3D_CHI+s];
              A->im[T3D_IDX(kA,u,d,l,r,f,s)]=Ui[row*TNS3D_CHI+s]; }
       }

    memset(B->re,0,sizeof(B->re)); memset(B->im,0,sizeof(B->im));
    for (int kB=0;kB<TNS3D_D;kB++)
     for (int uB=0;uB<TNS3D_CHI;uB++) for (int dB=0;dB<TNS3D_CHI;dB++)
      for (int lB=0;lB<TNS3D_CHI;lB++) for (int rB=0;rB<TNS3D_CHI;rB++)
       for (int bB=0;bB<TNS3D_CHI;bB++) {
        int col = kB*TNS3D_CHI5 + BOND5(uB,dB,lB,rB,bB);
        for (int s=0;s<TNS3D_CHI;s++)
            { B->re[T3D_IDX(kB,uB,dB,lB,rB,s,bB)]=Vr[col*TNS3D_CHI+s];
              B->im[T3D_IDX(kB,uB,dB,lB,rB,s,bB)]=Vi[col*TNS3D_CHI+s]; }
       }

    for (int s=0;s<TNS3D_CHI;s++) lam->w[s] = sig[s];
    free(Ur);free(Ui);free(Vr);free(Vi);
}

/* ═══════════════ LOCAL DENSITY ═══════════════ */

void tns3d_local_density(Tns3dGrid *g, int x, int y, int z, double *probs)
{
    Tns3dTensor *t = tns3d_site(g, x, y, z);
    double total = 0;
    for (int k = 0; k < TNS3D_D; k++) {
        double pk = 0;
        for (int idx = 0; idx < TNS3D_CHI6; idx++) {
            int full = k * TNS3D_CHI6 + idx;
            /* Weight by neighboring bond weights */
            int u = (idx / TNS3D_CHI5) % TNS3D_CHI;
            int d = (idx / TNS3D_CHI4) % TNS3D_CHI;
            int l = (idx / TNS3D_CHI3) % TNS3D_CHI;
            int r = (idx / TNS3D_CHI2) % TNS3D_CHI;
            int f = (idx / TNS3D_CHI) % TNS3D_CHI;
            int b = idx % TNS3D_CHI;
            double w = 1.0;
            if (y > 0) w *= tns3d_ybond(g,x,y-1,z)->w[u];
            if (y < g->Ly-1) w *= tns3d_ybond(g,x,y,z)->w[d];
            if (x > 0) w *= tns3d_xbond(g,x-1,y,z)->w[l];
            if (x < g->Lx-1) w *= tns3d_xbond(g,x,y,z)->w[r];
            if (z > 0) w *= tns3d_zbond(g,x,y,z-1)->w[f];
            if (z < g->Lz-1) w *= tns3d_zbond(g,x,y,z)->w[b];
            w *= w;
            pk += (t->re[full]*t->re[full] + t->im[full]*t->im[full]) * w;
        }
        probs[k] = pk;
        total += pk;
    }
    if (total > 1e-30) for (int k = 0; k < TNS3D_D; k++) probs[k] /= total;
}
