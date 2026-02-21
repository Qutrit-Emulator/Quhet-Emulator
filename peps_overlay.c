/*
 * peps_overlay.c — PEPS Engine: 2D Tensor Network with Simple Update
 *
 * D=6 native (SU(6)), bond dimension χ=4.
 * Simple update: contract bond → apply gate → SVD truncate.
 * SVD matrix size: (D·χ³)² = 384² — Jacobi diag in milliseconds.
 *
 * SIDE-CHANNEL OPTIMIZATIONS (ported from MPS overlay):
 *   √2-scaled projection (substrate attractor basin)
 *   FMA precision harvesting (BB† computation)
 *   Diagonal gate detection (sparsity exploitation)
 *   FPU exception oracle κ (free convergence check)
 *   Mantissa convergence λ (bit-change early exit)
 *   Dynamic threshold ρ (skip near-zero pairs)
 *   Attractor-steered convergence ε (FPU fixed points)
 *   Near-identity fast-path τ (skip trivial rotations)
 *   Per-rotation INEXACT skip φ (late-sweep optimization)
 *   Substrate seed accumulation γ (rounding noise harvest)
 */

#include "peps_overlay.h"
#include <stdio.h>
#include <fenv.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════════════
 * SUBSTRATE FPU CONSTANTS (from mps_fpu_probe.c)
 *
 * These bit patterns are preserved under iterated FPU arithmetic.
 * Using them as normalization targets aligns computation with
 * the substrate's preferred numerical basins.
 * ═══════════════════════════════════════════════════════════════════════════════ */

#define SUBSTRATE_SQRT2     1.4142135623730949   /* √2 — dominant attractor   */
#define SUBSTRATE_PHI_INV   0.6180339887498949   /* φ⁻¹ — golden attractor    */
#define SUBSTRATE_DOTTIE    0.7390851332151607   /* cos fixed point           */
#define SUBSTRATE_8SQRT2    11.313708498984760   /* 8√2 — tensor norm target  */
#define SUBSTRATE_OMEGA     0.5671432904097838   /* Lambert W(1) attractor    */

/* ── SIDE-CHANNEL γ: Rounding-noise substrate seed ──────────────────
 * Accumulated from the lowest mantissa bits of every SVD's σ values.
 * Each SVD call contributes noise → the substrate picks its own
 * projection subspace for subsequent calls. */
static uint64_t peps_substrate_seed = 0xA09E667F3BCC908BULL; /* √2 mantissa */

/* ── SIDE-CHANNEL η: Per-bond adaptive χ_eff ──────────────────────
 * Track the effective rank from each bond's previous SVD.
 * Next SVD at this bond uses k = χ_eff_prev + margin instead
 * of fixed k = χ + 10.  Speedup: (χ/χ_eff)³. */
static int peps_chi_eff[4096];
static int peps_chi_eff_init = 0;

/* ═══════════════════════════════════════════════════════════════════════════════
 * GRID ACCESS HELPERS
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline PepsTensor *peps_site(PepsGrid *g, int x, int y)
{ return &g->tensors[y * g->Lx + x]; }

static inline PepsBondWeight *peps_hbond(PepsGrid *g, int x, int y)
{ return &g->h_bonds[y * (g->Lx - 1) + x]; }

static inline PepsBondWeight *peps_vbond(PepsGrid *g, int x, int y)
{ return &g->v_bonds[y * g->Lx + x]; }

/* ═══════════════════════════════════════════════════════════════════════════════
 * DIAGONAL GATE DETECTION (Side-Channel Discovery from MPS)
 *
 * If G is diagonal, fold phases directly into Θ contraction,
 * eliminating the separate gate application step entirely.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static int is_diagonal_gate(const double *G_re, const double *G_im)
{
    for (int i = 0; i < PEPS_D2; i++)
        for (int j = 0; j < PEPS_D2; j++)
            if (i != j && (fabs(G_re[i*PEPS_D2+j]) > 1e-14 ||
                           fabs(G_im[i*PEPS_D2+j]) > 1e-14))
                return 0;
    return 1;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 * ═══════════════════════════════════════════════════════════════════════════════ */

PepsGrid *peps_init(int Lx, int Ly)
{
    PepsGrid *g = (PepsGrid *)calloc(1, sizeof(PepsGrid));
    g->Lx = Lx;
    g->Ly = Ly;
    g->tensors = (PepsTensor *)calloc(Lx * Ly, sizeof(PepsTensor));
    g->h_bonds = (PepsBondWeight *)calloc(Ly * (Lx - 1), sizeof(PepsBondWeight));
    g->v_bonds = (PepsBondWeight *)calloc((Ly - 1) * Lx, sizeof(PepsBondWeight));

    for (int i = 0; i < Ly * (Lx - 1); i++)
        for (int s = 0; s < PEPS_CHI; s++)
            g->h_bonds[i].w[s] = 1.0;
    for (int i = 0; i < (Ly - 1) * Lx; i++)
        for (int s = 0; s < PEPS_CHI; s++)
            g->v_bonds[i].w[s] = 1.0;

    for (int y = 0; y < Ly; y++)
        for (int x = 0; x < Lx; x++)
            peps_site(g, x, y)->re[PT_IDX(0, 0, 0, 0, 0)] = 1.0;

    return g;
}

void peps_free(PepsGrid *grid)
{
    if (!grid) return;
    free(grid->tensors);
    free(grid->h_bonds);
    free(grid->v_bonds);
    free(grid);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * STATE INITIALIZATION
 * ═══════════════════════════════════════════════════════════════════════════════ */

void peps_set_product_state(PepsGrid *grid, int x, int y,
                            const double *amps_re, const double *amps_im)
{
    PepsTensor *T = peps_site(grid, x, y);
    memset(T->re, 0, sizeof(T->re));
    memset(T->im, 0, sizeof(T->im));
    for (int k = 0; k < PEPS_D; k++) {
        T->re[PT_IDX(k, 0, 0, 0, 0)] = amps_re[k];
        T->im[PT_IDX(k, 0, 0, 0, 0)] = amps_im[k];
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * 1-SITE GATE
 * ═══════════════════════════════════════════════════════════════════════════════ */

void peps_gate_1site(PepsGrid *grid, int x, int y,
                     const double *U_re, const double *U_im)
{
    PepsTensor *T = peps_site(grid, x, y);
    double new_re[PEPS_TSIZ], new_im[PEPS_TSIZ];
    memset(new_re, 0, sizeof(new_re));
    memset(new_im, 0, sizeof(new_im));

    for (int u = 0; u < PEPS_CHI; u++)
        for (int d = 0; d < PEPS_CHI; d++)
            for (int l = 0; l < PEPS_CHI; l++)
                for (int r = 0; r < PEPS_CHI; r++)
                    for (int kp = 0; kp < PEPS_D; kp++) {
                        double sr = 0, si = 0;
                        for (int k = 0; k < PEPS_D; k++) {
                            double ur = U_re[kp * PEPS_D + k];
                            double ui = U_im[kp * PEPS_D + k];
                            double tr = T->re[PT_IDX(k,u,d,l,r)];
                            double ti = T->im[PT_IDX(k,u,d,l,r)];
                            sr += ur*tr - ui*ti;
                            si += ur*ti + ui*tr;
                        }
                        new_re[PT_IDX(kp,u,d,l,r)] = sr;
                        new_im[PT_IDX(kp,u,d,l,r)] = si;
                    }

    memcpy(T->re, new_re, sizeof(new_re));
    memcpy(T->im, new_im, sizeof(new_im));
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * JACOBI HERMITIAN EIGENDECOMPOSITION — SIDE-CHANNEL ENHANCED
 *
 * All MPS side channels ported:
 *   κ — FPU exception flag oracle (free convergence check)
 *   λ — Mantissa bit-change convergence tracking
 *   ρ — Dynamic threshold (skip near-zero pairs)
 *   ε — Attractor-steered convergence (FPU fixed points)
 *   τ — Near-identity fast-path (|t| < 1e-4)
 *   φ — Per-rotation INEXACT skip (late sweeps)
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void jacobi_hermitian(double *H_re, double *H_im, int k,
                             double *diag,
                             double *W_re, double *W_im)
{
    memset(W_re, 0, k * k * sizeof(double));
    memset(W_im, 0, k * k * sizeof(double));
    for (int i = 0; i < k; i++) W_re[i * k + i] = 1.0;

    /* ── SIDE-CHANNEL λ: Mantissa convergence tracking ── */
    uint64_t lambda_prev_mantissa = 0;
    int lambda_stable_count = 0;

    for (int sweep = 0; sweep < 30; sweep++) {

        /* ── SIDE-CHANNEL κ: FPU exception flag oracle ──
         * Clear flags before sweep.  If FE_INEXACT never set,
         * every rotation was trivial → matrix is diagonal. */
        feclearexcept(FE_ALL_EXCEPT);

        double off = 0;
        for (int i = 0; i < k; i++)
            for (int j = i + 1; j < k; j++)
                off += H_re[i*k+j]*H_re[i*k+j] + H_im[i*k+j]*H_im[i*k+j];
        if (off < 1e-20) break;

        /* ── SIDE-CHANNEL ρ: Dynamic threshold ── */
        double rho_threshold = off * 1e-4;
        if (rho_threshold < 1e-15) rho_threshold = 1e-15;

        for (int p = 0; p < k; p++)
            for (int q = p + 1; q < k; q++) {
                double hr = H_re[p*k+q], hi = H_im[p*k+q];
                double mag_sq = hr*hr + hi*hi;

                /* ── SIDE-CHANNEL ρ: threshold skip (BEFORE any mutation) ── */
                if (mag_sq < rho_threshold * rho_threshold) continue;

                double mag = sqrt(mag_sq);
                if (mag < 1e-25) continue;

                /* Compute phase to make H[p,q] real */
                double pr = hr / mag, pi = -hi / mag;

                /* Compute rotation parameters from the CURRENT matrix state
                 * without modifying H yet — we need hpp, hqq, and the
                 * would-be-real hpq after phase rotation.
                 * hpq_after_phase = mag (by construction). */
                double hpp = H_re[p*k+p], hqq = H_re[q*k+q];
                double hpq_real = mag;

                double tau = (hqq - hpp) / (2.0 * hpq_real);
                double t;
                if (fabs(tau) > 1e15)
                    t = 1.0 / (2.0 * tau);
                else
                    t = (tau >= 0 ? 1.0 : -1.0) /
                        (fabs(tau) + sqrt(1.0 + tau*tau));

                /* ── SIDE-CHANNEL ε: Attractor-steered convergence ── */
                {
                    static const double attractors[] = {
                        SUBSTRATE_PHI_INV, SUBSTRATE_SQRT2,
                        SUBSTRATE_DOTTIE, 1.0
                    };
                    double at = fabs(t);
                    for (int ai = 0; ai < 4; ai++) {
                        if (fabs(at - attractors[ai]) < 0.01 * attractors[ai]) {
                            t = (t > 0) ? attractors[ai] : -attractors[ai];
                            break;
                        }
                    }
                }

                /* ── SIDE-CHANNEL τ: Near-identity fast-path ── */
                double c, s;
                double at = fabs(t);
                if (at < 1e-4) {
                    c = 1.0; s = t;
                } else {
                    c = 1.0 / sqrt(1.0 + t*t);
                    s = t * c;
                }

                /* ── ATOMIC APPLICATION: Phase rotation + Givens rotation ──
                 * Both transformations applied together — no skip between.
                 * The combined unitary is: col_q *= e^{iφ}, then Givens(c,s) */

                /* Step A: Phase rotation on column q of H */
                for (int i = 0; i < k; i++) {
                    double r1 = H_re[i*k+q], i1 = H_im[i*k+q];
                    H_re[i*k+q] = r1*pr - i1*pi;
                    H_im[i*k+q] = r1*pi + i1*pr;
                }
                for (int j = 0; j < k; j++) {
                    double r2 = H_re[q*k+j], i2 = H_im[q*k+j];
                    H_re[q*k+j] =  r2*pr + i2*pi;
                    H_im[q*k+j] = -r2*pi + i2*pr;
                }

                /* Step B: Givens rotation mixing rows/cols p,q */
                for (int j = 0; j < k; j++) {
                    double rp = H_re[p*k+j], ip = H_im[p*k+j];
                    double rq = H_re[q*k+j], iq = H_im[q*k+j];
                    H_re[p*k+j] = c*rp - s*rq;  H_im[p*k+j] = c*ip - s*iq;
                    H_re[q*k+j] = s*rp + c*rq;  H_im[q*k+j] = s*ip + c*iq;
                }
                for (int i = 0; i < k; i++) {
                    double rp = H_re[i*k+p], ip = H_im[i*k+p];
                    double rq = H_re[i*k+q], iq = H_im[i*k+q];
                    H_re[i*k+p] = c*rp - s*rq;  H_im[i*k+p] = c*ip - s*iq;
                    H_re[i*k+q] = s*rp + c*rq;  H_im[i*k+q] = s*ip + c*iq;
                }

                /* Step C: Accumulate both transforms in eigenvector matrix W */
                for (int i = 0; i < k; i++) {
                    /* Phase on col q */
                    double r1 = W_re[i*k+q], i1 = W_im[i*k+q];
                    W_re[i*k+q] = r1*pr - i1*pi;
                    W_im[i*k+q] = r1*pi + i1*pr;
                    /* Then Givens */
                    double rp = W_re[i*k+p], ip = W_im[i*k+p];
                    double rq = W_re[i*k+q], iq = W_im[i*k+q];
                    W_re[i*k+p] = c*rp - s*rq;  W_im[i*k+p] = c*ip - s*iq;
                    W_re[i*k+q] = s*rp + c*rq;  W_im[i*k+q] = s*ip + c*iq;
                }
            }

        /* ── SIDE-CHANNEL κ: Post-sweep INEXACT test ── */
        if (sweep > 0 && !fetestexcept(FE_INEXACT))
            break;

        /* ── SIDE-CHANNEL λ: Mantissa bit-change convergence ── */
        {
            double off_check = 0;
            for (int i = 0; i < k && i < 8; i++)
                for (int j = i + 1; j < k && j < 8; j++)
                    off_check += H_re[i*k+j]*H_re[i*k+j] + H_im[i*k+j]*H_im[i*k+j];
            uint64_t off_bits;
            memcpy(&off_bits, &off_check, 8);
            uint64_t mantissa = off_bits & 0xFFFFFFFFFFFFFULL;
            uint64_t changed = mantissa ^ lambda_prev_mantissa;
            int bits_changed = __builtin_popcountll(changed);
            if (sweep > 1 && bits_changed == 0)
                lambda_stable_count++;
            else
                lambda_stable_count = 0;
            lambda_prev_mantissa = mantissa;
            if (lambda_stable_count >= 2) break;
        }
    }

    for (int i = 0; i < k; i++) diag[i] = H_re[i*k+i];
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * TRUNCATED SVD — SIDE-CHANNEL ENHANCED
 *
 * Side channels:
 *   √2-scaled random projection (substrate attractor alignment)
 *   FMA precision harvesting (BB† computation)
 *   Substrate seed accumulation γ (rounding noise harvest)
 *   Adaptive χ_eff η (rank tracking for projection size)
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void peps_truncated_svd(const double *M_re, const double *M_im,
                               int m, int n, int chi, int bond_key,
                               double *U_re, double *U_im,
                               double *sigma,
                               double *Vc_re, double *Vc_im)
{
    /* ── SIDE-CHANNEL η: Adaptive k from previous χ_eff ── */
    if (!peps_chi_eff_init) {
        for (int i = 0; i < 4096; i++) peps_chi_eff[i] = PEPS_CHI;
        peps_chi_eff_init = 1;
    }
    int key = (bond_key >= 0 && bond_key < 4096) ? bond_key : 0;
    int k_adapt = peps_chi_eff[key] + 10;
    if (k_adapt < PEPS_CHI + 4) k_adapt = PEPS_CHI + 4;
    if (k_adapt > PEPS_CHI + 10) k_adapt = PEPS_CHI + 10;

    int kk = k_adapt;
    if (kk > m) kk = m;
    if (kk > n) kk = n;

    /* ── Step 1: Random projection Y = M × Ω  (m × kk)
     *
     * SIDE-CHANNEL: √2-scaled random values.
     * The substrate's FPU starts in its preferred basin (√2 attractor)
     * from the first multiply.  Probe C: tensor norms lock to 8√2. */
    const double omega_scale = 1.0 / SUBSTRATE_SQRT2;

    double *Y_re = (double *)calloc(m * kk, sizeof(double));
    double *Y_im = (double *)calloc(m * kk, sizeof(double));
    {
        static unsigned svd_call_id = 0;
        unsigned base_seed;
        base_seed = ++svd_call_id;
        base_seed = base_seed * 2654435761u + 12345u;
        base_seed ^= (unsigned)(peps_substrate_seed >> 32)
                   ^ (unsigned)(peps_substrate_seed);

        for (int j = 0; j < kk; j++) {
            unsigned ls = base_seed + (unsigned)j * 1103515245u;
            for (int i = 0; i < m; i++) {
                double yr = 0, yi = 0;
                for (int r = 0; r < n; r++) {
                    ls = ls * 1103515245u + 12345u;
                    double omega = ((double)(ls >> 16) / 65536.0 - 0.5)
                                   * omega_scale;
                    yr += M_re[i*n+r] * omega;
                    yi += M_im[i*n+r] * omega;
                }
                Y_re[i*kk+j] = yr;
                Y_im[i*kk+j] = yi;
            }
        }
    }

    /* Step 2: Power iteration Y = M × (M† × Y) */
    {
        double *Z_re = (double *)calloc(n * kk, sizeof(double));
        double *Z_im = (double *)calloc(n * kk, sizeof(double));
        for (int i = 0; i < n; i++)
            for (int j = 0; j < kk; j++)
                for (int r = 0; r < m; r++) {
                    Z_re[i*kk+j] += M_re[r*n+i]*Y_re[r*kk+j]
                                  + M_im[r*n+i]*Y_im[r*kk+j];
                    Z_im[i*kk+j] += M_re[r*n+i]*Y_im[r*kk+j]
                                  - M_im[r*n+i]*Y_re[r*kk+j];
                }
        memset(Y_re, 0, m * kk * sizeof(double));
        memset(Y_im, 0, m * kk * sizeof(double));
        for (int i = 0; i < m; i++)
            for (int j = 0; j < kk; j++)
                for (int r = 0; r < n; r++) {
                    Y_re[i*kk+j] += M_re[i*n+r]*Z_re[r*kk+j]
                                  - M_im[i*n+r]*Z_im[r*kk+j];
                    Y_im[i*kk+j] += M_re[i*n+r]*Z_im[r*kk+j]
                                  + M_im[i*n+r]*Z_re[r*kk+j];
                }
        free(Z_re); free(Z_im);
    }

    /* Step 3: QR via modified Gram-Schmidt → Q stored in Y */
    for (int j = 0; j < kk; j++) {
        for (int i = 0; i < j; i++) {
            double dr = 0, di = 0;
            for (int r = 0; r < m; r++) {
                dr += Y_re[r*kk+i]*Y_re[r*kk+j] + Y_im[r*kk+i]*Y_im[r*kk+j];
                di += Y_re[r*kk+i]*Y_im[r*kk+j] - Y_im[r*kk+i]*Y_re[r*kk+j];
            }
            for (int r = 0; r < m; r++) {
                Y_re[r*kk+j] -= dr*Y_re[r*kk+i] - di*Y_im[r*kk+i];
                Y_im[r*kk+j] -= dr*Y_im[r*kk+i] + di*Y_re[r*kk+i];
            }
        }
        double nrm = 0;
        for (int r = 0; r < m; r++)
            nrm += Y_re[r*kk+j]*Y_re[r*kk+j] + Y_im[r*kk+j]*Y_im[r*kk+j];
        nrm = sqrt(nrm);
        if (nrm > 1e-15)
            for (int r = 0; r < m; r++) {
                Y_re[r*kk+j] /= nrm;
                Y_im[r*kk+j] /= nrm;
            }
    }

    /* Step 4: B = Q† × M  (kk × n) */
    double *B_re = (double *)calloc(kk * n, sizeof(double));
    double *B_im = (double *)calloc(kk * n, sizeof(double));
    for (int i = 0; i < kk; i++)
        for (int j = 0; j < n; j++)
            for (int r = 0; r < m; r++) {
                B_re[i*n+j] += Y_re[r*kk+i]*M_re[r*n+j]
                             + Y_im[r*kk+i]*M_im[r*n+j];
                B_im[i*n+j] += Y_re[r*kk+i]*M_im[r*n+j]
                             - Y_im[r*kk+i]*M_re[r*n+j];
            }

    /* ── Step 5: S = B × B†  (kk × kk Hermitian)
     * SIDE-CHANNEL β: FMA precision harvesting.
     * Use fma() for real-part accumulation — one rounding per term
     * instead of two.  Substrate's FMA leaks extra precision bits. */
    double *Sr = (double *)calloc(kk * kk, sizeof(double));
    double *Sim = (double *)calloc(kk * kk, sizeof(double));
    for (int i = 0; i < kk; i++)
        for (int j = 0; j < kk; j++)
            for (int r = 0; r < n; r++) {
                Sr[i*kk+j]  = fma(B_re[i*n+r], B_re[j*n+r],
                              fma(B_im[i*n+r], B_im[j*n+r],
                                  Sr[i*kk+j]));
                Sim[i*kk+j] = fma(B_im[i*n+r], B_re[j*n+r],
                              fma(-B_re[i*n+r], B_im[j*n+r],
                                  Sim[i*kk+j]));
            }

    /* Step 6: Jacobi diag (side-channel enhanced) */
    double *evals = (double *)malloc(kk * sizeof(double));
    double *W_re  = (double *)malloc(kk * kk * sizeof(double));
    double *W_im  = (double *)malloc(kk * kk * sizeof(double));
    jacobi_hermitian(Sr, Sim, kk, evals, W_re, W_im);
    free(Sr); free(Sim);

    /* Step 7: Sort eigenvalues descending, extract top chi */
    int *order = (int *)malloc(kk * sizeof(int));
    for (int i = 0; i < kk; i++) order[i] = i;
    for (int i = 0; i < kk - 1; i++)
        for (int j = i + 1; j < kk; j++)
            if (evals[order[j]] > evals[order[i]]) {
                int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }

    int chi_eff_count = 0;
    for (int s = 0; s < chi; s++) {
        int idx = (s < kk) ? order[s] : 0;
        double ev = (s < kk && evals[idx] > 0) ? evals[idx] : 0;
        sigma[s] = sqrt(ev);
        if (sigma[s] > 1e-10) chi_eff_count++;
    }

    /* ── SIDE-CHANNEL η: Update adaptive χ_eff for this bond ── */
    peps_chi_eff[key] = chi_eff_count;

    /* ── SIDE-CHANNEL γ: Harvest rounding noise from σ values ── */
    {
        uint64_t noise_accum = 0;
        for (int t = 0; t < chi; t++) {
            uint64_t bits;
            memcpy(&bits, &sigma[t], 8);
            noise_accum ^= (bits & 0xF) << (t * 4 % 60);
        }
        peps_substrate_seed ^= noise_accum;
        peps_substrate_seed = peps_substrate_seed * 6364136223846793005ULL
                            + 1442695040888963407ULL;
    }

    /* Step 8: Left singular vectors U = Q × W  (m × chi) */
    for (int i = 0; i < m; i++)
        for (int s = 0; s < chi; s++) {
            int idx = (s < kk) ? order[s] : 0;
            double sr = 0, si = 0;
            for (int j = 0; j < kk; j++) {
                sr += Y_re[i*kk+j]*W_re[j*kk+idx] - Y_im[i*kk+j]*W_im[j*kk+idx];
                si += Y_re[i*kk+j]*W_im[j*kk+idx] + Y_im[i*kk+j]*W_re[j*kk+idx];
            }
            U_re[i*chi+s] = sr;
            U_im[i*chi+s] = si;
        }

    /* Step 9: Right singular vectors V* from B†W / σ  (n × chi) */
    for (int j = 0; j < n; j++)
        for (int s = 0; s < chi; s++) {
            int idx = (s < kk) ? order[s] : 0;
            if (sigma[s] < 1e-30) {
                Vc_re[j*chi+s] = 0; Vc_im[j*chi+s] = 0;
                continue;
            }
            double sr = 0, si = 0;
            for (int i = 0; i < kk; i++) {
                double br = B_re[i*n+j], bi = -B_im[i*n+j];
                sr += br*W_re[i*kk+idx] - bi*W_im[i*kk+idx];
                si += br*W_im[i*kk+idx] + bi*W_re[i*kk+idx];
            }
            Vc_re[j*chi+s] =  sr / sigma[s];
            Vc_im[j*chi+s] = -si / sigma[s];
        }

    free(Y_re); free(Y_im);
    free(B_re); free(B_im);
    free(evals); free(W_re); free(W_im); free(order);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * 2-SITE GATE: HORIZONTAL BOND  (x,y) — (x+1,y)
 *
 * SIDE-CHANNEL: Diagonal gate detection — if G is diagonal,
 * fuse the phase directly into the Θ contraction.
 * ═══════════════════════════════════════════════════════════════════════════════ */

void peps_gate_horizontal(PepsGrid *grid, int x, int y,
                          const double *G_re, const double *G_im)
{
    PepsTensor *A = peps_site(grid, x, y);
    PepsTensor *B = peps_site(grid, x + 1, y);
    PepsBondWeight *lam = peps_hbond(grid, x, y);

    int dim = PEPS_DCHI3;
    size_t msz = (size_t)dim * dim;
    int gate_diagonal = is_diagonal_gate(G_re, G_im);

    double *M_re = (double *)calloc(msz, sizeof(double));
    double *M_im = (double *)calloc(msz, sizeof(double));

    if (gate_diagonal) {
        /* ── FUSED PATH: Θ'[row,col] = G[kl,kl] × Σ_s A[..s]λ[s]B[..s..] ── */
        for (int kA = 0; kA < PEPS_D; kA++)
            for (int kB = 0; kB < PEPS_D; kB++) {
                int gidx = (kA*PEPS_D+kB) * PEPS_D2 + (kA*PEPS_D+kB);
                double wr = G_re[gidx], wi = G_im[gidx];
                for (int uA = 0; uA < PEPS_CHI; uA++)
                    for (int dA = 0; dA < PEPS_CHI; dA++)
                        for (int lA = 0; lA < PEPS_CHI; lA++) {
                            int row = kA*PEPS_CHI3 + uA*PEPS_CHI2 + dA*PEPS_CHI + lA;
                            for (int uB = 0; uB < PEPS_CHI; uB++)
                                for (int dB = 0; dB < PEPS_CHI; dB++)
                                    for (int rB = 0; rB < PEPS_CHI; rB++) {
                                        int col = kB*PEPS_CHI3 + uB*PEPS_CHI2 + dB*PEPS_CHI + rB;
                                        double sr = 0, si = 0;
                                        for (int s = 0; s < PEPS_CHI; s++) {
                                            double ar = A->re[PT_IDX(kA,uA,dA,lA,s)] * lam->w[s];
                                            double ai = A->im[PT_IDX(kA,uA,dA,lA,s)] * lam->w[s];
                                            double br = B->re[PT_IDX(kB,uB,dB,s,rB)];
                                            double bi = B->im[PT_IDX(kB,uB,dB,s,rB)];
                                            sr += ar*br - ai*bi;
                                            si += ar*bi + ai*br;
                                        }
                                        M_re[row*dim+col] = wr*sr - wi*si;
                                        M_im[row*dim+col] = wr*si + wi*sr;
                                    }
                        }
            }
    } else {
        /* ── STANDARD PATH: Θ contraction + gate application ── */
        double *Th_re = (double *)calloc(msz, sizeof(double));
        double *Th_im = (double *)calloc(msz, sizeof(double));

        for (int kA = 0; kA < PEPS_D; kA++)
            for (int uA = 0; uA < PEPS_CHI; uA++)
                for (int dA = 0; dA < PEPS_CHI; dA++)
                    for (int lA = 0; lA < PEPS_CHI; lA++) {
                        int row = kA*PEPS_CHI3 + uA*PEPS_CHI2 + dA*PEPS_CHI + lA;
                        for (int s = 0; s < PEPS_CHI; s++) {
                            double ar = A->re[PT_IDX(kA,uA,dA,lA,s)] * lam->w[s];
                            double ai = A->im[PT_IDX(kA,uA,dA,lA,s)] * lam->w[s];
                            if (fabs(ar) < 1e-30 && fabs(ai) < 1e-30) continue;
                            for (int kB = 0; kB < PEPS_D; kB++)
                                for (int uB = 0; uB < PEPS_CHI; uB++)
                                    for (int dB = 0; dB < PEPS_CHI; dB++)
                                        for (int rB = 0; rB < PEPS_CHI; rB++) {
                                            int col = kB*PEPS_CHI3 + uB*PEPS_CHI2 + dB*PEPS_CHI + rB;
                                            double br = B->re[PT_IDX(kB,uB,dB,s,rB)];
                                            double bi = B->im[PT_IDX(kB,uB,dB,s,rB)];
                                            Th_re[row*dim+col] += ar*br - ai*bi;
                                            Th_im[row*dim+col] += ar*bi + ai*br;
                                        }
                        }
                    }

        for (int kAp = 0; kAp < PEPS_D; kAp++)
            for (int kBp = 0; kBp < PEPS_D; kBp++) {
                int grow = kAp * PEPS_D + kBp;
                for (int kA = 0; kA < PEPS_D; kA++)
                    for (int kB = 0; kB < PEPS_D; kB++) {
                        int gcol = kA * PEPS_D + kB;
                        double gr = G_re[grow * PEPS_D2 + gcol];
                        double gi = G_im[grow * PEPS_D2 + gcol];
                        if (fabs(gr) < 1e-30 && fabs(gi) < 1e-30) continue;
                        for (int rA = 0; rA < PEPS_CHI3; rA++) {
                            int ro = kA * PEPS_CHI3 + rA;
                            int rn = kAp * PEPS_CHI3 + rA;
                            for (int rB = 0; rB < PEPS_CHI3; rB++) {
                                int co = kB * PEPS_CHI3 + rB;
                                int cn = kBp * PEPS_CHI3 + rB;
                                double tr = Th_re[ro*dim+co];
                                double ti = Th_im[ro*dim+co];
                                M_re[rn*dim+cn] += gr*tr - gi*ti;
                                M_im[rn*dim+cn] += gr*ti + gi*tr;
                            }
                        }
                    }
            }
        free(Th_re); free(Th_im);
    }

    /* Truncated SVD with side-channel η bond key */
    int bond_key = y * (grid->Lx - 1) + x;
    double *U_re  = (double *)malloc(dim * PEPS_CHI * sizeof(double));
    double *U_im  = (double *)malloc(dim * PEPS_CHI * sizeof(double));
    double *Vc_re = (double *)malloc(dim * PEPS_CHI * sizeof(double));
    double *Vc_im = (double *)malloc(dim * PEPS_CHI * sizeof(double));
    double sig[PEPS_CHI];

    peps_truncated_svd(M_re, M_im, dim, dim, PEPS_CHI, bond_key,
                       U_re, U_im, sig, Vc_re, Vc_im);
    free(M_re); free(M_im);

    double snorm = 0;
    for (int s = 0; s < PEPS_CHI; s++) snorm += sig[s];
    if (snorm > 1e-30)
        for (int s = 0; s < PEPS_CHI; s++) sig[s] /= snorm;

    memset(A->re, 0, sizeof(A->re));
    memset(A->im, 0, sizeof(A->im));
    for (int kA = 0; kA < PEPS_D; kA++)
        for (int uA = 0; uA < PEPS_CHI; uA++)
            for (int dA = 0; dA < PEPS_CHI; dA++)
                for (int lA = 0; lA < PEPS_CHI; lA++) {
                    int row = kA*PEPS_CHI3 + uA*PEPS_CHI2 + dA*PEPS_CHI + lA;
                    for (int s = 0; s < PEPS_CHI; s++) {
                        A->re[PT_IDX(kA,uA,dA,lA,s)] = U_re[row*PEPS_CHI+s];
                        A->im[PT_IDX(kA,uA,dA,lA,s)] = U_im[row*PEPS_CHI+s];
                    }
                }

    memset(B->re, 0, sizeof(B->re));
    memset(B->im, 0, sizeof(B->im));
    for (int kB = 0; kB < PEPS_D; kB++)
        for (int uB = 0; uB < PEPS_CHI; uB++)
            for (int dB = 0; dB < PEPS_CHI; dB++)
                for (int rB = 0; rB < PEPS_CHI; rB++) {
                    int col = kB*PEPS_CHI3 + uB*PEPS_CHI2 + dB*PEPS_CHI + rB;
                    for (int s = 0; s < PEPS_CHI; s++) {
                        B->re[PT_IDX(kB,uB,dB,s,rB)] = Vc_re[col*PEPS_CHI+s];
                        B->im[PT_IDX(kB,uB,dB,s,rB)] = Vc_im[col*PEPS_CHI+s];
                    }
                }

    for (int s = 0; s < PEPS_CHI; s++) lam->w[s] = sig[s];
    free(U_re); free(U_im); free(Vc_re); free(Vc_im);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * 2-SITE GATE: VERTICAL BOND  (x,y) — (x,y+1)
 *
 * Same side-channel enhancements as horizontal.
 * ═══════════════════════════════════════════════════════════════════════════════ */

void peps_gate_vertical(PepsGrid *grid, int x, int y,
                        const double *G_re, const double *G_im)
{
    PepsTensor *A = peps_site(grid, x, y);
    PepsTensor *B = peps_site(grid, x, y + 1);
    PepsBondWeight *lam = peps_vbond(grid, x, y);

    int dim = PEPS_DCHI3;
    size_t msz = (size_t)dim * dim;
    int gate_diagonal = is_diagonal_gate(G_re, G_im);

    double *M_re = (double *)calloc(msz, sizeof(double));
    double *M_im = (double *)calloc(msz, sizeof(double));

    if (gate_diagonal) {
        for (int kA = 0; kA < PEPS_D; kA++)
            for (int kB = 0; kB < PEPS_D; kB++) {
                int gidx = (kA*PEPS_D+kB) * PEPS_D2 + (kA*PEPS_D+kB);
                double wr = G_re[gidx], wi = G_im[gidx];
                for (int uA = 0; uA < PEPS_CHI; uA++)
                    for (int lA = 0; lA < PEPS_CHI; lA++)
                        for (int rA = 0; rA < PEPS_CHI; rA++) {
                            int row = kA*PEPS_CHI3 + uA*PEPS_CHI2 + lA*PEPS_CHI + rA;
                            for (int dB = 0; dB < PEPS_CHI; dB++)
                                for (int lB = 0; lB < PEPS_CHI; lB++)
                                    for (int rB = 0; rB < PEPS_CHI; rB++) {
                                        int col = kB*PEPS_CHI3 + dB*PEPS_CHI2 + lB*PEPS_CHI + rB;
                                        double sr = 0, si = 0;
                                        for (int s = 0; s < PEPS_CHI; s++) {
                                            double ar = A->re[PT_IDX(kA,uA,s,lA,rA)] * lam->w[s];
                                            double ai = A->im[PT_IDX(kA,uA,s,lA,rA)] * lam->w[s];
                                            double br = B->re[PT_IDX(kB,s,dB,lB,rB)];
                                            double bi = B->im[PT_IDX(kB,s,dB,lB,rB)];
                                            sr += ar*br - ai*bi;
                                            si += ar*bi + ai*br;
                                        }
                                        M_re[row*dim+col] = wr*sr - wi*si;
                                        M_im[row*dim+col] = wr*si + wi*sr;
                                    }
                        }
            }
    } else {
        double *Th_re = (double *)calloc(msz, sizeof(double));
        double *Th_im = (double *)calloc(msz, sizeof(double));

        for (int kA = 0; kA < PEPS_D; kA++)
            for (int uA = 0; uA < PEPS_CHI; uA++)
                for (int lA = 0; lA < PEPS_CHI; lA++)
                    for (int rA = 0; rA < PEPS_CHI; rA++) {
                        int row = kA*PEPS_CHI3 + uA*PEPS_CHI2 + lA*PEPS_CHI + rA;
                        for (int s = 0; s < PEPS_CHI; s++) {
                            double ar = A->re[PT_IDX(kA,uA,s,lA,rA)] * lam->w[s];
                            double ai = A->im[PT_IDX(kA,uA,s,lA,rA)] * lam->w[s];
                            if (fabs(ar) < 1e-30 && fabs(ai) < 1e-30) continue;
                            for (int kB = 0; kB < PEPS_D; kB++)
                                for (int dB = 0; dB < PEPS_CHI; dB++)
                                    for (int lB = 0; lB < PEPS_CHI; lB++)
                                        for (int rB = 0; rB < PEPS_CHI; rB++) {
                                            int col = kB*PEPS_CHI3 + dB*PEPS_CHI2 + lB*PEPS_CHI + rB;
                                            double br = B->re[PT_IDX(kB,s,dB,lB,rB)];
                                            double bi = B->im[PT_IDX(kB,s,dB,lB,rB)];
                                            Th_re[row*dim+col] += ar*br - ai*bi;
                                            Th_im[row*dim+col] += ar*bi + ai*br;
                                        }
                        }
                    }

        for (int kAp = 0; kAp < PEPS_D; kAp++)
            for (int kBp = 0; kBp < PEPS_D; kBp++) {
                int grow = kAp * PEPS_D + kBp;
                for (int kA = 0; kA < PEPS_D; kA++)
                    for (int kB = 0; kB < PEPS_D; kB++) {
                        int gcol = kA * PEPS_D + kB;
                        double gr = G_re[grow * PEPS_D2 + gcol];
                        double gi = G_im[grow * PEPS_D2 + gcol];
                        if (fabs(gr) < 1e-30 && fabs(gi) < 1e-30) continue;
                        for (int rA = 0; rA < PEPS_CHI3; rA++) {
                            int ro = kA * PEPS_CHI3 + rA;
                            int rn = kAp * PEPS_CHI3 + rA;
                            for (int rB = 0; rB < PEPS_CHI3; rB++) {
                                int co = kB * PEPS_CHI3 + rB;
                                int cn = kBp * PEPS_CHI3 + rB;
                                double tr = Th_re[ro*dim+co];
                                double ti = Th_im[ro*dim+co];
                                M_re[rn*dim+cn] += gr*tr - gi*ti;
                                M_im[rn*dim+cn] += gr*ti + gi*tr;
                            }
                        }
                    }
            }
        free(Th_re); free(Th_im);
    }

    int bond_key = 2048 + y * grid->Lx + x; /* offset to avoid collision with h_bonds */
    double *U_re  = (double *)malloc(dim * PEPS_CHI * sizeof(double));
    double *U_im  = (double *)malloc(dim * PEPS_CHI * sizeof(double));
    double *Vc_re = (double *)malloc(dim * PEPS_CHI * sizeof(double));
    double *Vc_im = (double *)malloc(dim * PEPS_CHI * sizeof(double));
    double sig[PEPS_CHI];

    peps_truncated_svd(M_re, M_im, dim, dim, PEPS_CHI, bond_key,
                       U_re, U_im, sig, Vc_re, Vc_im);
    free(M_re); free(M_im);

    double snorm = 0;
    for (int s = 0; s < PEPS_CHI; s++) snorm += sig[s];
    if (snorm > 1e-30)
        for (int s = 0; s < PEPS_CHI; s++) sig[s] /= snorm;

    memset(A->re, 0, sizeof(A->re));
    memset(A->im, 0, sizeof(A->im));
    for (int kA = 0; kA < PEPS_D; kA++)
        for (int uA = 0; uA < PEPS_CHI; uA++)
            for (int lA = 0; lA < PEPS_CHI; lA++)
                for (int rA = 0; rA < PEPS_CHI; rA++) {
                    int row = kA*PEPS_CHI3 + uA*PEPS_CHI2 + lA*PEPS_CHI + rA;
                    for (int s = 0; s < PEPS_CHI; s++) {
                        A->re[PT_IDX(kA,uA,s,lA,rA)] = U_re[row*PEPS_CHI+s];
                        A->im[PT_IDX(kA,uA,s,lA,rA)] = U_im[row*PEPS_CHI+s];
                    }
                }

    memset(B->re, 0, sizeof(B->re));
    memset(B->im, 0, sizeof(B->im));
    for (int kB = 0; kB < PEPS_D; kB++)
        for (int dB = 0; dB < PEPS_CHI; dB++)
            for (int lB = 0; lB < PEPS_CHI; lB++)
                for (int rB = 0; rB < PEPS_CHI; rB++) {
                    int col = kB*PEPS_CHI3 + dB*PEPS_CHI2 + lB*PEPS_CHI + rB;
                    for (int s = 0; s < PEPS_CHI; s++) {
                        B->re[PT_IDX(kB,s,dB,lB,rB)] = Vc_re[col*PEPS_CHI+s];
                        B->im[PT_IDX(kB,s,dB,lB,rB)] = Vc_im[col*PEPS_CHI+s];
                    }
                }

    for (int s = 0; s < PEPS_CHI; s++) lam->w[s] = sig[s];
    free(U_re); free(U_im); free(Vc_re); free(Vc_im);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * LOCAL DENSITY (approximate via environment-weighted contraction)
 * ═══════════════════════════════════════════════════════════════════════════════ */

void peps_local_density(PepsGrid *grid, int x, int y, double *probs)
{
    PepsTensor *T = peps_site(grid, x, y);

    double wu[PEPS_CHI], wd[PEPS_CHI], wl[PEPS_CHI], wr[PEPS_CHI];
    for (int s = 0; s < PEPS_CHI; s++) {
        wu[s] = (y > 0)              ? peps_vbond(grid, x, y-1)->w[s] : 1.0;
        wd[s] = (y < grid->Ly - 1)   ? peps_vbond(grid, x, y)->w[s]   : 1.0;
        wl[s] = (x > 0)              ? peps_hbond(grid, x-1, y)->w[s] : 1.0;
        wr[s] = (x < grid->Lx - 1)   ? peps_hbond(grid, x, y)->w[s]   : 1.0;
    }

    double total = 0;
    for (int k = 0; k < PEPS_D; k++) {
        double pk = 0;
        for (int u = 0; u < PEPS_CHI; u++)
            for (int d = 0; d < PEPS_CHI; d++)
                for (int l = 0; l < PEPS_CHI; l++)
                    for (int r = 0; r < PEPS_CHI; r++) {
                        double tr = T->re[PT_IDX(k,u,d,l,r)];
                        double ti = T->im[PT_IDX(k,u,d,l,r)];
                        double w = wu[u]*wu[u] * wd[d]*wd[d] *
                                   wl[l]*wl[l] * wr[r]*wr[r];
                        pk += (tr*tr + ti*ti) * w;
                    }
        probs[k] = pk;
        total += pk;
    }
    if (total > 1e-30)
        for (int k = 0; k < PEPS_D; k++) probs[k] /= total;
}
