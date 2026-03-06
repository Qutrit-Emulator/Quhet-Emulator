/* ═══════════════════════════════════════════════════════════════════════════
 * vesica_wave_test.c — Full Fold (Vesica+Wave) vs SVD
 *
 * The previous test used ONLY the vesica channels. This one uses BOTH.
 * The fold is unitary: vesica + wave = lossless. So reconstruction should
 * be exact. The question: does the full folded spectrum naturally sort?
 *
 * For each Θ matrix:
 *   1. 15-syntheme sweep → optimal fold
 *   2. Fold into 6 channels: 3 vesica + 3 wave
 *   3. Compute per-channel norms (diagonal block norms in folded basis)
 *   4. Compare ordering against SVD σ values
 *   5. Reconstruct from full fold → measure error (should be ~0)
 *   6. Reconstruct from fold with magnitude-sorted truncation → compare to SVD truncation
 *
 * Build:
 *   gcc -O2 -march=native -o vesica_wave_test vesica_wave_test.c \
 *       quhit_core.c quhit_gates.c quhit_measure.c quhit_entangle.c \
 *       quhit_register.c quhit_substrate.c quhit_triality.c quhit_triadic.c \
 *       quhit_lazy.c quhit_calibrate.c quhit_svd_gate.c s6_exotic.c \
 *       mps_overlay.c peps_overlay.c peps3d_overlay.c peps4d_overlay.c \
 *       peps5d_overlay.c peps6d_overlay.c -lm -fopenmp -msse2
 * ═══════════════════════════════════════════════════════════════════════════ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mps_overlay.h"
#include "s6_exotic.h"
#include "tensor_svd.h"

#define D 6

/* ── Descending sort with index tracking ── */
typedef struct { double val; int idx; } IdxVal;
static int idxval_cmp(const void *a, const void *b) {
    double da = ((const IdxVal *)a)->val, db = ((const IdxVal *)b)->val;
    return (db > da) - (db < da);
}

/* ── PRNG ── */
static uint64_t g_rng = 42;
static void rng_seed(uint64_t s) { g_rng = s; }
static double rng_gauss(void) {
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = ((double)(g_rng >> 32) + 1.0) / 4294967298.0;
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = ((double)(g_rng >> 32)) / 4294967296.0;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TEST MATRICES
 * ═══════════════════════════════════════════════════════════════════════════ */

static void mat_random(double *re, double *im, int m, int n) {
    for (int i = 0; i < m*n; i++) { re[i] = rng_gauss(); im[i] = rng_gauss(); }
}

static void mat_bell(double *re, double *im) {
    memset(re, 0, D*D*sizeof(double)); memset(im, 0, D*D*sizeof(double));
    double c = 1.0/sqrt(3.0);
    re[0*D+3]=c; re[1*D+4]=c; re[2*D+5]=c;
}

static void mat_ghz(double *re, double *im) {
    memset(re, 0, D*D*sizeof(double)); memset(im, 0, D*D*sizeof(double));
    double c = 1.0/sqrt(6.0);
    for (int k=0;k<D;k++) re[k*D+k]=c;
}

static void mat_w_state(double *re, double *im) {
    /* W-type: (|01⟩ + |12⟩ + |23⟩ + |34⟩ + |45⟩ + |50⟩) / √6 */
    memset(re, 0, D*D*sizeof(double)); memset(im, 0, D*D*sizeof(double));
    double c = 1.0/sqrt(6.0);
    for (int k=0;k<D;k++) re[k*D+((k+1)%D)]=c;
}

static void mat_dft(double *re, double *im) {
    double c = 1.0/(double)D;
    for (int j=0;j<D;j++) for (int k=0;k<D;k++) {
        double a = 2.0*M_PI*j*k/D;
        re[j*D+k] = c*cos(a); im[j*D+k] = c*sin(a);
    }
}

static void mat_rank1(double *re, double *im) {
    /* |ψ⟩ = |+⟩⊗|+⟩ */
    memset(re, 0, D*D*sizeof(double)); memset(im, 0, D*D*sizeof(double));
    double c = 1.0/D;
    for (int j=0;j<D;j++) for (int k=0;k<D;k++) re[j*D+k]=c;
}

static void mat_rank2(double *re, double *im) {
    /* Two terms: |0,0⟩/√2 + |3,3⟩/√2 */
    memset(re, 0, D*D*sizeof(double)); memset(im, 0, D*D*sizeof(double));
    double c = 1.0/sqrt(2.0);
    re[0*D+0]=c; re[3*D+3]=c;
}

static void mat_cz_entangled(double *re, double *im) {
    /* |+⟩⊗|+⟩ after CZ: CZ|j,k⟩ = ω^{jk}|j,k⟩ */
    double c = 1.0/D;
    for (int j=0;j<D;j++) for (int k=0;k<D;k++) {
        double a = 2.0*M_PI*j*k/D;
        re[j*D+k] = c*cos(a); im[j*D+k] = c*sin(a);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CORE: Full-fold analysis + SVD comparison
 * ═══════════════════════════════════════════════════════════════════════════ */

static void run_one(const char *name, const double *Th_re, const double *Th_im,
                    int *n_sorted_out, double *corr_out, double *full_err_out,
                    double *trunc3_fold_err_out, double *trunc3_svd_err_out)
{
    int m = D, n = D, nEA = 1, nEB = 1;
    size_t msz = (size_t)m * n;

    /* ── SVD ground truth ── */
    double sU_re[D*D], sU_im[D*D], sigma[D], sV_re[D*D], sV_im[D*D];
    tsvd_truncated(Th_re, Th_im, m, n, D, sU_re, sU_im, sigma, sV_re, sV_im);

    /* ── 15-syntheme sweep ── */
    double *FF_re = calloc(msz, sizeof(double));
    double *FF_im = calloc(msz, sizeof(double));

    int best_si = 7; double best_wf = 1.0;
    for (int si = 0; si < 15; si++) {
        tsvd_fold_syntheme(Th_re, Th_im, m, n, nEA, nEB,
                           tsvd_s6_synthemes[si], FF_re, FF_im);
        double wf = tsvd_wave_energy(FF_re, FF_im, m, n, nEA, nEB);
        if (wf < best_wf) { best_wf = wf; best_si = si; }
    }

    const int (*pairs)[2] = tsvd_s6_synthemes[best_si];
    tsvd_fold_syntheme(Th_re, Th_im, m, n, nEA, nEB, pairs, FF_re, FF_im);

    /* ── Extract ALL 6 channel amplitudes ──
     * In the folded D×D matrix, rows 0-2 are vesica, rows 3-5 are wave.
     * Same for columns. So we have a 2×2 block structure:
     *   VV (vesica×vesica)  VW (vesica×wave)
     *   WV (wave×vesica)    WW (wave×wave)
     *
     * Per-row norm gives 6 values — the amplitude in each channel. */
    IdxVal fold_spectrum[D];
    for (int r = 0; r < D; r++) {
        double rnorm2 = 0;
        for (int c = 0; c < D; c++) {
            double re = FF_re[r*n+c], im = FF_im[r*n+c];
            rnorm2 += re*re + im*im;
        }
        fold_spectrum[r].val = sqrt(rnorm2);
        fold_spectrum[r].idx = r;
    }

    /* Check if naturally sorted BEFORE sort */
    int naturally_sorted = 1;
    for (int i = 1; i < D; i++) {
        if (fold_spectrum[i].val > fold_spectrum[i-1].val + 1e-12) {
            naturally_sorted = 0; break;
        }
    }
    *n_sorted_out = naturally_sorted;

    /* Sort descending */
    qsort(fold_spectrum, D, sizeof(IdxVal), idxval_cmp);

    /* ── Pearson correlation ── */
    double ms = 0, mf = 0;
    for (int i=0;i<D;i++) { ms += sigma[i]; mf += fold_spectrum[i].val; }
    ms /= D; mf /= D;

    double cov=0, vs=0, vf=0;
    for (int i=0;i<D;i++) {
        double ds = sigma[i]-ms, df = fold_spectrum[i].val-mf;
        cov += ds*df; vs += ds*ds; vf += df*df;
    }
    double denom = sqrt(vs*vf);
    *corr_out = (denom > 1e-30) ? cov/denom : 0;

    /* ── Full reconstruction from fold (should be ~0 err) ──
     * Unfold ALL 6 channels (vesica+wave) back to physical basis. */
    double recon_re[D*D], recon_im[D*D];
    memset(recon_re, 0, sizeof(recon_re));
    memset(recon_im, 0, sizeof(recon_im));

    for (int rr = 0; rr < D; rr++) {
        int kA_fold = rr;
        /* Map fold row → physical rows */
        int is_wave_r = (kA_fold >= 3);
        int pair_r = is_wave_r ? kA_fold - 3 : kA_fold;
        int a_r = pairs[pair_r][0], b_r = pairs[pair_r][1];

        for (int cc = 0; cc < D; cc++) {
            int kB_fold = cc;
            int is_wave_c = (kB_fold >= 3);
            int pair_c = is_wave_c ? kB_fold - 3 : kB_fold;
            int a_c = pairs[pair_c][0], b_c = pairs[pair_c][1];

            double fr = FF_re[rr*n+cc], fi = FF_im[rr*n+cc];

            /* Unfold: vesica contributes +1/√2 to both, wave ±1/√2 */
            double sign_r_a = TSVD_INV_SQRT2;
            double sign_r_b = is_wave_r ? -TSVD_INV_SQRT2 : TSVD_INV_SQRT2;
            double sign_c_a = TSVD_INV_SQRT2;
            double sign_c_b = is_wave_c ? -TSVD_INV_SQRT2 : TSVD_INV_SQRT2;

            /* 4 contributions: (a_r,a_c), (a_r,b_c), (b_r,a_c), (b_r,b_c) */
            recon_re[a_r*D+a_c] += sign_r_a * sign_c_a * fr;
            recon_im[a_r*D+a_c] += sign_r_a * sign_c_a * fi;
            recon_re[a_r*D+b_c] += sign_r_a * sign_c_b * fr;
            recon_im[a_r*D+b_c] += sign_r_a * sign_c_b * fi;
            recon_re[b_r*D+a_c] += sign_r_b * sign_c_a * fr;
            recon_im[b_r*D+a_c] += sign_r_b * sign_c_a * fi;
            recon_re[b_r*D+b_c] += sign_r_b * sign_c_b * fr;
            recon_im[b_r*D+b_c] += sign_r_b * sign_c_b * fi;
        }
    }

    double err2 = 0, orig2 = 0;
    for (int i = 0; i < m*n; i++) {
        double dr = Th_re[i]-recon_re[i], di = Th_im[i]-recon_im[i];
        err2 += dr*dr + di*di;
        orig2 += Th_re[i]*Th_re[i] + Th_im[i]*Th_im[i];
    }
    *full_err_out = (orig2 > 1e-30) ? sqrt(err2/orig2) : 0;

    /* ── Truncated reconstruction: keep top-3 fold channels ──
     * Zero out the bottom 3 channels in folded space, then unfold.
     * Compare error against SVD rank-3 truncation error. */

    /* Fold truncation: zero out rows belonging to bottom-3 channels */
    double *FF_trunc_re = calloc(msz, sizeof(double));
    double *FF_trunc_im = calloc(msz, sizeof(double));
    memcpy(FF_trunc_re, FF_re, msz*sizeof(double));
    memcpy(FF_trunc_im, FF_im, msz*sizeof(double));

    for (int rank = 3; rank < D; rank++) {
        int row = fold_spectrum[rank].idx;
        for (int c = 0; c < n; c++) {
            FF_trunc_re[row*n+c] = 0;
            FF_trunc_im[row*n+c] = 0;
        }
        /* Also zero cols for this channel */
        for (int r = 0; r < m; r++) {
            FF_trunc_re[r*n+row] = 0;
            FF_trunc_im[r*n+row] = 0;
        }
    }

    /* Unfold truncated */
    double trunc_re[D*D], trunc_im[D*D];
    memset(trunc_re, 0, sizeof(trunc_re));
    memset(trunc_im, 0, sizeof(trunc_im));

    for (int rr = 0; rr < D; rr++) {
        int is_wave_r = (rr >= 3);
        int pair_r = is_wave_r ? rr - 3 : rr;
        int a_r = pairs[pair_r][0], b_r = pairs[pair_r][1];
        for (int cc = 0; cc < D; cc++) {
            int is_wave_c = (cc >= 3);
            int pair_c = is_wave_c ? cc - 3 : cc;
            int a_c = pairs[pair_c][0], b_c = pairs[pair_c][1];
            double fr = FF_trunc_re[rr*n+cc], fi = FF_trunc_im[rr*n+cc];
            double sr_a = TSVD_INV_SQRT2;
            double sr_b = is_wave_r ? -TSVD_INV_SQRT2 : TSVD_INV_SQRT2;
            double sc_a = TSVD_INV_SQRT2;
            double sc_b = is_wave_c ? -TSVD_INV_SQRT2 : TSVD_INV_SQRT2;
            trunc_re[a_r*D+a_c] += sr_a*sc_a*fr; trunc_im[a_r*D+a_c] += sr_a*sc_a*fi;
            trunc_re[a_r*D+b_c] += sr_a*sc_b*fr; trunc_im[a_r*D+b_c] += sr_a*sc_b*fi;
            trunc_re[b_r*D+a_c] += sr_b*sc_a*fr; trunc_im[b_r*D+a_c] += sr_b*sc_a*fi;
            trunc_re[b_r*D+b_c] += sr_b*sc_b*fr; trunc_im[b_r*D+b_c] += sr_b*sc_b*fi;
        }
    }

    double fold_trunc_err2 = 0;
    for (int i = 0; i < m*n; i++) {
        double dr = Th_re[i]-trunc_re[i], di = Th_im[i]-trunc_im[i];
        fold_trunc_err2 += dr*dr + di*di;
    }
    *trunc3_fold_err_out = (orig2>1e-30) ? sqrt(fold_trunc_err2/orig2) : 0;

    /* SVD rank-3 reconstruction error */
    double svd_trunc_re[D*D], svd_trunc_im[D*D];
    memset(svd_trunc_re, 0, sizeof(svd_trunc_re));
    memset(svd_trunc_im, 0, sizeof(svd_trunc_im));
    for (int s = 0; s < 3; s++) {
        if (sigma[s] < 1e-30) continue;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) {
                double ur = sU_re[i*D+s], ui = sU_im[i*D+s];
                double vr = sV_re[s*n+j], vi = sV_im[s*n+j];
                svd_trunc_re[i*n+j] += sigma[s] * (ur*vr - ui*vi);
                svd_trunc_im[i*n+j] += sigma[s] * (ur*vi + ui*vr);
            }
    }
    double svd_trunc_err2 = 0;
    for (int i = 0; i < m*n; i++) {
        double dr = Th_re[i]-svd_trunc_re[i], di = Th_im[i]-svd_trunc_im[i];
        svd_trunc_err2 += dr*dr + di*di;
    }
    *trunc3_svd_err_out = (orig2>1e-30) ? sqrt(svd_trunc_err2/orig2) : 0;

    /* ── Print ── */
    printf("  %-20s  #%-2d  %5.1f%%  %-3s  %+.3f  %.1e  ",
           name, best_si, best_wf*100,
           naturally_sorted ? "YES" : "NO ", *corr_out, *full_err_out);
    printf("SVD=[");
    for (int i=0;i<D;i++) printf("%.3f%s", sigma[i], i<D-1?",":" ");
    printf("] Fold=[");
    for (int i=0;i<D;i++) printf("%.3f%s", fold_spectrum[i].val, i<D-1?",":" ");
    printf("]\n");
    printf("  %-20s          Rank-3 truncation error: Fold=%.4f  SVD=%.4f  (ratio: %.2f×)\n",
           "", *trunc3_fold_err_out, *trunc3_svd_err_out,
           (*trunc3_svd_err_out > 1e-30) ? *trunc3_fold_err_out / *trunc3_svd_err_out : 0);

    free(FF_re); free(FF_im);
    free(FF_trunc_re); free(FF_trunc_im);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
int main(void)
{
    s6_exotic_init();
    rng_seed(12345);

    printf("\n");
    printf("  ╔════════════════════════════════════════════════════════════════╗\n");
    printf("  ║                                                              ║\n");
    printf("  ║   FULL FOLD (Vesica+Wave) vs SVD — Does Geometry Win?       ║\n");
    printf("  ║                                                              ║\n");
    printf("  ║   Using ALL 6 channels (3 vesica + 3 wave).                 ║\n");
    printf("  ║   Full fold is unitary → reconstruction should be exact.    ║\n");
    printf("  ║   Question: does the spectrum naturally sort?               ║\n");
    printf("  ║   And: can fold rank-3 truncation match SVD rank-3?        ║\n");
    printf("  ║                                                              ║\n");
    printf("  ╚════════════════════════════════════════════════════════════════╝\n\n");

    printf("  %-20s  Syn  Wave%%  Sort  Corr  FullErr  Spectra\n", "Test");
    printf("  %-20s  ───  ─────  ────  ────  ───────  ───────\n", "────────────────────");

    double Th_re[D*D], Th_im[D*D];
    int sorted; double corr, ferr, fterr, sterr;

    /* Structured states */
    mat_bell(Th_re, Th_im);
    run_one("Bell (antipodal)", Th_re, Th_im, &sorted, &corr, &ferr, &fterr, &sterr);

    mat_ghz(Th_re, Th_im);
    run_one("GHZ (diagonal)", Th_re, Th_im, &sorted, &corr, &ferr, &fterr, &sterr);

    mat_w_state(Th_re, Th_im);
    run_one("W-state (cyclic)", Th_re, Th_im, &sorted, &corr, &ferr, &fterr, &sterr);

    mat_dft(Th_re, Th_im);
    run_one("DFT-entangled", Th_re, Th_im, &sorted, &corr, &ferr, &fterr, &sterr);

    mat_rank1(Th_re, Th_im);
    run_one("Rank-1 (|+⟩⊗|+⟩)", Th_re, Th_im, &sorted, &corr, &ferr, &fterr, &sterr);

    mat_rank2(Th_re, Th_im);
    run_one("Rank-2 (|00⟩+|33⟩)", Th_re, Th_im, &sorted, &corr, &ferr, &fterr, &sterr);

    mat_cz_entangled(Th_re, Th_im);
    run_one("CZ-entangled", Th_re, Th_im, &sorted, &corr, &ferr, &fterr, &sterr);

    /* Random matrices */
    printf("\n");
    int total_sorted = 0;
    double sum_corr = 0, sum_ferr = 0, sum_fterr = 0, sum_sterr = 0;
    int N_RAND = 20;

    for (int t = 0; t < N_RAND; t++) {
        char label[32]; snprintf(label, sizeof(label), "Random #%d", t+1);
        mat_random(Th_re, Th_im, D, D);
        run_one(label, Th_re, Th_im, &sorted, &corr, &ferr, &fterr, &sterr);
        total_sorted += sorted;
        sum_corr += corr;
        sum_ferr += ferr;
        sum_fterr += fterr;
        sum_sterr += sterr;
    }

    printf("\n  ═══════════════════════════════════════════════════════════════\n");
    printf("  SUMMARY — %d random matrices\n", N_RAND);
    printf("  ═══════════════════════════════════════════════════════════════\n\n");
    printf("    Naturally sorted:          %d / %d (%.0f%%)\n",
           total_sorted, N_RAND, 100.0*total_sorted/N_RAND);
    printf("    Avg SVD-fold correlation:  %.4f\n", sum_corr/N_RAND);
    printf("    Avg full recon error:      %.2e\n", sum_ferr/N_RAND);
    printf("    Avg rank-3 fold error:     %.4f\n", sum_fterr/N_RAND);
    printf("    Avg rank-3 SVD error:      %.4f\n", sum_sterr/N_RAND);
    printf("    Fold/SVD truncation ratio: %.2f×\n\n",
           (sum_sterr > 1e-30) ? sum_fterr/sum_sterr : 0);

    if (sum_ferr/N_RAND < 1e-10)
        printf("    ✓ FULL FOLD IS LOSSLESS — vesica+wave reconstructs exactly\n");
    if (sum_fterr/sum_sterr < 1.5)
        printf("    ✓ FOLD TRUNCATION COMPETITIVE — within 1.5× of optimal SVD\n");
    if (sum_fterr/sum_sterr < 1.05)
        printf("    ★ FOLD TRUNCATION MATCHES SVD — SVD is REDUNDANT\n");
    if (total_sorted > N_RAND * 0.7)
        printf("    ✓ NATURAL SORTING — the geometry does the work\n");

    printf("\n  ═══════════════════════════════════════════════════════════════\n\n");
    return 0;
}
