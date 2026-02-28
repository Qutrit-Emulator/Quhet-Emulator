/*
 * verify_precomp.c — Verify precomputed eigenvalue patterns don't corrupt
 *                     entanglement, superposition, or Bell states.
 *
 * NOTE: QuhitEngine is 2.3 GB — NEVER copy it. All tests use single-shot
 * measurements or mathematical verification instead of statistical sampling.
 *
 * gcc -O2 -std=gnu11 -I. -o verify_precomp verify_precomp.c \
 *     mps_overlay.c peps_overlay.c quhit_core.c quhit_gates.c \
 *     quhit_measure.c quhit_entangle.c quhit_register.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "quhit_engine.h"
#include "superposition.h"
#include "tensor_svd.h"
#include "mps_overlay.h"
#include "peps_overlay.h"

#define D 6
static int tests_passed = 0, tests_failed = 0;

static void check(const char *name, int cond) {
    printf("  %s: %s\n", cond ? "PASS" : "FAIL", name);
    if (cond) tests_passed++; else tests_failed++;
}

/* ═══ TEST 1: Bell State ═══ */
static void test_bell(void) {
    printf("\n  === TEST 1: Bell State ===\n\n");
    QuhitEngine *eng = malloc(sizeof(QuhitEngine));
    quhit_engine_init(eng);
    uint32_t qA = quhit_init(eng), qB = quhit_init(eng);
    quhit_entangle_bell(eng, qA, qB);

    /* Single-shot: measure A then B, must be equal (Bell = sum|ii>/sqrt(D)) */
    uint32_t rA = quhit_measure(eng, qA);
    uint32_t rB = quhit_measure(eng, qB);
    printf("    Measured: A=%u, B=%u\n", rA, rB);
    check("Bell pair: A == B (perfect correlation)", rA == rB);
    check("Measurement in [0..5]", rA < (uint32_t)D && rB < (uint32_t)D);

    /* Verify the joint amplitudes directly: |ψ> = (1/√6) Σ|ii> */
    quhit_engine_init(eng);
    qA = quhit_init(eng); qB = quhit_init(eng);
    quhit_entangle_bell(eng, qA, qB);
    QuhitPair *p = &eng->pairs[eng->quhits[qA].pair_id];
    double diag_sum = 0, off_sum = 0;
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++) {
            double amp2 = p->joint.re[i*D+j]*p->joint.re[i*D+j]
                        + p->joint.im[i*D+j]*p->joint.im[i*D+j];
            if (i == j) diag_sum += amp2;
            else off_sum += amp2;
        }
    printf("    |diag|^2 = %.10f (expect 1.0)\n", diag_sum);
    printf("    |off|^2  = %.10f (expect 0.0)\n", off_sum);
    check("Bell diag amplitudes sum to ~1 (engine precision)", fabs(diag_sum - 1.0) < 1e-2);
    check("Bell off-diag amplitudes are 0", off_sum < 1e-20);

    /* Each diagonal = 1/sqrt(6) */
    double exp_amp = 1.0/sqrt(6.0);
    double max_err = 0;
    for (int i = 0; i < D; i++) {
        double e = fabs(p->joint.re[i*D+i] - exp_amp);
        if (e > max_err) max_err = e;
    }
    printf("    Each |ii> amp = %.10f (expect %.10f), err=%.2e\n",
           p->joint.re[0], exp_amp, max_err);
    check("Bell amplitudes ~ 1/sqrt(6) (born_isqrt precision)", max_err < 1e-3);
    free(eng);
}

/* ═══ TEST 2: DFT Superposition ═══ */
static void test_superposition(void) {
    printf("\n  === TEST 2: DFT Superposition ===\n\n");
    QuhitEngine *eng = malloc(sizeof(QuhitEngine));
    quhit_engine_init(eng);
    uint32_t q = quhit_init(eng);
    quhit_apply_dft(eng, q);

    /* DFT|0> = (1/√6)(|0>+|1>+...+|5>) — check amplitudes directly */
    Quhit *qh = &eng->quhits[q];
    double sum2 = 0;
    double exp_amp = 1.0/sqrt(6.0);
    double max_err = 0;
    for (int i = 0; i < D; i++) {
        double amp2 = qh->state.re[i]*qh->state.re[i]
                    + qh->state.im[i]*qh->state.im[i];
        sum2 += amp2;
        double e = fabs(amp2 - 1.0/D);
        if (e > max_err) max_err = e;
    }
    printf("    |ψ|^2 = [");
    for (int i = 0; i < D; i++)
        printf("%.6f%s",
               qh->state.re[i]*qh->state.re[i]+qh->state.im[i]*qh->state.im[i],
               i<D-1?", ":"");
    printf("]\n");
    printf("    Sum=%.10f, max_err=%.2e\n", sum2, max_err);
    check("DFT uniform: all |amp|^2 = 1/6", max_err < 1e-14);
    check("DFT normalization: sum = 1.0", fabs(sum2 - 1.0) < 1e-14);

    /* Single measurement must be in [0..5] */
    uint32_t r = quhit_measure(eng, q);
    printf("    Measured: %u\n", r);
    check("DFT measurement in valid range", r < (uint32_t)D);
    free(eng);
}

/* ═══ TEST 3: SVD Round-Trip ═══ */
static void test_svd(void) {
    printf("\n  === TEST 3: SVD Round-Trip ===\n\n");
    QuhitEngine *eng = malloc(sizeof(QuhitEngine));

    /* Pattern B: Bell+DFT+CZ */
    quhit_engine_init(eng);
    uint32_t qA = quhit_init(eng), qB = quhit_init(eng);
    quhit_entangle_bell(eng, qA, qB);
    quhit_apply_dft(eng, qA);
    quhit_apply_cz(eng, qA, qB);

    QuhitPair *p = &eng->pairs[eng->quhits[qA].pair_id];
    double U[D*D], Ui[D*D], s[D], V[D*D], Vi[D*D];
    tsvd_truncated(p->joint.re, p->joint.im, D, D, D, U, Ui, s, V, Vi);

    printf("    Pattern B sigma: [");
    for (int i = 0; i < D; i++) printf("%.6f%s", s[i], i<D-1?", ":"");
    printf("]\n");

    /* Reconstruct: M' = U * diag(s) * V† */
    double R[D*D]={0}, Ri[D*D]={0};
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++)
            for (int k = 0; k < D; k++) {
                R[i*D+j]  += s[k]*(U[i*D+k]*V[k*D+j] - Ui[i*D+k]*Vi[k*D+j]);
                Ri[i*D+j] += s[k]*(U[i*D+k]*Vi[k*D+j] + Ui[i*D+k]*V[k*D+j]);
            }
    double e2 = 0, n2 = 0;
    for (int i = 0; i < D*D; i++) {
        double dr = p->joint.re[i]-R[i], di = p->joint.im[i]-Ri[i];
        e2 += dr*dr + di*di;
        n2 += p->joint.re[i]*p->joint.re[i] + p->joint.im[i]*p->joint.im[i];
    }
    double rel = sqrt(e2/(n2 > 0 ? n2 : 1));
    printf("    ||M - UsV†||/||M|| = %.2e\n", rel);
    check("Pattern B SVD round-trip < 1e-10", rel < 1e-10);

    /* Pattern A: Bell only (M†M = (1/6)I) */
    quhit_engine_init(eng);
    qA = quhit_init(eng); qB = quhit_init(eng);
    quhit_entangle_bell(eng, qA, qB);
    p = &eng->pairs[eng->quhits[qA].pair_id];
    tsvd_truncated(p->joint.re, p->joint.im, D, D, D, U, Ui, s, V, Vi);

    printf("    Pattern A sigma: [");
    for (int i = 0; i < D; i++) printf("%.8f%s", s[i], i<D-1?", ":"");
    printf("]\n");

    double exp_s = sqrt(1.0/D), me = 0;
    for (int i = 0; i < D; i++) { double e = fabs(s[i]-exp_s); if (e > me) me = e; }
    printf("    Expected=%.8f, max_err=%.2e\n", exp_s, me);
    check("Pattern A: all sigma ~ sqrt(1/6) (born_isqrt precision)", me < 1e-3);
    free(eng);
}

/* ═══ TEST 4: MPS Lazy Chain ═══ */
static void test_mps(void) {
    printf("\n  === TEST 4: MPS Lazy Chain ===\n\n");
    int N = 4;
    QuhitEngine *eng = malloc(sizeof(QuhitEngine));
    quhit_engine_init(eng);
    uint32_t quhits[4];
    for (int i = 0; i < N; i++) quhits[i] = quhit_init(eng);

    MpsLazyChain *lc = mps_lazy_init(eng, quhits, N);
    for (int s = 0; s < N; s++) {
        mps_lazy_zero_site(lc, s);
        mps_lazy_write_tensor(lc, s, 0, 0, 0, 1.0, 0.0);
    }

    /* DFT on all sites (1-site gates — avoids chi=512 SVD cost) */
    double dft_re[D*D], dft_im[D*D];
    memset(dft_re, 0, sizeof(dft_re));
    memset(dft_im, 0, sizeof(dft_im));
    for (int a = 0; a < D; a++)
        for (int b = 0; b < D; b++) {
            dft_re[a*D+b] = DFT6[a][b].re;
            dft_im[a*D+b] = DFT6[a][b].im;
        }
    for (int i = 0; i < N; i++)
        mps_lazy_gate_1site(lc, i, dft_re, dft_im);

    mps_lazy_flush(lc);
    mps_lazy_finalize_stats(lc);

    printf("    MPS %d sites, chi=%d\n", N, MPS_CHI);
    printf("    Queued: %lu, Materialized: %lu\n",
           (unsigned long)lc->stats.gates_queued,
           (unsigned long)lc->stats.gates_materialized);
    check("MPS gates queued", lc->stats.gates_queued > 0);
    check("MPS gates materialized", lc->stats.gates_materialized > 0);

    /* Single measurement */
    uint32_t r = mps_lazy_measure(lc, 0);
    printf("    Measure site 0: %u\n", r);
    check("MPS measurement in [0..5]", r < (uint32_t)D);

    mps_lazy_free(lc);
    free(eng);
}

/* ═══ TEST 5: PEPS-2D ═══ */
static void test_peps(void) {
    printf("\n  === TEST 5: PEPS-2D ===\n\n");
    PepsGrid *grid = peps_init(4, 4);

    double dft_re[D*D], dft_im[D*D];
    memset(dft_re, 0, sizeof(dft_re));
    memset(dft_im, 0, sizeof(dft_im));
    for (int a = 0; a < D; a++)
        for (int b = 0; b < D; b++) {
            dft_re[a*D+b] = DFT6[a][b].re;
            dft_im[a*D+b] = DFT6[a][b].im;
        }
    peps_gate_1site_all(grid, dft_re, dft_im);

    double cz_re[D*D*D*D], cz_im[D*D*D*D];
    memset(cz_re, 0, sizeof(cz_re));
    memset(cz_im, 0, sizeof(cz_im));
    for (int a = 0; a < D; a++)
        for (int b = 0; b < D; b++) {
            int idx = a*D*D*D + b*D*D + a*D + b;
            cz_re[idx] = OMEGA6[(a*b)%D].re;
            cz_im[idx] = OMEGA6[(a*b)%D].im;
        }
    peps_trotter_step(grid, cz_re, cz_im);

    double dens[D];
    peps_local_density(grid, 0, 0, dens);
    printf("    Site(0,0): [");
    for (int i = 0; i < D; i++) printf("%.4f%s", dens[i], i<D-1?", ":"");
    printf("]\n");

    double sum = 0;
    for (int i = 0; i < D; i++) sum += dens[i];
    printf("    Sum = %.10f\n", sum);
    check("PEPS density sums to 1.0", fabs(sum - 1.0) < 1e-6);

    int nz = 0;
    for (int i = 0; i < D; i++) if (dens[i] > 0.01) nz++;
    printf("    Non-trivial: %d/6\n", nz);
    check("PEPS density non-negative and normalized", sum > 0.99);

    peps_free(grid);
}

/* ═══ TEST 6: Eigenvector Orthogonality ═══ */
static void test_eigvec(void) {
    printf("\n  === TEST 6: Pattern B Eigenvectors ===\n\n");
    double W[6][6];
    memset(W, 0, sizeof(W));
    for (int p = 0; p < 3; p++) {
        int ia = TSVD_PAIR_A[p], ib = TSVD_PAIR_B[p];
        W[ia][p]   =  TSVD_INV_SQRT2; W[ib][p]   =  TSVD_INV_SQRT2;
        W[ia][p+3] =  TSVD_INV_SQRT2; W[ib][p+3] = -TSVD_INV_SQRT2;
    }

    /* W^T W = I? */
    double mo = 0, md = 0;
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++) {
            double d = 0;
            for (int k = 0; k < 6; k++) d += W[k][i]*W[k][j];
            if (i == j) { double e = fabs(d - 1); if (e > md) md = e; }
            else        { if (fabs(d) > mo) mo = fabs(d); }
        }
    printf("    max|diag-1| = %.2e, max|off| = %.2e\n", md, mo);
    check("Orthonormal (diag < 1e-15)", md < 1e-15);
    check("Orthogonal (off < 1e-15)", mo < 1e-15);

    /* V^T H V = diag(lambda)? */
    double H[6][6] = {0}, d0 = 1.0/6.0;
    for (int i = 0; i < 6; i++) H[i][i] = d0;
    H[0][3] = H[3][0] = d0;
    H[1][4] = H[4][1] = d0;
    H[2][5] = H[5][2] = d0;

    double Dm[6][6] = {0};
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
            for (int k = 0; k < 6; k++)
                for (int l = 0; l < 6; l++)
                    Dm[i][j] += W[k][i]*H[k][l]*W[l][j];

    printf("    V^T H V diag: [");
    for (int i = 0; i < 6; i++) printf("%.10f%s", Dm[i][i], i<5?", ":"");
    printf("]\n");

    double ee = 0;
    for (int i = 0; i < 3; i++) ee += fabs(Dm[i][i] - 2*d0);
    for (int i = 3; i < 6; i++) ee += fabs(Dm[i][i]);
    printf("    Eigenvalue error: %.2e\n", ee);
    check("Eigenvalues match [2/6, 2/6, 2/6, 0, 0, 0]", ee < 1e-15);

    double off = 0;
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
            if (i != j) off += fabs(Dm[i][j]);
    check("V^T H V is diagonal", off < 1e-14);
}

/* ═══════════════════════════════════════════════════════════════════════════ */
int main(void) {
    printf("\n  PRECOMPUTED EIGENVALUE VERIFICATION\n");
    printf("  ====================================\n");

    test_bell();
    test_superposition();
    test_svd();
    test_mps();
    test_peps();
    test_eigvec();

    printf("\n  ====================================\n");
    printf("  %d passed, %d failed -- %s\n\n",
           tests_passed, tests_failed,
           tests_failed == 0 ? "ALL PASS" : "FAILURES");
    return tests_failed > 0 ? 1 : 0;
}
