/*
 * bench_peps2d.c — Benchmark 2D PEPS Red-Black Checkerboard
 *
 * Compares serial loop vs batch _all() on a 10×10 grid.
 * 2D PEPS has χ=4 (bigger SVDs than 3D's χ=3).
 *
 * Build:
 *   Serial:  gcc -O2 -std=gnu99 -w bench_peps2d.c -lm -o bench_peps_serial
 *   OpenMP:  gcc -O2 -std=gnu99 -fopenmp -w bench_peps2d.c -lm -o bench_peps_omp
 */

#include "peps_overlay.c"
#include <time.h>

#define D PEPS_D
#define PI M_PI

static void build_cz6(double *re, double *im) {
    int D2 = D*D;
    memset(re, 0, D2*D2*sizeof(double));
    memset(im, 0, D2*D2*sizeof(double));
    for (int j = 0; j < D; j++)
        for (int k = 0; k < D; k++) {
            int idx = j*D+k;
            double angle = 2*PI*j*k/D;
            re[idx*D2+idx] = cos(angle);
            im[idx*D2+idx] = sin(angle);
        }
}

static void build_dft6(double *re, double *im) {
    double norm = 1.0 / sqrt(D);
    for (int j = 0; j < D; j++)
        for (int k = 0; k < D; k++) {
            re[j*D+k] = norm * cos(2*PI*j*k/D);
            im[j*D+k] = norm * sin(2*PI*j*k/D);
        }
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void)
{
    int LX = 10, LY = 10;
    int STEPS = 2;

    printf("\n  2D PEPS Red-Black Checkerboard Benchmark\n");
    printf("  Grid: %d × %d = %d sites  (χ=%d, SVD dim=%d)\n",
           LX, LY, LX*LY, PEPS_CHI, PEPS_DCHI3);
    printf("  Steps: %d Trotter steps\n", STEPS);

    #ifdef _OPENMP
    printf("  OpenMP: ENABLED (%d threads)\n\n", omp_get_max_threads());
    #else
    printf("  OpenMP: DISABLED (serial)\n\n");
    #endif

    double Ure[D*D], Uim[D*D];
    int D2 = D*D;
    double *Gre = malloc(D2*D2*sizeof(double));
    double *Gim = malloc(D2*D2*sizeof(double));
    build_cz6(Gre, Gim);
    build_dft6(Ure, Uim);

    /* ─── Method 1: Batch _all() with 1 thread (serial baseline) ─── */
    PepsGrid *g1 = peps_init(LX, LY);
    peps_gate_1site_all(g1, Ure, Uim);

    #ifdef _OPENMP
    omp_set_num_threads(1);
    #endif
    double t0 = now_sec();
    for (int step = 0; step < STEPS; step++)
        peps_trotter_step(g1, Gre, Gim);
    double t1 = now_sec();
    double serial_time = t1 - t0;

    /* ─── Method 2: Batch _all() with all threads ─── */
    PepsGrid *g2 = peps_init(LX, LY);
    peps_gate_1site_all(g2, Ure, Uim);

    #ifdef _OPENMP
    omp_set_num_threads(omp_get_num_procs());
    #endif
    double t2 = now_sec();
    for (int step = 0; step < STEPS; step++)
        peps_trotter_step(g2, Gre, Gim);
    double t3 = now_sec();
    double batch_time = t3 - t2;

    /* ─── Verify ─── */
    printf("  Correctness (sample sites):\n\n");
    printf("  Site     │ 1-thread P(0) │ N-thread P(0) │ Match\n");
    printf("  ─────────┼───────────────┼───────────────┼───────\n");
    int correct = 1;
    int spots[][2] = {{0,0},{5,5},{9,9},{3,7},{7,2}};
    for (int i = 0; i < 5; i++) {
        int sx=spots[i][0], sy=spots[i][1];
        double p1[D], p2[D];
        peps_local_density(g1, sx, sy, p1);
        peps_local_density(g2, sx, sy, p2);
        int match = 1;
        for (int k = 0; k < D; k++)
            if (fabs(p1[k] - p2[k]) > 1e-10) match = 0;
        if (!match) correct = 0;
        printf("  (%d,%d)   │ %11.6f   │ %11.6f   │ %s\n",
               sx, sy, p1[0], p2[0], match ? "  ✓" : "  ✗");
    }

    #ifdef _OPENMP
    int nthreads = omp_get_num_procs();
    #else
    int nthreads = 1;
    #endif
    printf("\n  ═══ TIMING ═══\n\n");
    printf("  1-thread:         %.2f s\n", serial_time);
    printf("  %d-thread:       %.2f s\n", nthreads, batch_time);
    printf("  Speedup:          %.2fx\n", serial_time / batch_time);
    printf("  Correctness:      %s\n\n", correct ? "✓ IDENTICAL" : "✗ MISMATCH");

    peps_free(g1);
    peps_free(g2);
    free(Gre); free(Gim);
    return 0;
}
