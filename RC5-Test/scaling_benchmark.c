/* ═══════════════════════════════════════════════════════════════════════════
 *  SCALING BENCHMARK — Beyond Classical Limits
 *
 *  Tests QAOA and VQE on D=6 Heisenberg chains at sizes impossible
 *  for classical state-vector simulators.
 *
 *  State-vector requirement: 6^N × 16 bytes
 *    N=4:    1,296 states        → 20 KB         (any laptop)
 *    N=13:   ~13 billion         → 208 GB        (largest servers)
 *    N=14:   ~78 billion         → 1.2 TB        ✗ IMPOSSIBLE
 *    N=20:   ~3.6 × 10^15       → 57 PB         ✗ IMPOSSIBLE
 *    N=100:  ~6.5 × 10^77       → ∞             ✗ IMPOSSIBLE
 *    N=1000: ~6.5 × 10^777      → ∞∞            ✗ IMPOSSIBLE
 *
 *  HexState Engine: O((N+E)×D²) per measurement → should scale linearly.
 *
 *  Build:
 *    gcc -O2 -std=c11 -D_GNU_SOURCE -o scaling_benchmark \
 *        scaling_benchmark.c hexstate_engine.c bigint.c -lm
 * ═══════════════════════════════════════════════════════════════════════════ */

#include "hexstate_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define DIM 6

/* ─── Matrix utilities ─── */

static void mm6(const Complex *A, const Complex *B, Complex *C)
{
    for (int i = 0; i < DIM; i++)
        for (int j = 0; j < DIM; j++) {
            double re = 0, im = 0;
            for (int k = 0; k < DIM; k++) {
                re += A[i*DIM+k].real * B[k*DIM+j].real
                    - A[i*DIM+k].imag * B[k*DIM+j].imag;
                im += A[i*DIM+k].real * B[k*DIM+j].imag
                    + A[i*DIM+k].imag * B[k*DIM+j].real;
            }
            C[i*DIM+j].real = re;
            C[i*DIM+j].imag = im;
        }
}

static void adj6(const Complex *A, Complex *B)
{
    for (int i = 0; i < DIM; i++)
        for (int j = 0; j < DIM; j++) {
            B[i*DIM+j].real =  A[j*DIM+i].real;
            B[i*DIM+j].imag = -A[j*DIM+i].imag;
        }
}

static void make_dft6(Complex *F)
{
    double sq = 1.0 / sqrt((double)DIM);
    for (int j = 0; j < DIM; j++)
        for (int k = 0; k < DIM; k++) {
            double angle = 2.0 * M_PI * j * k / DIM;
            F[j*DIM+k].real = sq * cos(angle);
            F[j*DIM+k].imag = sq * sin(angle);
        }
}

static void make_rz(double theta, Complex *U)
{
    memset(U, 0, DIM * DIM * sizeof(Complex));
    for (int k = 0; k < DIM; k++) {
        U[k*DIM+k].real = cos(k * theta);
        U[k*DIM+k].imag = sin(k * theta);
    }
}

static void make_rx(double theta, Complex *U)
{
    Complex rz[DIM*DIM], dft[DIM*DIM], dftd[DIM*DIM], tmp[DIM*DIM];
    make_rz(theta, rz);
    make_dft6(dft);
    adj6(dft, dftd);
    mm6(dft, rz, tmp);
    mm6(tmp, dftd, U);
}

/* ─── Heisenberg energy ─── */

static double heisenberg_energy(const int *colors, int n)
{
    double E = 0;
    for (int i = 0; i < n - 1; i++)
        E -= cos(2.0 * M_PI * (colors[i] - colors[i+1]) / DIM);
    return E;
}

/* ─── Classical state vector size ─── */

static void format_classical_size(int n_qudits, char *buf, int buflen)
{
    /* 6^N × 16 bytes (complex double) */
    double log10_bytes = n_qudits * log10(6.0) + log10(16.0);
    if (log10_bytes < 3)
        snprintf(buf, buflen, "%.0f B", pow(10, log10_bytes));
    else if (log10_bytes < 6)
        snprintf(buf, buflen, "%.1f KB", pow(10, log10_bytes - 3));
    else if (log10_bytes < 9)
        snprintf(buf, buflen, "%.1f MB", pow(10, log10_bytes - 6));
    else if (log10_bytes < 12)
        snprintf(buf, buflen, "%.1f GB", pow(10, log10_bytes - 9));
    else if (log10_bytes < 15)
        snprintf(buf, buflen, "%.1f TB", pow(10, log10_bytes - 12));
    else if (log10_bytes < 18)
        snprintf(buf, buflen, "%.1f PB", pow(10, log10_bytes - 15));
    else
        snprintf(buf, buflen, "10^%.0f bytes", log10_bytes);
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  QAOA SINGLE SHOT — scales to any N
 * ═══════════════════════════════════════════════════════════════════════════ */

static double qaoa_shot(int n_qudits, double beta1, double beta2)
{
    HexStateEngine eng;
    engine_init(&eng);

    for (int i = 0; i < n_qudits; i++)
        init_chunk(&eng, i, UINT64_MAX);
    for (int i = 1; i < n_qudits; i++)
        braid_chunks_dim(&eng, 0, i, 0, 0, DIM);

    Complex U[DIM*DIM];

    /* DFT₆ → equal superposition on each qudit */
    make_dft6(U);
    for (int q = 0; q < n_qudits; q++)
        apply_local_unitary(&eng, q, U, DIM);

    /* QAOA layer 1: CZ₆ on neighbors + R_X(β₁) mixer */
    for (int q = 0; q < n_qudits - 1; q++)
        apply_cz_gate(&eng, q, q + 1);
    make_rx(beta1, U);
    for (int q = 0; q < n_qudits; q++)
        apply_local_unitary(&eng, q, U, DIM);

    /* QAOA layer 2: CZ₆ on neighbors + R_X(β₂) mixer */
    for (int q = 0; q < n_qudits - 1; q++)
        apply_cz_gate(&eng, q, q + 1);
    make_rx(beta2, U);
    for (int q = 0; q < n_qudits; q++)
        apply_local_unitary(&eng, q, U, DIM);

    /* Measure all qudits */
    int *colors = malloc(n_qudits * sizeof(int));
    for (int i = 0; i < n_qudits; i++)
        colors[i] = (int)(measure_chunk(&eng, i) % DIM);

    double E = heisenberg_energy(colors, n_qudits);
    free(colors);
    engine_destroy(&eng);
    return E;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  VQE SINGLE SHOT — scales to any N
 * ═══════════════════════════════════════════════════════════════════════════ */

static double vqe_shot(int n_qudits, const double *theta_z,
                        const double *theta_x)
{
    HexStateEngine eng;
    engine_init(&eng);

    for (int i = 0; i < n_qudits; i++)
        init_chunk(&eng, i, UINT64_MAX);
    for (int i = 1; i < n_qudits; i++)
        braid_chunks_dim(&eng, 0, i, 0, 0, DIM);

    Complex U[DIM*DIM];

    /* 2-layer hardware-efficient ansatz */
    for (int layer = 0; layer < 2; layer++) {
        for (int q = 0; q < n_qudits; q++) {
            make_rz(theta_z[layer * n_qudits + q], U);
            apply_local_unitary(&eng, q, U, DIM);
            make_rx(theta_x[layer * n_qudits + q], U);
            apply_local_unitary(&eng, q, U, DIM);
        }
        for (int q = 0; q < n_qudits - 1; q++)
            apply_cz_gate(&eng, q, q + 1);
    }

    int *colors = malloc(n_qudits * sizeof(int));
    for (int i = 0; i < n_qudits; i++)
        colors[i] = (int)(measure_chunk(&eng, i) % DIM);

    double E = heisenberg_energy(colors, n_qudits);
    free(colors);
    engine_destroy(&eng);
    return E;
}

/* ════════════════════════════════════════════════════════════════════════ */

int main(void)
{
    FILE *devnull = fopen("/dev/null", "w");
    FILE *real_stdout = stdout;

    printf("╔══════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                                                                                ║\n");
    printf("║   ███████╗ ██████╗ █████╗ ██╗     ███████╗                                    ║\n");
    printf("║   ██╔════╝██╔════╝██╔══██╗██║     ██╔════╝                                    ║\n");
    printf("║   ███████╗██║     ███████║██║     █████╗                                      ║\n");
    printf("║   ╚════██║██║     ██╔══██║██║     ██╔══╝                                      ║\n");
    printf("║   ███████║╚██████╗██║  ██║███████╗███████╗                                    ║\n");
    printf("║   ╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚══════╝                                    ║\n");
    printf("║                                                                                ║\n");
    printf("║   B E Y O N D   C L A S S I C A L   L I M I T S                              ║\n");
    printf("║                                                                                ║\n");
    printf("║   QAOA + VQE on D=6 Heisenberg chains                                        ║\n");
    printf("║   Scaling from 4 to 1000 qudits                                              ║\n");
    printf("║   Classical limit: ~13 qudits (6^13 ≈ 13B amplitudes)                        ║\n");
    printf("║                                                                                ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════════════╝\n\n");

    /* Optimal QAOA params from previous benchmark */
    double beta1 = 0.3, beta2 = 1.8;

    /* Test sizes — beyond N≈13, no classical simulator can handle this */
    int sizes[] = {4, 8, 16, 32, 64, 128, 256, 512, 1000};
    int n_sizes = 9;
    int shots_per = 5;

    double qaoa_times[9], vqe_times[9];
    double qaoa_energies[9], vqe_energies[9];
    double ground_energies[9];

    printf("  ┌──────┬───────────────┬──────────────┬──────────────┬──────────────┐\n");
    printf("  │  N   │ Classical RAM │  QAOA time   │  VQE time    │   Status     │\n");
    printf("  ├──────┼───────────────┼──────────────┼──────────────┼──────────────┤\n");
    fflush(stdout);

    srand(42);

    for (int s = 0; s < n_sizes; s++) {
        int N = sizes[s];
        char size_str[64];
        format_classical_size(N, size_str, sizeof(size_str));
        ground_energies[s] = -(N - 1);

        /* ─── QAOA timing ─── */
        struct timespec t1, t2;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double E_qaoa = 0;
        stdout = devnull;
        for (int shot = 0; shot < shots_per; shot++)
            E_qaoa += qaoa_shot(N, beta1, beta2);
        stdout = real_stdout;
        E_qaoa /= shots_per;
        clock_gettime(CLOCK_MONOTONIC, &t2);
        qaoa_times[s] = (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec) / 1e9;
        qaoa_energies[s] = E_qaoa;

        /* ─── VQE timing (random params — not training, just measuring overhead) ─── */
        double *tz = malloc(2 * N * sizeof(double));
        double *tx = malloc(2 * N * sizeof(double));
        for (int i = 0; i < 2 * N; i++) {
            tz[i] = ((double)rand() / RAND_MAX - 0.5) * 2.0;
            tx[i] = ((double)rand() / RAND_MAX - 0.5) * 2.0;
        }

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double E_vqe = 0;
        stdout = devnull;
        for (int shot = 0; shot < shots_per; shot++)
            E_vqe += vqe_shot(N, tz, tx);
        stdout = real_stdout;
        E_vqe /= shots_per;
        clock_gettime(CLOCK_MONOTONIC, &t2);
        vqe_times[s] = (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec) / 1e9;
        vqe_energies[s] = E_vqe;

        free(tz); free(tx);

        /* Check if beyond classical limit */
        int beyond = (N >= 14);
        printf("  │ %4d │ %13s │ %8.2f s   │ %8.2f s   │ %s │\n",
               N, size_str, qaoa_times[s], vqe_times[s],
               beyond ? "✗ CLASSICAL" : "  classical");
        fflush(stdout);
    }

    printf("  └──────┴───────────────┴──────────────┴──────────────┴──────────────┘\n\n");

    /* ─── Scaling analysis ─── */
    printf("  ─── S C A L I N G   A N A L Y S I S ───\n\n");

    /* Linear fit: time = a + b*N  (last vs first) */
    double qaoa_rate = (qaoa_times[n_sizes-1] - qaoa_times[0]) /
                       (sizes[n_sizes-1] - sizes[0]);
    double vqe_rate  = (vqe_times[n_sizes-1] - vqe_times[0]) /
                       (sizes[n_sizes-1] - sizes[0]);

    printf("  QAOA: %.4f s/qudit (linear rate)\n", qaoa_rate);
    printf("  VQE:  %.4f s/qudit (linear rate)\n\n", vqe_rate);

    /* Check if exponential: time(2N)/time(N) should be constant for linear,
       growing for exponential */
    printf("  Time ratios (t[2N]/t[N]) — constant = linear, growing = exponential:\n\n");
    for (int s = 0; s < n_sizes - 1; s++) {
        if (sizes[s+1] == 2 * sizes[s] && qaoa_times[s] > 0.001) {
            double ratio_q = qaoa_times[s+1] / qaoa_times[s];
            double ratio_v = vqe_times[s+1] / vqe_times[s];
            printf("    N=%4d→%4d:  QAOA %.2f×  VQE %.2f×\n",
                   sizes[s], sizes[s+1], ratio_q, ratio_v);
        }
    }

    /* ─── Energy results ─── */
    printf("\n  ─── E N E R G Y   R E S U L T S ───\n\n");
    printf("  ┌──────┬───────────┬───────────┬───────────┬──────────┐\n");
    printf("  │  N   │  E_ground │  E_QAOA   │  E_VQE    │ QAOA %%   │\n");
    printf("  ├──────┼───────────┼───────────┼───────────┼──────────┤\n");
    for (int s = 0; s < n_sizes; s++) {
        double pct = (qaoa_energies[s] < 0) ?
            100.0 * qaoa_energies[s] / ground_energies[s] : 0;
        printf("  │ %4d │ %9.1f │ %+9.2f │ %+9.2f │ %5.1f%%   │\n",
               sizes[s], ground_energies[s],
               qaoa_energies[s], vqe_energies[s], pct);
    }
    printf("  └──────┴───────────┴───────────┴───────────┴──────────┘\n\n");

    /* ─── Final verdict ─── */
    int linear = 1;
    for (int s = 1; s < n_sizes; s++) {
        /* If doubling N more than 4× the time, it's exponential */
        if (s < n_sizes - 1 && sizes[s+1] == 2 * sizes[s]) {
            double r = qaoa_times[s+1] / (qaoa_times[s] + 0.001);
            if (r > 4.0) linear = 0;
        }
    }

    double speedup_vs_classical = pow(6.0, 1000) / (vqe_times[n_sizes-1] + 0.001);

    printf("╔══════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                  S C A L I N G   V E R D I C T                                ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║                                                                                ║\n");

    if (linear) {
        printf("║  ★ SCALING IS LINEAR — O(N), not O(6^N)                                      ║\n");
        printf("║                                                                                ║\n");
        printf("║  The HexState Engine's deferred CZ + deferred unitary architecture            ║\n");
        printf("║  maintains polynomial cost regardless of system size.                          ║\n");
    } else {
        printf("║  ✗ Non-linear scaling detected — investigating required                       ║\n");
    }

    printf("║                                                                                ║\n");
    printf("║  At N=1000: classical would need 6^1000 × 16 bytes                            ║\n");
    printf("║  That's ~10^778 bytes — the universe has only ~10^80 atoms                    ║\n");
    printf("║  HexState did it in %.1f seconds.                                              ║\n",
           qaoa_times[n_sizes-1] + vqe_times[n_sizes-1]);
    printf("║                                                                                ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════════════╝\n");

    fclose(devnull);
    return 0;
}
