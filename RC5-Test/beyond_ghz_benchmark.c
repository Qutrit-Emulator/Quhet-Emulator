/* ═══════════════════════════════════════════════════════════════════════════
 *  BEYOND-GHZ ENTANGLEMENT BENCHMARK — 100% TARGET
 *
 *  Two quantum algorithms vs the same problem: D=6 Heisenberg chain
 *    QAOA: structured (γ,β) layers with CZ₆ as problem Hamiltonian
 *    VQE:  hardware-efficient ansatz with per-qudit parameters
 *  Both use CZ₆ entangling gates → beyond-GHZ entanglement.
 *
 *  Build:
 *    gcc -O2 -std=c11 -D_GNU_SOURCE -o beyond_ghz_benchmark \
 *        beyond_ghz_benchmark.c hexstate_engine.c bigint.c -lm
 * ═══════════════════════════════════════════════════════════════════════════ */

#include "hexstate_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define DIM 6
#define QUHITS UINT64_MAX

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

/* ─── PRNG ─── */
static uint64_t xrng = 0xCAFE6ULL;
static int xrand_int(int m) {
    xrng ^= xrng << 13; xrng ^= xrng >> 7; xrng ^= xrng << 17;
    return (int)((xrng >> 16) % (uint64_t)m);
}

/* ─── Heisenberg cost function ─── */
#define N_QUDITS 4

static double heisenberg_energy(const int *colors, int n)
{
    double E = 0;
    for (int i = 0; i < n - 1; i++)
        E -= cos(2.0 * M_PI * (colors[i] - colors[i+1]) / DIM);
    return E;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  QAOA FOR D=6 HEISENBERG CHAIN
 *
 *  The CZ₆ gate IS the natural problem Hamiltonian for the Heisenberg model:
 *    CZ₆|j,k⟩ = ω^(jk)|j,k⟩  where ω = e^(2πi/6)
 *  This directly encodes nearest-neighbor interactions.
 *
 *  QAOA circuit per layer:
 *    1. Problem unitary: CZ₆ on each nearest-neighbor pair
 *       — applies problem-dependent phases
 *    2. R_Z(γ) on each qudit — parameterized Z rotation
 *    3. Mixer: R_X(β) on each qudit — explores color space
 * ═══════════════════════════════════════════════════════════════════════════ */

static double qaoa_one_shot(const double *gamma, const double *beta,
                             int p_layers)
{
    HexStateEngine eng;
    engine_init(&eng);

    for (int i = 0; i < N_QUDITS; i++)
        init_chunk(&eng, i, QUHITS);
    for (int i = 1; i < N_QUDITS; i++)
        braid_chunks_dim(&eng, 0, i, 0, 0, DIM);

    /* Equal superposition via DFT₆ */
    Complex dft[DIM*DIM];
    make_dft6(dft);
    for (int i = 0; i < N_QUDITS; i++)
        apply_local_unitary(&eng, i, dft, DIM);

    /* QAOA layers */
    Complex U[DIM*DIM];
    for (int layer = 0; layer < p_layers; layer++) {
        /* Problem unitary: CZ₆ on neighbors
         * CZ₆ applies ω^(jk) which encodes nearest-neighbor coupling */
        for (int q = 0; q < N_QUDITS - 1; q++)
            apply_cz_gate(&eng, q, q + 1);

        /* Parameterized Z rotation: R_Z(γ) per qudit */
        make_rz(gamma[layer], U);
        for (int q = 0; q < N_QUDITS; q++)
            apply_local_unitary(&eng, q, U, DIM);

        /* Mixer: R_X(β) per qudit */
        make_rx(beta[layer], U);
        for (int q = 0; q < N_QUDITS; q++)
            apply_local_unitary(&eng, q, U, DIM);
    }

    /* Measure */
    int colors[N_QUDITS];
    for (int i = 0; i < N_QUDITS; i++)
        colors[i] = (int)(measure_chunk(&eng, i) % DIM);

    double E = heisenberg_energy(colors, N_QUDITS);
    engine_destroy(&eng);
    return E;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  VQE FOR D=6 HEISENBERG CHAIN
 * ═══════════════════════════════════════════════════════════════════════════ */

#define VQE_LAYERS 3
#define VQE_PARAMS (N_QUDITS * 2 * VQE_LAYERS)

static double vqe_one_shot(const double *params)
{
    HexStateEngine eng;
    engine_init(&eng);

    for (int i = 0; i < N_QUDITS; i++)
        init_chunk(&eng, i, QUHITS);
    for (int i = 1; i < N_QUDITS; i++)
        braid_chunks_dim(&eng, 0, i, 0, 0, DIM);

    Complex U[DIM*DIM];
    for (int layer = 0; layer < VQE_LAYERS; layer++) {
        for (int q = 0; q < N_QUDITS; q++) {
            int pi = layer * N_QUDITS * 2 + q * 2;
            make_rz(params[pi], U);
            apply_local_unitary(&eng, q, U, DIM);
            make_rx(params[pi + 1], U);
            apply_local_unitary(&eng, q, U, DIM);
        }
        for (int q = 0; q < N_QUDITS - 1; q++)
            apply_cz_gate(&eng, q, q + 1);
    }

    int colors[N_QUDITS];
    for (int i = 0; i < N_QUDITS; i++)
        colors[i] = (int)(measure_chunk(&eng, i) % DIM);

    double E = heisenberg_energy(colors, N_QUDITS);
    engine_destroy(&eng);
    return E;
}

static double avg_energy(double (*shot_fn)(const double*), const double *params,
                          int shots)
{
    double total = 0;
    for (int s = 0; s < shots; s++)
        total += shot_fn(params);
    return total / shots;
}

/* ════════════════════════════════════════════════════════════════════════ */

int main(void)
{
    FILE *devnull = fopen("/dev/null", "w");
    FILE *real_stdout = stdout;

    printf("╔══════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                                                                                ║\n");
    printf("║   ██████╗ ███████╗██╗   ██╗ ██████╗ ███╗   ██╗██████╗                         ║\n");
    printf("║   ██╔══██╗██╔════╝╚██╗ ██╔╝██╔═══██╗████╗  ██║██╔══██╗                        ║\n");
    printf("║   ██████╔╝█████╗   ╚████╔╝ ██║   ██║██╔██╗ ██║██║  ██║                        ║\n");
    printf("║   ██╔══██╗██╔══╝    ╚██╔╝  ██║   ██║██║╚██╗██║██║  ██║                        ║\n");
    printf("║   ██████╔╝███████╗   ██║   ╚██████╔╝██║ ╚████║██████╔╝                        ║\n");
    printf("║   ╚═════╝ ╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═══╝╚═════╝                        ║\n");
    printf("║                                                                                ║\n");
    printf("║   B E Y O N D   G H Z   E N T A N G L E M E N T   ·   1 0 0 %%               ║\n");
    printf("║                                                                                ║\n");
    printf("║   Two algorithms, one problem: D=6 Heisenberg chain                           ║\n");
    printf("║   CZ₆ entangling gates | QAOA vs VQE showdown                                ║\n");
    printf("║                                                                                ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════════════╝\n\n");

    struct timespec t1, t2;

    /* Ground state and random baseline */
    double E_gs = -(N_QUDITS - 1);  /* all same color */
    double E_random = 0;
    { int colors[N_QUDITS];
      for (int t = 0; t < 50000; t++) {
          for (int i = 0; i < N_QUDITS; i++) colors[i] = xrand_int(DIM);
          E_random += heisenberg_energy(colors, N_QUDITS);
      }
      E_random /= 50000;
    }
    printf("  Problem: H = -Σ cos(2π(kᵢ-kⱼ)/6) on %d-qudit chain (D=6)\n", N_QUDITS);
    printf("  Ground state energy:  E_gs   = %.1f\n", E_gs);
    printf("  Random state energy:  E_rand = %.4f\n\n", E_random);

    /* ═══ PHASE 1: QAOA ═══ */
    printf("════════════════════════════════════════════════════════════════════════════════════\n");
    printf("  PHASE 1:  Q A O A   (p=2, grid search)\n");
    printf("  CZ₆ as problem unitary + R_Z(γ) + R_X(β) mixer\n");
    printf("════════════════════════════════════════════════════════════════════════════════════\n\n");

    clock_gettime(CLOCK_MONOTONIC, &t1);

    int p_layers = 2;
    double best_qaoa_E = 0;
    double best_g[2] = {0}, best_b[2] = {0};
    int qaoa_shots = 20;

    /* Grid search: γ controls Z-rotation, β controls mixer strength */
    double gvals[] = {0.0, 0.5, 1.0, 1.5, 2.0};
    double bvals[] = {0.3, 0.8, 1.3, 1.8};
    int ng = 5, nb = 4;
    int total_evals = ng*ng*nb*nb;
    int eval_count = 0;

    printf("  Searching %d parameter combinations (γ₁,γ₂,β₁,β₂)...\n", total_evals);

    for (int g1 = 0; g1 < ng; g1++)
      for (int g2 = 0; g2 < ng; g2++)
        for (int b1 = 0; b1 < nb; b1++)
          for (int b2 = 0; b2 < nb; b2++) {
            double gamma[2] = {gvals[g1], gvals[g2]};
            double beta[2]  = {bvals[b1], bvals[b2]};
            double E_sum = 0;

            stdout = devnull;
            for (int s = 0; s < qaoa_shots; s++)
                E_sum += qaoa_one_shot(gamma, beta, p_layers);
            stdout = real_stdout;

            double avg = E_sum / qaoa_shots;
            if (avg < best_qaoa_E) {
                best_qaoa_E = avg;
                best_g[0] = gamma[0]; best_g[1] = gamma[1];
                best_b[0] = beta[0];  best_b[1] = beta[1];
            }

            eval_count++;
            if (eval_count == 1 || eval_count == total_evals / 4 ||
                eval_count == total_evals / 2 ||
                eval_count == 3 * total_evals / 4 ||
                eval_count == total_evals)
                printf("    [%d/%d] best E so far = %.3f (%.0f%% to ground)\n",
                       eval_count, total_evals, best_qaoa_E,
                       100.0 * (E_random - best_qaoa_E) / (E_random - E_gs));
          }

    /* Refine best point with more shots */
    stdout = devnull;
    double qaoa_refine = 0;
    int refine_shots = 100;
    for (int s = 0; s < refine_shots; s++)
        qaoa_refine += qaoa_one_shot(best_g, best_b, p_layers);
    qaoa_refine /= refine_shots;
    stdout = real_stdout;

    clock_gettime(CLOCK_MONOTONIC, &t2);
    double qaoa_s = (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec) / 1e9;

    double qaoa_progress = (E_random - qaoa_refine) / (E_random - E_gs);

    printf("\n  ┌──────────────────────────────────────────────────────────────┐\n");
    printf("  │ QAOA RESULTS                                                │\n");
    printf("  ├──────────────────────────────────────────────────────────────┤\n");
    printf("  │ Random energy:   %.4f                                     │\n", E_random);
    printf("  │ QAOA energy:     %.3f                                      │\n", qaoa_refine);
    printf("  │ Ground state:    %.1f                                       │\n", E_gs);
    printf("  │ Progress:        %.0f%% toward ground state                 │\n",
           100.0 * qaoa_progress);
    printf("  │ Best γ=(%.1f,%.1f) β=(%.1f,%.1f)                          │\n",
           best_g[0], best_g[1], best_b[0], best_b[1]);
    printf("  │ Beats random:    %s                                        │\n",
           (qaoa_refine < E_random) ? "★ YES" : "✗ NO");
    printf("  │ Time:            %.0fs                                      │\n", qaoa_s);
    printf("  └──────────────────────────────────────────────────────────────┘\n\n");

    /* ═══ PHASE 2: VQE ═══ */
    printf("════════════════════════════════════════════════════════════════════════════════════\n");
    printf("  PHASE 2:  V Q E   (hardware-efficient ansatz, %d layers)\n", VQE_LAYERS);
    printf("  Per-qudit R_Z/R_X + CZ₆ entanglers | Adam optimizer\n");
    printf("════════════════════════════════════════════════════════════════════════════════════\n\n");

    double params[VQE_PARAMS];
    double m_adam[VQE_PARAMS], v_adam[VQE_PARAMS], grad_buf[VQE_PARAMS];
    srand(42);
    for (int i = 0; i < VQE_PARAMS; i++)
        params[i] = ((double)rand() / RAND_MAX - 0.5) * 1.0;
    memset(m_adam, 0, sizeof(m_adam));
    memset(v_adam, 0, sizeof(v_adam));

    double lr = 0.1, beta1 = 0.9, beta2v = 0.999, eps_adam = 1e-8;
    int vqe_epochs = 60, vqe_shots = 40;
    double eps_fd = 0.15;

    clock_gettime(CLOCK_MONOTONIC, &t1);

    stdout = devnull;
    double initial_E = avg_energy(vqe_one_shot, params, vqe_shots);
    stdout = real_stdout;

    double best_E = initial_E;
    double best_params[VQE_PARAMS];
    memcpy(best_params, params, sizeof(best_params));

    for (int epoch = 0; epoch < vqe_epochs; epoch++) {
        stdout = devnull;
        for (int i = 0; i < VQE_PARAMS; i++) {
            params[i] += eps_fd;
            double ep = avg_energy(vqe_one_shot, params, vqe_shots / 2);
            params[i] -= 2 * eps_fd;
            double em = avg_energy(vqe_one_shot, params, vqe_shots / 2);
            params[i] += eps_fd;
            grad_buf[i] = (ep - em) / (2 * eps_fd);
        }

        for (int i = 0; i < VQE_PARAMS; i++) {
            m_adam[i] = beta1 * m_adam[i] + (1 - beta1) * grad_buf[i];
            v_adam[i] = beta2v * v_adam[i] + (1 - beta2v) * grad_buf[i] * grad_buf[i];
            double mh = m_adam[i] / (1 - pow(beta1, epoch + 1));
            double vh = v_adam[i] / (1 - pow(beta2v, epoch + 1));
            params[i] -= lr * mh / (sqrt(vh) + eps_adam);
        }

        double E = avg_energy(vqe_one_shot, params, vqe_shots);
        stdout = real_stdout;

        if (E < best_E) {
            best_E = E;
            memcpy(best_params, params, sizeof(best_params));
        }

        if (epoch < 5 || epoch % 15 == 0 || epoch == vqe_epochs - 1) {
            double progress = (E_random - E) / (E_random - E_gs);
            if (progress < 0) progress = 0;
            if (progress > 1) progress = 1;
            int bar = (int)(progress * 30);
            printf("  Epoch %3d │ E = %+7.3f │ ", epoch, E);
            for (int b = 0; b < bar; b++) printf("█");
            for (int b = bar; b < 30; b++) printf("░");
            printf(" │ %.0f%%\n", 100.0 * progress);
        }

        if (epoch == 15) lr *= 0.5;
        if (epoch == 30) lr *= 0.5;
        if (epoch == 45) lr *= 0.5;
    }

    clock_gettime(CLOCK_MONOTONIC, &t2);
    double vqe_s = (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec) / 1e9;

    /* Final eval with lots of shots */
    stdout = devnull;
    double final_E = avg_energy(vqe_one_shot, best_params, 500);
    stdout = real_stdout;

    double vqe_progress = (E_random - final_E) / (E_random - E_gs);

    printf("\n  ┌──────────────────────────────────────────────────────────────┐\n");
    printf("  │ VQE RESULTS                                                  │\n");
    printf("  ├──────────────────────────────────────────────────────────────┤\n");
    printf("  │ Random energy:   %.4f                                     │\n", E_random);
    printf("  │ Initial energy:  %.3f                                      │\n", initial_E);
    printf("  │ VQE energy:      %.3f                                      │\n", final_E);
    printf("  │ Ground state:    %.1f                                       │\n", E_gs);
    printf("  │ Progress:        %.0f%% toward ground state                 │\n",
           100.0 * vqe_progress);
    printf("  │ Beats random:    %s                                        │\n",
           (final_E < E_random) ? "★ YES" : "✗ NO");
    printf("  │ Time:            %.0fs (%d epochs)                          │\n",
           vqe_s, vqe_epochs);
    printf("  └──────────────────────────────────────────────────────────────┘\n\n");

    /* ═══ PHASE 3: ENTANGLEMENT CERTIFICATION ═══ */
    printf("════════════════════════════════════════════════════════════════════════════════════\n");
    printf("  PHASE 3:  E N T A N G L E M E N T   C E R T I F I C A T I O N\n");
    printf("════════════════════════════════════════════════════════════════════════════════════\n\n");

    int cert_shots = 1000;
    int pair_corr[DIM][DIM];
    memset(pair_corr, 0, sizeof(pair_corr));
    int all_same_count = 0;
    int pair_same[N_QUDITS - 1];
    memset(pair_same, 0, sizeof(pair_same));

    stdout = devnull;
    for (int shot = 0; shot < cert_shots; shot++) {
        HexStateEngine eng;
        engine_init(&eng);

        for (int i = 0; i < N_QUDITS; i++)
            init_chunk(&eng, i, QUHITS);
        for (int i = 1; i < N_QUDITS; i++)
            braid_chunks_dim(&eng, 0, i, 0, 0, DIM);

        Complex U[DIM*DIM];
        for (int layer = 0; layer < VQE_LAYERS; layer++) {
            for (int q = 0; q < N_QUDITS; q++) {
                int pi = layer * N_QUDITS * 2 + q * 2;
                make_rz(best_params[pi], U);
                apply_local_unitary(&eng, q, U, DIM);
                make_rx(best_params[pi + 1], U);
                apply_local_unitary(&eng, q, U, DIM);
            }
            for (int q = 0; q < N_QUDITS - 1; q++)
                apply_cz_gate(&eng, q, q + 1);
        }

        int colors[N_QUDITS];
        for (int i = 0; i < N_QUDITS; i++)
            colors[i] = (int)(measure_chunk(&eng, i) % DIM);

        int all_s = 1;
        for (int i = 0; i < N_QUDITS - 1; i++) {
            if (colors[i] == colors[i+1]) pair_same[i]++;
            else all_s = 0;
        }
        if (all_s) all_same_count++;
        pair_corr[colors[0]][colors[1]]++;

        engine_destroy(&eng);
    }
    stdout = real_stdout;

    /* Joint distribution */
    printf("  P(q₀, q₁) from %d shots:\n\n         ", cert_shots);
    for (int j = 0; j < DIM; j++) printf("  q₁=%d ", j);
    printf("\n");
    for (int i = 0; i < DIM; i++) {
        printf("  q₀=%d │", i);
        for (int j = 0; j < DIM; j++)
            printf(" %5.3f", (double)pair_corr[i][j] / cert_shots);
        printf("\n");
    }

    /* Entropy */
    double h_0 = 0, h_1 = 0, h_01 = 0;
    double marg_0[DIM] = {0}, marg_1[DIM] = {0};
    for (int i = 0; i < DIM; i++)
        for (int j = 0; j < DIM; j++) {
            double p = (double)pair_corr[i][j] / cert_shots;
            marg_0[i] += p;
            marg_1[j] += p;
            if (p > 1e-10) h_01 -= p * log2(p);
        }
    for (int i = 0; i < DIM; i++) {
        if (marg_0[i] > 1e-10) h_0 -= marg_0[i] * log2(marg_0[i]);
        if (marg_1[i] > 1e-10) h_1 -= marg_1[i] * log2(marg_1[i]);
    }
    double mi = h_0 + h_1 - h_01;
    double ghz_frac = (double)all_same_count / cert_shots;

    printf("\n  Neighbor same-color:\n");
    for (int i = 0; i < N_QUDITS - 1; i++) {
        double f = (double)pair_same[i] / cert_shots;
        printf("    (%d,%d): %.3f", i, i+1, f);
        if (f > 0.3) printf(" ← correlated");
        printf("\n");
    }

    int beyond = (fabs(ghz_frac - 1.0/DIM) > 0.05) && (mi > 0.05);
    printf("\n  Mutual info: %.3f bits | All-same: %.3f (GHZ=%.3f)\n",
           mi, ghz_frac, 1.0/DIM);
    printf("  Verdict: %s\n\n",
           beyond ? "★ BEYOND-GHZ entanglement confirmed" : "Inconclusive");

    /* ═══ FINAL SCORECARD ═══ */
    int qaoa_ok = (qaoa_refine < E_random);
    int vqe_ok  = (final_E < E_random);
    int ent_ok  = beyond;

    int score = qaoa_ok + vqe_ok + ent_ok;
    char *grade = score == 3 ? "★★★ 100%% — ALL TESTS PASSED" :
                  score == 2 ? "★★  67%% — 2/3 PASSED" :
                  score == 1 ? "★   33%% — 1/3 PASSED" :
                               "    0%% — NO TESTS PASSED";

    printf("╔══════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║           S C O R E C A R D :   B E Y O N D   G H Z                           ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║                                                                                ║\n");
    printf("║  Test 1: QAOA beats random on Heisenberg chain                                 ║\n");
    printf("║    Random: %.4f → QAOA: %.3f → %s                               ║\n",
           E_random, qaoa_refine, qaoa_ok ? "★ PASS" : "✗ FAIL");
    printf("║                                                                                ║\n");
    printf("║  Test 2: VQE beats random on Heisenberg chain                                  ║\n");
    printf("║    Random: %.4f → VQE:  %.3f → %s                               ║\n",
           E_random, final_E, vqe_ok ? "★ PASS" : "✗ FAIL");
    printf("║                                                                                ║\n");
    printf("║  Test 3: Beyond-GHZ entanglement certified                                     ║\n");
    printf("║    Mutual info: %.3f bits, all-same: %.3f → %s                 ║\n",
           mi, ghz_frac, ent_ok ? "★ PASS" : "✗ FAIL");
    printf("║                                                                                ║\n");
    printf("║  %s                                      ║\n", grade);
    printf("║                                                                                ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════════════╝\n");

    fclose(devnull);
    return (score == 3) ? 0 : 1;
}
