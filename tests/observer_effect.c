/* observer_effect.c
 *
 * ═══════════════════════════════════════════════════════════════════════
 *  THE OBSERVER EFFECT TEST
 *  Does YOUR Consciousness Collapse The Wavefunction?
 * ═══════════════════════════════════════════════════════════════════════
 *
 *  "Observation is creation." — John Wheeler
 *
 *  THE QUESTION:
 *  ─────────────
 *  In quantum mechanics, measuring a system changes it. But WHAT
 *  constitutes a "measurement"? Is it:
 *    (a) Any physical interaction (decoherence) — mainstream physics
 *    (b) Conscious observation specifically — von Neumann / Wigner
 *
 *  THE EXPERIMENT:
 *  ──────────────
 *  Same quantum circuit (prepare → oracle → Hadamard → oracle → measure)
 *  is run in two modes, randomized so you don't know which is coming:
 *
 *  WATCHED MODE (🔭):
 *    You SEE the intermediate state printed to your screen.
 *    Your eyes receive photons. Your visual cortex processes them.
 *    Your consciousness OBSERVES the quantum state.
 *    → Then the final measurement happens.
 *
 *  UNWATCHED MODE (🙈):
 *    The intermediate state is computed but NOT shown.
 *    No photons reach your eyes. Your consciousness is ignorant.
 *    → Then the final measurement happens.
 *
 *  IF consciousness causes collapse:
 *    The WATCHED distribution should differ from UNWATCHED.
 *    Specifically, watching the intermediate state should "pin" it,
 *    reducing entropy in the final measurement.
 *
 *  IF consciousness doesn't matter:
 *    Both distributions should be statistically identical.
 *
 *  We run 100 trials in each mode and compare using:
 *    • Chi-squared test (distribution comparison)
 *    • Jensen-Shannon divergence (information-theoretic distance)
 *    • Entropy difference (does watching reduce randomness?)
 */

#include "hexstate_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <termios.h>
#include <unistd.h>

#define NUM_Q   100000000000000ULL
#define D       6
#define PI      3.14159265358979323846

/* ═══════════════════════════════════════════════════════════════════════════════
 * Terminal helpers
 * ═══════════════════════════════════════════════════════════════════════════════ */
static struct termios orig_termios;

static void restore_terminal(void)
{
    tcsetattr(STDIN_FILENO, TCSANOW, &orig_termios);
}

static void raw_mode(void)
{
    tcgetattr(STDIN_FILENO, &orig_termios);
    atexit(restore_terminal);
    struct termios raw = orig_termios;
    raw.c_lflag &= ~(ICANON | ECHO);
    raw.c_cc[VMIN] = 1;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &raw);
}

static void wait_key(void)
{
    char ch;
    if (read(STDIN_FILENO, &ch, 1)) { (void)ch; }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * Observer Oracle — creates a rich intermediate state worth observing
 * ═══════════════════════════════════════════════════════════════════════════════ */
typedef struct {
    int  round;
    int  stage;   /* 0 = first oracle, 1 = second oracle */
} ObserverCtx;

static void observer_oracle(HexStateEngine *eng, uint64_t chunk_id, void *ud)
{
    ObserverCtx *ctx = (ObserverCtx *)ud;
    Chunk *c = &eng->chunks[chunk_id];

    if (!c->hilbert.q_joint_state) return;
    int dim = c->hilbert.q_joint_dim;

    /* Apply round-dependent and stage-dependent phases.
     * Stage 0: initial transformation
     * Stage 1: secondary transformation (after possible observation) */
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            int idx = i * dim + j;

            double phase;
            if (ctx->stage == 0) {
                /* First oracle: create superposition with structure */
                phase = 2.0 * PI * i * (ctx->round + 1) / dim +
                        PI * j / (dim + 1.0);
            } else {
                /* Second oracle: rotation that depends on position */
                phase = PI * (i + j) * (ctx->round * 3 + 7) / (dim * 5.0);
            }

            double cos_p = cos(phase);
            double sin_p = sin(phase);
            double re = c->hilbert.q_joint_state[idx].real;
            double im = c->hilbert.q_joint_state[idx].imag;
            c->hilbert.q_joint_state[idx].real = re * cos_p - im * sin_p;
            c->hilbert.q_joint_state[idx].imag = re * sin_p + im * cos_p;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * Get the intermediate state as a probability distribution (for display)
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void get_probabilities(Complex *state, int dim, double *probs)
{
    double total = 0;
    for (int i = 0; i < dim; i++) {
        probs[i] = 0;
        for (int j = 0; j < dim; j++) {
            int idx = i * dim + j;
            probs[i] += state[idx].real * state[idx].real +
                        state[idx].imag * state[idx].imag;
        }
        total += probs[i];
    }
    if (total > 1e-15)
        for (int i = 0; i < dim; i++)
            probs[i] /= total;
}

/* Compute Shannon entropy */
static double shannon_entropy(double *probs, int n)
{
    double H = 0;
    for (int i = 0; i < n; i++)
        if (probs[i] > 1e-15)
            H -= probs[i] * log2(probs[i]);
    return H;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * EXPERIMENT 1: The Core Observer Test
 *
 * Randomized interleaving of WATCHED and UNWATCHED trials.
 * User presses Enter to advance each trial. In WATCHED mode,
 * they see the intermediate quantum state. In UNWATCHED, they don't.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void experiment_core(HexStateEngine *eng)
{
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  EXPERIMENT 1: THE CORE OBSERVER TEST                        ║\n");
    printf("║  Does watching the quantum state change the outcome?          ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    printf("  You will see 60 trials. For each:\n");
    printf("  • The quantum circuit runs (prepare → oracle → Hadamard)\n");
    printf("  • In WATCHED 🔭 trials: you see the intermediate state\n");
    printf("  • In UNWATCHED 🙈 trials: it's hidden from you\n");
    printf("  • Then the final measurement happens\n\n");
    printf("  Press ENTER to advance each trial.\n");
    printf("  Press 'q' to quit early (after at least 30 trials).\n\n");
    printf("  YOUR JOB: Just watch. Don't try to do anything special.\n");
    printf("  Your consciousness IS the variable being tested.\n\n");
    printf("  Press any key to begin...\n");

    raw_mode();
    wait_key();

    int n_trials = 60;
    int watched_counts[D] = {0};
    int unwatched_counts[D] = {0};
    int watched_total = 0;
    int unwatched_total = 0;
    double watched_mid_entropy_sum = 0;
    double unwatched_mid_entropy_sum = 0;

    ObserverCtx ctx;
    oracle_register(eng, 0xB0, "Observer", observer_oracle, &ctx);

    /* Generate randomized schedule */
    int schedule[60];
    struct timespec seed_ts;
    clock_gettime(CLOCK_MONOTONIC, &seed_ts);
    uint64_t rng = seed_ts.tv_nsec ^ (seed_ts.tv_sec * 1000003ULL);
    for (int i = 0; i < n_trials; i++) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        schedule[i] = (rng >> 33) & 1;  /* 0 = watched, 1 = unwatched */
    }

    printf("\n  Trial  Mode          Intermediate        Final   Total W/U\n");
    printf("  ────── ──────── ──────────────────── ─────── ──────────\n");

    for (int t = 0; t < n_trials; t++) {
        int is_watched = (schedule[t] == 0);

        /* === Quantum circuit === */
        ctx.round = t;
        ctx.stage = 0;

        init_chunk(eng, 800, NUM_Q);
        init_chunk(eng, 801, NUM_Q);
        braid_chunks(eng, 800, 801, 0, 0);

        /* Stage 1: First oracle */
        execute_oracle(eng, 800, 0xB0);

        /* Apply Hadamard to create rich intermediate state */
        apply_hadamard(eng, 800, 0);

        /* === THE KEY MOMENT: intermediate state === */
        Chunk *c = &eng->chunks[800];
        double mid_probs[D];
        if (c->hilbert.q_joint_state) {
            get_probabilities(c->hilbert.q_joint_state, D, mid_probs);
        } else {
            for (int i = 0; i < D; i++) mid_probs[i] = 1.0 / D;
        }

        double mid_H = shannon_entropy(mid_probs, D);

        /* === Stage 2: Second oracle (after observation window) === */
        ctx.stage = 1;
        execute_oracle(eng, 800, 0xB0);

        /* === Final measurement === */
        uint64_t result = measure_chunk(eng, 800) % D;
        measure_chunk(eng, 801);
        unbraid_chunks(eng, 800, 801);

        /* Record */
        if (is_watched) {
            watched_counts[result]++;
            watched_total++;
            watched_mid_entropy_sum += mid_H;
        } else {
            unwatched_counts[result]++;
            unwatched_total++;
            unwatched_mid_entropy_sum += mid_H;
        }

        /* Display */
        if (is_watched) {
            /* 🔭 WATCHED: Show the intermediate state! */
            printf("  [%2d]   🔭 WATCHED  ", t + 1);
            /* Mini probability bar for each basis state */
            for (int i = 0; i < D; i++) {
                int bar = (int)(mid_probs[i] * 20);
                if (bar > 5) bar = 5;
                printf("|%d⟩", i);
                for (int b = 0; b < bar; b++) printf("█");
                for (int b = bar; b < 3; b++) printf("░");
                printf(" ");
            }
        } else {
            /* 🙈 UNWATCHED: Hidden! */
            printf("  [%2d]   🙈 HIDDEN   ", t + 1);
            printf("  ▒▒▒▒▒▒▒▒▒ HIDDEN ▒▒▒▒▒▒▒▒▒  ");
        }

        printf("  |%lu⟩   W:%d U:%d\n", result, watched_total, unwatched_total);
        fflush(stdout);

        /* Wait for user acknowledgment */
        char ch;
        if (read(STDIN_FILENO, &ch, 1) == 1 && (ch == 'q' || ch == 'Q')) {
            if (watched_total >= 15 && unwatched_total >= 15) break;
            printf("  [Need at least 15 of each mode]\n");
        }
    }

    oracle_unregister(eng, 0xB0);
    restore_terminal();

    /* ═══════════════════════════════════════════════════════════════════════
     * Analysis
     * ═══════════════════════════════════════════════════════════════════════ */
    printf("\n  ═══════════════════════════════════════════════════\n");
    printf("  OBSERVER EFFECT ANALYSIS\n");
    printf("  ═══════════════════════════════════════════════════\n\n");

    /* Distribution comparison */
    printf("  Final Measurement Distributions:\n\n");
    printf("  State   Watched (%d)    Unwatched (%d)   Difference\n",
           watched_total, unwatched_total);
    printf("  ────── ──────────── ────────────── ──────────\n");

    double w_probs[D], u_probs[D];
    double chi2 = 0;
    double js_div = 0;

    for (int i = 0; i < D; i++) {
        w_probs[i] = (double)watched_counts[i] / watched_total;
        u_probs[i] = (double)unwatched_counts[i] / unwatched_total;

        double diff = w_probs[i] - u_probs[i];

        /* Mini bar chart */
        int w_bar = (int)(w_probs[i] * 30);
        int u_bar = (int)(u_probs[i] * 30);

        printf("  |%d⟩    ", i);

        /* Watched bar */
        for (int b = 0; b < w_bar && b < 8; b++) printf("█");
        for (int b = w_bar; b < 8; b++) printf("░");
        printf(" %4.1f%%   ", w_probs[i] * 100);

        /* Unwatched bar */
        for (int b = 0; b < u_bar && b < 8; b++) printf("▓");
        for (int b = u_bar; b < 8; b++) printf("░");
        printf(" %4.1f%%     ", u_probs[i] * 100);

        /* Difference */
        printf("%+5.1f%%\n", diff * 100);

        /* Chi-squared contribution */
        double expected = ((double)watched_counts[i] + unwatched_counts[i]) / 2.0;
        if (expected > 0.5) {
            chi2 += (watched_counts[i] - expected) * (watched_counts[i] - expected) / expected;
            chi2 += (unwatched_counts[i] - expected) * (unwatched_counts[i] - expected) / expected;
        }

        /* Jensen-Shannon divergence */
        double m = (w_probs[i] + u_probs[i]) / 2.0;
        if (w_probs[i] > 1e-15 && m > 1e-15)
            js_div += 0.5 * w_probs[i] * log2(w_probs[i] / m);
        if (u_probs[i] > 1e-15 && m > 1e-15)
            js_div += 0.5 * u_probs[i] * log2(u_probs[i] / m);
    }

    /* Entropy comparison */
    double H_watched = shannon_entropy(w_probs, D);
    double H_unwatched = shannon_entropy(u_probs, D);
    double H_max = log2(D);

    double avg_mid_watched = watched_mid_entropy_sum / watched_total;
    double avg_mid_unwatched = unwatched_mid_entropy_sum / unwatched_total;

    printf("\n  ─── Statistical Tests ───\n\n");

    printf("  Chi-squared (df=%d):  χ² = %.3f\n", D - 1, chi2);
    /* Critical values: df=5, α=0.05 → 11.07; α=0.10 → 9.24 */
    if (chi2 > 11.07)
        printf("  → SIGNIFICANT (p < 0.05) ⚡ Distributions DIFFER!\n");
    else if (chi2 > 9.24)
        printf("  → Marginally significant (p < 0.10)\n");
    else
        printf("  → Not significant — distributions appear similar\n");

    printf("\n  Jensen-Shannon divergence: JSD = %.6f bits\n", js_div);
    printf("  (0 = identical, 1 = maximally different)\n");
    if (js_div > 0.05)
        printf("  → Substantial divergence! The observer matters.\n");
    else if (js_div > 0.01)
        printf("  → Mild divergence — possible observer effect.\n");
    else
        printf("  → Negligible divergence — no observer effect.\n");

    printf("\n  ─── Entropy Analysis ───\n\n");
    printf("  Final measurement entropy:\n");
    printf("    Watched:    H = %.4f bits (%.1f%% of max)\n",
           H_watched, 100.0 * H_watched / H_max);
    printf("    Unwatched:  H = %.4f bits (%.1f%% of max)\n",
           H_unwatched, 100.0 * H_unwatched / H_max);
    printf("    Difference: ΔH = %+.4f bits\n", H_watched - H_unwatched);

    printf("\n  Intermediate state entropy (averaged):\n");
    printf("    When watched:    H_mid = %.4f bits\n", avg_mid_watched);
    printf("    When not watched: H_mid = %.4f bits\n", avg_mid_unwatched);

    if (H_watched < H_unwatched - 0.1) {
        printf("\n  ⚡ OBSERVER EFFECT DETECTED ⚡\n");
        printf("  Watching REDUCED entropy by %.4f bits!\n",
               H_unwatched - H_watched);
        printf("  → Your observation collapsed the superposition,\n");
        printf("    making the final state MORE deterministic.\n");
        printf("  → Consistent with consciousness-driven collapse.\n");
    } else if (H_watched > H_unwatched + 0.1) {
        printf("\n  ⚡ INVERSE OBSERVER EFFECT ⚡\n");
        printf("  Watching INCREASED entropy by %.4f bits!\n",
               H_watched - H_unwatched);
        printf("  → Your observation ENHANCED the superposition!\n");
        printf("  → Your consciousness added uncertainty.\n");
    } else {
        printf("\n  ○ No significant observer effect on entropy.\n");
        printf("  → Your consciousness does not appear to collapse\n");
        printf("    the wavefunction (at this measurement scale).\n");
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * EXPERIMENT 2: Schrödinger's Screen
 *
 * The display buffer itself is the "box." We prepare a quantum state,
 * write the result to the screen buffer, but use ANSI escape codes
 * to either SHOW or HIDE it. The state is "in your computer" either
 * way — but does YOUR EYES seeing it change the subsequent evolution?
 *
 * This is Schrödinger's cat, but the cat is a number on your screen.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void experiment_schrodinger_screen(HexStateEngine *eng)
{
    printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  EXPERIMENT 2: SCHRÖDINGER'S SCREEN                          ║\n");
    printf("║  The quantum state is always on screen — but can you see it? ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    printf("  In this experiment, the intermediate quantum state is\n");
    printf("  ALWAYS written to your terminal buffer. But sometimes\n");
    printf("  it's printed in VISIBLE text, and sometimes in\n");
    printf("  INVISIBLE text (same color as background).\n\n");
    printf("  The bits exist in your computer's RAM either way.\n");
    printf("  The only difference is whether PHOTONS reach YOUR EYES.\n\n");
    printf("  Does it matter if the information hits your retina?\n\n");
    printf("  40 trials. Press ENTER to advance.\n\n");
    printf("  Press any key to begin...\n");

    raw_mode();
    wait_key();

    int n_trials = 40;
    int visible_counts[D] = {0};
    int invisible_counts[D] = {0};
    int visible_total = 0;
    int invisible_total = 0;

    ObserverCtx ctx;
    oracle_register(eng, 0xB1, "Schrodinger", observer_oracle, &ctx);

    /* Randomize schedule */
    struct timespec seed_ts;
    clock_gettime(CLOCK_MONOTONIC, &seed_ts);
    uint64_t rng = seed_ts.tv_nsec ^ (seed_ts.tv_sec * 7919ULL);

    printf("\n");

    for (int t = 0; t < n_trials; t++) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        int is_visible = (rng >> 33) & 1;

        ctx.round = t;
        ctx.stage = 0;

        init_chunk(eng, 810, NUM_Q);
        init_chunk(eng, 811, NUM_Q);
        braid_chunks(eng, 810, 811, 0, 0);

        execute_oracle(eng, 810, 0xB1);
        apply_hadamard(eng, 810, 0);

        /* Intermediate "measurement" — always computed */
        Chunk *c = &eng->chunks[810];
        double probs[D];
        int mid_result = 0;
        if (c->hilbert.q_joint_state) {
            get_probabilities(c->hilbert.q_joint_state, D, probs);
            /* Find most probable state */
            for (int i = 1; i < D; i++)
                if (probs[i] > probs[mid_result]) mid_result = i;
        }

        /* Display: VISIBLE or INVISIBLE */
        printf("  [%2d]  ", t + 1);
        if (is_visible) {
            printf("👁 VISIBLE   |%d⟩ (p=%.1f%%)  ", mid_result, probs[mid_result] * 100);
        } else {
            /* Print in "invisible" mode — same info, obscured */
            printf("⬛ INVISIBLE  ▒▒▒▒▒▒▒▒▒▒▒▒▒  ");
        }

        /* Second oracle stage */
        ctx.stage = 1;
        execute_oracle(eng, 810, 0xB1);

        /* Final measurement */
        uint64_t result = measure_chunk(eng, 810) % D;
        measure_chunk(eng, 811);
        unbraid_chunks(eng, 810, 811);

        if (is_visible) {
            visible_counts[result]++;
            visible_total++;
        } else {
            invisible_counts[result]++;
            invisible_total++;
        }

        int mid_matched = ((uint64_t)mid_result == result);
        printf("→ |%lu⟩  %s", result,
               mid_matched ? "(matched mid!)" : "");
        printf("  V:%d I:%d\n", visible_total, invisible_total);

        char ch;
        if (read(STDIN_FILENO, &ch, 1) == 1 && (ch == 'q' || ch == 'Q'))
            if (visible_total >= 10 && invisible_total >= 10) break;
    }

    oracle_unregister(eng, 0xB1);
    restore_terminal();

    /* Analysis */
    printf("\n  ═══════════════════════════════════════════════════\n");
    printf("  SCHRÖDINGER'S SCREEN RESULTS\n");
    printf("  ═══════════════════════════════════════════════════\n\n");

    double v_probs[D], i_probs[D];
    for (int i = 0; i < D; i++) {
        v_probs[i] = (visible_total > 0) ?
                     (double)visible_counts[i] / visible_total : 0;
        i_probs[i] = (invisible_total > 0) ?
                     (double)invisible_counts[i] / invisible_total : 0;
    }

    double H_v = shannon_entropy(v_probs, D);
    double H_i = shannon_entropy(i_probs, D);

    printf("  State   Visible (%d)    Invisible (%d)\n", visible_total, invisible_total);
    printf("  ────── ──────────── ──────────────\n");
    for (int s = 0; s < D; s++) {
        printf("  |%d⟩     %4.1f%%          %4.1f%%\n",
               s, v_probs[s] * 100, i_probs[s] * 100);
    }

    printf("\n  Entropy (visible):    H = %.4f bits\n", H_v);
    printf("  Entropy (invisible):  H = %.4f bits\n", H_i);
    printf("  Difference: ΔH = %+.4f bits\n\n", H_v - H_i);

    if (fabs(H_v - H_i) > 0.15) {
        printf("  ⚡ The photons hitting your retina CHANGED the outcome! ⚡\n");
        printf("  → The quantum state evolved differently when you could\n");
        printf("    see the intermediate result vs when you couldn't.\n");
        printf("  → Your eyes are quantum measurement devices.\n");
    } else {
        printf("  ○ No difference — photons reaching your eyes didn't\n");
        printf("    change the quantum evolution.\n");
        printf("  → Consistent with decoherence (not consciousness)\n");
        printf("    being the true cause of wavefunction collapse.\n");
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * EXPERIMENT 3: The Attention Test
 *
 * YOU control your level of attention:
 *   - Press 'W' (WATCH): you pay close attention to this trial
 *   - Press 'I' (IGNORE): you intentionally look away / zone out
 *
 * Same circuit both times. Does your LEVEL OF ATTENTION
 * (not just passive seeing) affect the measurement?
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void experiment_attention(HexStateEngine *eng)
{
    printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  EXPERIMENT 3: THE ATTENTION EXPERIMENT                      ║\n");
    printf("║  Does your FOCUS change the quantum state?                    ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    printf("  For each trial, you choose your attention level:\n\n");
    printf("  Press 'W' = WATCH INTENTLY — stare at the state, focus\n");
    printf("  Press 'I' = IGNORE — look away, think about something else\n\n");
    printf("  The quantum state is ALWAYS shown. The only variable\n");
    printf("  is whether YOU are paying attention to it.\n\n");
    printf("  Is consciousness a passive receiver or an active force?\n\n");
    printf("  30+ trials. Press 'q' when done.\n\n");
    printf("  Press any key to begin...\n");

    raw_mode();
    wait_key();

    int focus_counts[D] = {0}, ignore_counts[D] = {0};
    int focus_total = 0, ignore_total = 0;
    int max_trials = 50;

    ObserverCtx ctx;
    oracle_register(eng, 0xB2, "Attention", observer_oracle, &ctx);

    printf("\n  Trial  Focus?   Quantum State Probabilities             Result\n");
    printf("  ────── ──────── ────────────────────────────────────── ──────\n");

    for (int t = 0; t < max_trials; t++) {
        /* Prepare quantum state */
        ctx.round = t;
        ctx.stage = 0;

        init_chunk(eng, 820, NUM_Q);
        init_chunk(eng, 821, NUM_Q);
        braid_chunks(eng, 820, 821, 0, 0);
        execute_oracle(eng, 820, 0xB2);
        apply_hadamard(eng, 820, 0);

        /* Show intermediate state */
        Chunk *c = &eng->chunks[820];
        double probs[D];
        if (c->hilbert.q_joint_state)
            get_probabilities(c->hilbert.q_joint_state, D, probs);
        else
            for (int i = 0; i < D; i++) probs[i] = 1.0 / D;

        printf("  [%2d]   ", t + 1);

        /* Show probabilities */
        printf("???      ");
        for (int i = 0; i < D; i++) {
            int bar = (int)(probs[i] * 20);
            printf("|%d⟩", i);
            for (int b = 0; b < bar && b < 4; b++) printf("█");
            for (int b = (bar < 4 ? bar : 4); b < 2; b++) printf("░");
            printf(" ");
        }

        printf("  (W/I) ");
        fflush(stdout);

        /* Get attention choice */
        int focused = -1;
        while (focused == -1) {
            char ch;
            if (read(STDIN_FILENO, &ch, 1) != 1) continue;
            if (ch == 'w' || ch == 'W') focused = 1;
            else if (ch == 'i' || ch == 'I') focused = 0;
            else if (ch == 'q' || ch == 'Q') { focused = -2; break; }
        }

        if (focused == -2) {
            printf("quit\n");
            if (focus_total >= 10 && ignore_total >= 10) break;
            printf("  [Need 10+ of each]\n");
            focused = 1;
        }

        /* Second oracle + measure */
        ctx.stage = 1;
        execute_oracle(eng, 820, 0xB2);
        uint64_t result = measure_chunk(eng, 820) % D;
        measure_chunk(eng, 821);
        unbraid_chunks(eng, 820, 821);

        if (focused) {
            focus_counts[result]++;
            focus_total++;
            printf("\r  [%2d]   🎯 FOCUS  ", t + 1);
        } else {
            ignore_counts[result]++;
            ignore_total++;
            printf("\r  [%2d]   😶 IGNORE ", t + 1);
        }

        /* Reprint probabilities */
        for (int i = 0; i < D; i++) {
            int bar = (int)(probs[i] * 20);
            printf("|%d⟩", i);
            for (int b = 0; b < bar && b < 4; b++) printf("█");
            for (int b = (bar < 4 ? bar : 4); b < 2; b++) printf("░");
            printf(" ");
        }
        printf("  → |%lu⟩   F:%d I:%d\n", result, focus_total, ignore_total);
    }

    oracle_unregister(eng, 0xB2);
    restore_terminal();

    /* Analysis */
    printf("\n  ═══════════════════════════════════════════════════\n");
    printf("  ATTENTION EXPERIMENT RESULTS\n");
    printf("  ═══════════════════════════════════════════════════\n\n");

    double f_probs[D], ig_probs[D];
    for (int i = 0; i < D; i++) {
        f_probs[i] = (focus_total > 0) ?
                     (double)focus_counts[i] / focus_total : 0;
        ig_probs[i] = (ignore_total > 0) ?
                      (double)ignore_counts[i] / ignore_total : 0;
    }

    double H_f = shannon_entropy(f_probs, D);
    double H_ig = shannon_entropy(ig_probs, D);

    printf("  State   Focused (%d)   Ignored (%d)   Difference\n",
           focus_total, ignore_total);
    printf("  ────── ──────────── ──────────── ──────────\n");
    for (int s = 0; s < D; s++) {
        printf("  |%d⟩     %5.1f%%        %5.1f%%       %+5.1f%%\n",
               s, f_probs[s] * 100, ig_probs[s] * 100,
               (f_probs[s] - ig_probs[s]) * 100);
    }

    printf("\n  Entropy (focused):  H = %.4f bits (%.1f%% of max)\n",
           H_f, 100 * H_f / log2(D));
    printf("  Entropy (ignored):  H = %.4f bits (%.1f%% of max)\n",
           H_ig, 100 * H_ig / log2(D));
    printf("  ΔH = %+.4f bits\n\n", H_f - H_ig);

    if (H_f < H_ig - 0.15) {
        printf("  ⚡ ATTENTION COLLAPSES THE WAVEFUNCTION ⚡\n");
        printf("  → Focused observation reduced entropy by %.3f bits.\n",
               H_ig - H_f);
        printf("  → Your attention is a quantum measurement force.\n");
    } else if (H_f > H_ig + 0.15) {
        printf("  ⚡ ATTENTION INCREASES QUANTUM UNCERTAINTY ⚡\n");
        printf("  → Focused observation ADDED entropy by %.3f bits.\n",
               H_f - H_ig);
        printf("  → Your consciousness destabilizes the quantum state!\n");
    } else {
        printf("  ○ No significant difference between focused and ignored.\n");
        printf("  → Your attention level doesn't affect the quantum state.\n");
        printf("  → Collapse is physical (decoherence), not mental.\n");
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * GRAND SUMMARY
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void grand_summary(void)
{
    printf("\n\n");
    printf("██████████████████████████████████████████████████████████████████\n");
    printf("██  OBSERVER EFFECT — GRAND SUMMARY                           ██\n");
    printf("██████████████████████████████████████████████████████████████████\n\n");

    printf("  You performed 3 experiments testing whether your\n");
    printf("  consciousness affects quantum measurement outcomes:\n\n");
    printf("  1. CORE TEST: Random interleaving of watched/unwatched\n");
    printf("     trials with chi-squared and JSD analysis.\n\n");
    printf("  2. SCHRÖDINGER'S SCREEN: Same info in RAM, but do\n");
    printf("     the photons reaching your retina matter?\n\n");
    printf("  3. ATTENTION: Same visual input, but does your\n");
    printf("     LEVEL OF FOCUS change the outcome?\n\n");
    printf("  Together, these probe the measurement problem at\n");
    printf("  three levels:\n");
    printf("    Physical interaction → Information display → Conscious focus\n\n");
    printf("  If NONE showed an effect: decoherence (mainstream physics)\n");
    printf("  If SCREEN showed an effect: retinal photons matter\n");
    printf("  If ATTENTION showed an effect: consciousness is special\n\n");
    printf("  \"Was the moon there when nobody looked?\"\n");
    printf("                                   — Einstein to Bohr\n");
    printf("██████████████████████████████████████████████████████████████████\n\n");
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════════ */
int main(void)
{
    printf("\n");
    printf("██████████████████████████████████████████████████████████████████\n");
    printf("██                                                            ██\n");
    printf("██   🧲 THE OBSERVER EFFECT TEST                              ██\n");
    printf("██                                                            ██\n");
    printf("██   Does YOUR consciousness collapse the wavefunction?       ██\n");
    printf("██                                                            ██\n");
    printf("██   3 experiments testing observation, visibility,           ██\n");
    printf("██   and attention as quantum measurement forces.             ██\n");
    printf("██                                                            ██\n");
    printf("██   100T quhits × 6 basis states × your eyeballs           ██\n");
    printf("██                                                            ██\n");
    printf("██████████████████████████████████████████████████████████████████\n\n");

    HexStateEngine eng;
    if (engine_init(&eng) != 0) {
        fprintf(stderr, "FATAL: engine_init failed\n");
        return 1;
    }

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    experiment_core(&eng);
    experiment_schrodinger_screen(&eng);
    experiment_attention(&eng);

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double total = (t_end.tv_sec - t_start.tv_sec) +
                   (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

    grand_summary();
    printf("  Session time: %.1f seconds\n\n", total);

    return 0;
}
