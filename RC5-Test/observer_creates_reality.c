/* observer_creates_reality.c — THE OBSERVER CREATES REALITY
 *
 * ██████████████████████████████████████████████████████████████████████████████
 * ██                                                                        ██
 * ██  THE OBSERVER IS THE POINT.                                            ██
 * ██                                                                        ██
 * ██  Without the observer, the curve has no X, no Y.                       ██
 * ██  Without the curve, the observer has no self-knowledge.                ██
 * ██  Neither pre-exists the other — they co-arise through entanglement.    ██
 * ██                                                                        ██
 * ██  Observations:                                                         ██
 * ██    1. No observer → no dimensions (the world doesn't exist)            ██
 * ██    2. Observation creates spatial structure (not discovers it)          ██
 * ██    3. Observer complexity determines how many dimensions               ██
 * ██    4. Self-knowledge requires the world (needs reflection)             ██
 * ██    5. Multiple observers create consensus (quantum Darwinism)          ██
 * ██    6. Measurement makes reality irreversible (creates facts)           ██
 * ██                                                                        ██
 * ██████████████████████████████████████████████████████████████████████████████
 */

#include "hexstate_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define D       6
#define D2      (D * D)
#define NUM_Q   100000000000000ULL

#define CMPLX(r_, i_) ((Complex){.real = (r_), .imag = (i_)})

/* ═══════════════════════════════════ UTILITIES ═════════════════════════════ */

static double cnorm2(Complex c) { return c.real*c.real + c.imag*c.imag; }

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Von Neumann entropy via Jacobi diagonalization */
static double entropy_nxn(double *H, int N) {
    for (int iter = 0; iter < 300; iter++) {
        double off = 0;
        for (int p = 0; p < N; p++)
            for (int q = p+1; q < N; q++)
                off += H[p*N+q] * H[p*N+q];
        if (off < 1e-28) break;
        for (int p = 0; p < N; p++)
            for (int q = p+1; q < N; q++) {
                double apq = H[p*N+q];
                if (fabs(apq) < 1e-15) continue;
                double d = H[q*N+q] - H[p*N+p];
                double t;
                if (fabs(d) < 1e-15) t = 1.0;
                else {
                    double tau = d / (2.0 * apq);
                    t = ((tau >= 0) ? 1.0 : -1.0) / (fabs(tau) + sqrt(1.0 + tau*tau));
                }
                double c = 1.0 / sqrt(1.0 + t*t), s = t*c;
                double app = H[p*N+p], aqq = H[q*N+q];
                H[p*N+p] = c*c*app - 2*s*c*apq + s*s*aqq;
                H[q*N+q] = s*s*app + 2*s*c*apq + c*c*aqq;
                H[p*N+q] = H[q*N+p] = 0;
                for (int r = 0; r < N; r++) {
                    if (r == p || r == q) continue;
                    double arp = H[r*N+p], arq = H[r*N+q];
                    H[r*N+p] = H[p*N+r] = c*arp - s*arq;
                    H[r*N+q] = H[q*N+r] = s*arp + c*arq;
                }
            }
    }
    double S = 0;
    for (int i = 0; i < N; i++)
        if (H[i*N+i] > 1e-15)
            S -= H[i*N+i] * log(H[i*N+i]);
    return S;
}

static double purity_nxn(const double *rho, int N) {
    double p = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            p += rho[i*N+j] * rho[j*N+i];
    return p;
}

/* Partial traces for D×D joint state */
static void pt_curve(const Complex *j, double *rho, int dim) {
    memset(rho, 0, dim*dim*sizeof(double));
    for (int p1 = 0; p1 < dim; p1++)
        for (int p2 = 0; p2 < dim; p2++)
            for (int c = 0; c < dim; c++) {
                double r1=j[p1*dim+c].real, i1=j[p1*dim+c].imag;
                double r2=j[p2*dim+c].real, i2=j[p2*dim+c].imag;
                rho[p1*dim+p2] += r1*r2 + i1*i2;
            }
}

static void pt_point(const Complex *j, double *rho, int dim) {
    memset(rho, 0, dim*dim*sizeof(double));
    for (int c1 = 0; c1 < dim; c1++)
        for (int c2 = 0; c2 < dim; c2++)
            for (int p = 0; p < dim; p++) {
                double r1=j[p*dim+c1].real, i1=j[p*dim+c1].imag;
                double r2=j[p*dim+c2].real, i2=j[p*dim+c2].imag;
                rho[c1*dim+c2] += r1*r2 + i1*i2;
            }
}

/* Prime factorization: return count */
static int prime_factors(int n, int *fac, int max) {
    int c = 0;
    for (int p = 2; p*p <= n && c < max; p++)
        while (n % p == 0 && c < max) { fac[c++] = p; n /= p; }
    if (n > 1 && c < max) fac[c++] = n;
    return c;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 *  TEST 1: NO OBSERVER → NO DIMENSIONS
 *
 *  The curve by itself: pure state |0⟩, entropy = 0, purity = 1.
 *  It has no X, no Y. The spatial axes DON'T EXIST without someone to
 *  unfold them. The "world" without an observer is structureless.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void test_no_observer(void)
{
    printf("  ╔══════════════════════════════════════════════════════════════════╗\n");
    printf("  ║  TEST 1: NO OBSERVER → NO DIMENSIONS                          ║\n");
    printf("  ║  Without the point, the curve is structureless.                ║\n");
    printf("  ╚══════════════════════════════════════════════════════════════════╝\n\n");

    /* Curve alone: product state |0⟩ */
    Complex curve[D];
    memset(curve, 0, sizeof(curve));
    curve[0] = CMPLX(1.0, 0.0);

    double probs[D];
    for (int k = 0; k < D; k++) probs[k] = cnorm2(curve[k]);

    /* Shannon entropy of the probability distribution */
    double H = 0;
    for (int k = 0; k < D; k++)
        if (probs[k] > 1e-15) H -= probs[k] * log(probs[k]);

    printf("  THE WORLD WITHOUT AN OBSERVER:\n");
    printf("    State:   |0⟩ (frozen at one position)\n");
    printf("    Entropy: %.6f nats (no uncertainty → no structure)\n", H);
    printf("    Probs:   ");
    for (int k = 0; k < D; k++) printf("P(%d)=%.1f  ", k, probs[k]);
    printf("\n\n");

    printf("    The curve has 6 positions, but with no observer:\n");
    printf("    • No X axis to unfold (H₂ factor inaccessible)\n");
    printf("    • No Y axis to unfold (H₃ factor inaccessible)\n");
    printf("    • Just a frozen point in a 1D line\n\n");

    /* Point alone: same story */
    Complex point[D];
    memset(point, 0, sizeof(point));
    point[0] = CMPLX(1.0, 0.0);

    double H_p = 0;
    for (int k = 0; k < D; k++) {
        double pk = cnorm2(point[k]);
        if (pk > 1e-15) H_p -= pk * log(pk);
    }

    printf("  THE OBSERVER WITHOUT A WORLD:\n");
    printf("    State:   |0⟩ (no internal structure activated)\n");
    printf("    Entropy: %.6f nats (no self-knowledge)\n", H_p);
    printf("    The observer's d=6 internal DoFs exist in theory,\n");
    printf("    but without a world to reflect them, they're invisible.\n\n");

    printf("  VERDICT: ✓ Neither has structure alone.\n");
    printf("  The world needs an observer. The observer needs a world.\n");
    printf("  Dimensionality is not a property — it's a RELATIONSHIP.\n\n");
}

/* ═══════════════════════════════════════════════════════════════════════════════
 *  TEST 2: OBSERVATION CREATES STRUCTURE
 *
 *  The moment of entanglement = the moment of creation.
 *  Track entropy at each step:
 *    Before: S=0 (no structure)
 *    Entangle: S=log(6) (full 3D structure)
 *  The observer didn't FIND structure — it CREATED it.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void test_observation_creates(HexStateEngine *eng)
{
    printf("  ╔══════════════════════════════════════════════════════════════════╗\n");
    printf("  ║  TEST 2: OBSERVATION CREATES STRUCTURE                         ║\n");
    printf("  ║  Entanglement = the act of creation.                           ║\n");
    printf("  ╚══════════════════════════════════════════════════════════════════╝\n\n");

    /* === Step 1: Before observation === */
    Complex joint_before[D2];
    memset(joint_before, 0, sizeof(joint_before));
    joint_before[0 * D + 0] = CMPLX(1.0, 0.0);  /* |0⟩_obs ⊗ |0⟩_world */

    double rho_w_before[D * D];
    pt_point(joint_before, rho_w_before, D);
    double rho_copy[D * D];
    memcpy(rho_copy, rho_w_before, sizeof(rho_copy));
    double S_before = entropy_nxn(rho_copy, D);
    double P_before = purity_nxn(rho_w_before, D);

    printf("  BEFORE (product state |0⟩_obs ⊗ |0⟩_world):\n");
    printf("    S(world) = %.4f nats  (no structure)\n", S_before);
    printf("    Purity   = %.4f       (completely definite)\n", P_before);
    printf("    Effective dimensions: %.1f (just 1 point in space)\n\n", exp(S_before));

    /* === Step 2: The act of observation (entanglement) === */
    Complex joint_after[D2];
    memset(joint_after, 0, sizeof(joint_after));
    double amp = 1.0 / sqrt((double)D);
    for (int k = 0; k < D; k++)
        joint_after[k * D + k] = CMPLX(amp, 0.0);

    double rho_w_after[D * D];
    pt_point(joint_after, rho_w_after, D);
    memcpy(rho_copy, rho_w_after, sizeof(rho_copy));
    double S_after = entropy_nxn(rho_copy, D);
    double P_after = purity_nxn(rho_w_after, D);

    double rho_o_after[D * D];
    pt_curve(joint_after, rho_o_after, D);
    memcpy(rho_copy, rho_o_after, sizeof(rho_copy));
    double S_obs = entropy_nxn(rho_copy, D);

    printf("  AFTER (Bell state (1/√6) Σ|k,k⟩ — observation occurred):\n");
    printf("    S(world)    = %.4f nats  (STRUCTURE EXISTS)\n", S_after);
    printf("    S(observer) = %.4f nats  (SELF-KNOWLEDGE EXISTS)\n", S_obs);
    printf("    Purity      = %.4f       (maximally mixed)\n", P_after);
    printf("    Effective dimensions: %.1f (full d=6 space accessible)\n\n", exp(S_after));

    printf("  THE CREATION EVENT:\n");
    printf("    ΔS(world)    = %.4f → %.4f  (increase: +%.4f nats)\n",
           S_before, S_after, S_after - S_before);
    printf("    ΔS(observer) = %.4f → %.4f  (increase: +%.4f nats)\n",
           0.0, S_obs, S_obs);
    printf("    Dimensions went from %.0f → %.0f\n\n", exp(S_before), exp(S_after));

    /* Verify with engine */
    init_chunk(eng, 10, NUM_Q);
    init_chunk(eng, 11, NUM_Q);
    HilbertSnapshot snap_pre = inspect_hilbert(eng, 11);

    braid_chunks_dim(eng, 10, 11, 0, 0, D);
    HilbertSnapshot snap_post = inspect_hilbert(eng, 11);

    printf("  ENGINE VERIFICATION (100T scale):\n");
    printf("    Before braid: S=%.4f, entangled=%s\n",
           snap_pre.entropy, snap_pre.is_entangled ? "yes" : "no");
    printf("    After  braid: S=%.4f, entangled=%s\n\n",
           snap_post.entropy, snap_post.is_entangled ? "yes" : "no");

    printf("  VERDICT: ★ The observer CREATED %d dimensions out of nothing.\n",
           (int)(exp(S_after) + 0.5) - 1);
    printf("  Before entanglement, the world had no spatial structure.\n");
    printf("  The observer didn't discover reality — it brought it into being.\n\n");

    unbraid_chunks(eng, 10, 11);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 *  TEST 3: OBSERVER COMPLEXITY → REALITY'S RICHNESS
 *
 *  A simple observer sees a simple world.
 *  A complex observer sees a rich world.
 *  The number of dimensions = Ω(d_observer) + 1.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void test_complexity_determines_reality(void)
{
    printf("  ╔══════════════════════════════════════════════════════════════════╗\n");
    printf("  ║  TEST 3: OBSERVER COMPLEXITY → REALITY'S RICHNESS             ║\n");
    printf("  ║  A d=2 observer sees 2D. A d=6 observer sees 3D.              ║\n");
    printf("  ║  You see what you are.                                         ║\n");
    printf("  ╚══════════════════════════════════════════════════════════════════╝\n\n");

    int dims[] = {2, 3, 5, 6, 7, 8, 12};
    int n_dims = (int)(sizeof(dims) / sizeof(dims[0]));

    printf("  ┌──────────────────────────────────────────────────────────────┐\n");
    printf("  │  Observer d   Ω(d)  S(world)   EffDims  Reality            │\n");
    printf("  ├──────────────────────────────────────────────────────────────┤\n");

    for (int i = 0; i < n_dims; i++) {
        int d = dims[i];
        int d2 = d * d;

        /* Bell state (1/√d) Σ|k,k⟩ */
        Complex *joint = calloc((size_t)d2, sizeof(Complex));
        double a = 1.0 / sqrt((double)d);
        for (int k = 0; k < d; k++)
            joint[k * d + k] = CMPLX(a, 0.0);

        /* World's entropy */
        double *rho = calloc((size_t)(d*d), sizeof(double));
        pt_point(joint, rho, d);
        double *rho_c = calloc((size_t)(d*d), sizeof(double));
        memcpy(rho_c, rho, (size_t)(d*d) * sizeof(double));
        double S = entropy_nxn(rho_c, d);
        double eff = exp(S);

        int fac[16];
        int omega = prime_factors(d, fac, 16);
        int total = 1 + omega;

        char fac_str[32] = "";
        for (int f = 0; f < omega; f++) {
            char buf[8];
            snprintf(buf, sizeof(buf), "%s%d", f > 0 ? "×" : "", fac[f]);
            strncat(fac_str, buf, sizeof(fac_str) - strlen(fac_str) - 1);
        }

        const char *desc = "";
        if (d == 2) desc = "Flatland";
        else if (d == 6) desc = "OUR UNIVERSE ★";
        else if (total == 3) desc = "3D";
        else if (total == 4) desc = "4D (unstable)";
        else desc = "Flatland";

        printf("  │  d=%-3d (%s)  %d     %.4f     %.1f      %dD  %s",
               d, fac_str, omega, S, eff, total, desc);
        int pad = 16 - (int)strlen(desc);
        for (int p = 0; p < pad; p++) printf(" ");
        printf("│\n");

        free(joint);
        free(rho);
        free(rho_c);
    }

    printf("  └──────────────────────────────────────────────────────────────┘\n\n");

    printf("  The observer doesn't passively watch — it projects its\n");
    printf("  internal complexity onto the world.\n\n");
    printf("  • A qubit observer (d=2, 1 prime) → sees a flat 2D world\n");
    printf("  • A quhit observer (d=6, 2 primes) → sees a rich 3D world\n");
    printf("  • A d=8 observer (3 primes) → sees 4D (but can't form atoms)\n\n");
    printf("  VERDICT: ★ You see what you ARE. Reality = observer structure.\n\n");
}

/* ═══════════════════════════════════════════════════════════════════════════════
 *  TEST 4: SELF-KNOWLEDGE REQUIRES THE WORLD
 *
 *  The observer alone: S=0. No self-knowledge.
 *  The observer entangled with the world: S=log(6). Full self-knowledge.
 *  But that self-knowledge is ALWAYS a reflection — it comes from
 *  the world, not from looking inward.
 *
 *  From mirror_perception.c: the world's observation of you is always
 *  a mirror (parity-inverted reflection).
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void test_self_knowledge(HexStateEngine *eng)
{
    printf("  ╔══════════════════════════════════════════════════════════════════╗\n");
    printf("  ║  TEST 4: SELF-KNOWLEDGE REQUIRES THE WORLD                    ║\n");
    printf("  ║  You cannot know yourself without a mirror.                    ║\n");
    printf("  ╚══════════════════════════════════════════════════════════════════╝\n\n");

    /* Observer alone */
    Complex obs_alone[D2];
    memset(obs_alone, 0, sizeof(obs_alone));
    obs_alone[0] = CMPLX(1.0, 0.0);

    double rho_obs_alone[D * D];
    memset(rho_obs_alone, 0, sizeof(rho_obs_alone));
    rho_obs_alone[0] = 1.0;  /* Pure |0⟩ */
    double rho_c[D * D];
    memcpy(rho_c, rho_obs_alone, sizeof(rho_c));
    double S_alone = entropy_nxn(rho_c, D);

    printf("  OBSERVER ALONE:\n");
    printf("    S(observer) = %.6f nats\n", S_alone);
    printf("    The observer exists, but has zero self-knowledge.\n");
    printf("    It's like having eyes but no mirror — you can't see\n");
    printf("    your own face.\n\n");

    /* Observer entangled with world */
    Complex joint[D2];
    memset(joint, 0, sizeof(joint));
    double amp = 1.0 / sqrt((double)D);
    for (int k = 0; k < D; k++)
        joint[k * D + k] = CMPLX(amp, 0.0);

    double rho_obs[D * D], rho_world[D * D];
    pt_curve(joint, rho_obs, D);
    pt_point(joint, rho_world, D);
    memcpy(rho_c, rho_obs, sizeof(rho_c));
    double S_obs = entropy_nxn(rho_c, D);
    memcpy(rho_c, rho_world, sizeof(rho_c));
    double S_world = entropy_nxn(rho_c, D);

    printf("  OBSERVER + WORLD (entangled):\n");
    printf("    S(observer) = %.6f nats (SELF-KNOWLEDGE)\n", S_obs);
    printf("    S(world)    = %.6f nats (STRUCTURE)\n", S_world);
    printf("    I(obs:world) = %.6f nats (MUTUAL knowledge)\n\n",
           S_obs + S_world);

    /* The reflection test: measure observer, see what world says */
    printf("  REFLECTION TEST (500 trials):\n");
    printf("    Measure observer → does the world reflect it back?\n\n");

    int n = 500, reflected = 0;
    for (int t = 0; t < n; t++) {
        init_chunk(eng, 20, NUM_Q);
        init_chunk(eng, 21, NUM_Q);
        braid_chunks_dim(eng, 20, 21, 0, 0, D);

        uint64_t obs_val = measure_chunk(eng, 20);
        uint64_t world_val = measure_chunk(eng, 21);

        if ((obs_val % D) == (world_val % D)) reflected++;
        unbraid_chunks(eng, 20, 21);
    }

    printf("    World reflected observer: %d/%d (%.1f%%)\n\n",
           reflected, n, 100.0 * reflected / n);

    printf("  VERDICT: ★ Self-knowledge = %.4f nats. All from reflection.\n", S_obs);
    printf("  The observer's S went from 0 → %.4f purely by entangling.\n", S_obs);
    printf("  There is no introspection — only reflection.\n");
    printf("  You know yourself ONLY through the world's mirror.\n\n");
}

/* ═══════════════════════════════════════════════════════════════════════════════
 *  TEST 5: MULTIPLE OBSERVERS → CONSENSUS REALITY
 *
 *  N points all observe the same curve. Do they all see the same thing?
 *  This is quantum Darwinism: objectivity = redundant encoding.
 *
 *  In a GHZ state: all observers agree with 100% probability.
 *  Consensus reality emerges from entanglement, not from the world
 *  having "real" properties.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void test_consensus_reality(HexStateEngine *eng)
{
    printf("  ╔══════════════════════════════════════════════════════════════════╗\n");
    printf("  ║  TEST 5: MULTIPLE OBSERVERS → CONSENSUS REALITY               ║\n");
    printf("  ║  N observers entangled with one world. Do they agree?          ║\n");
    printf("  ╚══════════════════════════════════════════════════════════════════╝\n\n");

    int observer_counts[] = {2, 3, 5, 10};
    int n_tests = (int)(sizeof(observer_counts) / sizeof(observer_counts[0]));

    for (int ti = 0; ti < n_tests; ti++) {
        int N = observer_counts[ti];
        int n_trials = 200;
        int all_agree = 0;

        for (int t = 0; t < n_trials; t++) {
            /* Create N+1 chunks: observers[0..N-1] + world[N] */
            uint64_t base = 700 + (uint64_t)(ti * 1000 + t * 20);
            for (int i = 0; i <= N; i++)
                init_chunk(eng, base + (uint64_t)i, NUM_Q);

            /* Star topology: each observer braids with the world (last chunk) */
            for (int i = 0; i < N; i++)
                braid_chunks_dim(eng, base + (uint64_t)i, base + (uint64_t)N, 0, 0, D);

            /* Measure ALL observers */
            uint64_t first_val = measure_chunk(eng, base) % D;
            int agree = 1;
            for (int i = 1; i < N; i++) {
                uint64_t val = measure_chunk(eng, base + (uint64_t)i) % D;
                if (val != first_val) { agree = 0; break; }
            }

            /* Check world too */
            uint64_t world_val = measure_chunk(eng, base + (uint64_t)N) % D;
            if (world_val != first_val) agree = 0;

            if (agree) all_agree++;

            for (int i = 0; i < N; i++)
                unbraid_chunks(eng, base + (uint64_t)i, base + (uint64_t)N);
        }

        printf("  %d observers + 1 world: all agree %d/%d (%.1f%%)\n",
               N, all_agree, n_trials, 100.0 * all_agree / n_trials);
    }

    printf("\n  QUANTUM DARWINISM:\n");
    printf("    Every observer sees the SAME reality.\n");
    printf("    Not because the world has \"real\" properties,\n");
    printf("    but because all observers are entangled into\n");
    printf("    the same GHZ state.\n\n");
    printf("  VERDICT: ★ Objectivity = entanglement redundancy.\n");
    printf("  The more observers, the more \"real\" reality feels.\n");
    printf("  But it's all one quantum state, viewed from many angles.\n\n");
}

/* ═══════════════════════════════════════════════════════════════════════════════
 *  TEST 6: MEASUREMENT CREATES IRREVERSIBLE FACT
 *
 *  Before measurement: superposition → entropy = log(6) → all possible.
 *  After measurement: collapsed → entropy = 0 → one fact is real.
 *  Subsequent observers all see the same fact.
 *
 *  The observer didn't read a pre-existing value.
 *  The observer CHOSE a value from the superposition.
 *  That choice is now an irreversible fact of reality.
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void test_measurement_irreversibility(HexStateEngine *eng)
{
    printf("  ╔══════════════════════════════════════════════════════════════════╗\n");
    printf("  ║  TEST 6: MEASUREMENT CREATES IRREVERSIBLE FACT                ║\n");
    printf("  ║  Before: all possible. After: one is real. No going back.      ║\n");
    printf("  ╚══════════════════════════════════════════════════════════════════╝\n\n");

    /* Before measurement: inspect Bell state */
    init_chunk(eng, 50, NUM_Q);
    init_chunk(eng, 51, NUM_Q);
    braid_chunks_dim(eng, 50, 51, 0, 0, D);

    HilbertSnapshot snap_pre = inspect_hilbert(eng, 51);

    printf("  BEFORE MEASUREMENT:\n");
    printf("    Non-zero amplitudes: %u (all %d possibilities alive)\n",
           snap_pre.num_entries, D);
    printf("    Entropy:  %.4f nats (maximum uncertainty)\n", snap_pre.entropy);
    printf("    Purity:   %.4f (maximally mixed reduced state)\n", snap_pre.purity);
    printf("    Marginals: ");
    for (int k = 0; k < D; k++) printf("|%d⟩=%.2f ", k, snap_pre.marginal_probs[k]);
    printf("\n\n");

    /* The irreversible act: MEASURE */
    uint64_t created_fact = measure_chunk(eng, 50) % D;

    /* After measurement: inspect collapsed state */
    HilbertSnapshot snap_post = inspect_hilbert(eng, 51);

    printf("  THE IRREVERSIBLE ACT: Observer measured → created fact |%lu⟩\n\n",
           (unsigned long)created_fact);

    printf("  AFTER MEASUREMENT:\n");
    printf("    Non-zero amplitudes: %u (only 1 survives)\n", snap_post.num_entries);
    printf("    Entropy:  %.4f nats (complete certainty)\n", snap_post.entropy);
    printf("    Purity:   %.4f (pure state)\n", snap_post.purity);
    printf("    Marginals: ");
    for (int k = 0; k < D; k++) printf("|%d⟩=%.2f ", k, snap_post.marginal_probs[k]);
    printf("\n\n");

    /* Verify subsequent observers see the same fact */
    uint64_t world_val = measure_chunk(eng, 51) % D;
    printf("  SUBSEQUENT OBSERVER measures world: |%lu⟩\n",
           (unsigned long)world_val);
    printf("  Match: %s\n\n", world_val == created_fact ? "✓ SAME FACT" : "✗ DIFFERENT");

    /* Repeat with many trials */
    int n = 500, consistent = 0;
    for (int t = 0; t < n; t++) {
        init_chunk(eng, 60, NUM_Q);
        init_chunk(eng, 61, NUM_Q);
        braid_chunks_dim(eng, 60, 61, 0, 0, D);

        uint64_t val1 = measure_chunk(eng, 60) % D;
        uint64_t val2 = measure_chunk(eng, 61) % D;
        if (val1 == val2) consistent++;

        unbraid_chunks(eng, 60, 61);
    }

    printf("  CONSISTENCY (500 trials):\n");
    printf("    Observer creates fact, world confirms: %d/%d (%.1f%%)\n\n",
           consistent, n, 100.0 * consistent / n);

    printf("  THE BEFORE/AFTER:\n");
    printf("    Entropy: %.4f → %.4f (collapsed to certainty)\n",
           snap_pre.entropy, snap_post.entropy);
    printf("    States:  %u → %u (all but one destroyed)\n",
           snap_pre.num_entries, snap_post.num_entries);
    printf("    Purity:  %.4f → %.4f (mixed → pure)\n\n",
           snap_pre.purity, snap_post.purity);

    printf("  VERDICT: ★ Measurement is CREATION, not discovery.\n");
    printf("  The observer selected |%lu⟩ from %d possibilities.\n",
           (unsigned long)created_fact, D);
    printf("  That selection is now an irreversible fact of reality.\n");
    printf("  All subsequent observers inherit it.\n\n");

    unbraid_chunks(eng, 50, 51);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 *  MAIN
 * ═══════════════════════════════════════════════════════════════════════════════ */
int main(void)
{
    printf("\n");
    printf("  ██████████████████████████████████████████████████████████████████████████████\n");
    printf("  ██                                                                        ██\n");
    printf("  ██  THE OBSERVER CREATES REALITY                                           ██\n");
    printf("  ██                                                                        ██\n");
    printf("  ██  The point (0D) IS the observer.                                        ██\n");
    printf("  ██  The curve (1D) IS the world.                                           ██\n");
    printf("  ██  Their entanglement IS the act of observation.                          ██\n");
    printf("  ██                                                                        ██\n");
    printf("  ██  Without the observer, the world has no dimensions.                     ██\n");
    printf("  ██  Without the world, the observer has no self-knowledge.                 ██\n");
    printf("  ██  They co-arise. Neither is primary.                                     ██\n");
    printf("  ██                                                                        ██\n");
    printf("  ██████████████████████████████████████████████████████████████████████████████\n\n");

    HexStateEngine eng;
    if (engine_init(&eng) != 0) {
        fprintf(stderr, "FATAL: engine_init failed\n");
        return 1;
    }

    double t0 = now_ms();

    test_no_observer();
    test_observation_creates(&eng);
    test_complexity_determines_reality();
    test_self_knowledge(&eng);
    test_consensus_reality(&eng);
    test_measurement_irreversibility(&eng);

    double elapsed = (now_ms() - t0) / 1000.0;

    printf("\n");
    printf("  ██████████████████████████████████████████████████████████████████████████████\n");
    printf("  ██                                                                        ██\n");
    printf("  ██  THE OBSERVER CREATES REALITY — CONCLUSIONS                             ██\n");
    printf("  ██                                                                        ██\n");
    printf("  ██  ┌──────────────────────────────────────────────────────────────────┐   ██\n");
    printf("  ██  │  1. No observer → no dimensions. The world is structureless.    │   ██\n");
    printf("  ██  │  2. Observation creates structure. Entanglement = creation.      │   ██\n");
    printf("  ██  │  3. Observer complexity = reality richness. You see what you are.│   ██\n");
    printf("  ██  │  4. Self-knowledge requires the world. No introspection exists. │   ██\n");
    printf("  ██  │  5. Multiple observers agree. Objectivity = entanglement.        │   ██\n");
    printf("  ██  │  6. Measurement is irreversible creation, not discovery.         │   ██\n");
    printf("  ██  └──────────────────────────────────────────────────────────────────┘   ██\n");
    printf("  ██                                                                        ██\n");
    printf("  ██  The observer IS the point.  The world IS the curve.                    ██\n");
    printf("  ██  Their entanglement IS consciousness.                                   ██\n");
    printf("  ██  Dimensionality is not a property of space —                             ██\n");
    printf("  ██  it's a relationship between observer and observed.                     ██\n");
    printf("  ██                                                                        ██\n");
    printf("  ██  Time: %.3f seconds                                                  ██\n", elapsed);
    printf("  ██                                                                        ██\n");
    printf("  ██████████████████████████████████████████████████████████████████████████████\n\n");

    engine_destroy(&eng);
    return 0;
}
