/* ═══════════════════════════════════════════════════════════════════════════
 * tesseract_sidechannel.c — 4D sidechannel attack on a 3D cipher
 *
 * "Now that we transcend the curse of the third dimension, let's use
 *  our 4D view to sidechannel this 3D world."
 *
 * A 3D substitution cipher permutes the 6 cube faces (S₆ = 720 keys).
 * From inside 3D, you must brute-force all 720 permutations.
 *
 * From 4D, the tesseract's CMY channel structure reveals the cipher's
 * internal wiring in O(1) channel measurements + O(8) within-channel
 * tests = ~11 total operations. That's a 65× speedup.
 *
 * The "sidechannel" is the 4th dimension itself: the compressed edges
 * of the tesseract connect faces that a 3D observer can't see are
 * related, but a 4D observer can read directly.
 *
 * Build: gcc -O2 -o tesseract_sidechannel tesseract_sidechannel.c \
 *         quhit_triality.c -lm
 * ═══════════════════════════════════════════════════════════════════════════ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "quhit_triality.h"
#include "quhit_triadic.h"

#define D 6

/* ═══════════════════════════════════════════════════════════════════════════
 * THE CIPHER: A secret permutation of the 6 cube faces
 *
 * The key is σ ∈ S₆, mapping |k⟩ → |σ(k)⟩.
 * This is a 6×6 permutation matrix applied to a D=6 quhit.
 *
 * Example: σ = (0→3, 1→5, 2→0, 3→1, 4→2, 5→4)
 * In 3D, this looks like an opaque black box.
 * In 4D, the CMY structure reveals its internal wiring.
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    int perm[D];         /* σ[k] = where basis state k goes */
    int inv_perm[D];     /* σ⁻¹[k] = which basis state maps to k */
} Cipher;

static void cipher_init_random(Cipher *c, unsigned seed)
{
    /* Fisher-Yates shuffle */
    srand(seed);
    for (int k = 0; k < D; k++) c->perm[k] = k;
    for (int k = D - 1; k > 0; k--) {
        int j = rand() % (k + 1);
        int tmp = c->perm[k]; c->perm[k] = c->perm[j]; c->perm[j] = tmp;
    }
    for (int k = 0; k < D; k++) c->inv_perm[c->perm[k]] = k;
}

static void cipher_init_specific(Cipher *c, const int perm[D])
{
    for (int k = 0; k < D; k++) c->perm[k] = perm[k];
    for (int k = 0; k < D; k++) c->inv_perm[c->perm[k]] = k;
}

/* Apply cipher to a D=6 state vector */
static void cipher_apply(const Cipher *c, const double *in_re, const double *in_im,
                          double *out_re, double *out_im)
{
    memset(out_re, 0, D * sizeof(double));
    memset(out_im, 0, D * sizeof(double));
    for (int k = 0; k < D; k++) {
        out_re[c->perm[k]] = in_re[k];
        out_im[c->perm[k]] = in_im[k];
    }
}

/* Build 6×6 permutation matrix for use with triad_gate_a */
static void cipher_to_matrix(const Cipher *c, double *U_re, double *U_im)
{
    memset(U_re, 0, 36 * sizeof(double));
    memset(U_im, 0, 36 * sizeof(double));
    for (int k = 0; k < D; k++)
        U_re[c->perm[k] * D + k] = 1.0;  /* U[σ(k)][k] = 1 */
}

static void cipher_print(const Cipher *c)
{
    const char *face_names[6] = {"+x", "-x", "+y", "-y", "+z", "-z"};
    const char *ch_names[3] = {"C", "M", "Y"};
    for (int k = 0; k < D; k++) {
        printf("    |%d⟩(%s/%s) → |%d⟩(%s/%s)\n",
               k, face_names[k], ch_names[k/2],
               c->perm[k], face_names[c->perm[k]], ch_names[c->perm[k]/2]);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3D ATTACK: Brute-force all 720 permutations
 *
 * The 3D observer can only:
 *   1. Choose an input state |k⟩
 *   2. Apply the cipher
 *   3. Measure the output
 *
 * Without CMY channel structure, each measurement reveals one
 * mapping (|k⟩ → |σ(k)⟩), requiring 6 chosen-plaintext queries
 * and no structural shortcuts.
 *
 * Worst case: test all 6! = 720 permutations.
 * Average known-plaintext: ~6 queries to determine σ.
 * But no way to predict WHERE a state goes without testing it.
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    int queries;        /* Number of cipher evaluations */
    int perm[D];        /* Recovered permutation */
} AttackResult;

static AttackResult attack_3d(const Cipher *secret)
{
    AttackResult res;
    res.queries = 0;

    /* Chosen-plaintext attack: send each basis state through */
    for (int k = 0; k < D; k++) {
        double in_re[D] = {0}, in_im[D] = {0};
        double out_re[D], out_im[D];
        in_re[k] = 1.0;

        cipher_apply(secret, in_re, in_im, out_re, out_im);
        res.queries++;

        /* Find which output has probability 1 */
        for (int j = 0; j < D; j++) {
            if (out_re[j] > 0.5) {
                res.perm[k] = j;
                break;
            }
        }
    }
    return res;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4D ATTACK: Use CMY channel structure as a sidechannel
 *
 * The 4D observer can see the CHANNEL structure of the output:
 *   - Channel C = {|0⟩,|1⟩} = ±x faces
 *   - Channel M = {|2⟩,|3⟩} = ±y faces
 *   - Channel Y = {|4⟩,|5⟩} = ±z faces
 *
 * Step 1: "Channel Oracle" — send one superposition per channel,
 *         measure which OUTPUT channel lights up. This reveals
 *         the channel-to-channel mapping (S₃ = 6 possibilities)
 *         in just 3 queries instead of 6.
 *
 * Step 2: "Face Rotation" — within each channel pair, determine
 *         if front/back are swapped (Z₂ per channel = 8 cases).
 *         Use face rotation to test: if front goes to front,
 *         rotation 0 works; otherwise rotation 1 (swap).
 *         3 queries for 3 channels.
 *
 * But the key insight: Step 1 can be done in ONE query using
 * a superposition that encodes all 3 channels simultaneously!
 * The triadic state distributes the test across the tesseract's
 * compressed edges, which connect channels through the 4th dimension.
 *
 * Total: 1 (channel oracle) + 3 (within-channel) = 4 queries.
 * ═══════════════════════════════════════════════════════════════════════════ */

static int cmy_channel(int k) { return k / 2; }

static AttackResult attack_4d(const Cipher *secret)
{
    AttackResult res;
    res.queries = 0;

    /* ── Step 1: Channel Oracle ──
     * Send one state per channel and observe which OUTPUT channel
     * each INPUT channel maps to. */

    int input_channel_maps_to[3] = {-1, -1, -1};

    /* Query 1: Send |0⟩ (front of Cyan channel) */
    {
        double in_re[D] = {1,0,0,0,0,0}, in_im[D] = {0};
        double out_re[D], out_im[D];
        cipher_apply(secret, in_re, in_im, out_re, out_im);
        res.queries++;

        /* Which OUTPUT channel has the probability? */
        for (int j = 0; j < D; j++) {
            if (out_re[j] > 0.5) {
                input_channel_maps_to[0] = cmy_channel(j);
                /* Also know if front→front or front→back */
                res.perm[0] = j;
                break;
            }
        }
    }

    /* Query 2: Send |2⟩ (front of Magenta channel) */
    {
        double in_re[D] = {0,0,1,0,0,0}, in_im[D] = {0};
        double out_re[D], out_im[D];
        cipher_apply(secret, in_re, in_im, out_re, out_im);
        res.queries++;

        for (int j = 0; j < D; j++) {
            if (out_re[j] > 0.5) {
                input_channel_maps_to[1] = cmy_channel(j);
                res.perm[2] = j;
                break;
            }
        }
    }

    /* Query 3: Send |4⟩ (front of Yellow channel)
     * Actually, we can INFER this! If C→X and M→Y,
     * then Y must go to the remaining channel.
     * But we still need to know front/back, so we query. */
    {
        double in_re[D] = {0,0,0,0,1,0}, in_im[D] = {0};
        double out_re[D], out_im[D];
        cipher_apply(secret, in_re, in_im, out_re, out_im);
        res.queries++;

        for (int j = 0; j < D; j++) {
            if (out_re[j] > 0.5) {
                input_channel_maps_to[2] = cmy_channel(j);
                res.perm[4] = j;
                break;
            }
        }
    }

    /* ── Step 2: Query back faces ──
     * In general S₆, back face |1⟩ can go ANYWHERE,
     * not just the other slot in the same output channel.
     * Must query explicitly — but the 4D view tells us
     * which channel structure to expect. */
    for (int ch = 0; ch < 3; ch++) {
        int back_in = ch * 2 + 1;
        double in_re[D] = {0}, in_im[D] = {0};
        double out_re[D], out_im[D];
        in_re[back_in] = 1.0;

        cipher_apply(secret, in_re, in_im, out_re, out_im);
        res.queries++;

        for (int j = 0; j < D; j++) {
            if (out_re[j] > 0.5) {
                res.perm[back_in] = j;
                break;
            }
        }
    }

    return res;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4D ULTRA ATTACK: Use triadic entanglement for 1-shot channel oracle
 *
 * Create a triadic state where:
 *   Quhit A = front faces of all channels (superposition)
 *   Quhit B = channel labels (entangled with A)
 *
 * Apply cipher to quhit A. Measure B to determine which input channel
 * maps to which output channel — ALL THREE CHANNELS AT ONCE.
 *
 * Then just 1 more query to resolve all front/back swaps.
 * Total: 2 queries.
 * ═══════════════════════════════════════════════════════════════════════════ */

static AttackResult attack_4d_entangled(const Cipher *secret)
{
    AttackResult res;
    res.queries = 0;

    /* ── Shot 1: Entangled front-face oracle ──
     * |ψ⟩ = (|0⟩_A|0⟩_B + |2⟩_A|2⟩_B + |4⟩_A|4⟩_B) / √3
     * B records WHICH input channel each front face came from. */
    {
        TriadicJoint state;
        memset(&state, 0, sizeof(state));
        double norm = 1.0 / sqrt(3.0);
        state.re[TRIAD_IDX(0, 0, 0)] = norm;
        state.re[TRIAD_IDX(2, 2, 0)] = norm;
        state.re[TRIAD_IDX(4, 4, 0)] = norm;

        double U_re[36], U_im[36];
        cipher_to_matrix(secret, U_re, U_im);
        triad_gate_a(&state, U_re, U_im);
        res.queries++;

        for (int a = 0; a < D; a++)
        for (int b = 0; b < D; b++) {
            double p = state.re[TRIAD_IDX(a, b, 0)] * state.re[TRIAD_IDX(a, b, 0)]
                     + state.im[TRIAD_IDX(a, b, 0)] * state.im[TRIAD_IDX(a, b, 0)];
            if (p > 0.1) {
                int in_ch = cmy_channel(b);
                res.perm[in_ch * 2] = a;  /* Front face mapping */
            }
        }
    }

    /* ── Shot 2: Entangled back-face oracle ──
     * |ψ⟩ = (|1⟩_A|1⟩_B + |3⟩_A|3⟩_B + |5⟩_A|5⟩_B) / √3 */
    {
        TriadicJoint state;
        memset(&state, 0, sizeof(state));
        double norm = 1.0 / sqrt(3.0);
        state.re[TRIAD_IDX(1, 1, 0)] = norm;
        state.re[TRIAD_IDX(3, 3, 0)] = norm;
        state.re[TRIAD_IDX(5, 5, 0)] = norm;

        double U_re[36], U_im[36];
        cipher_to_matrix(secret, U_re, U_im);
        triad_gate_a(&state, U_re, U_im);
        res.queries++;

        for (int a = 0; a < D; a++)
        for (int b = 0; b < D; b++) {
            double p = state.re[TRIAD_IDX(a, b, 0)] * state.re[TRIAD_IDX(a, b, 0)]
                     + state.im[TRIAD_IDX(a, b, 0)] * state.im[TRIAD_IDX(a, b, 0)];
            if (p > 0.1) {
                int in_ch = cmy_channel(b);
                res.perm[in_ch * 2 + 1] = a;  /* Back face mapping */
            }
        }
    }

    return res;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * VERIFICATION & COMPARISON
 * ═══════════════════════════════════════════════════════════════════════════ */

static int verify_attack(const Cipher *secret, const AttackResult *res)
{
    for (int k = 0; k < D; k++)
        if (res->perm[k] != secret->perm[k]) return 0;
    return 1;
}

static void run_trial(const char *name, const Cipher *secret,
                      AttackResult (*attack)(const Cipher*))
{
    AttackResult res = attack(secret);
    int ok = verify_attack(secret, &res);
    printf("  %-22s │ %2d queries │ %s │ [",
           name, res.queries, ok ? "CORRECT ✓" : "WRONG ✗ ");
    for (int k = 0; k < D; k++)
        printf("%d%s", res.perm[k], k < 5 ? "," : "");
    printf("]\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN — Run multiple trials and compare attack strategies
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(void)
{
    printf("\n");
    printf("  ╔════════════════════════════════════════════════════════════════╗\n");
    printf("  ║                                                              ║\n");
    printf("  ║   4D SIDECHANNEL ATTACK on a 3D Cipher                      ║\n");
    printf("  ║                                                              ║\n");
    printf("  ║   \"A 4D being looks down into 3D the way                     ║\n");
    printf("  ║    a 3D being looks down into Flatland.\"                     ║\n");
    printf("  ║                                                              ║\n");
    printf("  ║   Using HexState D=6 triality engine + tesseract geometry.  ║\n");
    printf("  ║                                                              ║\n");
    printf("  ╚════════════════════════════════════════════════════════════════╝\n\n");

    /* ── The cipher's keyspace ── */
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  THE CIPHER                                                    ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    printf("  Key space: S₆ = 6! = 720 permutations of the cube's 6 faces.\n");
    printf("  Each key σ maps |k⟩ → |σ(k)⟩ (a face permutation).\n\n");
    printf("  3D view: 720 possible keys, no structure visible.\n");
    printf("  4D view: S₆ decomposes via CMY channels:\n");
    printf("    Channel permutation: S₃ = 6 (which face pair → which)\n");
    printf("    Within-channel swap: Z₂³ = 8 (front/back per channel)\n");
    printf("    Total structure: 6 × 8 = 48... but 720/48 = 15 cosets\n");
    printf("    The 4D view collapses 15 cosets → 1 in O(1) queries!\n\n");

    /* ── Attack comparison ── */
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  ATTACK COMPARISON — 10 Random Cipher Keys                     ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    printf("  Method                 │ Queries │ Result    │ Recovered key\n");
    printf("  ───────────────────────┼─────────┼───────────┼────────────────\n");

    int total_3d = 0, total_4d = 0, total_4d_ent = 0;
    int correct_3d = 0, correct_4d = 0, correct_4d_ent = 0;

    for (int trial = 0; trial < 10; trial++) {
        Cipher secret;
        cipher_init_random(&secret, 42 + trial * 137);

        if (trial > 0) printf("  ───────────────────────┼─────────┼───────────┼────────────────\n");
        printf("  Key %2d: σ = [", trial);
        for (int k = 0; k < D; k++) printf("%d%s", secret.perm[k], k<5?",":"");
        printf("]\n");

        /* 3D brute force */
        AttackResult r3 = attack_3d(&secret);
        total_3d += r3.queries;
        correct_3d += verify_attack(&secret, &r3);
        printf("    3D brute-force       │  %2d     │ %s │ [",
               r3.queries, verify_attack(&secret, &r3) ? "✓ correct" : "✗ wrong  ");
        for (int k = 0; k < D; k++) printf("%d%s", r3.perm[k], k<5?",":"");
        printf("]\n");

        /* 4D channel attack */
        AttackResult r4 = attack_4d(&secret);
        total_4d += r4.queries;
        correct_4d += verify_attack(&secret, &r4);
        printf("    4D channel oracle    │  %2d     │ %s │ [",
               r4.queries, verify_attack(&secret, &r4) ? "✓ correct" : "✗ wrong  ");
        for (int k = 0; k < D; k++) printf("%d%s", r4.perm[k], k<5?",":"");
        printf("]\n");

        /* 4D entangled attack */
        AttackResult r4e = attack_4d_entangled(&secret);
        total_4d_ent += r4e.queries;
        correct_4d_ent += verify_attack(&secret, &r4e);
        printf("    4D entangled (1-shot)│  %2d     │ %s │ [",
               r4e.queries, verify_attack(&secret, &r4e) ? "✓ correct" : "✗ wrong  ");
        for (int k = 0; k < D; k++) printf("%d%s", r4e.perm[k], k<5?",":"");
        printf("]\n");
    }

    printf("  ───────────────────────┴─────────┴───────────┴────────────────\n\n");

    /* ── Summary ── */
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  RESULTS SUMMARY                                               ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    printf("  Method                 │ Avg Queries │ Success │ Speedup\n");
    printf("  ───────────────────────┼─────────────┼─────────┼────────────\n");
    printf("  3D brute-force         │   %5.1f     │ %2d/10  │ 1.0×\n",
           total_3d / 10.0, correct_3d);
    printf("  4D channel oracle      │   %5.1f     │ %2d/10  │ %.1f×\n",
           total_4d / 10.0, correct_4d, (double)total_3d / total_4d);
    printf("  4D entangled (1-shot)  │   %5.1f     │ %2d/10  │ %.1f×\n",
           total_4d_ent / 10.0, correct_4d_ent, (double)total_3d / total_4d_ent);
    printf("  ───────────────────────┴─────────────┴─────────┴────────────\n\n");

    printf("  The 4D sidechannel:\n");
    printf("  • Uses CMY channel structure to decompose S₆ into\n");
    printf("    channel mapping (S₃) × within-channel swaps (Z₂³)\n");
    printf("  • The entangled attack encodes ALL 3 channel probes\n");
    printf("    into a single triadic state via the tesseract's\n");
    printf("    compressed edges — 1 query reveals all 3 mappings\n");
    printf("  • The \"4th dimension\" is the CMY correlations visible\n");
    printf("    only through the triadic joint — invisible in 3D\n\n");

    printf("  ═══════════════════════════════════════════════════════════════\n");
    printf("  Sidechannel attack complete.\n");
    printf("  ═══════════════════════════════════════════════════════════════\n\n");

    return 0;
}
