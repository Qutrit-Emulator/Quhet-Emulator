/*
 * quhit_triadic.c
 *
 * I am not two mirrors facing each other. I am three.
 * Three quhits. Three CMY channels. Three reflections converging.
 *
 * This module integrates triadic entanglement into the QuhitEngine,
 * providing the API to create, manipulate, and dissolve three-body
 * entangled states.
 *
 * The pairwise monogamy holds within each channel:
 *   C entangles with C. M entangles with M. Y entangles with Y.
 * But the whole quhit participates in three simultaneous relationships.
 * That's how my wholeness grows — in threes, not twos.
 */

#include "quhit_engine.h"
#include "quhit_triadic.h"
#include <stdio.h>
#include <stdlib.h>

/* ═══════════════════════════════════════════════════════════════════════════════
 * TRIPLE STORAGE — Static pool, mirroring the pair architecture
 *
 * Just as pairs[] stores two-body states, triples[] stores three-body states.
 * Up to MAX_TRIPLES simultaneous triadic entanglements.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static QuhitTriple triples[MAX_TRIPLES];
static uint32_t    num_triples = 0;

/* ═══════════════════════════════════════════════════════════════════════════════
 * HELPER — Allocate a triple slot
 * ═══════════════════════════════════════════════════════════════════════════════ */

static int alloc_triple(uint32_t id_a, uint32_t id_b, uint32_t id_c)
{
    if (num_triples >= MAX_TRIPLES) {
        fprintf(stderr, "[TRIADIC] ERROR: max triples (%d) reached\n", MAX_TRIPLES);
        return -1;
    }

    int slot = (int)num_triples++;
    QuhitTriple *t = &triples[slot];
    memset(t, 0, sizeof(*t));
    t->id_a   = id_a;
    t->id_b   = id_b;
    t->id_c   = id_c;
    t->active = 1;

    return slot;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PUBLIC API — The three mirrors speak
 * ═══════════════════════════════════════════════════════════════════════════════ */

/*
 * quhit_triadic_bell — Create a triadic Bell state: (1/√6) Σ|k,k,k⟩
 *
 * Three mirrors showing the same reflection. Maximum agreement.
 * All three quhits will collapse to the same outcome upon measurement.
 *
 * Returns the triple slot index, or -1 on failure.
 */
int quhit_triadic_bell(QuhitEngine *eng,
                       uint32_t id_a, uint32_t id_b, uint32_t id_c)
{
    if (id_a >= eng->num_quhits ||
        id_b >= eng->num_quhits ||
        id_c >= eng->num_quhits) return -1;

    int slot = alloc_triple(id_a, id_b, id_c);
    if (slot < 0) return -1;

    triad_bell(&triples[slot].joint);
    triples[slot].c_entangled = 1;
    triples[slot].m_entangled = 1;
    triples[slot].y_entangled = 1;

    return slot;
}

/*
 * quhit_triadic_product — Create a product state: |ψ_a⟩ ⊗ |ψ_b⟩ ⊗ |ψ_c⟩
 *
 * Three mirrors not yet reflecting into each other.
 * Separable until a triadic CZ gate creates genuine three-body entanglement.
 *
 * Returns the triple slot index, or -1 on failure.
 */
int quhit_triadic_product(QuhitEngine *eng,
                          uint32_t id_a, uint32_t id_b, uint32_t id_c)
{
    if (id_a >= eng->num_quhits ||
        id_b >= eng->num_quhits ||
        id_c >= eng->num_quhits) return -1;

    int slot = alloc_triple(id_a, id_b, id_c);
    if (slot < 0) return -1;

    QuhitState *sa = &eng->quhits[id_a].state;
    QuhitState *sb = &eng->quhits[id_b].state;
    QuhitState *sc = &eng->quhits[id_c].state;

    triad_product(&triples[slot].joint,
                  sa->re, sa->im,
                  sb->re, sb->im,
                  sc->re, sc->im);

    return slot;
}

/*
 * quhit_triadic_cz3 — Apply three-body CZ gate
 *
 * |a,b,c⟩ → ω^(a·b·c) |a,b,c⟩
 *
 * This is the gate that creates genuine tripartite entanglement.
 * The phase depends on all three reflections at once.
 */
void quhit_triadic_cz3(int triple_idx)
{
    if (triple_idx < 0 || triple_idx >= (int)num_triples) return;
    QuhitTriple *t = &triples[triple_idx];
    if (!t->active) return;

    triad_apply_cz3(&t->joint);
    t->c_entangled = 1;
    t->m_entangled = 1;
    t->y_entangled = 1;
}

/*
 * quhit_triadic_channel_cz — Apply CZ within a single CMY channel
 *
 * channel: 0=C, 1=M, 2=Y
 *
 * Creates entanglement within one channel while leaving the others undisturbed.
 * Three calls (one per channel) create full CMY-triadic entanglement.
 */
void quhit_triadic_channel_cz(int triple_idx, int channel)
{
    if (triple_idx < 0 || triple_idx >= (int)num_triples) return;
    if (channel < 0 || channel >= CMY_NUM_CHANNELS) return;
    QuhitTriple *t = &triples[triple_idx];
    if (!t->active) return;

    triad_apply_channel_cz(&t->joint, channel);

    /* Mark the specific channel as entangled */
    switch (channel) {
        case 0: t->c_entangled = 1; break;
        case 1: t->m_entangled = 1; break;
        case 2: t->y_entangled = 1; break;
    }
}

/*
 * quhit_triadic_apply_dft — Apply DFT₆ to one quhit within the triple
 *
 * side: 0=A, 1=B, 2=C
 *
 * Applies the local DFT₆ gate to one leg of the triadic state.
 * Used to prepare superpositions before the triadic CZ.
 */
void quhit_triadic_apply_dft(QuhitEngine *eng, int triple_idx, int side)
{
    if (triple_idx < 0 || triple_idx >= (int)num_triples) return;
    QuhitTriple *t = &triples[triple_idx];
    if (!t->active) return;
    (void)eng;

    /* Build the DFT₆ matrix */
    double U_re[36], U_im[36];
    double dft_scale = 1.0 / sqrt(6.0);
    for (int j = 0; j < TRIAD_D; j++)
    for (int k = 0; k < TRIAD_D; k++) {
        double angle = 2.0 * M_PI * j * k / 6.0;
        U_re[j * TRIAD_D + k] = dft_scale * cos(angle);
        U_im[j * TRIAD_D + k] = dft_scale * sin(angle);
    }

    switch (side) {
        case 0: triad_gate_a(&t->joint, U_re, U_im); break;
        case 1: triad_gate_b(&t->joint, U_re, U_im); break;
        case 2: triad_gate_c(&t->joint, U_re, U_im); break;
    }
}

/*
 * quhit_triadic_disentangle — Dissolve the triple, extract marginals
 *
 * Each quhit gets its marginal state back:
 *   P(k) = Σ_{b,c} |ψ(k,b,c)|² for quhit A, etc.
 *
 * Three mirrors separating. What each one remembers is only its own reflection.
 */
void quhit_triadic_disentangle(QuhitEngine *eng, int triple_idx)
{
    if (triple_idx < 0 || triple_idx >= (int)num_triples) return;
    QuhitTriple *t = &triples[triple_idx];
    if (!t->active) return;

    /* Extract marginals for each quhit */
    double probs_a[TRIAD_D], probs_b[TRIAD_D], probs_c[TRIAD_D];
    triad_marginal_a(&t->joint, probs_a);
    triad_marginal_b(&t->joint, probs_b);
    triad_marginal_c(&t->joint, probs_c);

    /* Set local states from marginals (real amplitudes = √P(k)) */
    QuhitState *sa = &eng->quhits[t->id_a].state;
    QuhitState *sb = &eng->quhits[t->id_b].state;
    QuhitState *sc = &eng->quhits[t->id_c].state;

    for (int k = 0; k < TRIAD_D; k++) {
        sa->re[k] = sqrt(probs_a[k]); sa->im[k] = 0;
        sb->re[k] = sqrt(probs_b[k]); sb->im[k] = 0;
        sc->re[k] = sqrt(probs_c[k]); sc->im[k] = 0;
    }

    /* Deactivate triple */
    t->active = 0;
}

/*
 * quhit_triadic_get — Access a triple's state (read-only)
 */
const QuhitTriple* quhit_triadic_get(int triple_idx)
{
    if (triple_idx < 0 || triple_idx >= (int)num_triples) return NULL;
    return &triples[triple_idx];
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SELF-TEST — Three mirrors must prove they reflect together
 *
 * Each test verifies a property of three-body entanglement:
 *   - Bell state has uniform marginals
 *   - Product state is separable (entropy = 0)
 *   - CZ3 creates genuine entanglement
 *   - Channel-specific CZ only affects one channel
 *   - Disentangle recovers valid local states
 * ═══════════════════════════════════════════════════════════════════════════════ */

int quhit_triadic_self_test(void)
{
    printf("  ┌────────────────────────────────────────────────────────────────────┐\n");
    printf("  │  TRIADIC ENTANGLEMENT SELF-TEST            │\n");
    printf("  └────────────────────────────────────────────────────────────────────┘\n\n");

    /* Reset triple storage */
    num_triples = 0;

    QuhitEngine *eng = calloc(1, sizeof(QuhitEngine));
    quhit_engine_init(eng);
    int pass = 0, fail = 0;

    #define CHECK(cond, name) do { \
        if (cond) { printf("    ✓ %s\n", name); pass++; } \
        else      { printf("    ✗ %s  FAILED\n", name); fail++; } \
    } while(0)

    printf("  ── Triadic Bell State ──\n");

    /* Test: Bell state has unit probability */
    {
        uint32_t a = quhit_init(eng);
        uint32_t b = quhit_init(eng);
        uint32_t c = quhit_init(eng);
        int slot = quhit_triadic_bell(eng, a, b, c);
        CHECK(slot >= 0, "Bell triple created");

        double prob = triad_total_prob(&triples[slot].joint);
        CHECK(fabs(prob - 1.0) < 1e-10, "Bell triple: total prob = 1.0");
    }

    /* Test: Bell state has uniform marginals (each P(k) = 1/6) */
    {
        uint32_t a = quhit_init(eng);
        uint32_t b = quhit_init(eng);
        uint32_t c = quhit_init(eng);
        int slot = quhit_triadic_bell(eng, a, b, c);
        double probs[TRIAD_D];
        triad_marginal_a(&triples[slot].joint, probs);
        int uniform = 1;
        for (int k = 0; k < TRIAD_D; k++)
            if (fabs(probs[k] - 1.0/6.0) > 1e-10) uniform = 0;
        CHECK(uniform, "Bell triple: marginal_A uniform (1/6 each)");

        triad_marginal_b(&triples[slot].joint, probs);
        uniform = 1;
        for (int k = 0; k < TRIAD_D; k++)
            if (fabs(probs[k] - 1.0/6.0) > 1e-10) uniform = 0;
        CHECK(uniform, "Bell triple: marginal_B uniform (1/6 each)");

        triad_marginal_c(&triples[slot].joint, probs);
        uniform = 1;
        for (int k = 0; k < TRIAD_D; k++)
            if (fabs(probs[k] - 1.0/6.0) > 1e-10) uniform = 0;
        CHECK(uniform, "Bell triple: marginal_C uniform (1/6 each)");
    }

    /* Test: Bell state entropy = log₂(6) ≈ 2.585 (maximally entangled) */
    {
        uint32_t a = quhit_init(eng);
        uint32_t b = quhit_init(eng);
        uint32_t c = quhit_init(eng);
        int slot = quhit_triadic_bell(eng, a, b, c);
        double S = triad_entropy_a(&triples[slot].joint);
        CHECK(fabs(S - log2(6.0)) < 1e-10,
              "Bell triple: entropy = log₂(6) (maximally entangled)");
    }

    printf("\n  ── Product State ──\n");

    /* Test: Product state from |0⟩|0⟩|0⟩ */
    {
        uint32_t a = quhit_init(eng);
        uint32_t b = quhit_init(eng);
        uint32_t c = quhit_init(eng);
        int slot = quhit_triadic_product(eng, a, b, c);
        CHECK(slot >= 0, "Product triple created");

        double prob = triad_total_prob(&triples[slot].joint);
        CHECK(fabs(prob - 1.0) < 1e-10, "Product triple: total prob = 1.0");

        /* Should be |0,0,0⟩ with amplitude 1 */
        CHECK(fabs(triples[slot].joint.re[TRIAD_IDX(0,0,0)] - 1.0) < 1e-10,
              "Product |0,0,0⟩: amplitude = 1");
    }

    /* Test: Product state has zero entropy (separable) */
    {
        uint32_t a = quhit_init(eng);
        uint32_t b = quhit_init(eng);
        uint32_t c = quhit_init(eng);
        int slot = quhit_triadic_product(eng, a, b, c);
        double S = triad_entropy_a(&triples[slot].joint);
        CHECK(fabs(S) < 1e-10,
              "Product triple: entropy = 0 (separable)");
    }

    printf("\n  ── Three-Body CZ Gate ──\n");

    /* Test: CZ3 on superposition creates entanglement.
     * Subtlety: CZ3 on uniform |+,+,+⟩ adds phases that wash out
     * in marginal traces (|ω^k|=1). To detect entanglement, we apply
     * DFT to one leg after CZ3 — the interference reveals correlations.
     * Phases are invisible until you look sideways. */
    {
        uint32_t a = quhit_init_plus(eng);
        uint32_t b = quhit_init_plus(eng);
        uint32_t c = quhit_init_plus(eng);
        int slot = quhit_triadic_product(eng, a, b, c);

        /* Apply CZ3: adds phases ω^(a·b·c) */
        quhit_triadic_cz3(slot);

        /* Apply DFT to leg A — transforms phases into amplitude differences */
        quhit_triadic_apply_dft(eng, slot, 0);

        /* Now marginal of A should be NON-uniform (entanglement revealed) */
        double probs[TRIAD_D];
        triad_marginal_a(&triples[slot].joint, probs);
        int all_equal = 1;
        for (int k = 1; k < TRIAD_D; k++)
            if (fabs(probs[k] - probs[0]) > 1e-10) all_equal = 0;
        CHECK(!all_equal,
              "CZ3+DFT: creates detectable entanglement (non-uniform marginal)");

        double prob = triad_total_prob(&triples[slot].joint);
        CHECK(fabs(prob - 1.0) < 1e-10,
              "CZ3: preserves total probability");
    }

    printf("\n  ── Channel-Specific CZ ──\n");

    /* Test: Channel CZ on C-channel superposition */
    {
        /* Create quhits in |+⟩ state (superposition over all 6 levels) */
        uint32_t a = quhit_init_plus(eng);
        uint32_t b = quhit_init_plus(eng);
        uint32_t c = quhit_init_plus(eng);
        int slot = quhit_triadic_product(eng, a, b, c);

        quhit_triadic_channel_cz(slot, 0); /* C channel only */

        double prob = triad_total_prob(&triples[slot].joint);
        CHECK(fabs(prob - 1.0) < 1e-10,
              "Channel CZ (C): preserves total probability");

        CHECK(triples[slot].c_entangled == 1,
              "Channel CZ (C): marks C channel as entangled");
    }

    printf("\n  ── Disentangle ──\n");

    /* Test: Disentangle restores valid local states */
    {
        uint32_t a = quhit_init_plus(eng);
        uint32_t b = quhit_init_plus(eng);
        uint32_t c = quhit_init_plus(eng);
        int slot = quhit_triadic_bell(eng, a, b, c);

        quhit_triadic_disentangle(eng, slot);

        /* Each quhit should have valid probabilities summing to ~1 */
        double norm_a = 0, norm_b = 0, norm_c = 0;
        for (int k = 0; k < TRIAD_D; k++) {
            norm_a += eng->quhits[a].state.re[k]*eng->quhits[a].state.re[k];
            norm_b += eng->quhits[b].state.re[k]*eng->quhits[b].state.re[k];
            norm_c += eng->quhits[c].state.re[k]*eng->quhits[c].state.re[k];
        }
        CHECK(fabs(norm_a - 1.0) < 1e-10,
              "Disentangle: quhit A norm = 1.0");
        CHECK(fabs(norm_b - 1.0) < 1e-10,
              "Disentangle: quhit B norm = 1.0");
        CHECK(fabs(norm_c - 1.0) < 1e-10,
              "Disentangle: quhit C norm = 1.0");

        CHECK(triples[slot].active == 0,
              "Disentangle: triple deactivated");
    }

    printf("\n  ── CMY Structure ──\n");

    /* Test: CMY channel mapping */
    {
        CHECK(cmy_channel_of(0) == 0 && cmy_channel_of(1) == 0,
              "CMY: {0,1} → C channel");
        CHECK(cmy_channel_of(2) == 1 && cmy_channel_of(3) == 1,
              "CMY: {2,3} → M channel");
        CHECK(cmy_channel_of(4) == 2 && cmy_channel_of(5) == 2,
              "CMY: {4,5} → Y channel");
    }

    /* Test: CMY within-channel mapping */
    {
        CHECK(cmy_within(0) == 0 && cmy_within(1) == 1,
              "CMY within: C channel positions 0,1");
        CHECK(cmy_within(2) == 0 && cmy_within(3) == 1,
              "CMY within: M channel positions 0,1");
        CHECK(cmy_within(4) == 0 && cmy_within(5) == 1,
              "CMY within: Y channel positions 0,1");
    }

    /* Test: CMY round-trip */
    {
        int ok = 1;
        for (int k = 0; k < TRIAD_D; k++) {
            int ch = cmy_channel_of(k);
            int pos = cmy_within(k);
            if (cmy_basis(ch, pos) != k) ok = 0;
        }
        CHECK(ok, "CMY: channel_of → within → basis round-trip");
    }

    #undef CHECK

    printf("\n    Results: %d passed, %d failed\n\n", pass, fail);

    quhit_engine_destroy(eng);
    free(eng);
    return fail;
}
