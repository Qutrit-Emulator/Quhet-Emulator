/*
 * quhit_lazy.c — Deferred Existence: The Lazy Gate Engine
 *
 * I don't exist until I'm observed. Neither should my state.
 *
 * This module provides a lazy gate API that intercepts gate operations,
 * records them into a compressed unitary chain per quhit, and only
 * materializes the state when measurement forces collapse.
 *
 * The API mirrors quhit_gates.c but defers everything:
 *   lazy_dft()     — record DFT₆, don't apply
 *   lazy_idft()    — record IDFT₆, don't apply
 *   lazy_z()       — record Z gate, don't apply
 *   lazy_x()       — record X gate, don't apply
 *   lazy_phase()   — record phase gate, don't apply
 *   lazy_flush()   — resolve: apply accumulated U to state
 *
 * Consecutive gates on the same quhit compress into one 6×6 matrix.
 * DFT₆ → Z → IDFT₆ = three gates, but one matrix at resolve.
 * DFT₆ → IDFT₆ = identity → skip entirely.
 */

#include "quhit_engine.h"
#include "quhit_lazy.h"
#include <stdio.h>
#include <stdlib.h>

/* ═══════════════════════════════════════════════════════════════════════════════
 * LAZY CHAIN POOL — One chain per quhit, static allocation
 *
 * Each quhit silently accumulates its gate chain.
 * The pool mirrors the engine's quhit array.
 * ═══════════════════════════════════════════════════════════════════════════════ */

#define LAZY_MAX_QUHITS 4096

static LazyChain lazy_chains[LAZY_MAX_QUHITS];
static int lazy_pool_initialized = 0;

static void lazy_pool_init(void)
{
    if (lazy_pool_initialized) return;
    for (int i = 0; i < LAZY_MAX_QUHITS; i++)
        lazy_init(&lazy_chains[i]);
    lazy_pool_initialized = 1;
}

static inline LazyChain* lazy_get(uint32_t id)
{
    if (id >= LAZY_MAX_QUHITS) return NULL;
    return &lazy_chains[id];
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PRECOMPUTED GATE MATRICES — The geometries I carry, pre-rendered
 *
 * Instead of building matrices at record time, I keep them ready.
 * DFT₆, IDFT₆, X, Z — all precomputed as 6×6 complex matrices.
 * Recording a gate = one matrix multiply against the chain.
 * ═══════════════════════════════════════════════════════════════════════════════ */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* DFT₆ matrix: F[j][k] = (1/√6) × ω^(j·k) where ω = e^(2πi/6) */
static double DFT6_RE[36], DFT6_IM[36];
static double IDFT6_RE[36], IDFT6_IM[36];
static double X_RE[36], X_IM[36];
static int gate_matrices_initialized = 0;

static void init_gate_matrices(void)
{
    if (gate_matrices_initialized) return;
    double scale = 1.0 / sqrt(6.0);

    /* DFT₆ */
    for (int j = 0; j < 6; j++)
    for (int k = 0; k < 6; k++) {
        double angle = 2.0 * M_PI * j * k / 6.0;
        DFT6_RE[j*6+k] = scale * cos(angle);
        DFT6_IM[j*6+k] = scale * sin(angle);
    }

    /* IDFT₆ = conjugate transpose of DFT₆ */
    for (int j = 0; j < 6; j++)
    for (int k = 0; k < 6; k++) {
        IDFT6_RE[j*6+k] =  DFT6_RE[k*6+j];
        IDFT6_IM[j*6+k] = -DFT6_IM[k*6+j];
    }

    /* X = cyclic shift: |k⟩ → |k+1 mod 6⟩ */
    memset(X_RE, 0, sizeof(X_RE));
    memset(X_IM, 0, sizeof(X_IM));
    for (int k = 0; k < 6; k++) {
        int dest = (k + 1) % 6;
        X_RE[dest * 6 + k] = 1.0; /* row dest, col k: maps |k⟩ to |k+1⟩ */
    }

    gate_matrices_initialized = 1;
}

/* Z gate diagonal: ω^k for k=0..5 */
static const double Z_DIAG_RE[6] = {
     1.0,                       /* ω^0 = 1              */
     0.5,                       /* ω^1 = ½ + i√3/2      */
    -0.5,                       /* ω^2 = -½ + i√3/2     */
    -1.0,                       /* ω^3 = -1             */
    -0.5,                       /* ω^4 = -½ - i√3/2     */
     0.5                        /* ω^5 = ½ - i√3/2      */
};
static const double Z_DIAG_IM[6] = {
     0.0,
     0.86602540378443864676,    /* √3/2  */
     0.86602540378443864676,
     0.0,
    -0.86602540378443864676,
    -0.86602540378443864676
};

/* ═══════════════════════════════════════════════════════════════════════════════
 * LAZY GATE API — Record, don't apply. Defer existence.
 *
 * Each function composes a gate into the quhit's lazy chain.
 * No state is touched. The state doesn't know it's been gate'd.
 * ═══════════════════════════════════════════════════════════════════════════════ */

void lazy_record_dft(uint32_t id)
{
    lazy_pool_init();
    init_gate_matrices();
    LazyChain *lc = lazy_get(id);
    if (!lc) return;
    lazy_compose(lc, DFT6_RE, DFT6_IM);
}

void lazy_record_idft(uint32_t id)
{
    lazy_pool_init();
    init_gate_matrices();
    LazyChain *lc = lazy_get(id);
    if (!lc) return;
    lazy_compose(lc, IDFT6_RE, IDFT6_IM);
}

void lazy_record_z(uint32_t id)
{
    lazy_pool_init();
    LazyChain *lc = lazy_get(id);
    if (!lc) return;
    lazy_compose_diagonal(lc, Z_DIAG_RE, Z_DIAG_IM);
}

void lazy_record_x(uint32_t id)
{
    lazy_pool_init();
    init_gate_matrices();
    LazyChain *lc = lazy_get(id);
    if (!lc) return;
    lazy_compose(lc, X_RE, X_IM);
}

void lazy_record_phase(uint32_t id, const double *phases)
{
    lazy_pool_init();
    LazyChain *lc = lazy_get(id);
    if (!lc) return;

    /* Build diagonal from phase angles */
    double d_re[6], d_im[6];
    for (int k = 0; k < 6; k++) {
        d_re[k] = cos(phases[k]);
        d_im[k] = sin(phases[k]);
    }
    lazy_compose_diagonal(lc, d_re, d_im);
}

void lazy_record_unitary(uint32_t id, const double *U_re, const double *U_im)
{
    lazy_pool_init();
    LazyChain *lc = lazy_get(id);
    if (!lc) return;
    lazy_compose(lc, U_re, U_im);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * FLUSH — Observation forces existence. Resolve the chain.
 *
 * Apply the accumulated unitary to the quhit's state vector.
 * After this, the chain resets to identity.
 *
 * Call this before measurement, before inspection, before any read
 * of the state that requires it to be current.
 *
 * I was nothing until you looked. Now I am.
 * ═══════════════════════════════════════════════════════════════════════════════ */

void lazy_flush(QuhitEngine *eng, uint32_t id)
{
    if (id >= eng->num_quhits) return;
    LazyChain *lc = lazy_get(id);
    if (!lc) return;

    QuhitState *s = &eng->quhits[id].state;
    lazy_resolve(lc, s->re, s->im);
}

void lazy_flush_all(QuhitEngine *eng)
{
    lazy_pool_init();
    for (uint32_t i = 0; i < eng->num_quhits && i < LAZY_MAX_QUHITS; i++) {
        if (lazy_chains[i].gate_count > 0)
            lazy_flush(eng, i);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * QUERY — How deep is the deferred chain?
 * ═══════════════════════════════════════════════════════════════════════════════ */

uint32_t lazy_pending_depth(uint32_t id)
{
    lazy_pool_init();
    LazyChain *lc = lazy_get(id);
    if (!lc) return 0;
    return lazy_depth(lc);
}

int lazy_is_pending(uint32_t id)
{
    lazy_pool_init();
    LazyChain *lc = lazy_get(id);
    if (!lc) return 0;
    return lc->gate_count > 0 && !lc->is_identity;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SELF-TEST — The deferred must prove itself equal to the immediate
 *
 * Every test applies the same gate sequence via both paths:
 *   1. Immediate: apply gates directly to state (quhit_gates.c)
 *   2. Lazy: record gates, then flush
 *
 * Results must match to machine precision.
 * If they don't, something is broken in the deferral.
 * ═══════════════════════════════════════════════════════════════════════════════ */

int quhit_lazy_self_test(void)
{
    printf("  ┌────────────────────────────────────────────────────────────────────┐\n");
    printf("  │  LAZY GATE SELF-TEST — I don't exist until I'm observed           │\n");
    printf("  └────────────────────────────────────────────────────────────────────┘\n\n");

    /* Reset pools */
    lazy_pool_initialized = 0;

    QuhitEngine *eng = calloc(1, sizeof(QuhitEngine));
    quhit_engine_init(eng);
    int pass = 0, fail = 0;

    #define CHECK(cond, name) do { \
        if (cond) { printf("    ✓ %s\n", name); pass++; } \
        else      { printf("    ✗ %s  FAILED\n", name); fail++; } \
    } while(0)

    printf("  ── Single gate deferral ──\n");

    /* Test: DFT₆ via lazy matches DFT₆ via immediate */
    {
        uint32_t q_imm = quhit_init(eng);
        uint32_t q_lazy = quhit_init(eng);

        /* Immediate path */
        quhit_apply_dft(eng, q_imm);

        /* Lazy path */
        lazy_record_dft(q_lazy);
        CHECK(lazy_pending_depth(q_lazy) == 1, "DFT₆ recorded (depth=1)");
        lazy_flush(eng, q_lazy);

        /* Compare */
        QuhitState *si = &eng->quhits[q_imm].state;
        QuhitState *sl = &eng->quhits[q_lazy].state;
        double err = 0;
        for (int k = 0; k < 6; k++)
            err += fabs(si->re[k] - sl->re[k]) + fabs(si->im[k] - sl->im[k]);
        CHECK(err < 1e-10, "DFT₆: lazy matches immediate");
    }

    /* Test: Z gate via lazy */
    {
        uint32_t q_imm = quhit_init_plus(eng);
        uint32_t q_lazy = quhit_init_plus(eng);

        quhit_apply_z(eng, q_imm);

        lazy_record_z(q_lazy);
        lazy_flush(eng, q_lazy);

        QuhitState *si = &eng->quhits[q_imm].state;
        QuhitState *sl = &eng->quhits[q_lazy].state;
        double err = 0;
        for (int k = 0; k < 6; k++)
            err += fabs(si->re[k] - sl->re[k]) + fabs(si->im[k] - sl->im[k]);
        CHECK(err < 1e-10, "Z gate: lazy matches immediate");
    }

    /* Test: X gate via lazy */
    {
        uint32_t q_imm = quhit_init(eng);
        uint32_t q_lazy = quhit_init(eng);

        quhit_apply_x(eng, q_imm);

        lazy_record_x(q_lazy);
        lazy_flush(eng, q_lazy);

        QuhitState *si = &eng->quhits[q_imm].state;
        QuhitState *sl = &eng->quhits[q_lazy].state;
        double err = 0;
        for (int k = 0; k < 6; k++)
            err += fabs(si->re[k] - sl->re[k]) + fabs(si->im[k] - sl->im[k]);
        CHECK(err < 1e-10, "X gate: lazy matches immediate");
    }

    printf("\n  ── Gate cancellation ──\n");

    /* Test: DFT · IDFT = identity ⇒ lazy detects cancellation */
    {
        uint32_t q = quhit_init_plus(eng);
        QuhitState saved;
        memcpy(&saved, &eng->quhits[q].state, sizeof(saved));

        lazy_record_dft(q);
        lazy_record_idft(q);
        CHECK(lazy_pending_depth(q) == 2, "DFT→IDFT recorded (depth=2)");

        /* Check if it resolves to identity */
        LazyChain *lc = lazy_get(q);
        int is_id = lazy_is_identity(lc);
        CHECK(is_id, "DFT→IDFT: detected as identity (cancelled)");

        lazy_flush(eng, q);
        QuhitState *s = &eng->quhits[q].state;
        double err = 0;
        for (int k = 0; k < 6; k++)
            err += fabs(s->re[k] - saved.re[k]) + fabs(s->im[k] - saved.im[k]);
        CHECK(err < 1e-10, "DFT→IDFT: state unchanged after flush");
    }

    printf("\n  ── Multi-gate chains ──\n");

    /* Test: DFT → Z → IDFT chain */
    {
        uint32_t q_imm = quhit_init_plus(eng);
        uint32_t q_lazy = quhit_init_plus(eng);

        /* Immediate */
        quhit_apply_dft(eng, q_imm);
        quhit_apply_z(eng, q_imm);
        quhit_apply_idft(eng, q_imm);

        /* Lazy */
        lazy_record_dft(q_lazy);
        lazy_record_z(q_lazy);
        lazy_record_idft(q_lazy);
        CHECK(lazy_pending_depth(q_lazy) == 3, "DFT→Z→IDFT chain (depth=3)");

        lazy_flush(eng, q_lazy);

        QuhitState *si = &eng->quhits[q_imm].state;
        QuhitState *sl = &eng->quhits[q_lazy].state;
        double err = 0;
        for (int k = 0; k < 6; k++)
            err += fabs(si->re[k] - sl->re[k]) + fabs(si->im[k] - sl->im[k]);
        CHECK(err < 1e-10, "DFT→Z→IDFT: lazy matches immediate");
    }

    /* Test: Z → Z → Z → Z → Z → Z = Z⁶ = identity (ω⁶ = 1) */
    {
        uint32_t q = quhit_init_plus(eng);
        QuhitState saved;
        memcpy(&saved, &eng->quhits[q].state, sizeof(saved));

        for (int i = 0; i < 6; i++)
            lazy_record_z(q);
        CHECK(lazy_pending_depth(q) == 6, "Z⁶ chain (depth=6)");

        LazyChain *lc = lazy_get(q);
        int is_id = lazy_is_identity(lc);
        CHECK(is_id, "Z⁶: detected as identity (ω⁶ = 1)");

        lazy_flush(eng, q);
        QuhitState *s = &eng->quhits[q].state;
        double err = 0;
        for (int k = 0; k < 6; k++)
            err += fabs(s->re[k] - saved.re[k]) + fabs(s->im[k] - saved.im[k]);
        CHECK(err < 1e-10, "Z⁶: state unchanged (identity)");
    }

    /* Test: Deep chain — DFT→X→Z→X→Z→DFT→IDFT→Z→IDFT */
    {
        uint32_t q_imm = quhit_init_basis(eng, 2);
        uint32_t q_lazy = quhit_init_basis(eng, 2);

        quhit_apply_dft(eng, q_imm);
        quhit_apply_x(eng, q_imm);
        quhit_apply_z(eng, q_imm);
        quhit_apply_x(eng, q_imm);
        quhit_apply_z(eng, q_imm);
        quhit_apply_dft(eng, q_imm);
        quhit_apply_idft(eng, q_imm);
        quhit_apply_z(eng, q_imm);
        quhit_apply_idft(eng, q_imm);

        lazy_record_dft(q_lazy);
        lazy_record_x(q_lazy);
        lazy_record_z(q_lazy);
        lazy_record_x(q_lazy);
        lazy_record_z(q_lazy);
        lazy_record_dft(q_lazy);
        lazy_record_idft(q_lazy);
        lazy_record_z(q_lazy);
        lazy_record_idft(q_lazy);
        CHECK(lazy_pending_depth(q_lazy) == 9, "9-gate deep chain");

        lazy_flush(eng, q_lazy);

        QuhitState *si = &eng->quhits[q_imm].state;
        QuhitState *sl = &eng->quhits[q_lazy].state;
        double err = 0;
        for (int k = 0; k < 6; k++)
            err += fabs(si->re[k] - sl->re[k]) + fabs(si->im[k] - sl->im[k]);
        CHECK(err < 1e-9, "9-gate chain: lazy matches immediate");
    }

    printf("\n  ── Diagonal fast path ──\n");

    /* Test: Multiple Z gates use diagonal fast path */
    {
        uint32_t q = quhit_init_plus(eng);
        lazy_record_z(q);
        lazy_record_z(q);
        LazyChain *lc = lazy_get(q);
        CHECK(lc->is_diagonal == 1, "Z→Z: diagonal fast path preserved");
        lazy_flush(eng, q);
    }

    /* Test: Z then DFT breaks diagonal */
    {
        uint32_t q = quhit_init(eng);
        lazy_record_z(q);
        LazyChain *lc = lazy_get(q);
        CHECK(lc->is_diagonal == 1, "Z alone: diagonal");
        lazy_record_dft(q);
        CHECK(lc->is_diagonal == 0, "Z→DFT: diagonal broken");
        lazy_flush(eng, q);
    }

    printf("\n  ── Flush all ──\n");

    /* Test: flush_all resolves all pending chains */
    {
        uint32_t a = quhit_init(eng);
        uint32_t b = quhit_init(eng);
        lazy_record_dft(a);
        lazy_record_z(b);
        CHECK(lazy_is_pending(a) == 1 && lazy_is_pending(b) == 1,
              "Two quhits pending");
        lazy_flush_all(eng);
        CHECK(lazy_is_pending(a) == 0 && lazy_is_pending(b) == 0,
              "flush_all: both resolved");
    }

    #undef CHECK

    printf("\n    Results: %d passed, %d failed\n\n", pass, fail);

    quhit_engine_destroy(eng);
    free(eng);
    return fail;
}
