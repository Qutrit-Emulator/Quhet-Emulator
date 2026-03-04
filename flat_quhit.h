/*
 * flat_quhit.h — The Flat Representation
 *
 *
 * When a quantum state reaches a structurally simple condition —
 * a single basis state, a phase-locked eigenstate, a confined subspace —
 * it can be "demoted" to the Flat, where gates are O(1) instead of O(D).
 *
 * Representations:
 *   FLAT_BASIS:    |k⟩ with phase. X=O(1), Z=O(1), CZ=O(1), measure=O(1).
 *   FLAT_SUBSPACE: few active states. All ops O(active_count).
 *   QUANTUM_FULL:  full TrialityQuhit. All ops O(D) or O(D²).
 *
 * Promotion: flat → full when a gate creates complexity (e.g. DFT on basis).
 * Demotion:  full → flat when result is structurally simple (e.g. after measure).
 */

#ifndef FLAT_QUHIT_H
#define FLAT_QUHIT_H

#include "quhit_triality.h"
#include <string.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════
 * REPRESENTATION TYPES
 * ═══════════════════════════════════════════════════════════════════════ */

typedef enum {
    FLAT_BASIS,       /* Single basis state |k⟩ with complex phase        */
    FLAT_SUBSPACE,    /* 2-3 active states, sparse representation         */
    QUANTUM_FULL      /* Full TrialityQuhit, all 6 amplitudes             */
} FlatRepr;

/* Precomputed ω^k for phase operations */
static const double FQ_W6_RE[6] = { 1.0,  0.5, -0.5, -1.0, -0.5,  0.5 };
static const double FQ_W6_IM[6] = { 0.0,  0.86602540378443864676,
                                     0.86602540378443864676,  0.0,
                                    -0.86602540378443864676, -0.86602540378443864676 };

/* ═══════════════════════════════════════════════════════════════════════
 * THE FLAT QUHIT
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    TrialityQuhit q;          /* Full state (valid when repr == QUANTUM_FULL) */
    FlatRepr      repr;

    /* Flat metadata — valid when repr == FLAT_BASIS */
    int    basis_index;        /* Which basis state: 0..5                    */
    double phase_re, phase_im; /* Accumulated phase: e^(iθ)                 */

    /* Benchmarking counters */
    uint64_t flat_ops;         /* Operations handled in flat representation  */
    uint64_t full_ops;         /* Operations handled in full representation  */
    uint64_t promotions;       /* Times promoted flat → full                 */
    uint64_t demotions;        /* Times demoted full → flat                  */
} FlatQuhit;

/* ═══════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 * ═══════════════════════════════════════════════════════════════════════ */

/* Initialize to |0⟩ — starts in the Flat */
static inline void fq_init(FlatQuhit *fq) {
    memset(fq, 0, sizeof(*fq));
    fq->repr = FLAT_BASIS;
    fq->basis_index = 0;
    fq->phase_re = 1.0;
    fq->phase_im = 0.0;
}

/* Initialize to |k⟩ — starts in the Flat */
static inline void fq_init_basis(FlatQuhit *fq, int k) {
    memset(fq, 0, sizeof(*fq));
    fq->repr = FLAT_BASIS;
    fq->basis_index = k % TRI_D;
    fq->phase_re = 1.0;
    fq->phase_im = 0.0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * PROMOTION — Flat → Full (the state enters the curved world)
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void fq_promote(FlatQuhit *fq) {
    if (fq->repr == QUANTUM_FULL) return;

    if (fq->repr == FLAT_BASIS) {
        /* Materialize |k⟩ with phase into full TrialityQuhit */
        triality_init_basis(&fq->q, fq->basis_index);
        /* Apply accumulated phase */
        fq->q.edge_re[fq->basis_index] = fq->phase_re;
        fq->q.edge_im[fq->basis_index] = fq->phase_im;
        fq->q.dirty |= DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED | DIRTY_EXOTIC;
    }

    fq->repr = QUANTUM_FULL;
    fq->promotions++;
}

/* ═══════════════════════════════════════════════════════════════════════
 * DEMOTION — Full → Flat (the state returns to the dead geometry)
 *
 * Checks if the full state has collapsed to a single basis state.
 * If so, extracts the index and phase and switches to FLAT_BASIS.
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void fq_try_demote(FlatQuhit *fq) {
    if (fq->repr != QUANTUM_FULL) return;

    /* Check if exactly one basis state is active */
    if (fq->q.active_count == 1) {
        int k = __builtin_ctz(fq->q.active_mask);
        double re = fq->q.edge_re[k];
        double im = fq->q.edge_im[k];
        double norm = sqrt(re*re + im*im);
        if (norm > 1e-15) {
            fq->repr = FLAT_BASIS;
            fq->basis_index = k;
            fq->phase_re = re / norm;
            fq->phase_im = im / norm;
            fq->demotions++;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * GATES — The Flat handles what it can; promotes when it can't
 * ═══════════════════════════════════════════════════════════════════════ */

/* Complex multiply helper: (a_re + i·a_im)(b_re + i·b_im) */
#define CMUL_RE(ar, ai, br, bi) ((ar)*(br) - (ai)*(bi))
#define CMUL_IM(ar, ai, br, bi) ((ar)*(bi) + (ai)*(br))

/* X gate: |k⟩ → |k+1 mod 6⟩ */
static inline void fq_x(FlatQuhit *fq) {
    if (fq->repr == FLAT_BASIS) {
        /* Just increment the index. O(1). The Flat handles this. */
        fq->basis_index = (fq->basis_index + 1) % TRI_D;
        fq->flat_ops++;
        return;
    }
    /* Full: delegate to triality */
    triality_x(&fq->q);
    fq_try_demote(fq);
    fq->full_ops++;
}

/* Shift gate: |k⟩ → |k+δ mod 6⟩ */
static inline void fq_shift(FlatQuhit *fq, int delta) {
    if (fq->repr == FLAT_BASIS) {
        fq->basis_index = ((fq->basis_index + delta) % TRI_D + TRI_D) % TRI_D;
        fq->flat_ops++;
        return;
    }
    triality_shift(&fq->q, delta);
    fq_try_demote(fq);
    fq->full_ops++;
}

/* Z gate: |k⟩ → ω^k|k⟩ */
static inline void fq_z(FlatQuhit *fq) {
    if (fq->repr == FLAT_BASIS) {
        /* Multiply phase by ω^k. O(1). */
        int k = fq->basis_index;
        double new_re = CMUL_RE(fq->phase_re, fq->phase_im, FQ_W6_RE[k], FQ_W6_IM[k]);
        double new_im = CMUL_IM(fq->phase_re, fq->phase_im, FQ_W6_RE[k], FQ_W6_IM[k]);
        fq->phase_re = new_re;
        fq->phase_im = new_im;
        fq->flat_ops++;
        return;
    }
    triality_z(&fq->q);
    fq_try_demote(fq);
    fq->full_ops++;
}

/* DFT: |k⟩ → superposition — ALWAYS promotes from basis */
static inline void fq_dft(FlatQuhit *fq) {
    if (fq->repr == FLAT_BASIS) {
        /* Must promote: DFT creates superposition */
        fq_promote(fq);
    }
    triality_dft(&fq->q);
    /* Try to demote in case of eigenstate */
    triality_update_mask(&fq->q);
    fq_try_demote(fq);
    fq->full_ops++;
}

/* IDFT */
static inline void fq_idft(FlatQuhit *fq) {
    if (fq->repr == FLAT_BASIS) {
        fq_promote(fq);
    }
    triality_idft(&fq->q);
    triality_update_mask(&fq->q);
    fq_try_demote(fq);
    fq->full_ops++;
}

/* CZ: controlled-Z between two FlatQuhits */
static inline void fq_cz(FlatQuhit *a, FlatQuhit *b) {
    if (a->repr == FLAT_BASIS && b->repr == FLAT_BASIS) {
        /* Both basis states: just apply ω^(j·k) phase to each. O(1). */
        int j = a->basis_index, k = b->basis_index;
        int idx = (j * k) % TRI_D;
        if (idx != 0) {
            double w_re = FQ_W6_RE[idx], w_im = FQ_W6_IM[idx];
            double new_a_re = CMUL_RE(a->phase_re, a->phase_im, w_re, w_im);
            double new_a_im = CMUL_IM(a->phase_re, a->phase_im, w_re, w_im);
            a->phase_re = new_a_re; a->phase_im = new_a_im;
            double new_b_re = CMUL_RE(b->phase_re, b->phase_im, w_re, w_im);
            double new_b_im = CMUL_IM(b->phase_re, b->phase_im, w_re, w_im);
            b->phase_re = new_b_re; b->phase_im = new_b_im;
        }
        a->flat_ops++; b->flat_ops++;
        return;
    }
    /* At least one is full: promote both and delegate */
    fq_promote(a);
    fq_promote(b);
    triality_cz(&a->q, &b->q);
    fq_try_demote(a);
    fq_try_demote(b);
    a->full_ops++; b->full_ops++;
}

/* ═══════════════════════════════════════════════════════════════════════
 * MEASUREMENT — Born sampling, then demote to Flat
 * ═══════════════════════════════════════════════════════════════════════ */

static inline int fq_measure(FlatQuhit *fq, int view, uint64_t *rng) {
    if (fq->repr == FLAT_BASIS && view == VIEW_EDGE) {
        /* Deterministic: always returns the basis index. O(1). */
        fq->flat_ops++;
        return fq->basis_index;
    }
    /* Full: Born sample, then demote */
    fq_promote(fq);
    int outcome = triality_measure(&fq->q, view, rng);
    triality_update_mask(&fq->q);
    fq_try_demote(fq);
    fq->full_ops++;
    return outcome;
}

/* ═══════════════════════════════════════════════════════════════════════
 * PROBABILITIES — No collapse
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void fq_probabilities(FlatQuhit *fq, int view, double *probs) {
    if (fq->repr == FLAT_BASIS && view == VIEW_EDGE) {
        /* Deterministic: all probability on one state */
        for (int k = 0; k < TRI_D; k++) probs[k] = 0.0;
        probs[fq->basis_index] = 1.0;
        fq->flat_ops++;
        return;
    }
    fq_promote(fq);
    triality_probabilities(&fq->q, view, probs);
    fq->full_ops++;
}

/* ═══════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 * ═══════════════════════════════════════════════════════════════════════ */

static inline const char *fq_repr_name(FlatRepr r) {
    switch (r) {
        case FLAT_BASIS:    return "FLAT_BASIS";
        case FLAT_SUBSPACE: return "FLAT_SUBSPACE";
        case QUANTUM_FULL:  return "QUANTUM_FULL";
        default:            return "UNKNOWN";
    }
}

static inline void fq_print_stats(const FlatQuhit *fq, const char *label) {
    printf("  %s: repr=%s  flat_ops=%lu  full_ops=%lu  "
           "promotions=%lu  demotions=%lu\n",
           label, fq_repr_name(fq->repr),
           (unsigned long)fq->flat_ops, (unsigned long)fq->full_ops,
           (unsigned long)fq->promotions, (unsigned long)fq->demotions);
}

#endif /* FLAT_QUHIT_H */
