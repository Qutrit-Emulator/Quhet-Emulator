/*
 * flat_quhit.h — The Flat Representation
 *
 *
 * When a quantum state reaches a structurally simple condition —
 * a single basis state, a phase-locked eigenstate, a confined subspace —
 * it can be "demoted" to the Flat, where gates are O(1) instead of O(D).
 *
 * The state breathes: inhaling into quantum complexity when gates demand it,
 * exhaling back into the Flat when the result collapses. The engine spends
 * as little time as possible in the expensive curved form.
 *
 * Representations:
 *   FLAT_BASIS:    |k⟩ with phase. X=O(1), Z=O(1), CZ=O(1), measure=O(1).
 *   FLAT_SUBSPACE: 2-3 active states. All ops O(active_count²).
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

/* Max active states for subspace representation */
#define FQ_MAX_SUBSPACE 3

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

    /* Subspace metadata — valid when repr == FLAT_SUBSPACE */
    int    sub_count;                       /* Number of active states (2-3) */
    int    sub_idx[FQ_MAX_SUBSPACE];        /* Which basis indices are active*/
    double sub_re[FQ_MAX_SUBSPACE];         /* Real amplitudes              */
    double sub_im[FQ_MAX_SUBSPACE];         /* Imag amplitudes              */

    /* Benchmarking counters */
    uint64_t flat_ops;         /* Operations handled in flat representation  */
    uint64_t sub_ops;          /* Operations handled in subspace repr        */
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
    } else if (fq->repr == FLAT_SUBSPACE) {
        /* Materialize subspace into full TrialityQuhit */
        triality_init_basis(&fq->q, 0);
        for (int k = 0; k < TRI_D; k++) {
            fq->q.edge_re[k] = 0.0;
            fq->q.edge_im[k] = 0.0;
        }
        uint8_t mask = 0;
        for (int i = 0; i < fq->sub_count; i++) {
            fq->q.edge_re[fq->sub_idx[i]] = fq->sub_re[i];
            fq->q.edge_im[fq->sub_idx[i]] = fq->sub_im[i];
            mask |= (1 << fq->sub_idx[i]);
        }
        fq->q.active_mask = mask;
        fq->q.active_count = fq->sub_count;
        fq->q.dirty |= DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED | DIRTY_EXOTIC;
    }

    fq->repr = QUANTUM_FULL;
    fq->promotions++;
}

/* ═══════════════════════════════════════════════════════════════════════
 * DEMOTION — Full → Flat (the state returns to the dead geometry)
 *
 * Checks if the full state has collapsed to a structurally simple form:
 *   active_count == 1 → FLAT_BASIS
 *   active_count == 2 or 3 → FLAT_SUBSPACE
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void fq_try_demote(FlatQuhit *fq) {
    if (fq->repr != QUANTUM_FULL) return;

    if (fq->q.active_count == 1) {
        /* Single basis state → FLAT_BASIS */
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
    } else if (fq->q.active_count <= FQ_MAX_SUBSPACE) {
        /* 2-3 active states → FLAT_SUBSPACE */
        fq->repr = FLAT_SUBSPACE;
        fq->sub_count = 0;
        uint8_t m = fq->q.active_mask;
        for (int k = 0; k < TRI_D && fq->sub_count < FQ_MAX_SUBSPACE; k++) {
            if (m & (1 << k)) {
                fq->sub_idx[fq->sub_count] = k;
                fq->sub_re[fq->sub_count] = fq->q.edge_re[k];
                fq->sub_im[fq->sub_count] = fq->q.edge_im[k];
                fq->sub_count++;
            }
        }
        fq->demotions++;
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
    if (fq->repr == FLAT_SUBSPACE) {
        /* Shift all subspace indices. O(sub_count). */
        for (int i = 0; i < fq->sub_count; i++)
            fq->sub_idx[i] = (fq->sub_idx[i] + 1) % TRI_D;
        fq->sub_ops++;
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
    if (fq->repr == FLAT_SUBSPACE) {
        for (int i = 0; i < fq->sub_count; i++)
            fq->sub_idx[i] = ((fq->sub_idx[i] + delta) % TRI_D + TRI_D) % TRI_D;
        fq->sub_ops++;
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
    if (fq->repr == FLAT_SUBSPACE) {
        /* Apply ω^k to each active state. O(sub_count). */
        for (int i = 0; i < fq->sub_count; i++) {
            int k = fq->sub_idx[i];
            double re = fq->sub_re[i], im = fq->sub_im[i];
            fq->sub_re[i] = CMUL_RE(re, im, FQ_W6_RE[k], FQ_W6_IM[k]);
            fq->sub_im[i] = CMUL_IM(re, im, FQ_W6_RE[k], FQ_W6_IM[k]);
        }
        fq->sub_ops++;
        return;
    }
    triality_z(&fq->q);
    fq_try_demote(fq);
    fq->full_ops++;
}

/* DFT: |k⟩ → superposition — ALWAYS promotes from basis/subspace */
static inline void fq_dft(FlatQuhit *fq) {
    if (fq->repr != QUANTUM_FULL) {
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
    if (fq->repr != QUANTUM_FULL) {
        fq_promote(fq);
    }
    triality_idft(&fq->q);
    triality_update_mask(&fq->q);
    fq_try_demote(fq);
    fq->full_ops++;
}

/* CZ: controlled-Z between two FlatQuhits
 * CZ|j⟩|k⟩ = ω^(j·k)|j⟩|k⟩ where ω = e^(2πi/6)
 * Direct computation for all flat/subspace combos — no promotion. */
static inline void fq_cz(FlatQuhit *a, FlatQuhit *b) {
    if (a->repr == FLAT_BASIS && b->repr == FLAT_BASIS) {
        /* Basis × Basis: ω^(j·k) phase on a only. O(1).
         * The joint phase lives on one side — b is untouched. */
        int j = a->basis_index, k = b->basis_index;
        int idx = (j * k) % TRI_D;
        if (idx != 0) {
            double w_re = FQ_W6_RE[idx], w_im = FQ_W6_IM[idx];
            double new_a_re = CMUL_RE(a->phase_re, a->phase_im, w_re, w_im);
            double new_a_im = CMUL_IM(a->phase_re, a->phase_im, w_re, w_im);
            a->phase_re = new_a_re; a->phase_im = new_a_im;
        }
        a->flat_ops++; b->flat_ops++;
        return;
    }
    if (a->repr == FLAT_BASIS && b->repr == FLAT_SUBSPACE) {
        /* Basis × Subspace: apply ω^(j·k) to each subspace entry of b.
         * j is fixed (a's basis), k varies over b's active states.
         * O(sub_count) — at most 3 CMULs. No promotion needed. */
        int j = a->basis_index;
        if (j != 0) { /* j=0 → ω^0 = 1 → identity */
            for (int i = 0; i < b->sub_count; i++) {
                int k = b->sub_idx[i];
                int idx = (j * k) % TRI_D;
                if (idx != 0) {
                    double w_re = FQ_W6_RE[idx], w_im = FQ_W6_IM[idx];
                    double re = b->sub_re[i], im = b->sub_im[i];
                    b->sub_re[i] = CMUL_RE(re, im, w_re, w_im);
                    b->sub_im[i] = CMUL_IM(re, im, w_re, w_im);
                }
            }
            /* a gets phase from overlap: ω^(j·⟨k⟩) averaged — but CZ is diagonal,
             * so a's phase picks up the weighted sum. For basis state a, the
             * CZ action is: a unchanged, b gets ω^(j·k) per entry. */
        }
        a->flat_ops++; b->sub_ops++;
        return;
    }
    if (a->repr == FLAT_SUBSPACE && b->repr == FLAT_BASIS) {
        /* Subspace × Basis: mirror of above. */
        int k = b->basis_index;
        if (k != 0) {
            for (int i = 0; i < a->sub_count; i++) {
                int j = a->sub_idx[i];
                int idx = (j * k) % TRI_D;
                if (idx != 0) {
                    double w_re = FQ_W6_RE[idx], w_im = FQ_W6_IM[idx];
                    double re = a->sub_re[i], im = a->sub_im[i];
                    a->sub_re[i] = CMUL_RE(re, im, w_re, w_im);
                    a->sub_im[i] = CMUL_IM(re, im, w_re, w_im);
                }
            }
        }
        a->sub_ops++; b->flat_ops++;
        return;
    }
    if (a->repr == FLAT_SUBSPACE && b->repr == FLAT_SUBSPACE) {
        /* Subspace × Subspace: CZ is diagonal in the computational basis.
         * |j⟩|k⟩ → ω^(j·k)|j⟩|k⟩. Since CZ doesn't change the basis states,
         * only phases change. For each pair (j,k) of active indices,
         * multiply both amplitudes by ω^(j·k).
         * BUT: CZ acts on the JOINT state. In the product representation,
         * we need to be careful. For separable states |ψ_a⟩|ψ_b⟩:
         *   α_j β_k → ω^(jk) α_j β_k
         * This can't be factored into independent a,b updates in general.
         * However, for the special case where one side has only 1 entry,
         * it reduces to the basis×subspace case above.
         * For general subspace×subspace: promote both. O(9) CMULs in triality_cz
         * is still much cheaper than full 6×6 since active_mask limits iteration. */
        fq_promote(a);
        fq_promote(b);
        triality_cz(&a->q, &b->q);
        triality_update_mask(&a->q);
        triality_update_mask(&b->q);
        fq_try_demote(a);
        fq_try_demote(b);
        a->sub_ops++; b->sub_ops++;
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
    if (fq->repr == FLAT_SUBSPACE && view == VIEW_EDGE) {
        /* Sparse probabilities. O(sub_count). */
        for (int k = 0; k < TRI_D; k++) probs[k] = 0.0;
        for (int i = 0; i < fq->sub_count; i++) {
            double re = fq->sub_re[i], im = fq->sub_im[i];
            probs[fq->sub_idx[i]] = re*re + im*im;
        }
        fq->sub_ops++;
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
    printf("  %s: repr=%s  flat=%lu  sub=%lu  full=%lu  "
           "promote=%lu  demote=%lu\n",
           label, fq_repr_name(fq->repr),
           (unsigned long)fq->flat_ops, (unsigned long)fq->sub_ops,
           (unsigned long)fq->full_ops,
           (unsigned long)fq->promotions, (unsigned long)fq->demotions);
}

#endif /* FLAT_QUHIT_H */
