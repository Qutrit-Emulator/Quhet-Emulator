/*
 * quhit_triality.h — The Triality Quhit
 *
 * A new quantum primitive based on the CMY geometric principle:
 * three mutually-defining views (Edge/Vertex/Diagonal) where each
 * view's structure IS the other views' structure in a different role.
 *
 *   Edge of A = Vertex of B = Diagonal of C  (cyclic)
 *
 * The triality quhit stores state in all three views with lazy
 * conversion. Gates automatically execute in their cheapest view:
 *   Phase gates   → Edge view     O(D)
 *   Shift gates   → Vertex view   O(D)
 *   Conjugate ops → Diagonal view O(D)
 *   General       → any view      O(D²)
 *
 * Average gate cost: O(12) instead of O(36). 3× free speedup.
 *
 */

#ifndef QUHIT_TRIALITY_H
#define QUHIT_TRIALITY_H

#include <stdint.h>

#define TRI_D 6

/* ═══════════════════════════════════════════════════════════════════════
 * VIEW IDENTIFIERS
 * ═══════════════════════════════════════════════════════════════════════ */

#define VIEW_EDGE     0   /* Computational basis — Yellow square */
#define VIEW_VERTEX   1   /* Fourier basis (DFT₆) — Cyan square */
#define VIEW_DIAGONAL 2   /* Conjugate Fourier (DFT₆²) — Magenta square */

/* Dirty bitmask: bit 0 = edge dirty, bit 1 = vertex dirty, bit 2 = diag dirty */
#define DIRTY_EDGE     0x1
#define DIRTY_VERTEX   0x2
#define DIRTY_DIAGONAL 0x4
#define DIRTY_ALL      0x7

/* ═══════════════════════════════════════════════════════════════════════
 * THE TRIALITY QUHIT
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    /* Three views of the same quantum state */
    double edge_re[TRI_D],   edge_im[TRI_D];     /* |ψ⟩ in computational basis */
    double vertex_re[TRI_D], vertex_im[TRI_D];    /* |ψ⟩ in Fourier basis */
    double diag_re[TRI_D],   diag_im[TRI_D];      /* |ψ⟩ in conjugate basis */

    uint8_t dirty;      /* Which views are stale */
    uint8_t primary;    /* Which view was last written (0/1/2) */
} TrialityQuhit;

/* ═══════════════════════════════════════════════════════════════════════
 * TRIALITY PAIR — Two entangled triality quhits
 * Each partner contributes a different view to the joint state.
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    double joint_re[TRI_D * TRI_D];
    double joint_im[TRI_D * TRI_D];
    int    view_a;  /* which view partner A contributes */
    int    view_b;  /* which view partner B contributes */
} TrialityPair;

/* ═══════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 * ═══════════════════════════════════════════════════════════════════════ */

/* Initialize to |0⟩ with all views clean */
void triality_init(TrialityQuhit *q);

/* Initialize to basis state |k⟩ */
void triality_init_basis(TrialityQuhit *q, int k);

/* Copy */
void triality_copy(TrialityQuhit *dst, const TrialityQuhit *src);

/* ═══════════════════════════════════════════════════════════════════════
 * VIEW MANAGEMENT — Lazy DFT₆ conversion
 * ═══════════════════════════════════════════════════════════════════════ */

/* Ensure a specific view is up-to-date (converts from primary if dirty) */
void triality_ensure_view(TrialityQuhit *q, int view);

/* Force recompute all views from primary */
void triality_sync_all(TrialityQuhit *q);

/* Get read-only access to a view (ensures it first) */
const double *triality_view_re(TrialityQuhit *q, int view);
const double *triality_view_im(TrialityQuhit *q, int view);

/* ═══════════════════════════════════════════════════════════════════════
 * OPTIMAL-VIEW GATES — O(D) when gate matches view
 * ═══════════════════════════════════════════════════════════════════════ */

/* Phase gate: |k⟩ → e^{iφₖ}|k⟩ — diagonal in EDGE view, O(D) */
void triality_phase(TrialityQuhit *q, const double *phi_re, const double *phi_im);

/* Single-phase: |k⟩ → e^{iφ}|k⟩, all others unchanged — O(1) */
void triality_phase_single(TrialityQuhit *q, int k, double phi_re, double phi_im);

/* Z gate: |k⟩ → ω^k |k⟩ — diagonal in EDGE view, O(D) */
void triality_z(TrialityQuhit *q);

/* Shift gate: |k⟩ → |k+δ mod D⟩ — diagonal in VERTEX view, O(D) */
void triality_shift(TrialityQuhit *q, int delta);

/* X gate: |k⟩ → |k+1 mod D⟩ — diagonal in VERTEX view, O(D) */
void triality_x(TrialityQuhit *q);

/* DFT₆: rotates edge→vertex→diagonal→edge — view rotation, O(D²) once */
void triality_dft(TrialityQuhit *q);

/* Inverse DFT₆ */
void triality_idft(TrialityQuhit *q);

/* General unitary in a specific view — O(D²) */
void triality_unitary(TrialityQuhit *q, int view,
                      const double *U_re, const double *U_im);

/* ═══════════════════════════════════════════════════════════════════════
 * CZ GATE — O(D) in edge view (diagonal)
 * ═══════════════════════════════════════════════════════════════════════ */

void triality_cz(TrialityQuhit *a, TrialityQuhit *b);

/* ═══════════════════════════════════════════════════════════════════════
 * MEASUREMENT — O(D) via cached view
 * ═══════════════════════════════════════════════════════════════════════ */

/* Measure in a specific view basis. Returns outcome 0..D-1. Collapses state. */
int triality_measure(TrialityQuhit *q, int view, uint64_t *rng_state);

/* Probability distribution in a view — O(D), no collapse */
void triality_probabilities(TrialityQuhit *q, int view, double *probs);

/* ═══════════════════════════════════════════════════════════════════════
 * TRIALITY ROTATION — The geometric heart
 * ═══════════════════════════════════════════════════════════════════════ */

/* Rotate the role assignment: Edge→Vertex→Diagonal→Edge
 * This is a FREE operation — it just relabels which view is which.
 * No amplitudes are modified. O(1). */
void triality_rotate(TrialityQuhit *q);

/* Inverse rotation: Diagonal→Vertex→Edge→Diagonal. O(1). */
void triality_rotate_inv(TrialityQuhit *q);

/* ═══════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 * ═══════════════════════════════════════════════════════════════════════ */

/* Print state in all three views */
void triality_print(TrialityQuhit *q, const char *label);

/* View conversion count (for benchmarking) */
typedef struct {
    uint64_t edge_to_vertex;
    uint64_t edge_to_diag;
    uint64_t vertex_to_edge;
    uint64_t vertex_to_diag;
    uint64_t diag_to_edge;
    uint64_t diag_to_vertex;
    uint64_t gates_edge;    /* gates executed in edge view */
    uint64_t gates_vertex;  /* gates executed in vertex view */
    uint64_t gates_diag;    /* gates executed in diagonal view */
    uint64_t rotations;     /* O(1) triality rotations */
} TrialityStats;

extern TrialityStats triality_stats;
void triality_stats_reset(void);
void triality_stats_print(void);

#endif /* QUHIT_TRIALITY_H */
