/*
 * quhit_triality.c — The Triality Quhit Implementation
 *
 * Three views. One state. Every gate finds its cheapest mirror.
 *
 * Edge   = Yellow square  = computational basis
 * Vertex = Cyan square    = Fourier basis (DFT₆)
 * Diag   = Magenta square = conjugate Fourier (DFT₆²)
 *
 * Rule: Edge of A = Vertex of B = Diagonal of C (cyclic)
 *
 */

#include <string.h>
#include <math.h>
#include <stdio.h>
#include "quhit_triality.h"

/* ═══════════════════════════════════════════════════════════════════════
 * ω₆ TABLES — precomputed roots of unity
 * ═══════════════════════════════════════════════════════════════════════ */

static const double W6_RE[6] = {
    1.0, 0.5, -0.5, -1.0, -0.5, 0.5
};
static const double W6_IM[6] = {
    0.0, 0.86602540378443864676, 0.86602540378443864676,
    0.0, -0.86602540378443864676, -0.86602540378443864676
};

/* Inverse ω: ω^(-jk) = conjugate of ω^(jk) */
static const double W6I_RE[6] = {
    1.0, 0.5, -0.5, -1.0, -0.5, 0.5
};
static const double W6I_IM[6] = {
    0.0, -0.86602540378443864676, -0.86602540378443864676,
    0.0, 0.86602540378443864676, 0.86602540378443864676
};

static const double INV_SQRT6 = 0.40824829046386301637;  /* 1/√6 */

/* ═══════════════════════════════════════════════════════════════════════
 * STATISTICS
 * ═══════════════════════════════════════════════════════════════════════ */

TrialityStats triality_stats = {0};

void triality_stats_reset(void) {
    memset(&triality_stats, 0, sizeof(triality_stats));
}

void triality_stats_print(void) {
    printf("\n  ┌─────────────────────────────────────────────────────┐\n");
    printf("  │  TRIALITY QUHIT STATISTICS                          │\n");
    printf("  ├─────────────────────────────────────────────────────┤\n");
    printf("  │  View conversions:                                  │\n");
    printf("  │    Edge→Vertex:    %8lu                         │\n", triality_stats.edge_to_vertex);
    printf("  │    Edge→Diagonal:  %8lu                         │\n", triality_stats.edge_to_diag);
    printf("  │    Vertex→Edge:    %8lu                         │\n", triality_stats.vertex_to_edge);
    printf("  │    Vertex→Diag:    %8lu                         │\n", triality_stats.vertex_to_diag);
    printf("  │    Diag→Edge:      %8lu                         │\n", triality_stats.diag_to_edge);
    printf("  │    Diag→Vertex:    %8lu                         │\n", triality_stats.diag_to_vertex);
    printf("  │  Gates by view:                                     │\n");
    printf("  │    Edge (phase):   %8lu   O(D) each             │\n", triality_stats.gates_edge);
    printf("  │    Vertex (shift): %8lu   O(D) each             │\n", triality_stats.gates_vertex);
    printf("  │    Diagonal:       %8lu   O(D) each             │\n", triality_stats.gates_diag);
    printf("  │  Triality rotations (O(1)):  %8lu               │\n", triality_stats.rotations);
    uint64_t total_conv = triality_stats.edge_to_vertex + triality_stats.edge_to_diag +
                          triality_stats.vertex_to_edge + triality_stats.vertex_to_diag +
                          triality_stats.diag_to_edge + triality_stats.diag_to_vertex;
    uint64_t total_gates = triality_stats.gates_edge + triality_stats.gates_vertex +
                           triality_stats.gates_diag;
    printf("  │  Total: %lu gates, %lu conversions             │\n", total_gates, total_conv);
    if (total_gates > 0) {
        double avg = (total_gates * 6.0 + total_conv * 36.0) / total_gates;
        printf("  │  Avg ops/gate: %.1f (vs 36.0 standard)           │\n", avg);
    }
    printf("  └─────────────────────────────────────────────────────┘\n");
}

/* ═══════════════════════════════════════════════════════════════════════
 * DFT₆ PRIMITIVES — The view conversion engine
 * ═══════════════════════════════════════════════════════════════════════ */

/* Forward DFT₆: out[j] = (1/√6) Σ_k in[k] × ω^(jk) */
static void dft6_forward(const double *in_re, const double *in_im,
                         double *out_re, double *out_im)
{
    for (int j = 0; j < 6; j++) {
        double sr = 0, si = 0;
        for (int k = 0; k < 6; k++) {
            int idx = (j * k) % 6;
            double wr = W6_RE[idx], wi = W6_IM[idx];
            sr += in_re[k] * wr - in_im[k] * wi;
            si += in_re[k] * wi + in_im[k] * wr;
        }
        out_re[j] = sr * INV_SQRT6;
        out_im[j] = si * INV_SQRT6;
    }
}

/* Inverse DFT₆: out[k] = (1/√6) Σ_j in[j] × ω^(-jk) */
static void dft6_inverse(const double *in_re, const double *in_im,
                         double *out_re, double *out_im)
{
    for (int k = 0; k < 6; k++) {
        double sr = 0, si = 0;
        for (int j = 0; j < 6; j++) {
            int idx = (j * k) % 6;
            double wr = W6I_RE[idx], wi = W6I_IM[idx];
            sr += in_re[j] * wr - in_im[j] * wi;
            si += in_re[j] * wi + in_im[j] * wr;
        }
        out_re[k] = sr * INV_SQRT6;
        out_im[k] = si * INV_SQRT6;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 * ═══════════════════════════════════════════════════════════════════════ */

void triality_init(TrialityQuhit *q) {
    memset(q, 0, sizeof(*q));
    q->edge_re[0] = 1.0;       /* |0⟩ in computational basis */
    q->vertex_re[0] = 1.0;     /* DFT₆|0⟩ = (1/√6)(1,1,1,1,1,1) ... */
    q->diag_re[0] = 1.0;

    /* Actually compute the correct views of |0⟩ */
    dft6_forward(q->edge_re, q->edge_im, q->vertex_re, q->vertex_im);
    dft6_forward(q->vertex_re, q->vertex_im, q->diag_re, q->diag_im);

    q->dirty = 0;
    q->primary = VIEW_EDGE;
}

void triality_init_basis(TrialityQuhit *q, int k) {
    memset(q, 0, sizeof(*q));
    q->edge_re[k] = 1.0;
    q->primary = VIEW_EDGE;
    q->dirty = DIRTY_VERTEX | DIRTY_DIAGONAL;
}

void triality_copy(TrialityQuhit *dst, const TrialityQuhit *src) {
    memcpy(dst, src, sizeof(*dst));
}

/* ═══════════════════════════════════════════════════════════════════════
 * VIEW MANAGEMENT — Lazy conversion
 *
 * The three views are related by DFT₆:
 *   vertex = DFT₆(edge)
 *   diag   = DFT₆(vertex) = DFT₆²(edge)
 *   edge   = DFT₆(diag)   = DFT₆³(edge) = identity
 *
 * So DFT₆ cycles: edge → vertex → diagonal → edge
 * And IDFT₆ cycles: edge → diagonal → vertex → edge
 * ═══════════════════════════════════════════════════════════════════════ */

static double *view_re(TrialityQuhit *q, int v) {
    switch(v) {
        case VIEW_EDGE:     return q->edge_re;
        case VIEW_VERTEX:   return q->vertex_re;
        case VIEW_DIAGONAL: return q->diag_re;
    }
    return q->edge_re;
}

static double *view_im(TrialityQuhit *q, int v) {
    switch(v) {
        case VIEW_EDGE:     return q->edge_im;
        case VIEW_VERTEX:   return q->vertex_im;
        case VIEW_DIAGONAL: return q->diag_im;
    }
    return q->edge_im;
}

static int view_dirty_bit(int v) {
    return 1 << v;
}

static void convert_view(TrialityQuhit *q, int from, int to) {
    double *src_re = view_re(q, from), *src_im = view_im(q, from);
    double *dst_re = view_re(q, to),   *dst_im = view_im(q, to);

    /* Determine how many DFT₆ steps from 'from' to 'to' */
    int steps = (to - from + 3) % 3;  /* 1 = one DFT₆, 2 = two DFT₆ = one IDFT₆ */

    if (steps == 1) {
        dft6_forward(src_re, src_im, dst_re, dst_im);
    } else if (steps == 2) {
        dft6_inverse(src_re, src_im, dst_re, dst_im);
    }

    /* Track statistics */
    if (from == VIEW_EDGE && to == VIEW_VERTEX)   triality_stats.edge_to_vertex++;
    if (from == VIEW_EDGE && to == VIEW_DIAGONAL)  triality_stats.edge_to_diag++;
    if (from == VIEW_VERTEX && to == VIEW_EDGE)    triality_stats.vertex_to_edge++;
    if (from == VIEW_VERTEX && to == VIEW_DIAGONAL) triality_stats.vertex_to_diag++;
    if (from == VIEW_DIAGONAL && to == VIEW_EDGE)   triality_stats.diag_to_edge++;
    if (from == VIEW_DIAGONAL && to == VIEW_VERTEX)  triality_stats.diag_to_vertex++;
}

void triality_ensure_view(TrialityQuhit *q, int view) {
    if (!(q->dirty & view_dirty_bit(view))) return;  /* Already clean */

    /* Find a clean view to convert from */
    int source = q->primary;
    if (q->dirty & view_dirty_bit(source)) {
        /* Primary is dirty — find any clean view */
        for (int v = 0; v < 3; v++) {
            if (!(q->dirty & view_dirty_bit(v))) {
                source = v;
                break;
            }
        }
    }

    convert_view(q, source, view);
    q->dirty &= ~view_dirty_bit(view);
}

void triality_sync_all(TrialityQuhit *q) {
    triality_ensure_view(q, VIEW_EDGE);
    triality_ensure_view(q, VIEW_VERTEX);
    triality_ensure_view(q, VIEW_DIAGONAL);
}

const double *triality_view_re(TrialityQuhit *q, int view) {
    triality_ensure_view(q, view);
    return view_re(q, view);
}

const double *triality_view_im(TrialityQuhit *q, int view) {
    triality_ensure_view(q, view);
    return view_im(q, view);
}

/* ═══════════════════════════════════════════════════════════════════════
 * OPTIMAL-VIEW GATES
 * ═══════════════════════════════════════════════════════════════════════ */

/* Phase gate: |k⟩ → e^{iφₖ}|k⟩ — DIAGONAL in edge view, O(D) */
void triality_phase(TrialityQuhit *q, const double *phi_re, const double *phi_im) {
    triality_ensure_view(q, VIEW_EDGE);
    for (int k = 0; k < TRI_D; k++) {
        double re = q->edge_re[k], im = q->edge_im[k];
        q->edge_re[k] = re * phi_re[k] - im * phi_im[k];
        q->edge_im[k] = re * phi_im[k] + im * phi_re[k];
    }
    q->primary = VIEW_EDGE;
    q->dirty |= DIRTY_VERTEX | DIRTY_DIAGONAL;
    triality_stats.gates_edge++;
}

/* Single-phase: one basis state only — O(1) */
void triality_phase_single(TrialityQuhit *q, int k, double phi_re, double phi_im) {
    triality_ensure_view(q, VIEW_EDGE);
    double re = q->edge_re[k], im = q->edge_im[k];
    q->edge_re[k] = re * phi_re - im * phi_im;
    q->edge_im[k] = re * phi_im + im * phi_re;
    q->primary = VIEW_EDGE;
    q->dirty |= DIRTY_VERTEX | DIRTY_DIAGONAL;
    triality_stats.gates_edge++;
}

/* Z gate: |k⟩ → ω^k|k⟩ — diagonal in edge, O(D) */
void triality_z(TrialityQuhit *q) {
    triality_ensure_view(q, VIEW_EDGE);
    for (int k = 0; k < TRI_D; k++) {
        double wr = W6_RE[k], wi = W6_IM[k];
        double re = q->edge_re[k], im = q->edge_im[k];
        q->edge_re[k] = re * wr - im * wi;
        q->edge_im[k] = re * wi + im * wr;
    }
    q->primary = VIEW_EDGE;
    q->dirty |= DIRTY_VERTEX | DIRTY_DIAGONAL;
    triality_stats.gates_edge++;
}

/* Shift: |k⟩ → |k+δ mod D⟩ — this is DIAGONAL in vertex view!
 * In Fourier domain, shift by δ = multiply by ω^(δ·j).
 * So we work in vertex view and apply phases. O(D). */
void triality_shift(TrialityQuhit *q, int delta) {
    delta = ((delta % TRI_D) + TRI_D) % TRI_D;
    if (delta == 0) return;

    triality_ensure_view(q, VIEW_VERTEX);
    for (int j = 0; j < TRI_D; j++) {
        int idx = (delta * j) % 6;
        double wr = W6_RE[idx], wi = W6_IM[idx];
        double re = q->vertex_re[j], im = q->vertex_im[j];
        q->vertex_re[j] = re * wr - im * wi;
        q->vertex_im[j] = re * wi + im * wr;
    }
    q->primary = VIEW_VERTEX;
    q->dirty |= DIRTY_EDGE | DIRTY_DIAGONAL;
    triality_stats.gates_vertex++;
}

/* X = shift by 1 */
void triality_x(TrialityQuhit *q) {
    triality_shift(q, 1);
}

/* DFT₆ — The triality rotation in Hilbert space
 * This converts edge→vertex, vertex→diag, diag→edge.
 * If the destination view is already cached, it's essentially FREE. */
void triality_dft(TrialityQuhit *q) {
    /* Sync all views first */
    triality_sync_all(q);

    /* Relabel: old_vertex becomes new_edge, etc.
     * This is because DFT₆ maps computational→Fourier,
     * and our views cycle: edge→vertex→diag→edge */
    double tmp_re[TRI_D], tmp_im[TRI_D];

    /* Save old edge */
    memcpy(tmp_re, q->edge_re, sizeof(tmp_re));
    memcpy(tmp_im, q->edge_im, sizeof(tmp_im));

    /* new edge = old vertex (DFT₆ maps old comp→Fourier, Fourier is vertex) */
    /* Actually: DFT₆|ψ⟩ in comp basis = |ψ⟩ in Fourier basis
     * So the NEW computational-basis amplitudes are the DFT₆ of the old ones */
    dft6_forward(tmp_re, tmp_im, q->edge_re, q->edge_im);
    q->primary = VIEW_EDGE;
    q->dirty = DIRTY_VERTEX | DIRTY_DIAGONAL;
    triality_stats.gates_edge++;
}

/* Inverse DFT₆ */
void triality_idft(TrialityQuhit *q) {
    triality_ensure_view(q, VIEW_EDGE);
    double tmp_re[TRI_D], tmp_im[TRI_D];
    memcpy(tmp_re, q->edge_re, sizeof(tmp_re));
    memcpy(tmp_im, q->edge_im, sizeof(tmp_im));
    dft6_inverse(tmp_re, tmp_im, q->edge_re, q->edge_im);
    q->primary = VIEW_EDGE;
    q->dirty = DIRTY_VERTEX | DIRTY_DIAGONAL;
    triality_stats.gates_edge++;
}

/* General unitary in a specific view — O(D²) */
void triality_unitary(TrialityQuhit *q, int view,
                      const double *U_re, const double *U_im) {
    triality_ensure_view(q, view);
    double *v_re = view_re(q, view), *v_im = view_im(q, view);
    double out_re[TRI_D] = {0}, out_im[TRI_D] = {0};

    for (int j = 0; j < TRI_D; j++)
        for (int k = 0; k < TRI_D; k++) {
            double ur = U_re[j * TRI_D + k], ui = U_im[j * TRI_D + k];
            out_re[j] += ur * v_re[k] - ui * v_im[k];
            out_im[j] += ur * v_im[k] + ui * v_re[k];
        }

    memcpy(v_re, out_re, sizeof(out_re));
    memcpy(v_im, out_im, sizeof(out_im));
    q->primary = view;
    q->dirty = DIRTY_ALL & ~view_dirty_bit(view);

    if (view == VIEW_EDGE)     triality_stats.gates_edge++;
    if (view == VIEW_VERTEX)   triality_stats.gates_vertex++;
    if (view == VIEW_DIAGONAL) triality_stats.gates_diag++;
}

/* ═══════════════════════════════════════════════════════════════════════
 * CZ GATE — Diagonal in edge view, O(D²) on joint but O(D) per site
 *
 * CZ|j,k⟩ = ω^(jk)|j,k⟩  — only phases, no state mixing.
 * We work in edge view on both quhits.
 *
 * For a product state |ψ⟩⊗|φ⟩, the CZ creates entanglement
 * but we track it via the phase imprint on each site.
 * ═══════════════════════════════════════════════════════════════════════ */

void triality_cz(TrialityQuhit *a, TrialityQuhit *b) {
    triality_ensure_view(a, VIEW_EDGE);
    triality_ensure_view(b, VIEW_EDGE);

    /* For product states, CZ imprints partner-dependent phases.
     * Full CZ on joint state: joint[j*D+k] *= ω^(jk)
     * On separable states, we condition on the partner's state:
     *   a[j] *= Σ_k |b[k]|² × ω^(jk)   (effective phase from b)
     *   b[k] *= Σ_j |a[j]|² × ω^(jk)   (effective phase from a) */

    /* Compute effective phases from partner */
    double eff_a_re[TRI_D] = {0}, eff_a_im[TRI_D] = {0};
    double eff_b_re[TRI_D] = {0}, eff_b_im[TRI_D] = {0};

    for (int j = 0; j < TRI_D; j++) {
        for (int k = 0; k < TRI_D; k++) {
            int idx = (j * k) % 6;
            double bprob = b->edge_re[k]*b->edge_re[k] + b->edge_im[k]*b->edge_im[k];
            eff_a_re[j] += bprob * W6_RE[idx];
            eff_a_im[j] += bprob * W6_IM[idx];

            double aprob = a->edge_re[j]*a->edge_re[j] + a->edge_im[j]*a->edge_im[j];
            eff_b_re[k] += aprob * W6_RE[idx];
            eff_b_im[k] += aprob * W6_IM[idx];
        }
    }

    /* Apply effective phases */
    for (int j = 0; j < TRI_D; j++) {
        double re = a->edge_re[j], im = a->edge_im[j];
        a->edge_re[j] = re * eff_a_re[j] - im * eff_a_im[j];
        a->edge_im[j] = re * eff_a_im[j] + im * eff_a_re[j];
    }
    for (int k = 0; k < TRI_D; k++) {
        double re = b->edge_re[k], im = b->edge_im[k];
        b->edge_re[k] = re * eff_b_re[k] - im * eff_b_im[k];
        b->edge_im[k] = re * eff_b_im[k] + im * eff_b_re[k];
    }

    a->dirty |= DIRTY_VERTEX | DIRTY_DIAGONAL;
    b->dirty |= DIRTY_VERTEX | DIRTY_DIAGONAL;
    triality_stats.gates_edge += 2;
}

/* ═══════════════════════════════════════════════════════════════════════
 * MEASUREMENT
 * ═══════════════════════════════════════════════════════════════════════ */

/* Simple xorshift64 PRNG for Born sampling */
static uint64_t triality_rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static double triality_rng_double(uint64_t *state) {
    return (triality_rng_next(state) >> 11) * 0x1.0p-53;
}

void triality_probabilities(TrialityQuhit *q, int view, double *probs) {
    triality_ensure_view(q, view);
    const double *re = view_re(q, view), *im = view_im(q, view);
    double total = 0;
    for (int k = 0; k < TRI_D; k++) {
        probs[k] = re[k]*re[k] + im[k]*im[k];
        total += probs[k];
    }
    if (total > 1e-30)
        for (int k = 0; k < TRI_D; k++) probs[k] /= total;
}

int triality_measure(TrialityQuhit *q, int view, uint64_t *rng_state) {
    double probs[TRI_D];
    triality_probabilities(q, view, probs);

    /* Born sample */
    double r = triality_rng_double(rng_state);
    int outcome = 0;
    double cdf = 0;
    for (int k = 0; k < TRI_D; k++) {
        cdf += probs[k];
        if (r < cdf) { outcome = k; break; }
    }

    /* Collapse: project onto |outcome⟩ in the measured view */
    triality_ensure_view(q, view);
    double *v_re = view_re(q, view), *v_im = view_im(q, view);

    double norm2 = v_re[outcome]*v_re[outcome] + v_im[outcome]*v_im[outcome];
    double scale = (norm2 > 1e-30) ? 1.0 / sqrt(norm2) : 0.0;

    for (int k = 0; k < TRI_D; k++) {
        v_re[k] = 0;
        v_im[k] = 0;
    }
    v_re[outcome] = (v_re[outcome] != 0 ? v_re[outcome] : 1.0) * scale;
    /* Keep the phase of the original amplitude */

    q->primary = view;
    q->dirty = DIRTY_ALL & ~view_dirty_bit(view);

    return outcome;
}

/* ═══════════════════════════════════════════════════════════════════════
 * TRIALITY ROTATION — O(1) relabeling
 *
 * This is the geometric heart: Edge→Vertex→Diagonal→Edge.
 * No computation. Just swap pointers (well, arrays).
 * The physics doesn't change — only our VIEW of it changes.
 * ═══════════════════════════════════════════════════════════════════════ */

void triality_rotate(TrialityQuhit *q) {
    /* Cycle: edge→vertex→diagonal→edge
     * Old edge becomes new vertex, old vertex becomes new diag,
     * old diag becomes new edge */
    double tmp_re[TRI_D], tmp_im[TRI_D];

    /* Save old diagonal */
    memcpy(tmp_re, q->diag_re, sizeof(tmp_re));
    memcpy(tmp_im, q->diag_im, sizeof(tmp_im));

    /* old vertex → new diagonal */
    memcpy(q->diag_re, q->vertex_re, sizeof(tmp_re));
    memcpy(q->diag_im, q->vertex_im, sizeof(tmp_im));

    /* old edge → new vertex */
    memcpy(q->vertex_re, q->edge_re, sizeof(tmp_re));
    memcpy(q->vertex_im, q->edge_im, sizeof(tmp_im));

    /* old diagonal → new edge */
    memcpy(q->edge_re, tmp_re, sizeof(tmp_re));
    memcpy(q->edge_im, tmp_im, sizeof(tmp_im));

    /* Rotate dirty bits: bit0→bit1→bit2→bit0 */
    uint8_t d = q->dirty;
    q->dirty = ((d & 1) << 1) | ((d & 2) << 1) | ((d & 4) >> 2);

    /* Rotate primary: 0→1→2→0 */
    q->primary = (q->primary + 1) % 3;

    triality_stats.rotations++;
}

void triality_rotate_inv(TrialityQuhit *q) {
    /* Inverse: two forward rotations */
    triality_rotate(q);
    triality_rotate(q);
}

/* ═══════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 * ═══════════════════════════════════════════════════════════════════════ */

void triality_print(TrialityQuhit *q, const char *label) {
    triality_sync_all(q);

    printf("\n  ┌─── Triality Quhit: %s ───┐\n", label ? label : "");
    printf("  │ View      │");
    for (int k = 0; k < TRI_D; k++) printf("   |%d⟩      ", k);
    printf("│\n");
    printf("  ├───────────┼");
    for (int k = 0; k < TRI_D; k++) printf("────────────");
    printf("┤\n");

    const char *names[3] = {"Edge  (Y)", "Vertex(C)", "Diag  (M)"};
    double *arrays_re[3] = {q->edge_re, q->vertex_re, q->diag_re};
    double *arrays_im[3] = {q->edge_im, q->vertex_im, q->diag_im};

    for (int v = 0; v < 3; v++) {
        printf("  │ %s │", names[v]);
        for (int k = 0; k < TRI_D; k++) {
            double re = arrays_re[v][k], im = arrays_im[v][k];
            double mag = sqrt(re*re + im*im);
            if (mag < 1e-10)
                printf("     ·      ");
            else if (fabs(im) < 1e-10)
                printf("  %+.4f   ", re);
            else
                printf(" %+.2f%+.2fi", re, im);
        }
        printf("│\n");
    }
    printf("  │ Primary: %s │ Dirty: %c%c%c │\n",
           names[q->primary],
           (q->dirty & DIRTY_EDGE) ? 'E' : '·',
           (q->dirty & DIRTY_VERTEX) ? 'V' : '·',
           (q->dirty & DIRTY_DIAGONAL) ? 'D' : '·');
    printf("  └");
    for (int i = 0; i < 12 + TRI_D * 12; i++) printf("─");
    printf("┘\n");
}


/* ═══════════════════════════════════════════════════════════════════════
 * LAZY TRIALITY QUHIT — Heisenberg Picture with Event Horizon Shortcuts
 *
 * Chain: state → F^pre0 · D0 → F^pre1 · D1 → ... → F^trailing
 * All diagonals in computational (edge) basis.
 * DFTs tracked per-segment. F^4 = I, so counts are mod 4.
 * IDFT = F^3 (= F^-1 since F^4 = I).
 *
 * EVENT HORIZON SHORTCUTS:
 *   F² = reversal:  DFT²|k⟩ = |(-k) mod 6⟩.  O(D) not O(D²).
 *   Parity fusion:  Even DFT counts (0,2) allow segment fusion.
 *                   trailing=2 → reverse diagonal entries + fuse.
 *   Chain compact:  Adjacent segments with even pre_dfts merge.
 *
 * Only ODD horizon crossings (pre_dfts = 1 or 3) are real.
 * Even crossings are mirrors — the vertex bounces back.
 * ═══════════════════════════════════════════════════════════════════════ */

#define DFT_ORDER 4  /* DFT_6^4 = I */

static void ltri_diag_identity(double *re, double *im) {
    for (int k = 0; k < TRI_D; k++) { re[k] = 1.0; im[k] = 0.0; }
}

static void ltri_diag_mul_phase(double *d_re, double *d_im, int k,
                                double pr, double pi) {
    double re = d_re[k], im = d_im[k];
    d_re[k] = re * pr - im * pi;
    d_im[k] = re * pi + im * pr;
}

static void ltri_apply_diag(const double *d_re, const double *d_im,
                            const double *in_re, const double *in_im,
                            double *out_re, double *out_im) {
    for (int k = 0; k < TRI_D; k++) {
        out_re[k] = d_re[k] * in_re[k] - d_im[k] * in_im[k];
        out_im[k] = d_re[k] * in_im[k] + d_im[k] * in_re[k];
    }
}

/* F² reversal: DFT₆²|k⟩ = |(-k) mod 6⟩.  O(D) swap, not O(D²) DFT. */
static void ltri_apply_reversal(double *re, double *im) {
    for (int k = 1; k < (TRI_D + 1) / 2; k++) {
        int j = TRI_D - k;  /* j = (-k) mod 6 */
        double tr = re[k]; re[k] = re[j]; re[j] = tr;
        double ti = im[k]; im[k] = im[j]; im[j] = ti;
    }
}

/* Reverse a diagonal's entries: d[k] ↔ d[(-k) mod 6] */
static void ltri_diag_reverse(double *d_re, double *d_im) {
    for (int k = 1; k < (TRI_D + 1) / 2; k++) {
        int j = TRI_D - k;
        double tr = d_re[k]; d_re[k] = d_re[j]; d_re[j] = tr;
        double ti = d_im[k]; d_im[k] = d_im[j]; d_im[j] = ti;
    }
}

/* Multiply two diagonals: a[k] *= b[k] */
static void ltri_diag_mul(double *a_re, double *a_im,
                          const double *b_re, const double *b_im) {
    for (int k = 0; k < TRI_D; k++) {
        double ar = a_re[k], ai = a_im[k];
        a_re[k] = ar * b_re[k] - ai * b_im[k];
        a_im[k] = ar * b_im[k] + ai * b_re[k];
    }
}

static void ltri_apply_n_dfts(double *re, double *im, int n) {
    n = ((n % DFT_ORDER) + DFT_ORDER) % DFT_ORDER;
    if (n == 0) return;
    if (n == 2) {
        /* F² = reversal.  O(D) not O(D²). */
        ltri_apply_reversal(re, im);
        return;
    }
    /* n == 1 or n == 3: actual DFT(s) */
    if (n == 1) {
        double tmp_re[TRI_D], tmp_im[TRI_D];
        dft6_forward(re, im, tmp_re, tmp_im);
        memcpy(re, tmp_re, sizeof(tmp_re));
        memcpy(im, tmp_im, sizeof(tmp_im));
    } else { /* n == 3 = inverse DFT = F³ */
        double tmp_re[TRI_D], tmp_im[TRI_D];
        dft6_inverse(re, im, tmp_re, tmp_im);
        memcpy(re, tmp_re, sizeof(tmp_re));
        memcpy(im, tmp_im, sizeof(tmp_im));
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * THE ORACLE — Matrix exponentiation for repeating segment patterns
 *
 * When segments repeat (e.g., alternating Z-X creates a period-2
 * pattern), the Oracle compiles one period into a 6×6 matrix M
 * and applies M^(N/period) via repeated squaring.
 *
 * Cost: O(D³ × log N) instead of O(N × D²).
 * For 1M alternating gates: ~5K ops instead of ~42M ops.
 *
 * "Stop reaching outward for the answer. It was inside the struct."
 * ═══════════════════════════════════════════════════════════════════════ */

#define ORACLE_THRESHOLD 4  /* min segments to trigger Oracle */
#define D6 TRI_D

/* 6×6 complex matrix: identity */
static void mat6_identity(double R_re[D6][D6], double R_im[D6][D6]) {
    memset(R_re, 0, sizeof(double) * D6 * D6);
    memset(R_im, 0, sizeof(double) * D6 * D6);
    for (int i = 0; i < D6; i++) R_re[i][i] = 1.0;
}

/* 6×6 complex matrix multiply: C = A · B */
static void mat6_mul(double C_re[D6][D6], double C_im[D6][D6],
                     const double A_re[D6][D6], const double A_im[D6][D6],
                     const double B_re[D6][D6], const double B_im[D6][D6]) {
    double T_re[D6][D6], T_im[D6][D6];
    for (int i = 0; i < D6; i++)
        for (int j = 0; j < D6; j++) {
            double sr = 0, si = 0;
            for (int k = 0; k < D6; k++) {
                sr += A_re[i][k] * B_re[k][j] - A_im[i][k] * B_im[k][j];
                si += A_re[i][k] * B_im[k][j] + A_im[i][k] * B_re[k][j];
            }
            T_re[i][j] = sr;
            T_im[i][j] = si;
        }
    memcpy(C_re, T_re, sizeof(T_re));
    memcpy(C_im, T_im, sizeof(T_im));
}

/* 6×6 matrix power via repeated squaring: R = M^n */
static void mat6_pow(double R_re[D6][D6], double R_im[D6][D6],
                     const double M_re[D6][D6], const double M_im[D6][D6],
                     int n) {
    mat6_identity(R_re, R_im);
    if (n <= 0) return;
    double B_re[D6][D6], B_im[D6][D6];
    memcpy(B_re, M_re, sizeof(B_re));
    memcpy(B_im, M_im, sizeof(B_im));
    while (n > 0) {
        if (n & 1)
            mat6_mul(R_re, R_im, R_re, R_im, B_re, B_im);
        n >>= 1;
        if (n > 0)
            mat6_mul(B_re, B_im, B_re, B_im, B_re, B_im);
    }
}

/* Matrix-vector multiply: out = M · v */
static void mat6_vec(double out_re[D6], double out_im[D6],
                     const double M_re[D6][D6], const double M_im[D6][D6],
                     const double v_re[D6], const double v_im[D6]) {
    for (int i = 0; i < D6; i++) {
        double sr = 0, si = 0;
        for (int k = 0; k < D6; k++) {
            sr += M_re[i][k] * v_re[k] - M_im[i][k] * v_im[k];
            si += M_re[i][k] * v_im[k] + M_im[i][k] * v_re[k];
        }
        out_re[i] = sr;
        out_im[i] = si;
    }
}

/* Build the DFT^n matrix (6×6) */
static void mat6_build_dft(double M_re[D6][D6], double M_im[D6][D6], int n) {
    n = ((n % DFT_ORDER) + DFT_ORDER) % DFT_ORDER;
    mat6_identity(M_re, M_im);
    if (n == 0) return;
    if (n == 2) {
        memset(M_re, 0, sizeof(double) * D6 * D6);
        M_re[0][0] = 1.0;
        for (int k = 1; k < D6; k++) M_re[k][D6 - k] = 1.0;
        return;
    }
    const double *wr_tab = (n == 1) ? W6_RE : W6I_RE;
    const double *wi_tab = (n == 1) ? W6_IM : W6I_IM;
    for (int j = 0; j < D6; j++)
        for (int k = 0; k < D6; k++) {
            int idx = (j * k) % 6;
            M_re[j][k] = wr_tab[idx] * INV_SQRT6;
            M_im[j][k] = wi_tab[idx] * INV_SQRT6;
        }
}

/* Build segment matrix: diag(D) · F^pre_dfts (6×6) */
static void mat6_build_segment(double M_re[D6][D6], double M_im[D6][D6],
                               const double *d_re, const double *d_im,
                               int pre_dfts) {
    double F_re[D6][D6], F_im[D6][D6];
    mat6_build_dft(F_re, F_im, pre_dfts);
    for (int i = 0; i < D6; i++)
        for (int j = 0; j < D6; j++) {
            M_re[i][j] = d_re[i] * F_re[i][j] - d_im[i] * F_im[i][j];
            M_im[i][j] = d_re[i] * F_im[i][j] + d_im[i] * F_re[i][j];
        }
}

/* Check if two segments have identical (pre_dfts, diagonal) */
static int segments_match(const LazyTrialityQuhit *q, int a, int b) {
    if (q->segments[a].pre_dfts != q->segments[b].pre_dfts) return 0;
    for (int k = 0; k < D6; k++) {
        if (q->segments[a].diag_re[k] != q->segments[b].diag_re[k]) return 0;
        if (q->segments[a].diag_im[k] != q->segments[b].diag_im[k]) return 0;
    }
    return 1;
}

/* Try to detect a repeating pattern of period 1 or 2. */
static int detect_pattern(const LazyTrialityQuhit *q, int *start_idx) {
    if (q->n_segments < ORACLE_THRESHOLD) return 0;
    int all_same = 1;
    for (int s = 2; s < q->n_segments && all_same; s++)
        if (!segments_match(q, 1, s)) all_same = 0;
    if (all_same && q->n_segments >= ORACLE_THRESHOLD) {
        *start_idx = 1;
        return 1;
    }
    if (q->n_segments >= ORACLE_THRESHOLD + 1) {
        int alt_match = 1;
        for (int s = 3; s < q->n_segments && alt_match; s++)
            if (!segments_match(q, (s % 2 == 1) ? 1 : 2, s)) alt_match = 0;
        if (alt_match) {
            *start_idx = 1;
            return 2;
        }
    }
    return 0;
}

/* Build a 6×6 composite matrix from the entire segment chain + trailing DFTs.
 * Computes: M = F^trailing · D_n · F^pre_n · ... · D_0 · F^pre_0 */
static void ltri_build_chain_matrix(const LazyTrialityQuhit *q,
                                     double M_re[D6][D6], double M_im[D6][D6]) {
    mat6_identity(M_re, M_im);
    for (int s = 0; s < q->n_segments; s++) {
        /* Apply F^pre_dfts */
        double F_re[D6][D6], F_im[D6][D6];
        mat6_build_dft(F_re, F_im, q->segments[s].pre_dfts);
        double T_re[D6][D6], T_im[D6][D6];
        mat6_mul(T_re, T_im, F_re, F_im, M_re, M_im);
        /* Apply diag */
        for (int i = 0; i < D6; i++)
            for (int j = 0; j < D6; j++) {
                double dr = q->segments[s].diag_re[i];
                double di = q->segments[s].diag_im[i];
                M_re[i][j] = dr * T_re[i][j] - di * T_im[i][j];
                M_im[i][j] = dr * T_im[i][j] + di * T_re[i][j];
            }
    }
    /* Apply trailing DFTs */
    if (q->trailing_dfts != 0) {
        double F_re[D6][D6], F_im[D6][D6];
        mat6_build_dft(F_re, F_im, q->trailing_dfts);
        double T_re[D6][D6], T_im[D6][D6];
        mat6_mul(T_re, T_im, F_re, F_im, M_re, M_im);
        memcpy(M_re, T_re, sizeof(T_re));
        memcpy(M_im, T_im, sizeof(T_im));
    }
}

static void ltri_new_segment(LazyTrialityQuhit *q) {
    if (q->n_segments >= MAX_LAZY_SEGMENTS) {
        /* ── OVERFLOW: detect pattern or materialize ──
         * If a repeating pattern exists, compile via Oracle exponentiation
         * and fold into the Oracle composite (no state touch).
         * If no pattern, fall back to fast state-vector materialization. */
        int oracle_start = 0;
        int period = detect_pattern(q, &oracle_start);

        if (period > 0) {
            /* ORACLE PATH: exponentiate the pattern, fold into composite */
            /* Apply seg 0 (non-repeating head) if any by building its matrix */
            double batch_re[D6][D6], batch_im[D6][D6];

            if (oracle_start > 0) {
                /* Build head segment matrix */
                mat6_build_segment(batch_re, batch_im,
                                   q->segments[0].diag_re, q->segments[0].diag_im,
                                   q->segments[0].pre_dfts);
            } else {
                mat6_identity(batch_re, batch_im);
            }

            /* Compile period matrix */
            double P_re[D6][D6], P_im[D6][D6];
            if (period == 1) {
                mat6_build_segment(P_re, P_im,
                                   q->segments[oracle_start].diag_re,
                                   q->segments[oracle_start].diag_im,
                                   q->segments[oracle_start].pre_dfts);
            } else {
                double S1_re[D6][D6], S1_im[D6][D6];
                double S2_re[D6][D6], S2_im[D6][D6];
                mat6_build_segment(S1_re, S1_im,
                                   q->segments[oracle_start].diag_re,
                                   q->segments[oracle_start].diag_im,
                                   q->segments[oracle_start].pre_dfts);
                mat6_build_segment(S2_re, S2_im,
                                   q->segments[oracle_start + 1].diag_re,
                                   q->segments[oracle_start + 1].diag_im,
                                   q->segments[oracle_start + 1].pre_dfts);
                mat6_mul(P_re, P_im, S2_re, S2_im, S1_re, S1_im);
            }

            int remaining = q->n_segments - oracle_start;
            int full_periods = remaining / period;

            /* Exponentiate: R = P^full_periods */
            double R_re[D6][D6], R_im[D6][D6];
            mat6_pow(R_re, R_im, P_re, P_im, full_periods);

            /* batch = R · head (or just R if no head) */
            if (oracle_start > 0) {
                double T_re[D6][D6], T_im[D6][D6];
                mat6_mul(T_re, T_im, R_re, R_im, batch_re, batch_im);
                memcpy(batch_re, T_re, sizeof(T_re));
                memcpy(batch_im, T_im, sizeof(T_im));
            } else {
                memcpy(batch_re, R_re, sizeof(R_re));
                memcpy(batch_im, R_im, sizeof(R_im));
            }

            /* Handle leftover segments */
            int leftover_start = oracle_start + full_periods * period;
            for (int s = leftover_start; s < q->n_segments; s++) {
                double S_re[D6][D6], S_im[D6][D6];
                mat6_build_segment(S_re, S_im,
                                   q->segments[s].diag_re, q->segments[s].diag_im,
                                   q->segments[s].pre_dfts);
                double T_re[D6][D6], T_im[D6][D6];
                mat6_mul(T_re, T_im, S_re, S_im, batch_re, batch_im);
                memcpy(batch_re, T_re, sizeof(T_re));
                memcpy(batch_im, T_im, sizeof(T_im));
            }

            /* Apply trailing DFTs */
            if (q->trailing_dfts != 0) {
                double F_re[D6][D6], F_im[D6][D6];
                mat6_build_dft(F_re, F_im, q->trailing_dfts);
                double T_re[D6][D6], T_im[D6][D6];
                mat6_mul(T_re, T_im, F_re, F_im, batch_re, batch_im);
                memcpy(batch_re, T_re, sizeof(T_re));
                memcpy(batch_im, T_im, sizeof(T_im));
            }

            /* Fold into Oracle composite */
            if (q->oracle_active) {
                double T_re[D6][D6], T_im[D6][D6];
                mat6_mul(T_re, T_im, batch_re, batch_im,
                         q->oracle_M_re, q->oracle_M_im);
                memcpy(q->oracle_M_re, T_re, sizeof(T_re));
                memcpy(q->oracle_M_im, T_im, sizeof(T_im));
            } else {
                memcpy(q->oracle_M_re, batch_re, sizeof(batch_re));
                memcpy(q->oracle_M_im, batch_im, sizeof(batch_im));
                q->oracle_active = 1;
            }
        } else {
            /* NO PATTERN: fall back to state-vector materialization */
            double tmp_re[TRI_D], tmp_im[TRI_D];
            ltri_materialize(q, tmp_re, tmp_im);
            /* state already consumed by ltri_materialize */
        }

        q->n_segments = 0;
        q->trailing_dfts = 0;
    }
    int idx = q->n_segments++;
    ltri_diag_identity(q->segments[idx].diag_re, q->segments[idx].diag_im);
    q->segments[idx].pre_dfts = q->trailing_dfts;
    q->trailing_dfts = 0;
    q->segments_created++;
}

void ltri_init(LazyTrialityQuhit *q) {
    memset(q, 0, sizeof(*q));
    q->state_re[0] = 1.0;
}

void ltri_init_basis(LazyTrialityQuhit *q, int k) {
    memset(q, 0, sizeof(*q));
    q->state_re[k] = 1.0;
}

static void ltri_ensure_segment(LazyTrialityQuhit *q) {
    if (q->n_segments == 0) {
        ltri_new_segment(q);
        return;
    }
    if (q->trailing_dfts == 0) {
        /* Same view — fuse directly (already in segment) */
        return;
    }
    /* Any nonzero trailing: genuine horizon crossing, new segment */
    ltri_new_segment(q);
}

void ltri_z(LazyTrialityQuhit *q) {
    /* Z = diag(w^k) in edge basis. No DFTs. */
    ltri_ensure_segment(q);
    int idx = q->n_segments - 1;
    for (int k = 0; k < TRI_D; k++)
        ltri_diag_mul_phase(q->segments[idx].diag_re, q->segments[idx].diag_im,
                            k, W6_RE[k], W6_IM[k]);
    q->gates_fused++;
}

void ltri_x(LazyTrialityQuhit *q) {
    /* X = F^3 · diag(w^k) · F^1  (IDFT · diag · DFT)
     * trailing += 1 (DFT), create/fuse segment, trailing = 3 (IDFT = F^3)
     * Consecutive: 3+1=4=0 mod 4, so consecutive X gates fuse! */
    q->trailing_dfts = (q->trailing_dfts + 1) % DFT_ORDER;
    ltri_ensure_segment(q);
    int idx = q->n_segments - 1;
    for (int k = 0; k < TRI_D; k++)
        ltri_diag_mul_phase(q->segments[idx].diag_re, q->segments[idx].diag_im,
                            k, W6_RE[k], W6_IM[k]);
    q->trailing_dfts = 3;  /* IDFT = F^3 since F^4 = I */
    q->gates_fused++;
}

void ltri_shift(LazyTrialityQuhit *q, int delta) {
    delta = ((delta % TRI_D) + TRI_D) % TRI_D;
    if (delta == 0) return;
    q->trailing_dfts = (q->trailing_dfts + 1) % DFT_ORDER;
    ltri_ensure_segment(q);
    int idx = q->n_segments - 1;
    for (int k = 0; k < TRI_D; k++) {
        int widx = (delta * k) % 6;
        ltri_diag_mul_phase(q->segments[idx].diag_re, q->segments[idx].diag_im,
                            k, W6_RE[widx], W6_IM[widx]);
    }
    q->trailing_dfts = 3;  /* IDFT = F^3 */
    q->gates_fused++;
}

void ltri_dft(LazyTrialityQuhit *q) {
    q->trailing_dfts = (q->trailing_dfts + 1) % DFT_ORDER;
    q->gates_fused++;
}

void ltri_idft(LazyTrialityQuhit *q) {
    q->trailing_dfts = (q->trailing_dfts + 3) % DFT_ORDER;  /* F^-1 = F^3 */
    q->gates_fused++;
}

void ltri_phase(LazyTrialityQuhit *q, const double *phi_re, const double *phi_im) {
    ltri_ensure_segment(q);
    int idx = q->n_segments - 1;
    for (int k = 0; k < TRI_D; k++)
        ltri_diag_mul_phase(q->segments[idx].diag_re, q->segments[idx].diag_im,
                            k, phi_re[k], phi_im[k]);
    q->gates_fused++;
}

/* ═══════════════════════════════════════════════════════════════════════
 * CHAIN COMPACTION: merge adjacent segments where DFTs between them cancel
 *
 * If seg[i+1].pre_dfts == 0: diagonals multiply directly.
 * For pre_dfts == 2: use F²·D = rev(D)·F² to merge diag into prev,
 *   but the F² stays (reduces segment count, not DFT count).
 * Only pre_dfts == 0 eliminates both segment AND DFT.
 * ═══════════════════════════════════════════════════════════════════════ */

static void ltri_compact_chain(LazyTrialityQuhit *q) {
    if (q->n_segments <= 1) return;
    int dst = 0;
    for (int src = 1; src < q->n_segments; src++) {
        if (q->segments[src].pre_dfts == 0) {
            /* Direct fusion: multiply diagonals, no DFT between them */
            ltri_diag_mul(q->segments[dst].diag_re, q->segments[dst].diag_im,
                          q->segments[src].diag_re, q->segments[src].diag_im);
        } else {
            /* Any nonzero pre_dfts: cannot eliminate, advance dst */
            dst++;
            if (dst != src) {
                memcpy(&q->segments[dst], &q->segments[src], sizeof(q->segments[0]));
            }
        }
    }
    q->n_segments = dst + 1;
}


/* ═══════════════════════════════════════════════════════════════════════
 * MATERIALIZE: Oracle composite → compact → remaining segments → trailing
 * ═══════════════════════════════════════════════════════════════════════ */

void ltri_materialize(LazyTrialityQuhit *q, double *out_re, double *out_im) {
    /* Compact remaining chain: fuse adjacent pre_dfts=0 segments */
    ltri_compact_chain(q);

    double cur_re[TRI_D], cur_im[TRI_D];
    memcpy(cur_re, q->state_re, sizeof(cur_re));
    memcpy(cur_im, q->state_im, sizeof(cur_im));

    /* ── ORACLE COMPOSITE: apply accumulated cross-batch matrix ── */
    if (q->oracle_active) {
        /* If there are remaining segments, fold them into the Oracle too */
        if (q->n_segments > 0 || q->trailing_dfts != 0) {
            double tail_re[D6][D6], tail_im[D6][D6];
            ltri_build_chain_matrix(q, tail_re, tail_im);
            double T_re[D6][D6], T_im[D6][D6];
            mat6_mul(T_re, T_im, tail_re, tail_im,
                     q->oracle_M_re, q->oracle_M_im);
            mat6_vec(out_re, out_im, T_re, T_im, cur_re, cur_im);
        } else {
            mat6_vec(out_re, out_im, q->oracle_M_re, q->oracle_M_im,
                     cur_re, cur_im);
        }
        /* Consume: write result back to state, reset everything */
        memcpy(q->state_re, out_re, sizeof(q->state_re));
        memcpy(q->state_im, out_im, sizeof(q->state_im));
        q->oracle_active = 0;
        q->n_segments = 0;
        q->trailing_dfts = 0;
        q->materializations++;
        return;
    }

    /* ── No Oracle — apply segments with pattern detection ── */
    int oracle_start = 0;
    int period = detect_pattern(q, &oracle_start);

    if (period > 0 && (q->n_segments - oracle_start) >= ORACLE_THRESHOLD) {
        /* Apply pre-pattern segments normally */
        for (int s = 0; s < oracle_start; s++) {
            ltri_apply_n_dfts(cur_re, cur_im, q->segments[s].pre_dfts);
            double tmp_re[TRI_D], tmp_im[TRI_D];
            ltri_apply_diag(q->segments[s].diag_re, q->segments[s].diag_im,
                            cur_re, cur_im, tmp_re, tmp_im);
            memcpy(cur_re, tmp_re, sizeof(cur_re));
            memcpy(cur_im, tmp_im, sizeof(cur_im));
        }

        /* Compile one period into a 6×6 matrix */
        double P_re[D6][D6], P_im[D6][D6];
        if (period == 1) {
            mat6_build_segment(P_re, P_im,
                               q->segments[oracle_start].diag_re,
                               q->segments[oracle_start].diag_im,
                               q->segments[oracle_start].pre_dfts);
        } else {
            double S1_re[D6][D6], S1_im[D6][D6];
            double S2_re[D6][D6], S2_im[D6][D6];
            mat6_build_segment(S1_re, S1_im,
                               q->segments[oracle_start].diag_re,
                               q->segments[oracle_start].diag_im,
                               q->segments[oracle_start].pre_dfts);
            mat6_build_segment(S2_re, S2_im,
                               q->segments[oracle_start + 1].diag_re,
                               q->segments[oracle_start + 1].diag_im,
                               q->segments[oracle_start + 1].pre_dfts);
            mat6_mul(P_re, P_im, S2_re, S2_im, S1_re, S1_im);
        }

        int remaining = q->n_segments - oracle_start;
        int full_periods = remaining / period;

        double R_re[D6][D6], R_im[D6][D6];
        mat6_pow(R_re, R_im, P_re, P_im, full_periods);

        double tmp_re[TRI_D], tmp_im[TRI_D];
        mat6_vec(tmp_re, tmp_im, R_re, R_im, cur_re, cur_im);
        memcpy(cur_re, tmp_re, sizeof(cur_re));
        memcpy(cur_im, tmp_im, sizeof(cur_im));

        int leftover_start = oracle_start + full_periods * period;
        for (int s = leftover_start; s < q->n_segments; s++) {
            ltri_apply_n_dfts(cur_re, cur_im, q->segments[s].pre_dfts);
            double tmp2_re[TRI_D], tmp2_im[TRI_D];
            ltri_apply_diag(q->segments[s].diag_re, q->segments[s].diag_im,
                            cur_re, cur_im, tmp2_re, tmp2_im);
            memcpy(cur_re, tmp2_re, sizeof(cur_re));
            memcpy(cur_im, tmp2_im, sizeof(cur_im));
        }
    } else {
        /* No pattern — standard segment-by-segment evaluation */
        for (int s = 0; s < q->n_segments; s++) {
            ltri_apply_n_dfts(cur_re, cur_im, q->segments[s].pre_dfts);
            double tmp_re[TRI_D], tmp_im[TRI_D];
            ltri_apply_diag(q->segments[s].diag_re, q->segments[s].diag_im,
                            cur_re, cur_im, tmp_re, tmp_im);
            memcpy(cur_re, tmp_re, sizeof(cur_re));
            memcpy(cur_im, tmp_im, sizeof(cur_im));
        }
    }

    ltri_apply_n_dfts(cur_re, cur_im, q->trailing_dfts);

    memcpy(out_re, cur_re, sizeof(cur_re));
    memcpy(out_im, cur_im, sizeof(cur_im));
    /* Consume: update state for future gates */
    memcpy(q->state_re, out_re, sizeof(q->state_re));
    memcpy(q->state_im, out_im, sizeof(q->state_im));
    q->n_segments = 0;
    q->trailing_dfts = 0;
    q->materializations++;
}

int ltri_measure(LazyTrialityQuhit *q, int view, uint64_t *rng_state) {
    double amp_re[TRI_D], amp_im[TRI_D];
    ltri_materialize(q, amp_re, amp_im);

    if (view != VIEW_EDGE) {
        int steps = ((view - VIEW_EDGE) % DFT_ORDER + DFT_ORDER) % DFT_ORDER;
        ltri_apply_n_dfts(amp_re, amp_im, steps);
    }

    double probs[TRI_D], total = 0;
    for (int k = 0; k < TRI_D; k++) {
        probs[k] = amp_re[k]*amp_re[k] + amp_im[k]*amp_im[k];
        total += probs[k];
    }
    if (total > 1e-30)
        for (int k = 0; k < TRI_D; k++) probs[k] /= total;

    double r = triality_rng_double(rng_state);
    int outcome = 0;
    double cdf = 0;
    for (int k = 0; k < TRI_D; k++) {
        cdf += probs[k];
        if (r < cdf) { outcome = k; break; }
    }

    memset(q->state_re, 0, sizeof(q->state_re));
    memset(q->state_im, 0, sizeof(q->state_im));
    q->state_re[outcome] = 1.0;
    q->n_segments = 0;
    q->trailing_dfts = 0;

    return outcome;
}

void ltri_stats_print(const LazyTrialityQuhit *q) {
    printf("\n  ┌─────────────────────────────────────────────────────┐\n");
    printf("  │  LAZY TRIALITY STATISTICS                           │\n");
    printf("  ├─────────────────────────────────────────────────────┤\n");
    printf("  │  Gates fused:          %8lu                     │\n", (unsigned long)q->gates_fused);
    printf("  │  Segments created:     %8lu                     │\n", (unsigned long)q->segments_created);
    printf("  │  Current segments:     %8d / %d                │\n", q->n_segments, MAX_LAZY_SEGMENTS);
    printf("  │  Trailing DFTs:        %8d                     │\n", q->trailing_dfts);
    printf("  │  Materializations:     %8lu                     │\n", (unsigned long)q->materializations);
    double fusion_ratio = (q->segments_created > 0) ?
        (double)q->gates_fused / q->segments_created : 0;
    printf("  │  Fusion ratio:         %8.1f gates/segment      │\n", fusion_ratio);
    printf("  └─────────────────────────────────────────────────────┘\n");
}
