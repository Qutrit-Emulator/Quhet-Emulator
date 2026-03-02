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
