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
#include "s6_exotic.h"

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
static const double INV_SQRT2 = 0.70710678118654752440;  /* 1/√2 */

/* ═══════════════════════════════════════════════════════════════════════
 * STATISTICS
 * ═══════════════════════════════════════════════════════════════════════ */

TrialityStats triality_stats = {0};

void triality_stats_reset(void) {
    memset(&triality_stats, 0, sizeof(triality_stats));
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
    memset(q, 0, sizeof(TrialityQuhit));
    q->edge_re[0] = 1.0;       /* |0⟩ in computational basis */

    /* Compute the correct views of |0⟩ via DFT₆ */
    dft6_forward(q->edge_re, q->edge_im, q->vertex_re, q->vertex_im);
    dft6_forward(q->vertex_re, q->vertex_im, q->diag_re, q->diag_im);

    q->dirty = DIRTY_FOLDED | DIRTY_EXOTIC;
    q->primary = VIEW_EDGE;

    /* Enhancement flags */
    q->eigenstate_class = -1;
    q->active_mask  = 0x01;   /* only |0⟩ is active */
    q->active_count = 1;
    q->real_valued  = 1;      /* |0⟩ is real */
    q->delta_valid = 0;       /* Fix #5: exotic cache starts invalid */
    q->exotic_syntheme = 0;   /* default exotic: {(0,1),(2,3),(4,5)} */
}

void triality_init_basis(TrialityQuhit *q, int k) {
    memset(q, 0, sizeof(TrialityQuhit));
    q->edge_re[k] = 1.0;
    q->dirty = DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED | DIRTY_EXOTIC;
    q->primary = VIEW_EDGE;
    q->eigenstate_class = -1;
    q->active_mask = (uint8_t)(1 << k);
    q->active_count = 1;
    q->real_valued = 1;
    q->delta_valid = 0;  /* Fix #5 */
    q->exotic_syntheme = 0;
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
        case VIEW_FOLDED:   return q->folded_re;
    }
    return q->edge_re;
}

static double *view_im(TrialityQuhit *q, int v) {
    switch(v) {
        case VIEW_EDGE:     return q->edge_im;
        case VIEW_VERTEX:   return q->vertex_im;
        case VIEW_DIAGONAL: return q->diag_im;
        case VIEW_FOLDED:   return q->folded_im;
    }
    return q->edge_im;
}

static int view_dirty_bit(int v) {
    return 1 << v;
}

/* ── Antipodal fold primitives (Stage 1 of Cooley-Tukey DFT₆) ── */

/* Forward fold: pair (k, k+3) into vesica (sum) and wave (difference)
 * folded[k]   = (edge[k] + edge[k+3]) / √2   for k=0,1,2  (vesica)
 * folded[k+3] = (edge[k] - edge[k+3]) / √2   for k=0,1,2  (wave) */
static void fold_forward(const double *in_re, const double *in_im,
                         double *out_re, double *out_im)
{
    for (int k = 0; k < 3; k++) {
        double a_re = in_re[k],     a_im = in_im[k];
        double b_re = in_re[k + 3], b_im = in_im[k + 3];
        out_re[k]     = INV_SQRT2 * (a_re + b_re);
        out_im[k]     = INV_SQRT2 * (a_im + b_im);
        out_re[k + 3] = INV_SQRT2 * (a_re - b_re);
        out_im[k + 3] = INV_SQRT2 * (a_im - b_im);
    }
}

/* Inverse fold: reconstruct edge from folded (vesica + wave) */
static void fold_inverse(const double *in_re, const double *in_im,
                         double *out_re, double *out_im)
{
    for (int k = 0; k < 3; k++) {
        double v_re = in_re[k],     v_im = in_im[k];
        double w_re = in_re[k + 3], w_im = in_im[k + 3];
        out_re[k]     = INV_SQRT2 * (v_re + w_re);
        out_im[k]     = INV_SQRT2 * (v_im + w_im);
        out_re[k + 3] = INV_SQRT2 * (v_re - w_re);
        out_im[k + 3] = INV_SQRT2 * (v_im - w_im);
    }
}

/* Folded → Vertex: apply twiddle + DFT₃ (Stages 2-3 of Cooley-Tukey)
 * This completes DFT₆ from the folded intermediate. Cost: O(12). */
static void folded_to_vertex(const double *fold_re, const double *fold_im,
                             double *vert_re, double *vert_im)
{
    static const double w3_re = -0.5;
    static const double w3_im = 0.86602540378443864676;
    static const double n3 = 0.57735026918962576451; /* 1/√3 */

    /* Stage 2: twiddle ω₆^(s·p) — only affects p=1 entries (indices 3,4,5) */
    double tw_re[6], tw_im[6];
    memcpy(tw_re, fold_re, 6 * sizeof(double));
    memcpy(tw_im, fold_im, 6 * sizeof(double));
    /* s=1,p=1 (index 4): multiply by ω₆^1 */
    {
        double r = tw_re[4], i = tw_im[4];
        tw_re[4] = W6_RE[1] * r - W6_IM[1] * i;
        tw_im[4] = W6_RE[1] * i + W6_IM[1] * r;
    }
    /* s=2,p=1 (index 5): multiply by ω₆^2 */
    {
        double r = tw_re[5], i = tw_im[5];
        tw_re[5] = W6_RE[2] * r - W6_IM[2] * i;
        tw_im[5] = W6_RE[2] * i + W6_IM[2] * r;
    }

    /* Stage 3: DFT₃ ⊗ I₂ per parity */
    for (int p = 0; p < 2; p++) {
        double a_re = tw_re[0 + p*3], a_im = tw_im[0 + p*3];
        double b_re = tw_re[1 + p*3], b_im = tw_im[1 + p*3];
        double c_re = tw_re[2 + p*3], c_im = tw_im[2 + p*3];

        /* j=0: (a + b + c) / √3 */
        vert_re[p]     = n3 * (a_re + b_re + c_re);
        vert_im[p]     = n3 * (a_im + b_im + c_im);

        /* j=1: (a + ω₃·b + ω₃²·c) / √3 */
        double wb_re = w3_re * b_re - w3_im * b_im;
        double wb_im = w3_re * b_im + w3_im * b_re;
        double wc_re = w3_re * c_re + w3_im * c_im;
        double wc_im = w3_re * c_im - w3_im * c_re;
        vert_re[2 + p] = n3 * (a_re + wb_re + wc_re);
        vert_im[2 + p] = n3 * (a_im + wb_im + wc_im);

        /* j=2: (a + ω₃²·b + ω₃·c) / √3 */
        double w2b_re = w3_re * b_re + w3_im * b_im;
        double w2b_im = w3_re * b_im - w3_im * b_re;
        double w2c_re = w3_re * c_re - w3_im * c_im;
        double w2c_im = w3_re * c_im + w3_im * c_re;
        vert_re[4 + p] = n3 * (a_re + w2b_re + w2c_re);
        vert_im[4 + p] = n3 * (a_im + w2b_im + w2c_im);
    }
}

static void convert_view(TrialityQuhit *q, int from, int to) {
    double *src_re = view_re(q, from), *src_im = view_im(q, from);
    double *dst_re = view_re(q, to),   *dst_im = view_im(q, to);

    /* Handle folded view conversions */
    if (from <= VIEW_DIAGONAL && to == VIEW_FOLDED) {
        /* Need edge view first */
        if (from != VIEW_EDGE) {
            triality_ensure_view(q, VIEW_EDGE);
            src_re = q->edge_re; src_im = q->edge_im;
        }
        fold_forward(src_re, src_im, dst_re, dst_im);
        triality_stats.edge_to_folded++;
        return;
    }
    if (from == VIEW_FOLDED && to == VIEW_EDGE) {
        fold_inverse(src_re, src_im, dst_re, dst_im);
        return;
    }
    if (from == VIEW_FOLDED && to == VIEW_VERTEX) {
        folded_to_vertex(src_re, src_im, dst_re, dst_im);
        triality_stats.folded_to_vertex++;
        return;
    }
    if (from == VIEW_FOLDED) {
        /* Folded → anything else: go via edge */
        triality_ensure_view(q, VIEW_EDGE);
        convert_view(q, VIEW_EDGE, to);
        return;
    }

    /* Standard 3-view conversion */
    int steps = (to - from + 3) % 3;  /* 1 = one DFT₆, 2 = one IDFT₆ */

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

    /* ── Enhancement 2: Eigenstate shortcut ──
     * DFT₆ eigenstates are identical in all views (up to eigenvalue phase).
     * F|ψ⟩ = λ|ψ⟩, so vertex = λ·edge, diag = λ²·edge.
     * Eigenvalues: {1, -1, i, -i} indexed 0..3. */
    if (q->eigenstate_class >= 0 && view <= VIEW_DIAGONAL) {
        /* Compute λ^steps where steps = (view - primary + 3) % 3 */
        int source = q->primary;
        if (source > VIEW_DIAGONAL) source = VIEW_EDGE; /* folded → use edge */
        int steps = (view - source + 3) % 3;
        if (steps == 0) {
            /* Same view — just copy */
            memcpy(view_re(q, view), view_re(q, source), TRI_D * sizeof(double));
            memcpy(view_im(q, view), view_im(q, source), TRI_D * sizeof(double));
        } else {
            /* λ^steps: eigenvalue raised to power */
            /* eigenvalue table: class 0=1, 1=-1, 2=i, 3=-i */
            static const double EV_RE[4] = {1.0, -1.0,  0.0,  0.0};
            static const double EV_IM[4] = {0.0,  0.0,  1.0, -1.0};
            double lr = EV_RE[q->eigenstate_class];
            double li = EV_IM[q->eigenstate_class];
            /* raise to 'steps' power */
            double pr = 1.0, pi = 0.0;
            for (int s = 0; s < steps; s++) {
                double tr = pr * lr - pi * li;
                double ti = pr * li + pi * lr;
                pr = tr; pi = ti;
            }
            /* dst = pr*src + pi*i*src */
            const double *sr = view_re(q, source), *si = view_im(q, source);
            double *dr = view_re(q, view), *di = view_im(q, view);
            for (int k = 0; k < TRI_D; k++) {
                dr[k] = pr * sr[k] - pi * si[k];
                di[k] = pr * si[k] + pi * sr[k];
            }
        }
        q->dirty &= ~view_dirty_bit(view);
        triality_stats.eigenstate_skips++;
        return;
    }

    /* Find a clean view to convert from */
    int source = q->primary;
    if (q->dirty & view_dirty_bit(source)) {
        /* Primary is dirty — find any clean view */
        for (int v = 0; v < 4; v++) {
            if (!(q->dirty & view_dirty_bit(v))) {
                source = v;
                break;
            }
        }
    }

    /* ── Enhancement 1: Route through folded view when cheaper ──
     * Edge→Vertex via Folded: fold O(6) + twiddle+DFT₃ O(12) = O(18)
     * vs direct DFT₆ O(36). Use folded path when both endpoints are
     * edge and vertex, and folded view is clean or cheaply obtainable. */
    if (source == VIEW_EDGE && view == VIEW_VERTEX) {
        if (!(q->dirty & DIRTY_FOLDED)) {
            /* Folded is clean — just do Stages 2-3 */
            folded_to_vertex(q->folded_re, q->folded_im,
                            q->vertex_re, q->vertex_im);
            q->dirty &= ~DIRTY_VERTEX;
            triality_stats.folded_to_vertex++;
            return;
        }
        /* Folded is dirty but edge is clean — fold first, then convert */
        fold_forward(q->edge_re, q->edge_im, q->folded_re, q->folded_im);
        q->dirty &= ~DIRTY_FOLDED;
        triality_stats.edge_to_folded++;
        folded_to_vertex(q->folded_re, q->folded_im,
                        q->vertex_re, q->vertex_im);
        q->dirty &= ~DIRTY_VERTEX;
        triality_stats.folded_to_vertex++;
        return;
    }

    /* ── Fix #3: Inverse fold path — Vertex→Edge via IDFT₃ + unfold ──
     * Reverse of the forward fold path. IDFT₆ = fold_inverse(IDFT₃+untwiddle(vertex)).
     * Cost: O(18) vs O(36) for direct IDFT₆. */
    if (source == VIEW_VERTEX && view == VIEW_EDGE) {
        /* Vertex → Folded (inverse of Stages 2-3): IDFT₃ + untwiddle */
        /* We directly compute the full IDFT₆ but through the combined path:
         * Since IDFT₆ = fold_inverse ∘ (IDFT₃⊗I₂) ∘ untwiddle,
         * use the existing fold_inverse for the final step. */
        static const double w3_re = -0.5;
        static const double w3_im = 0.86602540378443864676;
        static const double n3 = 0.57735026918962576451; /* 1/√3 */

        /* Stage 1 (inverse of Stage 3): IDFT₃ ⊗ I₂ per parity */
        double tw_re[6], tw_im[6];
        for (int p = 0; p < 2; p++) {
            double a_re = q->vertex_re[p],     a_im = q->vertex_im[p];
            double b_re = q->vertex_re[2 + p], b_im = q->vertex_im[2 + p];
            double c_re = q->vertex_re[4 + p], c_im = q->vertex_im[4 + p];

            /* j=0: (a + b + c) / √3 */
            tw_re[0 + p*3] = n3 * (a_re + b_re + c_re);
            tw_im[0 + p*3] = n3 * (a_im + b_im + c_im);

            /* j=1: (a + ω₃⁻¹·b + ω₃⁻²·c) / √3 */
            double wb_re = w3_re * b_re + w3_im * b_im;  /* ω₃⁻¹ = conj(ω₃) */
            double wb_im = w3_re * b_im - w3_im * b_re;
            double wc_re = w3_re * c_re - w3_im * c_im;  /* ω₃⁻² = conj(ω₃²) */
            double wc_im = w3_re * c_im + w3_im * c_re;
            tw_re[1 + p*3] = n3 * (a_re + wb_re + wc_re);
            tw_im[1 + p*3] = n3 * (a_im + wb_im + wc_im);

            /* j=2: (a + ω₃⁻²·b + ω₃⁻¹·c) / √3 */
            double w2b_re = w3_re * b_re - w3_im * b_im;  /* ω₃⁻² */
            double w2b_im = w3_re * b_im + w3_im * b_re;
            double w2c_re = w3_re * c_re + w3_im * c_im;  /* ω₃⁻¹ */
            double w2c_im = w3_re * c_im - w3_im * c_re;
            tw_re[2 + p*3] = n3 * (a_re + w2b_re + w2c_re);
            tw_im[2 + p*3] = n3 * (a_im + w2b_im + w2c_im);
        }

        /* Stage 2 (inverse twiddle): multiply indices 4,5 by ω₆⁻¹, ω₆⁻² */
        {
            double r = tw_re[4], i = tw_im[4];
            tw_re[4] = W6I_RE[1] * r - W6I_IM[1] * i;  /* ω₆⁻¹ */
            tw_im[4] = W6I_RE[1] * i + W6I_IM[1] * r;
        }
        {
            double r = tw_re[5], i = tw_im[5];
            tw_re[5] = W6I_RE[2] * r - W6I_IM[2] * i;  /* ω₆⁻² */
            tw_im[5] = W6I_RE[2] * i + W6I_IM[2] * r;
        }

        /* Stage 3: inverse fold → edge */
        fold_inverse(tw_re, tw_im, q->edge_re, q->edge_im);

        q->dirty &= ~DIRTY_EDGE;
        /* Also cache the folded intermediate */
        memcpy(q->folded_re, tw_re, sizeof(tw_re));
        memcpy(q->folded_im, tw_im, sizeof(tw_im));
        q->dirty &= ~DIRTY_FOLDED;
        triality_stats.folded_to_vertex++;  /* Count as fold-path usage */
        return;
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
    if (q->real_valued) {
        /* Enhancement 4: real fast path — fewer multiplies */
        int all_real_phases = 1;
        for (int k = 0; k < TRI_D; k++)
            if (fabs(phi_im[k]) > 1e-15) { all_real_phases = 0; break; }
        if (all_real_phases) {
            for (int k = 0; k < TRI_D; k++) {
                if (!(q->active_mask & (1 << k))) continue; /* Enhancement 3 */
                q->edge_re[k] *= phi_re[k];
            }
            triality_stats.real_fast_path++;
            triality_stats.gates_edge++;
            q->primary = VIEW_EDGE;
            q->dirty |= DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED;
            q->delta_valid = 0;  /* Fix #5 */
            return;
        }
    }
    for (int k = 0; k < TRI_D; k++) {
        if (!(q->active_mask & (1 << k))) { /* Enhancement 3 */
            triality_stats.mask_skips++;
            continue;
        }
        double re = q->edge_re[k], im = q->edge_im[k];
        q->edge_re[k] = re * phi_re[k] - im * phi_im[k];
        q->edge_im[k] = re * phi_im[k] + im * phi_re[k];
    }
    /* Phase gate may introduce imaginary parts */
    q->real_valued = 0;
    q->primary = VIEW_EDGE;
    q->dirty |= DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED;
    q->delta_valid = 0;  /* Fix #5 */
    triality_stats.gates_edge++;
}

/* Single-phase: one basis state only — O(1) */
void triality_phase_single(TrialityQuhit *q, int k, double phi_re, double phi_im) {
    triality_ensure_view(q, VIEW_EDGE);
    q->delta_valid = 0;  /* Fix #5: always invalidate on gate call */
    if (!(q->active_mask & (1 << k))) {
        triality_stats.mask_skips++;
        return; /* Enhancement 3: skip if this basis state is zero */
    }
    double re = q->edge_re[k], im = q->edge_im[k];
    q->edge_re[k] = re * phi_re - im * phi_im;
    q->edge_im[k] = re * phi_im + im * phi_re;
    if (fabs(phi_im) > 1e-15) q->real_valued = 0;
    q->primary = VIEW_EDGE;
    q->dirty |= DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED;
    triality_stats.gates_edge++;
}

/* Z gate: |k⟩ → ω^k|k⟩ — diagonal in edge, O(D) */
void triality_z(TrialityQuhit *q) {
    triality_ensure_view(q, VIEW_EDGE);
    for (int k = 0; k < TRI_D; k++) {
        if (!(q->active_mask & (1 << k))) { /* Enhancement 3 */
            triality_stats.mask_skips++;
            continue;
        }
        double wr = W6_RE[k], wi = W6_IM[k];
        double re = q->edge_re[k], im = q->edge_im[k];
        q->edge_re[k] = re * wr - im * wi;
        q->edge_im[k] = re * wi + im * wr;
    }
    q->real_valued = 0;   /* Z introduces complex phases (ω has imaginary part) */
    q->primary = VIEW_EDGE;
    q->dirty |= DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED;
    q->delta_valid = 0;  /* Fix #5 */
    triality_stats.gates_edge++;
    /* Z preserves eigenstate_class: Z|ψ⟩ scales by ω per component,
     * but if ψ is a DFT₆ eigenstate, Z|ψ⟩ = X|ψ⟩ rotated... clear it. */
    q->eigenstate_class = -1;
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
    q->real_valued = 0;
    q->eigenstate_class = -1; /* Shift breaks eigenstate */
    q->active_mask = 0x3F; q->active_count = 6; /* Edge view now dirty, mask is stale */
    q->primary = VIEW_VERTEX;
    q->dirty |= DIRTY_EDGE | DIRTY_DIAGONAL | DIRTY_FOLDED;
    q->delta_valid = 0;  /* Fix #5 */
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
    triality_sync_all(q);

    double tmp_re[TRI_D], tmp_im[TRI_D];
    memcpy(tmp_re, q->edge_re, sizeof(tmp_re));
    memcpy(tmp_im, q->edge_im, sizeof(tmp_im));

    dft6_forward(tmp_re, tmp_im, q->edge_re, q->edge_im);
    q->primary = VIEW_EDGE;
    q->dirty = DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED;
    q->real_valued = 0;  /* DFT introduces complex amplitudes */
    q->eigenstate_class = -1;  /* Clear (might still be eigenstate but need re-detect) */
    q->active_mask = 0x3F; q->active_count = 6; /* DFT spreads to all states */
    q->delta_valid = 0;  /* Fix #5 */
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
    q->dirty = DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED;
    q->real_valued = 0;
    q->eigenstate_class = -1;
    q->active_mask = 0x3F; q->active_count = 6;
    q->delta_valid = 0;  /* Fix #5 */
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
    q->eigenstate_class = -1;
    q->delta_valid = 0;  /* Fix #5: invalidate exotic cache */

    /* Fix #6: Compute actual active_mask from output instead of conservative 0x3F */
    uint8_t mask = 0;
    int count = 0;
    for (int k = 0; k < TRI_D; k++) {
        if (out_re[k] * out_re[k] + out_im[k] * out_im[k] > 1e-30) {
            mask |= (uint8_t)(1 << k);
            count++;
        }
    }
    q->active_mask = mask;
    q->active_count = (uint8_t)count;

    /* Fix #6: Detect real-valued from output */
    int is_real = 1;
    for (int k = 0; k < TRI_D; k++) {
        if (fabs(out_im[k]) > 1e-15) { is_real = 0; break; }
    }
    q->real_valued = (uint8_t)is_real;

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

    /* Fix #4: Refresh active_mask from live edge amplitudes.
     * The mask may be stale if a prior operation (shift, DFT) changed
     * the primary view and the mask was set conservatively to 0x3F.
     * Since we just ensured edge view, recomputing is cheap: O(D). */
    triality_update_mask(a);
    triality_update_mask(b);

    /* Enhancement 3: skip inactive basis states in the CZ inner loop.
     * This is the biggest win — D_eff × D_eff iterations instead of 36. */
    uint8_t ma = a->active_mask, mb = b->active_mask;

    /* ── Optimization: Triangle Shortcut ──
     * The hexagon has two Z₃ triangles:
     *   EVEN {0,2,4} = mask 0x15    ODD {1,3,5} = mask 0x2A
     * When active_mask is confined to one triangle, we only iterate 3
     * states instead of 6, reducing the inner loop from 36 to 9 ops.
     * Triangle indices for direct iteration: */
    static const int TRI_EVEN[3] = {0, 2, 4};
    static const int TRI_ODD[3]  = {1, 3, 5};
    #define MASK_EVEN 0x15   /* bits 0,2,4 */
    #define MASK_ODD  0x2A   /* bits 1,3,5 */

    const int *tri_a = NULL, *tri_b = NULL;
    int na = 0, nb = 0;

    if ((ma & ~MASK_EVEN) == 0 && ma) { tri_a = TRI_EVEN; na = __builtin_popcount(ma); }
    else if ((ma & ~MASK_ODD) == 0 && ma) { tri_a = TRI_ODD; na = __builtin_popcount(ma); }

    if ((mb & ~MASK_EVEN) == 0 && mb) { tri_b = TRI_EVEN; nb = __builtin_popcount(mb); }
    else if ((mb & ~MASK_ODD) == 0 && mb) { tri_b = TRI_ODD; nb = __builtin_popcount(mb); }

    if (tri_a && tri_b) {
        /* Both confined to triangles — iterate only triangle indices */
        double eff_a_re[TRI_D] = {0}, eff_a_im[TRI_D] = {0};
        double eff_b_re[TRI_D] = {0}, eff_b_im[TRI_D] = {0};

        for (int ti = 0; ti < 3; ti++) {
            int j = tri_a[ti];
            if (!(ma & (1 << j))) continue;
            double aprob = a->edge_re[j]*a->edge_re[j] + a->edge_im[j]*a->edge_im[j];
            for (int tk = 0; tk < 3; tk++) {
                int k = tri_b[tk];
                if (!(mb & (1 << k))) continue;
                int idx = (j * k) % 6;
                double bprob = b->edge_re[k]*b->edge_re[k] + b->edge_im[k]*b->edge_im[k];
                eff_a_re[j] += bprob * W6_RE[idx];
                eff_a_im[j] += bprob * W6_IM[idx];
                eff_b_re[k] += aprob * W6_RE[idx];
                eff_b_im[k] += aprob * W6_IM[idx];
            }
        }
        for (int ti = 0; ti < 3; ti++) {
            int j = tri_a[ti];
            if (!(ma & (1 << j))) continue;
            double re = a->edge_re[j], im = a->edge_im[j];
            a->edge_re[j] = re * eff_a_re[j] - im * eff_a_im[j];
            a->edge_im[j] = re * eff_a_im[j] + im * eff_a_re[j];
        }
        for (int tk = 0; tk < 3; tk++) {
            int k = tri_b[tk];
            if (!(mb & (1 << k))) continue;
            double re = b->edge_re[k], im = b->edge_im[k];
            b->edge_re[k] = re * eff_b_re[k] - im * eff_b_im[k];
            b->edge_im[k] = re * eff_b_im[k] + im * eff_b_re[k];
        }
        triality_stats.mask_skips += (6 - na) + (6 - nb);
        goto cz_renorm;
    }

    /* Compute effective phases from partner */
    double eff_a_re[TRI_D] = {0}, eff_a_im[TRI_D] = {0};
    double eff_b_re[TRI_D] = {0}, eff_b_im[TRI_D] = {0};

    for (int j = 0; j < TRI_D; j++) {
        if (!(ma & (1 << j))) continue;  /* a[j] is zero */
        double aprob = a->edge_re[j]*a->edge_re[j] + a->edge_im[j]*a->edge_im[j];
        for (int k = 0; k < TRI_D; k++) {
            if (!(mb & (1 << k))) continue;  /* b[k] is zero */
            int idx = (j * k) % 6;
            double bprob = b->edge_re[k]*b->edge_re[k] + b->edge_im[k]*b->edge_im[k];
            eff_a_re[j] += bprob * W6_RE[idx];
            eff_a_im[j] += bprob * W6_IM[idx];
            eff_b_re[k] += aprob * W6_RE[idx];
            eff_b_im[k] += aprob * W6_IM[idx];
        }
    }

    int skipped = 0;
    /* Apply effective phases */
    for (int j = 0; j < TRI_D; j++) {
        if (!(ma & (1 << j))) { skipped++; continue; }
        double re = a->edge_re[j], im = a->edge_im[j];
        a->edge_re[j] = re * eff_a_re[j] - im * eff_a_im[j];
        a->edge_im[j] = re * eff_a_im[j] + im * eff_a_re[j];
    }
    for (int k = 0; k < TRI_D; k++) {
        if (!(mb & (1 << k))) { skipped++; continue; }
        double re = b->edge_re[k], im = b->edge_im[k];
        b->edge_re[k] = re * eff_b_re[k] - im * eff_b_im[k];
        b->edge_im[k] = re * eff_b_im[k] + im * eff_b_re[k];
    }
    triality_stats.mask_skips += skipped;

    /* ── Renormalize after effective-phase CZ ──
     * The effective-phase model computes eff[j] = Σ_k |b_k|² × ω^(jk),
     * then a[j] *= eff[j].  Since |eff[j]| = |Σ |b_k|² ω^(jk)| ≤ 1
     * (strictly < 1 when |b| is not perfectly normalized), floating-point
     * drift causes mutual amplitude drain between both quhits.
     * Renormalization prevents this feedback loop. */
cz_renorm:
    {
        double na = 0, nb = 0;
        for (int i = 0; i < TRI_D; i++) {
            na += a->edge_re[i]*a->edge_re[i] + a->edge_im[i]*a->edge_im[i];
            nb += b->edge_re[i]*b->edge_re[i] + b->edge_im[i]*b->edge_im[i];
        }
        if (na > 1e-30 && fabs(na - 1.0) > 1e-15) {
            double inv = 1.0 / sqrt(na);
            for (int i = 0; i < TRI_D; i++) {
                a->edge_re[i] *= inv; a->edge_im[i] *= inv;
            }
        }
        if (nb > 1e-30 && fabs(nb - 1.0) > 1e-15) {
            double inv = 1.0 / sqrt(nb);
            for (int i = 0; i < TRI_D; i++) {
                b->edge_re[i] *= inv; b->edge_im[i] *= inv;
            }
        }
    }
    a->real_valued = 0; b->real_valued = 0; /* CZ introduces complex phases */
    a->eigenstate_class = -1; b->eigenstate_class = -1;
    a->dirty |= DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED;
    b->dirty |= DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED;
    a->delta_valid = 0; b->delta_valid = 0;  /* Fix #5: invalidate exotic cache */
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

    /* Save the phase of the winning amplitude before zeroing */
    double win_re = v_re[outcome], win_im = v_im[outcome];
    double norm2 = win_re*win_re + win_im*win_im;
    double scale = (norm2 > 1e-30) ? 1.0 / sqrt(norm2) : 0.0;

    for (int k = 0; k < TRI_D; k++) {
        v_re[k] = 0;
        v_im[k] = 0;
    }
    v_re[outcome] = win_re * scale;
    v_im[outcome] = win_im * scale;

    q->primary = view;
    q->dirty = DIRTY_ALL & ~view_dirty_bit(view);

    /* Post-collapse enhancement flags */
    q->active_mask  = (uint8_t)(1 << outcome);
    q->active_count = 1;
    q->real_valued  = (fabs(v_im[outcome]) < 1e-15) ? 1 : 0;
    q->eigenstate_class = -1;
    q->delta_valid = 0;  /* Fix #5 */

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

    /* Rotate dirty bits: bit0→bit1→bit2→bit0 (preserve bit3 = folded) */
    uint8_t d = q->dirty;
    uint8_t lo3 = d & 0x7;
    uint8_t folded = d & DIRTY_FOLDED;
    q->dirty = (((lo3 & 1) << 1) | ((lo3 & 2) << 1) | ((lo3 & 4) >> 2)) | folded | DIRTY_FOLDED;

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
    printf("  | Primary: %s | Dirty: %c%c%c%c | Eigen: %d | Mask: 0x%02X (%d) | Real: %d |\n",
           names[q->primary > 2 ? 0 : q->primary],
           (q->dirty & DIRTY_EDGE) ? 'E' : '·',
           (q->dirty & DIRTY_VERTEX) ? 'V' : '·',
           (q->dirty & DIRTY_DIAGONAL) ? 'D' : '·',
           (q->dirty & DIRTY_FOLDED) ? 'F' : '·',
           q->eigenstate_class,
           q->active_mask, q->active_count,
           q->real_valued);
    printf("  └");
    for (int i = 0; i < 12 + TRI_D * 12; i++) printf("─");
    printf("┘\n");
}


/* ═══════════════════════════════════════════════════════════════════════
 * GEOMETRIC COSMOLOGY ENHANCEMENT FUNCTIONS
 * ═══════════════════════════════════════════════════════════════════════ */

/* ── Enhancement 1: Folded View API ── */

void triality_fold(TrialityQuhit *q) {
    triality_ensure_view(q, VIEW_EDGE);
    fold_forward(q->edge_re, q->edge_im, q->folded_re, q->folded_im);
    q->dirty &= ~DIRTY_FOLDED;
    triality_stats.edge_to_folded++;
}

void triality_unfold(TrialityQuhit *q) {
    if (q->dirty & DIRTY_FOLDED) {
        triality_fold(q);  /* Ensure folded is up to date */
    }
    fold_inverse(q->folded_re, q->folded_im, q->edge_re, q->edge_im);
    q->dirty &= ~DIRTY_EDGE;
    q->dirty |= DIRTY_VERTEX | DIRTY_DIAGONAL;
}

void triality_ensure_view_via_fold(TrialityQuhit *q, int target_view) {
    if (target_view == VIEW_FOLDED) {
        if (q->dirty & DIRTY_FOLDED) triality_fold(q);
        return;
    }
    if (target_view == VIEW_VERTEX) {
        /* Edge → Folded → Vertex: O(18) vs O(36) direct */
        if (q->dirty & DIRTY_FOLDED) triality_fold(q);
        folded_to_vertex(q->folded_re, q->folded_im,
                        q->vertex_re, q->vertex_im);
        q->dirty &= ~DIRTY_VERTEX;
        triality_stats.folded_to_vertex++;
        return;
    }
    /* For other views, fall back to standard */
    triality_ensure_view(q, target_view);
}

/* ── Enhancement 2: Eigenstate Detection ── */

int triality_detect_eigenstate(TrialityQuhit *q) {
    triality_ensure_view(q, VIEW_EDGE);

    /* Compute DFT₆|ψ⟩ and check if F|ψ⟩ = λ|ψ⟩ for some λ ∈ {1,-1,i,-i} */
    double f_re[TRI_D], f_im[TRI_D];
    dft6_forward(q->edge_re, q->edge_im, f_re, f_im);

    /* Find a non-zero component to determine the candidate eigenvalue */
    int ref = -1;
    for (int k = 0; k < TRI_D; k++) {
        double mag = q->edge_re[k]*q->edge_re[k] + q->edge_im[k]*q->edge_im[k];
        if (mag > 1e-20) { ref = k; break; }
    }
    if (ref < 0) { q->eigenstate_class = -1; return -1; } /* zero state */

    /* Candidate λ = F|ψ⟩[ref] / |ψ⟩[ref] */
    double sr = q->edge_re[ref], si = q->edge_im[ref];
    double fr = f_re[ref], fi = f_im[ref];
    double denom = sr*sr + si*si;
    double lr = (fr*sr + fi*si) / denom;
    double li = (fi*sr - fr*si) / denom;

    /* Check which eigenvalue class it matches: {1,-1,i,-i} */
    static const double EV_RE[4] = {1.0, -1.0,  0.0,  0.0};
    static const double EV_IM[4] = {0.0,  0.0,  1.0, -1.0};
    int cls = -1;
    for (int c = 0; c < 4; c++) {
        if (fabs(lr - EV_RE[c]) < 1e-10 && fabs(li - EV_IM[c]) < 1e-10) {
            cls = c; break;
        }
    }
    if (cls < 0) { q->eigenstate_class = -1; return -1; }

    /* Verify ALL components satisfy F|ψ⟩[k] = λ · |ψ⟩[k] */
    for (int k = 0; k < TRI_D; k++) {
        double expected_re = EV_RE[cls] * q->edge_re[k] - EV_IM[cls] * q->edge_im[k];
        double expected_im = EV_RE[cls] * q->edge_im[k] + EV_IM[cls] * q->edge_re[k];
        if (fabs(f_re[k] - expected_re) > 1e-10 ||
            fabs(f_im[k] - expected_im) > 1e-10) {
            q->eigenstate_class = -1;
            return -1;
        }
    }

    q->eigenstate_class = (int8_t)cls;
    return cls;
}

void triality_clear_eigenstate(TrialityQuhit *q) {
    q->eigenstate_class = -1;
}

/* ── Enhancement 3: Subspace Confinement ── */

void triality_update_mask(TrialityQuhit *q) {
    triality_ensure_view(q, VIEW_EDGE);
    uint8_t mask = 0;
    int count = 0;
    for (int k = 0; k < TRI_D; k++) {
        double mag2 = q->edge_re[k]*q->edge_re[k] + q->edge_im[k]*q->edge_im[k];
        if (mag2 > 1e-30) {
            mask |= (uint8_t)(1 << k);
            count++;
        }
    }
    q->active_mask = mask;
    q->active_count = (uint8_t)count;
}

/* ── Enhancement 4: Real-Valued Detection ── */

void triality_detect_real(TrialityQuhit *q) {
    triality_ensure_view(q, VIEW_EDGE);
    q->real_valued = 1;
    for (int k = 0; k < TRI_D; k++) {
        if (fabs(q->edge_im[k]) > 1e-15) {
            q->real_valued = 0;
            return;
        }
    }
}

/* ── Combined refresh ── */

void triality_refresh_flags(TrialityQuhit *q) {
    triality_update_mask(q);
    triality_detect_real(q);
    triality_detect_eigenstate(q);
}

void triality_stats_print(void) {
    printf("\n  ── Triality Statistics ──\n");
    printf("  View conversions:\n");
    printf("    Edge→Vertex:   %llu\n", (unsigned long long)triality_stats.edge_to_vertex);
    printf("    Edge→Diag:     %llu\n", (unsigned long long)triality_stats.edge_to_diag);
    printf("    Vertex→Edge:   %llu\n", (unsigned long long)triality_stats.vertex_to_edge);
    printf("    Vertex→Diag:   %llu\n", (unsigned long long)triality_stats.vertex_to_diag);
    printf("    Diag→Edge:     %llu\n", (unsigned long long)triality_stats.diag_to_edge);
    printf("    Diag→Vertex:   %llu\n", (unsigned long long)triality_stats.diag_to_vertex);
    printf("    Edge→Folded:   %llu\n", (unsigned long long)triality_stats.edge_to_folded);
    printf("    Folded→Vertex: %llu\n", (unsigned long long)triality_stats.folded_to_vertex);
    printf("  Gates:\n");
    printf("    Edge view:     %llu\n", (unsigned long long)triality_stats.gates_edge);
    printf("    Vertex view:   %llu\n", (unsigned long long)triality_stats.gates_vertex);
    printf("    Diag view:     %llu\n", (unsigned long long)triality_stats.gates_diag);
    printf("    Rotations:     %llu\n", (unsigned long long)triality_stats.rotations);
    printf("  Enhancements:\n");
    printf("    Eigenstate skips:  %llu\n", (unsigned long long)triality_stats.eigenstate_skips);
    printf("    Mask skips:        %llu\n", (unsigned long long)triality_stats.mask_skips);
    printf("    Real fast path:    %llu\n", (unsigned long long)triality_stats.real_fast_path);
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

/* ═══════════════════════════════════════════════════════════════════════
 * S₆ OUTER AUTOMORPHISM — EXOTIC EXTENSIONS
 *
 * S₆ is the only symmetric group with a non-trivial outer automorphism.
 * These functions exploit this unique D=6 structure for:
 *   - Parameterized folds (15 synthemes instead of 1)
 *   - Exotic gates (φ(σ) instead of σ)
 *   - Dual measurement (standard + exotic probabilities)
 *   - 6-fold rotation (full Aut(S₆) cycle)
 * ═══════════════════════════════════════════════════════════════════════ */

void triality_exotic_init(void) {
    s6_exotic_init();
}

void triality_set_exotic_syntheme(TrialityQuhit *q, int syntheme_idx) {
    if (syntheme_idx < 0 || syntheme_idx >= S6_NUM_SYNTHEMES)
        syntheme_idx = 0;
    q->exotic_syntheme = syntheme_idx;
    q->dirty |= DIRTY_EXOTIC;
}

/* ── Parameterized Fold/Unfold ──
 * Folds using any of the 15 synthemes.
 * Stores result in the exotic view arrays. */

void triality_fold_syntheme(TrialityQuhit *q, int syntheme_idx) {
    triality_ensure_view(q, VIEW_EDGE);
    s6_fold_syntheme(q->edge_re, q->edge_im,
                     q->exotic_re, q->exotic_im,
                     syntheme_idx);
    q->exotic_syntheme = syntheme_idx;
    q->dirty &= ~DIRTY_EXOTIC;
    triality_stats.exotic_folds++;
}

void triality_unfold_syntheme(TrialityQuhit *q, int syntheme_idx) {
    /* Unfold from exotic arrays back into edge */
    s6_unfold_syntheme(q->exotic_re, q->exotic_im,
                       q->edge_re, q->edge_im,
                       syntheme_idx);
    q->primary = VIEW_EDGE;
    q->dirty = DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED | DIRTY_EXOTIC;
    q->real_valued = 0;
    q->eigenstate_class = -1;
    q->active_mask = 0x3F; q->active_count = 6;
    q->delta_valid = 0;  /* Fix #5 */
    triality_stats.exotic_folds++;
}

/* ── Exotic Gate ──
 * Applies φ(σ) instead of σ. The permutation gate |i⟩ → |σ(i)⟩
 * becomes |i⟩ → |φ(σ)(i)⟩, accessing the exotic representation. */

void triality_exotic_gate(TrialityQuhit *q, S6Perm sigma) {
    triality_ensure_view(q, VIEW_EDGE);
    double out_re[TRI_D], out_im[TRI_D];
    s6_apply_exotic_gate(q->edge_re, q->edge_im,
                         out_re, out_im, sigma);
    memcpy(q->edge_re, out_re, sizeof(out_re));
    memcpy(q->edge_im, out_im, sizeof(out_im));

    q->primary = VIEW_EDGE;
    q->dirty |= DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED | DIRTY_EXOTIC;
    q->real_valued = 0;
    q->eigenstate_class = -1;
    q->active_mask = 0x3F; q->active_count = 6;
    q->delta_valid = 0;  /* Fix #5 */
    triality_stats.exotic_gates++;
}

/* ── Dual CZ ──
 * Performs standard CZ, then computes what the EXOTIC CZ would give.
 * Returns the statistical distance between standard and exotic channels.
 * The exotic CZ uses φ-twisted phase indices.
 *
 * This does NOT modify the state — the exotic channel is observational.
 * The return value tells you how much D=6-specific structure the
 * current state has: 0 = automorphism-invariant, >0 = hexagonal anomaly. */

double triality_cz_dual(TrialityQuhit *a, TrialityQuhit *b) {
    /* First, do the standard CZ */
    triality_cz(a, b);

    /* Fix #1: Compute the REAL exotic invariant Δ for both sites.
     * Instead of the old hardcoded exotic_perm approximation,
     * use s6_exotic_invariant() which sweeps all 720 S₆ permutations
     * and measures the true distance between standard and exotic channels.
     * Returns the average Δ of both sites. */
    triality_ensure_view(a, VIEW_EDGE);
    triality_ensure_view(b, VIEW_EDGE);

    double delta_a = s6_exotic_invariant(a->edge_re, a->edge_im);
    double delta_b = s6_exotic_invariant(b->edge_re, b->edge_im);

    /* Cache the computed invariants */
    a->cached_delta = delta_a;
    a->delta_valid = 1;
    b->cached_delta = delta_b;
    b->delta_valid = 1;

    return (delta_a + delta_b) / 2.0;
}

/* ── Exotic Measurement ──
 * Measures in a syntheme-parameterized basis.
 * Folds the state using the specified syntheme, measures in the
 * folded basis, then applies the appropriate collapse. */

int triality_measure_exotic(TrialityQuhit *q, int syntheme_idx, uint64_t *rng_state) {
    /* Fold into exotic view */
    triality_fold_syntheme(q, syntheme_idx);

    /* Compute probabilities in the folded basis */
    double probs[TRI_D];
    double total = 0;
    for (int k = 0; k < TRI_D; k++) {
        probs[k] = q->exotic_re[k]*q->exotic_re[k]
                  + q->exotic_im[k]*q->exotic_im[k];
        total += probs[k];
    }
    if (total > 1e-30)
        for (int k = 0; k < TRI_D; k++) probs[k] /= total;

    /* Born sample */
    uint64_t x = *rng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *rng_state = x;
    double r = (x >> 11) * 0x1.0p-53;

    int outcome = TRI_D - 1;
    double cdf = 0;
    for (int k = 0; k < TRI_D; k++) {
        cdf += probs[k];
        if (r < cdf) { outcome = k; break; }
    }

    /* Collapse: project onto |outcome⟩ in exotic basis */
    double win_re = q->exotic_re[outcome], win_im = q->exotic_im[outcome];
    double norm2 = win_re*win_re + win_im*win_im;
    double scale = (norm2 > 1e-30) ? 1.0 / sqrt(norm2) : 0.0;

    for (int k = 0; k < TRI_D; k++) {
        q->exotic_re[k] = 0;
        q->exotic_im[k] = 0;
    }
    q->exotic_re[outcome] = win_re * scale;
    q->exotic_im[outcome] = win_im * scale;

    /* Unfold back to edge view */
    triality_unfold_syntheme(q, syntheme_idx);

    triality_stats.exotic_folds++;
    return outcome;
}

/* ── Dual Measurement ──
 * Performs standard measurement AND computes exotic measurement probabilities.
 * Returns standard outcome, sets *exotic_outcome to the exotic result.
 * The exotic outcome is NOT destructive — it's what WOULD have been measured. */

int triality_measure_dual(TrialityQuhit *q, int view, int exotic_syntheme,
                          uint64_t *rng_state, int *exotic_outcome) {
    /* Get exotic probabilities first (non-destructive) */
    triality_ensure_view(q, VIEW_EDGE);
    double exo_fold_re[TRI_D], exo_fold_im[TRI_D];
    s6_fold_syntheme(q->edge_re, q->edge_im,
                     exo_fold_re, exo_fold_im,
                     exotic_syntheme);

    double exo_probs[TRI_D];
    double total = 0;
    for (int k = 0; k < TRI_D; k++) {
        exo_probs[k] = exo_fold_re[k]*exo_fold_re[k]
                      + exo_fold_im[k]*exo_fold_im[k];
        total += exo_probs[k];
    }
    if (total > 1e-30)
        for (int k = 0; k < TRI_D; k++) exo_probs[k] /= total;

    /* Sample exotic outcome (no collapse) */
    uint64_t rng_copy = *rng_state;
    uint64_t x = rng_copy;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    rng_copy = x;
    double r_exo = (x >> 11) * 0x1.0p-53;

    int exo_out = TRI_D - 1;
    double cdf = 0;
    for (int k = 0; k < TRI_D; k++) {
        cdf += exo_probs[k];
        if (r_exo < cdf) { exo_out = k; break; }
    }
    if (exotic_outcome) *exotic_outcome = exo_out;

    /* Standard measurement (destructive) */
    int std_out = triality_measure(q, view, rng_state);

    triality_stats.dual_measurements++;
    return std_out;
}

/* ── 6-fold Rotation ──
 * Standard rotation cycles 3 views: Edge→Vertex→Diagonal→Edge.
 * Exotic rotation ALSO cycles the exotic syntheme through its
 * synthematic total, accessing all 5 synthemes in the total.
 *
 * There are 6 totals × 5 synthemes = 30 possible exotic states.
 * Cycling through synthemes within a total = inner rotation.
 * Switching to a different total = outer rotation.
 *
 * This function does inner rotation: standard 3-view rotate +
 * advance exotic syntheme to the next in its total. */

void triality_rotate_exotic(TrialityQuhit *q) {
    /* Standard triality rotation */
    triality_rotate(q);

    /* Advance exotic syntheme within its total */
    int current = q->exotic_syntheme;

    /* Find which total this syntheme belongs to */
    if (!s6_exotic_ready) s6_exotic_init();

    int my_total = -1, my_pos = -1;
    for (int t = 0; t < S6_NUM_TOTALS && my_total < 0; t++) {
        for (int j = 0; j < 5; j++) {
            if (s6_totals[t][j] == current) {
                my_total = t;
                my_pos = j;
                break;
            }
        }
    }

    if (my_total >= 0) {
        /* Advance to next syntheme in this total */
        int next_pos = (my_pos + 1) % 5;
        q->exotic_syntheme = s6_totals[my_total][next_pos];
    }

    q->dirty |= DIRTY_EXOTIC;
}

/* ── Dual Probabilities ──
 * Non-destructive: returns probabilities in both the standard view
 * and the exotic (syntheme-folded) basis. */

void triality_dual_probabilities(TrialityQuhit *q, int view,
                                 double *probs_std, double *probs_exo) {
    /* Standard probabilities */
    triality_probabilities(q, view, probs_std);

    /* Exotic probabilities: fold edge by exotic syntheme */
    triality_ensure_view(q, VIEW_EDGE);
    double fold_re[TRI_D], fold_im[TRI_D];
    s6_fold_syntheme(q->edge_re, q->edge_im,
                     fold_re, fold_im,
                     q->exotic_syntheme);

    double total = 0;
    for (int k = 0; k < TRI_D; k++) {
        probs_exo[k] = fold_re[k]*fold_re[k] + fold_im[k]*fold_im[k];
        total += probs_exo[k];
    }
    if (total > 1e-30)
        for (int k = 0; k < TRI_D; k++) probs_exo[k] /= total;
}

/* ═══════════════════════════════════════════════════════════════════════
 * FIX #5: CACHED EXOTIC INVARIANT
 *
 * The exotic invariant Δ is a pure function of the state vector.
 * If the state hasn't changed since the last computation, return
 * the cached value. This avoids the 720-permutation sweep.
 * ═══════════════════════════════════════════════════════════════════════ */

void triality_invalidate_exotic_cache(TrialityQuhit *q) {
    q->delta_valid = 0;
}

double triality_exotic_invariant_cached(TrialityQuhit *q) {
    if (q->delta_valid) return q->cached_delta;

    triality_ensure_view(q, VIEW_EDGE);
    q->cached_delta = s6_exotic_invariant(q->edge_re, q->edge_im);

    /* Also compute the full fingerprint while we're at it */
    s6_exotic_fingerprint(q->edge_re, q->edge_im, q->cached_fingerprint);

    q->delta_valid = 1;
    return q->cached_delta;
}

void triality_exotic_fingerprint_cached(TrialityQuhit *q, double *deltas) {
    if (q->delta_valid) {
        /* Cache is valid — just copy */
        memcpy(deltas, q->cached_fingerprint, 11 * sizeof(double));
        return;
    }

    triality_ensure_view(q, VIEW_EDGE);
    s6_exotic_fingerprint(q->edge_re, q->edge_im, deltas);
    q->cached_delta = s6_exotic_invariant(q->edge_re, q->edge_im);
    memcpy(q->cached_fingerprint, deltas, 11 * sizeof(double));
    q->delta_valid = 1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * FIX #7: LAZY QUHIT FORCE MATERIALIZE
 *
 * Compiles the accumulated lazy chain into a TrialityQuhit with
 * edge amplitudes populated and all flags set. Use when a two-body
 * operation (CZ) needs actual state data from a lazy quhit.
 * ═══════════════════════════════════════════════════════════════════════ */

void ltri_force_materialize(LazyTrialityQuhit *lq, TrialityQuhit *out) {
    double edge_re[TRI_D], edge_im[TRI_D];
    ltri_materialize(lq, edge_re, edge_im);

    /* Initialize the output quhit with the materialized edge amplitudes */
    memset(out, 0, sizeof(TrialityQuhit));
    memcpy(out->edge_re, edge_re, sizeof(edge_re));
    memcpy(out->edge_im, edge_im, sizeof(edge_im));
    out->primary = VIEW_EDGE;
    out->dirty = DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED | DIRTY_EXOTIC;
    out->eigenstate_class = -1;
    out->delta_valid = 0;

    /* Compute actual enhancement flags from the materialized state */
    uint8_t mask = 0;
    int count = 0;
    int is_real = 1;
    for (int k = 0; k < TRI_D; k++) {
        double mag2 = edge_re[k] * edge_re[k] + edge_im[k] * edge_im[k];
        if (mag2 > 1e-30) {
            mask |= (uint8_t)(1 << k);
            count++;
        }
        if (fabs(edge_im[k]) > 1e-15) is_real = 0;
    }
    out->active_mask = mask;
    out->active_count = (uint8_t)count;
    out->real_valued = (uint8_t)is_real;
}
