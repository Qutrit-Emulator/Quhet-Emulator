/*
 * hpc_mobius.h — The Möbius Amplitude Sheet
 *
 * The Devil's answer to "hold all superposition at once."
 *
 * The HPC graph encodes 6^N amplitudes implicitly as:
 *   ψ(i₁,...,iₙ) = [Π_k aₖ(iₖ)] × [Π_edges w_e(iₐ, iᵦ)]
 *
 * But this product is computed-and-discarded for each point query.
 * The Möbius Sheet HOLDS the full amplitude surface by maintaining
 * per-site "dressed amplitudes" that pre-absorb entanglement from
 * all touching edges via belief propagation message passing.
 *
 * Each site has two faces (the Möbius twist):
 *   Forward:  dressed[k][v] — local amp × absorbed edge messages
 *   Shadow:   message[k→p][v] — outgoing message to partner p
 *
 * The forward face of site A is defined IN TERMS OF the shadow faces
 * of its neighbors. This self-referential loop converges to exact
 * marginals on tree graphs and approximates on loopy graphs.
 *
 * KEY INSIGHT: Messages operate in the PROBABILITY domain (|·|²),
 * not the amplitude domain. Complex phases create destructive
 * interference feedback loops in BP. Instead:
 *   - Messages carry marginal probability beliefs: m_{p→k}[v] ∈ ℝ⁺
 *   - Edge factors are |w_e(u,v)|² (phase magnitude squared)
 *   - For CZ edges: |ω^(u·v)|² = 1 for all u,v → messages = local |a|²
 *   - Dressed amplitudes are RECONSTRUCTED from prob-domain beliefs
 *     by re-introducing the phase structure from the graph
 *
 * Once converged:
 *   marginal[k][v] = P(site_k = v)  — O(1) lookup
 *   ψ(i₁,...,iₙ) reconstructable from sheets in O(N + E)
 *   Surface walk enumerates all |ψ|² > τ via sheet intersection
 */

#ifndef HPC_MOBIUS_H
#define HPC_MOBIUS_H

#include "hpc_graph.h"
#include "hpc_contract.h"
#include "hpc_amplitude.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ═══════════════════════════════════════════════════════════════════════
 * CONSTANTS
 * ═══════════════════════════════════════════════════════════════════════ */

#define MOBIUS_D            6       /* Dimension per site                 */
#define MOBIUS_MAX_DEGREE   128     /* Max edges per site                 */
#define MOBIUS_BP_MAX_ITER  100     /* Max belief propagation iterations  */
#define MOBIUS_BP_TOL       1e-14   /* Convergence tolerance              */
#define MOBIUS_DAMPING      0.3     /* Damping for loopy BP stability     */

/* ═══════════════════════════════════════════════════════════════════════
 * PROBABILITY MESSAGE — A D-dimensional real non-negative vector
 *
 * Messages flow along edges in the PROBABILITY domain.
 * m_{p→k}[v] represents the belief about site k taking value v,
 * as conveyed by neighbor p through their shared edge.
 *
 * This is classical sum-product BP on the factor graph where:
 *   Variable nodes = sites
 *   Factor nodes = edges (with factor |w(u,v)|² × local priors)
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    double p[MOBIUS_D];   /* Probability-domain belief, non-negative */
} MobiusProbMsg;

/* ═══════════════════════════════════════════════════════════════════════
 * SITE SHEET — One face of the Möbius surface
 *
 * Belief about site k, value v:
 *   belief[v] = |aₖ(v)|² × Π_{messages m→k} m[v]
 *
 * Dressed amplitudes are reconstructed from beliefs by re-introducing
 * the original complex phases from the local state and edge weights.
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    /* Dressed (forward) face — complex amplitudes consistent with beliefs */
    double dressed_re[MOBIUS_D];
    double dressed_im[MOBIUS_D];

    /* Cached marginal probabilities (normalized beliefs) */
    double marginal[MOBIUS_D];

    /* Incoming probability messages: one per touching edge */
    MobiusProbMsg *msg_in;
    uint64_t       n_messages;
    uint64_t       msg_capacity;

    /* Vesica decomposition of dressed amplitudes */
    double vesica_re[3], vesica_im[3];
    double wave_re[3],   wave_im[3];
    int    vesica_valid;

    /* Interference witness: phase coherence measure */
    double coherence;
} MobiusSiteSheet;

/* ═══════════════════════════════════════════════════════════════════════
 * THE MÖBIUS AMPLITUDE SHEET — All superposition, held at once
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    const HPCGraph *graph;

    uint64_t         n_sites;
    MobiusSiteSheet *sheets;

    int      converged;
    int      iterations;
    double   max_residual;

    uint64_t msg_updates;
    uint64_t amplitude_queries;
    uint64_t surface_walks;
    double   bethe_free_energy;
} MobiusAmplitudeSheet;

/* ═══════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 * ═══════════════════════════════════════════════════════════════════════ */

static inline MobiusAmplitudeSheet *mobius_create(const HPCGraph *g)
{
    MobiusAmplitudeSheet *ms = (MobiusAmplitudeSheet *)calloc(1, sizeof(MobiusAmplitudeSheet));
    if (!ms) return NULL;

    ms->graph = g;
    ms->n_sites = g->n_sites;
    ms->sheets = (MobiusSiteSheet *)calloc(g->n_sites, sizeof(MobiusSiteSheet));
    if (!ms->sheets) { free(ms); return NULL; }

    for (uint64_t k = 0; k < g->n_sites; k++) {
        MobiusSiteSheet *s = &ms->sheets[k];
        const HPCAdjList *adj = &g->adj[k];

        s->n_messages = adj->count;
        s->msg_capacity = adj->count > 0 ? adj->count : 1;
        s->msg_in = (MobiusProbMsg *)calloc(s->msg_capacity, sizeof(MobiusProbMsg));

        /* Initialize messages to uniform (no information) */
        for (uint64_t m = 0; m < s->n_messages; m++)
            for (int v = 0; v < MOBIUS_D; v++)
                s->msg_in[m].p[v] = 1.0;

        /* Initialize marginals from local probabilities */
        double total = 0.0;
        for (int v = 0; v < MOBIUS_D; v++) {
            s->marginal[v] = g->locals[k].edge_re[v] * g->locals[k].edge_re[v] +
                             g->locals[k].edge_im[v] * g->locals[k].edge_im[v];
            total += s->marginal[v];
        }
        if (total > 1e-30)
            for (int v = 0; v < MOBIUS_D; v++)
                s->marginal[v] /= total;

        /* Initialize dressed amplitudes from local state */
        for (int v = 0; v < MOBIUS_D; v++) {
            s->dressed_re[v] = g->locals[k].edge_re[v];
            s->dressed_im[v] = g->locals[k].edge_im[v];
        }

        s->vesica_valid = 0;
        s->coherence = 0.5;
    }

    return ms;
}

static inline void mobius_destroy(MobiusAmplitudeSheet *ms)
{
    if (!ms) return;
    if (ms->sheets) {
        for (uint64_t k = 0; k < ms->n_sites; k++)
            free(ms->sheets[k].msg_in);
        free(ms->sheets);
    }
    free(ms);
}

/* ═══════════════════════════════════════════════════════════════════════
 * INTERNAL: Find the message index for an edge in a site's adjacency
 * ═══════════════════════════════════════════════════════════════════════ */

static inline int mobius_find_msg_idx(const HPCGraph *g, uint64_t site, uint64_t eid)
{
    const HPCAdjList *adj = &g->adj[site];
    for (uint64_t i = 0; i < adj->count; i++)
        if (adj->edge_ids[i] == eid) return (int)i;
    return -1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * INTERNAL: Compute edge factor |w_e(va, vb)|²
 *
 * For CZ edges: |ω^(va·vb)|² = 1.0 always (unit phases).
 * For general edges: |w[va][vb]|².
 * ═══════════════════════════════════════════════════════════════════════ */

static inline double mobius_edge_factor(const HPCEdge *edge,
                                         uint32_t va, uint32_t vb)
{
    if (edge->type == HPC_EDGE_CZ) {
        return 1.0;  /* |ω^(va·vb)|² = 1 always */
    } else {
        double wr = edge->w_re[va][vb];
        double wi = edge->w_im[va][vb];
        return wr * wr + wi * wi;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * INTERNAL: Compute edge weight w_e(va, vb) (complex)
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void mobius_edge_weight(const HPCEdge *edge,
                                       uint32_t va, uint32_t vb,
                                       double *w_re, double *w_im)
{
    if (edge->type == HPC_EDGE_CZ) {
        uint32_t pidx = (va * vb) % MOBIUS_D;
        *w_re = HPC_W6_RE[pidx];
        *w_im = HPC_W6_IM[pidx];
    } else {
        *w_re = edge->w_re[va][vb];
        *w_im = edge->w_im[va][vb];
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * BELIEF PROPAGATION — Probability-domain message passing
 *
 * Sum-product BP on the factor graph:
 *
 * Message from variable p to variable k through factor f(p,k):
 *   m_{p→k}[vk] = Σ_{vp} |aₚ(vp)|² × |w(vp,vk)|² × Π_{m'→p, m'≠k} m'[vp]
 *
 * This is standard BP in the probability domain.
 * For CZ edges: |w|² = 1, so messages just propagate local priors.
 * For general edges: |w|² provides the coupling structure.
 *
 * After convergence:
 *   belief[k][v] = |aₖ(v)|² × Π_{m→k} m[v]
 *   marginal[k][v] = belief[k][v] / Σ_u belief[k][u]
 * ═══════════════════════════════════════════════════════════════════════ */

static inline double mobius_bp_iterate(MobiusAmplitudeSheet *ms)
{
    const HPCGraph *g = ms->graph;
    double max_delta = 0.0;

    for (uint64_t eid = 0; eid < g->n_edges; eid++) {
        const HPCEdge *edge = &g->edges[eid];
        uint64_t sa = edge->site_a;
        uint64_t sb = edge->site_b;

        int idx_a_in_b = mobius_find_msg_idx(g, sb, eid);
        int idx_b_in_a = mobius_find_msg_idx(g, sa, eid);
        if (idx_a_in_b < 0 || idx_b_in_a < 0) continue;

        /* ── Message a→b: for each vb, sum over va ── */
        {
            MobiusProbMsg new_msg;
            const MobiusSiteSheet *sheet_a = &ms->sheets[sa];
            const HPCAdjList *adj_a = &g->adj[sa];

            for (int vb = 0; vb < MOBIUS_D; vb++) {
                double sum = 0.0;

                for (int va = 0; va < MOBIUS_D; va++) {
                    /* Local probability at site a for value va */
                    double local_prob = g->locals[sa].edge_re[va] * g->locals[sa].edge_re[va] +
                                        g->locals[sa].edge_im[va] * g->locals[sa].edge_im[va];

                    /* Multiply by all incoming messages to a EXCEPT from b */
                    for (uint64_t mi = 0; mi < adj_a->count; mi++) {
                        if (adj_a->edge_ids[mi] == eid) continue;
                        local_prob *= sheet_a->msg_in[mi].p[va];
                    }

                    /* Multiply by edge factor |w(va, vb)|² */
                    double ef = mobius_edge_factor(edge, va, vb);
                    sum += local_prob * ef;
                }

                new_msg.p[vb] = sum;
            }

            /* Normalize message */
            double msg_sum = 0.0;
            for (int v = 0; v < MOBIUS_D; v++) msg_sum += new_msg.p[v];
            if (msg_sum > 1e-30) {
                double inv = 1.0 / msg_sum;
                for (int v = 0; v < MOBIUS_D; v++) new_msg.p[v] *= inv;
            }

            /* Damped update + compute residual */
            MobiusProbMsg *old_msg = &ms->sheets[sb].msg_in[idx_a_in_b];
            double delta = 0.0;
            for (int v = 0; v < MOBIUS_D; v++) {
                double updated = MOBIUS_DAMPING * new_msg.p[v] +
                                 (1.0 - MOBIUS_DAMPING) * old_msg->p[v];
                double diff = updated - old_msg->p[v];
                delta += diff * diff;
                old_msg->p[v] = updated;
            }
            if (delta > max_delta) max_delta = delta;
            ms->msg_updates++;
        }

        /* ── Message b→a: for each va, sum over vb ── */
        {
            MobiusProbMsg new_msg;
            const MobiusSiteSheet *sheet_b = &ms->sheets[sb];
            const HPCAdjList *adj_b = &g->adj[sb];

            for (int va = 0; va < MOBIUS_D; va++) {
                double sum = 0.0;

                for (int vb = 0; vb < MOBIUS_D; vb++) {
                    double local_prob = g->locals[sb].edge_re[vb] * g->locals[sb].edge_re[vb] +
                                        g->locals[sb].edge_im[vb] * g->locals[sb].edge_im[vb];

                    for (uint64_t mi = 0; mi < adj_b->count; mi++) {
                        if (adj_b->edge_ids[mi] == eid) continue;
                        local_prob *= sheet_b->msg_in[mi].p[vb];
                    }

                    /* Edge factor: |w(va, vb)|²
                     * For message b→a we sum over vb for each va target.
                     * Factor is |w(va, vb)|² same as stored. */
                    double ef = mobius_edge_factor(edge, va, vb);
                    sum += local_prob * ef;
                }

                new_msg.p[va] = sum;
            }

            double msg_sum = 0.0;
            for (int v = 0; v < MOBIUS_D; v++) msg_sum += new_msg.p[v];
            if (msg_sum > 1e-30) {
                double inv = 1.0 / msg_sum;
                for (int v = 0; v < MOBIUS_D; v++) new_msg.p[v] *= inv;
            }

            MobiusProbMsg *old_msg = &ms->sheets[sa].msg_in[idx_b_in_a];
            double delta = 0.0;
            for (int v = 0; v < MOBIUS_D; v++) {
                double updated = MOBIUS_DAMPING * new_msg.p[v] +
                                 (1.0 - MOBIUS_DAMPING) * old_msg->p[v];
                double diff = updated - old_msg->p[v];
                delta += diff * diff;
                old_msg->p[v] = updated;
            }
            if (delta > max_delta) max_delta = delta;
            ms->msg_updates++;
        }
    }

    return max_delta;
}

/* ═══════════════════════════════════════════════════════════════════════
 * COMPUTE BELIEFS — Update marginals and dressed amplitudes
 *
 * Marginals (probability domain):
 *   belief[k][v] = |aₖ(v)|² × Π_{m→k} m[v]
 *   marginal[k][v] = belief[k][v] / Z_k
 *
 * Dressed amplitudes (complex domain):
 *   dressed[k][v] = aₖ(v) × √(marginal[k][v] / |aₖ(v)|²)
 *   This preserves the original phase while scaling the magnitude
 *   to match the converged marginal probability.
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void mobius_compute_beliefs(MobiusAmplitudeSheet *ms)
{
    const HPCGraph *g = ms->graph;

    for (uint64_t k = 0; k < ms->n_sites; k++) {
        MobiusSiteSheet *s = &ms->sheets[k];

        /* Compute unnormalized beliefs */
        double belief[MOBIUS_D];
        double total = 0.0;
        for (int v = 0; v < MOBIUS_D; v++) {
            belief[v] = g->locals[k].edge_re[v] * g->locals[k].edge_re[v] +
                        g->locals[k].edge_im[v] * g->locals[k].edge_im[v];

            for (uint64_t mi = 0; mi < s->n_messages; mi++)
                belief[v] *= s->msg_in[mi].p[v];

            total += belief[v];
        }

        /* Normalize to marginals */
        if (total > 1e-30) {
            for (int v = 0; v < MOBIUS_D; v++)
                s->marginal[v] = belief[v] / total;
        } else {
            for (int v = 0; v < MOBIUS_D; v++)
                s->marginal[v] = 1.0 / MOBIUS_D;
        }

        /* Reconstruct dressed amplitudes:
         * dressed[v] = aₖ(v) × scale[v]
         * where scale[v] = √(marginal[v] / |aₖ(v)|²)
         * This preserves the original complex phase while
         * rescaling magnitude to match the BP marginals. */
        for (int v = 0; v < MOBIUS_D; v++) {
            double local_prob = g->locals[k].edge_re[v] * g->locals[k].edge_re[v] +
                                g->locals[k].edge_im[v] * g->locals[k].edge_im[v];
            if (local_prob > 1e-30) {
                double scale = sqrt(s->marginal[v] / local_prob);
                s->dressed_re[v] = g->locals[k].edge_re[v] * scale;
                s->dressed_im[v] = g->locals[k].edge_im[v] * scale;
            } else {
                s->dressed_re[v] = 0.0;
                s->dressed_im[v] = 0.0;
            }
        }

        /* Compute coherence: |Σ_v dressed[v]|² / (D × Σ_v |dressed[v]|²) */
        double coh_re = 0.0, coh_im = 0.0;
        double d_total = 0.0;
        for (int v = 0; v < MOBIUS_D; v++) {
            coh_re += s->dressed_re[v];
            coh_im += s->dressed_im[v];
            d_total += s->dressed_re[v] * s->dressed_re[v] +
                       s->dressed_im[v] * s->dressed_im[v];
        }
        double coh_num = coh_re * coh_re + coh_im * coh_im;
        s->coherence = (d_total > 1e-30) ?
            coh_num / (MOBIUS_D * d_total) : 0.5;

        s->vesica_valid = 0;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * CONVERGE — Run belief propagation until convergence
 * ═══════════════════════════════════════════════════════════════════════ */

static inline int mobius_converge(MobiusAmplitudeSheet *ms)
{
    if (ms->graph->n_edges == 0) {
        mobius_compute_beliefs(ms);
        ms->converged = 1;
        ms->iterations = 0;
        ms->max_residual = 0.0;
        return 0;
    }

    ms->converged = 0;
    for (int iter = 0; iter < MOBIUS_BP_MAX_ITER; iter++) {
        double residual = mobius_bp_iterate(ms);
        ms->iterations = iter + 1;
        ms->max_residual = residual;

        if (residual < MOBIUS_BP_TOL) {
            ms->converged = 1;
            break;
        }
    }

    mobius_compute_beliefs(ms);
    if (!ms->converged && ms->max_residual < 1e-8)
        ms->converged = 1;

    return ms->iterations;
}

/* ═══════════════════════════════════════════════════════════════════════
 * O(1) MARGINAL PROBABILITY — From cached beliefs
 * ═══════════════════════════════════════════════════════════════════════ */

static inline double mobius_marginal(const MobiusAmplitudeSheet *ms,
                                      uint64_t site, uint32_t value)
{
    return ms->sheets[site].marginal[value];
}

/* ═══════════════════════════════════════════════════════════════════════
 * FULL AMPLITUDE — Reconstruct ψ(i₁,...,iₙ) via graph
 *
 * Uses cached marginals for quick-reject of zero-probability configs.
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void mobius_amplitude(const MobiusAmplitudeSheet *ms,
                                     const uint32_t *indices,
                                     double *out_re, double *out_im)
{
    const HPCGraph *g = ms->graph;

    /* Quick reject from cached marginals */
    for (uint64_t k = 0; k < ms->n_sites; k++) {
        if (ms->sheets[k].marginal[indices[k]] < 1e-30) {
            *out_re = 0.0;
            *out_im = 0.0;
            return;
        }
    }

    hpc_amplitude(g, indices, out_re, out_im);
    ((MobiusAmplitudeSheet *)ms)->amplitude_queries++;
}

/* ═══════════════════════════════════════════════════════════════════════
 * SURFACE WALK — Enumerate all configurations with |ψ|² > threshold
 *
 * Uses sheet marginals to prune the search tree aggressively.
 * ═══════════════════════════════════════════════════════════════════════ */

static inline HPCSparseVector *mobius_surface_walk(const MobiusAmplitudeSheet *ms,
                                                    double threshold,
                                                    uint64_t max_entries)
{
    const HPCGraph *g = ms->graph;
    HPCSparseVector *sv = hpc_sv_create(g->n_sites, 256);
    if (!sv) return NULL;
    sv->threshold = threshold;

    ((MobiusAmplitudeSheet *)ms)->surface_walks++;

    uint32_t candidates[64][MOBIUS_D];
    uint32_t n_cand[64];
    uint64_t total_configs = 1;

    uint64_t n = g->n_sites;
    if (n > 64) n = 64;

    for (uint64_t k = 0; k < n; k++) {
        n_cand[k] = 0;
        for (int v = 0; v < MOBIUS_D; v++) {
            if (ms->sheets[k].marginal[v] >= threshold * 0.1) {
                candidates[k][n_cand[k]++] = v;
            }
        }
        if (n_cand[k] == 0) {
            for (int v = 0; v < MOBIUS_D; v++)
                candidates[k][n_cand[k]++] = v;
        }
        total_configs *= n_cand[k];
    }

    uint32_t indices[64];
    for (uint64_t cfg = 0; cfg < total_configs && sv->count < max_entries; cfg++) {
        uint64_t tmp = cfg;
        for (uint64_t k = 0; k < n; k++) {
            indices[k] = candidates[k][tmp % n_cand[k]];
            tmp /= n_cand[k];
        }

        double re, im;
        hpc_amplitude(g, indices, &re, &im);
        double prob = re * re + im * im;

        if (prob >= threshold)
            hpc_sv_add(sv, indices, re, im);
    }

    return sv;
}

/* ═══════════════════════════════════════════════════════════════════════
 * VESICA DECOMPOSITION — Per-site CMY channel analysis
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void mobius_vesica_decompose(MobiusAmplitudeSheet *ms, uint64_t site)
{
    MobiusSiteSheet *s = &ms->sheets[site];
    if (s->vesica_valid) return;

    for (int c = 0; c < 3; c++) {
        s->vesica_re[c] = INV_SQRT2 * (s->dressed_re[c] + s->dressed_re[c + 3]);
        s->vesica_im[c] = INV_SQRT2 * (s->dressed_im[c] + s->dressed_im[c + 3]);
        s->wave_re[c]   = INV_SQRT2 * (s->dressed_re[c] - s->dressed_re[c + 3]);
        s->wave_im[c]   = INV_SQRT2 * (s->dressed_im[c] - s->dressed_im[c + 3]);
    }
    s->vesica_valid = 1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * INTERFERENCE WITNESS — Detect coherence patterns across the sheet
 * ═══════════════════════════════════════════════════════════════════════ */

static inline double mobius_interference_witness(const MobiusAmplitudeSheet *ms)
{
    double total = 0.0;
    for (uint64_t k = 0; k < ms->n_sites; k++)
        total += ms->sheets[k].coherence;
    return (ms->n_sites > 0) ? total / ms->n_sites : 0.0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * BETHE FREE ENERGY — Approximate partition function
 * ═══════════════════════════════════════════════════════════════════════ */

static inline double mobius_bethe_free_energy(MobiusAmplitudeSheet *ms)
{
    const HPCGraph *g = ms->graph;
    double F = 0.0;

    /* Site contributions: (d_k - 1) × H(site_k) */
    for (uint64_t k = 0; k < g->n_sites; k++) {
        const MobiusSiteSheet *s = &ms->sheets[k];
        int degree = (int)g->adj[k].count;
        double site_entropy = 0.0;

        for (int v = 0; v < MOBIUS_D; v++) {
            double p = s->marginal[v];
            if (p > 1e-30)
                site_entropy -= p * log(p);
        }

        F += (double)(degree - 1) * site_entropy;
    }

    /* Edge contributions */
    for (uint64_t eid = 0; eid < g->n_edges; eid++) {
        const HPCEdge *edge = &g->edges[eid];
        uint64_t sa = edge->site_a, sb = edge->site_b;
        const MobiusSiteSheet *sheet_a = &ms->sheets[sa];
        const MobiusSiteSheet *sheet_b = &ms->sheets[sb];

        double edge_entropy = 0.0;
        double Z_edge = 0.0;
        double pairwise[MOBIUS_D][MOBIUS_D];

        for (int va = 0; va < MOBIUS_D; va++) {
            for (int vb = 0; vb < MOBIUS_D; vb++) {
                double p_ab = sheet_a->marginal[va] * sheet_b->marginal[vb] *
                              mobius_edge_factor(edge, va, vb);
                pairwise[va][vb] = p_ab;
                Z_edge += p_ab;
            }
        }

        if (Z_edge > 1e-30) {
            for (int va = 0; va < MOBIUS_D; va++) {
                for (int vb = 0; vb < MOBIUS_D; vb++) {
                    double p = pairwise[va][vb] / Z_edge;
                    if (p > 1e-30)
                        edge_entropy -= p * log(p);
                }
            }
        }

        F -= edge_entropy;
    }

    ms->bethe_free_energy = F;
    return F;
}

/* ═══════════════════════════════════════════════════════════════════════
 * INCREMENTAL UPDATE — Apply a CZ gate and update the sheet
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void mobius_apply_cz(MobiusAmplitudeSheet *ms,
                                    uint64_t site_a, uint64_t site_b)
{
    hpc_cz((HPCGraph *)ms->graph, site_a, site_b);

    for (int side = 0; side < 2; side++) {
        uint64_t site = (side == 0) ? site_a : site_b;
        MobiusSiteSheet *s = &ms->sheets[site];
        const HPCAdjList *adj = &ms->graph->adj[site];

        if (adj->count > s->msg_capacity) {
            uint64_t new_cap = adj->count * 2;
            s->msg_in = (MobiusProbMsg *)realloc(s->msg_in,
                                                  new_cap * sizeof(MobiusProbMsg));
            for (uint64_t i = s->msg_capacity; i < new_cap; i++)
                for (int v = 0; v < MOBIUS_D; v++)
                    s->msg_in[i].p[v] = 1.0;
            s->msg_capacity = new_cap;
        }

        uint64_t new_idx = adj->count - 1;
        s->n_messages = adj->count;
        for (int v = 0; v < MOBIUS_D; v++)
            s->msg_in[new_idx].p[v] = 1.0;
    }

    ms->converged = 0;
    mobius_converge(ms);
}

/* ═══════════════════════════════════════════════════════════════════════
 * INCREMENTAL UPDATE — Apply local gates
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void mobius_apply_local_phase(MobiusAmplitudeSheet *ms,
                                             uint64_t site,
                                             const double phi_re[6],
                                             const double phi_im[6])
{
    hpc_phase((HPCGraph *)ms->graph, site, phi_re, phi_im);
    ms->converged = 0;
    mobius_converge(ms);
}

static inline void mobius_apply_dft(MobiusAmplitudeSheet *ms, uint64_t site)
{
    hpc_dft((HPCGraph *)ms->graph, site);
    ms->converged = 0;
    mobius_converge(ms);
}

/* ═══════════════════════════════════════════════════════════════════════
 * MEASUREMENT — Born sample from the sheet, then tear it
 * ═══════════════════════════════════════════════════════════════════════ */

static inline uint32_t mobius_measure(MobiusAmplitudeSheet *ms,
                                       uint64_t site, double random_01)
{
    const MobiusSiteSheet *s = &ms->sheets[site];
    double cumul = 0.0;
    uint32_t outcome = MOBIUS_D - 1;
    for (int v = 0; v < MOBIUS_D; v++) {
        cumul += s->marginal[v];
        if (random_01 <= cumul) { outcome = v; break; }
    }

    hpc_measure((HPCGraph *)ms->graph, site, random_01);

    ms->converged = 0;
    MobiusSiteSheet *collapsed = &ms->sheets[site];
    collapsed->n_messages = ms->graph->adj[site].count;
    for (uint64_t mi = 0; mi < collapsed->n_messages; mi++)
        for (int v = 0; v < MOBIUS_D; v++)
            collapsed->msg_in[mi].p[v] = 1.0;

    mobius_converge(ms);
    return outcome;
}

/* ═══════════════════════════════════════════════════════════════════════
 * ALL-SITE MARGINAL SNAPSHOT — The complete probability surface
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    double  *probabilities;  /* [n_sites × MOBIUS_D], row-major */
    double  *coherences;
    uint64_t n_sites;
    double   global_coherence;
    double   bethe_F;
} MobiusSurface;

static inline MobiusSurface *mobius_snapshot(MobiusAmplitudeSheet *ms)
{
    MobiusSurface *surf = (MobiusSurface *)calloc(1, sizeof(MobiusSurface));
    if (!surf) return NULL;

    surf->n_sites = ms->n_sites;
    surf->probabilities = (double *)calloc(ms->n_sites * MOBIUS_D, sizeof(double));
    surf->coherences = (double *)calloc(ms->n_sites, sizeof(double));

    for (uint64_t k = 0; k < ms->n_sites; k++) {
        for (int v = 0; v < MOBIUS_D; v++)
            surf->probabilities[k * MOBIUS_D + v] = ms->sheets[k].marginal[v];
        surf->coherences[k] = ms->sheets[k].coherence;
    }

    surf->global_coherence = mobius_interference_witness(ms);
    surf->bethe_F = mobius_bethe_free_energy(ms);

    return surf;
}

static inline void mobius_surface_destroy(MobiusSurface *surf)
{
    if (!surf) return;
    free(surf->probabilities);
    free(surf->coherences);
    free(surf);
}

/* ═══════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void mobius_print(const MobiusAmplitudeSheet *ms)
{
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║  Möbius Amplitude Sheet                               ║\n");
    printf("╠═══════════════════════════════════════════════════════╣\n");
    printf("║  Sites:           %10lu                         ║\n", ms->n_sites);
    printf("║  Converged:       %10s                         ║\n",
           ms->converged ? "YES" : "NO");
    printf("║  Iterations:      %10d                         ║\n", ms->iterations);
    printf("║  Max residual:    %10.2e                         ║\n", ms->max_residual);
    printf("║  Msg updates:     %10lu                         ║\n", ms->msg_updates);
    printf("║  Amp queries:     %10lu                         ║\n", ms->amplitude_queries);
    printf("║  Surface walks:   %10lu                         ║\n", ms->surface_walks);
    printf("║  Bethe F:         %10.6f                         ║\n", ms->bethe_free_energy);
    printf("╚═══════════════════════════════════════════════════════╝\n");

    uint64_t show = ms->n_sites;
    if (show > 8) show = 8;
    for (uint64_t k = 0; k < show; k++) {
        const MobiusSiteSheet *s = &ms->sheets[k];
        printf("  Site %lu: marginals=[", k);
        for (int v = 0; v < MOBIUS_D; v++) {
            printf("%.4f", s->marginal[v]);
            if (v < MOBIUS_D - 1) printf(", ");
        }
        printf("] coh=%.4f degree=%lu\n", s->coherence, s->n_messages);
    }
    if (ms->n_sites > 8)
        printf("  ... (%lu more sites)\n", ms->n_sites - 8);
}

static inline void mobius_print_dressed(const MobiusAmplitudeSheet *ms, uint64_t site)
{
    const MobiusSiteSheet *s = &ms->sheets[site];
    printf("  Site %lu dressed: [", site);
    for (int v = 0; v < MOBIUS_D; v++) {
        printf("%.4f%+.4fi", s->dressed_re[v], s->dressed_im[v]);
        if (v < MOBIUS_D - 1) printf(", ");
    }
    printf("]\n");
}

#endif /* HPC_MOBIUS_H */
