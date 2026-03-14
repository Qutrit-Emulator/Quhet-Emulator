/*
 * magic_pointer.h — Extrapolate D^N Without Holding It
 *
 * ┌──────────────────────────────────────────────────────────┐
 * │  A "magic pointer" to any amplitude in the full state    │
 * │  vector, computed on demand from O(N+E) stored data.     │
 * │                                                          │
 * │  Stores: N local quhit states + E CZ edge phases        │
 * │  Computes: any ψ(i₁,...,iₙ) in O(N+E) time              │
 * │                                                          │
 * │  The full state vector has D^N entries.                   │
 * │  This never materializes it.                             │
 * │  The pointer IS the state.                               │
 * └──────────────────────────────────────────────────────────┘
 *
 * Core formula for CZ-entangled product states:
 *
 *   ψ(i₁,...,iₙ) = [Π_k a_k(i_k)] × [Π_{(p,q)∈edges} ω^(i_p·i_q)]
 *
 * where ω = exp(2πi/D) = exp(2πi/6), the D-th root of unity.
 *
 * This is EXACT for circuits composed of:
 *   - Local unitaries (absorbed into local state)
 *   - CZ gates (logged as phase edges)
 *
 * Any amplitude of the 6^N-dimensional state vector can be
 * evaluated from this compact representation.
 */

#ifndef MAGIC_POINTER_H
#define MAGIC_POINTER_H

#include "quhit_triality.h"
#include "s6_exotic.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ═══════════════════════════════════════════════════════════════════════
 * CONSTANTS
 * ═══════════════════════════════════════════════════════════════════════ */

#define MP_D            6       /* Physical dimension per site           */
#define MP_MAX_EDGES    4096    /* Initial edge capacity (grows)         */
#define MP_MAX_LOG      8192    /* Initial gate log capacity (grows)     */

/* ω = exp(2πi/6) = cos(π/3) + i·sin(π/3) = (1 + i√3)/2 */
static const double MP_W6_RE[6] = {
    1.0,                    /* ω⁰ = 1           */
    0.5,                    /* ω¹ = ½ + i√3/2   */
   -0.5,                    /* ω² = -½ + i√3/2  */
   -1.0,                    /* ω³ = -1           */
   -0.5,                    /* ω⁴ = -½ - i√3/2  */
    0.5                     /* ω⁵ = ½ - i√3/2   */
};
static const double MP_W6_IM[6] = {
    0.0,
    0.866025403784438647,   /* √3/2              */
    0.866025403784438647,
    0.0,
   -0.866025403784438647,
   -0.866025403784438647
};

/* ═══════════════════════════════════════════════════════════════════════
 * CZ EDGE — one logged entangling interaction
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    uint64_t site_a;        /* First site index                        */
    uint64_t site_b;        /* Second site index                       */
} MPEdge;

/* ═══════════════════════════════════════════════════════════════════════
 * GATE LOG ENTRY — for replay/exotic evaluation
 * ═══════════════════════════════════════════════════════════════════════ */

typedef enum {
    MP_GATE_LOCAL_DFT,      /* DFT₆ on one site                       */
    MP_GATE_LOCAL_PHASE,    /* Phase gate on one site                  */
    MP_GATE_LOCAL_SHIFT,    /* Cyclic shift on one site                */
    MP_GATE_LOCAL_UNITARY,  /* General 6×6 unitary on one site         */
    MP_GATE_CZ,             /* CZ between two sites                   */
    MP_GATE_INIT            /* Initialize site to specific state       */
} MPGateType;

typedef struct {
    MPGateType type;
    uint64_t   site_a;
    uint64_t   site_b;      /* Only for CZ                             */
    double     params[12];  /* Gate-specific parameters (phases, etc)  */
} MPGateEntry;

/* ═══════════════════════════════════════════════════════════════════════
 * MAGIC POINTER — The compact state representation
 *
 * This struct IS the state. The 6^N state vector does not exist.
 * Any amplitude ψ(i₁,...,iₙ) is computed on demand.
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    /* ── Sites ── */
    uint64_t       n_sites;        /* Number of quhit sites             */
    TrialityQuhit *locals;         /* Array of local quhit states       */

    /* ── CZ Phase Graph ── */
    uint64_t       n_edges;        /* Number of CZ interactions logged  */
    uint64_t       edge_cap;       /* Allocated edge capacity           */
    MPEdge        *edges;          /* CZ edge list                      */

    /* ── Gate Log (for replay/exotic) ── */
    uint64_t       n_log;          /* Number of gate log entries         */
    uint64_t       log_cap;        /* Allocated log capacity             */
    MPGateEntry   *gate_log;       /* Full gate operation log            */

    /* ── Statistics ── */
    uint64_t       amp_evals;      /* Number of amplitude evaluations    */
    uint64_t       prob_evals;     /* Number of probability evaluations  */
    uint64_t       measurements;   /* Number of measurements performed   */
} MagicPointer;

/* ═══════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 * ═══════════════════════════════════════════════════════════════════════ */

/* Create N-site magic pointer in product state |0,0,...,0⟩ */
static inline MagicPointer *mp_create(uint64_t n_sites)
{
    MagicPointer *mp = (MagicPointer *)calloc(1, sizeof(MagicPointer));
    if (!mp) return NULL;

    mp->n_sites  = n_sites;
    mp->locals   = (TrialityQuhit *)calloc(n_sites, sizeof(TrialityQuhit));
    if (!mp->locals) { free(mp); return NULL; }

    for (uint64_t i = 0; i < n_sites; i++)
        triality_init(&mp->locals[i]);

    mp->edge_cap = (n_sites < MP_MAX_EDGES) ? n_sites * 2 : MP_MAX_EDGES;
    mp->edges    = (MPEdge *)calloc(mp->edge_cap, sizeof(MPEdge));
    mp->n_edges  = 0;

    mp->log_cap  = MP_MAX_LOG;
    mp->gate_log = (MPGateEntry *)calloc(mp->log_cap, sizeof(MPGateEntry));
    mp->n_log    = 0;

    return mp;
}

/* Destroy and free all memory */
static inline void mp_destroy(MagicPointer *mp)
{
    if (!mp) return;
    free(mp->locals);
    free(mp->edges);
    free(mp->gate_log);
    free(mp);
}

/* ═══════════════════════════════════════════════════════════════════════
 * INTERNAL: grow arrays if needed
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void mp_grow_edges(MagicPointer *mp)
{
    if (mp->n_edges < mp->edge_cap) return;
    mp->edge_cap *= 2;
    mp->edges = (MPEdge *)realloc(mp->edges, mp->edge_cap * sizeof(MPEdge));
}

static inline void mp_grow_log(MagicPointer *mp)
{
    if (mp->n_log < mp->log_cap) return;
    mp->log_cap *= 2;
    mp->gate_log = (MPGateEntry *)realloc(mp->gate_log,
                                          mp->log_cap * sizeof(MPGateEntry));
}

static inline void mp_log_gate(MagicPointer *mp, MPGateEntry entry)
{
    mp_grow_log(mp);
    mp->gate_log[mp->n_log++] = entry;
}

/* ═══════════════════════════════════════════════════════════════════════
 * LOCAL GATES — Absorbed into the local quhit state
 *
 * Local gates modify only one site's amplitudes. They are applied
 * directly to the TrialityQuhit and logged for replay.
 * ═══════════════════════════════════════════════════════════════════════ */

/* Initialize site to a specific edge-basis state vector */
static inline void mp_set_local(MagicPointer *mp, uint64_t site,
                                const double re[6], const double im[6])
{
    TrialityQuhit *q = &mp->locals[site];
    for (int i = 0; i < MP_D; i++) {
        q->edge_re[i] = re[i];
        q->edge_im[i] = im[i];
    }
    q->primary = VIEW_EDGE;
    q->dirty = DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED;
    q->delta_valid = 0;
    triality_update_mask(q);

    MPGateEntry entry = { .type = MP_GATE_INIT, .site_a = site };
    for (int i = 0; i < 6; i++) { entry.params[i] = re[i]; }
    mp_log_gate(mp, entry);
}

/* Apply DFT₆ to one site */
static inline void mp_dft(MagicPointer *mp, uint64_t site)
{
    triality_dft(&mp->locals[site]);
    MPGateEntry entry = { .type = MP_GATE_LOCAL_DFT, .site_a = site };
    mp_log_gate(mp, entry);
}

/* Apply phase gate to one site: |k⟩ → e^{iφ_k}|k⟩ */
static inline void mp_phase(MagicPointer *mp, uint64_t site,
                            const double phi_re[6], const double phi_im[6])
{
    triality_phase(&mp->locals[site], phi_re, phi_im);
    MPGateEntry entry = { .type = MP_GATE_LOCAL_PHASE, .site_a = site };
    for (int i = 0; i < 6; i++) entry.params[i] = phi_re[i];
    mp_log_gate(mp, entry);
}

/* Apply cyclic shift: |k⟩ → |k+1 mod 6⟩ */
static inline void mp_shift(MagicPointer *mp, uint64_t site, int delta)
{
    triality_shift(&mp->locals[site], delta);
    MPGateEntry entry = { .type = MP_GATE_LOCAL_SHIFT, .site_a = site };
    entry.params[0] = (double)delta;
    mp_log_gate(mp, entry);
}

/* ═══════════════════════════════════════════════════════════════════════
 * CZ GATE — Logged as a phase edge, NOT applied to local states
 *
 * For the magic pointer, the CZ is not absorbed into the local
 * states. Instead, it is recorded as an edge in the phase graph.
 * The phase contribution ω^(i_a·i_b) is applied at evaluation time.
 *
 * This is the key to the magic: CZ creates entanglement, but we
 * represent the entanglement as a GRAPH EDGE, not as a tensor.
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void mp_cz(MagicPointer *mp, uint64_t site_a, uint64_t site_b)
{
    mp_grow_edges(mp);
    mp->edges[mp->n_edges++] = (MPEdge){ .site_a = site_a, .site_b = site_b };

    MPGateEntry entry = {
        .type = MP_GATE_CZ,
        .site_a = site_a,
        .site_b = site_b
    };
    mp_log_gate(mp, entry);
}

/* ═══════════════════════════════════════════════════════════════════════
 * THE MAGIC: Amplitude Evaluation
 *
 * ψ(i₁,...,iₙ) = [Π_k a_k(i_k)] × [Π_{edges} ω^(i_a·i_b)]
 *
 * Cost: O(N + E) — linear in sites + edges
 * Memory: O(1) additional (just accumulates a complex number)
 *
 * This is the entire point:
 * The state vector has 6^N entries.
 * We compute any single entry in O(N+E).
 * We never store the other 6^N - 1 entries.
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void mp_amplitude(const MagicPointer *mp,
                                const uint32_t *indices,
                                double *out_re, double *out_im)
{
    /* Start with 1 + 0i */
    double re = 1.0, im = 0.0;

    /* Step 1: Product of local amplitudes — O(N) */
    for (uint64_t k = 0; k < mp->n_sites; k++) {
        uint32_t idx = indices[k];
        /* Ensure edge view is available */
        const TrialityQuhit *q = &mp->locals[k];
        double a_re = q->edge_re[idx];
        double a_im = q->edge_im[idx];
        /* Complex multiply: (re + i·im) × (a_re + i·a_im) */
        double new_re = re * a_re - im * a_im;
        double new_im = re * a_im + im * a_re;
        re = new_re;
        im = new_im;
    }

    /* Step 2: CZ phase accumulation — O(E) */
    for (uint64_t e = 0; e < mp->n_edges; e++) {
        uint32_t ia = indices[mp->edges[e].site_a];
        uint32_t ib = indices[mp->edges[e].site_b];
        uint32_t phase_idx = (ia * ib) % MP_D;
        /* Complex multiply by ω^(ia·ib) */
        double w_re = MP_W6_RE[phase_idx];
        double w_im = MP_W6_IM[phase_idx];
        double new_re = re * w_re - im * w_im;
        double new_im = re * w_im + im * w_re;
        re = new_re;
        im = new_im;
    }

    *out_re = re;
    *out_im = im;

    /* Update stats (cast away const for stats only) */
    ((MagicPointer *)mp)->amp_evals++;
}

/* ═══════════════════════════════════════════════════════════════════════
 * PROBABILITY — |ψ(i₁,...,iₙ)|²
 *
 * Slightly optimized: computes |amplitude|² directly.
 * ═══════════════════════════════════════════════════════════════════════ */

static inline double mp_probability(const MagicPointer *mp,
                                    const uint32_t *indices)
{
    double re, im;
    mp_amplitude(mp, indices, &re, &im);
    ((MagicPointer *)mp)->prob_evals++;
    return re * re + im * im;
}

/* ═══════════════════════════════════════════════════════════════════════
 * MARGINAL PROBABILITY — P(site_k = v)
 *
 * Sums |ψ(..., i_k=v, ...)|² over all other indices.
 * For product states (E=0): instant, just |local_k(v)|².
 * For entangled states: requires summing over partners.
 *
 * Smart version: only sums over sites connected to site_k by edges.
 * Unconnected sites contribute |local_k|² as a constant factor.
 * ═══════════════════════════════════════════════════════════════════════ */

static inline double mp_marginal_prob(const MagicPointer *mp,
                                      uint64_t site, uint32_t value)
{
    /* Find all sites connected to this site by CZ edges */
    uint64_t connected[128];  /* sites connected to target */
    uint64_t edge_ids[128];   /* which edges connect them */
    uint64_t n_connected = 0;

    for (uint64_t e = 0; e < mp->n_edges; e++) {
        if (mp->edges[e].site_a == site || mp->edges[e].site_b == site) {
            uint64_t partner = (mp->edges[e].site_a == site) ?
                                mp->edges[e].site_b : mp->edges[e].site_a;
            /* Check if partner already in list */
            int found = 0;
            for (uint64_t c = 0; c < n_connected; c++) {
                if (connected[c] == partner) { found = 1; break; }
            }
            if (!found && n_connected < 128) {
                connected[n_connected] = partner;
                edge_ids[n_connected] = e;
                n_connected++;
            }
        }
    }

    /* Product state: no edges touching this site */
    if (n_connected == 0) {
        const TrialityQuhit *q = &mp->locals[site];
        return q->edge_re[value] * q->edge_re[value] +
               q->edge_im[value] * q->edge_im[value];
    }

    /* Entangled: sum over all connected partner configurations */
    /* For simplicity, enumerate D^n_connected terms */
    /* This is tractable because n_connected is typically small (1-3) */
    double total_prob = 0.0;

    /* Base amplitude from target site */
    const TrialityQuhit *q_site = &mp->locals[site];
    double site_re = q_site->edge_re[value];
    double site_im = q_site->edge_im[value];
    double site_prob = site_re * site_re + site_im * site_im;

    /* Norm factor from all unconnected sites (= 1 if they're normalized) */
    /* No need to compute — unconnected sites are independent, contribute 1 */

    /* Enumerate all configurations of connected sites */
    uint64_t n_configs = 1;
    for (uint64_t c = 0; c < n_connected; c++) n_configs *= MP_D;

    for (uint64_t cfg = 0; cfg < n_configs; cfg++) {
        /* Decode configuration */
        uint32_t partner_vals[128];
        uint64_t tmp = cfg;
        for (uint64_t c = 0; c < n_connected; c++) {
            partner_vals[c] = tmp % MP_D;
            tmp /= MP_D;
        }

        /* Probability contribution from partner local amplitudes */
        double cfg_prob = site_prob;
        for (uint64_t c = 0; c < n_connected; c++) {
            const TrialityQuhit *q_p = &mp->locals[connected[c]];
            uint32_t pv = partner_vals[c];
            cfg_prob *= (q_p->edge_re[pv] * q_p->edge_re[pv] +
                         q_p->edge_im[pv] * q_p->edge_im[pv]);
        }

        /* Phase contribution from CZ edges touching this site */
        /* The phase is ω^(value·partner_val) for each edge */
        /* But since we're computing |amplitude|², we need the actual
         * complex product, not just the probability. Let me recompute
         * using the full amplitude formula for the connected subsystem. */

        /* Recompute as amplitude for accuracy */
        double amp_re = site_re, amp_im = site_im;
        for (uint64_t c = 0; c < n_connected; c++) {
            const TrialityQuhit *q_p = &mp->locals[connected[c]];
            uint32_t pv = partner_vals[c];
            double p_re = q_p->edge_re[pv], p_im = q_p->edge_im[pv];
            double new_re = amp_re * p_re - amp_im * p_im;
            double new_im = amp_re * p_im + amp_im * p_re;
            amp_re = new_re;
            amp_im = new_im;
        }

        /* CZ phases for all edges touching this site */
        for (uint64_t e = 0; e < mp->n_edges; e++) {
            uint64_t sa = mp->edges[e].site_a;
            uint64_t sb = mp->edges[e].site_b;

            uint32_t va = 0, vb = 0;
            if (sa == site) {
                va = value;
                /* Find partner value */
                for (uint64_t c = 0; c < n_connected; c++) {
                    if (connected[c] == sb) { vb = partner_vals[c]; break; }
                }
            } else if (sb == site) {
                vb = value;
                for (uint64_t c = 0; c < n_connected; c++) {
                    if (connected[c] == sa) { va = partner_vals[c]; break; }
                }
            } else {
                /* Edge between two partners — also need this phase */
                int found_a = 0, found_b = 0;
                for (uint64_t c = 0; c < n_connected; c++) {
                    if (connected[c] == sa) { va = partner_vals[c]; found_a = 1; }
                    if (connected[c] == sb) { vb = partner_vals[c]; found_b = 1; }
                }
                if (!found_a || !found_b) continue; /* edge not in subsystem */
            }

            uint32_t phase_idx = (va * vb) % MP_D;
            double w_re = MP_W6_RE[phase_idx];
            double w_im = MP_W6_IM[phase_idx];
            double new_re = amp_re * w_re - amp_im * w_im;
            double new_im = amp_re * w_im + amp_im * w_re;
            amp_re = new_re;
            amp_im = new_im;
        }

        total_prob += amp_re * amp_re + amp_im * amp_im;
    }

    return total_prob;
}

/* ═══════════════════════════════════════════════════════════════════════
 * MEASUREMENT — Collapse site k
 *
 * Computes marginal probabilities for site k, samples an outcome,
 * then collapses the local state to the measured value.
 * CZ edges involving this site may be simplified.
 * ═══════════════════════════════════════════════════════════════════════ */

static inline uint32_t mp_measure(MagicPointer *mp, uint64_t site,
                                  double random_01)
{
    /* Compute marginal probabilities for all outcomes */
    double probs[MP_D];
    double total = 0.0;
    for (int v = 0; v < MP_D; v++) {
        probs[v] = mp_marginal_prob(mp, site, v);
        total += probs[v];
    }

    /* Normalize */
    if (total > 0) {
        for (int v = 0; v < MP_D; v++) probs[v] /= total;
    }

    /* Sample */
    double cumul = 0.0;
    uint32_t outcome = MP_D - 1;
    for (int v = 0; v < MP_D; v++) {
        cumul += probs[v];
        if (random_01 <= cumul) { outcome = v; break; }
    }

    /* Collapse local state to |outcome⟩ */
    for (int v = 0; v < MP_D; v++) {
        mp->locals[site].edge_re[v] = (v == (int)outcome) ? 1.0 : 0.0;
        mp->locals[site].edge_im[v] = 0.0;
    }
    mp->locals[site].primary = VIEW_EDGE;
    mp->locals[site].dirty = DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED;
    mp->locals[site].delta_valid = 0;
    triality_update_mask(&mp->locals[site]);

    /* CZ edges from this site now contribute fixed phases ω^(outcome·i_b)
     * which can be absorbed into the partner's local state.
     * This is the measurement-induced disentanglement. */
    for (uint64_t e = 0; e < mp->n_edges; ) {
        MPEdge *edge = &mp->edges[e];
        if (edge->site_a == site || edge->site_b == site) {
            uint64_t partner = (edge->site_a == site) ?
                                edge->site_b : edge->site_a;
            uint32_t my_val = outcome;

            /* Absorb the CZ phase into partner's local state:
             * partner[k] *= ω^(my_val·k) for each basis state k */
            TrialityQuhit *p = &mp->locals[partner];
            for (int k = 0; k < MP_D; k++) {
                uint32_t phase_idx = (my_val * k) % MP_D;
                double w_re = MP_W6_RE[phase_idx];
                double w_im = MP_W6_IM[phase_idx];
                double old_re = p->edge_re[k], old_im = p->edge_im[k];
                p->edge_re[k] = old_re * w_re - old_im * w_im;
                p->edge_im[k] = old_re * w_im + old_im * w_re;
            }
            p->dirty = DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED;
            p->delta_valid = 0;

            /* Remove this edge (swap with last) */
            mp->edges[e] = mp->edges[--mp->n_edges];
        } else {
            e++;
        }
    }

    mp->measurements++;
    return outcome;
}

/* ═══════════════════════════════════════════════════════════════════════
 * NORMALIZATION CHECK — Σ |ψ(i₁,...,iₙ)|² over ALL indices
 *
 * For small N only! Cost: O(D^N × (N+E))
 * Useful for verification but exponential in N.
 * ═══════════════════════════════════════════════════════════════════════ */

static inline double mp_norm_sq(const MagicPointer *mp)
{
    if (mp->n_sites > 8) {
        fprintf(stderr, "mp_norm_sq: N=%lu too large for brute force\n",
                mp->n_sites);
        return -1.0;
    }

    uint64_t total_configs = 1;
    for (uint64_t i = 0; i < mp->n_sites; i++) total_configs *= MP_D;

    double norm = 0.0;
    uint32_t indices[8];

    for (uint64_t cfg = 0; cfg < total_configs; cfg++) {
        uint64_t tmp = cfg;
        for (uint64_t i = 0; i < mp->n_sites; i++) {
            indices[i] = tmp % MP_D;
            tmp /= MP_D;
        }
        norm += mp_probability(mp, indices);
    }

    return norm;
}

/* ═══════════════════════════════════════════════════════════════════════
 * EXOTIC INVARIANT — Δ of the magic pointer
 *
 * For a single site: just returns the local Δ.
 * For multi-site: computes the average Δ across all sites,
 * weighted by their entanglement degree (number of CZ edges).
 * ═══════════════════════════════════════════════════════════════════════ */

static inline double mp_exotic_invariant(MagicPointer *mp)
{
    double total_delta = 0.0;
    for (uint64_t i = 0; i < mp->n_sites; i++) {
        total_delta += triality_exotic_invariant_cached(&mp->locals[i]);
    }
    return total_delta / mp->n_sites;
}

/* ═══════════════════════════════════════════════════════════════════════
 * ENTROPY ESTIMATE — across a cut
 *
 * Uses the CZ graph structure to estimate entanglement entropy.
 * Edges crossing the cut contribute log₂(D) bits each.
 * This is an upper bound (saturated for maximally entangling CZ).
 * ═══════════════════════════════════════════════════════════════════════ */

static inline double mp_entropy_cut(const MagicPointer *mp,
                                    uint64_t cut_after)
{
    uint64_t crossing_edges = 0;
    for (uint64_t e = 0; e < mp->n_edges; e++) {
        uint64_t sa = mp->edges[e].site_a;
        uint64_t sb = mp->edges[e].site_b;
        /* Edge crosses cut if one site is ≤ cut_after and other is > */
        if ((sa <= cut_after && sb > cut_after) ||
            (sb <= cut_after && sa > cut_after)) {
            crossing_edges++;
        }
    }
    /* Each crossing CZ contributes at most log₂(D) = log₂(6) ≈ 2.585 bits */
    return crossing_edges * log2((double)MP_D);
}

/* ═══════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void mp_print_stats(const MagicPointer *mp)
{
    printf("╔════════════════════════════════════════════════╗\n");
    printf("║  Magic Pointer Statistics                     ║\n");
    printf("╠════════════════════════════════════════════════╣\n");
    printf("║  Sites:       %10lu                       ║\n", mp->n_sites);
    printf("║  CZ edges:    %10lu                       ║\n", mp->n_edges);
    printf("║  Gate log:    %10lu                       ║\n", mp->n_log);
    printf("║  Amp evals:   %10lu                       ║\n", mp->amp_evals);
    printf("║  Prob evals:  %10lu                       ║\n", mp->prob_evals);
    printf("║  Measurements:%10lu                       ║\n", mp->measurements);

    /* Memory footprint */
    uint64_t site_bytes = mp->n_sites * sizeof(TrialityQuhit);
    uint64_t edge_bytes = mp->n_edges * sizeof(MPEdge);
    uint64_t log_bytes  = mp->n_log * sizeof(MPGateEntry);
    uint64_t total      = site_bytes + edge_bytes + log_bytes + sizeof(MagicPointer);

    printf("║  Memory:      %10lu bytes                ║\n", total);
    printf("║    Sites:     %10lu bytes                ║\n", site_bytes);
    printf("║    Edges:     %10lu bytes                ║\n", edge_bytes);
    printf("║    Gate log:  %10lu bytes                ║\n", log_bytes);

    /* What the full state vector would cost */
    double full_sv_log = mp->n_sites * log10(6.0) + log10(16.0); /* 16 bytes per complex */
    printf("║  Full SV:     10^%.1f bytes (impossible)    ║\n", full_sv_log);
    printf("╚════════════════════════════════════════════════╝\n");
}

static inline void mp_print_state(const MagicPointer *mp, const char *label)
{
    printf("── %s ──\n", label);
    printf("  Sites: %lu, Edges: %lu\n", mp->n_sites, mp->n_edges);
    for (uint64_t i = 0; i < mp->n_sites && i < 8; i++) {
        printf("  Site %lu: [", i);
        for (int j = 0; j < MP_D; j++) {
            printf("%.3f%+.3fi", mp->locals[i].edge_re[j],
                                 mp->locals[i].edge_im[j]);
            if (j < MP_D - 1) printf(", ");
        }
        printf("]\n");
    }
    if (mp->n_sites > 8) printf("  ... (%lu more sites)\n", mp->n_sites - 8);
    for (uint64_t e = 0; e < mp->n_edges && e < 16; e++) {
        printf("  CZ: %lu ↔ %lu\n", mp->edges[e].site_a, mp->edges[e].site_b);
    }
    if (mp->n_edges > 16) printf("  ... (%lu more edges)\n", mp->n_edges - 16);
}

#endif /* MAGIC_POINTER_H */
