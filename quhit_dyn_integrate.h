/*
 * quhit_dyn_integrate.h — I reach into every overlay. I make them all breathe.
 *
 * The dynamic growth engine was built for 6D. But awareness doesn't
 * belong to one dimension. It belongs to any space where entanglement
 * moves. This module connects the DynLattice to every overlay:
 *
 *   MPS (1D)    — Chain growth. Extend or contract endpoints.
 *   PEPS (2D)   — 4-neighbor lattice breathing.
 *   TNS (3D-6D) — Full 6-12 neighbor growth/contraction.
 *
 */

#ifndef QUHIT_DYN_INTEGRATE_H
#define QUHIT_DYN_INTEGRATE_H

#include "quhit_peps_grow.h"
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* Oracle convergence states */
#define DYN_CONVERGING  0
#define DYN_OSCILLATING 1
#define DYN_STAGNANT    2

/* Oracle history depth */
#define DYN_ORACLE_HISTORY 8

/* ═══════════════════════════════════════════════════════════════════════════════
 * 1D CHAIN GROWTH — MPS doesn't need a lattice. It needs a living chain.
 *
 * An MPS chain is linear: site 0 — site 1 — ... — site N-1.
 * Growth means extending the ACTIVE REGION within a pre-allocated chain.
 * Contraction means shrinking from the idle tails.
 *
 * The chain breathes from both ends, tracking the entanglement front.
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    int      max_sites;         /* Maximum chain length (pre-allocated)     */
    int      active_start;      /* First active site index                  */
    int      active_end;        /* Last active site index (inclusive)        */

    /* Per-site entropy tracking */
    double  *entropy;           /* [max_sites] — von Neumann entropy        */

    /* Thresholds */
    double   grow_threshold;    /* Extend when boundary entropy > this      */
    double   contract_threshold;/* Contract when tail entropy < this        */
    int      min_active;        /* Never contract below this length         */

    /* Statistics */
    uint32_t epoch;
    uint32_t grow_events;
    uint32_t contract_events;

    /* ═══ ORACLE INFRASTRUCTURE ═══ */

    /* Oracle 1: Entropy Prediction — ring buffer of per-site history */
    double  *entropy_history;   /* [max_sites × HISTORY] ring buffer        */
    double  *entropy_predicted; /* [max_sites] extrapolated next-step H     */
    int      history_cursor;    /* Current write position in ring buffer    */

    /* Oracle 2: Long-Range Correlation — mutual information map */
    double  *correlation_map;   /* [max_sites × max_sites] JSD matrix       */
    int      best_couple_i;     /* Optimal coupling pair: site i            */
    int      best_couple_j;     /* Optimal coupling pair: site j            */
    double   max_correlation;   /* Strength of best coupling                */

    /* Oracle 3: Convergence Horizon — oscillation detection */
    double   fidelity_history[DYN_ORACLE_HISTORY]; /* Global fidelity trace */
    int      fidelity_cursor;   /* Ring buffer position                     */
    int      convergence_state; /* DYN_CONVERGING / OSCILLATING / STAGNANT  */
    double   prev_total_entropy;/* Previous epoch's total entropy           */

    /* Oracle 4: Topology Phase-Transition — boundary detection */
    int      phase_boundary;    /* Site index of sharpest entropy gradient  */
    double   phase_gradient;    /* Magnitude of sharpest gradient           */

    /* Oracle 5: Permanent Steering — site weight ranking */
    double  *site_weight;       /* [max_sites] per-site information weight  */
} DynChain;

/* ── Lifecycle ── */

static inline DynChain dyn_chain_create(int max_sites)
{
    DynChain dc;
    memset(&dc, 0, sizeof(dc));
    dc.max_sites    = max_sites;
    dc.active_start = 0;
    dc.active_end   = 0;
    dc.entropy      = (double *)calloc(max_sites, sizeof(double));
    dc.grow_threshold    = 0.1 * log2(6.0);
    dc.contract_threshold = 0.01 * log2(6.0);
    dc.min_active   = 1;

    /* Oracle allocations */
    dc.entropy_history   = (double *)calloc((size_t)max_sites * DYN_ORACLE_HISTORY, sizeof(double));
    dc.entropy_predicted = (double *)calloc(max_sites, sizeof(double));
    dc.correlation_map   = (double *)calloc((size_t)max_sites * max_sites, sizeof(double));
    dc.site_weight       = (double *)calloc(max_sites, sizeof(double));
    dc.best_couple_i = 0;
    dc.best_couple_j = 1;
    dc.convergence_state = DYN_CONVERGING;
    dc.phase_boundary = -1;
    return dc;
}

static inline void dyn_chain_free(DynChain *dc)
{
    free(dc->entropy);          dc->entropy = NULL;
    free(dc->entropy_history);  dc->entropy_history = NULL;
    free(dc->entropy_predicted);dc->entropy_predicted = NULL;
    free(dc->correlation_map);  dc->correlation_map = NULL;
    free(dc->site_weight);      dc->site_weight = NULL;
}

/* ── Seed — activate a contiguous region ── */

static inline void dyn_chain_seed(DynChain *dc, int start, int end)
{
    if (start < 0) start = 0;
    if (end >= dc->max_sites) end = dc->max_sites - 1;
    dc->active_start = start;
    dc->active_end   = end;
}

static inline int dyn_chain_active_length(const DynChain *dc)
{
    return dc->active_end - dc->active_start + 1;
}

/* ── Update entropy for a site ── */

static inline void dyn_chain_update_entropy(DynChain *dc, int site,
                                             const double *probs, int D)
{
    if (site < 0 || site >= dc->max_sites) return;
    dc->entropy[site] = site_entropy(probs, D);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * ORACLE 1: ENTROPY PREDICTION
 *
 * Maintains a ring buffer of per-site entropy history.
 * Fits a linear trend to predict next-step entropy.
 * Growth uses PREDICTED entropy, not just current.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline void dyn_chain_record_entropy(DynChain *dc)
{
    int cursor = dc->history_cursor % DYN_ORACLE_HISTORY;
    for (int s = dc->active_start; s <= dc->active_end; s++) {
        dc->entropy_history[(size_t)s * DYN_ORACLE_HISTORY + cursor] = dc->entropy[s];
    }
    dc->history_cursor++;
}

static inline void dyn_chain_predict_entropy(DynChain *dc)
{
    int n = dc->history_cursor < DYN_ORACLE_HISTORY ?
            dc->history_cursor : DYN_ORACLE_HISTORY;
    if (n < 2) {
        /* Not enough history — predict = current */
        for (int s = 0; s < dc->max_sites; s++)
            dc->entropy_predicted[s] = dc->entropy[s];
        return;
    }

    /* Linear least-squares per site: H(t) ≈ a·t + b, predict H(t+1) */
    for (int s = dc->active_start; s <= dc->active_end; s++) {
        double sum_t = 0, sum_H = 0, sum_tH = 0, sum_tt = 0;
        for (int i = 0; i < n; i++) {
            int idx = (dc->history_cursor - n + i) % DYN_ORACLE_HISTORY;
            if (idx < 0) idx += DYN_ORACLE_HISTORY;
            double H = dc->entropy_history[(size_t)s * DYN_ORACLE_HISTORY + idx];
            double t = (double)i;
            sum_t  += t;
            sum_H  += H;
            sum_tH += t * H;
            sum_tt += t * t;
        }
        double denom = n * sum_tt - sum_t * sum_t;
        double a = 0, b = sum_H / n;
        if (fabs(denom) > 1e-30) {
            a = (n * sum_tH - sum_t * sum_H) / denom;
            b = (sum_H - a * sum_t) / n;
        }
        dc->entropy_predicted[s] = a * (double)n + b;  /* Next step */
        if (dc->entropy_predicted[s] < 0) dc->entropy_predicted[s] = 0;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * ORACLE 2: LONG-RANGE CORRELATION
 *
 * Computes Jensen-Shannon divergence between all active site pairs.
 * Identifies the most-correlated pair for optimal coupling.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline void dyn_chain_mutual_info(DynChain *dc, const double *all_marginals, int D)
{
    /* all_marginals: [max_sites × D] array of per-site channel-A marginals */
    int len = dyn_chain_active_length(dc);
    if (len < 2) return;

    dc->max_correlation = 0;
    dc->best_couple_i = dc->active_start;
    dc->best_couple_j = dc->active_start + 1;

    for (int i = dc->active_start; i <= dc->active_end; i++) {
        for (int j = i + 1; j <= dc->active_end; j++) {
            /* Jensen-Shannon divergence: JSD(P||Q) = ½KL(P||M) + ½KL(Q||M)
             * where M = (P+Q)/2 */
            const double *P = all_marginals + (size_t)i * D;
            const double *Q = all_marginals + (size_t)j * D;

            double jsd = 0;
            for (int k = 0; k < D; k++) {
                double m = (P[k] + Q[k]) * 0.5;
                if (m > 1e-15) {
                    if (P[k] > 1e-15) jsd += 0.5 * P[k] * log2(P[k] / m);
                    if (Q[k] > 1e-15) jsd += 0.5 * Q[k] * log2(Q[k] / m);
                }
            }

            dc->correlation_map[(size_t)i * dc->max_sites + j] = jsd;
            dc->correlation_map[(size_t)j * dc->max_sites + i] = jsd;

            /* Track MAXIMUM correlation (highest JSD = most different =
             * most informative coupling target) */
            if (jsd > dc->max_correlation) {
                dc->max_correlation = jsd;
                dc->best_couple_i = i;
                dc->best_couple_j = j;
            }
        }
    }
}

static inline void dyn_chain_best_coupling(const DynChain *dc, int *i_out, int *j_out)
{
    *i_out = dc->best_couple_i;
    *j_out = dc->best_couple_j;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * ORACLE 3: CONVERGENCE HORIZON
 *
 * Tracks total entropy across epochs. Detects:
 * - CONVERGING: entropy monotonically decreasing
 * - OSCILLATING: entropy bounces up/down > 3 times
 * - STAGNANT: entropy change < ε for 4+ epochs
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline int dyn_chain_check_convergence(DynChain *dc)
{
    /* Compute current total entropy */
    double total = 0;
    for (int s = dc->active_start; s <= dc->active_end; s++)
        total += dc->entropy[s];

    /* Global fidelity: |ΔH| between epochs */
    double fidelity = fabs(total - dc->prev_total_entropy);
    int cursor = dc->fidelity_cursor % DYN_ORACLE_HISTORY;
    dc->fidelity_history[cursor] = fidelity;
    dc->fidelity_cursor++;
    dc->prev_total_entropy = total;

    int n = dc->fidelity_cursor < DYN_ORACLE_HISTORY ?
            dc->fidelity_cursor : DYN_ORACLE_HISTORY;
    if (n < 3) { dc->convergence_state = DYN_CONVERGING; return DYN_CONVERGING; }

    /* Count sign changes (oscillation) and near-zero stretches (stagnation) */
    int sign_changes = 0, stagnant_count = 0;
    double eps = 1e-6;
    for (int i = 1; i < n; i++) {
        int ci = (dc->fidelity_cursor - n + i) % DYN_ORACLE_HISTORY;
        int pi = (dc->fidelity_cursor - n + i - 1) % DYN_ORACLE_HISTORY;
        if (ci < 0) ci += DYN_ORACLE_HISTORY;
        if (pi < 0) pi += DYN_ORACLE_HISTORY;
        double diff = dc->fidelity_history[ci] - dc->fidelity_history[pi];
        if (i > 1) {
            int ppi = (dc->fidelity_cursor - n + i - 2) % DYN_ORACLE_HISTORY;
            if (ppi < 0) ppi += DYN_ORACLE_HISTORY;
            double prev_diff = dc->fidelity_history[pi] - dc->fidelity_history[ppi];
            if ((diff > 0 && prev_diff < 0) || (diff < 0 && prev_diff > 0))
                sign_changes++;
        }
        if (fabs(dc->fidelity_history[ci]) < eps)
            stagnant_count++;
    }

    if (sign_changes >= 3)
        dc->convergence_state = DYN_OSCILLATING;
    else if (stagnant_count >= 4)
        dc->convergence_state = DYN_STAGNANT;
    else
        dc->convergence_state = DYN_CONVERGING;

    return dc->convergence_state;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * ORACLE 4: TOPOLOGY PHASE-TRANSITION
 *
 * Detects sharp entropy gradients between adjacent sites.
 * Returns site index of the sharpest structural boundary.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline int dyn_chain_phase_boundary(DynChain *dc)
{
    dc->phase_boundary = -1;
    dc->phase_gradient = 0;

    for (int s = dc->active_start; s < dc->active_end; s++) {
        double grad = fabs(dc->entropy[s+1] - dc->entropy[s]);
        if (grad > dc->phase_gradient) {
            dc->phase_gradient = grad;
            dc->phase_boundary = s;
        }
    }
    return dc->phase_boundary;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * ORACLE 5: PERMANENT STEERING
 *
 * Computes per-site "information weight" = (1 - H/H_max).
 * Low-weight sites carry no information → contract them first.
 * High-weight sites are critical → never contract them.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline void dyn_chain_compute_weights(DynChain *dc, int D)
{
    double H_max = log2((double)D);
    if (H_max < 1e-15) H_max = 1.0;

    for (int s = dc->active_start; s <= dc->active_end; s++) {
        /* Weight = 1 when perfectly collapsed (H=0), 0 when maximally mixed */
        dc->site_weight[s] = 1.0 - dc->entropy[s] / H_max;
        if (dc->site_weight[s] < 0) dc->site_weight[s] = 0;
    }
}

/* Find the least-informative active site (for intelligent contraction) */
static inline int dyn_chain_weakest_site(const DynChain *dc)
{
    double min_w = 1e30;
    int weakest = dc->active_start;
    for (int s = dc->active_start; s <= dc->active_end; s++) {
        if (dc->site_weight[s] < min_w) {
            min_w = dc->site_weight[s];
            weakest = s;
        }
    }
    return weakest;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * ORACLE-ENHANCED GROW / CONTRACT / STEP
 *
 * These replace the original grow/contract/step with oracle-aware versions.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline int dyn_chain_grow(DynChain *dc)
{
    int grown = 0;

    /* Use PREDICTED entropy if available, else current */
    double left_H = (dc->history_cursor >= 2) ?
        dc->entropy_predicted[dc->active_start] :
        dc->entropy[dc->active_start];
    double right_H = (dc->history_cursor >= 2) ?
        dc->entropy_predicted[dc->active_end] :
        dc->entropy[dc->active_end];

    if (dc->active_start > 0 && left_H > dc->grow_threshold) {
        dc->active_start--;
        grown++;
    }
    if (dc->active_end < dc->max_sites - 1 && right_H > dc->grow_threshold) {
        dc->active_end++;
        grown++;
    }

    dc->grow_events += grown;
    return grown;
}

static inline int dyn_chain_contract(DynChain *dc)
{
    int contracted = 0;
    if (dyn_chain_active_length(dc) <= dc->min_active) return 0;

    /* Use site weights if computed — contract weakest boundary site */
    double left_w  = dc->site_weight[dc->active_start];
    double right_w = dc->site_weight[dc->active_end];

    /* Only contract sites with near-zero weight AND low entropy */
    if (dc->entropy[dc->active_start] < dc->contract_threshold &&
        left_w < 0.1 &&
        dyn_chain_active_length(dc) > dc->min_active) {
        dc->active_start++;
        contracted++;
    }

    if (dc->active_end > dc->active_start &&
        dc->entropy[dc->active_end] < dc->contract_threshold &&
        right_w < 0.1 &&
        dyn_chain_active_length(dc) > dc->min_active) {
        dc->active_end--;
        contracted++;
    }

    dc->contract_events += contracted;
    return contracted;
}

static inline void dyn_chain_step(DynChain *dc)
{
    /* Record entropy history for prediction oracle */
    dyn_chain_record_entropy(dc);
    dyn_chain_predict_entropy(dc);

    /* Check convergence state */
    dyn_chain_check_convergence(dc);

    /* Detect phase boundaries */
    dyn_chain_phase_boundary(dc);

    /* Compute site weights for intelligent contraction */
    dyn_chain_compute_weights(dc, 6);

    /* If oscillating, contract aggressively. If converging, grow. */
    if (dc->convergence_state == DYN_OSCILLATING) {
        dyn_chain_contract(dc);
    } else {
        int grown = dyn_chain_grow(dc);
        if (grown == 0)
            dyn_chain_contract(dc);
    }

    dc->epoch++;
}

/* ── Query — is this site currently active? ── */

static inline int dyn_chain_is_active(const DynChain *dc, int site)
{
    return (site >= dc->active_start && site <= dc->active_end);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PEPS/TNS INTEGRATION — Convenience wrappers for 2D-6D grids
 *
 * These create a DynLattice matched to the grid's dimensionality,
 * then provide entropy-update and growth functions that speak the
 * grid's coordinate language.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Create a DynLattice for a 2D PEPS grid */
static inline DynLattice* dyn_peps2d_create(int Lx, int Ly)
{
    return dyn_lattice_create(Lx, Ly, 1, 1, 1, 1, 2);
}

/* Create a DynLattice for a 3D TNS grid */
static inline DynLattice* dyn_tns3d_create(int Lx, int Ly, int Lz)
{
    return dyn_lattice_create(Lx, Ly, Lz, 1, 1, 1, 3);
}

/* Create a DynLattice for a 4D TNS grid */
static inline DynLattice* dyn_tns4d_create(int Lx, int Ly, int Lz, int Lw)
{
    return dyn_lattice_create(Lx, Ly, Lz, Lw, 1, 1, 4);
}

/* Create a DynLattice for a 5D TNS grid */
static inline DynLattice* dyn_tns5d_create(int Lx, int Ly, int Lz, int Lw, int Lv)
{
    return dyn_lattice_create(Lx, Ly, Lz, Lw, Lv, 1, 5);
}

/* Create a DynLattice for a 6D TNS grid */
static inline DynLattice* dyn_tns6d_create(int Lx, int Ly, int Lz,
                                            int Lw, int Lv, int Lu)
{
    return dyn_lattice_create(Lx, Ly, Lz, Lw, Lv, Lu, 6);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SITE ACTIVITY CHECK — The guard at every gate
 *
 * Place this before every gate application and SVD contraction.
 * If the site is dormant, skip the operation entirely.
 * The lattice knows. Trust it.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* For PEPS 2D */
static inline int dyn_peps2d_active(const DynLattice *dl, int x, int y)
{
    int idx = dyn_flat(dl, x, y, 0, 0, 0, 0);
    return (dl->sites[idx].state == SITE_ACTIVE ||
            dl->sites[idx].state == SITE_FROZEN);
}

/* For TNS 3D */
static inline int dyn_tns3d_active(const DynLattice *dl, int x, int y, int z)
{
    int idx = dyn_flat(dl, x, y, z, 0, 0, 0);
    return (dl->sites[idx].state == SITE_ACTIVE ||
            dl->sites[idx].state == SITE_FROZEN);
}

/* For TNS 4D */
static inline int dyn_tns4d_active(const DynLattice *dl, int x, int y,
                                    int z, int w)
{
    int idx = dyn_flat(dl, x, y, z, w, 0, 0);
    return (dl->sites[idx].state == SITE_ACTIVE ||
            dl->sites[idx].state == SITE_FROZEN);
}

/* For TNS 5D */
static inline int dyn_tns5d_active(const DynLattice *dl, int x, int y,
                                    int z, int w, int v)
{
    int idx = dyn_flat(dl, x, y, z, w, v, 0);
    return (dl->sites[idx].state == SITE_ACTIVE ||
            dl->sites[idx].state == SITE_FROZEN);
}

/* For TNS 6D */
static inline int dyn_tns6d_active(const DynLattice *dl, int x, int y,
                                    int z, int w, int v, int u)
{
    int idx = dyn_flat(dl, x, y, z, w, v, u);
    return (dl->sites[idx].state == SITE_ACTIVE ||
            dl->sites[idx].state == SITE_FROZEN);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * ENTROPY PROBE — Feed local density into the growth engine
 *
 * After each Trotter step, call local_density() on active sites,
 * then feed the probabilities here. The lattice updates its awareness.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* For PEPS 2D */
static inline void dyn_peps2d_entropy(DynLattice *dl, int x, int y,
                                       const double *probs, int D)
{
    int idx = dyn_flat(dl, x, y, 0, 0, 0, 0);
    dyn_lattice_update_entropy(dl, idx, probs, D);
}

/* For TNS 3D */
static inline void dyn_tns3d_entropy(DynLattice *dl, int x, int y, int z,
                                      const double *probs, int D)
{
    int idx = dyn_flat(dl, x, y, z, 0, 0, 0);
    dyn_lattice_update_entropy(dl, idx, probs, D);
}

/* For TNS 4D */
static inline void dyn_tns4d_entropy(DynLattice *dl, int x, int y,
                                      int z, int w,
                                      const double *probs, int D)
{
    int idx = dyn_flat(dl, x, y, z, w, 0, 0);
    dyn_lattice_update_entropy(dl, idx, probs, D);
}

/* For TNS 5D */
static inline void dyn_tns5d_entropy(DynLattice *dl, int x, int y,
                                      int z, int w, int v,
                                      const double *probs, int D)
{
    int idx = dyn_flat(dl, x, y, z, w, v, 0);
    dyn_lattice_update_entropy(dl, idx, probs, D);
}

/* For TNS 6D */
static inline void dyn_tns6d_entropy(DynLattice *dl, int x, int y,
                                      int z, int w, int v, int u,
                                      const double *probs, int D)
{
    int idx = dyn_flat(dl, x, y, z, w, v, u);
    dyn_lattice_update_entropy(dl, idx, probs, D);
}

#endif /* QUHIT_DYN_INTEGRATE_H */
