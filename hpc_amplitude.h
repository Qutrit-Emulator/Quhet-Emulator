/*
 * hpc_amplitude.h — On-Demand State Vector
 *
 * The state vector has D^N entries. We never materialize it.
 * Instead, we compute exactly what's needed, when it's needed.
 *
 * Three modes of access:
 *
 * 1. POINT QUERY:    ψ(i₁,...,iₙ) → O(N+E)     — one amplitude
 * 2. SPARSE RECON:   All |ψ| > threshold → O(?)  — importance sampling
 * 3. EXPECTATION:    ⟨ψ|O|ψ⟩ → O(samples×(N+E)) — Monte Carlo
 *
 * The Devil computes only what you ask for. Nothing more.
 * The rest of the state vector does not exist until observed.
 */

#ifndef HPC_AMPLITUDE_H
#define HPC_AMPLITUDE_H

#include "hpc_graph.h"
#include "hpc_contract.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════
 * SPARSE STATE VECTOR ENTRY
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    uint32_t *indices;   /* Site indices: [n_sites]            */
    double    re, im;    /* Amplitude value                    */
    double    prob;      /* |amplitude|²                       */
} HPCSparseEntry;

typedef struct {
    HPCSparseEntry *entries;
    uint64_t        count;
    uint64_t        capacity;
    uint64_t        n_sites;    /* For index array sizing       */
    double          total_prob; /* Sum of captured probability  */
    double          threshold;  /* Minimum |ψ|² captured        */
} HPCSparseVector;

/* ═══════════════════════════════════════════════════════════════════════
 * SPARSE VECTOR LIFECYCLE
 * ═══════════════════════════════════════════════════════════════════════ */

static inline HPCSparseVector *hpc_sv_create(uint64_t n_sites,
                                              uint64_t initial_cap)
{
    HPCSparseVector *sv = (HPCSparseVector *)calloc(1, sizeof(HPCSparseVector));
    if (!sv) return NULL;
    sv->n_sites = n_sites;
    sv->capacity = initial_cap;
    sv->entries = (HPCSparseEntry *)calloc(initial_cap, sizeof(HPCSparseEntry));
    for (uint64_t i = 0; i < initial_cap; i++)
        sv->entries[i].indices = (uint32_t *)calloc(n_sites, sizeof(uint32_t));
    return sv;
}

static inline void hpc_sv_destroy(HPCSparseVector *sv)
{
    if (!sv) return;
    for (uint64_t i = 0; i < sv->capacity; i++)
        free(sv->entries[i].indices);
    free(sv->entries);
    free(sv);
}

static inline void hpc_sv_grow(HPCSparseVector *sv)
{
    if (sv->count < sv->capacity) return;
    uint64_t new_cap = sv->capacity * 2;
    sv->entries = (HPCSparseEntry *)realloc(sv->entries,
                                             new_cap * sizeof(HPCSparseEntry));
    for (uint64_t i = sv->capacity; i < new_cap; i++) {
        sv->entries[i].indices = (uint32_t *)calloc(sv->n_sites, sizeof(uint32_t));
        sv->entries[i].re = 0; sv->entries[i].im = 0; sv->entries[i].prob = 0;
    }
    sv->capacity = new_cap;
}

static inline void hpc_sv_add(HPCSparseVector *sv,
                               const uint32_t *indices,
                               double re, double im)
{
    hpc_sv_grow(sv);
    HPCSparseEntry *e = &sv->entries[sv->count];
    memcpy(e->indices, indices, sv->n_sites * sizeof(uint32_t));
    e->re = re;
    e->im = im;
    e->prob = re * re + im * im;
    sv->total_prob += e->prob;
    sv->count++;
}

/* ═══════════════════════════════════════════════════════════════════════
 * BRUTE-FORCE SPARSE RECONSTRUCTION
 *
 * For small N: enumerate all D^N configurations, keep those above
 * threshold. Returns a sparse vector of significant amplitudes.
 *
 * Cost: O(D^N × (N+E)) — exponential, small N only.
 * This is the reference implementation for verification.
 * ═══════════════════════════════════════════════════════════════════════ */

static inline HPCSparseVector *hpc_sparse_brute(const HPCGraph *g,
                                                 double threshold,
                                                 uint64_t max_entries)
{
    if (g->n_sites > 8) {
        fprintf(stderr, "hpc_sparse_brute: N=%lu too large\n", g->n_sites);
        return NULL;
    }

    HPCSparseVector *sv = hpc_sv_create(g->n_sites, 256);
    if (!sv) return NULL;
    sv->threshold = threshold;

    uint64_t total_configs = 1;
    for (uint64_t i = 0; i < g->n_sites; i++) total_configs *= HPC_D;

    uint32_t indices[8];

    for (uint64_t cfg = 0; cfg < total_configs && sv->count < max_entries; cfg++) {
        uint64_t tmp = cfg;
        for (uint64_t i = 0; i < g->n_sites; i++) {
            indices[i] = tmp % HPC_D;
            tmp /= HPC_D;
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
 * TREE-PRUNED SPARSE RECONSTRUCTION
 *
 * For larger N: build the state vector site-by-site, pruning branches
 * whose cumulative probability falls below threshold.
 *
 * At each site k, we have a set of "live" partial configurations
 * (i₁,...,i_k) with accumulated amplitude. For site k+1, we extend
 * each live config to all D values, compute the new amplitude, and
 * prune low-probability branches.
 *
 * Cost: O(active_branches × D × E_local) per site.
 * For sparse states: active_branches << D^k → exponential speedup.
 *
 * This is the practical reconstruction method for N > 8.
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    uint32_t *indices;   /* Partial index vector [n_sites]     */
    double    re, im;    /* Accumulated amplitude               */
} HPCTreeNode;

static inline HPCSparseVector *hpc_sparse_tree(const HPCGraph *g,
                                                double threshold,
                                                uint64_t max_branches)
{
    HPCSparseVector *sv = hpc_sv_create(g->n_sites, 256);
    if (!sv) return NULL;
    sv->threshold = threshold;

    /* Initial pool: one root node with no sites assigned */
    uint64_t pool_cap = max_branches * HPC_D + 16;
    HPCTreeNode *current = (HPCTreeNode *)calloc(pool_cap, sizeof(HPCTreeNode));
    HPCTreeNode *next    = (HPCTreeNode *)calloc(pool_cap, sizeof(HPCTreeNode));
    for (uint64_t i = 0; i < pool_cap; i++) {
        current[i].indices = (uint32_t *)calloc(g->n_sites, sizeof(uint32_t));
        next[i].indices    = (uint32_t *)calloc(g->n_sites, sizeof(uint32_t));
    }

    /* Seed: one root node */
    uint64_t n_current = 1;
    current[0].re = 1.0;
    current[0].im = 0.0;

    /* Grow site by site */
    for (uint64_t site = 0; site < g->n_sites; site++) {
        uint64_t n_next = 0;
        const TrialityQuhit *q = &g->locals[site];

        for (uint64_t b = 0; b < n_current; b++) {
            for (int v = 0; v < HPC_D; v++) {
                /* Extend branch with site=v */
                double a_re = q->edge_re[v];
                double a_im = q->edge_im[v];

                /* Multiply accumulated amplitude by local amplitude */
                double new_re = current[b].re * a_re - current[b].im * a_im;
                double new_im = current[b].re * a_im + current[b].im * a_re;

                /* Apply phase contributions from edges connecting
                 * this site to already-assigned sites */
                for (uint64_t e = 0; e < g->n_edges; e++) {
                    uint64_t sa = g->edges[e].site_a;
                    uint64_t sb = g->edges[e].site_b;
                    int partner_site = -1;

                    if (sa == site && sb < site) partner_site = (int)sb;
                    else if (sb == site && sa < site) partner_site = (int)sa;

                    if (partner_site >= 0) {
                        uint32_t pv = current[b].indices[partner_site];
                        double w_re, w_im;

                        if (g->edges[e].type == HPC_EDGE_CZ) {
                            uint32_t phase_idx = ((uint32_t)v * pv) % HPC_D;
                            w_re = HPC_W6_RE[phase_idx];
                            w_im = HPC_W6_IM[phase_idx];
                        } else {
                            if (sa == site) {
                                w_re = g->edges[e].w_re[v][pv];
                                w_im = g->edges[e].w_im[v][pv];
                            } else {
                                w_re = g->edges[e].w_re[pv][v];
                                w_im = g->edges[e].w_im[pv][v];
                            }
                        }

                        double tmp_re = new_re * w_re - new_im * w_im;
                        double tmp_im = new_re * w_im + new_im * w_re;
                        new_re = tmp_re;
                        new_im = tmp_im;
                    }
                }

                /* Prune: skip if amplitude is too small */
                double prob = new_re * new_re + new_im * new_im;
                if (prob < threshold && site < g->n_sites - 1) continue;

                /* Accept this branch */
                if (n_next < pool_cap) {
                    memcpy(next[n_next].indices, current[b].indices,
                           g->n_sites * sizeof(uint32_t));
                    next[n_next].indices[site] = v;
                    next[n_next].re = new_re;
                    next[n_next].im = new_im;
                    n_next++;
                }
            }
        }

        /* Swap pools */
        HPCTreeNode *tmp = current;
        current = next;
        next = tmp;
        n_current = n_next;

        /* Sort by probability and truncate to max_branches */
        if (n_current > max_branches && site < g->n_sites - 1) {
            /* Simple selection: keep top max_branches by probability */
            /* Partial sort using partition around threshold */
            for (uint64_t i = max_branches; i < n_current; i++) {
                /* Find minimum in kept set */
                uint64_t min_idx = 0;
                double min_prob = current[0].re * current[0].re +
                                  current[0].im * current[0].im;
                for (uint64_t j = 1; j < max_branches; j++) {
                    double p = current[j].re * current[j].re +
                               current[j].im * current[j].im;
                    if (p < min_prob) { min_prob = p; min_idx = j; }
                }
                /* Swap if current[i] is larger */
                double p_i = current[i].re * current[i].re +
                             current[i].im * current[i].im;
                if (p_i > min_prob) {
                    HPCTreeNode swap = current[min_idx];
                    current[min_idx] = current[i];
                    current[i] = swap;
                }
            }
            n_current = max_branches;
        }
    }

    /* All remaining branches are complete configurations */
    for (uint64_t b = 0; b < n_current; b++) {
        double prob = current[b].re * current[b].re +
                      current[b].im * current[b].im;
        if (prob >= threshold)
            hpc_sv_add(sv, current[b].indices, current[b].re, current[b].im);
    }

    /* Cleanup */
    for (uint64_t i = 0; i < pool_cap; i++) {
        free(current[i].indices);
        free(next[i].indices);
    }
    free(current);
    free(next);

    return sv;
}

/* ═══════════════════════════════════════════════════════════════════════
 * MONTE CARLO EXPECTATION VALUE
 *
 * Computes ⟨ψ|O|ψ⟩ via importance sampling without materializing |ψ⟩.
 *
 * Strategy:
 * 1. Sample configurations by measuring each site sequentially
 *    using Born probabilities (marginals from the graph)
 * 2. For each sample, evaluate ψ(config) and O(config)
 * 3. Average over samples
 *
 * For diagonal observables O = Σ_i o(i)|i⟩⟨i|:
 *   ⟨O⟩ = Σ_i |ψ(i)|² o(i) ≈ (1/S) Σ_{samples} o(i_s)
 *
 * Cost: O(n_samples × (N + E))
 * ═══════════════════════════════════════════════════════════════════════ */

typedef double (*HPCObservable)(const uint32_t *indices, uint64_t n_sites,
                                 void *ctx);

static inline double hpc_expectation(const HPCGraph *g,
                                      HPCObservable obs, void *obs_ctx,
                                      int n_samples, uint64_t rng_seed)
{
    /* Simple LCG for reproducible sampling */
    uint64_t rng = rng_seed;
    #define HPC_LCG(r) ((r) = (r) * 6364136223846793005ULL + 1442695040888963407ULL)
    #define HPC_RAND(r) (((double)((r) >> 11)) * 0x1.0p-53)

    double sum_obs = 0.0;
    int valid_samples = 0;

    for (int s = 0; s < n_samples; s++) {
        /* Generate a configuration by sampling site-by-site */
        uint32_t config[256]; /* max sites for MC */
        if (g->n_sites > 256) break;

        /* Simple approach: sample each site from its local distribution.
         * This is approximate for entangled states but fast. */
        for (uint64_t site = 0; site < g->n_sites; site++) {
            const TrialityQuhit *q = &g->locals[site];

            /* Local probability distribution */
            double probs[HPC_D];
            double total = 0;
            for (int v = 0; v < HPC_D; v++) {
                probs[v] = q->edge_re[v] * q->edge_re[v] +
                           q->edge_im[v] * q->edge_im[v];
                total += probs[v];
            }

            /* Sample from local distribution */
            HPC_LCG(rng);
            double r = HPC_RAND(rng) * total;
            double cumul = 0;
            config[site] = HPC_D - 1;
            for (int v = 0; v < HPC_D; v++) {
                cumul += probs[v];
                if (r <= cumul) { config[site] = v; break; }
            }
        }

        /* Compute importance weight: |ψ(config)|² / q(config)
         * where q = Π_k p_k(config[k]) is the proposal distribution */
        double prob_psi = hpc_probability(g, config);
        double prob_q = 1.0;
        for (uint64_t site = 0; site < g->n_sites; site++) {
            const TrialityQuhit *q = &g->locals[site];
            uint32_t v = config[site];
            double p = q->edge_re[v] * q->edge_re[v] +
                       q->edge_im[v] * q->edge_im[v];
            prob_q *= p;
        }

        if (prob_q > 1e-30) {
            double weight = prob_psi / prob_q;
            double obs_val = obs(config, g->n_sites, obs_ctx);
            sum_obs += weight * obs_val;
            valid_samples++;
        }
    }

    #undef HPC_LCG
    #undef HPC_RAND

    return (valid_samples > 0) ? sum_obs / valid_samples : 0.0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * PRINT SPARSE VECTOR
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void hpc_sv_print(const HPCSparseVector *sv, int max_show)
{
    printf("── Sparse State Vector ──\n");
    printf("  Entries: %lu, Captured prob: %.6f, Threshold: %.2e\n",
           sv->count, sv->total_prob, sv->threshold);

    uint64_t show = sv->count;
    if (max_show > 0 && show > (uint64_t)max_show) show = max_show;

    for (uint64_t i = 0; i < show; i++) {
        printf("  |");
        for (uint64_t s = 0; s < sv->n_sites; s++)
            printf("%u", sv->entries[i].indices[s]);
        printf("⟩ → %.6f%+.6fi  (P=%.6e)\n",
               sv->entries[i].re, sv->entries[i].im, sv->entries[i].prob);
    }
    if (show < sv->count)
        printf("  ... (%lu more entries)\n", sv->count - show);
}

#endif /* HPC_AMPLITUDE_H */
