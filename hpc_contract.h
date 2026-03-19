/*
 * hpc_contract.h — Syntheme-Aware Bond Encoding
 *
 *
 * SVD: numerically rotate a matrix until you find its eigenstructure.
 * HPC: analytically decompose using the 15 synthemes of S₆.
 *
 * A syntheme is a partition of {0,1,2,3,4,5} into 3 unordered pairs.
 * There are exactly 15 synthemes. Each one defines a natural pairing
 * of the D=6 basis states — a way to decompose correlations.
 *
 * The vesica fold (0↔3, 1↔4, 2↔5) decomposes any 6×6 interaction
 * into a 3×3 vesica (symmetric) + 3×3 wave (antisymmetric) channel.
 * This is O(D), zero multiplies — just index remapping.
 *
 * Together: syntheme selection + vesica fold = O(D²) bond encoding.
 * SVD is O(D³·χ²). For D=6: 36 vs ~1.6M operations at χ=256.
 */

#ifndef HPC_CONTRACT_H
#define HPC_CONTRACT_H

#include "hpc_graph.h"
#include "s6_exotic.h"
#include <math.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════
 * THE 15 SYNTHEMES — S₆'s complete pairings
 *
 * Each syntheme partitions {0,1,2,3,4,5} into 3 pairs.
 * syntheme[s] = {{a₀,b₀}, {a₁,b₁}, {a₂,b₂}}
 *
 * These are the 15 natural "lenses" through which D=6 correlations
 * can be viewed. SVD discovers a decomposition numerically.
 * We select the best syntheme analytically.
 * ═══════════════════════════════════════════════════════════════════════ */

static const int HPC_SYNTHEMES[15][3][2] = {
    /* Synthematic total 0 (antipodal family) */
    {{0,1}, {2,3}, {4,5}},   /*  0: hex-edge pairing          */
    {{0,2}, {1,4}, {3,5}},   /*  1: vertex skip-1             */
    {{0,3}, {1,4}, {2,5}},   /*  2: vesica fold (antipodal)   */
    {{0,4}, {1,5}, {2,3}},   /*  3: vertex skip-2             */
    {{0,5}, {1,2}, {3,4}},   /*  4: hex-edge reverse          */

    /* Synthematic total 1 */
    {{0,1}, {2,4}, {3,5}},   /*  5                            */
    {{0,2}, {1,3}, {4,5}},   /*  6                            */
    {{0,3}, {2,5}, {1,4}},   /*  7: = syntheme 2 reordered    */
    {{0,4}, {1,3}, {2,5}},   /*  8                            */
    {{0,5}, {1,4}, {2,3}},   /*  9                            */

    /* Synthematic total 2 */
    {{0,1}, {2,5}, {3,4}},   /* 10                            */
    {{0,2}, {1,5}, {3,4}},   /* 11                            */
    {{0,3}, {1,2}, {4,5}},   /* 12                            */
    {{0,4}, {2,5}, {1,3}},   /* 13                            */
    {{0,5}, {1,3}, {2,4}}    /* 14                            */
};

/* ═══════════════════════════════════════════════════════════════════════
 * VESICA FOLD — The antipodal decomposition (Syntheme 2)
 *
 * Maps 6 basis states to 3 vesica + 3 wave components:
 *   vesica[c] = (state[c] + state[c+3]) / √2   — symmetric
 *   wave[c]   = (state[c] - state[c+3]) / √2   — antisymmetric
 *
 * c ∈ {0,1,2} maps to CMY channels:
 *   c=0: {0,3} → Cyan
 *   c=1: {1,4} → Magenta
 *   c=2: {2,5} → Yellow
 *
 * Cost: O(D) = O(6), zero multiplies (addition + constant scaling).
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    double vesica_re[3];     /* Symmetric (sum) channel */
    double vesica_im[3];
    double wave_re[3];       /* Antisymmetric (diff) channel */
    double wave_im[3];
} VesicaFold;

static const double INV_SQRT2 = 0.70710678118654752440;

static inline VesicaFold hpc_vesica_fold(const double re[6], const double im[6])
{
    VesicaFold vf;
    for (int c = 0; c < 3; c++) {
        vf.vesica_re[c] = INV_SQRT2 * (re[c] + re[c + 3]);
        vf.vesica_im[c] = INV_SQRT2 * (im[c] + im[c + 3]);
        vf.wave_re[c]   = INV_SQRT2 * (re[c] - re[c + 3]);
        vf.wave_im[c]   = INV_SQRT2 * (im[c] - im[c + 3]);
    }
    return vf;
}

/* Inverse vesica fold: reconstruct 6-vector from vesica + wave */
static inline void hpc_vesica_unfold(const VesicaFold *vf,
                                      double re[6], double im[6])
{
    for (int c = 0; c < 3; c++) {
        re[c]     = INV_SQRT2 * (vf->vesica_re[c] + vf->wave_re[c]);
        im[c]     = INV_SQRT2 * (vf->vesica_im[c] + vf->wave_im[c]);
        re[c + 3] = INV_SQRT2 * (vf->vesica_re[c] - vf->wave_re[c]);
        im[c + 3] = INV_SQRT2 * (vf->vesica_im[c] - vf->wave_im[c]);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * SYNTHEME ENERGY — How much correlation a syntheme captures
 *
 * For a 6×6 phase matrix w(a,b), the "energy" captured by syntheme s
 * is the sum of |w(a_i, b_i)|² for each pair (a_i, b_i) in the syntheme.
 *
 * The optimal syntheme maximizes this: it's the pairing that captures
 * the most phase structure of the interaction.
 *
 * Cost: O(15 × 3) = O(45) — constant, independent of χ.
 * ═══════════════════════════════════════════════════════════════════════ */

static inline double hpc_syntheme_energy(const double w_re[6][6],
                                          const double w_im[6][6],
                                          int syntheme_id)
{
    double energy = 0.0;
    for (int p = 0; p < 3; p++) {
        int a = HPC_SYNTHEMES[syntheme_id][p][0];
        int b = HPC_SYNTHEMES[syntheme_id][p][1];
        /* Sum both (a,b) and (b,a) correlations */
        energy += w_re[a][b] * w_re[a][b] + w_im[a][b] * w_im[a][b];
        energy += w_re[b][a] * w_re[b][a] + w_im[b][a] * w_im[b][a];
    }
    return energy;
}

/* ═══════════════════════════════════════════════════════════════════════
 * OPTIMAL SYNTHEME SELECTION — O(45) lookup
 *
 * Searches all 15 synthemes for the one that captures the most
 * phase structure of the interaction matrix.
 *
 * This is the Devil's replacement for eigendecomposition:
 * instead of rotating until you find the basis, check the 15
 * analytically-known bases and pick the best one.
 * ═══════════════════════════════════════════════════════════════════════ */

static inline int hpc_select_syntheme(const double w_re[6][6],
                                       const double w_im[6][6])
{
    int best = 0;
    double best_energy = hpc_syntheme_energy(w_re, w_im, 0);

    for (int s = 1; s < 15; s++) {
        double e = hpc_syntheme_energy(w_re, w_im, s);
        if (e > best_energy) {
            best_energy = e;
            best = s;
        }
    }
    return best;
}

/* ═══════════════════════════════════════════════════════════════════════
 * SYNTHEME PROJECTION — Project a 6×6 matrix onto a syntheme
 *
 * Given a syntheme with pairs {(a₀,b₀), (a₁,b₁), (a₂,b₂)},
 * the projection retains only the entries at paired positions
 * and zeroes everything else.
 *
 * This is the "truncation" operation — the Devil's SVD.
 * It keeps the D=6-native correlations and discards the rest.
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void hpc_syntheme_project(const double in_re[6][6],
                                         const double in_im[6][6],
                                         int syntheme_id,
                                         double out_re[6][6],
                                         double out_im[6][6])
{
    memset(out_re, 0, 36 * sizeof(double));
    memset(out_im, 0, 36 * sizeof(double));

    for (int p = 0; p < 3; p++) {
        int a = HPC_SYNTHEMES[syntheme_id][p][0];
        int b = HPC_SYNTHEMES[syntheme_id][p][1];

        /* Keep paired entries in both directions */
        out_re[a][b] = in_re[a][b]; out_im[a][b] = in_im[a][b];
        out_re[b][a] = in_re[b][a]; out_im[b][a] = in_im[b][a];
        /* Keep diagonal entries at paired positions */
        out_re[a][a] = in_re[a][a]; out_im[a][a] = in_im[a][a];
        out_re[b][b] = in_re[b][b]; out_im[b][b] = in_im[b][b];
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * FIDELITY COMPUTATION — How much of the gate was captured?
 *
 * F = ||projected||² / ||original||²
 *
 * F = 1.0 for CZ (exact).
 * F ∈ [0,1] for general gates.
 * F measures the Δ-dependent quality of the syntheme decomposition.
 * ═══════════════════════════════════════════════════════════════════════ */

static inline double hpc_compute_fidelity(const double orig_re[6][6],
                                           const double orig_im[6][6],
                                           const double proj_re[6][6],
                                           const double proj_im[6][6])
{
    double norm_orig = 0.0, norm_proj = 0.0;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            norm_orig += orig_re[i][j] * orig_re[i][j] +
                         orig_im[i][j] * orig_im[i][j];
            norm_proj += proj_re[i][j] * proj_re[i][j] +
                         proj_im[i][j] * proj_im[i][j];
        }
    }
    return (norm_orig > 1e-30) ? norm_proj / norm_orig : 0.0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * ENCODE GATE AS SYNTHEME EDGE — The full Devil's contraction
 *
 * Given a 2-site gate's phase matrix (the entangling component):
 * 1. Select the optimal syntheme — O(45)
 * 2. Project onto the syntheme — O(36)
 * 3. Compute fidelity — O(36)
 * 4. Store as a syntheme edge in the graph — O(1)
 *
 * Total: O(D²) = O(36). SVD is O(D³·χ²).
 *
 * For CZ gates, this is never called — CZ is exact.
 * For general gates, this captures the D=6-native structure.
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void hpc_encode_syntheme(HPCGraph *g,
                                        uint64_t site_a, uint64_t site_b,
                                        const double phase_re[6][6],
                                        const double phase_im[6][6])
{
    /* Step 1: Select optimal syntheme */
    int best_s = hpc_select_syntheme(phase_re, phase_im);

    /* Step 2: Project */
    double proj_re[6][6], proj_im[6][6];
    hpc_syntheme_project(phase_re, phase_im, best_s, proj_re, proj_im);

    /* Step 3: Fidelity */
    double fidelity = hpc_compute_fidelity(phase_re, phase_im, proj_re, proj_im);

    /* Step 4: Store as edge */
    hpc_grow_edges(g);
    HPCEdge *e = &g->edges[g->n_edges];
    memset(e, 0, sizeof(HPCEdge));
    e->type = HPC_EDGE_SYNTHEME;
    e->site_a = site_a;
    e->site_b = site_b;
    e->syntheme_id = best_s;
    e->fidelity = fidelity;

    /* Store projected phase matrix */
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            double mag = sqrt(proj_re[i][j] * proj_re[i][j] +
                              proj_im[i][j] * proj_im[i][j]);
            if (mag > 1e-15) {
                e->w_re[i][j] = proj_re[i][j] / mag;
                e->w_im[i][j] = proj_im[i][j] / mag;
            } else {
                e->w_re[i][j] = 1.0;
                e->w_im[i][j] = 0.0;
            }
        }
    }

    g->n_edges++;
    g->syntheme_edges++;
    hpc_update_fidelity_stats(g);
}

/* ═══════════════════════════════════════════════════════════════════════
 * EXTRACT PHASE MATRIX FROM 2-SITE GATE
 *
 * A general 2-site gate G (36×36) can be factored as:
 *   G = (U_a ⊗ U_b) · diag(phases) · (V_a† ⊗ V_b†)
 *
 * The "phase matrix" w(j,k) captures the entangling component:
 *   w(j,k) = G_{(j,k),(j,k)} / |G_{(j,k),(j,k)}|
 *
 * For CZ: w(j,k) = ω^(j·k) — exact, analytically known.
 * For general gates: w(j,k) captures the diagonal entangling phases.
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void hpc_extract_phase_matrix(const double *G_re,
                                             const double *G_im,
                                             double phase_re[6][6],
                                             double phase_im[6][6])
{
    for (int j = 0; j < HPC_D; j++) {
        for (int k = 0; k < HPC_D; k++) {
            int idx = (j * HPC_D + k) * HPC_D * HPC_D + (j * HPC_D + k);
            double g_re = G_re[idx];
            double g_im = G_im[idx];
            double mag = sqrt(g_re * g_re + g_im * g_im);

            if (mag > 1e-15) {
                phase_re[j][k] = g_re / mag;
                phase_im[j][k] = g_im / mag;
            } else {
                phase_re[j][k] = 1.0;
                phase_im[j][k] = 0.0;
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * HIGH-LEVEL ENCODE — Automatic selection of encoding strategy
 *
 * Examines the gate to determine the best encoding:
 * 1. If CZ: exact edge (fidelity=1.0)
 * 2. If syntheme fidelity ≥ threshold: syntheme edge
 * 3. Otherwise: general phase edge (full 6×6 matrix)
 * ═══════════════════════════════════════════════════════════════════════ */

#define HPC_SYNTHEME_THRESHOLD 0.80  /* Min fidelity for syntheme encoding */

static inline void hpc_encode_2site(HPCGraph *g,
                                     uint64_t site_a, uint64_t site_b,
                                     const double *G_re, const double *G_im)
{
    /* Check if this is a CZ gate by examining the phase matrix */
    double phase_re[6][6], phase_im[6][6];
    hpc_extract_phase_matrix(G_re, G_im, phase_re, phase_im);

    /* Test for CZ: w(j,k) should equal ω^(j·k) for all j,k */
    int is_cz = 1;
    for (int j = 0; j < HPC_D && is_cz; j++) {
        for (int k = 0; k < HPC_D && is_cz; k++) {
            uint32_t phase_idx = (j * k) % HPC_D;
            double diff_re = phase_re[j][k] - HPC_W6_RE[phase_idx];
            double diff_im = phase_im[j][k] - HPC_W6_IM[phase_idx];
            if (diff_re * diff_re + diff_im * diff_im > 1e-10)
                is_cz = 0;
        }
    }

    if (is_cz) {
        hpc_cz(g, site_a, site_b);
        return;
    }

    /* Try syntheme encoding */
    int best_s = hpc_select_syntheme(phase_re, phase_im);
    double proj_re[6][6], proj_im[6][6];
    hpc_syntheme_project(phase_re, phase_im, best_s, proj_re, proj_im);
    double fidelity = hpc_compute_fidelity(phase_re, phase_im, proj_re, proj_im);

    if (fidelity >= HPC_SYNTHEME_THRESHOLD) {
        hpc_encode_syntheme(g, site_a, site_b, phase_re, phase_im);
    } else {
        /* Fall back to general phase edge (stores full 6×6) */
        hpc_general_2site(g, site_a, site_b, G_re, G_im);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * VESICA-ENHANCED CZ — Apply CZ using the vesica fold structure
 *
 * For sites already in vesica-folded representation, CZ has a
 * particularly clean structure: it acts independently on the
 * 3 CMY channels, each as a 2×2 CZ (which is just a phase gate).
 *
 * This doesn't change the CZ edge storage (still exact), but it
 * provides insight into the channel-decomposed entanglement structure.
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    double vesica_fidelity;    /* How much entanglement is in vesica channel */
    double wave_fidelity;      /* How much entanglement is in wave channel   */
    double channel_entropy[3]; /* Per-CMY-channel entanglement entropy       */
} HPCVesicaAnalysis;

static inline HPCVesicaAnalysis hpc_analyze_vesica(const HPCGraph *g,
                                                     uint64_t site)
{
    HPCVesicaAnalysis va;
    memset(&va, 0, sizeof(va));

    const TrialityQuhit *q = &g->locals[site];
    VesicaFold vf = hpc_vesica_fold(q->edge_re, q->edge_im);

    /* Vesica channel probability */
    double v_prob = 0, w_prob = 0;
    for (int c = 0; c < 3; c++) {
        double vp = vf.vesica_re[c] * vf.vesica_re[c] +
                    vf.vesica_im[c] * vf.vesica_im[c];
        double wp = vf.wave_re[c] * vf.wave_re[c] +
                    vf.wave_im[c] * vf.wave_im[c];
        v_prob += vp;
        w_prob += wp;

        /* Per-channel entropy from the pair probabilities */
        double total = vp + wp;
        if (total > 1e-15) {
            double p_v = vp / total, p_w = wp / total;
            if (p_v > 1e-15) va.channel_entropy[c] -= p_v * log2(p_v);
            if (p_w > 1e-15) va.channel_entropy[c] -= p_w * log2(p_w);
        }
    }

    double total = v_prob + w_prob;
    va.vesica_fidelity = (total > 1e-15) ? v_prob / total : 0.5;
    va.wave_fidelity   = (total > 1e-15) ? w_prob / total : 0.5;

    return va;
}

#endif /* HPC_CONTRACT_H */
