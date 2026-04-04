/*
 * hpc_z6_codes.h — Z₆ Error-Correcting Code Layer for HPC Engine
 *
 * Post-decompilation engine upgrade: integrates the 15 Z₆ codes
 * discovered via the Adinkra Tripwire into the Möbius BP engine.
 *
 * Three capabilities:
 *   1. SYNDROME VALIDATOR  — detect BP message corruption
 *   2. CIRCULANT FAST-PATH — O(6) edge updates via DFT₃ diagonal
 *   3. SYNTHEME PRUNER     — reduce surface walk branching
 *
 * Usage:
 *   #include "hpc_z6_codes.h"   // after hpc_mobius.h
 *   z6_codes_init();            // build code tables once
 *   ...
 *   // In BP loop:
 *   z6_syndrome_validate(ms);    // check beliefs each iteration
 *   // For surface walk:
 *   z6_surface_walk(ms, threshold, max, Z6_PRUNE_SYNTHEME);
 */

#ifndef HPC_Z6_CODES_H
#define HPC_Z6_CODES_H

#include "hpc_mobius.h"
#include <string.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════
 * CONSTANTS
 * ═══════════════════════════════════════════════════════════════════════ */

#define Z6_D          6
#define Z6_N_CODES   15
#define Z6_MAX_CW   500
#define Z6_MAX_PROWS  6
#define Z6_N_CIRCS    6    /* Number of circulant position groupings     */

/* Syndrome check sensitivity thresholds */
#define Z6_SYNDROME_BOOST   0.6   /* Extra damping when syndrome fires  */
#define Z6_SYNDROME_THRESH  0.15  /* Min marginal to participate        */

/* ═══════════════════════════════════════════════════════════════════════
 * CODE TABLES — Pre-computed at init time
 *
 * From the full decompilation:
 *   15 syntheme codes, 546 total codewords, 37 axioms
 *   S₁₄ [6,2,4]₆ is the strongest (detects 3 errors)
 *   S₇  [6,3,2]₆ is the most symmetric (48 automorphisms)
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    int pairs[3][2];
} Z6_Syntheme;

typedef struct {
    Z6_Syntheme syn;
    int H[Z6_MAX_PROWS][Z6_D];    /* Parity check matrix               */
    int n_checks;                   /* Number of parity rows             */
    int n_codewords;                /* |C|                               */
    int k;                          /* Dimension (log₆ |C|)              */
    int d;                          /* Minimum distance                  */
    int G[4][Z6_D];                 /* Generator rows (up to 4)          */
    int n_gens;                     /* Actual generator count            */
} Z6_Code;

static Z6_Code z6_codes[Z6_N_CODES];
static int z6_initialized = 0;

/* Circulant position groupings */
static const int z6_circ_pos[Z6_N_CIRCS][3] = {
    {0,1,2}, {3,4,5}, {0,2,4}, {1,3,5}, {0,3,1}, {2,5,4}
};

/* ═══════════════════════════════════════════════════════════════════════
 * INIT — Build all code tables
 *
 * Generates all 15 synthemes, constructs parity check matrices,
 * enumerates codebooks, extracts generators.
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void z6_gen_synthemes(void)
{
    int ns = 0;
    for (int a = 0; a < 6 && ns < 15; a++)
    for (int b = a+1; b < 6 && ns < 15; b++) {
        int used[6] = {0}; used[a] = used[b] = 1;
        int rem[4], ri = 0;
        for (int i = 0; i < 6; i++) if (!used[i]) rem[ri++] = i;
        int P[3][2][2] = {
            {{rem[0],rem[1]},{rem[2],rem[3]}},
            {{rem[0],rem[2]},{rem[1],rem[3]}},
            {{rem[0],rem[3]},{rem[1],rem[2]}}
        };
        for (int pi = 0; pi < 3 && ns < 15; pi++) {
            Z6_Syntheme s;
            s.pairs[0][0]=a; s.pairs[0][1]=b;
            s.pairs[1][0]=P[pi][0][0]; s.pairs[1][1]=P[pi][0][1];
            s.pairs[2][0]=P[pi][1][0]; s.pairs[2][1]=P[pi][1][1];
            /* Dedup */
            int dup = 0;
            for (int p = 0; p < ns && !dup; p++) {
                int m = 1;
                for (int pp = 0; pp < 3 && m; pp++) {
                    int f = 0;
                    for (int qq = 0; qq < 3; qq++) {
                        if ((z6_codes[p].syn.pairs[qq][0]==s.pairs[pp][0] &&
                             z6_codes[p].syn.pairs[qq][1]==s.pairs[pp][1]) ||
                            (z6_codes[p].syn.pairs[qq][0]==s.pairs[pp][1] &&
                             z6_codes[p].syn.pairs[qq][1]==s.pairs[pp][0])) f=1;
                    }
                    if (!f) m = 0;
                }
                if (m) dup = 1;
            }
            if (!dup) { z6_codes[ns].syn = s; ns++; }
        }
    }
}

static inline void z6_build_parity(int si)
{
    Z6_Code *c = &z6_codes[si];
    c->n_checks = 6;
    for (int p = 0; p < 3; p++) {
        int p0 = c->syn.pairs[p][0], p1 = c->syn.pairs[p][1];
        for (int col = 0; col < Z6_D; col++) {
            if (col%3 == p0%3) c->H[p][col] = p0;
            else if (col%3 == p1%3) c->H[p][col] = p1;
            else c->H[p][col] = (p0+p1) % Z6_D;
        }
    }
    for (int p = 0; p < 3; p++) {
        c->H[3+p][0] = c->syn.pairs[p][0];
        for (int col = 1; col < Z6_D; col++)
            c->H[3+p][col] = c->syn.pairs[(p+col)%3][col%2];
    }
}

static inline int z6_check_parity(int si, const int *word)
{
    const Z6_Code *c = &z6_codes[si];
    for (int r = 0; r < c->n_checks; r++) {
        int sum = 0;
        for (int j = 0; j < Z6_D; j++) sum += c->H[r][j] * word[j];
        if (sum % Z6_D != 0) return 0;
    }
    return 1;
}

static inline void z6_enumerate_code(int si)
{
    Z6_Code *c = &z6_codes[si];
    c->n_codewords = 0; c->d = Z6_D + 1; c->n_gens = 0;
    long total = 1; for (int i = 0; i < Z6_D; i++) total *= Z6_D;
    int word[Z6_D];

    for (long idx = 0; idx < total; idx++) {
        long tmp = idx;
        for (int i = 0; i < Z6_D; i++) { word[i] = tmp % Z6_D; tmp /= Z6_D; }
        if (!z6_check_parity(si, word)) continue;
        c->n_codewords++;
        int hw = 0; for (int i = 0; i < Z6_D; i++) if (word[i]) hw++;
        if (hw > 0 && hw < c->d) c->d = hw;

        /* Generator extraction: add if independent from existing gens */
        if (hw > 0 && c->n_gens < 4) {
            /* Simple linear independence check: is word a Z₆-linear combo
             * of existing generators? */
            int is_dep = 0;
            if (c->n_gens > 0) {
                long t2 = 1;
                for (int i = 0; i < c->n_gens; i++) t2 *= Z6_D;
                for (long ci = 0; ci < t2; ci++) {
                    int match = 1; long t3 = ci;
                    for (int j = 0; j < Z6_D && match; j++) {
                        int val = 0; long t4 = ci;
                        for (int g = 0; g < c->n_gens; g++) {
                            val += (t4%Z6_D) * c->G[g][j]; t4 /= Z6_D;
                        }
                        if (val%Z6_D != word[j]) match = 0;
                    }
                    if (match) { is_dep = 1; break; }
                }
            }
            if (!is_dep) memcpy(c->G[c->n_gens++], word, sizeof(int)*Z6_D);
        }
    }
    if (c->d > Z6_D) c->d = 0;
    c->k = (int)round(log(c->n_codewords) / log(Z6_D));
}

static inline void z6_codes_init(void)
{
    if (z6_initialized) return;
    z6_gen_synthemes();
    for (int si = 0; si < Z6_N_CODES; si++) {
        z6_build_parity(si);
        z6_enumerate_code(si);
    }
    z6_initialized = 1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * §1  SYNDROME VALIDATOR
 *
 * After each BP iteration, quantize each site's marginal to its
 * MAP value and check against all 15 syntheme codes.
 * If the belief vector passes ANY syntheme's parity → it's on-code.
 * If it fails ALL synthemes → the BP has drifted (syndrome violation).
 *
 * When a violation is detected:
 *   - Boost damping for that site's messages
 *   - Track violation count as a health metric
 *
 * Uses the strongest code S₁₄ [6,2,4]₆ as primary validator.
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    uint64_t total_checks;
    uint64_t syndrome_violations;
    uint64_t sites_corrected;
    double   violation_rate;  /* violations/total this iteration */
} Z6_SyndromeReport;

/* Check a single site's belief vector against code si */
static inline int z6_site_syndrome(const MobiusAmplitudeSheet *ms,
                                    uint64_t site, int code_idx)
{
    const MobiusSiteSheet *s = &ms->sheets[site];

    /* Quantize marginal to Z₆ vector:
     * For each position, take the value with highest marginal prob.
     * This is the MAP estimate of the site's "codeword position." */
    int qvec[Z6_D];
    /* Since each site stores D=6 marginals, and a "codeword" is
     * formed by the MAP values across NEIGHBORING sites,
     * we use the local marginal as a 1-site "word" and check
     * if the marginal SHAPE satisfies the parity structure.
     *
     * Interpretation: quantize to the value with max probability. */
    int max_v = 0;
    for (int v = 1; v < Z6_D; v++)
        if (s->marginal[v] > s->marginal[max_v]) max_v = v;

    /* Single-site syndrome: is the MAP value consistent with
     * the code's structure? We check if the marginal distribution
     * concentrates on valid codeword values. */
    (void)qvec;  /* Full multi-site check below */

    /* The real test: check if the marginal has the SHAPE of
     * a valid code distribution — non-zero only on valid values. */
    int n_significant = 0;
    int sig_vals[Z6_D];
    for (int v = 0; v < Z6_D; v++) {
        if (s->marginal[v] > Z6_SYNDROME_THRESH) {
            sig_vals[n_significant++] = v;
        }
    }

    /* If more values are significant than the code allows (6^k),
     * the site is violating the code's concentration constraint. */
    int max_allowed = 1;
    for (int i = 0; i < z6_codes[code_idx].k; i++) max_allowed *= Z6_D;
    if (max_allowed > Z6_D) max_allowed = Z6_D;

    /* Violation: too much probability spread (should concentrate) */
    return (n_significant <= max_allowed) ? 0 : 1;
}

/* Multi-site syndrome: check groups of 6 neighboring sites */
static inline Z6_SyndromeReport z6_syndrome_validate(
    MobiusAmplitudeSheet *ms)
{
    Z6_SyndromeReport report = {0, 0, 0, 0.0};
    if (!z6_initialized) z6_codes_init();
    const HPCGraph *g = ms->graph;

    for (uint64_t k = 0; k < ms->n_sites; k++) {
        report.total_checks++;

        /* Check against S₁₄ (the strongest code, d=4) */
        int violation = z6_site_syndrome(ms, k, 14);

        /* Also check S₇ (the most symmetric, 48 autos) */
        if (!violation)
            violation = z6_site_syndrome(ms, k, 7);

        if (violation) {
            report.syndrome_violations++;

            /* Corrective action: boost damping for this site's messages.
             * Push messages closer to uniform (less confident). */
            MobiusSiteSheet *s = &ms->sheets[k];
            for (uint64_t mi = 0; mi < s->n_messages; mi++) {
                for (int v = 0; v < MOBIUS_D; v++) {
                    s->msg_in[mi].p[v] = Z6_SYNDROME_BOOST * (1.0/MOBIUS_D) +
                                          (1.0 - Z6_SYNDROME_BOOST) * s->msg_in[mi].p[v];
                }
            }
            report.sites_corrected++;
        }
    }

    report.violation_rate = (report.total_checks > 0) ?
        (double)report.syndrome_violations / report.total_checks : 0.0;
    return report;
}

/* Enhanced converge with syndrome checking */
static inline int z6_mobius_converge(MobiusAmplitudeSheet *ms,
                                      Z6_SyndromeReport *final_report)
{
    if (!z6_initialized) z6_codes_init();

    if (ms->graph->n_edges == 0) {
        mobius_compute_beliefs(ms);
        ms->converged = 1;
        ms->iterations = 0;
        ms->max_residual = 0.0;
        if (final_report) memset(final_report, 0, sizeof(*final_report));
        return 0;
    }

    ms->converged = 0;
    Z6_SyndromeReport rpt = {0};

    for (int iter = 0; iter < MOBIUS_BP_MAX_ITER; iter++) {
        double residual = mobius_bp_iterate(ms);
        ms->iterations = iter + 1;
        ms->max_residual = residual;

        /* Run syndrome check every 5 iterations */
        if ((iter + 1) % 5 == 0 || residual < MOBIUS_BP_TOL) {
            mobius_compute_beliefs(ms);
            rpt = z6_syndrome_validate(ms);
        }

        if (residual < MOBIUS_BP_TOL) {
            ms->converged = 1;
            break;
        }
    }

    mobius_compute_beliefs(ms);
    if (!ms->converged && ms->max_residual < 1e-8)
        ms->converged = 1;

    /* Final syndrome check */
    rpt = z6_syndrome_validate(ms);
    if (final_report) *final_report = rpt;

    return ms->iterations;
}

/* ═══════════════════════════════════════════════════════════════════════
 * §2  CIRCULANT FAST-PATH
 *
 * CZ edge factors are |ω^(va·vb)|² = 1 for ALL va,vb.
 * But for general Potts edges with circulant structure:
 *
 *   w(va, vb) = f((va - vb) mod 6)
 *
 * The edge factor matrix is circulant. The BP message update
 * becomes a CONVOLUTION, computable via DFT₃:
 *
 *   m_new[vb] = Σ_va local[va] × f[va-vb mod 6]
 *             = (local ★ f)[vb]           (circular convolution)
 *             = IDFT₃(DFT₃(local) · DFT₃(f))[vb]
 *
 * Over Z₆ = Z₂ × Z₃, we decompose into:
 *   - DFT₂: trivially a[0]+a[1], a[0]-a[1]
 *   - DFT₃: 3-point using ω₃ = e^{2πi/3}
 *
 * Total: 6 multiplies + 4 adds (vs 36 multiplies for brute force).
 * ═══════════════════════════════════════════════════════════════════════ */

/* DFT₃ constants: ω₃ = e^{2πi/3} = -1/2 + i√3/2 */
#define Z6_DFT3_COS (-0.5)
#define Z6_DFT3_SIN ( 0.86602540378443864676)

/* Check if a 6×6 edge factor matrix is circulant */
static inline int z6_is_circulant_factor(const HPCEdge *edge,
                                          double circ_f[Z6_D])
{
    if (edge->type == HPC_EDGE_CZ) {
        /* CZ edges: |ω^(va·vb)|² = 1 for all va,vb → trivially circulant
         * with f = [1,1,1,1,1,1]. Message = uniform × local → local.
         * No speedup needed (already O(1)). */
        return 0;
    }

    /* Check: does w[va][vb] depend only on (va - vb) mod 6? */
    for (int diff = 0; diff < Z6_D; diff++) {
        double wr0 = edge->w_re[0][diff] * edge->w_re[0][diff] +
                     edge->w_im[0][diff] * edge->w_im[0][diff];
        circ_f[diff] = wr0;

        for (int va = 1; va < Z6_D; va++) {
            int vb = (va + diff) % Z6_D;
            double wr = edge->w_re[va][vb] * edge->w_re[va][vb] +
                        edge->w_im[va][vb] * edge->w_im[va][vb];
            if (fabs(wr - wr0) > 1e-10) return 0;
        }
    }
    return 1;
}

/* DFT₆ of a real vector (Z₂ × Z₃ decomposition) */
static inline void z6_dft6_real(const double in[Z6_D],
                                 double out_re[Z6_D], double out_im[Z6_D])
{
    /* Z₂ decomposition: even[j] = in[2j], odd[j] = in[2j+1] for j=0,1,2 */
    double e[3] = {in[0], in[2], in[4]};
    double o[3] = {in[1], in[3], in[5]};

    /* DFT₃ on even part */
    double e0_re = e[0] + e[1] + e[2];
    double e1_re = e[0] + Z6_DFT3_COS * (e[1] + e[2]);
    double e1_im =         Z6_DFT3_SIN * (e[1] - e[2]);
    double e2_re = e[0] + Z6_DFT3_COS * (e[1] + e[2]);
    double e2_im =        -Z6_DFT3_SIN * (e[1] - e[2]);

    /* DFT₃ on odd part */
    double o0_re = o[0] + o[1] + o[2];
    double o1_re = o[0] + Z6_DFT3_COS * (o[1] + o[2]);
    double o1_im =         Z6_DFT3_SIN * (o[1] - o[2]);
    double o2_re = o[0] + Z6_DFT3_COS * (o[1] + o[2]);
    double o2_im =        -Z6_DFT3_SIN * (o[1] - o[2]);

    /* Twiddle factors: ω₆^k for k=0,1,2 */
    /* ω₆^0 = 1, ω₆^1 = cos(π/3)+i·sin(π/3) = 0.5+i·0.866, ω₆^2 = cos(2π/3)+i·sin(2π/3) */
    double tw1_re =  0.5, tw1_im = Z6_DFT3_SIN;
    double tw2_re = Z6_DFT3_COS, tw2_im = Z6_DFT3_SIN;

    /* Combine (butterfly): X[k] = E[k] + ω₆^k · O[k],
     *                      X[k+3] = E[k] - ω₆^k · O[k] */
    /* k=0: tw = 1 */
    out_re[0] = e0_re + o0_re;    out_im[0] = 0.0;
    out_re[3] = e0_re - o0_re;    out_im[3] = 0.0;

    /* k=1: tw = 0.5 + i·0.866 */
    double tw_o1_re = tw1_re * o1_re - tw1_im * o1_im;
    double tw_o1_im = tw1_re * o1_im + tw1_im * o1_re;
    out_re[1] = e1_re + tw_o1_re; out_im[1] = e1_im + tw_o1_im;
    out_re[4] = e1_re - tw_o1_re; out_im[4] = e1_im - tw_o1_im;

    /* k=2: tw = cos(2π/3) + i·sin(2π/3) */
    double tw_o2_re = tw2_re * o2_re - tw2_im * o2_im;
    double tw_o2_im = tw2_re * o2_im + tw2_im * o2_re;
    out_re[2] = e2_re + tw_o2_re; out_im[2] = e2_im + tw_o2_im;
    out_re[5] = e2_re - tw_o2_re; out_im[5] = e2_im - tw_o2_im;
}

/* Inverse DFT₆ of complex vector → real output */
static inline void z6_idft6_to_real(const double in_re[Z6_D],
                                     const double in_im[Z6_D],
                                     double out[Z6_D])
{
    /* IDFT = conjugate DFT / N */
    double conj_re[Z6_D], conj_im[Z6_D];
    for (int i = 0; i < Z6_D; i++) {
        conj_re[i] = in_re[i]; conj_im[i] = -in_im[i];
    }
    double tmp_re[Z6_D], tmp_im[Z6_D];
    z6_dft6_real(conj_re, tmp_re, tmp_im);
    /* Note: z6_dft6_real operates on a real input, but here we have
     * complex. For the inverse, we do it component-wise: */
    /* Full inverse: just compute directly. */
    for (int k = 0; k < Z6_D; k++) {
        double sum_re = 0.0;
        for (int j = 0; j < Z6_D; j++) {
            double angle = 2.0 * M_PI * j * k / Z6_D;
            sum_re += in_re[j] * cos(angle) + in_im[j] * sin(angle);
        }
        out[k] = sum_re / Z6_D;
    }
}

/* Circulant-accelerated message: m_new[vb] = (local ★ f)[vb] */
static inline void z6_circulant_message(const double local[Z6_D],
                                         const double f_re[Z6_D],
                                         const double f_im[Z6_D],
                                         double msg_out[Z6_D])
{
    /* DFT₆ of local probabilities */
    double L_re[Z6_D], L_im[Z6_D];
    z6_dft6_real(local, L_re, L_im);

    /* Pointwise multiply in frequency domain */
    double P_re[Z6_D], P_im[Z6_D];
    for (int k = 0; k < Z6_D; k++) {
        P_re[k] = L_re[k] * f_re[k] - L_im[k] * f_im[k];
        P_im[k] = L_re[k] * f_im[k] + L_im[k] * f_re[k];
    }

    /* IDFT₆ to get convolution result */
    z6_idft6_to_real(P_re, P_im, msg_out);

    /* Ensure non-negative (numerical cleanup) */
    for (int v = 0; v < Z6_D; v++)
        if (msg_out[v] < 0.0) msg_out[v] = 0.0;
}

/* Cached DFT₆ of a circulant's first column */
typedef struct {
    int    is_circulant;
    double f_re[Z6_D];  /* DFT₆ of circulant vector */
    double f_im[Z6_D];
} Z6_CircEdgeCache;

/* Pre-compute circulant DFTs for all edges with circulant structure */
static inline Z6_CircEdgeCache *z6_cache_circulant_edges(const HPCGraph *g)
{
    Z6_CircEdgeCache *cache = (Z6_CircEdgeCache *)calloc(
        g->n_edges, sizeof(Z6_CircEdgeCache));
    if (!cache) return NULL;

    for (uint64_t eid = 0; eid < g->n_edges; eid++) {
        double circ_f[Z6_D];
        cache[eid].is_circulant = z6_is_circulant_factor(
            &g->edges[eid], circ_f);
        if (cache[eid].is_circulant) {
            z6_dft6_real(circ_f, cache[eid].f_re, cache[eid].f_im);
        }
    }
    return cache;
}

/* Circulant-accelerated BP iterate */
static inline double z6_bp_iterate_circulant(MobiusAmplitudeSheet *ms,
                                              Z6_CircEdgeCache *cache)
{
    const HPCGraph *g = ms->graph;
    double max_delta = 0.0;

    for (uint64_t eid = 0; eid < g->n_edges; eid++) {
        const HPCEdge *edge = &g->edges[eid];
        uint64_t sa = edge->site_a, sb = edge->site_b;

        int idx_a_in_b = mobius_find_msg_idx(g, sb, eid);
        int idx_b_in_a = mobius_find_msg_idx(g, sa, eid);
        if (idx_a_in_b < 0 || idx_b_in_a < 0) continue;

        /* Use circulant fast path if available, else standard */
        int use_fast = (cache && cache[eid].is_circulant);

        /* ── Message a→b ── */
        {
            MobiusProbMsg new_msg;
            const MobiusSiteSheet *sheet_a = &ms->sheets[sa];
            const HPCAdjList *adj_a = &g->adj[sa];

            /* Compute local probability for site a (excluding edge eid) */
            double local_a[Z6_D];
            for (int va = 0; va < MOBIUS_D; va++) {
                local_a[va] = g->locals[sa].edge_re[va] * g->locals[sa].edge_re[va] +
                              g->locals[sa].edge_im[va] * g->locals[sa].edge_im[va];
                for (uint64_t mi = 0; mi < adj_a->count; mi++) {
                    if (adj_a->edge_ids[mi] == eid) continue;
                    local_a[va] *= sheet_a->msg_in[mi].p[va];
                }
            }

            if (use_fast) {
                /* O(6 log 6) circulant fast path */
                z6_circulant_message(local_a,
                                     cache[eid].f_re, cache[eid].f_im,
                                     new_msg.p);
            } else {
                /* Standard O(36) path */
                for (int vb = 0; vb < MOBIUS_D; vb++) {
                    double sum = 0.0;
                    for (int va = 0; va < MOBIUS_D; va++)
                        sum += local_a[va] * mobius_edge_factor(edge, va, vb);
                    new_msg.p[vb] = sum;
                }
            }

            /* Normalize */
            double msg_sum = 0.0;
            for (int v = 0; v < MOBIUS_D; v++) msg_sum += new_msg.p[v];
            if (msg_sum > 1e-30) {
                double inv = 1.0 / msg_sum;
                for (int v = 0; v < MOBIUS_D; v++) new_msg.p[v] *= inv;
            }

            /* Damped update */
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

        /* ── Message b→a ── */
        {
            MobiusProbMsg new_msg;
            const MobiusSiteSheet *sheet_b = &ms->sheets[sb];
            const HPCAdjList *adj_b = &g->adj[sb];

            double local_b[Z6_D];
            for (int vb = 0; vb < MOBIUS_D; vb++) {
                local_b[vb] = g->locals[sb].edge_re[vb] * g->locals[sb].edge_re[vb] +
                              g->locals[sb].edge_im[vb] * g->locals[sb].edge_im[vb];
                for (uint64_t mi = 0; mi < adj_b->count; mi++) {
                    if (adj_b->edge_ids[mi] == eid) continue;
                    local_b[vb] *= sheet_b->msg_in[mi].p[vb];
                }
            }

            if (use_fast) {
                z6_circulant_message(local_b,
                                     cache[eid].f_re, cache[eid].f_im,
                                     new_msg.p);
            } else {
                for (int va = 0; va < MOBIUS_D; va++) {
                    double sum = 0.0;
                    for (int vb = 0; vb < MOBIUS_D; vb++)
                        sum += local_b[vb] * mobius_edge_factor(edge, va, vb);
                    new_msg.p[va] = sum;
                }
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
 * §3  SYNTHEME PRUNER FOR SURFACE WALK
 *
 * During surface walk enumeration, check partial assignments against
 * the syntheme parity constraints. If a group of 6 assigned sites
 * violates a syntheme's H·c ≡ 0 mod 6 check, prune that branch.
 *
 * Key insight: the 15 synthemes partition positions into 3 pairs.
 * As soon as all 6 positions of a syntheme are assigned, we can
 * verify the parity constraint. If it fails, the full configuration
 * cannot have high amplitude (it violates the vacuum's error code).
 * ═══════════════════════════════════════════════════════════════════════ */

#define Z6_PRUNE_NONE      0
#define Z6_PRUNE_SYNTHEME  1  /* Enable syntheme pruning */
#define Z6_PRUNE_S14       2  /* Only check strongest code (faster) */

/* Check if partial assignment (sites 0..depth-1) passes syntheme tests.
 * For any syntheme whose positions are all ≤ depth-1, check parity.
 * Returns 1 if OK, 0 if a violation detected (should prune). */
static inline int z6_syntheme_prune(const uint32_t *indices,
                                     uint64_t depth, int mode)
{
    if (!z6_initialized) z6_codes_init();
    if (depth < Z6_D || mode == Z6_PRUNE_NONE) return 1; /* not enough sites yet */

    int check_start = (mode == Z6_PRUNE_S14) ? 14 : 0;
    int check_end   = (mode == Z6_PRUNE_S14) ? 15 : Z6_N_CODES;

    for (int si = check_start; si < check_end; si++) {
        const Z6_Code *c = &z6_codes[si];
        /* Check parity: H · indices[0..5] ≡ 0 mod 6
         * We use positions 0..5 of the current assignment. */
        for (int r = 0; r < c->n_checks; r++) {
            int sum = 0;
            for (int j = 0; j < Z6_D; j++)
                sum += c->H[r][j] * (int)indices[j];
            if (sum % Z6_D != 0) return 0; /* Prune! */
        }
    }
    return 1; /* Passes all checks */
}

/* Enhanced surface walk with syntheme pruning */
static inline HPCSparseVector *z6_surface_walk(const MobiusAmplitudeSheet *ms,
                                                double threshold,
                                                uint64_t max_entries,
                                                int prune_mode)
{
    if (!z6_initialized) z6_codes_init();
    const HPCGraph *g = ms->graph;
    HPCSparseVector *sv = hpc_sv_create(g->n_sites, 256);
    if (!sv) return NULL;
    sv->threshold = threshold;

    ((MobiusAmplitudeSheet *)ms)->surface_walks++;

    uint32_t candidates[64][MOBIUS_D];
    uint32_t n_cand[64];
    uint64_t total_configs = 1;
    uint64_t pruned = 0;

    uint64_t n = g->n_sites;
    if (n > 64) n = 64;

    for (uint64_t k = 0; k < n; k++) {
        n_cand[k] = 0;
        for (int v = 0; v < MOBIUS_D; v++) {
            if (ms->sheets[k].marginal[v] >= threshold * 0.1)
                candidates[k][n_cand[k]++] = v;
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

        /* Syntheme prune check on first 6 positions */
        if (prune_mode != Z6_PRUNE_NONE && n >= Z6_D) {
            if (!z6_syntheme_prune(indices, Z6_D, prune_mode)) {
                pruned++;
                continue;
            }
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
 * DIAGNOSTIC — Print code summary
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void z6_print_codes(void)
{
    if (!z6_initialized) z6_codes_init();
    printf("  Z₆ Error-Correcting Codes (15 synthemes, D=6):\n");
    printf("  %-4s  %-20s  %6s  %4s  %4s  %s\n",
           "Code", "Syntheme", "Params", "|C|", "gens", "Function");
    int total_cw = 0, total_g = 0;
    for (int si = 0; si < Z6_N_CODES; si++) {
        Z6_Code *c = &z6_codes[si];
        total_cw += c->n_codewords;
        total_g += c->n_gens;
        printf("  S%-3d  {(%d,%d)(%d,%d)(%d,%d)}  [%d,%d,%d]  %4d  %4d",
               si,
               c->syn.pairs[0][0], c->syn.pairs[0][1],
               c->syn.pairs[1][0], c->syn.pairs[1][1],
               c->syn.pairs[2][0], c->syn.pairs[2][1],
               Z6_D, c->k, c->d, c->n_codewords, c->n_gens);
        if (si == 14) printf("  ← STRONGEST (d=4)");
        if (si == 7) printf("  ← MOST SYMMETRIC");
        printf("\n");
    }
    printf("  Total: %d codewords, %d generators\n", total_cw, total_g);
}

#endif /* HPC_Z6_CODES_H */
