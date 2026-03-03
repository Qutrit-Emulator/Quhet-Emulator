/* s6_exotic.c — S₆ Outer Automorphism Implementation
 *
 * Constructs φ via synthematic totals at initialization.
 * Provides exotic gates, parameterized folds, and dual measurement.
 */

#include <string.h>
#include <stdio.h>
#include <math.h>
#include "s6_exotic.h"

static const double INV_SQRT2 = 0.70710678118654752440;

/* ═══════════════════════════════════════════════════════════════════════════
 * SYNTHEMES — 15 partitions of {0,..,5} into 3 pairs
 *
 * Canonical form: pairs sorted by first element, a < c < e.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* We enumerate all 15 at compile time */
const S6Syntheme s6_synthemes[S6_NUM_SYNTHEMES] = {
    [0]  = {{{0,1},{2,3},{4,5}}},   /* T0 member */
    [1]  = {{{0,1},{2,4},{3,5}}},
    [2]  = {{{0,1},{2,5},{3,4}}},
    [3]  = {{{0,2},{1,3},{4,5}}},
    [4]  = {{{0,2},{1,4},{3,5}}},   /* T0 member */
    [5]  = {{{0,2},{1,5},{3,4}}},
    [6]  = {{{0,3},{1,2},{4,5}}},
    [7]  = {{{0,3},{1,4},{2,5}}},   /* DEFAULT fold — the standard antipodal pairing */
    [8]  = {{{0,3},{1,5},{2,4}}},   /* T0 member */
    [9]  = {{{0,4},{1,2},{3,5}}},
    [10] = {{{0,4},{1,3},{2,5}}},   /* T0 member */
    [11] = {{{0,4},{1,5},{2,3}}},
    [12] = {{{0,5},{1,2},{3,4}}},   /* T0 member */
    [13] = {{{0,5},{1,3},{2,4}}},
    [14] = {{{0,5},{1,4},{2,3}}},
};

/* ═══════════════════════════════════════════════════════════════════════════
 * TOTALS — 6 sets of 5 synthemes covering all 15 pairs
 *
 * Built at init time by brute-force search over C(15,5) = 3003 subsets.
 * ═══════════════════════════════════════════════════════════════════════════ */

int s6_totals[S6_NUM_TOTALS][5];
S6Perm s6_phi[S6_ORDER];
int s6_exotic_ready = 0;

/* Check if 5 syntheme indices form a total (cover all 15 pairs exactly once) */
static int check_total(const int idx[5]) {
    int covered[6][6] = {{0}};
    for (int si = 0; si < 5; si++) {
        const S6Syntheme *s = &s6_synthemes[idx[si]];
        for (int p = 0; p < 3; p++) {
            int a = s->pairs[p][0], b = s->pairs[p][1];
            if (covered[a][b]) return 0;
            covered[a][b] = covered[b][a] = 1;
        }
    }
    for (int a = 0; a < 6; a++)
        for (int b = a+1; b < 6; b++)
            if (!covered[a][b]) return 0;
    return 1;
}

static int find_all_totals(void) {
    int n = 0;
    for (int a = 0; a < 15 && n < 6; a++)
    for (int b = a+1; b < 15 && n < 6; b++)
    for (int c = b+1; c < 15 && n < 6; c++)
    for (int d = c+1; d < 15 && n < 6; d++)
    for (int e = d+1; e < 15 && n < 6; e++) {
        int idx[5] = {a,b,c,d,e};
        if (check_total(idx)) {
            for (int i = 0; i < 5; i++) s6_totals[n][i] = idx[i];
            n++;
        }
    }
    return n;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PERMUTATION PRIMITIVES
 * ═══════════════════════════════════════════════════════════════════════════ */

S6Perm s6_from_int(int n) {
    n = ((n % 720) + 720) % 720;
    int avail[6] = {0,1,2,3,4,5}, fact[6] = {120,24,6,2,1,1};
    S6Perm r;
    for (int i = 0; i < 6; i++) {
        int d = n / fact[i]; n %= fact[i];
        r.p[i] = avail[d];
        for (int j = d; j < 5-i; j++) avail[j] = avail[j+1];
    }
    return r;
}

int s6_to_int_perm(S6Perm a) {
    int used[6]={0}, result=0, fact[6]={120,24,6,2,1,1};
    for (int i = 0; i < 6; i++) {
        int rank = 0;
        for (int j = 0; j < a.p[i]; j++) if (!used[j]) rank++;
        result += rank * fact[i]; used[a.p[i]] = 1;
    }
    return result;
}

S6Perm s6_compose_perm(S6Perm a, S6Perm b) {
    S6Perm r;
    for (int i = 0; i < 6; i++) r.p[i] = b.p[a.p[i]];
    return r;
}

S6Perm s6_inverse(S6Perm a) {
    S6Perm r;
    for (int i = 0; i < 6; i++) r.p[a.p[i]] = i;
    return r;
}

int s6_perm_eq(S6Perm a, S6Perm b) {
    return memcmp(a.p, b.p, sizeof(a.p)) == 0;
}

int s6_fixed_points(S6Perm a) {
    int c = 0;
    for (int i = 0; i < 6; i++) if (a.p[i] == i) c++;
    return c;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * OUTER AUTOMORPHISM CONSTRUCTION
 *
 * For each σ ∈ S₆: apply σ to each total's synthemes, find which
 * target total ALL 5 image synthemes land in → φ(σ).
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Apply σ to a syntheme: permute all elements in all pairs */
static S6Syntheme apply_sigma(S6Perm sigma, const S6Syntheme *s) {
    S6Syntheme r;
    for (int p = 0; p < 3; p++) {
        int a = sigma.p[s->pairs[p][0]];
        int b = sigma.p[s->pairs[p][1]];
        if (a > b) { int t = a; a = b; b = t; }
        r.pairs[p][0] = a; r.pairs[p][1] = b;
    }
    /* Sort pairs by first element */
    for (int i = 0; i < 2; i++)
        for (int j = i+1; j < 3; j++)
            if (r.pairs[j][0] < r.pairs[i][0]) {
                S6Syntheme tmp = r;
                r.pairs[i][0] = tmp.pairs[j][0]; r.pairs[i][1] = tmp.pairs[j][1];
                r.pairs[j][0] = tmp.pairs[i][0]; r.pairs[j][1] = tmp.pairs[i][1];
            }
    return r;
}

/* Find index of a syntheme in the table */
static int find_synth_idx(const S6Syntheme *s) {
    for (int i = 0; i < S6_NUM_SYNTHEMES; i++)
        if (memcmp(&s6_synthemes[i], s, sizeof(S6Syntheme)) == 0) return i;
    return -1;
}

/* Map a total under σ: apply σ to all 5 synthemes, find target total */
static int map_total_under(S6Perm sigma, int total_idx) {
    int img_synth[5];
    for (int j = 0; j < 5; j++) {
        S6Syntheme img = apply_sigma(sigma, &s6_synthemes[s6_totals[total_idx][j]]);
        img_synth[j] = find_synth_idx(&img);
        if (img_synth[j] < 0) return -1;
    }
    for (int t = 0; t < S6_NUM_TOTALS; t++) {
        int all = 1;
        for (int j = 0; j < 5 && all; j++) {
            int found = 0;
            for (int k = 0; k < 5; k++)
                if (s6_totals[t][k] == img_synth[j]) { found = 1; break; }
            if (!found) all = 0;
        }
        if (all) return t;
    }
    return -1;
}

void s6_exotic_init(void) {
    if (s6_exotic_ready) return;

    int n_totals = find_all_totals();
    if (n_totals != 6) {
        fprintf(stderr, "[S6_EXOTIC] FATAL: found %d totals (expected 6)\n", n_totals);
        return;
    }

    /* Build φ for all 720 elements */
    for (int idx = 0; idx < 720; idx++) {
        S6Perm sigma = s6_from_int(idx);
        for (int t = 0; t < 6; t++) {
            int img = map_total_under(sigma, t);
            if (img < 0) {
                s6_phi[idx] = S6_IDENTITY;
                break;
            }
            s6_phi[idx].p[t] = img;
        }
    }

    s6_exotic_ready = 1;
}

S6Perm s6_apply_phi(S6Perm sigma) {
    if (!s6_exotic_ready) s6_exotic_init();
    int idx = s6_to_int_perm(sigma);
    return s6_phi[idx];
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SYNTHEME-PARAMETERIZED FOLD
 *
 * Instead of always pairing (k, k+3), pair according to syntheme s.
 * Output layout: out[0..2] = vesica, out[3..5] = wave.
 * ═══════════════════════════════════════════════════════════════════════════ */

void s6_fold_syntheme(const double *in_re, const double *in_im,
                      double *out_re, double *out_im,
                      int syntheme_idx) {
    if (syntheme_idx < 0 || syntheme_idx >= S6_NUM_SYNTHEMES)
        syntheme_idx = 7; /* fallback to default */

    const S6Syntheme *s = &s6_synthemes[syntheme_idx];
    for (int p = 0; p < 3; p++) {
        int k = s->pairs[p][0], k2 = s->pairs[p][1];
        out_re[p]     = INV_SQRT2 * (in_re[k] + in_re[k2]);
        out_im[p]     = INV_SQRT2 * (in_im[k] + in_im[k2]);
        out_re[p + 3] = INV_SQRT2 * (in_re[k] - in_re[k2]);
        out_im[p + 3] = INV_SQRT2 * (in_im[k] - in_im[k2]);
    }
}

void s6_unfold_syntheme(const double *in_re, const double *in_im,
                        double *out_re, double *out_im,
                        int syntheme_idx) {
    if (syntheme_idx < 0 || syntheme_idx >= S6_NUM_SYNTHEMES)
        syntheme_idx = 7;

    const S6Syntheme *s = &s6_synthemes[syntheme_idx];
    /* Zero output first — different synthemes write to different indices */
    memset(out_re, 0, 6 * sizeof(double));
    memset(out_im, 0, 6 * sizeof(double));

    for (int p = 0; p < 3; p++) {
        int k = s->pairs[p][0], k2 = s->pairs[p][1];
        double v_re = in_re[p],     v_im = in_im[p];
        double w_re = in_re[p + 3], w_im = in_im[p + 3];
        out_re[k]  = INV_SQRT2 * (v_re + w_re);
        out_im[k]  = INV_SQRT2 * (v_im + w_im);
        out_re[k2] = INV_SQRT2 * (v_re - w_re);
        out_im[k2] = INV_SQRT2 * (v_im - w_im);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * OPTIMAL SYNTHEME SELECTION
 *
 * Given an active_mask (6-bit bitmask of nonzero basis states),
 * find the syntheme whose pairing puts the most active states into
 * the SAME pair. This maximizes the efficiency of the fold stage.
 *
 * If both active states are in the same pair, the fold concentrates
 * all amplitude into one slot → O(1) downstream.
 * ═══════════════════════════════════════════════════════════════════════════ */

int s6_optimal_syntheme(uint8_t active_mask) {
    int best_synth = 7; /* default: antipodal */
    int best_score = -1;

    for (int si = 0; si < S6_NUM_SYNTHEMES; si++) {
        const S6Syntheme *s = &s6_synthemes[si];
        int score = 0;
        for (int p = 0; p < 3; p++) {
            int k1 = s->pairs[p][0], k2 = s->pairs[p][1];
            int a1 = (active_mask >> k1) & 1;
            int a2 = (active_mask >> k2) & 1;
            /* Score: count pairs where BOTH are active (good: concentrate)
             * or NEITHER is active (good: skip entire pair) */
            if (a1 && a2) score += 2;  /* both active → concentrated */
            if (!a1 && !a2) score += 1; /* both dead → skippable */
        }
        if (score > best_score) {
            best_score = score;
            best_synth = si;
        }
    }
    return best_synth;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * EXOTIC GATE — Apply φ(σ) instead of σ
 * ═══════════════════════════════════════════════════════════════════════════ */

void s6_apply_exotic_gate(const double *in_re, const double *in_im,
                          double *out_re, double *out_im,
                          S6Perm sigma) {
    if (!s6_exotic_ready) s6_exotic_init();
    S6Perm phi_sigma = s6_apply_phi(sigma);

    double tmp_re[6], tmp_im[6];
    for (int i = 0; i < 6; i++) {
        tmp_re[phi_sigma.p[i]] = in_re[i];
        tmp_im[phi_sigma.p[i]] = in_im[i];
    }
    memcpy(out_re, tmp_re, 6 * sizeof(double));
    memcpy(out_im, tmp_im, 6 * sizeof(double));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * DUAL MEASUREMENT — Standard and exotic probabilities
 *
 * Standard: probs[k] = |ψ[k]|²
 * Exotic: probabilities after applying the "exotic permutation"
 * π_exotic = φ(transposition (01)) = triple transposition (01)(23)(45).
 * This gives probabilities in a basis that the standard basis cannot see.
 * ═══════════════════════════════════════════════════════════════════════════ */

void s6_dual_probabilities(const double *re, const double *im,
                           double *probs_std, double *probs_exo) {
    /* Standard probabilities */
    for (int k = 0; k < 6; k++)
        probs_std[k] = re[k]*re[k] + im[k]*im[k];

    /* Exotic probabilities: apply (01)(23)(45) to indices
     * This is the image of the simplest transposition under φ */
    static const int exotic_perm[6] = {1,0,3,2,5,4};
    for (int k = 0; k < 6; k++) {
        int ek = exotic_perm[k];
        probs_exo[k] = re[ek]*re[ek] + im[ek]*im[ek];
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * EXOTIC INVARIANT Δ
 *
 * Δ(ψ) = Σ_{σ ∈ S₆} |⟨ψ|P_σ|ψ⟩ - ⟨ψ|P_{φ(σ)}|ψ⟩|²
 *
 * For each permutation σ:
 *   ⟨ψ|P_σ|ψ⟩ = Σ_k conj(ψ_k) · ψ_{σ(k)}
 *   ⟨ψ|P_{φ(σ)}|ψ⟩ = Σ_k conj(ψ_k) · ψ_{φ(σ)(k)}
 *
 * The difference measures how much the state distinguishes between
 * the standard and exotic representations. This is a D=6-exclusive
 * quantum number — it cannot exist in any other dimension.
 *
 * Cost: O(720 × 6) ≈ 4320 operations.
 * ═══════════════════════════════════════════════════════════════════════════ */

double s6_exotic_invariant(const double *re, const double *im) {
    if (!s6_exotic_ready) s6_exotic_init();

    double delta = 0;

    for (int idx = 0; idx < 720; idx++) {
        S6Perm sigma = s6_from_int(idx);
        S6Perm phi_sigma = s6_phi[idx];

        /* ⟨ψ|P_σ|ψ⟩ = Σ_k conj(ψ_k) · ψ_{σ(k)} */
        double std_re = 0, std_im = 0;
        double exo_re = 0, exo_im = 0;

        for (int k = 0; k < 6; k++) {
            /* conj(ψ_k) = (re[k], -im[k]) */
            double ck_re = re[k], ck_im = -im[k];

            /* Standard: ψ_{σ(k)} */
            int sk = sigma.p[k];
            std_re += ck_re * re[sk] - ck_im * im[sk];
            std_im += ck_re * im[sk] + ck_im * re[sk];

            /* Exotic: ψ_{φ(σ)(k)} */
            int ek = phi_sigma.p[k];
            exo_re += ck_re * re[ek] - ck_im * im[ek];
            exo_im += ck_re * im[ek] + ck_im * re[ek];
        }

        /* |std - exo|² */
        double diff_re = std_re - exo_re;
        double diff_im = std_im - exo_im;
        delta += diff_re * diff_re + diff_im * diff_im;
    }

    return delta;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * EXOTIC ENTROPY ΔS
 *
 * ΔS = S_std - S_exo
 *
 * S_std = -Σ p_k log(p_k) where p_k = |ψ_k|²
 * S_exo = -Σ q_k log(q_k) where q_k = |fold_k|² (syntheme-parameterized)
 *
 * ΔS > 0: exotic channel is more ordered (lower entropy)
 * ΔS < 0: standard channel is more ordered
 * ΔS = 0: both channels see the same disorder
 * ═══════════════════════════════════════════════════════════════════════════ */

double s6_exotic_entropy(const double *re, const double *im,
                         int syntheme_idx) {
    /* Standard entropy */
    double S_std = 0;
    double total = 0;
    for (int k = 0; k < 6; k++) {
        double p = re[k]*re[k] + im[k]*im[k];
        if (p > 1e-30) S_std -= p * log(p);
        total += p;
    }
    /* Normalize */
    if (total > 1e-30) S_std = S_std / total + log(total);

    /* Exotic entropy: fold by syntheme */
    double fold_re[6], fold_im[6];
    s6_fold_syntheme(re, im, fold_re, fold_im, syntheme_idx);

    double S_exo = 0;
    total = 0;
    for (int k = 0; k < 6; k++) {
        double p = fold_re[k]*fold_re[k] + fold_im[k]*fold_im[k];
        if (p > 1e-30) S_exo -= p * log(p);
        total += p;
    }
    if (total > 1e-30) S_exo = S_exo / total + log(total);

    return S_std - S_exo;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * EXOTIC FINGERPRINT — Per-conjugacy-class breakdown
 *
 * Returns 11 values, one per conjugacy class of S₆.
 * class_deltas[c] = (1/|C_c|) Σ_{σ ∈ C_c} |⟨ψ|P_σ|ψ⟩ - ⟨ψ|P_{φ(σ)}|ψ⟩|²
 *
 * The 11 classes (ordered by partition):
 *   0: 1⁶ (identity)      5: 3·2·1
 *   1: 2·1⁴               6: 4·1²
 *   2: 2²·1²              7: 4·2
 *   3: 2³                  8: 5·1
 *   4: 3·1³               9: 3²
 *  10: 6
 *
 * Classes where φ swaps the cycle type (1↔3, 4↔9, 6↔7) will have
 * the largest deltas. Classes where φ preserves the type (0, 2, 5, 8, 10)
 * may still have nonzero deltas (individual elements are rearranged).
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Cycle type → class index mapping */
static int cycle_type_to_class(S6Perm sigma) {
    int vis[6] = {0}, lens[6], n = 0;
    for (int i = 0; i < 6; i++) {
        if (vis[i]) continue;
        int len = 0, j = i;
        while (!vis[j]) { vis[j] = 1; j = sigma.p[j]; len++; }
        lens[n++] = len;
    }
    /* Sort descending */
    for (int i = 0; i < n-1; i++)
        for (int j = i+1; j < n; j++)
            if (lens[j] > lens[i]) { int t = lens[i]; lens[i] = lens[j]; lens[j] = t; }

    /* Map to class index based on sorted partition */
    if (n == 6) return 0;  /* 1⁶ */
    if (n == 5) return 1;  /* 2·1⁴ */
    if (n == 4 && lens[0] == 2 && lens[1] == 2) return 2;  /* 2²·1² */
    if (n == 4 && lens[0] == 3) return 4;  /* 3·1³ */
    if (n == 3 && lens[0] == 2 && lens[1] == 2 && lens[2] == 2) return 3;  /* 2³ */
    if (n == 3 && lens[0] == 3 && lens[1] == 2) return 5;  /* 3·2·1 */
    if (n == 3 && lens[0] == 4) return 6;  /* 4·1² */
    if (n == 2 && lens[0] == 3 && lens[1] == 3) return 9;  /* 3² */
    if (n == 2 && lens[0] == 4) return 7;  /* 4·2 */
    if (n == 2 && lens[0] == 5) return 8;  /* 5·1 */
    if (n == 1) return 10; /* 6 */
    return 0;
}

void s6_exotic_fingerprint(const double *re, const double *im,
                           double *class_deltas) {
    if (!s6_exotic_ready) s6_exotic_init();

    double class_sums[11] = {0};
    int class_counts[11] = {0};

    for (int idx = 0; idx < 720; idx++) {
        S6Perm sigma = s6_from_int(idx);
        S6Perm phi_sigma = s6_phi[idx];

        double std_re = 0, std_im = 0;
        double exo_re = 0, exo_im = 0;

        for (int k = 0; k < 6; k++) {
            double ck_re = re[k], ck_im = -im[k];
            int sk = sigma.p[k];
            std_re += ck_re * re[sk] - ck_im * im[sk];
            std_im += ck_re * im[sk] + ck_im * re[sk];
            int ek = phi_sigma.p[k];
            exo_re += ck_re * re[ek] - ck_im * im[ek];
            exo_im += ck_re * im[ek] + ck_im * re[ek];
        }

        double diff_re = std_re - exo_re;
        double diff_im = std_im - exo_im;
        double d2 = diff_re * diff_re + diff_im * diff_im;

        int cls = cycle_type_to_class(sigma);
        class_sums[cls] += d2;
        class_counts[cls]++;
    }

    for (int c = 0; c < 11; c++)
        class_deltas[c] = (class_counts[c] > 0) ?
                           class_sums[c] / class_counts[c] : 0;
}
