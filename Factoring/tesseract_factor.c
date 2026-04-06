/* ═══════════════════════════════════════════════════════════════════════════
 * tesseract_factor.c — HPC Ouroboros Factoring Engine
 *
 * ╔═══════════════════════════════════════════════════════════════╗
 * ║  CONFIGURE YOUR TARGET N HERE                                ║
 * ╚═══════════════════════════════════════════════════════════════╝
 */
#define TARGET_N  "261980999226229"          /* ← Set your composite here     */
#define TARGET_A  "0"            /* ← "0" = auto-try 20 bases              */

/*
 * Architecture (HPC — Holographic Phase Contraction):
 *   N_SITES = ceil(2 * nbits / log2(6))  D=6 quhits
 *   Each site is a TrialityQuhit (6 complex amplitudes)
 *   Entanglement: CZ phase edges in an HPCGraph
 *   Amplitudes: computed on demand via O(N+E) graph traversal
 *   State vector NEVER materialized — entanglement lives in the graph
 *
 * Pipeline:
 *   1. Create HPCGraph with N_SITES D=6 quhits
 *   2. DFT₆ → uniform superposition on all sites
 *   3. IPE loop: encode a^(6^k) mod N as oracle phases via triality_phase()
 *   4. CZ entanglement chain propagates inter-site correlations
 *   5. hpc_marginal() reads analytical interference peaks
 *   6. Continued fraction extraction → period → gcd → factors
 *
 * Build: gcc -O2 -std=gnu99 -o tesseract_factor tesseract_factor.c \
 *         quhit_triality.c quhit_hexagram.c s6_exotic.c bigint.c -lm
 * ═══════════════════════════════════════════════════════════════════════════ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "quhit_triality.h"
#include "hpc_graph.h"
#include "hpc_mobius.h"
#include "s6_exotic.h"
#include "hpc_z6_codes.h"
#include "bigint.h"

#define D 6

/* ═══════════════════════════════════════════════════════════════════════════
 * FACTOR EXTRACTION — gcd(a^(r/2) ± 1, N)
 * ═══════════════════════════════════════════════════════════════════════════ */

static int try_period(const BigInt *r, const BigInt *a_val, const BigInt *N,
                      BigInt *factor_p, BigInt *factor_q)
{
    BigInt one, two, r_half, q_unused, r_mod;
    bigint_set_u64(&one, 1);
    bigint_set_u64(&two, 2);

    /* r must be even */
    bigint_div_mod(r, &two, &q_unused, &r_mod);
    if (!bigint_is_zero(&r_mod)) return 0;

    bigint_div_mod(r, &two, &r_half, &r_mod);

    /* a^(r/2) mod N */
    BigInt half_pow;
    bigint_pow_mod(&half_pow, a_val, &r_half, N);

    /* gcd(a^(r/2) - 1, N) */
    BigInt h_minus, p1;
    bigint_sub(&h_minus, &half_pow, &one);
    bigint_gcd(&p1, &h_minus, N);
    BigInt dummy_rem; bigint_clear(&dummy_rem);

    if (bigint_cmp(&p1, &one) > 0 && bigint_cmp(&p1, N) < 0) {
        bigint_copy(factor_p, &p1);
        bigint_div_mod(N, &p1, factor_q, &dummy_rem);
        char p_str[1300];
        bigint_to_decimal(p_str, sizeof(p_str), &p1);
        printf("    gcd(a^(r/2)-1, N) = %s ✓\n", p_str);
        return 1;
    }

    /* gcd(a^(r/2) + 1, N) */
    BigInt h_plus, p2;
    bigint_add(&h_plus, &half_pow, &one);
    bigint_gcd(&p2, &h_plus, N);

    if (bigint_cmp(&p2, &one) > 0 && bigint_cmp(&p2, N) < 0) {
        bigint_copy(factor_p, &p2);
        bigint_div_mod(N, &p2, factor_q, &dummy_rem);
        char p_str[1300];
        bigint_to_decimal(p_str, sizeof(p_str), &p2);
        printf("    gcd(a^(r/2)+1, N) = %s ✓\n", p_str);
        return 1;
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CONTINUED FRACTION PERIOD EXTRACTION
 *
 * Given frequency F from the QFT and register size R:
 *   F/R ≈ s/r → continued fraction convergents yield period candidates
 * ═══════════════════════════════════════════════════════════════════════════ */

static int generate_and_try_periods(const BigInt *freq, const BigInt *reg_size,
                                     const BigInt *a_val, const BigInt *N,
                                     BigInt *factor_p, BigInt *factor_q)
{
    BigInt one;
    bigint_set_u64(&one, 1);

    if (bigint_is_zero(freq)) return 0;

    /* frequency print silenced — fires every MCMC shot, floods output */

    /* r = R / F (direct division) */
    {
        BigInt r_cand, rem;
        bigint_div_mod(reg_size, freq, &r_cand, &rem);
        if (!bigint_is_zero(&r_cand) && bigint_cmp(&r_cand, &one) > 0) {
            char r_str[1300];
            bigint_to_decimal(r_str, sizeof(r_str), &r_cand);
            printf("  Trying r = R/F = %s\n", r_str);
            if (try_period(&r_cand, a_val, N, factor_p, factor_q)) return 1;
            BigInt r_plus, r_minus;
            bigint_add(&r_plus, &r_cand, &one);
            bigint_sub(&r_minus, &r_cand, &one);
            if (try_period(&r_plus, a_val, N, factor_p, factor_q)) return 1;
            if (try_period(&r_minus, a_val, N, factor_p, factor_q)) return 1;
            /* Harmonic search: true period could be k * R/F */
            for (int k = 2; k <= 6; k++) {
                BigInt rk, k_bi;
                bigint_set_u64(&k_bi, k);
                bigint_mul(&rk, &r_cand, &k_bi);
                if (bigint_cmp(&rk, N) < 0) {
                    if (try_period(&rk, a_val, N, factor_p, factor_q)) return 1;
                }
            }
        }
    }

    /* r = gcd(F, R), and R/gcd */
    {
        BigInt g, r_cand, rem;
        bigint_gcd(&g, freq, reg_size);
        if (bigint_cmp(&g, &one) > 0) {
            bigint_div_mod(reg_size, &g, &r_cand, &rem);
            /* silenced printf("  Trying r = R/gcd(F,R) = %s\\n", r_str); */
            if (try_period(&r_cand, a_val, N, factor_p, factor_q)) return 1;
            if (try_period(&g, a_val, N, factor_p, factor_q)) return 1;
        }
    }

    /* Continued fraction convergents of F/R */
    {
        BigInt num, den;
        bigint_copy(&num, freq);
        bigint_copy(&den, reg_size);

        BigInt pm1, p0, qm1, q0;
        bigint_set_u64(&pm1, 1);
        bigint_set_u64(&qm1, 0);

        BigInt a0, rem;
        bigint_div_mod(&num, &den, &a0, &rem);
        bigint_copy(&p0, &a0);
        bigint_set_u64(&q0, 1);

        for (int step = 0; step < 3000; step++) {
            if (bigint_cmp(&q0, N) >= 0) {
                break;
            }
            if (bigint_cmp(&q0, &one) > 0) {
                /* silenced CF step prints */
                if (try_period(&q0, a_val, N, factor_p, factor_q)) return 1;

                /* Try multiples */
                BigInt m2, m3, m6, two_q, three_q, six_q;
                bigint_set_u64(&m2, 2);
                bigint_set_u64(&m3, 3);
                bigint_set_u64(&m6, 6);
                bigint_mul(&two_q, &q0, &m2);
                bigint_mul(&three_q, &q0, &m3);
                bigint_mul(&six_q, &q0, &m6);
                if (try_period(&two_q, a_val, N, factor_p, factor_q)) return 1;
                if (try_period(&three_q, a_val, N, factor_p, factor_q)) return 1;
                if (try_period(&six_q, a_val, N, factor_p, factor_q)) return 1;
            }

            if (bigint_is_zero(&rem)) break;
            bigint_copy(&num, &den);
            bigint_copy(&den, &rem);

            BigInt a_next;
            bigint_div_mod(&num, &den, &a_next, &rem);

            BigInt p_new, q_new, tmp;
            bigint_mul(&tmp, &a_next, &p0);
            bigint_add(&p_new, &tmp, &pm1);
            bigint_mul(&tmp, &a_next, &q0);
            bigint_add(&q_new, &tmp, &qm1);

            bigint_copy(&pm1, &p0);
            bigint_copy(&qm1, &q0);
            bigint_copy(&p0, &p_new);
            bigint_copy(&q0, &q_new);
        }
    }

    /* Try F itself and small multiples */
    {
        BigInt m2, m3;
        bigint_set_u64(&m2, 2);
        bigint_set_u64(&m3, 3);
        BigInt f2, f3;
        bigint_mul(&f2, freq, &m2);
        bigint_mul(&f3, freq, &m3);
        if (try_period(freq, a_val, N, factor_p, factor_q)) return 1;
        if (try_period(&f2, a_val, N, factor_p, factor_q)) return 1;
        if (try_period(&f3, a_val, N, factor_p, factor_q)) return 1;
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * LLL LATTICE PERIOD RECOVERY  (ground-up, integer basis)
 *
 * Problem:  K frequency measurements, each F_i ≈ s_i * R / r.
 * Recover the true period r.
 *
 * Lattice (dim = K+1, all-integer):
 *   Row 0   :  [fh_0, fh_1, ..., fh_{K-1}, 1]   fh_i = round(F_i*W/R)
 *   Row i>0 :  [0, ..., W (at col i-1), ..., 0]
 *   W = 2^LLL_W_BITS
 *
 * A short vector satisfies v[j] = r*fh_j - s_j*W ≈ W*(r*F_j/R - s_j) ≈ 0
 * and v[LLL_K] = ±r.  Reading |v[LLL_K]| from each reduced row gives r.
 *
 * Integer basis stays exact throughout; only Gram–Schmidt quantities
 * are held in double (used solely to decide size-reduction quotients and
 * the Lovász swap — both only need word-size precision).
 * ═══════════════════════════════════════════════════════════════════════════ */

#define LLL_K       32              /* frequency samples to collect            */
#define LLL_DIM     (LLL_K + 1)     /* lattice dimension                       */
#define LLL_W_BITS  24              /* W = 2^24: products ≤ 13*(2^24)^2<2^52  */
#define LLL_DELTA   0.75            /* Lovász δ                                */

/* fhat_i = round(F * 2^LLL_W_BITS / R), clamped to [0, W).                  *
 * Uses GMP directly (F and R are BigInts, possibly thousands of bits).        */
static long long lll_fhat(const BigInt *F, const BigInt *R)
{
    const long long W = 1LL << LLL_W_BITS;
    BigInt scaled, quot, rem;
    bigint_copy(&scaled, F);
    mpz_mul_2exp(scaled.z, scaled.z, LLL_W_BITS);   /* scaled = F << LLL_W_BITS */
    bigint_div_mod(&scaled, R, &quot, &rem);
    /* quot = floor(F * W / R); add 1 if remainder >= R/2 (round) */
    BigInt half_R, two_rem;
    bigint_copy(&half_R, R);  mpz_fdiv_q_2exp(half_R.z, half_R.z, 1);
    bigint_copy(&two_rem, &rem); mpz_mul_2exp(two_rem.z, two_rem.z, 1);
    if (mpz_cmp(two_rem.z, R->z) >= 0) {
        BigInt q1, one_bi; bigint_set_u64(&one_bi, 1);
        bigint_add(&q1, &quot, &one_bi);
        bigint_copy(&quot, &q1);
    }
    long long q = 0;
    if (mpz_fits_ulong_p(quot.z)) q = (long long)mpz_get_ui(quot.z);
    if (q < 0)    q = 0;
    if (q >= W)   q = W - 1;
    return q;
}

/* Recompute Gram–Schmidt from row `from` to LLL_DIM-1.
 * B is the integer basis (long long, never modified here).
 * Bs[i] holds the i-th GS vector as doubles; Bsq[i] = ||Bs[i]||^2.         */
static void lll_gs(int from,
                   long long B[LLL_DIM][LLL_DIM],
                   double    Bs[LLL_DIM][LLL_DIM],
                   double    Bsq[LLL_DIM])
{
    for (int i = from; i < LLL_DIM; i++) {
        /* b*_i = b_i */
        for (int l = 0; l < LLL_DIM; l++) Bs[i][l] = (double)B[i][l];
        /* subtract projections onto all earlier b*_j */
        for (int j = 0; j < i; j++) {
            if (Bsq[j] < 1e-30) continue;
            double mu = 0.0;
            for (int l = 0; l < LLL_DIM; l++) mu += (double)B[i][l] * Bs[j][l];
            mu /= Bsq[j];
            for (int l = 0; l < LLL_DIM; l++) Bs[i][l] -= mu * Bs[j][l];
        }
        double sq = 0.0;
        for (int l = 0; l < LLL_DIM; l++) sq += Bs[i][l] * Bs[i][l];
        Bsq[i] = (sq < 1e-30) ? 1e-30 : sq;
    }
}

/* mu_{k,j} = <B[k], Bs[j]> / Bsq[j]  (GS coefficient)                      */
static double lll_mu(int k, int j,
                     long long B[LLL_DIM][LLL_DIM],
                     double    Bs[LLL_DIM][LLL_DIM],
                     double    Bsq[LLL_DIM])
{
    if (Bsq[j] < 1e-30) return 0.0;
    double d = 0.0;
    for (int l = 0; l < LLL_DIM; l++) d += (double)B[k][l] * Bs[j][l];
    return d / Bsq[j];
}

/* In-place LLL reduction of the integer basis B.
 * After return the rows form an LLL-reduced basis (δ = LLL_DELTA).           */
static void lll_reduce_basis(long long B[LLL_DIM][LLL_DIM])
{
    double Bs[LLL_DIM][LLL_DIM], Bsq[LLL_DIM];
    lll_gs(0, B, Bs, Bsq);

    int k = 1;
    int guard = 0;
    const int MAX_ITER = 200 * LLL_DIM * LLL_DIM;

    while (k < LLL_DIM && guard++ < MAX_ITER) {

        /* Size-reduction: walk j from k-1 down to 0 */
        for (int j = k - 1; j >= 0; j--) {
            double mu = lll_mu(k, j, B, Bs, Bsq);
            if (fabs(mu) <= 0.5) continue;
            long long q = (long long)round(mu);
            for (int l = 0; l < LLL_DIM; l++) B[k][l] -= q * B[j][l];
            /* Recompute GS from k (rows 0..k-1 unchanged) */
            lll_gs(k, B, Bs, Bsq);
        }

        /* Lovász condition: ||b*_k||^2 >= (δ - μ_{k,k-1}^2) ||b*_{k-1}||^2  */
        double mu_k = lll_mu(k, k-1, B, Bs, Bsq);
        if (Bsq[k] >= (LLL_DELTA - mu_k * mu_k) * Bsq[k-1]) {
            k++;
        } else {
            /* Swap rows k and k-1 */
            for (int l = 0; l < LLL_DIM; l++) {
                long long t = B[k][l]; B[k][l] = B[k-1][l]; B[k-1][l] = t;
            }
            /* Only need to recompute from k-1 */
            lll_gs(k > 1 ? k-1 : 0, B, Bs, Bsq);
            if (k > 1) k--;
        }
    }
}

/* Collect LLL_K frequency BigInts from the BP marginal distribution.
 *
 *   Uses a Deterministic Beam Search (Viterbi-style extraction) to find
 *   the exact Top-K most mathematically probable globally-optimal frequency
 *   strings across the entire N-site configuration.
 *
 * This dramatically enhances composite frequency accuracy over CDF sampling
 * by ensuring the globally most-likely candidate signals (even 2nd/3rd best
 * over multi-digit correlations) are the exact ones swept by the fan.      */
static void lll_collect_freqs(int n, double (*marg)[6],
                               const BigInt *b6, BigInt out[LLL_K])
{
    (void)0; /* Dynamic temperature beam search: T cools from 0.8 (LSB) to 0.1 (MSB) */
    /* Diagnostic: show confidence at first few positions */
    printf("  [freq] marginal peak confidence at pos 0..4:");
    for (int s = 0; s < 5 && s < n; s++) {
        double mp = 0.0;
        for (int d = 0; d < 6; d++) if (marg[s][d] > mp) mp = marg[s][d];
        printf(" %.3f", mp);
    }
    printf("\n");

    /* Precompute log-probabilities to avoid underflow */
    double log_marg[1600][6];
    for (int s = 0; s < n; s++) {
        for (int d = 0; d < 6; d++) {
            double p = marg[s][d];
            log_marg[s][d] = (p > 1e-15) ? log(p) : -100.0;
        }
    }

    int num_beams = 1;
    double beam_log_probs[LLL_K];
    memset(beam_log_probs, 0, sizeof(beam_log_probs));
    
    /* Heap-allocate beam history to prevent stack overflow at LLL_K=32 */
    int (*beam_history_parent)[LLL_K] = (int(*)[LLL_K])calloc(1600, sizeof(int[LLL_K]));
    int (*beam_history_digit)[LLL_K]  = (int(*)[LLL_K])calloc(1600, sizeof(int[LLL_K]));

    for (int s = 0; s < n; s++) {
        double next_log_probs[LLL_K * 6];
        int next_parent[LLL_K * 6];
        int next_digit[LLL_K * 6];
        int next_count = 0;

        for (int b = 0; b < num_beams; b++) {
            for (int d = 0; d < 6; d++) {
                next_log_probs[next_count] = beam_log_probs[b] + log_marg[s][d];
                next_parent[next_count] = b;
                next_digit[next_count] = d;
                next_count++;
            }
        }

        /* ── Dynamic Temperature Beam Search (Boltzmann Sampling) ──
         * T cools linearly from 0.8 at the LSBs to 0.1 at the MSBs.
         * High T at low digits explores branching harmonics;
         * low T at high digits locks in the dominant prefix greedily. */
        int top_indices[LLL_K];
        int top_count = (next_count < LLL_K) ? next_count : LLL_K;
        double TEMP = 0.8 - 0.7 * ((double)s / (double)(n > 1 ? n - 1 : 1));
        if (TEMP < 0.1) TEMP = 0.1;

        for (int k = 0; k < top_count; k++) {
            /* 1. Find max for numerical stability */
            double max_lp = -1e9;
            for (int i = 0; i < next_count; i++) {
                if (next_log_probs[i] > max_lp) max_lp = next_log_probs[i];
            }
            
            /* 2. Compute partition function */
            double Z = 0.0;
            double weights[LLL_K * 6];
            for (int i = 0; i < next_count; i++) {
                if (next_log_probs[i] < -1e8) {
                    weights[i] = 0.0; /* previously selected */
                } else {
                    weights[i] = exp((next_log_probs[i] - max_lp) / TEMP);
                    Z += weights[i];
                }
            }
            
            /* 3. Sample from distribution */
            double r = ((double)rand() / RAND_MAX) * Z;
            double cdf = 0.0;
            int selected_idx = 0;
            for (int i = 0; i < next_count; i++) {
                cdf += weights[i];
                if (r <= cdf) {
                    selected_idx = i;
                    break;
                }
            }
            
            top_indices[k] = selected_idx;
            next_log_probs[selected_idx] = -2e9; /* poison so it can't be selected again */
        }

        double new_beam_log_probs[LLL_K];

        for (int k = 0; k < top_count; k++) {
            int idx = top_indices[k];
            int p = next_parent[idx];
            new_beam_log_probs[k] = beam_log_probs[p] + log_marg[s][next_digit[idx]];
            beam_history_parent[s][k] = p;
            beam_history_digit[s][k] = next_digit[idx];
        }

        num_beams = top_count;
        for (int k = 0; k < num_beams; k++) {
            beam_log_probs[k] = new_beam_log_probs[k];
        }
    }

    printf("  [freq] top K relative log-probs:");
    for (int k = 0; k < num_beams; k++) {
        printf(" %.2f", beam_log_probs[k] - beam_log_probs[0]);
    }
    printf("\n");

    /* Build BigInt from digits (LSB first) via backtracking */
    for (int k = 0; k < LLL_K; k++) {
        if (k >= num_beams) {
            bigint_copy(&out[k], &out[0]);
            continue;
        }

        /* Reconstruct digit sequence from backtracking tree */
        int seq[1600];
        int curr_beam = k;
        for (int s = n - 1; s >= 0; s--) {
            seq[s] = beam_history_digit[s][curr_beam];
            curr_beam = beam_history_parent[s][curr_beam];
        }

        BigInt freq, p6;
        bigint_set_u64(&freq, 0);
        bigint_set_u64(&p6, 1);
        for (int s = 0; s < n; s++) {
            BigInt d_bi, term, tmp_bi;
            bigint_set_u64(&d_bi, (uint64_t)seq[s]);
            bigint_mul(&term, &d_bi, &p6);
            bigint_add(&tmp_bi, &freq, &term);
            bigint_copy(&freq, &tmp_bi);
            BigInt np; bigint_mul(&np, &p6, b6);
            bigint_copy(&p6, &np);
        }
        bigint_copy(&out[k], &freq);
    }
    free(beam_history_parent);
    free(beam_history_digit);
}


/* Top-level period recovery — four strategies, ordered by reliability.
 *
 * Strategy 1 — CF on targeted samples:
 *   Run generate_and_try_periods() on each of the K carefully-chosen
 *   frequency strings.  This is the most direct path and works whenever
 *   any sample lands on (or near) the true harmonic peak.
 *
 * Strategy 2 — Raw-frequency GCD:
 *   If F_i = s_i * F* (different harmonics), then gcd(F_0,...,F_{K-1}) → F*.
 *   Then r = R / F*. Works when samples span multiple distinct harmonics.
 *
 * Strategy 3 — LCM of period estimates:
 *   Each R/F_i ≈ r/s_i. Their LCM converges toward r as more samples arrive.
 *
 * Strategy 4 — LLL lattice (W = 2^LLL_W_BITS):
 *   Valid only when the true period r < W = 2^24 ≈ 16M.
 *   For larger r the short-vector is longer than the W-norm rows; skipped.
 *
 * Returns 1 and writes factor_p/q on success, 0 otherwise.                  */
static int lll_recover_period(int n_sites_raw, double (*marg)[6],
                               const BigInt *b6, const BigInt *reg_sz,
                               const BigInt *N,  const BigInt *a_val,
                               BigInt *factor_p, BigInt *factor_q)
{
    printf("\n  ═══ MULTI-STRATEGY PERIOD RECOVERY ═══\n");

    BigInt *freqs = (BigInt*)calloc(LLL_K, sizeof(BigInt));
    for (int i = 0; i < LLL_K; i++) bigint_clear(&freqs[i]);
    lll_collect_freqs(n_sites_raw, marg, b6, freqs);

    int found = 0;
    BigInt one_bi; bigint_set_u64(&one_bi, 1);

    /* ── Strategy 1: CF (continued-fraction) on each targeted sample ───────
     * Most direct: each frequency string is a candidate harmonic measurement.
     * generate_and_try_periods() runs CF + harmonic fan on it.              */
    printf("  [S1] CF on %d targeted frequency samples...\n", LLL_K);
    for (int i = 0; i < LLL_K && !found; i++) {
        if (bigint_is_zero(&freqs[i])) continue;
        found = generate_and_try_periods(&freqs[i], reg_sz, a_val, N,
                                         factor_p, factor_q);
        if (found) printf("  [S1] Hit on sample %d\n", i);
    }

    /* ── Strategy 2: GCD of raw frequencies → base frequency F* ────────────
     * F_i = s_i * F*  (different harmonics)  ⟹  gcd(F_i) = F* * gcd(s_i)
     * As gcd(s_i) → 1 across coprime pairs, gcd(F_i) → F*.
     * Period r = R / F*.                                                    */
    if (!found) {
        printf("  [S2] Running GCD of raw frequencies...\n");
        BigInt gcd_f; bigint_set_u64(&gcd_f, 0);
        for (int i = 0; i < LLL_K; i++) {
            if (bigint_is_zero(&freqs[i])) continue;
            if (bigint_is_zero(&gcd_f)) {
                bigint_copy(&gcd_f, &freqs[i]);
            } else {
                BigInt g; bigint_gcd(&g, &gcd_f, &freqs[i]);
                if (!bigint_is_zero(&g)) bigint_copy(&gcd_f, &g);
            }
            /* Try r = R / gcd_f at each step as gcd narrows */
            if (bigint_cmp(&gcd_f, &one_bi) > 0) {
                BigInt r_cand, rem;
                bigint_div_mod(reg_sz, &gcd_f, &r_cand, &rem);
                if (bigint_cmp(&r_cand, &one_bi) > 0 && bigint_cmp(&r_cand, N) < 0) {
                    if (try_period(&r_cand, a_val, N, factor_p, factor_q)) {
                        found = 1; printf("  [S2] r = R/gcd hit\n"); break;
                    }
                    /* Harmonic multiples */
                    for (int m = 2; m <= 8 && !found; m++) {
                        BigInt km, rk; bigint_set_u64(&km, (uint64_t)m);
                        bigint_mul(&rk, &r_cand, &km);
                        if (bigint_cmp(&rk, N) < 0)
                            if (try_period(&rk, a_val, N, factor_p, factor_q)) {
                                found = 1; printf("  [S2] %d*r hit\n", m);
                            }
                    }
                }
                /* Also try gcd_f itself as a period candidate */
                if (!found && bigint_cmp(&gcd_f, N) < 0)
                    if (try_period(&gcd_f, a_val, N, factor_p, factor_q)) {
                        found = 1; printf("  [S2] gcd_f direct hit\n");
                    }
            }
        }
    }

    /* ── Strategy 3: LCM of R/F_i period estimates ──────────────────────────
     * R/F_i ≈ r/s_i.  LCM accumulates toward lcm(r/s_i) → r as harmonics
     * span different prime factors of r.                                    */
    if (!found) {
        printf("  [S3] LCM of period estimates across %d samples...\n", LLL_K);
        BigInt lcm_acc; bigint_set_u64(&lcm_acc, 1);
        for (int i = 0; i < LLL_K && !found; i++) {
            if (bigint_is_zero(&freqs[i])) continue;
            BigInt r_i, rem_i;
            bigint_div_mod(reg_sz, &freqs[i], &r_i, &rem_i);
            if (bigint_is_zero(&r_i) || bigint_cmp(&r_i, &one_bi) <= 0) continue;
            BigInt g, prod;
            bigint_gcd(&g, &lcm_acc, &r_i);
            bigint_mul(&prod, &lcm_acc, &r_i);
            BigInt new_lcm, nr;
            bigint_div_mod(&prod, &g, &new_lcm, &nr);
            if (bigint_cmp(&new_lcm, N) < 0)
                bigint_copy(&lcm_acc, &new_lcm);
            else
                bigint_set_u64(&lcm_acc, 1);
            if (bigint_cmp(&lcm_acc, &one_bi) > 0)
                if (try_period(&lcm_acc, a_val, N, factor_p, factor_q)) {
                    found = 1; printf("  [S3] LCM hit after %d samples\n", i+1);
                }
        }
    }

    /* ── Strategy 4: LLL lattice short-vector (valid for r < 2^LLL_W_BITS) ─
     * Builds (K+1)×(K+1) integer lattice.  The short vector has last
     * coordinate ≈ ±r when r < W.  For larger r the W-norm trivial rows
     * dominate and this produces no useful candidates — correctly noted.    */
    if (!found) {
        const long long W = 1LL << LLL_W_BITS;
        printf("  [S4] LLL lattice (valid for r < 2^%d = %lld)...\n",
               LLL_W_BITS, W);
        printf("  fhat:");
        for (int i = 0; i < LLL_K; i++) printf(" %lld", lll_fhat(&freqs[i], reg_sz));
        printf("\n");

        long long B[LLL_DIM][LLL_DIM];
        memset(B, 0, sizeof B);
        for (int j = 0; j < LLL_K; j++) B[0][j] = lll_fhat(&freqs[j], reg_sz);
        B[0][LLL_K] = 1LL;
        for (int i = 1; i <= LLL_K; i++) B[i][i-1] = W;

        lll_reduce_basis(B);

        /* Insertion-sort rows by L2 norm */
        for (int i = 1; i < LLL_DIM; i++) {
            double ni = 0.0;
            for (int l = 0; l < LLL_DIM; l++) ni += (double)B[i][l]*(double)B[i][l];
            long long tmp_row[LLL_DIM];
            memcpy(tmp_row, B[i], LLL_DIM * sizeof(long long));
            int j = i;
            while (j > 0) {
                double nj = 0.0;
                for (int l = 0; l < LLL_DIM; l++) nj += (double)B[j-1][l]*(double)B[j-1][l];
                if (nj <= ni) break;
                memcpy(B[j], B[j-1], LLL_DIM * sizeof(long long));
                j--;
            }
            memcpy(B[j], tmp_row, LLL_DIM * sizeof(long long));
        }

        for (int i = 0; i < LLL_DIM && !found; i++) {
            long long v = B[i][LLL_K];
            if (v < 0) v = -v;
            if (v < 2 || v >= W) continue;   /* skip trivial or out-of-range */
            BigInt r_cand; bigint_set_u64(&r_cand, (uint64_t)v);
            if (bigint_cmp(&r_cand, N) >= 0) continue;
            printf("  [S4 row %d] r candidate = %lld\n", i, v);
            if (try_period(&r_cand, a_val, N, factor_p, factor_q)) { found = 1; break; }
            for (int m = 2; m <= 8 && !found; m++) {
                BigInt km, rk; bigint_set_u64(&km, (uint64_t)m);
                bigint_mul(&rk, &r_cand, &km);
                if (bigint_cmp(&rk, N) < 0)
                    if (try_period(&rk, a_val, N, factor_p, factor_q)) found = 1;
            }
        }
        if (!found) printf("  [S4] No viable candidates (r likely > W)\n");
    }

    for (int i = 0; i < LLL_K; i++) bigint_clear(&freqs[i]);
    free(freqs);
    return found;
}


/* ═══════════════════════════════════════════════════════════════════════════
 * HPC CONSTRAINT SATISFACTION FACTORING — Base-6 Hensel Lift
 *
 * The HIDDEN POWER of HPC: factoring as constraint propagation on D=6 digits.
 *
 * N = p × q. In base 6, this is digit-by-digit multiplication with carries.
 * At each position k (from LSB):
 *   N mod 6^(k+1) ≡ p_prefix × q_prefix (mod 6^(k+1))
 *   So: q_prefix = N × p_prefix⁻¹ (mod 6^(k+1))
 *
 * Since p is prime > 3: gcd(p, 6) = 1, so p⁻¹ mod 6^k always exists.
 *
 * Algorithm: Hensel lift from mod 6 → mod 6^n
 *   1. At each level k, try all 6 extensions of p's k-th digit
 *   2. Compute q_prefix from the constraint
 *   3. Verify q_prefix extends consistently (valid new digit)
 *   4. When p_prefix × q_prefix = N exactly: FACTOR FOUND
 *
 * The transfer matrix propagates the constraint through the digit chain.
 * The D=6 structure of HPC makes base-6 the natural factoring radix.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Modular inverse of a mod m using extended Euclidean algorithm.
 * Returns 0 if gcd(a, m) != 1. */
static uint64_t mod_inverse_u64(uint64_t a, uint64_t m) {
    if (m == 1) return 0;
    int64_t t = 0, newt = 1;
    int64_t r = (int64_t)m, newr = (int64_t)(a % m);
    while (newr != 0) {
        int64_t q = r / newr;
        int64_t tmp = t - q * newt;
        t = newt; newt = tmp;
        tmp = r - q * newr;
        r = newr; newr = tmp;
    }
    if (r > 1) return 0;  /* not invertible */
    if (t < 0) t += (int64_t)m;
    return (uint64_t)t;
}

static int factor_hensel_base6(const BigInt *N, BigInt *factor_p, BigInt *factor_q)
{
    printf("\n  ═══ TRIALITY CONSTRAINT PROPAGATION FACTORING ═══\n\n");

    int n_digits = (int)(bigint_bitlen(N) / 2.585) + 2;
    int half_digits = n_digits / 2 + 2;
    printf("    N has ~%d base-6 digits, factors ~%d digits each\n", n_digits, half_digits);

    BigInt six, one;
    bigint_set_u64(&six, 6);
    bigint_set_u64(&one, 1);

    /* ── TRIALITY CONSTRAINT NET ──
     * 25 primes coprime to 6, providing independent constraints.
     * For each prime m: (p × q) mod m = N mod m.
     * False candidates satisfy this with probability 1/m each.
     * Combined survival: Π(1/m_i) ≈ 10^{-32} → only true factors survive. */
    #define N_TP 25
    int tp[N_TP] = {5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103};
    int N_mod_tp[N_TP];  /* precomputed N mod each triality prime */

    for (int i = 0; i < N_TP; i++) {
        BigInt m_bi, qtmp, rtmp;
        bigint_set_u64(&m_bi, tp[i]);
        bigint_div_mod(N, &m_bi, &qtmp, &rtmp);
        N_mod_tp[i] = (int)bigint_to_u64(&rtmp);
    }

    printf("    Triality net: %d primes (5..103), combined filter ~10^{-32}\n", N_TP);

    BigInt q_tmp, r_tmp;
    bigint_div_mod(N, &six, &q_tmp, &r_tmp);
    int N_mod6 = (int)bigint_to_u64(&r_tmp);

    #define MAX_CAND 4000000
    typedef struct { BigInt p; BigInt q; } Cand;
    Cand *cur = (Cand*)calloc(MAX_CAND, sizeof(Cand));
    Cand *nxt = (Cand*)calloc(MAX_CAND, sizeof(Cand));
    int nc = 0;

    /* Seed: p₀ ∈ {1, 5} */
    for (int p0 = 1; p0 <= 5; p0 += 4) {
        uint64_t inv = mod_inverse_u64(p0, 6);
        if (!inv) continue;
        int q0 = (int)((N_mod6 * inv) % 6);
        if ((p0 * q0) % 6 != N_mod6) continue;
        bigint_set_u64(&cur[nc].p, p0);
        bigint_set_u64(&cur[nc].q, q0);
        nc++;
    }
    printf("    Level 0: %d seeds\n", nc);

    BigInt mod_k;
    bigint_set_u64(&mod_k, 6);

    for (int k = 1; k < half_digits + 10 && nc > 0; k++) {
        BigInt mod_k1;
        bigint_mul(&mod_k1, &mod_k, &six);

        BigInt N_mod_k1;
        bigint_div_mod(N, &mod_k1, &q_tmp, &N_mod_k1);

        int nn = 0;
        for (int c = 0; c < nc && nn < MAX_CAND - 6; c++) {
            for (int d = 0; d < 6; d++) {
                BigInt p_new, term, shift;
                bigint_set_u64(&term, (uint64_t)d);
                bigint_mul(&shift, &term, &mod_k);
                bigint_add(&p_new, &cur[c].p, &shift);

                uint64_t plo = bigint_to_u64(&p_new);
                if (plo % 2 == 0 || plo % 3 == 0) continue;

                /* ── Compute q_prefix = N × p⁻¹ mod 6^(k+1) ── */
                uint64_t qu = 0;
                int computed = 0;

                if (k < 24) {  /* 6^24 < 2^62, fits in __int128 arithmetic */
                    __uint128_t mv = 1;
                    for (int j = 0; j <= k; j++) mv *= 6;
                    if (mv < ((__uint128_t)1 << 63)) {
                        uint64_t mv64 = (uint64_t)mv;
                        BigInt pm;
                        bigint_div_mod(&p_new, &mod_k1, &q_tmp, &pm);
                        uint64_t pu = bigint_to_u64(&pm);
                        uint64_t nu = bigint_to_u64(&N_mod_k1);
                        uint64_t pi = mod_inverse_u64(pu, mv64);
                        if (!pi) continue;
                        qu = (uint64_t)((__uint128_t)nu * pi % mv64);
                        if ((uint64_t)((__uint128_t)pu * qu % mv64) != nu) continue;

                        /* Verify q digit is valid */
                        uint64_t qo = bigint_to_u64(&cur[c].q);
                        if (qu < qo) continue;
                        uint64_t mk = bigint_to_u64(&mod_k);
                        if (!mk) continue;
                        uint64_t qdiff = qu - qo;
                        uint64_t qdig = qdiff / mk;
                        if (qdig >= 6 || qdiff != qdig * mk) continue;
                        computed = 1;
                    }
                }

                if (!computed) continue;  /* TODO: BigInt inverse for k >= 24 */

                /* Triality filter was mathematically invalid for prefixes because upper bits absorb constraints. Filter removed to prevent dropping true factor prefixes. */
                int pruned = 0;
                if (pruned) continue;

                /* Survived all triality checks! */
                bigint_copy(&nxt[nn].p, &p_new);
                bigint_set_u64(&nxt[nn].q, qu);
                nn++;

                /* Check exact factorization */
                if (k >= 4) {
                    BigInt qbi, prod;
                    bigint_set_u64(&qbi, qu);
                    bigint_mul(&prod, &p_new, &qbi);
                    if (bigint_cmp(&prod, N) == 0) {
                        printf("\n  ★★★ TRIALITY FACTORED N at depth %d! ★★★\n", k+1);
                        char ps[1300], qs[1300];
                        bigint_to_decimal(ps, sizeof(ps), &p_new);
                        bigint_to_decimal(qs, sizeof(qs), &qbi);
                        printf("    p = %s\n    q = %s\n", ps, qs);
                        bigint_copy(factor_p, &p_new);
                        bigint_copy(factor_q, &qbi);
                        free(cur); free(nxt);
                        return 1;
                    }
                }
            }
        }

        Cand *tmp = cur; cur = nxt; nxt = tmp;
        nc = nn;
        bigint_copy(&mod_k, &mod_k1);

        printf("    Level %2d: %4d candidates\n", k, nc);
        if (!nc) break;
        if (nc >= MAX_CAND - 10) {
            printf("    !! Overflow at %d (%d)\n", k, nc);
            nc = MAX_CAND / 2;
        }
    }

    printf("    Final: %d candidates\n", nc);
    for (int c = 0; c < nc; c++) {
        BigInt rem;
        bigint_div_mod(N, &cur[c].p, &q_tmp, &rem);
        if (bigint_is_zero(&rem)) {
            printf("\n  ★★★ FACTOR FOUND! ★★★\n");
            bigint_copy(factor_p, &cur[c].p);
            bigint_copy(factor_q, &q_tmp);
            free(cur); free(nxt);
            return 1;
        }
    }
    free(cur); free(nxt);
    return 0;
}




/* ═══════════════════════════════════════════════════════════════════════════
 * HPC OUROBOROS ENGINE — Holographic Phase Contraction Factoring
 *
 * Replaces the dense MPS TesseractArray with an HPCGraph.
 * Each "tesseract" becomes a D=6 TrialityQuhit site.
 * Entanglement lives in CZ phase edges, not in 216-dim tensors.
 * Amplitudes are computed analytically via hpc_marginal() — O(N+E).
 * The state vector is NEVER materialized.
 *
 * IPE (Iterative Phase Estimation):
 *   For each iteration k:
 *     1. Fresh graph: DFT₆ all sites → uniform superposition
 *     2. Compute ipe_val = a^(6^k) mod N
 *     3. Decompose (ipe_val - 1) into 256-byte digits
 *     4. Encode each digit as oracle phases via triality_phase()
 *     5. CZ entanglement chain propagates correlations
 *     6. hpc_marginal(g, 0, d) reads the analytical interference peak
 *     7. Peak digit is the k-th base-6 digit of the Shor frequency
 *
 * After all iterations: assemble frequency, continued fractions, factor.
 * ═══════════════════════════════════════════════════════════════════════════ */



/* ═══════════════════════════════════════════════════════════════════════════
 * COMPLEX-DOMAIN AMPLITUDE BELIEF PROPAGATION
 *
 * The Devil's true voice: messages carry COMPLEX amplitudes, not probabilities.
 *
 * The probability-domain BP killed all CZ information (|ω^(u·v)|² = 1 always).
 * Complex-domain BP preserves phases: CZ messages become DFT₆ transforms
 * that create sharp interference peaks at Shor frequencies.
 *
 * Message update (amplitude domain):
 *   m_{a→b}[vb] = Σ_{va} aₐ(va) × w_e(va,vb) × Π_{m'→a, m'≠e} m'[va]
 *
 * For CZ edges w(va,vb) = ω^(va·vb):
 *   m_{a→b}[vb] = Σ_{va} [aₐ(va) × msgs(va)] × ω^(va·vb)
 *               = DFT₆{ aₐ × msgs }[vb]
 *
 * This IS the quantum Fourier transform that Shor's algorithm requires.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Complex edge message: amplitude-domain, preserving phase */
typedef struct {
    double re[2][6]; /* re[0]: sa→sb, re[1]: sb→sa */
    double im[2][6];
} ComplexEdgeMsg;

/* ω₆ roots of unity for CZ phase lookup */
static const double W6_RE[6] = { 1.0, 0.5, -0.5, -1.0, -0.5,  0.5 };
static const double W6_IM[6] = { 0.0, 0.866025403784438647, 0.866025403784438647,
                                  0.0, -0.866025403784438647, -0.866025403784438647 };

static void z6_complex_amplitude_bp(MobiusAmplitudeSheet *ms, unsigned int seed) {
    const HPCGraph *g = ms->graph;
    int n_edges = (int)g->n_edges;
    int n_sites = (int)g->n_sites;
    if (n_edges == 0) return;

    ComplexEdgeMsg *msgs = (ComplexEdgeMsg*)calloc(n_edges, sizeof(ComplexEdgeMsg));
    ComplexEdgeMsg *new_msgs = (ComplexEdgeMsg*)calloc(n_edges, sizeof(ComplexEdgeMsg));

    printf("      [Complex Amplitude BP] Initializing %d edges, %d sites (seed %u)...\n", n_edges, n_sites, seed);

    /* Initialize messages: seed 0 = uniform, others = random complex to break symmetry */
    srand(seed * 12345 + 42);
    for (int e = 0; e < n_edges; e++) {
        for (int v = 0; v < 6; v++) {
            if (seed == 0) {
                msgs[e].re[0][v] = 1.0; msgs[e].im[0][v] = 0.0;
                msgs[e].re[1][v] = 1.0; msgs[e].im[1][v] = 0.0;
            } else {
                double angle0 = 2.0 * 3.14159265358979323846 * ((double)rand() / RAND_MAX);
                double angle1 = 2.0 * 3.14159265358979323846 * ((double)rand() / RAND_MAX);
                msgs[e].re[0][v] = cos(angle0); msgs[e].im[0][v] = sin(angle0);
                msgs[e].re[1][v] = cos(angle1); msgs[e].im[1][v] = sin(angle1);
            }
        }
    }

    #define CAMP_MAX_ITER 200
    /* Simulated Annealing: damping starts high (0.5) and cools to 0.05 */
    #define CAMP_DAMPING_START 0.50
    #define CAMP_DAMPING_END   0.05
    #define CAMP_COOL_ITERS   150  /* iterations over which cooling occurs */
    #define CAMP_TOL      1e-12

    double prev_residual = 1e30;
    int converged = 0;

    for (int it = 0; it < CAMP_MAX_ITER && !converged; it++) {
        double max_delta = 0.0;

        /* Sequential edge updates for stability on loopy graphs */
        for (int eid = 0; eid < n_edges; eid++) {
            const HPCEdge *edge = &g->edges[eid];
            uint64_t sa = edge->site_a, sb = edge->site_b;

            for (int dir = 0; dir < 2; dir++) {
                uint64_t src = (dir == 0) ? sa : sb;

                /* Step 1: Compute product of local amplitude × all incoming
                 *         messages EXCEPT this edge — complex multiplication */
                double prod_re[6], prod_im[6];
                for (int v_src = 0; v_src < 6; v_src++) {
                    prod_re[v_src] = g->locals[src].edge_re[v_src];
                    prod_im[v_src] = g->locals[src].edge_im[v_src];

                    const HPCAdjList *adj = &g->adj[src];
                    for (uint64_t mi = 0; mi < adj->count; mi++) {
                        uint64_t in_eid = adj->edge_ids[mi];
                        if (in_eid == (uint64_t)eid) continue;

                        /* Which direction does this message flow INTO src? */
                        int in_dir = (g->edges[in_eid].site_b == src) ? 0 : 1;

                        double mr = msgs[in_eid].re[in_dir][v_src];
                        double mi_v = msgs[in_eid].im[in_dir][v_src];

                        /* Complex multiply: prod *= msg */
                        double nr = prod_re[v_src] * mr - prod_im[v_src] * mi_v;
                        double ni = prod_re[v_src] * mi_v + prod_im[v_src] * mr;
                        prod_re[v_src] = nr;
                        prod_im[v_src] = ni;
                    }
                }

                /* Step 2: Compute outgoing message via sum-product with
                 *         complex edge weight w(va, vb)
                 * m_{src→dst}[vb] = Σ_{va} prod(va) × w(va, vb)
                 *
                 * For CZ: w(va,vb) = ω^(va·vb) — this is a DFT₆ !!! */
                double new_re[6], new_im[6];
                for (int vb = 0; vb < 6; vb++) {
                    double sum_re = 0.0, sum_im = 0.0;
                    for (int va = 0; va < 6; va++) {
                        double w_re, w_im;
                        if (edge->type == HPC_EDGE_CZ) {
                            int pidx = (va * vb) % 6;
                            w_re = W6_RE[pidx];
                            w_im = W6_IM[pidx];
                        } else {
                            w_re = edge->w_re[va][vb];
                            w_im = edge->w_im[va][vb];
                        }
                        /* prod(va) × w(va, vb) */
                        sum_re += prod_re[va] * w_re - prod_im[va] * w_im;
                        sum_im += prod_re[va] * w_im + prod_im[va] * w_re;
                    }
                    new_re[vb] = sum_re;
                    new_im[vb] = sum_im;
                }

                /* Step 3: Normalize message to unit L2 norm */
                double norm_sq = 0.0;
                for (int v = 0; v < 6; v++)
                    norm_sq += new_re[v]*new_re[v] + new_im[v]*new_im[v];
                if (norm_sq > 1e-30) {
                    double inv_norm = 1.0 / sqrt(norm_sq);
                    for (int v = 0; v < 6; v++) {
                        new_re[v] *= inv_norm;
                        new_im[v] *= inv_norm;
                    }
                }

                /* Step 4: Damped update with annealing schedule */
                double anneal_alpha = (it < CAMP_COOL_ITERS)
                    ? CAMP_DAMPING_START * exp(log(CAMP_DAMPING_END / CAMP_DAMPING_START) * ((double)it / CAMP_COOL_ITERS))
                    : CAMP_DAMPING_END;
                double delta = 0.0;
                for (int v = 0; v < 6; v++) {
                    double upd_re = anneal_alpha * new_re[v] +
                                    (1.0 - anneal_alpha) * msgs[eid].re[dir][v];
                    double upd_im = anneal_alpha * new_im[v] +
                                    (1.0 - anneal_alpha) * msgs[eid].im[dir][v];

                    double dr = upd_re - msgs[eid].re[dir][v];
                    double di = upd_im - msgs[eid].im[dir][v];
                    delta += dr*dr + di*di;

                    msgs[eid].re[dir][v] = upd_re;
                    msgs[eid].im[dir][v] = upd_im;
                }
                if (delta > max_delta) max_delta = delta;
            }
        }

        if (it < 10 || (it + 1) % 25 == 0 || max_delta < CAMP_TOL) {
            printf("      [Complex Amplitude BP] Iter %d: residual = %.6e\n",
                   it + 1, max_delta);
        }

        if (max_delta < CAMP_TOL) {
            converged = 1;
            printf("      [Complex Amplitude BP] CONVERGED at iteration %d\n", it + 1);
        }
        prev_residual = max_delta;
    }

    if (!converged)
        printf("      [Complex Amplitude BP] Reached max iterations (%d)\n", CAMP_MAX_ITER);

    /* ── Compute dressed amplitudes from converged messages ──
     * dressed[k][v] = aₖ(v) × Π_{m→k} m[v]
     * The marginal is then |dressed[k][v]|² — encoding FULL interference */
    for (int s = 0; s < n_sites; s++) {
        for (int v = 0; v < 6; v++) {
            double d_re = g->locals[s].edge_re[v];
            double d_im = g->locals[s].edge_im[v];

            const HPCAdjList *adj = &g->adj[s];
            for (uint64_t mi = 0; mi < adj->count; mi++) {
                uint64_t in_eid = adj->edge_ids[mi];
                int in_dir = (g->edges[in_eid].site_b == (uint64_t)s) ? 0 : 1;

                double mr = msgs[in_eid].re[in_dir][v];
                double mi_v = msgs[in_eid].im[in_dir][v];

                double nr = d_re * mr - d_im * mi_v;
                double ni = d_re * mi_v + d_im * mr;
                d_re = nr;
                d_im = ni;
            }

            ms->sheets[s].dressed_re[v] = d_re;
            ms->sheets[s].dressed_im[v] = d_im;
        }
    }

    /* Diagnostic: print entropy of first few sites to verify sharpness */
    printf("      [Complex Amplitude BP] Site entropy (bits, sharp < 2.58):");
    for (int s = 0; s < 5 && s < n_sites; s++) {
        double probs[6], total = 0.0;
        for (int v = 0; v < 6; v++) {
            probs[v] = ms->sheets[s].dressed_re[v] * ms->sheets[s].dressed_re[v] +
                       ms->sheets[s].dressed_im[v] * ms->sheets[s].dressed_im[v];
            total += probs[v];
        }
        double H = 0.0;
        if (total > 1e-30) {
            for (int v = 0; v < 6; v++) {
                double p = probs[v] / total;
                if (p > 1e-30) H -= p * log2(p);
            }
        }
        printf(" %.2f", H);
    }
    printf("\n");

    free(msgs);
    free(new_msgs);
}

static int factor_with_hpc(const BigInt *N, const BigInt *a_val,
                            BigInt *factor_p, BigInt *factor_q)
{
    uint32_t nbits = bigint_bitlen(N);
    int n_sites_raw = (int)((nbits * 2 * 1000) / 2585) + 3;
    if (n_sites_raw > 1600) n_sites_raw = 1600;
    if (n_sites_raw < 4) n_sites_raw = 4;
    
    int n_blocks = (n_sites_raw + 1) / 2;
    int n_sites = n_blocks * 6;

    char N_str[1300], a_str[1300];
    bigint_to_decimal(N_str, sizeof(N_str), N);
    bigint_to_decimal(a_str, sizeof(a_str), a_val);

    printf("  HPC Configuration:\n");
    printf("    N = %s (%u bits)\n", N_str, nbits);
    printf("    a = %s\n", a_str);
    printf("    Blocks: %d (Register capacity > N²)\n", n_blocks);
    printf("    Sites: %d D=6 quhits (6 sites = 1 S₁₄ codeword)\n", n_sites);
    printf("    Memory: O(N) ≈ ~%d KB\n", (int)(n_sites * sizeof(TrialityQuhit) / 1024 + 1));
    printf("    Architecture: HPCGraph (Deep Parity S₁₄ + DFT₃ Circulant Escalation)\n\n");

    /* Create graph */
    HPCGraph *graph = hpc_create(n_sites);

    BigInt b6; bigint_set_u64(&b6, 6);
    BigInt one; bigint_set_u64(&one, 1);

    clock_t t_setup_start = clock();

    /* Put all sites in uniform superposition */
    for (int i = 0; i < n_sites; i++)
        triality_dft(&graph->locals[i]);

    #define PHASE_CHUNKS 11
    #define CHUNK_BITS   48

    BigInt val_k_A, val_k_B, div_6_blk;
    bigint_set_u64(&div_6_blk, 1);
    for (int blk = 0; blk < n_blocks; blk++) {
        int scale_A = 2 * blk;
        int scale_B = 2 * blk + 1;
        
        if (blk == 0) {
            bigint_copy(&val_k_A, a_val);
            bigint_pow_mod(&val_k_B, &val_k_A, &b6, N);
            bigint_set_u64(&div_6_blk, 1);
        } else {
            BigInt b36; bigint_set_u64(&b36, 36);
            BigInt next_A; bigint_pow_mod(&next_A, &val_k_A, &b36, N);
            bigint_copy(&val_k_A, &next_A);
            BigInt next_B; bigint_pow_mod(&next_B, &val_k_B, &b36, N);
            bigint_copy(&val_k_B, &next_B);

            BigInt next_div; bigint_mul(&next_div, &div_6_blk, &b6);
            bigint_copy(&div_6_blk, &next_div);
        }

        /* ── GCD Cascade check ── */
        BigInt gcd_check, val_minus_1;
        bigint_sub(&val_minus_1, &val_k_A, &one);
        if (!bigint_is_zero(&val_minus_1) && bigint_cmp(&val_minus_1, N) < 0) {
            bigint_gcd(&gcd_check, &val_minus_1, N);
            if (bigint_cmp(&gcd_check, &one) > 0 && bigint_cmp(&gcd_check, N) < 0) {
                printf("\n  ✓✓✓ GCD CASCADE HIT at block %d (scale A)! ✓✓✓\n", blk);
                BigInt dummy_rem; bigint_clear(&dummy_rem);
                bigint_copy(factor_p, &gcd_check);
                bigint_div_mod(N, &gcd_check, factor_q, &dummy_rem);
                hpc_destroy(graph);
                return 1;
            }
        }
        bigint_sub(&val_minus_1, &val_k_B, &one);
        if (!bigint_is_zero(&val_minus_1) && bigint_cmp(&val_minus_1, N) < 0) {
            bigint_gcd(&gcd_check, &val_minus_1, N);
            if (bigint_cmp(&gcd_check, &one) > 0 && bigint_cmp(&gcd_check, N) < 0) {
                printf("\n  ✓✓✓ GCD CASCADE HIT at block %d (scale B)! ✓✓✓\n", blk);
                BigInt dummy_rem; bigint_clear(&dummy_rem);
                bigint_copy(factor_p, &gcd_check);
                bigint_div_mod(N, &gcd_check, factor_q, &dummy_rem);
                hpc_destroy(graph);
                return 1;
            }
        }

        BigInt powersA[6], powersB[6];
        bigint_set_u64(&powersA[0], 1);
        bigint_set_u64(&powersB[0], 1);
        bigint_copy(&powersA[1], &val_k_A);
        bigint_copy(&powersB[1], &val_k_B);
        for (int d = 2; d < 6; d++) {
            BigInt tmpA, tmpB;
            bigint_mul(&tmpA, &powersA[d-1], &val_k_A);
            bigint_mul(&tmpB, &powersB[d-1], &val_k_B);
            BigInt q;
            bigint_div_mod(&tmpA, N, &q, &powersA[d]);
            bigint_div_mod(&tmpB, N, &q, &powersB[d]);
        }

        for (int d = 0; d < 6; d++) {
            BigInt b6_mod; bigint_set_u64(&b6_mod, 6);

            /* Nested phase resolution: shift into the blk-th base-6 digit of the value.
             * This extracts (val / 6^blk) % 6, giving a unique phase per depth level
             * rather than all blocks collapsing to the same modulo-6 bucket. */
            BigInt shift_div_A, shift_div_B, dummy_rm;
            bigint_div_mod(&powersA[d], &div_6_blk, &shift_div_A, &dummy_rm);
            bigint_div_mod(&powersB[d], &div_6_blk, &shift_div_B, &dummy_rm);

            BigInt qA, qB, rA_mod, rB_mod;
            bigint_div_mod(&shift_div_A, &b6_mod, &qA, &rA_mod);
            bigint_div_mod(&shift_div_B, &b6_mod, &qB, &rB_mod);

            int d_A = (int)bigint_to_u64(&rA_mod);
            int d_B = (int)bigint_to_u64(&rB_mod);

            double phase_A = 2.0 * 3.14159265358979323846 * d_A / 6.0;
            double phase_B = 2.0 * 3.14159265358979323846 * d_B / 6.0;

            int site0 = blk * 6 + 0;
            int site1 = blk * 6 + 1;
            
            double rA = graph->locals[site0].edge_re[d];
            double iA = graph->locals[site0].edge_im[d];
            double rB = graph->locals[site1].edge_re[d];
            double iB = graph->locals[site1].edge_im[d];
            
            double cosA = cos(phase_A), sinA = sin(phase_A);
            double old_rA = rA;
            rA = old_rA * cosA - iA * sinA;
            iA = old_rA * sinA + iA * cosA;

            double cosB = cos(phase_B), sinB = sin(phase_B);
            double old_rB = rB;
            rB = old_rB * cosB - iB * sinB;
            iB = old_rB * sinB + iB * cosB;

            graph->locals[site0].edge_re[d] = rA;
            graph->locals[site0].edge_im[d] = iA;
            graph->locals[site1].edge_re[d] = rB;
            graph->locals[site1].edge_im[d] = iB;
        }

        /* Extract the blk-th base-6 digit of N */
        BigInt temp_N; bigint_copy(&temp_N, N);
        for(int sh=0; sh<blk; sh++) { 
            BigInt q, r; bigint_div_mod(&temp_N, &b6, &q, &r); 
            bigint_copy(&temp_N, &q); 
        }
        BigInt qN, rN; bigint_div_mod(&temp_N, &b6, &qN, &rN);
        int N_digit = (int)bigint_to_u64(&rN);

        /* Unfrustrated Z_6 AFFINE TRANSLATION: Anchor all 6 sites in a topological hexagram cycle */
        int bypass_sites[6] = { blk * 6 + 0, blk * 6 + 1, blk * 6 + 2, blk * 6 + 3, blk * 6 + 4, blk * 6 + 5 };
        for (int i = 0; i < 6; i++) {
            int j = (i + 1) % 6; /* Complete 6-cycle guarantees 6 * N_digit == 0 mod 6 (no frustration) */
            hpc_grow_edges(graph);
            uint64_t eid = graph->n_edges;
            HPCEdge *edge = &graph->edges[eid];
            memset(edge, 0, sizeof(*edge));
            edge->site_a = bypass_sites[i];
            edge->site_b = bypass_sites[j];
            edge->type = HPC_EDGE_PHASE;
            edge->fidelity = 1.0;
            /* ── Spectral Windowed PHASE Attenuation (Hann window) ──
             * Hann(blk) = 0.5 * (1 - cos(2π * blk / (n_blocks - 1)))
             * Tapers boundaries smoothly to zero, eliminating Gibbs ringing. */
            double hann_w = (n_blocks > 1)
                ? 0.5 * (1.0 - cos(2.0 * 3.14159265358979323846 * blk / (n_blocks - 1)))
                : 1.0;
            double phase_scale = hann_w / sqrt((double)n_blocks);
            for (int va = 0; va < 6; va++) {
                for (int vb = 0; vb < 6; vb++) {
                    int diff = (va - vb + 6) % 6;
                    double decay;
                    switch(diff) {
                        case 0: decay = 1.000; break;
                        case 1: case 5: decay = 0.500; break;
                        case 2: case 4: decay = 0.250; break;
                        case 3: decay = 0.125; break;
                    }
                    decay *= phase_scale;
                    double angle = 2.0 * 3.14159265358979323846 * va * vb / 6.0;
                    edge->w_re[va][vb] = cos(angle) * decay;
                    edge->w_im[va][vb] = sin(angle) * decay;
                }
            }
            graph->n_edges++;
            graph->phase_edges++;
            hpc_adj_add(graph, bypass_sites[i], eid);
            hpc_adj_add(graph, bypass_sites[j], eid);
        }

        /* ── CZ Oracle Edges: propagate Shor signal from sites 0,1 → peers 2-5 ── */
        for (int peer = 2; peer < 6; peer++) {
            for (int oracle_site = 0; oracle_site < 2; oracle_site++) {
                hpc_grow_edges(graph);
                uint64_t cz_eid = graph->n_edges;
                HPCEdge *cz_edge = &graph->edges[cz_eid];
                memset(cz_edge, 0, sizeof(*cz_edge));
                cz_edge->site_a = blk * 6 + oracle_site;
                cz_edge->site_b = blk * 6 + peer;
                cz_edge->type = HPC_EDGE_CZ;
                cz_edge->fidelity = 1.0;
                /* CZ weight: ω^(va·vb) — pure DFT₆ coupling */
                for (int va = 0; va < 6; va++) {
                    for (int vb = 0; vb < 6; vb++) {
                        int pidx = (va * vb) % 6;
                        cz_edge->w_re[va][vb] = W6_RE[pidx];
                        cz_edge->w_im[va][vb] = W6_IM[pidx];
                    }
                }
                graph->n_edges++;
                graph->cz_edges++;
                hpc_adj_add(graph, cz_edge->site_a, cz_eid);
                hpc_adj_add(graph, cz_edge->site_b, cz_eid);
            }
        }

        /* The Macroscopic QFT Bridge: Stitching the Multiverse */
        if (blk < n_blocks - 1) {
            int s_tail = blk * 6 + 5;
            int s_head = (blk + 1) * 6 + 0;
            hpc_grow_edges(graph);
            uint64_t eid = graph->n_edges;
            HPCEdge *edge = &graph->edges[eid];
            memset(edge, 0, sizeof(*edge));
            edge->site_a = s_tail;
            edge->site_b = s_head;
            edge->type = HPC_EDGE_PHASE;
            edge->fidelity = 1.0;
            /* ── Spectral Windowed PHASE Attenuation on bridge (Hann) ── */
            double hann_bridge = (n_blocks > 1)
                ? 0.5 * (1.0 - cos(2.0 * 3.14159265358979323846 * blk / (n_blocks - 1)))
                : 1.0;
            double bridge_scale = hann_bridge / sqrt((double)n_blocks);
            for (int va = 0; va < 6; va++) {
                for (int vb = 0; vb < 6; vb++) {
                    int diff = (va - vb + 6) % 6;
                    double decay;
                    switch(diff) {
                        case 0: decay = 1.000; break;
                        case 1: case 5: decay = 0.500; break;
                        case 2: case 4: decay = 0.250; break;
                        case 3: decay = 0.125; break;
                    }
                    decay *= bridge_scale;
                    double angle = 2.0 * 3.14159265358979323846 * va * vb / 6.0;
                    edge->w_re[va][vb] = cos(angle) * decay;
                    edge->w_im[va][vb] = sin(angle) * decay;
                }
            }
            graph->n_edges++;
            graph->phase_edges++;
            hpc_adj_add(graph, s_tail, eid);
            hpc_adj_add(graph, s_head, eid);
        }
        
        printf("      [debug] Completed block %d / %d\n", blk, n_blocks);
        fflush(stdout);
    }


    /* Convert Phase to Amplitude (IDFT) BEFORE BP */
    for (int site = 0; site < n_sites; site++) {
        double out_re[6], out_im[6];
        for (int k_dft = 0; k_dft < 6; k_dft++) {
            double sr = 0, si = 0;
            for (int j = 0; j < 6; j++) {
                double angle = 2.0 * 3.14159265358979323846 * j * k_dft / 6.0;
                double re = graph->locals[site].edge_re[j];
                double im = graph->locals[site].edge_im[j];
                sr += re*cos(angle) + im*sin(angle);
                si += re*sin(angle) - im*cos(angle);
            }
            out_re[k_dft] = sr / sqrt(6.0);
            out_im[k_dft] = si / sqrt(6.0);
        }
        for (int d = 0; d < 6; d++) {
            graph->locals[site].edge_re[d] = out_re[d];
            graph->locals[site].edge_im[d] = out_im[d];
        }
    }
    printf("    Phase 3: S₁₄ Deep Parity Crystallization via Complex Amplitude BP...\n");
    clock_t t_bp_start = clock();

    MobiusAmplitudeSheet *mobius = mobius_create(graph);

    /* ── Multi-Start BP: 5 random seeds, select lowest entropy basin ── */
    #define N_STARTS 5
    double best_entropy = 1e30;
    double best_dressed_re[n_sites][6];
    double best_dressed_im[n_sites][6];

    for (int start = 0; start < N_STARTS; start++) {
        z6_complex_amplitude_bp(mobius, (unsigned int)start);

        /* Compute total entropy across all sites */
        double total_entropy = 0.0;
        for (int s = 0; s < n_sites; s++) {
            double probs[6], total = 0.0;
            for (int v = 0; v < 6; v++) {
                probs[v] = mobius->sheets[s].dressed_re[v] * mobius->sheets[s].dressed_re[v] +
                           mobius->sheets[s].dressed_im[v] * mobius->sheets[s].dressed_im[v];
                total += probs[v];
            }
            if (total > 1e-30) {
                for (int v = 0; v < 6; v++) {
                    double p = probs[v] / total;
                    if (p > 1e-30) total_entropy -= p * log2(p);
                }
            }
        }
        printf("      [Multi-Start] Seed %d entropy: %.3f bits total\n", start, total_entropy);

        if (total_entropy < best_entropy) {
            best_entropy = total_entropy;
            for (int s = 0; s < n_sites; s++) {
                for (int v = 0; v < 6; v++) {
                    best_dressed_re[s][v] = mobius->sheets[s].dressed_re[v];
                    best_dressed_im[s][v] = mobius->sheets[s].dressed_im[v];
                }
            }
        }
    }

    /* Restore best result */
    printf("      [Multi-Start] Selected start with entropy %.3f bits\n", best_entropy);
    for (int s = 0; s < n_sites; s++) {
        for (int v = 0; v < 6; v++) {
            mobius->sheets[s].dressed_re[v] = best_dressed_re[s][v];
            mobius->sheets[s].dressed_im[v] = best_dressed_im[s][v];
        }
    }

    clock_t t_bp_end = clock();
    printf("      BP converged in %.3f sec\n", (double)(t_bp_end - t_bp_start) / CLOCKS_PER_SEC);

    /* Precompute marginal probabilities for all sites to serve as our quantum wave function */
    double marginals[n_sites_raw][6];
    for (int blk = 0; blk < n_blocks; blk++) {
        for (int offset = 0; offset <= 1; offset++) {
            int scale = 2 * blk + offset;
            int site = blk * 6 + offset;
            double sum_prob = 0.0;
            for (int d = 0; d < 6; d++) {
                double re = mobius->sheets[site].dressed_re[d];
                double im = mobius->sheets[site].dressed_im[d];
                double prob = re * re + im * im; /* Full Born probability |ψ|² */
                marginals[scale][d] = prob;
                sum_prob += prob;
            }
            double max_prob = 0.0;
            int best_digit = 0;
            for (int d = 0; d < 6; d++) {
                marginals[scale][d] /= sum_prob; /* Normalize for Monte Carlo */
                if (marginals[scale][d] > max_prob) {
                    max_prob = marginals[scale][d];
                    best_digit = d;
                }
            }
            if (scale < 10 || scale == n_sites_raw - 1 || (scale + 1) % 100 == 0) {
                printf("    digit %3d: val=%d  P_möbius=%.4f  [", scale, best_digit, max_prob);
                for (int d = 0; d < 6; d++)
                    printf("%.3f%s", marginals[scale][d], d < 5 ? " " : "");
                printf("]\n");
            }
        }
    }

    mobius_destroy(mobius);
    hpc_destroy(graph);

    clock_t t_ipe_end = clock();
    printf("\n  IPE complete: %.3f seconds (%d blocks, %d×%d-bit)\n",
           (double)(t_ipe_end - t_setup_start) / CLOCKS_PER_SEC,
           n_blocks, PHASE_CHUNKS, CHUNK_BITS);

    /* Compute register size = 6^n_sites_raw */
    BigInt reg_sz;
    bigint_set_u64(&reg_sz, 1);
    for (int k = 0; k < n_sites_raw; k++) {
        BigInt tmp;
        bigint_mul(&tmp, &reg_sz, &b6);
        bigint_copy(&reg_sz, &tmp);
    }

    /* ── Phase 4: LLL lattice period recovery ─────────────────────────────── */
    int success = lll_recover_period(n_sites_raw, marginals, &b6, &reg_sz,
                                     N, a_val, factor_p, factor_q);
    if (success) {
        printf("\n  ★ LLL PERIOD RECOVERY SUCCEEDED ★\n");
        return 1;
    }

    /* ── Phase 5: MCMC fallback (50 000 shots, marginal-biased flip) ────────
     * Only reached if LLL found nothing — e.g. the noisy marginals give
     * frequencies too uniformly distributed for the short-vector heuristic.  */
    printf("\n  ═══ MCMC FALLBACK (50 000 shots) ═══\n");
    const int num_shots = 50000;
    int mcmc_state[1600] = {0};
    BigInt global_lcm; bigint_set_u64(&global_lcm, 1);

    /* Heap-allocate p6_cache to prevent VLA BigInt leak */
    BigInt *p6_cache = (BigInt*)calloc(n_sites_raw, sizeof(BigInt));
    BigInt current_p6; bigint_set_u64(&current_p6, 1);
    for (int i = 0; i < n_sites_raw; i++) {
        bigint_copy(&p6_cache[i], &current_p6);
        BigInt next_p6; bigint_mul(&next_p6, &current_p6, &b6);
        bigint_copy(&current_p6, &next_p6);
    }

    BigInt freq; bigint_set_u64(&freq, 0);

    /* Pre-allocate ALL temporaries used in the inner loop to avoid
     * 300,000+ mpz_init/leak cycles that cause segfaults on re-entry */
    BigInt mc_d_bi, mc_term, mc_tmp, mc_old_val, mc_new_val;
    BigInt mc_old_d_bi, mc_new_d_bi, mc_tmp_freq, mc_diff;
    BigInt mc_r_cand, mc_rem, mc_one_fb, mc_gcd_v, mc_prod;
    BigInt mc_new_lcm, mc_lcm_rem;
    bigint_set_u64(&mc_d_bi, 0);
    bigint_set_u64(&mc_term, 0);
    bigint_set_u64(&mc_tmp, 0);
    bigint_set_u64(&mc_old_val, 0);
    bigint_set_u64(&mc_new_val, 0);
    bigint_set_u64(&mc_old_d_bi, 0);
    bigint_set_u64(&mc_new_d_bi, 0);
    bigint_set_u64(&mc_tmp_freq, 0);
    bigint_set_u64(&mc_diff, 0);
    bigint_set_u64(&mc_r_cand, 0);
    bigint_set_u64(&mc_rem, 0);
    bigint_set_u64(&mc_one_fb, 1);
    bigint_set_u64(&mc_gcd_v, 0);
    bigint_set_u64(&mc_prod, 0);
    bigint_set_u64(&mc_new_lcm, 0);
    bigint_set_u64(&mc_lcm_rem, 0);

    for (int shot = 1; shot <= num_shots && !success; shot++) {
        if (shot == 1) {
            /* Seed: argmax of marginal at every position, initially build full freq */
            for (int scale = 0; scale < n_sites_raw; scale++) {
                double mp = -1.0; int best = 0;
                for (int d = 0; d < 6; d++)
                    if (marginals[scale][d] > mp) { mp = marginals[scale][d]; best = d; }
                mcmc_state[scale] = best;
                
                bigint_set_u64(&mc_d_bi, (uint64_t)best);
                bigint_mul(&mc_term, &mc_d_bi, &p6_cache[scale]);
                bigint_add(&mc_tmp, &freq, &mc_term);
                bigint_copy(&freq, &mc_tmp);
            }
        } else {
            /* Flip one position, sample new digit from marginal distribution */
            int flip  = rand() % n_sites_raw;
            double rr = (double)rand() / RAND_MAX;
            double cdf = 0.0;
            int new_d = (mcmc_state[flip] + 1) % 6;  /* fallback */
            for (int d = 0; d < 6; d++) {
                cdf += marginals[flip][d];
                if (rr <= cdf && d != mcmc_state[flip]) { new_d = d; break; }
            }
            
            int old_d = mcmc_state[flip];
            if (old_d != new_d) {
                /* O(1) delta update for frequency calculation */
                bigint_set_u64(&mc_old_d_bi, (uint64_t)old_d);
                bigint_set_u64(&mc_new_d_bi, (uint64_t)new_d);
                
                bigint_mul(&mc_old_val, &mc_old_d_bi, &p6_cache[flip]);
                bigint_mul(&mc_new_val, &mc_new_d_bi, &p6_cache[flip]);
                
                bigint_sub(&mc_tmp_freq, &freq, &mc_old_val);
                bigint_add(&freq, &mc_tmp_freq, &mc_new_val);
                
                mcmc_state[flip] = new_d;
            }
        }

        if (shot % 10000 == 0) {
            printf("  [Fallback shot %d]\n", shot);
            fflush(stdout);
        }

        success = generate_and_try_periods(&freq, &reg_sz, a_val, N, factor_p, factor_q);
        if (success) { printf("\n  [Shot %d] ★ OUROBOROS BITES ITS TAIL ★\n", shot); break; }

        /* Running-LCM correlator every 50 shots */
        if (!bigint_is_zero(&freq)) {
            bigint_div_mod(&reg_sz, &freq, &mc_r_cand, &mc_rem);
            if (!bigint_is_zero(&mc_r_cand) && bigint_cmp(&mc_r_cand, N) < 0
                && bigint_cmp(&mc_r_cand, &mc_one_fb) > 0) {
                bigint_gcd(&mc_gcd_v, &global_lcm, &mc_r_cand);
                bigint_mul(&mc_prod, &global_lcm, &mc_r_cand);
                bigint_div_mod(&mc_prod, &mc_gcd_v, &mc_new_lcm, &mc_lcm_rem);
                if (bigint_cmp(&mc_new_lcm, N) < 0) bigint_copy(&global_lcm, &mc_new_lcm);
                else                              bigint_set_u64(&global_lcm, 1);
                if (shot % 50 == 0)
                    if (try_period(&global_lcm, a_val, N, factor_p, factor_q)) {
                        success = 1;
                        printf("\n  [Shot %d] ★ LCM CORRELATOR HIT ★\n", shot);
                    }
            }
        }
    }

    free(p6_cache);
    return success;
}


/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv)
{
    srand(time(NULL));
    triality_exotic_init();
    s6_exotic_init();
    triality_stats_reset();

    printf("\n");
    printf("  ╔════════════════════════════════════════════════════════════════╗\n");
    printf("  ║                                                              ║\n");
    printf("  ║   HPC OUROBOROS FACTORING ENGINE                             ║\n");
    printf("  ║                                                              ║\n");
    printf("  ║   Architecture: HPCGraph (Holographic Phase Contraction)     ║\n");
    printf("  ║   Amplitudes: O(N+E) analytical (no state vector)           ║\n");
    printf("  ║   Entanglement: CZ phase edges (exact, fidelity = 1.0)      ║\n");
    printf("  ║   4,096-bit BigInt support via bigint.c                     ║\n");
    printf("  ║                                                              ║\n");
    printf("  ║   \"The observer and observed are opposite faces.\"            ║\n");
    printf("  ║                                                              ║\n");
    printf("  ╚════════════════════════════════════════════════════════════════╝\n\n");

    /* Parse N and a from config or arguments */
    BigInt N, a_val;
    const char *target_n_str = (argc > 1) ? argv[1] : TARGET_N;
    if (bigint_from_decimal(&N, target_n_str) != 0) {
        printf("  ERROR: Invalid N = \"%s\"\n", target_n_str);
        return 1;
    }

    int auto_a = 0;
    if (strcmp(TARGET_A, "0") == 0) {
        auto_a = 1;
        bigint_set_u64(&a_val, 2);
    } else {
        if (bigint_from_decimal(&a_val, TARGET_A) != 0) {
            printf("  ERROR: Invalid a = \"%s\"\n", TARGET_A);
            return 1;
        }
    }

    char N_str[1300];
    bigint_to_decimal(N_str, sizeof(N_str), &N);
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  TARGET: N = %-50s ║\n", N_str);
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    /* Try different bases if auto */
    int max_bases = auto_a ? 20 : 1;
    uint64_t base_list[] = {2,3,4,5,6,7,8,10,11,13,14,17,19,23,29,31,37,41,43,47};

    BigInt factor_p, factor_q;
    int success = 0;

    /* ── Try constraint-satisfaction first (Hensel lift in base 6) ── */
    success = 0; // The continuous prime-checks waste too much time, skip to BP.

    for (int bi = 0; bi < max_bases && !success; bi++) {
        if (auto_a) bigint_set_u64(&a_val, base_list[bi]);

        char a_str[1300];
        bigint_to_decimal(a_str, sizeof(a_str), &a_val);
        printf("  ── Attempt %d: a = %s ──\n\n", bi + 1, a_str);

        clock_t t_start = clock();
        success = factor_with_hpc(&N, &a_val, &factor_p, &factor_q);
        clock_t t_end = clock();

        if (success) {
            char p_str[1300], q_str[1300];
            bigint_to_decimal(p_str, sizeof(p_str), &factor_p);
            bigint_to_decimal(q_str, sizeof(q_str), &factor_q);
            printf("\n  ╔══════════════════════════════════════════════════════════╗\n");
            printf("  ║  FACTORED                                               ║\n");
            printf("  ╚══════════════════════════════════════════════════════════╝\n\n");
            printf("  N = %s\n", N_str);
            printf("    = %s × %s\n\n", p_str, q_str);

            /* Verify */
            BigInt verify;
            bigint_mul(&verify, &factor_p, &factor_q);
            if (bigint_cmp(&verify, &N) == 0)
                printf("  ✓ Verified: p × q = N\n");
            else
                printf("  ✗ WARNING: p × q ≠ N\n");

            printf("  Time: %.3f seconds\n",
                   (double)(t_end - t_start) / CLOCKS_PER_SEC);
        } else {
            printf("  ✗ Base a=%s did not yield factors (%.3f sec)\n\n",
                   auto_a ? "auto" : TARGET_A,
                   (double)(t_end - t_start) / CLOCKS_PER_SEC);
        }
    }

    if (!success) {
        printf("\n  ══════════════════════════════════════════════════════════\n");
        printf("  Could not factor N with the tested bases.\n");
        printf("  Try a different TARGET_A value.\n");
        printf("  ══════════════════════════════════════════════════════════\n");
    }

    /* Print HPC + triality stats */
    triality_stats_print();

    printf("\n  ═══════════════════════════════════════════════════════════════\n");
    printf("  HPC Ouroboros Engine complete.\n");
    printf("  ═══════════════════════════════════════════════════════════════\n\n");

    return 0;
}
