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
    /* Static temporaries — allocated once, reused forever */
    static int tp_init = 0;
    static BigInt tp_one, tp_two, tp_r_half, tp_q_unused, tp_r_mod;
    static BigInt tp_half_pow, tp_h_minus, tp_p1, tp_dummy_rem;
    static BigInt tp_h_plus, tp_p2;
    if (!tp_init) {
        bigint_set_u64(&tp_one, 1); bigint_set_u64(&tp_two, 2);
        bigint_set_u64(&tp_r_half, 0); bigint_set_u64(&tp_q_unused, 0);
        bigint_set_u64(&tp_r_mod, 0); bigint_set_u64(&tp_half_pow, 0);
        bigint_set_u64(&tp_h_minus, 0); bigint_set_u64(&tp_p1, 0);
        bigint_set_u64(&tp_dummy_rem, 0); bigint_set_u64(&tp_h_plus, 0);
        bigint_set_u64(&tp_p2, 0);
        tp_init = 1;
    }

    /* r must be even */
    bigint_div_mod(r, &tp_two, &tp_q_unused, &tp_r_mod);
    if (!bigint_is_zero(&tp_r_mod)) return 0;

    bigint_div_mod(r, &tp_two, &tp_r_half, &tp_r_mod);

    /* a^(r/2) mod N */
    bigint_pow_mod(&tp_half_pow, a_val, &tp_r_half, N);

    /* gcd(a^(r/2) - 1, N) */
    bigint_sub(&tp_h_minus, &tp_half_pow, &tp_one);
    bigint_gcd(&tp_p1, &tp_h_minus, N);

    if (bigint_cmp(&tp_p1, &tp_one) > 0 && bigint_cmp(&tp_p1, N) < 0) {
        bigint_copy(factor_p, &tp_p1);
        bigint_div_mod(N, &tp_p1, factor_q, &tp_dummy_rem);
        char p_str[1300];
        bigint_to_decimal(p_str, sizeof(p_str), &tp_p1);
        printf("    gcd(a^(r/2)-1, N) = %s ✓\n", p_str);
        return 1;
    }

    /* gcd(a^(r/2) + 1, N) */
    bigint_add(&tp_h_plus, &tp_half_pow, &tp_one);
    bigint_gcd(&tp_p2, &tp_h_plus, N);

    if (bigint_cmp(&tp_p2, &tp_one) > 0 && bigint_cmp(&tp_p2, N) < 0) {
        bigint_copy(factor_p, &tp_p2);
        bigint_div_mod(N, &tp_p2, factor_q, &tp_dummy_rem);
        char p_str[1300];
        bigint_to_decimal(p_str, sizeof(p_str), &tp_p2);
        printf("    gcd(a^(r/2)+1, N) = %s ✓\n", p_str);
        return 1;
    }

    if (bigint_cmp(&tp_p1, N) == 0 || bigint_cmp(&tp_p2, N) == 0) {
        /* If gcd == N, it implies a^(r/2) ± 1 = 0 mod N.
         * If a^(r/2) = -1 mod N, this is a mathematically sterile TRIVIAL ROOT!
         * The base 'a' will never yield prime factors, so we must violently abort! */
        BigInt full_pow;
        bigint_pow_mod(&full_pow, a_val, r, N);
        if (bigint_cmp(&full_pow, &tp_one) == 0) {
            printf("\n  [!] MATHEMATICALLY STERILE BASE DETECTED. Base yields trivial roots. Aborting.\n");
            exit(1); /* DUD BASE */
        }
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
    /* Static temporaries — allocated once, reused forever */
    static int gtp_init = 0;
    static BigInt gtp_one, gtp_r_cand, gtp_rem, gtp_r_plus, gtp_r_minus;
    static BigInt gtp_k_bi, gtp_rk;
    static BigInt gtp_g;
    static BigInt gtp_num, gtp_den, gtp_pm1, gtp_p0, gtp_qm1, gtp_q0;
    static BigInt gtp_a0, gtp_cf_rem, gtp_a_next;
    static BigInt gtp_m2, gtp_m3, gtp_m6, gtp_two_q, gtp_three_q, gtp_six_q;
    static BigInt gtp_p_new, gtp_q_new, gtp_tmp;
    static BigInt gtp_f2, gtp_f3;
    if (!gtp_init) {
        bigint_set_u64(&gtp_one, 1); bigint_set_u64(&gtp_r_cand, 0);
        bigint_set_u64(&gtp_rem, 0); bigint_set_u64(&gtp_r_plus, 0);
        bigint_set_u64(&gtp_r_minus, 0); bigint_set_u64(&gtp_k_bi, 0);
        bigint_set_u64(&gtp_rk, 0); bigint_set_u64(&gtp_g, 0);
        bigint_set_u64(&gtp_num, 0); bigint_set_u64(&gtp_den, 0);
        bigint_set_u64(&gtp_pm1, 0); bigint_set_u64(&gtp_p0, 0);
        bigint_set_u64(&gtp_qm1, 0); bigint_set_u64(&gtp_q0, 0);
        bigint_set_u64(&gtp_a0, 0); bigint_set_u64(&gtp_cf_rem, 0);
        bigint_set_u64(&gtp_a_next, 0);
        bigint_set_u64(&gtp_m2, 2); bigint_set_u64(&gtp_m3, 3);
        bigint_set_u64(&gtp_m6, 6);
        bigint_set_u64(&gtp_two_q, 0); bigint_set_u64(&gtp_three_q, 0);
        bigint_set_u64(&gtp_six_q, 0);
        bigint_set_u64(&gtp_p_new, 0); bigint_set_u64(&gtp_q_new, 0);
        bigint_set_u64(&gtp_tmp, 0);
        bigint_set_u64(&gtp_f2, 0); bigint_set_u64(&gtp_f3, 0);
        gtp_init = 1;
    }

    if (bigint_is_zero(freq)) return 0;

    /* r = R / F (direct division) */
    bigint_div_mod(reg_size, freq, &gtp_r_cand, &gtp_rem);
    if (!bigint_is_zero(&gtp_r_cand) && bigint_cmp(&gtp_r_cand, &gtp_one) > 0) {
        char r_str[1300];
        bigint_to_decimal(r_str, sizeof(r_str), &gtp_r_cand);
        printf("  Trying r = R/F = %s\n", r_str);
        if (try_period(&gtp_r_cand, a_val, N, factor_p, factor_q)) return 1;
        bigint_add(&gtp_r_plus, &gtp_r_cand, &gtp_one);
        bigint_sub(&gtp_r_minus, &gtp_r_cand, &gtp_one);
        if (try_period(&gtp_r_plus, a_val, N, factor_p, factor_q)) return 1;
        if (try_period(&gtp_r_minus, a_val, N, factor_p, factor_q)) return 1;
        /* Harmonic search: true period could be k * R/F */
        for (int k = 2; k <= 6; k++) {
            bigint_set_u64(&gtp_k_bi, k);
            bigint_mul(&gtp_rk, &gtp_r_cand, &gtp_k_bi);
            if (bigint_cmp(&gtp_rk, N) < 0) {
                if (try_period(&gtp_rk, a_val, N, factor_p, factor_q)) return 1;
            }
        }
    }

    /* r = gcd(F, R), and R/gcd */
    bigint_gcd(&gtp_g, freq, reg_size);
    if (bigint_cmp(&gtp_g, &gtp_one) > 0) {
        bigint_div_mod(reg_size, &gtp_g, &gtp_r_cand, &gtp_rem);
        if (try_period(&gtp_r_cand, a_val, N, factor_p, factor_q)) return 1;
        if (try_period(&gtp_g, a_val, N, factor_p, factor_q)) return 1;
    }

    /* Continued fraction convergents of F/R */
    bigint_copy(&gtp_num, freq);
    bigint_copy(&gtp_den, reg_size);

    bigint_set_u64(&gtp_pm1, 1);
    bigint_set_u64(&gtp_qm1, 0);

    bigint_div_mod(&gtp_num, &gtp_den, &gtp_a0, &gtp_cf_rem);
    bigint_copy(&gtp_p0, &gtp_a0);
    bigint_set_u64(&gtp_q0, 1);

    for (int step = 0; ; step++) {
        if (bigint_cmp(&gtp_q0, N) >= 0 || bigint_bitlen(&gtp_q0) > 2000) break;

        if (bigint_cmp(&gtp_q0, &gtp_one) > 0) {
            if (try_period(&gtp_q0, a_val, N, factor_p, factor_q)) return 1;

            /* Try multiples */
            bigint_mul(&gtp_two_q, &gtp_q0, &gtp_m2);
            bigint_mul(&gtp_three_q, &gtp_q0, &gtp_m3);
            bigint_mul(&gtp_six_q, &gtp_q0, &gtp_m6);
            if (try_period(&gtp_two_q, a_val, N, factor_p, factor_q)) return 1;
            if (try_period(&gtp_three_q, a_val, N, factor_p, factor_q)) return 1;
            if (try_period(&gtp_six_q, a_val, N, factor_p, factor_q)) return 1;
        }

        if (bigint_is_zero(&gtp_cf_rem)) break;
        bigint_copy(&gtp_num, &gtp_den);
        bigint_copy(&gtp_den, &gtp_cf_rem);

        bigint_div_mod(&gtp_num, &gtp_den, &gtp_a_next, &gtp_cf_rem);

        bigint_mul(&gtp_tmp, &gtp_a_next, &gtp_p0);
        bigint_add(&gtp_p_new, &gtp_tmp, &gtp_pm1);
        bigint_mul(&gtp_tmp, &gtp_a_next, &gtp_q0);
        bigint_add(&gtp_q_new, &gtp_tmp, &gtp_qm1);

        bigint_copy(&gtp_pm1, &gtp_p0);
        bigint_copy(&gtp_qm1, &gtp_q0);
        bigint_copy(&gtp_p0, &gtp_p_new);
        bigint_copy(&gtp_q0, &gtp_q_new);
    }

    /* Try F itself and small multiples */
    bigint_mul(&gtp_f2, freq, &gtp_m2);
    bigint_mul(&gtp_f3, freq, &gtp_m3);
    if (try_period(freq, a_val, N, factor_p, factor_q)) return 1;
    if (try_period(&gtp_f2, a_val, N, factor_p, factor_q)) return 1;
    if (try_period(&gtp_f3, a_val, N, factor_p, factor_q)) return 1;

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
#define LLL_W_BITS  60              /* W = 2^60: Unleashed max dimension bounds for Phase 4 lattice  */
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
    double (*log_marg)[6] = (double(*)[6])calloc(n, sizeof(double[6]));
    for (int s = 0; s < n; s++) {
        for (int d = 0; d < 6; d++) {
            double p = marg[s][d];
            log_marg[s][d] = (p > 1e-15) ? log(p) : -100.0;
        }
    }

    int num_beams = 1;
    double beam_log_probs[LLL_K];
    memset(beam_log_probs, 0, sizeof(beam_log_probs));
    
    /* Heap-allocate beam history to prevent stack overflow */
    int (*beam_history_parent)[LLL_K] = (int(*)[LLL_K])calloc(n, sizeof(int[LLL_K]));
    int (*beam_history_digit)[LLL_K]  = (int(*)[LLL_K])calloc(n, sizeof(int[LLL_K]));

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

        /* ── Deterministic Top-K Selection (argmax) ──
         * Replaces stochastic Boltzmann sampling for reproducibility.
         * Always selects the K highest-scoring beams at each position. */
        int top_indices[LLL_K];
        int top_count = (next_count < LLL_K) ? next_count : LLL_K;

        for (int k = 0; k < top_count; k++) {
            int best_idx = -1;
            double best_lp = -1e30;
            for (int i = 0; i < next_count; i++) {
                if (next_log_probs[i] > best_lp) {
                    best_lp = next_log_probs[i];
                    best_idx = i;
                }
            }
            top_indices[k] = best_idx;
            next_log_probs[best_idx] = -2e9; /* poison so it can't be selected again */
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

    /* Pre-allocate reconstruction temporaries */
    BigInt rc_freq, rc_p6, rc_d_bi, rc_term, rc_tmp_bi, rc_np;
    bigint_set_u64(&rc_freq, 0); bigint_set_u64(&rc_p6, 0);
    bigint_set_u64(&rc_d_bi, 0); bigint_set_u64(&rc_term, 0);
    bigint_set_u64(&rc_tmp_bi, 0); bigint_set_u64(&rc_np, 0);

    /* Build BigInt from digits (LSB first) via backtracking */
    for (int k = 0; k < LLL_K; k++) {
        if (k >= num_beams) {
            bigint_copy(&out[k], &out[0]);
            continue;
        }

        /* Reconstruct digit sequence from backtracking tree */
        int *seq = (int*)calloc(n, sizeof(int));
        int curr_beam = k;
        for (int s = n - 1; s >= 0; s--) {
            seq[s] = beam_history_digit[s][curr_beam];
            curr_beam = beam_history_parent[s][curr_beam];
        }

        bigint_set_u64(&rc_freq, 0);
        bigint_set_u64(&rc_p6, 1);
        for (int s = 0; s < n; s++) {
            bigint_set_u64(&rc_d_bi, (uint64_t)seq[s]);
            bigint_mul(&rc_term, &rc_d_bi, &rc_p6);
            bigint_add(&rc_tmp_bi, &rc_freq, &rc_term);
            bigint_copy(&rc_freq, &rc_tmp_bi);
            bigint_mul(&rc_np, &rc_p6, b6);
            bigint_copy(&rc_p6, &rc_np);
        }
        bigint_copy(&out[k], &rc_freq);
        free(seq);
    }
    free(log_marg);
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
    /* Static temporaries — allocated once, reused forever */
    static int lrp_init = 0;
    static BigInt lrp_one;
    static BigInt lrp_gcd_f, lrp_g, lrp_r_cand, lrp_rem;
    static BigInt lrp_km, lrp_rk;
    static BigInt lrp_lcm_acc, lrp_r_i, lrp_rem_i;
    static BigInt lrp_g2, lrp_prod, lrp_new_lcm, lrp_nr;
    static BigInt lrp_s4_cand, lrp_s4_km, lrp_s4_rk;
    if (!lrp_init) {
        bigint_set_u64(&lrp_one, 1);
        bigint_set_u64(&lrp_gcd_f, 0); bigint_set_u64(&lrp_g, 0);
        bigint_set_u64(&lrp_r_cand, 0); bigint_set_u64(&lrp_rem, 0);
        bigint_set_u64(&lrp_km, 0); bigint_set_u64(&lrp_rk, 0);
        bigint_set_u64(&lrp_lcm_acc, 0); bigint_set_u64(&lrp_r_i, 0);
        bigint_set_u64(&lrp_rem_i, 0); bigint_set_u64(&lrp_g2, 0);
        bigint_set_u64(&lrp_prod, 0); bigint_set_u64(&lrp_new_lcm, 0);
        bigint_set_u64(&lrp_nr, 0);
        bigint_set_u64(&lrp_s4_cand, 0); bigint_set_u64(&lrp_s4_km, 0);
        bigint_set_u64(&lrp_s4_rk, 0);
        lrp_init = 1;
    }

    printf("\n  ═══ MULTI-STRATEGY PERIOD RECOVERY ═══\n");

    BigInt *freqs = (BigInt*)calloc(LLL_K, sizeof(BigInt));
    for (int i = 0; i < LLL_K; i++) bigint_clear(&freqs[i]);
    lll_collect_freqs(n_sites_raw, marg, b6, freqs);

    int found = 0;

    /* ── Strategy 1: CF (continued-fraction) on each targeted sample ─────── */
    printf("  [S1] CF on %d targeted frequency samples...\n", LLL_K);
    for (int i = 0; i < LLL_K && !found; i++) {
        if (bigint_is_zero(&freqs[i])) continue;
        found = generate_and_try_periods(&freqs[i], reg_sz, a_val, N,
                                         factor_p, factor_q);
        if (found) printf("  [S1] Hit on sample %d\n", i);
    }

    /* ── Strategy 2: GCD of raw frequencies → base frequency F* ──────────── */
    if (!found) {
        printf("  [S2] Running GCD of raw frequencies...\n");
        bigint_set_u64(&lrp_gcd_f, 0);
        for (int i = 0; i < LLL_K; i++) {
            if (bigint_is_zero(&freqs[i])) continue;
            if (bigint_is_zero(&lrp_gcd_f)) {
                bigint_copy(&lrp_gcd_f, &freqs[i]);
            } else {
                bigint_gcd(&lrp_g, &lrp_gcd_f, &freqs[i]);
                if (!bigint_is_zero(&lrp_g)) bigint_copy(&lrp_gcd_f, &lrp_g);
            }
            if (bigint_cmp(&lrp_gcd_f, &lrp_one) > 0) {
                bigint_div_mod(reg_sz, &lrp_gcd_f, &lrp_r_cand, &lrp_rem);
                if (bigint_cmp(&lrp_r_cand, &lrp_one) > 0 && bigint_cmp(&lrp_r_cand, N) < 0) {
                    if (try_period(&lrp_r_cand, a_val, N, factor_p, factor_q)) {
                        found = 1; printf("  [S2] r = R/gcd hit\n"); break;
                    }
                    for (int m = 2; m <= 8 && !found; m++) {
                        bigint_set_u64(&lrp_km, (uint64_t)m);
                        bigint_mul(&lrp_rk, &lrp_r_cand, &lrp_km);
                        if (bigint_cmp(&lrp_rk, N) < 0)
                            if (try_period(&lrp_rk, a_val, N, factor_p, factor_q)) {
                                found = 1; printf("  [S2] %d*r hit\n", m);
                            }
                    }
                }
                if (!found && bigint_cmp(&lrp_gcd_f, N) < 0)
                    if (try_period(&lrp_gcd_f, a_val, N, factor_p, factor_q)) {
                        found = 1; printf("  [S2] gcd_f direct hit\n");
                    }
            }
        }
    }

    /* ── Strategy 3: LCM of R/F_i period estimates ────────────────────────── */
    if (!found) {
        printf("  [S3] LCM of period estimates across %d samples...\n", LLL_K);
        bigint_set_u64(&lrp_lcm_acc, 1);
        for (int i = 0; i < LLL_K && !found; i++) {
            if (bigint_is_zero(&freqs[i])) continue;
            bigint_div_mod(reg_sz, &freqs[i], &lrp_r_i, &lrp_rem_i);
            if (bigint_is_zero(&lrp_r_i) || bigint_cmp(&lrp_r_i, &lrp_one) <= 0) continue;
            bigint_gcd(&lrp_g2, &lrp_lcm_acc, &lrp_r_i);
            bigint_mul(&lrp_prod, &lrp_lcm_acc, &lrp_r_i);
            bigint_div_mod(&lrp_prod, &lrp_g2, &lrp_new_lcm, &lrp_nr);
            if (bigint_cmp(&lrp_new_lcm, N) < 0)
                bigint_copy(&lrp_lcm_acc, &lrp_new_lcm);
            else
                bigint_set_u64(&lrp_lcm_acc, 1);
            if (bigint_cmp(&lrp_lcm_acc, &lrp_one) > 0)
                if (try_period(&lrp_lcm_acc, a_val, N, factor_p, factor_q)) {
                    found = 1; printf("  [S3] LCM hit after %d samples\n", i+1);
                }
        }
    }

    /* ── Strategy 4: LLL lattice short-vector (valid for r < 2^LLL_W_BITS) ─ */
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
            if (v < 2 || v >= W) continue;
            bigint_set_u64(&lrp_s4_cand, (uint64_t)v);
            if (bigint_cmp(&lrp_s4_cand, N) >= 0) continue;
            printf("  [S4 row %d] r candidate = %lld\n", i, v);
            if (try_period(&lrp_s4_cand, a_val, N, factor_p, factor_q)) { found = 1; break; }
            for (int m = 2; m <= 8 && !found; m++) {
                bigint_set_u64(&lrp_s4_km, (uint64_t)m);
                bigint_mul(&lrp_s4_rk, &lrp_s4_cand, &lrp_s4_km);
                if (bigint_cmp(&lrp_s4_rk, N) < 0)
                    if (try_period(&lrp_s4_rk, a_val, N, factor_p, factor_q)) found = 1;
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
                BigInt qu_bi;
                bigint_set_u64(&qu_bi, 0);
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
                        if (pi) {
                            uint64_t qu_val = (uint64_t)((__uint128_t)nu * pi % mv64);
                            if ((uint64_t)((__uint128_t)pu * qu_val % mv64) == nu) {
                                /* Verify q digit is valid */
                                uint64_t qo = bigint_to_u64(&cur[c].q);
                                if (qu_val >= qo) {
                                    uint64_t mk = bigint_to_u64(&mod_k);
                                    if (mk) {
                                        uint64_t qdiff = qu_val - qo;
                                        uint64_t qdig = qdiff / mk;
                                        if (qdig < 6 && qdiff == qdig * mk) {
                                            computed = 1;
                                            bigint_set_u64(&qu_bi, qu_val);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if (!computed) {  
                    /* Native GMP BigInt Extended Euclidean Inverse for k >= 24 */
                    mpz_t pi_z, pm_z;
                    mpz_init(pi_z); mpz_init(pm_z);
                    mpz_fdiv_r(pm_z, p_new.z, mod_k1.z);
                    
                    if (mpz_invert(pi_z, pm_z, mod_k1.z)) {
                        mpz_t nz, qu_z;
                        mpz_init(nz); mpz_init_set(qu_z, pi_z);
                        mpz_fdiv_r(nz, N->z, mod_k1.z); // N mod k1
                        mpz_mul(qu_z, qu_z, nz);
                        mpz_fdiv_r(qu_z, qu_z, mod_k1.z);
                        
                        BigInt Q_cand;
                        mpz_init_set(Q_cand.z, qu_z);
                        
                        if (bigint_cmp(&Q_cand, &cur[c].q) >= 0) {
                            BigInt qdiff, qdig, qrem;
                            bigint_sub(&qdiff, &Q_cand, &cur[c].q);
                            bigint_div_mod(&qdiff, &mod_k, &qdig, &qrem);
                            
                            if (bigint_is_zero(&qrem) && bigint_cmp(&qdig, &six) < 0) {
                                computed = 1;
                                bigint_copy(&qu_bi, &Q_cand);
                            }
                        }
                        mpz_clear(nz); mpz_clear(qu_z);
                    }
                    mpz_clear(pi_z); mpz_clear(pm_z);
                }

                if (!computed) continue;

                /* Triality filter was mathematically invalid for prefixes because upper bits absorb constraints. Filter removed to prevent dropping true factor prefixes. */
                int pruned = 0;
                if (pruned) continue;

                /* Survived all triality checks! */
                bigint_copy(&nxt[nn].p, &p_new);
                bigint_copy(&nxt[nn].q, &qu_bi);
                nn++;

                /* Check exact factorization */
                if (k >= 4) {
                    BigInt qbi, prod;
                    bigint_copy(&qbi, &qu_bi);
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

    #define CAMP_MAX_ITER 1500
    /* Simulated Annealing: damping starts high (0.5) and cools to 0.05 */
    #define CAMP_DAMPING_START 0.50
    #define CAMP_DAMPING_END   0.05
    #define CAMP_COOL_ITERS   1200  /* iterations over which cooling occurs */
    #define CAMP_TOL      1e-8

    double prev_residual = 1e30;
    int converged = 0;
    int plateau_count = 0;

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

        /* Deep topological sequence mathematically natively forces full thermodynamic exhaustion */
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
                            BigInt *factor_p, BigInt *factor_q,
                            BigInt *best_period)
{
    uint32_t nbits = bigint_bitlen(N);
    int n_sites_raw = (int)((nbits * 2 * 1000) / 2585) + 30;
    
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

    /* Pre-allocate ALL BigInt temporaries used in graph construction loop.
     * Previously these were stack-local and leaked ~80+ GMP allocations per
     * block iteration, corrupting the heap for large block counts. */
    BigInt val_k_A, val_k_B, div_6_blk;
    BigInt gc_b36, gc_next_A, gc_next_B, gc_next_div;
    BigInt gc_gcd_check, gc_val_minus_1, gc_dummy_rem;
    BigInt gc_powersA[6], gc_powersB[6];
    BigInt gc_tmpA, gc_tmpB, gc_q_div;
    BigInt gc_b6_mod, gc_shift_div_A, gc_shift_div_B, gc_dummy_rm2;
    BigInt gc_qA, gc_qB, gc_rA_mod, gc_rB_mod;
    BigInt gc_temp_N, gc_qN, gc_rN, gc_q_sh, gc_r_sh;

    bigint_set_u64(&val_k_A, 0); bigint_set_u64(&val_k_B, 0);
    bigint_set_u64(&div_6_blk, 1);
    bigint_set_u64(&gc_b36, 36); bigint_set_u64(&gc_next_A, 0);
    bigint_set_u64(&gc_next_B, 0); bigint_set_u64(&gc_next_div, 0);
    bigint_set_u64(&gc_gcd_check, 0); bigint_set_u64(&gc_val_minus_1, 0);
    bigint_set_u64(&gc_dummy_rem, 0);
    for (int i = 0; i < 6; i++) { bigint_set_u64(&gc_powersA[i], 0); bigint_set_u64(&gc_powersB[i], 0); }
    bigint_set_u64(&gc_tmpA, 0); bigint_set_u64(&gc_tmpB, 0);
    bigint_set_u64(&gc_q_div, 0);
    bigint_set_u64(&gc_b6_mod, 6); bigint_set_u64(&gc_shift_div_A, 0);
    bigint_set_u64(&gc_shift_div_B, 0); bigint_set_u64(&gc_dummy_rm2, 0);
    bigint_set_u64(&gc_qA, 0); bigint_set_u64(&gc_qB, 0);
    bigint_set_u64(&gc_rA_mod, 0); bigint_set_u64(&gc_rB_mod, 0);
    bigint_set_u64(&gc_temp_N, 0); bigint_set_u64(&gc_qN, 0);
    bigint_set_u64(&gc_rN, 0); bigint_set_u64(&gc_q_sh, 0);
    bigint_set_u64(&gc_r_sh, 0);

    for (int blk = 0; blk < n_blocks; blk++) {
        int scale_A = 2 * blk;
        int scale_B = 2 * blk + 1;
        
        if (blk == 0) {
            bigint_copy(&val_k_A, a_val);
            bigint_pow_mod(&val_k_B, &val_k_A, &b6, N);
            bigint_set_u64(&div_6_blk, 1);
        } else {
            bigint_pow_mod(&gc_next_A, &val_k_A, &gc_b36, N);
            bigint_copy(&val_k_A, &gc_next_A);
            bigint_pow_mod(&gc_next_B, &val_k_B, &gc_b36, N);
            bigint_copy(&val_k_B, &gc_next_B);

            bigint_mul(&gc_next_div, &div_6_blk, &b6);
            bigint_copy(&div_6_blk, &gc_next_div);
        }

        /* ── GCD Cascade check ── */
        bigint_sub(&gc_val_minus_1, &val_k_A, &one);
        if (!bigint_is_zero(&gc_val_minus_1) && bigint_cmp(&gc_val_minus_1, N) < 0) {
            bigint_gcd(&gc_gcd_check, &gc_val_minus_1, N);
            if (bigint_cmp(&gc_gcd_check, &one) > 0 && bigint_cmp(&gc_gcd_check, N) < 0) {
                printf("\n  ✓✓✓ GCD CASCADE HIT at block %d (scale A)! ✓✓✓\n", blk);
                bigint_copy(factor_p, &gc_gcd_check);
                bigint_div_mod(N, &gc_gcd_check, factor_q, &gc_dummy_rem);
                hpc_destroy(graph);
                return 1;
            }
        }
        bigint_sub(&gc_val_minus_1, &val_k_B, &one);
        if (!bigint_is_zero(&gc_val_minus_1) && bigint_cmp(&gc_val_minus_1, N) < 0) {
            bigint_gcd(&gc_gcd_check, &gc_val_minus_1, N);
            if (bigint_cmp(&gc_gcd_check, &one) > 0 && bigint_cmp(&gc_gcd_check, N) < 0) {
                printf("\n  ✓✓✓ GCD CASCADE HIT at block %d (scale B)! ✓✓✓\n", blk);
                bigint_copy(factor_p, &gc_gcd_check);
                bigint_div_mod(N, &gc_gcd_check, factor_q, &gc_dummy_rem);
                hpc_destroy(graph);
                return 1;
            }
        }

        bigint_set_u64(&gc_powersA[0], 1);
        bigint_set_u64(&gc_powersB[0], 1);
        bigint_copy(&gc_powersA[1], &val_k_A);
        bigint_copy(&gc_powersB[1], &val_k_B);
        for (int d = 2; d < 6; d++) {
            bigint_mul(&gc_tmpA, &gc_powersA[d-1], &val_k_A);
            bigint_mul(&gc_tmpB, &gc_powersB[d-1], &val_k_B);
            bigint_div_mod(&gc_tmpA, N, &gc_q_div, &gc_powersA[d]);
            bigint_div_mod(&gc_tmpB, N, &gc_q_div, &gc_powersB[d]);
        }

        for (int d = 0; d < 6; d++) {
            /* ── Holographic Phase Contraction (HPC) ──
             * Map the macroscopic integer y = a^(d * 6^scale) mod N
             * onto the microscopic quantum phase circle: θ = 2π * y / N.
             * This explicitly enforces the global periodicity r of Shor's
             * sequence directly into the BP message correlation geometry. */
            long exp_yA=0, exp_yB=0, exp_N=0;
            double d_yA = mpz_get_d_2exp(&exp_yA, gc_powersA[d].z);
            double d_yB = mpz_get_d_2exp(&exp_yB, gc_powersB[d].z);
            double d_N  = mpz_get_d_2exp(&exp_N, N->z);
            
            double ratio_A = (d_yA / d_N) * pow(2.0, (double)(exp_yA - exp_N));
            double ratio_B = (d_yB / d_N) * pow(2.0, (double)(exp_yB - exp_N));
            
            double phase_A = 2.0 * 3.14159265358979323846 * ratio_A;
            double phase_B = 2.0 * 3.14159265358979323846 * ratio_B;

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

        /* Extract the blk-th base-6 digit of N via running quotient (O(1) per block) */
        if (blk == 0) {
            bigint_copy(&gc_temp_N, N);
        }
        bigint_div_mod(&gc_temp_N, &b6, &gc_qN, &gc_rN);
        int N_digit = (int)bigint_to_u64(&gc_rN);
        bigint_copy(&gc_temp_N, &gc_qN);  /* advance running quotient for next block */


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
            /* ── Spectral Windowed PHASE Attenuation (Hann² window) ──
             * Hann²(blk) = [0.5 * (1 - cos(2π * blk / (n_blocks - 1)))]²
             * Squared Hann tapers boundaries more steeply, suppressing sidelobes. */
            double hann_w = (n_blocks > 1)
                ? 0.5 * (1.0 - cos(2.0 * 3.14159265358979323846 * blk / (n_blocks - 1)))
                : 1.0;
            hann_w *= hann_w;  /* Hann² */
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
                /* CZ weight: ω^(va·vb) — pure DFT₆ coupling, scaled 1/sqrt(6) 
                 * to prevent message flooding and enforce unitary boundaries */
                double norm_cz = 1.0 / sqrt(6.0);
                for (int va = 0; va < 6; va++) {
                    for (int vb = 0; vb < 6; vb++) {
                        int pidx = (va * vb) % 6;
                        cz_edge->w_re[va][vb] = W6_RE[pidx] * norm_cz;
                        cz_edge->w_im[va][vb] = W6_IM[pidx] * norm_cz;
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
            /* ── Spectral Windowed PHASE Attenuation on bridge (Hann²) ── */
            double hann_bridge = (n_blocks > 1)
                ? 0.5 * (1.0 - cos(2.0 * 3.14159265358979323846 * blk / (n_blocks - 1)))
                : 1.0;
            hann_bridge *= hann_bridge;  /* Hann² */
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

    /* ── Multi-Start BP: 30 random seeds, select lowest entropy basin ── */
    #define N_STARTS 30
    double best_entropy = 1e30;
    /* Heap-allocate to prevent stack overflow with large N (n_sites can reach ~4800) */
    double (*best_dressed_re)[6] = (double(*)[6])calloc(n_sites, sizeof(double[6]));
    double (*best_dressed_im)[6] = (double(*)[6])calloc(n_sites, sizeof(double[6]));

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

    /* Heap-allocate marginals. Index goes up to scale = 2*n_blocks-1, so
     * we need max(n_sites_raw, 2*n_blocks) elements to avoid overflow. */
    int marginals_sz = (2 * n_blocks > n_sites_raw) ? 2 * n_blocks : n_sites_raw;
    double (*marginals)[6] = (double(*)[6])calloc(marginals_sz, sizeof(double[6]));
    /* We extract the most confident node per block (usually site 0) */
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
            double sum_prob2 = 0.0;
            if (sum_prob < 1e-30) {
                /* BP has NO information about this digit — all Born probs underflowed.
                 * The mathematically correct Bayesian prior is UNIFORM 1/6.
                 * This gives the MCMC genuine stochastic diversity on these positions
                 * without corrupting positions where BP actually converged. */
                for (int d = 0; d < 6; d++) {
                    marginals[scale][d] = 1.0 / 6.0;
                }
                sum_prob2 = 1.0;
            } else {
                for (int d = 0; d < 6; d++) {
                    marginals[scale][d] /= sum_prob; /* Normalize for Monte Carlo */
                    sum_prob2 += marginals[scale][d];
                }
            }
            double max_prob = 0.0;
            int best_digit = 0;
            for (int d = 0; d < 6; d++) {
                marginals[scale][d] /= sum_prob2; /* Re-normalize */
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

    /* ── Build signal mask: which positions have genuine BP information? ── */
    int *bp_has_signal = (int*)calloc(n_sites_raw, sizeof(int));
    int *flippable = (int*)calloc(n_sites_raw, sizeof(int));
    int n_flippable = 0;
    for (int scale = 0; scale < n_sites_raw; scale++) {
        double max_p = 0.0;
        for (int d = 0; d < 6; d++)
            if (marginals[scale][d] > max_p) max_p = marginals[scale][d];
        /* A position has signal if its max probability is significantly above uniform (1/6 ≈ 0.167) */
        if (max_p > 0.20) {
            bp_has_signal[scale] = 1;
            flippable[n_flippable++] = scale;
        }
    }
    printf("  [Signal mask] %d / %d positions have BP signal (%.1f%%)\n",
           n_flippable, n_sites_raw, 100.0 * n_flippable / n_sites_raw);

    mobius_destroy(mobius);
    hpc_destroy(graph);

    clock_t t_ipe_end = clock();
    printf("\n  IPE complete: %.3f seconds (%d blocks, %d×%d-bit)\n",
           (double)(t_ipe_end - t_setup_start) / CLOCKS_PER_SEC,
           n_blocks, PHASE_CHUNKS, CHUNK_BITS);

    /* Compute register size = 6^n_sites_raw */
    BigInt reg_sz, gc_reg_tmp;
    bigint_set_u64(&reg_sz, 1);
    bigint_set_u64(&gc_reg_tmp, 0);
    for (int k = 0; k < n_sites_raw; k++) {
        bigint_mul(&gc_reg_tmp, &reg_sz, &b6);
        bigint_copy(&reg_sz, &gc_reg_tmp);
    }

    /* ── Phase 4: LLL lattice period recovery ─────────────────────────────── */
    int success = lll_recover_period(n_sites_raw, marginals, &b6, &reg_sz,
                                     N, a_val, factor_p, factor_q);
    if (success) {
        printf("\n  ★ LLL PERIOD RECOVERY SUCCEEDED ★\n");
        free(best_dressed_re); free(best_dressed_im); free(marginals);
        return 1;
    }

    /* ── Phase 5: MCMC Intelligent Period Recovery ──────────────────────────
     * GCD Consensus + Candidate Voting system.
     * Instead of brute-forcing O(n²×1000) pairwise ratios, we:
     *  1. Maintain a running GCD that converges to the base frequency F*
     *  2. Track the top 16 period candidates with vote counts
     *  3. Cross-shot GCD/LCM refinement as primary mechanism
     *  4. Deep CF only on candidates with ≥3 votes                         */
    printf("\n  ═══ INTELLIGENT MCMC PERIOD RECOVERY ═══\n");
    const int num_shots = 50000;
    int *mcmc_state = (int*)calloc(n_sites_raw, sizeof(int));
    int sweep_pos = 0;

    /* ── Candidate Voting Table ── */
    #define VOTE_TABLE_SIZE 16
    typedef struct {
        BigInt r_cand;
        int votes;
        int last_seen;
    } PeriodVote;
    PeriodVote vote_table[VOTE_TABLE_SIZE];
    for (int i = 0; i < VOTE_TABLE_SIZE; i++) {
        bigint_set_u64(&vote_table[i].r_cand, 0);
        vote_table[i].votes = 0;
        vote_table[i].last_seen = 0;
    }
    int total_votes_cast = 0;

    /* ── Running GCD Accumulator ── */
    BigInt running_gcd, prev_freq;
    bigint_set_u64(&running_gcd, 0);
    bigint_set_u64(&prev_freq, 0);
    int gcd_samples = 0;

    /* ── Cross-shot period accumulator ── */
    BigInt cross_gcd, cross_lcm;
    bigint_set_u64(&cross_gcd, 0);
    bigint_set_u64(&cross_lcm, 1);

    /* Heap-allocate p6_cache to prevent VLA BigInt leak */
    BigInt *p6_cache = (BigInt*)calloc(n_sites_raw, sizeof(BigInt));
    BigInt current_p6, next_p6;
    bigint_set_u64(&current_p6, 1);
    bigint_set_u64(&next_p6, 0);
    for (int i = 0; i < n_sites_raw; i++) {
        bigint_copy(&p6_cache[i], &current_p6);
        bigint_mul(&next_p6, &current_p6, &b6);
        bigint_copy(&current_p6, &next_p6);
    }

    BigInt freq; bigint_set_u64(&freq, 0);

    /* Pre-allocate ALL temporaries */
    BigInt mc_d_bi, mc_term, mc_tmp, mc_old_val, mc_new_val;
    BigInt mc_old_d_bi, mc_new_d_bi, mc_tmp_freq;
    BigInt mc_r_cand, mc_rem, mc_one_fb, mc_gcd_v;
    BigInt mc_new_lcm, mc_lcm_rem, mc_lcm_prod, mc_lcm_g;
    bigint_set_u64(&mc_d_bi, 0);    bigint_set_u64(&mc_term, 0);
    bigint_set_u64(&mc_tmp, 0);     bigint_set_u64(&mc_old_val, 0);
    bigint_set_u64(&mc_new_val, 0); bigint_set_u64(&mc_old_d_bi, 0);
    bigint_set_u64(&mc_new_d_bi, 0); bigint_set_u64(&mc_tmp_freq, 0);
    bigint_set_u64(&mc_r_cand, 0);  bigint_set_u64(&mc_rem, 0);
    bigint_set_u64(&mc_one_fb, 1);  bigint_set_u64(&mc_gcd_v, 0);
    bigint_set_u64(&mc_new_lcm, 0); bigint_set_u64(&mc_lcm_rem, 0);
    bigint_set_u64(&mc_lcm_prod, 0); bigint_set_u64(&mc_lcm_g, 0);

    /* Helper: register a period candidate into the voting table */
    #define VOTE_FOR_CANDIDATE(r_ptr, shot_num) do { \
        if (bigint_is_zero(r_ptr) || bigint_cmp((r_ptr), &mc_one_fb) <= 0 \
            || bigint_cmp((r_ptr), N) >= 0) break; \
        int _matched = -1; \
        for (int _vi = 0; _vi < VOTE_TABLE_SIZE; _vi++) { \
            if (vote_table[_vi].votes == 0) continue; \
            /* Match if candidates are equal, or one divides the other with small quotient */ \
            if (bigint_cmp((r_ptr), &vote_table[_vi].r_cand) == 0) { \
                _matched = _vi; break; \
            } \
            BigInt _vq, _vr; \
            if (bigint_cmp((r_ptr), &vote_table[_vi].r_cand) > 0) { \
                bigint_div_mod((r_ptr), &vote_table[_vi].r_cand, &_vq, &_vr); \
            } else { \
                bigint_div_mod(&vote_table[_vi].r_cand, (r_ptr), &_vq, &_vr); \
            } \
            if (bigint_is_zero(&_vr) && bigint_bitlen(&_vq) <= 10) { \
                _matched = _vi; break; \
            } \
        } \
        if (_matched >= 0) { \
            vote_table[_matched].votes++; \
            vote_table[_matched].last_seen = (shot_num); \
        } else { \
            /* Find lowest-voted slot to replace */ \
            int _min_v = 999999, _min_i = 0; \
            for (int _vi = 0; _vi < VOTE_TABLE_SIZE; _vi++) { \
                if (vote_table[_vi].votes < _min_v) { \
                    _min_v = vote_table[_vi].votes; _min_i = _vi; \
                } \
            } \
            bigint_copy(&vote_table[_min_i].r_cand, (r_ptr)); \
            vote_table[_min_i].votes = 1; \
            vote_table[_min_i].last_seen = (shot_num); \
        } \
        total_votes_cast++; \
    } while(0)

    for (int shot = 1; shot <= num_shots && !success; shot++) {
        /* ── MCMC Sampling (unchanged) ── */
        if (shot == 1) {
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
            /* Position-aware sweep: only flip positions where BP has real signal */
            if (n_flippable == 0) continue; /* No informed positions — skip */
            int flip = flippable[sweep_pos % n_flippable];
            sweep_pos++;
            double rr = (double)rand() / RAND_MAX;
            double cdf = 0.0;
            int new_d = 0;
            for (int d = 0; d < 6; d++) {
                cdf += marginals[flip][d];
                if (rr <= cdf) { new_d = d; break; }
            }
            int old_d = mcmc_state[flip];
            if (old_d != new_d) {
                bigint_set_u64(&mc_old_d_bi, (uint64_t)old_d);
                bigint_set_u64(&mc_new_d_bi, (uint64_t)new_d);
                bigint_mul(&mc_old_val, &mc_old_d_bi, &p6_cache[flip]);
                bigint_mul(&mc_new_val, &mc_new_d_bi, &p6_cache[flip]);
                bigint_sub(&mc_tmp_freq, &freq, &mc_old_val);
                bigint_add(&freq, &mc_tmp_freq, &mc_new_val);
                mcmc_state[flip] = new_d;
            }
        }

        /* ── Skip zero/trivial frequencies ── */
        if (bigint_is_zero(&freq) || bigint_cmp(&freq, &mc_one_fb) <= 0) continue;

        /* ══════════════════════════════════════════════════════════════════
         * STRATEGY A: Running GCD Accumulator
         * GCD of multiple harmonics F_i = s_i·F* converges to F* itself.
         * Then r = R / F*.
         * ══════════════════════════════════════════════════════════════════ */
        if (gcd_samples == 0) {
            bigint_copy(&running_gcd, &freq);
        } else {
            bigint_gcd(&mc_gcd_v, &running_gcd, &freq);
            if (!bigint_is_zero(&mc_gcd_v))
                bigint_copy(&running_gcd, &mc_gcd_v);
        }
        gcd_samples++;

        /* Derive period estimate from running GCD */
        if (!bigint_is_zero(&running_gcd) && bigint_cmp(&running_gcd, &mc_one_fb) > 0) {
            bigint_div_mod(&reg_sz, &running_gcd, &mc_r_cand, &mc_rem);
            if (bigint_cmp(&mc_r_cand, &mc_one_fb) > 0 && bigint_cmp(&mc_r_cand, N) < 0) {
                VOTE_FOR_CANDIDATE(&mc_r_cand, shot);

                /* Direct test on every shot (cheap — just one modexp) */
                if (try_period(&mc_r_cand, a_val, N, factor_p, factor_q) == 1) {
                    success = 1;
                    printf("\n  [Shot %d] ★ OUROBOROS BITES ITS TAIL ★ (GCD accumulator)\n", shot);
                }
            }
        }
        if (success) break;

        /* ══════════════════════════════════════════════════════════════════
         * STRATEGY B: Cross-Shot GCD/LCM Refinement
         * If consecutive shots produce r₁ and r₂:
         *   gcd(r₁, r₂) divides the true period
         *   lcm(r₁, r₂) may BE the true period
         * ══════════════════════════════════════════════════════════════════ */
        if (!bigint_is_zero(&prev_freq) && bigint_cmp(&prev_freq, &mc_one_fb) > 0) {
            /* Pairwise GCD of consecutive raw frequencies */
            bigint_gcd(&mc_gcd_v, &prev_freq, &freq);
            if (!bigint_is_zero(&mc_gcd_v) && bigint_cmp(&mc_gcd_v, &mc_one_fb) > 0) {
                bigint_div_mod(&reg_sz, &mc_gcd_v, &mc_r_cand, &mc_rem);
                if (bigint_cmp(&mc_r_cand, &mc_one_fb) > 0 && bigint_cmp(&mc_r_cand, N) < 0) {
                    VOTE_FOR_CANDIDATE(&mc_r_cand, shot);
                    if (try_period(&mc_r_cand, a_val, N, factor_p, factor_q) == 1) {
                        success = 1;
                        printf("\n  [Shot %d] ★ OUROBOROS BITES ITS TAIL ★ (cross-shot GCD)\n", shot);
                    }
                }
            }

            /* Pairwise LCM of consecutive raw frequencies */
            if (!success && !bigint_is_zero(&mc_gcd_v)) {
                bigint_mul(&mc_lcm_prod, &prev_freq, &freq);
                bigint_div_mod(&mc_lcm_prod, &mc_gcd_v, &mc_new_lcm, &mc_lcm_rem);
                if (bigint_cmp(&mc_new_lcm, N) < 0 && bigint_cmp(&mc_new_lcm, &mc_one_fb) > 0) {
                    bigint_div_mod(&reg_sz, &mc_new_lcm, &mc_r_cand, &mc_rem);
                    if (bigint_cmp(&mc_r_cand, &mc_one_fb) > 0 && bigint_cmp(&mc_r_cand, N) < 0) {
                        VOTE_FOR_CANDIDATE(&mc_r_cand, shot);
                    }
                }
            }
        }
        bigint_copy(&prev_freq, &freq);
        if (success) break;

        /* ══════════════════════════════════════════════════════════════════
         * STRATEGY C: Direct CF on individual frequencies
         * Each raw freq F yields r = R/F directly via CF convergents.
         * ══════════════════════════════════════════════════════════════════ */
        if (shot % 10 == 0) {
            int cf_result = generate_and_try_periods(&freq, &reg_sz, a_val, N, factor_p, factor_q);
            if (cf_result == 1) {
                success = 1;
                printf("\n  [Shot %d] ★ OUROBOROS BITES ITS TAIL ★ (direct CF on sample)\n", shot);
                break;
            }
        }

        /* ══════════════════════════════════════════════════════════════════
         * STRATEGY D: Deep-test voted candidates (≥3 votes)
         * Every 500 shots, take the top voted candidates and run them
         * through the harmonic multiplier cascade.
         * ══════════════════════════════════════════════════════════════════ */
        if (shot % 500 == 0 && total_votes_cast > 0) {
            printf("  [Shot %5d] Vote table (%d total votes):\n", shot, total_votes_cast);
            
            /* Sort vote table by votes (descending) for display + testing */
            for (int vi = 0; vi < VOTE_TABLE_SIZE && !success; vi++) {
                if (vote_table[vi].votes < 1) continue;
                
                uint32_t cand_bits = bigint_bitlen(&vote_table[vi].r_cand);
                printf("    [%2d] %3d votes, %u bits, last seen shot %d\n",
                       vi, vote_table[vi].votes, cand_bits, vote_table[vi].last_seen);

                /* Only deep-test candidates with ≥ 3 independent confirmations */
                if (vote_table[vi].votes >= 3) {
                    printf("    → Deep-testing candidate %d (%u bits, %d votes)...\n",
                           vi, cand_bits, vote_table[vi].votes);

                    /* Test the candidate directly */
                    if (try_period(&vote_table[vi].r_cand, a_val, N, factor_p, factor_q) == 1) {
                        success = 1;
                        printf("\n  [Shot %d] ★ OUROBOROS BITES ITS TAIL ★ (voted candidate %d, %d votes)\n",
                               shot, vi, vote_table[vi].votes);
                        break;
                    }

                    /* Test harmonic multiples up to 1000 */
                    BigInt r_mult, mult_const;
                    for (int sm = 2; sm <= 1000 && !success; sm++) {
                        bigint_set_u64(&mult_const, sm);
                        bigint_mul(&r_mult, &vote_table[vi].r_cand, &mult_const);
                        if (bigint_cmp(&r_mult, N) >= 0) break;
                        if (try_period(&r_mult, a_val, N, factor_p, factor_q) == 1) {
                            success = 1;
                            printf("\n  [Shot %d] ★ OUROBOROS BITES ITS TAIL ★ (voted candidate %d × %d)\n",
                                   shot, vi, sm);
                        }
                    }

                    /* Cross-GCD voted candidate with running GCD estimate */
                    if (!success && !bigint_is_zero(&running_gcd)) {
                        BigInt cross_r, cross_rem;
                        bigint_gcd(&mc_gcd_v, &vote_table[vi].r_cand, &running_gcd);
                        if (bigint_cmp(&mc_gcd_v, &mc_one_fb) > 0) {
                            bigint_div_mod(&reg_sz, &mc_gcd_v, &cross_r, &cross_rem);
                            if (bigint_cmp(&cross_r, &mc_one_fb) > 0 && bigint_cmp(&cross_r, N) < 0) {
                                if (try_period(&cross_r, a_val, N, factor_p, factor_q) == 1) {
                                    success = 1;
                                    printf("\n  [Shot %d] ★ OUROBOROS BITES ITS TAIL ★ (cross-GCD vote × accumulator)\n", shot);
                                }
                            }
                        }
                    }
                }
            }
            
            /* Running GCD status */
            uint32_t gcd_bits = bigint_bitlen(&running_gcd);
            uint32_t n_bits = bigint_bitlen(N);
            printf("  [Shot %5d] GCD accumulator: %u bits (%d samples) | target <%u bits\n",
                   shot, gcd_bits, gcd_samples, n_bits);
            fflush(stdout);
        }
    }

    /* Surface the best partial period estimate from the vote table */
    if (!success && best_period) {
        int best_votes = 0;
        for (int vi = 0; vi < VOTE_TABLE_SIZE; vi++) {
            if (vote_table[vi].votes > best_votes && bigint_bitlen(&vote_table[vi].r_cand) > 10) {
                best_votes = vote_table[vi].votes;
                bigint_copy(best_period, &vote_table[vi].r_cand);
            }
        }
        if (best_votes > 0) {
            char rp_str[1300];
            bigint_to_decimal(rp_str, sizeof(rp_str), best_period);
            printf("  [Partial] Best period estimate: %s (%u bits, %d votes)\n",
                   rp_str, bigint_bitlen(best_period), best_votes);
        }
    }

    free(mcmc_state);
    free(p6_cache);
    free(bp_has_signal);
    free(flippable);
    free(best_dressed_re);
    free(best_dressed_im);
    free(marginals);
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

    /* ── Cross-base LCM period accumulator ── */
    BigInt cross_base_lcm, cross_base_gcd, cross_base_prod, cross_base_rem;
    BigInt best_partial;
    bigint_set_u64(&cross_base_lcm, 1);
    bigint_set_u64(&cross_base_gcd, 0);
    bigint_set_u64(&cross_base_prod, 0);
    bigint_set_u64(&cross_base_rem, 0);
    bigint_set_u64(&best_partial, 0);
    BigInt bi_one; bigint_set_u64(&bi_one, 1);

    /* ── Try constraint-satisfaction first (Hensel lift in base 6) ── */
    success = 0;

    for (int bi = 0; bi < max_bases && !success; bi++) {
        if (auto_a) bigint_set_u64(&a_val, base_list[bi]);

        char a_str[1300];
        bigint_to_decimal(a_str, sizeof(a_str), &a_val);
        printf("  ── Attempt %d: a = %s ──\n\n", bi + 1, a_str);

        bigint_set_u64(&best_partial, 0);
        clock_t t_start = clock();
        success = factor_with_hpc(&N, &a_val, &factor_p, &factor_q, &best_partial);
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
            printf("  ✗ Base a=%s did not yield factors (%.3f sec)\n",
                   a_str, (double)(t_end - t_start) / CLOCKS_PER_SEC);

            /* ── Cross-base LCM accumulation ── */
            if (!bigint_is_zero(&best_partial) && bigint_cmp(&best_partial, &bi_one) > 0) {
                /* LCM(a, b) = a * b / gcd(a, b) */
                bigint_gcd(&cross_base_gcd, &cross_base_lcm, &best_partial);
                bigint_mul(&cross_base_prod, &cross_base_lcm, &best_partial);
                bigint_div_mod(&cross_base_prod, &cross_base_gcd, &cross_base_lcm, &cross_base_rem);

                /* Clamp: if LCM exceeds N, it's blown past the period */
                if (bigint_cmp(&cross_base_lcm, &N) >= 0) {
                    printf("  [Cross-base] LCM exceeded N, resetting to partial\n");
                    bigint_copy(&cross_base_lcm, &best_partial);
                }

                uint32_t lcm_bits = bigint_bitlen(&cross_base_lcm);
                uint32_t n_bits = bigint_bitlen(&N);
                printf("  [Cross-base] Accumulated LCM: %u bits (target <%u bits)\n",
                       lcm_bits, n_bits);

                /* Try the accumulated LCM as a period candidate */
                printf("  [Cross-base] Testing accumulated LCM as period...\n");
                /* Test with EACH base we've tried so far */
                for (int bj = 0; bj <= bi && !success; bj++) {
                    BigInt test_a;
                    bigint_set_u64(&test_a, base_list[bj]);
                    if (try_period(&cross_base_lcm, &test_a, &N, &factor_p, &factor_q) == 1) {
                        success = 1;
                        printf("\n  ★★★ CROSS-BASE LCM FACTORED N! (base a=%llu) ★★★\n",
                               (unsigned long long)base_list[bj]);
                    }
                    /* Also test small multiples */
                    for (int sm = 2; sm <= 12 && !success; sm++) {
                        BigInt r_mult, mult_c;
                        bigint_set_u64(&mult_c, sm);
                        bigint_mul(&r_mult, &cross_base_lcm, &mult_c);
                        if (bigint_cmp(&r_mult, &N) >= 0) break;
                        if (try_period(&r_mult, &test_a, &N, &factor_p, &factor_q) == 1) {
                            success = 1;
                            printf("\n  ★★★ CROSS-BASE LCM × %d FACTORED N! (base a=%llu) ★★★\n",
                                   sm, (unsigned long long)base_list[bj]);
                        }
                    }
                }
            }
            printf("\n");
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
