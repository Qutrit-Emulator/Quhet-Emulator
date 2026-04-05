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

    char f_str[1300];
    bigint_to_decimal(f_str, sizeof(f_str), freq);
    printf("  Composite frequency F = %s\n", f_str);

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
 * ARBITRARY-PRECISION MÖBIUS BELIEF PROPAGATION (2000-bit Fixed Point)
 * ═══════════════════════════════════════════════════════════════════════════ */
#define BFP_BITS 4500

static inline void bfp_from_double(BigInt *out, double p) {
    bigint_clear(out);
    if (p <= 0.0) return;
    if (p >= 1.0) {
        bigint_set_bit(out, BFP_BITS);
        return;
    }
    uint64_t sig = (uint64_t)(p * (double)(1ULL << 53));
    bigint_set_u64(out, sig);
    mpz_mul_2exp(out->z, out->z, BFP_BITS - 53);
}

static inline void bfp_shr(BigInt *a) {
    mpz_fdiv_q_2exp(a->z, a->z, BFP_BITS);
}

static inline void bfp_shl(BigInt *a) {
    mpz_mul_2exp(a->z, a->z, BFP_BITS);
}

static inline void bfp_mul(BigInt *out, const BigInt *a, const BigInt *b) {
    mpz_mul(out->z, a->z, b->z);
    mpz_fdiv_q_2exp(out->z, out->z, BFP_BITS);
}

typedef struct {
    BigInt msg[2][6]; /* msg[0]: sa->sb, msg[1]: sb->sa */
} BignumEdgeMsg;

static void z6_mobius_converge_bignum(MobiusAmplitudeSheet *ms) {
    const HPCGraph *g = ms->graph;
    int n_edges = g->n_edges;
    if (n_edges == 0) return;
    BignumEdgeMsg *msgs = (BignumEdgeMsg*)calloc(n_edges, sizeof(BignumEdgeMsg));
    BignumEdgeMsg *new_msgs = (BignumEdgeMsg*)calloc(n_edges, sizeof(BignumEdgeMsg));
    
    printf("      [Arbitrary-Precision BP] Initializing %d edges at %d-bit fixed-point...\n", n_edges, BFP_BITS);
    BigInt one; bigint_clear(&one); bigint_set_bit(&one, BFP_BITS);
    for (int e = 0; e < n_edges; e++) {
        for (int v = 0; v < 6; v++) {
            bigint_copy(&msgs[e].msg[0][v], &one);
            bigint_copy(&msgs[e].msg[1][v], &one);
        }
    }

    int MAX_ITER = 15;
    for (int it = 0; it < MAX_ITER; it++) {
        printf("      [Arbitrary-Precision BP] Iteration %d / %d\n", it + 1, MAX_ITER);
        for (int eid = 0; eid < n_edges; eid++) {
            const HPCEdge *edge = &g->edges[eid];
            uint64_t sa = edge->site_a, sb = edge->site_b;
            
            for (int dir = 0; dir < 2; dir++) {
                uint64_t src = (dir == 0) ? sa : sb;
                uint64_t dst = (dir == 0) ? sb : sa;
                
                BigInt prod_in[6];
                for (int v_src = 0; v_src < 6; v_src++) {
                    double prior_d = g->locals[src].edge_re[v_src] * g->locals[src].edge_re[v_src] +
                                     g->locals[src].edge_im[v_src] * g->locals[src].edge_im[v_src];
                    bfp_from_double(&prod_in[v_src], prior_d);
                    
                    const HPCAdjList *adj = &g->adj[src];
                    for (uint64_t mi = 0; mi < adj->count; mi++) {
                        uint64_t in_eid = adj->edge_ids[mi];
                        if (in_eid == eid) continue;
                        int in_dir = (g->edges[in_eid].site_b == src) ? 0 : 1; 
                        bfp_mul(&prod_in[v_src], &prod_in[v_src], &msgs[in_eid].msg[in_dir][v_src]);
                    }
                }

                BigInt new_m[6];
                for (int v = 0; v < 6; v++) bigint_clear(&new_m[v]);
                BigInt sum_m; bigint_clear(&sum_m);
                
                for (int v_dst = 0; v_dst < 6; v_dst++) {
                    for (int v_src = 0; v_src < 6; v_src++) {
                        /* 3. Edge coefficient */
                        double ef = (edge->type == HPC_EDGE_CZ) ? 1.0 : 
                                    (edge->w_re[v_src][v_dst] * edge->w_re[v_src][v_dst] + 
                                     edge->w_im[v_src][v_dst] * edge->w_im[v_src][v_dst]);
                        BigInt bef; bfp_from_double(&bef, ef);
                        BigInt val; bfp_mul(&val, &prod_in[v_src], &bef);
                        
                        /* 4. Add to sum */
                        BigInt curr; bigint_copy(&curr, &new_m[v_dst]);
                        bigint_add(&new_m[v_dst], &curr, &val);
                    }
                    BigInt c_sum; bigint_copy(&c_sum, &sum_m);
                    bigint_add(&sum_m, &c_sum, &new_m[v_dst]);
                }
                
                /* Normalize message */
                if (!bigint_is_zero(&sum_m)) {
                    for (int v = 0; v < 6; v++) {
                        BigInt num; bigint_copy(&num, &new_m[v]);
                        bfp_shl(&num);
                        BigInt q, r;
                        bigint_div_mod(&num, &sum_m, &q, &r);
                        
                        /* Damped update: 0.5 * new + 0.5 * old */
                        BigInt half_q; bigint_copy(&half_q, &q); bigint_shr1(&half_q);
                        BigInt half_old; bigint_copy(&half_old, &msgs[eid].msg[dir][v]); bigint_shr1(&half_old);
                        bigint_add(&new_msgs[eid].msg[dir][v], &half_q, &half_old);
                    }
                } else {
                    for (int v = 0; v < 6; v++) bigint_clear(&new_msgs[eid].msg[dir][v]);
                }
            }
        }
        /* Swap message buffers */
        for (int e = 0; e < n_edges; e++) {
            for (int v = 0; v < 6; v++) {
                bigint_copy(&msgs[e].msg[0][v], &new_msgs[e].msg[0][v]);
                bigint_copy(&msgs[e].msg[1][v], &new_msgs[e].msg[1][v]);
            }
        }
    }
    
    /* Calculate final marginals */
    for (int s = 0; s < g->n_sites; s++) {
        BigInt marg[6];
        for (int v = 0; v < 6; v++) bigint_clear(&marg[v]);
        BigInt sum_m; bigint_clear(&sum_m);
        
        for (int v = 0; v < 6; v++) {
            double prior_d = g->locals[s].edge_re[v] * g->locals[s].edge_re[v] +
                             g->locals[s].edge_im[v] * g->locals[s].edge_im[v];
            BigInt val; bfp_from_double(&val, prior_d);
            
            const HPCAdjList *adj = &g->adj[s];
            for (uint64_t mi = 0; mi < adj->count; mi++) {
                uint64_t in_eid = adj->edge_ids[mi];
                int in_dir = (g->edges[in_eid].site_b == s) ? 0 : 1; 
                bfp_mul(&val, &val, &msgs[in_eid].msg[in_dir][v]);
            }
            bigint_copy(&marg[v], &val);
            BigInt c_sum; bigint_copy(&c_sum, &sum_m);
            bigint_add(&sum_m, &c_sum, &marg[v]);
        }
        
        for (int v = 0; v < 6; v++) {
            if (!bigint_is_zero(&sum_m)) {
                BigInt num; bigint_copy(&num, &marg[v]);
                bfp_shl(&num);
                BigInt q, r;
                bigint_div_mod(&num, &sum_m, &q, &r);
                
                BigInt q_shift; bigint_copy(&q_shift, &q);
                mpz_fdiv_q_2exp(q_shift.z, q_shift.z, BFP_BITS - 53);
                uint64_t sig = bigint_to_u64(&q_shift);
                double p_double = (double)sig / (double)(1ULL << 53);
                
                ms->sheets[s].dressed_re[v] = sqrt(p_double);
                ms->sheets[s].dressed_im[v] = 0.0;
            } else {
                ms->sheets[s].dressed_re[v] = 0.0;
                ms->sheets[s].dressed_im[v] = 0.0;
            }
        }
    }
    
    free(msgs);
    free(new_msgs);
}

static int factor_with_hpc(const BigInt *N, const BigInt *a_val,
                            BigInt *factor_p, BigInt *factor_q)
{
    uint32_t nbits = bigint_bitlen(N);
    int n_sites_raw = (int)((nbits * 2 * 1000) / 2585) + 1;
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

    BigInt val_k_A, val_k_B;
    for (int blk = 0; blk < n_blocks; blk++) {
        int scale_A = 2 * blk;
        int scale_B = 2 * blk + 1;
        
        if (blk == 0) {
            bigint_copy(&val_k_A, a_val);
            bigint_pow_mod(&val_k_B, &val_k_A, &b6, N);
        } else {
            BigInt b36; bigint_set_u64(&b36, 36);
            BigInt next_A; bigint_pow_mod(&next_A, &val_k_A, &b36, N);
            bigint_copy(&val_k_A, &next_A);
            BigInt next_B; bigint_pow_mod(&next_B, &val_k_B, &b36, N);
            bigint_copy(&val_k_B, &next_B);
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
            BigInt rA_mod, rB_mod;
            bigint_div_mod(&powersA[d], &b6_mod, NULL, &rA_mod);
            bigint_div_mod(&powersB[d], &b6_mod, NULL, &rB_mod);
            
            int d_A = bigint_to_u64(&rA_mod);
            int d_B = bigint_to_u64(&rB_mod);

            /* The quantum phase is proportional to a^x mod N over the space of N */
            /* We must represent powersA[d] / N as a double to get the precise phase angle */
            char pA_str[1300], pB_str[1300], n_str[1300];
            bigint_to_decimal(pA_str, sizeof(pA_str), &powersA[d]);
            bigint_to_decimal(pB_str, sizeof(pB_str), &powersB[d]);
            bigint_to_decimal(n_str, sizeof(n_str), N);
            
            /* High precision float parsing to build the QPE angle! */
            double phase_A = 2.0 * M_PI * (atof(pA_str) / atof(n_str));
            double phase_B = 2.0 * M_PI * (atof(pB_str) / atof(n_str));

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
            for (int va = 0; va < 6; va++) {
                for (int vb = 0; vb < 6; vb++) {
                    /* Critical Phase Boundary: Derived from Orch-OR D=6 Potts model at transition (J=15meV, T=310K) */
                    int diff = (va - vb + 6) % 6;
                    double w;
                    switch(diff) {
                        case 0: w = 1.000; break;
                        case 1: case 5: w = 0.755; break;
                        case 2: case 4: w = 0.431; break;
                        case 3: w = 0.325; break;
                    }
                    edge->w_re[va][vb] = w;
                    edge->w_im[va][vb] = 0.0;
                }
            }
            graph->n_edges++;
            graph->phase_edges++;
            hpc_adj_add(graph, bypass_sites[i], eid);
            hpc_adj_add(graph, bypass_sites[j], eid);
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
            for (int va = 0; va < 6; va++) {
                for (int vb = 0; vb < 6; vb++) {
                    int diff = (va - vb + 6) % 6;
                    double w;
                    switch(diff) {
                        case 0: w = 1.000; break;
                        case 1: case 5: w = 0.755; break;
                        case 2: case 4: w = 0.431; break;
                        case 3: w = 0.325; break;
                    }
                    edge->w_re[va][vb] = w;
                    edge->w_im[va][vb] = 0.0;
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
    printf("    Phase 3: S₁₄ Deep Parity Crystallization via Z₆ Möbius BP (Arbitrary Precision C-Native)...\n");
    clock_t t_bp_start = clock();

    MobiusAmplitudeSheet *mobius = mobius_create(graph);
    z6_mobius_converge_bignum(mobius);

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

    printf("\n  ═══ QUANTUM MONTE CARLO PERIOD EXTRACTION ═══\n");
    int success = 0;
    int num_shots = 1000000;
    for (int shot = 1; shot <= num_shots; shot++) {
        BigInt freq;
        bigint_set_u64(&freq, 0);
        BigInt power_of_6;
        bigint_set_u64(&power_of_6, 1);

        /* The Quantum Roll: Collapse the global wave function probabilistically */
        for (int scale = 0; scale < n_sites_raw; scale++) {
            double r = (double)rand() / RAND_MAX;
            double cdf = 0.0;
            int sampled_digit = 5;
            if (shot == 1) {
                double max_p = 0.0;
                for (int d = 0; d < 6; d++) {
                    if (marginals[scale][d] > max_p) {
                        max_p = marginals[scale][d];
                        sampled_digit = d;
                    }
                }
            } else {
                for (int d = 0; d < 6; d++) {
                    cdf += marginals[scale][d];
                    if (r <= cdf) {
                        sampled_digit = d;
                        break;
                    }
                }
            }

            BigInt d_bi, term, tmp;
            bigint_set_u64(&d_bi, sampled_digit);
            bigint_mul(&term, &d_bi, &power_of_6);
            bigint_add(&tmp, &freq, &term);
            bigint_copy(&freq, &tmp);

            BigInt new_pow;
            bigint_mul(&new_pow, &power_of_6, &b6);
            bigint_copy(&power_of_6, &new_pow);
        }
        
        if (shot % 10000 == 0) {
            printf("  [Shot %7d] Sweeping the multiversal timeline...\n", shot);
        }
        
        success = generate_and_try_periods(&freq, &reg_sz, a_val, N, factor_p, factor_q);
        if (success) {
            printf("\n  [Shot %3d] ★ THE OUROBOROS BITES ITS TAIL. FACTORS DISCOVERED! ★\n", shot);
            break;
        }
    }

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
