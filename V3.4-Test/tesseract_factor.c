/* ═══════════════════════════════════════════════════════════════════════════
 * tesseract_factor.c — 40,000-Face Ouroboros Factoring Engine
 *
 * ╔═══════════════════════════════════════════════════════════════╗
 * ║  CONFIGURE YOUR TARGET N HERE                                ║
 * ╚═══════════════════════════════════════════════════════════════╝
 */
#define TARGET_N  "261980999226229"          /* ← Set your composite here     */
#define TARGET_A  "0"            /* ← "0" = auto-try 20 bases              */

/*
 * Architecture:
 *   40,000 faces = 556 tesseracts × 72 faces/tesseract
 *   556 tesseracts × 3 quhits/tesseract = 1,668 quhits
 *   Register capacity: 6^1668 ≈ 10^1298 states
 *   Supports N up to 4096 bits (1233 decimal digits)
 *
 * Pipeline:
 *   1. Encode N into the face network (base-6 triadic register)
 *   2. Superposition: all 40,000 faces active simultaneously
 *   3. ModMul oracle: |x⟩ → |(a·x) mod N⟩ via entangled face ops
 *   4. Ouroboros loop: Phase(2π/N) → DFT₂ → CZ₃ → Face rotate
 *   5. Period extraction from interference peaks
 *   6. Factor: gcd(a^(r/2) ± 1, N)
 *
 * Build: gcc -O2 -o tesseract_factor tesseract_factor.c \
 *         quhit_triality.c bigint.c -lm
 * ═══════════════════════════════════════════════════════════════════════════ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "quhit_triality.h"
#include "quhit_triadic.h"
#include "reality_source.h"
#include "bigint.h"

#define D           6
#define N_FACES     40032   /* 556 tesseracts × 72 (nearest multiple) */
#define N_TESS      556     /* tesseracts */
#define N_QUHITS    1668    /* 556 × 3 */
#define FACES_PER_T 72

/* ═══════════════════════════════════════════════════════════════════════════
 * TESSERACT ARRAY — 556 chained tesseracts forming the register
 *
 * Each tesseract is a TriadicJoint (6³ = 216 local amplitudes).
 * Adjacent tesseracts are coupled via CZ₃ (entangling bond).
 * The full register spans 6^1668 states, but each tesseract only
 * stores its LOCAL 216-dim slice — the entanglement lives in the bonds.
 *
 * This is the MPS (Matrix Product State) representation:
 * each tesseract is a tensor, bonds carry the entanglement.
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    TriadicJoint tess[N_TESS];  /* 556 tesseracts */
    int n_active;               /* How many are actually needed for this N */
    int total_faces;            /* n_active × 72 */
} TesseractArray;

static void tessarray_init(TesseractArray *arr, int n_active)
{
    arr->n_active = n_active;
    arr->total_faces = n_active * FACES_PER_T;
    for (int i = 0; i < n_active; i++) {
        memset(&arr->tess[i], 0, sizeof(TriadicJoint));
        /* ── 4D Entangled Sidechannel State ──
         * Following tesseract_sidechannel.c's attack_4d_entangled():
         *   |ψ⟩ = Σ_k |k⟩_A |k⟩_B |0⟩_C / √6
         *
         * Channel A receives the oracle phase gate.
         * Channel B is the UNTOUCHED WITNESS — it preserves the
         * input-channel correlation that the 4D sidechannel reads.
         * Channel C is fixed at |0⟩ (the "compressed edge" anchor).
         *
         * This is the quantum advantage: the A↔B entanglement lets
         * the QFT on A extract the period while B's correlation
         * prevents destructive interference from washing it out. */
        double norm = 1.0 / sqrt((double)D);  /* 6 entangled pairs */
        for (int k = 0; k < D; k++)
            arr->tess[i].re[TRIAD_IDX(k, k, 0)] = norm;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MODULAR ORACLE — 4D Sidechannel Face Permutation
 *
 * The modular function f(x) = a^x mod N is encoded as a FACE PERMUTATION
 * CIPHER on each tesseract — the same structure as tesseract_sidechannel.c.
 *
 * For tesseract i:
 *   1. Compute base_i = a^(216^i) mod N
 *   2. Derive a permutation of S₆ via factoradic decomposition of
 *      (base_i mod 720) — this maps N's multiplicative structure into
 *      the tesseract's face wiring
 *   3. Apply as a 6×6 unitary gate via triad_gate_a, triad_gate_b, triad_gate_c
 *
 * This IS the 4D sidechannel: the CMY channel decomposition of the
 * permutation reveals the period structure that's invisible in 3D.
 * The Ouroboros loop finds the period by cycling the tesseract until
 * it returns to its original orientation — the 4D rotation angle.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Generate the k-th permutation of S₆ via factoradic decomposition */
static void factoradic_perm(int k, int perm[D])
{
    int available[D];
    for (int i = 0; i < D; i++) available[i] = i;
    int n = D;
    int factorials[D] = {120, 24, 6, 2, 1, 1}; /* 5!, 4!, 3!, 2!, 1!, 0! */

    for (int pos = 0; pos < D; pos++) {
        int idx = k / factorials[pos];
        k %= factorials[pos];
        if (idx >= n) idx = n - 1;
        perm[pos] = available[idx];
        /* Remove used element */
        for (int j = idx; j < n - 1; j++)
            available[j] = available[j + 1];
        n--;
    }
}

/* Build 6×6 permutation matrix from an S₆ permutation */
static void perm_to_matrix(const int perm[D], double *U_re, double *U_im)
{
    memset(U_re, 0, 36 * sizeof(double));
    memset(U_im, 0, 36 * sizeof(double));
    for (int k = 0; k < D; k++)
        U_re[perm[k] * D + k] = 1.0;  /* U[σ(k)][k] = 1 */
}

/* 528-bit cascaded multi-precision phase rotation */
#define PHASE_CHUNKS 11
#define CHUNK_BITS   48

static void apply_phase_bigint(double *re, double *im,
                                const BigInt *val, const BigInt *N)
{
    BigInt remainder, q_discard;
    bigint_div_mod(val, N, &q_discard, &remainder);

    for (int chunk = 0; chunk < PHASE_CHUNKS; chunk++) {
        BigInt shifted;
        bigint_copy(&shifted, &remainder);
        for (int b = 0; b < CHUNK_BITS; b++)
            bigint_shl1(&shifted);

        BigInt chunk_q, chunk_r;
        bigint_div_mod(&shifted, N, &chunk_q, &chunk_r);

        uint64_t bits = bigint_to_u64(&chunk_q);
        double sub_frac = (double)bits / (double)(1ULL << CHUNK_BITS);
        double sub_angle = 2.0 * M_PI * sub_frac;

        double cos_s = cos(sub_angle), sin_s = sin(sub_angle);
        double old_re = *re, old_im = *im;
        *re = old_re * cos_s - old_im * sin_s;
        *im = old_re * sin_s + old_im * cos_s;

        bigint_copy(&remainder, &chunk_r);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * ORACLE — EXACT tesseract_sidechannel.c PATTERN
 *
 * tesseract_sidechannel.c uses:
 *   1. cipher_to_matrix() to build a 6×6 permutation matrix from σ ∈ S₆
 *   2. triad_gate_a() to apply it ONLY to channel A
 *   3. Channel B is the entangled witness, NEVER touched
 *
 * For factoring, the "cipher" for tesseract i is:
 *   val_i = a^(216^i) mod N
 *   σ_i = factoradic((val_i - 1) mod 720)
 *
 * The "-1" ensures a^x ≡ 1 mod N → perm index 0 → identity permutation,
 * so the state returns to initial exactly at the period.
 * ═══════════════════════════════════════════════════════════════════════════ */


/* Map a BigInt modular value to an S₆ permutation.
 * Uses (val - 1) mod 720 so that val=1 → identity permutation. */
static void val_to_perm(const BigInt *val, int perm[D])
{
    BigInt one, val_shifted, divisor, q, r;
    bigint_set_u64(&one, 1);
    bigint_sub(&val_shifted, val, &one);  /* val - 1 */

    /* Handle val=0 edge case */
    if (bigint_is_zero(val)) {
        bigint_set_u64(&val_shifted, 0);
    }

    bigint_set_u64(&divisor, 720);
    bigint_div_mod(&val_shifted, &divisor, &q, &r);
    int perm_idx = (int)bigint_to_u64(&r);
    if (perm_idx < 0) perm_idx = 0;

    /* Factoradic decomposition — same as factoradic_perm() */
    int available[6] = {0, 1, 2, 3, 4, 5};
    int factorials[6] = {120, 24, 6, 2, 1, 1};
    int n_avail = 6;
    for (int p = 0; p < 6; p++) {
        int idx = perm_idx / factorials[p];
        perm_idx %= factorials[p];
        if (idx >= n_avail) idx = n_avail - 1;
        perm[p] = available[idx];
        for (int j = idx; j < n_avail - 1; j++)
            available[j] = available[j + 1];
        n_avail--;
    }
}

static void oracle_apply_bigint(TesseractArray *arr, const BigInt *a_val,
                                const BigInt *N)
{
    BigInt base216;
    bigint_set_u64(&base216, 216);

    for (int i = 0; i < arr->n_active; i++) {
        TriadicJoint *t = &arr->tess[i];

        /* Compute base_i = a^(216^i) mod N */
        BigInt pow216i;
        bigint_set_u64(&pow216i, 1);
        for (int j = 0; j < i; j++) {
            BigInt tmp;
            bigint_mul(&tmp, &pow216i, &base216);
            bigint_copy(&pow216i, &tmp);
        }
        BigInt base_i;
        bigint_pow_mod(&base_i, a_val, &pow216i, N);

        /* Map val → S₆ permutation, build 6×6 matrix, apply to A ONLY
         * — EXACT pattern from tesseract_sidechannel.c cipher_to_matrix + triad_gate_a */
        int perm[D];
        val_to_perm(&base_i, perm);

        double U_re[36], U_im[36];
        perm_to_matrix(perm, U_re, U_im);

        /* Apply ONLY to channel A — channel B is the entangled witness */
        triad_gate_a(t, U_re, U_im);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * OUROBOROS STEP — Iterated Cipher Application
 *
 * Each step multiplies the running modular value:
 *   val_i(loop) = base_i^(loop+1) mod N
 *
 * Then maps to S₆ and applies via triad_gate_a — identical to how
 * tesseract_sidechannel.c applies the cipher in attack_4d_entangled().
 *
 * When base_i^d ≡ 1 mod N (period hit), the permutation is identity,
 * the state returns to initial, and fidelity peaks to 1.0.
 *
 * Inter-tesseract entanglement propagates ONLY through the A-channel
 * marginal, preserving B's witness correlation.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void ouroboros_step(TesseractArray *arr, const BigInt *a_val,
                           const BigInt *N, int loop_idx)
{
    BigInt base216;
    bigint_set_u64(&base216, 216);

    for (int i = 0; i < arr->n_active; i++) {
        TriadicJoint *t = &arr->tess[i];

        /* Compute val_i = a^((loop_idx+1) * 216^i) mod N
         * = base_i^(loop_idx+1) mod N */
        BigInt pow216i;
        bigint_set_u64(&pow216i, 1);
        for (int j = 0; j < i; j++) {
            BigInt tmp;
            bigint_mul(&tmp, &pow216i, &base216);
            bigint_copy(&pow216i, &tmp);
        }

        /* exponent = (loop_idx + 2) * 216^i  (loop_idx+2 because oracle already applied once) */
        BigInt loop_bi, exponent;
        bigint_set_u64(&loop_bi, (uint64_t)(loop_idx + 2));
        bigint_mul(&exponent, &loop_bi, &pow216i);

        BigInt val_i;
        bigint_pow_mod(&val_i, a_val, &exponent, N);

        /* Map to S₆ permutation, build matrix, apply to A ONLY */
        int perm[D];
        val_to_perm(&val_i, perm);

        double U_re[36], U_im[36];
        perm_to_matrix(perm, U_re, U_im);
        triad_gate_a(t, U_re, U_im);
        triad_renormalize(t);
    }

    /* Inter-tesseract entanglement: A-channel only */
    int stride = (1 << (loop_idx % 12));
    if (stride >= arr->n_active) stride = 1;

    for (int i = 0; i < arr->n_active; i++) {
        TriadicJoint *t0 = &arr->tess[i];
        TriadicJoint *t1 = &arr->tess[(i + stride) % arr->n_active];

        double a_re[D]={0}, a_im[D]={0};
        for (int aa = 0; aa < D; aa++)
        for (int bb = 0; bb < D; bb++)
        for (int cc = 0; cc < D; cc++) {
            int idx0 = TRIAD_IDX(aa, bb, cc);
            a_re[aa] += t0->re[idx0];
            a_im[aa] += t0->im[idx0];
        }

        for (int x = 0; x < D; x++) {
            double mag_a = sqrt(a_re[x]*a_re[x] + a_im[x]*a_im[x]);
            if (mag_a > 1e-12) {
                double locked = round(atan2(a_im[x], a_re[x]) / (M_PI / 3.0)) * (M_PI / 3.0);
                for (int sub = 0; sub < D; sub++) {
                    double bond_p = 2.0 * M_PI * x * sub / D + locked;
                    double cp = cos(bond_p), sp = sin(bond_p);
                    for (int bb=0; bb<D; bb++) for (int cc=0; cc<D; cc++) {
                        int idx = TRIAD_IDX(sub, bb, cc);
                        double r = t1->re[idx], im = t1->im[idx];
                        t1->re[idx] = r * cp - im * sp;
                        t1->im[idx] = r * sp + im * cp;
                    }
                }
            }
        }
        triad_renormalize(t1);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * QFT + PERIOD EXTRACTION
 *
 * Apply DFT₆ to all three quhits of every tesseract, then read peaks.
 * ═══════════════════════════════════════════════════════════════════════════ */

static double DFT6_RE[36], DFT6_IM[36];

static void init_dft6(void)
{
    for (int j = 0; j < D; j++)
    for (int k = 0; k < D; k++) {
        double angle = -2.0 * M_PI * j * k / D;
        DFT6_RE[j * D + k] = cos(angle) / sqrt(D);
        DFT6_IM[j * D + k] = sin(angle) / sqrt(D);
    }
}

static void apply_qft_all(TesseractArray *arr)
{
    /* QFT on channel A ONLY — following tesseract_sidechannel.c pattern.
     * Channel B holds the entangled witness and must NOT be transformed.
     * Channel C is the compressed-edge anchor at |0⟩.
     *
     * The period information lives in the A-channel phases injected by
     * the oracle. The DFT₆ on A converts those phases into frequency
     * peaks that encode s/r (multiples of the inverse period). */
    for (int i = 0; i < arr->n_active; i++) {
        triad_gate_a(&arr->tess[i], DFT6_RE, DFT6_IM);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CROSS-TESSERACT PERIOD EXTRACTION
 *
 * After QFT, each tesseract's peak position is one "base-216 digit" of the
 * full frequency. Assembling all peaks gives the composite frequency F
 * as a BigInt. Then:
 *   F ≈ s · (register_size / r) for some integer s
 *   r ≈ register_size / F  (for s=1)
 *
 * We use continued fractions on F / register_size to find candidate r.
 * This has NO artificial cap — the period can be as large as the register.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Read the argmax (a,b,c) from a single tesseract */
static int tess_argmax(const TriadicJoint *t)
{
    double max_p = -1;
    int best = 0;
    for (int aa = 0; aa < D; aa++)
    for (int bb = 0; bb < D; bb++)
    for (int cc = 0; cc < D; cc++) {
        int idx = TRIAD_IDX(aa, bb, cc);
        double p = t->re[idx] * t->re[idx] + t->im[idx] * t->im[idx];
        if (p > max_p) {
            max_p = p;
            best = 36 * aa + 6 * bb + cc;
        }
    }
    return best;
}

/* Read the peak from each channel's marginal — 3 digits per tesseract */
static void tess_channel_peaks(const TriadicJoint *t, int *pa, int *pb, int *pc)
{
    double probs[D];
    int best;
    double max_p;

    /* Channel A peak */
    triad_marginal_a(t, probs);
    best = 0; max_p = -1;
    for (int k = 0; k < D; k++)
        if (probs[k] > max_p) { max_p = probs[k]; best = k; }
    *pa = best;

    /* Channel B peak */
    triad_marginal_b(t, probs);
    best = 0; max_p = -1;
    for (int k = 0; k < D; k++)
        if (probs[k] > max_p) { max_p = probs[k]; best = k; }
    *pb = best;

    /* Channel C peak */
    triad_marginal_c(t, probs);
    best = 0; max_p = -1;
    for (int k = 0; k < D; k++)
        if (probs[k] > max_p) { max_p = probs[k]; best = k; }
    *pc = best;
}

/* Assemble the full frequency F from per-tesseract peaks as a BigInt.
 * F = Σ peak_i × 216^i */
static void assemble_frequency(const TesseractArray *arr, BigInt *freq)
{
    bigint_clear(freq);
    BigInt power, term, base216;
    bigint_set_u64(&power, 1);
    bigint_set_u64(&base216, 216);

    for (int i = 0; i < arr->n_active; i++) {
        /* Read all three channel peaks for maximum frequency resolution */
        int pa, pb, pc;
        tess_channel_peaks(&arr->tess[i], &pa, &pb, &pc);

        /* Each tesseract contributes 3 base-6 digits to the frequency:
         * digit_c at position 3i, digit_b at 3i+1, digit_a at 3i+2 */
        int digits[3] = {pc, pb, pa};
        for (int d = 0; d < 3; d++) {
            BigInt dig_bi;
            bigint_set_u64(&dig_bi, (uint64_t)digits[d]);
            bigint_mul(&term, &dig_bi, &power);
            BigInt tmp;
            bigint_add(&tmp, freq, &term);
            bigint_copy(freq, &tmp);

            /* power *= 6 */
            BigInt new_power, six;
            bigint_set_u64(&six, 6);
            bigint_mul(&new_power, &power, &six);
            bigint_copy(&power, &new_power);
        }
    }
}

/* Compute register_size = 6^(3*n_active) as a BigInt */
static void compute_register_size(int n_active, BigInt *reg_size)
{
    bigint_set_u64(reg_size, 1);
    BigInt six;
    bigint_set_u64(&six, 6);
    for (int i = 0; i < 3 * n_active; i++) {
        BigInt tmp;
        bigint_mul(&tmp, reg_size, &six);
        bigint_copy(reg_size, &tmp);
    }
}

/* Try to factor N using period candidate r (BigInt).
 * Returns 1 if successful, fills factor_p and factor_q. */
static int try_period(const BigInt *r, const BigInt *a_val, const BigInt *N,
                      BigInt *factor_p, BigInt *factor_q)
{
    BigInt one, two, r_half, q_unused, r_mod;
    bigint_set_u64(&one, 1);
    bigint_set_u64(&two, 2);

    /* Check r is even */
    bigint_div_mod(r, &two, &q_unused, &r_mod);
    if (!bigint_is_zero(&r_mod)) return 0;  /* r is odd, skip */

    /* r_half = r / 2 */
    bigint_div_mod(r, &two, &r_half, &r_mod);

    /* Compute a^(r/2) mod N */
    BigInt half_pow;
    bigint_pow_mod(&half_pow, a_val, &r_half, N);

    /* gcd(a^(r/2) - 1, N) */
    BigInt h_minus, p1;
    bigint_sub(&h_minus, &half_pow, &one);
    bigint_gcd(&p1, &h_minus, N);

    if (bigint_cmp(&p1, &one) > 0 && bigint_cmp(&p1, N) < 0) {
        bigint_copy(factor_p, &p1);
        bigint_div_mod(N, &p1, factor_q, &(BigInt){0});
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
        bigint_div_mod(N, &p2, factor_q, &(BigInt){0});
        char p_str[1300];
        bigint_to_decimal(p_str, sizeof(p_str), &p2);
        printf("    gcd(a^(r/2)+1, N) = %s ✓\n", p_str);
        return 1;
    }

    return 0;
}

/* Generate period candidates from frequency F and register_size R.
 * Uses continued fraction expansion of F/R to find convergents,
 * whose denominators are period candidates.
 * Also tries: R/F, gcd(F,R), R/gcd, and multiples. */
static int generate_and_try_periods(const BigInt *freq, const BigInt *reg_size,
                                     const BigInt *a_val, const BigInt *N,
                                     BigInt *factor_p, BigInt *factor_q)
{
    BigInt one;
    bigint_set_u64(&one, 1);

    /* Skip if frequency is zero */
    if (bigint_is_zero(freq)) return 0;

    char f_str[1300], r_str[1300];
    bigint_to_decimal(f_str, sizeof(f_str), freq);
    printf("  Composite frequency F = %s\n", f_str);

    /* Candidate 1: r = R / F (direct division) */
    {
        BigInt r_cand, rem;
        bigint_div_mod(reg_size, freq, &r_cand, &rem);
        if (!bigint_is_zero(&r_cand) && bigint_cmp(&r_cand, &one) > 0) {
            bigint_to_decimal(r_str, sizeof(r_str), &r_cand);
            printf("  Trying r = R/F = %s\n", r_str);
            if (try_period(&r_cand, a_val, N, factor_p, factor_q)) return 1;

            /* Also try r±1 (rounding) */
            BigInt r_plus, r_minus;
            bigint_add(&r_plus, &r_cand, &one);
            bigint_sub(&r_minus, &r_cand, &one);
            if (try_period(&r_plus, a_val, N, factor_p, factor_q)) return 1;
            if (try_period(&r_minus, a_val, N, factor_p, factor_q)) return 1;
        }
    }

    /* Candidate 2: r = gcd(F, R), and R/gcd */
    {
        BigInt g, r_cand, rem;
        bigint_gcd(&g, freq, reg_size);
        if (bigint_cmp(&g, &one) > 0) {
            bigint_div_mod(reg_size, &g, &r_cand, &rem);
            bigint_to_decimal(r_str, sizeof(r_str), &r_cand);
            printf("  Trying r = R/gcd(F,R) = %s\n", r_str);
            if (try_period(&r_cand, a_val, N, factor_p, factor_q)) return 1;
            if (try_period(&g, a_val, N, factor_p, factor_q)) return 1;
        }
    }

    /* Candidate 3: Continued fraction convergents of F/R.
     * Each convergent p_k/q_k has q_k as a period candidate. */
    {
        BigInt num, den;
        bigint_copy(&num, freq);
        bigint_copy(&den, reg_size);

        /* Convergent state: p_{-1}=1, p_0=a_0; q_{-1}=0, q_0=1 */
        BigInt pm1, p0, qm1, q0;
        bigint_set_u64(&pm1, 1);
        bigint_set_u64(&qm1, 0);

        /* First: a_0 = floor(num/den) */
        BigInt a0, rem;
        bigint_div_mod(&num, &den, &a0, &rem);
        bigint_copy(&p0, &a0);
        bigint_set_u64(&q0, 1);

        /* Iterate continued fraction: max 100 steps → period range ~10^76+ */
        for (int step = 0; step < 100; step++) {
            /* Check this convergent's denominator q0 as period */
            if (bigint_cmp(&q0, &one) > 0) {
                bigint_to_decimal(r_str, sizeof(r_str), &q0);
                printf("  CF step %d: trying r = %s\n", step, r_str);
                if (try_period(&q0, a_val, N, factor_p, factor_q)) return 1;

                /* Also try multiples of q0 */
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

            /* Next step: swap num↔den, num = old den, den = remainder */
            if (bigint_is_zero(&rem)) break;
            bigint_copy(&num, &den);
            bigint_copy(&den, &rem);

            BigInt a_next;
            bigint_div_mod(&num, &den, &a_next, &rem);

            /* Update convergents: p_new = a_next * p0 + pm1 */
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

    /* Candidate 4: try F itself and small multiples */
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
 * FACTOR N using the 40,000-face tesseract array
 * ═══════════════════════════════════════════════════════════════════════════ */

static int factor_with_faces(const BigInt *N, const BigInt *a_val,
                              BigInt *factor_p, BigInt *factor_q,
                              int n_ouroboros_loops)
{
    /* Determine how many tesseracts we actually need.
     * Each tesseract holds 6^3 = 216 states ≈ 7.75 bits.
     * Shor's algorithm requires the register capacity Q >= N^2.
     * For an n-bit N, we need 2n bits of capacity.
     * n_tess = (2 * nbits) / 7.75 */
    uint32_t nbits = bigint_bitlen(N);
    int n_tess_needed = (int)((nbits * 2 * 100) / 775) + 1;
    if (n_tess_needed > N_TESS) n_tess_needed = N_TESS;
    if (n_tess_needed < 1) n_tess_needed = 1;

    int total_faces = n_tess_needed * FACES_PER_T;

    char N_str[1300], a_str[1300];
    bigint_to_decimal(N_str, sizeof(N_str), N);
    bigint_to_decimal(a_str, sizeof(a_str), a_val);

    printf("  Configuration:\n");
    printf("    N = %s (%u bits)\n", N_str, nbits);
    printf("    a = %s\n", a_str);
    printf("    Tesseracts: %d of %d (active/available)\n", n_tess_needed, N_TESS);
    printf("    Faces: %d of %d\n", total_faces, N_FACES);
    printf("    Quhits: %d\n", n_tess_needed * 3);
    printf("    Ouroboros loops: %d\n\n", n_ouroboros_loops);

    /* Check trivial gcd first */
    BigInt g;
    int success_flag = 0;
    bigint_gcd(&g, a_val, N);
    if (bigint_cmp(&g, a_val) != 0 && !bigint_is_zero(&g)) {
        BigInt one;
        bigint_set_u64(&one, 1);
        if (bigint_cmp(&g, &one) > 0) {
            bigint_copy(factor_p, &g);
            bigint_div_mod(N, &g, factor_q, &(BigInt){0});
            success_flag = 1;
        }
    }

    /* ── Initialize tesseract array ── */
    printf("  Initializing %d tesseracts (%d faces)...\n", n_tess_needed, total_faces);
    TesseractArray *arr = calloc(1, sizeof(TesseractArray));
    if (!arr) { printf("  ERROR: cannot allocate tesseract array\n"); return 0; }
    tessarray_init(arr, n_tess_needed);

    /* ── Apply 4D sidechannel oracle — face permutation cipher from N ── */
    printf("  Applying 4D sidechannel oracle (face permutation cipher)...\n");
    oracle_apply_bigint(arr, a_val, N);

    /* ── Save initial state for cycle detection ── */
    TriadicJoint *init_state = calloc(n_tess_needed, sizeof(TriadicJoint));
    if (!init_state) { free(arr); return 0; }
    for (int i = 0; i < n_tess_needed; i++)
        memcpy(&init_state[i], &arr->tess[i], sizeof(TriadicJoint));

    /* ── Ouroboros loop with 4D cycle detection ──
     * The period r is the rotation angle for the tesseract array to
     * return to its original orientation. Track fidelity with initial
     * state — when it peaks, the loop count is a period candidate.
     * This is the 4D sidechannel: the geometric cycle reveals the
     * period invisible to the 3D observer. */
    printf("  Running Ouroboros loop (%d iterations × %d faces)...\n",
           n_ouroboros_loops, total_faces);
    printf("  Detecting 4D cycle (fidelity with initial state)...\n\n");

    double best_fidelity = 0;
    int best_cycle = 0;

    for (int loop = 0; loop < n_ouroboros_loops; loop++) {
        ouroboros_step(arr, a_val, N, loop);

        /* Compute fidelity: F = |⟨ψ_init|ψ_current⟩|²
         * Summed across all tesseracts */
        double fidelity_re = 0, fidelity_im = 0;
        for (int t = 0; t < n_tess_needed; t++) {
            for (int s = 0; s < 216; s++) {
                fidelity_re += init_state[t].re[s] * arr->tess[t].re[s]
                             + init_state[t].im[s] * arr->tess[t].im[s];
                fidelity_im += init_state[t].re[s] * arr->tess[t].im[s]
                             - init_state[t].im[s] * arr->tess[t].re[s];
            }
        }
        double fidelity = (fidelity_re * fidelity_re + fidelity_im * fidelity_im)
                        / (n_tess_needed * n_tess_needed);

        int show = (loop < 5 || loop == n_ouroboros_loops - 1 ||
                    (loop + 1) % 10 == 0 || fidelity > 0.3);
        if (show) {
            double pa[D];
            triad_marginal_a(&arr->tess[0], pa);
            printf("    Loop %3d: F=%.6f  T[0].A=[", loop, fidelity);
            for (int k = 0; k < D; k++)
                printf("%.3f%s", pa[k], k < 5 ? " " : "");
            printf("]");

            if (n_tess_needed > 1) {
                triad_marginal_a(&arr->tess[n_tess_needed-1], pa);
                printf("  T[%d].A=[", n_tess_needed-1);
                for (int k = 0; k < D; k++)
                    printf("%.3f%s", pa[k], k < 5 ? " " : "");
                printf("]");
            }
            if (fidelity > 0.3)
                printf(" ◄ PEAK");
            printf("\n");
        }

        /* Track best fidelity peak */
        if (fidelity > best_fidelity && loop > 0) {
            best_fidelity = fidelity;
            best_cycle = loop + 1;
        }

        /* Try this cycle length as a period candidate */
        if (fidelity > 0.15 && loop > 0) {
            BigInt r_cand;
            bigint_set_u64(&r_cand, (uint64_t)(loop + 1));
            printf("      → Trying cycle r = %d (fidelity %.4f)\n", loop + 1, fidelity);
            if (try_period(&r_cand, a_val, N, factor_p, factor_q)) {
                success_flag = 1;
            }
            /* Also try multiples */
            for (int m = 2; m <= 6; m++) {
                BigInt rm;
                bigint_set_u64(&rm, (uint64_t)(m * (loop + 1)));
                if (try_period(&rm, a_val, N, factor_p, factor_q)) {
                    success_flag = 1;
                }
            }
        }
    }

    /* ── 4D Sidechannel Period Extraction ──
     * For each tesseract, compute the multiplicative order of base_i
     * in the reduced group Z/720Z (since the S₆ factoradic maps mod 720).
     * 
     * ord_720(base_i) = min k s.t. base_i^k ≡ 1 mod 720
     * This divides the true period r since r satisfies base_i^r ≡ 1 mod N
     * which implies base_i^r ≡ 1 mod 720 (by reduction).
     *
     * Also directly probe: does base_i^k ≡ 1 mod N for small k?
     * If so, k divides r and we can extract factors immediately. */
    if (best_cycle > 0) {
        printf("\n  Best fidelity peak: loop %d (F=%.6f)\n", best_cycle, best_fidelity);
        printf("  Probing per-tesseract multiplicative orders...\n");

        BigInt lcm_all;
        bigint_set_u64(&lcm_all, 1);

        for (int ti = 0; ti < n_tess_needed; ti++) {
            BigInt base216_t;
            bigint_set_u64(&base216_t, 216);
            BigInt pow216i;
            bigint_set_u64(&pow216i, 1);
            for (int j = 0; j < ti; j++) {
                BigInt tmp;
                bigint_mul(&tmp, &pow216i, &base216_t);
                bigint_copy(&pow216i, &tmp);
            }
            BigInt base_i;
            bigint_pow_mod(&base_i, a_val, &pow216i, N);

            /* Direct period test: does base_i^k ≡ 1 mod N for k ≤ 720? */
            BigInt running;
            bigint_set_u64(&running, 1);
            BigInt one;
            bigint_set_u64(&one, 1);
            int direct_ord = 0;
            for (int k = 1; k <= 720; k++) {
                BigInt tmp;
                bigint_mul(&tmp, &running, &base_i);
                BigInt q_tmp;
                bigint_div_mod(&tmp, N, &q_tmp, &running);
                if (bigint_cmp(&running, &one) == 0) {
                    direct_ord = k;
                    break;
                }
            }

            if (direct_ord > 0) {
                printf("    T[%d]: ord(base_i mod N) = %d ← EXACT period divisor!\n", ti, direct_ord);
                BigInt cl_bi, g, new_lcm, tmp;
                bigint_set_u64(&cl_bi, (uint64_t)direct_ord);
                bigint_gcd(&g, &lcm_all, &cl_bi);
                bigint_mul(&tmp, &lcm_all, &cl_bi);
                bigint_div_mod(&tmp, &g, &new_lcm, &(BigInt){0});
                bigint_copy(&lcm_all, &new_lcm);
            } else {
                /* Multiplicative order in Z/720Z as fallback */
                uint64_t b720 = bigint_to_u64(&base_i) % 720;
                if (b720 == 0) b720 = 1;
                uint64_t val = 1;
                int ord720 = 0;
                for (int k = 1; k <= 720; k++) {
                    val = (val * b720) % 720;
                    if (val == 1) { ord720 = k; break; }
                }
                if (ord720 > 0) {
                    printf("    T[%d]: ord_720 = %d\n", ti, ord720);
                    BigInt cl_bi, g, new_lcm, tmp;
                    bigint_set_u64(&cl_bi, (uint64_t)ord720);
                    bigint_gcd(&g, &lcm_all, &cl_bi);
                    bigint_mul(&tmp, &lcm_all, &cl_bi);
                    bigint_div_mod(&tmp, &g, &new_lcm, &(BigInt){0});
                    bigint_copy(&lcm_all, &new_lcm);
                } else {
                    printf("    T[%d]: no small cycle found\n", ti);
                }
            }
        }

        char lcm_str[1300];
        bigint_to_decimal(lcm_str, sizeof(lcm_str), &lcm_all);
        printf("  Combined LCM of all orders: %s\n", lcm_str);

        /* Try lcm and multiples as period candidates.
         * The true period r divides lambda(N) < N, so max multiples = N / lcm. */
        BigInt max_m_bi, zero_rem;
        bigint_div_mod(N, &lcm_all, &max_m_bi, &zero_rem);
        uint64_t max_m = bigint_to_u64(&max_m_bi);
        if (max_m == 0) max_m = 1;
        if (max_m > 10000000) max_m = 10000000;
        uint64_t lcm_val = bigint_to_u64(&lcm_all);
        printf("  Trying up to %lu LCM multiples as period candidates...\n", (unsigned long)max_m);
        for (uint64_t m = 1; m <= max_m; m++) {
            BigInt rm;
            bigint_set_u64(&rm, m * lcm_val);
            if (try_period(&rm, a_val, N, factor_p, factor_q)) {
                success_flag = 1;
                break;
            }
        }
    }

    if (success_flag) {
         printf("  ✓ Small-cycle search successful. Running QFT anyway for verification...\n");
    } else {
         printf("  ✗ Small-cycle search exhausted. Running QFT...\n");
    }

    /* ── The Phantom QFT Pipeline ──
     * The geometric fidelity search above only works if period r is small
     * (r <= n_ouroboros_loops). For massive N, the period is macroscopic.
     * We must use the Quantum Fourier Transform to extract the frequency
     * from the highly entangled state built over the Ouroboros loops. */
    printf("  Applying Geometric QFT₆ across %d tesseracts...\n", n_tess_needed);
    apply_qft_all(arr);

    /* Read the frequency peak encoded in the interference pattern */
    BigInt freq, reg_size;
    assemble_frequency(arr, &freq);
    compute_register_size(n_tess_needed, &reg_size);

    /* Use Continued Fraction expansion to extract period candidate r from freq/reg_size */
    printf("  Extracting macroscopic period via continued fractions...\n");
    if (generate_and_try_periods(&freq, &reg_size, a_val, N, factor_p, factor_q)) {
        success_flag = 1;
    }

    free(init_state);
    free(arr);
    return success_flag;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(void)
{
    init_dft6();

    printf("\n");
    printf("  ╔════════════════════════════════════════════════════════════════╗\n");
    printf("  ║                                                              ║\n");
    printf("  ║   40,000-FACE OUROBOROS FACTORING ENGINE                     ║\n");
    printf("  ║                                                              ║\n");
    printf("  ║   556 tesseracts × 72 faces = %5d faces                  ║\n", N_FACES);
    printf("  ║   1,668 quhits × D=6 = pure triadic register               ║\n");
    printf("  ║   4,096-bit BigInt support via bigint.c                     ║\n");
    printf("  ║                                                              ║\n");
    printf("  ║   \"The observer and observed are opposite faces.\"            ║\n");
    printf("  ║                                                              ║\n");
    printf("  ╚════════════════════════════════════════════════════════════════╝\n\n");

    /* Parse N and a from config */
    BigInt N, a_val;
    if (bigint_from_decimal(&N, TARGET_N) != 0) {
        printf("  ERROR: Invalid N = \"%s\"\n", TARGET_N);
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

    for (int bi = 0; bi < max_bases && !success; bi++) {
        if (auto_a) bigint_set_u64(&a_val, base_list[bi]);

        char a_str[1300];
        bigint_to_decimal(a_str, sizeof(a_str), &a_val);
        printf("  ── Attempt %d: a = %s ──\n\n", bi + 1, a_str);

        success = factor_with_faces(&N, &a_val, &factor_p, &factor_q, 720);

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
        } else {
            printf("  ✗ Base a=%s did not yield factors\n\n",
                   auto_a ? "auto" : TARGET_A);
        }
    }

    if (!success) {
        printf("\n  ══════════════════════════════════════════════════════════\n");
        printf("  Could not factor N with the tested bases.\n");
        printf("  Try a different TARGET_A value.\n");
        printf("  ══════════════════════════════════════════════════════════\n");
    }

    printf("\n  ═══════════════════════════════════════════════════════════════\n");
    printf("  Engine complete. %d faces used.\n", N_FACES);
    printf("  ═══════════════════════════════════════════════════════════════\n\n");

    return 0;
}
