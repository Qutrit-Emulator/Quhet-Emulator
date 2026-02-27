/*
 * quhit_3ch_register.h — Sparse Channel Qubit Register
 *
 * Each entry: (c_vec, b, amplitude)
 *   c_vec = per-quhit channel index (N-digit base-3, stored in uint64_t)
 *   b     = per-quhit parity (N-bit)
 *   Together: k_i = c_i*2 + b_i gives the D=6 value for quhit i
 *
 * Both c_vec and b are SPARSE — most entries are zero.
 * Missing c_vec entries are DERIVABLE via DFT₃ / twiddle.
 *
 * DFT₆ on quhit i decomposes as:
 *   Stage 1: Hadamard on bit i of b
 *   Stage 2: Twiddle ω₆^(c_i · b_i) on entries where b_i=1
 *   Stage 3: DFT₃ on c_i across the 3 possible values, per fixed (rest, b)
 *
 * CZ on quhits i,j: phase ω₆^(k_i·k_j) where k_i = c_i*2 + b_i
 */

#ifndef QUHIT_3CH_REGISTER_H
#define QUHIT_3CH_REGISTER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════ Constants ═══════════════ */

static const double TCR_H2 = 0.70710678118654752440;
static const double TCR_W6_RE[6] = {1.0, 0.5, -0.5, -1.0, -0.5, 0.5};
static const double TCR_W6_IM[6] = {0.0, 0.86602540378443864676,
    0.86602540378443864676, 0.0, -0.86602540378443864676,
    -0.86602540378443864676};
static const double TCR_N3 = 0.57735026918962576451;   /* 1/√3 */
static const double TCR_W3R = -0.5;
static const double TCR_W3I = 0.86602540378443864676;

/* ═══════════════ Sparse Entry ═══════════════ */

typedef struct {
    uint64_t cvec;   /* per-quhit channel index, base-3 packed */
    uint64_t b;      /* per-quhit parity, N-bit */
    double   re, im;
} TCR_Entry;

/* ═══════════════ Register ═══════════════ */

typedef struct {
    TCR_Entry *entries;
    int        count;
    int        capacity;
    int        N;    /* number of quhits */
} ThreeCh_Register;

/* ═══════════════ Base-3 digit ops ═══════════════ */

static inline int tcr_get_c(uint64_t cvec, int qi) {
    uint64_t v = cvec;
    for (int i = 0; i < qi; i++) v /= 3;
    return v % 3;
}

static inline uint64_t tcr_set_c(uint64_t cvec, int qi, int val) {
    uint64_t pow3 = 1;
    for (int i = 0; i < qi; i++) pow3 *= 3;
    int old = (cvec / pow3) % 3;
    return cvec + (int64_t)(val - old) * pow3;
}

/* ═══════════════ Register ops ═══════════════ */

static inline ThreeCh_Register *tcr_alloc(int N) {
    ThreeCh_Register *r = (ThreeCh_Register *)calloc(1, sizeof(*r));
    r->N = N;
    r->capacity = 256;
    r->entries = (TCR_Entry *)calloc(r->capacity, sizeof(TCR_Entry));
    r->count = 0;
    return r;
}

static inline void tcr_free(ThreeCh_Register *r) {
    free(r->entries); free(r);
}

static inline void tcr_grow(ThreeCh_Register *r) {
    r->capacity *= 2;
    r->entries = (TCR_Entry *)realloc(r->entries,
                                      r->capacity * sizeof(TCR_Entry));
}

/* Find entry by (cvec, b), return index or -1 */
static inline int tcr_find(const ThreeCh_Register *r,
                           uint64_t cvec, uint64_t b) {
    for (int i = 0; i < r->count; i++)
        if (r->entries[i].cvec == cvec && r->entries[i].b == b)
            return i;
    return -1;
}

/* Add amplitude to (cvec, b) */
static inline void tcr_add(ThreeCh_Register *r,
                           uint64_t cvec, uint64_t b,
                           double re, double im) {
    int idx = tcr_find(r, cvec, b);
    if (idx >= 0) {
        r->entries[idx].re += re;
        r->entries[idx].im += im;
        return;
    }
    if (r->count >= r->capacity) tcr_grow(r);
    r->entries[r->count].cvec = cvec;
    r->entries[r->count].b = b;
    r->entries[r->count].re = re;
    r->entries[r->count].im = im;
    r->count++;
}

/* Set amplitude at (cvec, b) */
static inline void tcr_set(ThreeCh_Register *r,
                           uint64_t cvec, uint64_t b,
                           double re, double im) {
    int idx = tcr_find(r, cvec, b);
    if (idx >= 0) {
        r->entries[idx].re = re;
        r->entries[idx].im = im;
        return;
    }
    if (r->count >= r->capacity) tcr_grow(r);
    r->entries[r->count].cvec = cvec;
    r->entries[r->count].b = b;
    r->entries[r->count].re = re;
    r->entries[r->count].im = im;
    r->count++;
}

/* Trim near-zero entries */
static inline void tcr_trim(ThreeCh_Register *r, double eps) {
    int j = 0;
    for (int i = 0; i < r->count; i++) {
        double mag2 = r->entries[i].re * r->entries[i].re +
                      r->entries[i].im * r->entries[i].im;
        if (mag2 > eps * eps)
            r->entries[j++] = r->entries[i];
    }
    r->count = j;
}

/* ═══════════════ Init ═══════════════ */

/* All quhits to |0⟩: k=0 → (c=0, p=0) */
static inline void tcr_init_all_zero(ThreeCh_Register *r) {
    tcr_set(r, 0ULL, 0ULL, 1.0, 0.0);
}

/* ═══════════════ DFT₆ on quhit qi ═══════════════ */

static inline void tcr_apply_dft6(ThreeCh_Register *r, int qi) {
    uint64_t bmask = 1ULL << qi;

    /* Stage 1: Hadamard on bit qi of b */
    int old_count = r->count;
    TCR_Entry *old = (TCR_Entry *)malloc(old_count * sizeof(TCR_Entry));
    memcpy(old, r->entries, old_count * sizeof(TCR_Entry));
    r->count = 0;

    for (int i = 0; i < old_count; i++) {
        uint64_t cvec = old[i].cvec;
        uint64_t b = old[i].b;
        double re = old[i].re, im = old[i].im;
        int bit = (b >> qi) & 1;

        uint64_t b0 = b & ~bmask;
        uint64_t b1 = b | bmask;

        if (bit == 0) {
            /* |0⟩ → (|0⟩+|1⟩)/√2 */
            tcr_add(r, cvec, b0, TCR_H2 * re, TCR_H2 * im);
            tcr_add(r, cvec, b1, TCR_H2 * re, TCR_H2 * im);
        } else {
            /* |1⟩ → (|0⟩-|1⟩)/√2 */
            tcr_add(r, cvec, b0,  TCR_H2 * re,  TCR_H2 * im);
            tcr_add(r, cvec, b1, -TCR_H2 * re, -TCR_H2 * im);
        }
    }
    free(old);

    /* Stage 2: Twiddle ω₆^(c_qi · b_qi) on entries where b_qi=1 */
    for (int i = 0; i < r->count; i++) {
        if (!((r->entries[i].b >> qi) & 1)) continue;
        int c = tcr_get_c(r->entries[i].cvec, qi);
        if (c == 0) continue;
        double re = r->entries[i].re, im = r->entries[i].im;
        r->entries[i].re = TCR_W6_RE[c] * re - TCR_W6_IM[c] * im;
        r->entries[i].im = TCR_W6_RE[c] * im + TCR_W6_IM[c] * re;
    }

    /* Stage 3: DFT₃ on c_qi — for each unique (cvec_rest, b),
     * gather the 3 entries with c_qi ∈ {0,1,2}, apply DFT₃, scatter */

    /* Collect unique (cvec_with_c0, b) keys */
    int n = r->count;
    /* Work on a copy */
    old_count = r->count;
    old = (TCR_Entry *)malloc(old_count * sizeof(TCR_Entry));
    memcpy(old, r->entries, old_count * sizeof(TCR_Entry));

    /* Mark processed entries */
    uint8_t *done = (uint8_t *)calloc(old_count, 1);

    r->count = 0;

    for (int i = 0; i < old_count; i++) {
        if (done[i]) continue;
        uint64_t cvec0 = tcr_set_c(old[i].cvec, qi, 0);
        uint64_t b = old[i].b;

        /* Gather amplitudes for c_qi = 0, 1, 2 */
        double a_re[3] = {0}, a_im[3] = {0};
        for (int j = i; j < old_count; j++) {
            if (done[j]) continue;
            if (old[j].b != b) continue;
            if (tcr_set_c(old[j].cvec, qi, 0) != cvec0) continue;
            int c = tcr_get_c(old[j].cvec, qi);
            a_re[c] = old[j].re;
            a_im[c] = old[j].im;
            done[j] = 1;
        }

        /* DFT₃ */
        double o_re[3], o_im[3];
        /* j=0: (a+b+c)/√3 */
        o_re[0] = TCR_N3 * (a_re[0] + a_re[1] + a_re[2]);
        o_im[0] = TCR_N3 * (a_im[0] + a_im[1] + a_im[2]);
        /* j=1: (a + ω₃·b + ω₃²·c)/√3 */
        double wb_r = TCR_W3R*a_re[1] - TCR_W3I*a_im[1];
        double wb_i = TCR_W3R*a_im[1] + TCR_W3I*a_re[1];
        double wc_r = TCR_W3R*a_re[2] + TCR_W3I*a_im[2];
        double wc_i = TCR_W3R*a_im[2] - TCR_W3I*a_re[2];
        o_re[1] = TCR_N3 * (a_re[0] + wb_r + wc_r);
        o_im[1] = TCR_N3 * (a_im[0] + wb_i + wc_i);
        /* j=2: (a + ω₃²·b + ω₃·c)/√3 */
        double w2b_r = TCR_W3R*a_re[1] + TCR_W3I*a_im[1];
        double w2b_i = TCR_W3R*a_im[1] - TCR_W3I*a_re[1];
        double w2c_r = TCR_W3R*a_re[2] - TCR_W3I*a_im[2];
        double w2c_i = TCR_W3R*a_im[2] + TCR_W3I*a_re[2];
        o_re[2] = TCR_N3 * (a_re[0] + w2b_r + w2c_r);
        o_im[2] = TCR_N3 * (a_im[0] + w2b_i + w2c_i);

        /* Scatter */
        for (int c = 0; c < 3; c++) {
            double mag2 = o_re[c]*o_re[c] + o_im[c]*o_im[c];
            if (mag2 > 1e-30) {
                uint64_t cv = tcr_set_c(cvec0, qi, c);
                tcr_add(r, cv, b, o_re[c], o_im[c]);
            }
        }
    }

    free(old);
    free(done);
    tcr_trim(r, 1e-15);
}

/* ═══════════════ CZ Gate ═══════════════ */

static inline void tcr_apply_cz(ThreeCh_Register *r, int qi, int qj) {
    uint64_t bmask_i = 1ULL << qi;
    uint64_t bmask_j = 1ULL << qj;

    for (int e = 0; e < r->count; e++) {
        int c_i = tcr_get_c(r->entries[e].cvec, qi);
        int c_j = tcr_get_c(r->entries[e].cvec, qj);
        int p_i = (r->entries[e].b & bmask_i) ? 1 : 0;
        int p_j = (r->entries[e].b & bmask_j) ? 1 : 0;
        int k_i = 3 * p_i + c_i;   /* physical k = 3p + s */
        int k_j = 3 * p_j + c_j;
        int phase_idx = (k_i * k_j) % 6;
        if (phase_idx == 0) continue;
        double re = r->entries[e].re, im = r->entries[e].im;
        r->entries[e].re = TCR_W6_RE[phase_idx] * re - TCR_W6_IM[phase_idx] * im;
        r->entries[e].im = TCR_W6_RE[phase_idx] * im + TCR_W6_IM[phase_idx] * re;
    }
}

/* ═══════════════ Flat index ═══════════════ */

/* Convert (cvec, b) → flat 6^N index */
static inline uint64_t tcr_flat_index(int N, uint64_t cvec, uint64_t b) {
    uint64_t flat = 0, pow6 = 1;
    uint64_t cv = cvec;
    for (int i = 0; i < N; i++) {
        int c = cv % 3; cv /= 3;
        int p = (b >> i) & 1;
        flat += (c * 2 + p) * pow6;
        pow6 *= 6;
    }
    return flat;
}

/* ═══════════════ Statistics ═══════════════ */

static inline double tcr_total_prob(const ThreeCh_Register *r) {
    double s = 0;
    for (int i = 0; i < r->count; i++)
        s += r->entries[i].re * r->entries[i].re +
             r->entries[i].im * r->entries[i].im;
    return s;
}

static inline int tcr_total_nnz(const ThreeCh_Register *r) {
    return r->count;
}

static inline void tcr_print(const ThreeCh_Register *r, const char *label) {
    printf("  %s: N=%d, %d entries, prob=%.6f, memory=%d B\n",
           label, r->N, r->count, tcr_total_prob(r), r->count * 32);
    printf("    (vs 6^%d = %.0f full entries)\n", r->N, pow(6.0, r->N));
}

/* ═══════════════ Born probabilities ═══════════════ */

static inline void tcr_born_probs(const ThreeCh_Register *r,
                                  int qi, double probs[6]) {
    memset(probs, 0, 6 * sizeof(double));
    uint64_t bmask = 1ULL << qi;
    for (int e = 0; e < r->count; e++) {
        int c = tcr_get_c(r->entries[e].cvec, qi);
        int p = (r->entries[e].b & bmask) ? 1 : 0;
        int k = 3 * p + c;
        double mag2 = r->entries[e].re * r->entries[e].re +
                      r->entries[e].im * r->entries[e].im;
        probs[k] += mag2;
    }
}

/* Two-quhit correlation */
static inline double tcr_correlation(const ThreeCh_Register *r,
                                     int qi, int qj) {
    double corr = 0;
    uint64_t mi = 1ULL << qi, mj = 1ULL << qj;
    for (int e = 0; e < r->count; e++) {
        int c_i = tcr_get_c(r->entries[e].cvec, qi);
        int p_i = (r->entries[e].b & mi) ? 1 : 0;
        int k_i = 3 * p_i + c_i;
        int c_j = tcr_get_c(r->entries[e].cvec, qj);
        int p_j = (r->entries[e].b & mj) ? 1 : 0;
        int k_j = 3 * p_j + c_j;
        double mag2 = r->entries[e].re * r->entries[e].re +
                      r->entries[e].im * r->entries[e].im;
        corr += (double)(k_i * k_j) * mag2;
    }
    return corr;
}

#endif /* QUHIT_3CH_REGISTER_H */
