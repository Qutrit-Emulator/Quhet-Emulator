/*
 * quhit_3ch_register.h — Sparse (cvec, b) Register with DFT⁴=I CZ
 *
 * Full factored DFT₆ = Hadamard + Twiddle + DFT₃
 * CZ = diagonal in computational basis
 *
 * Key property: DFT₆⁴ = I, so DFT₆⁻¹ = DFT₆³
 * CZ in DFT basis: apply DFT₆⁻¹ on both quhits, CZ diagonal, DFT₆ back.
 * But since CZ is already diagonal in the (cvec, b) basis (k = 3p + s),
 * CZ just applies ω₆^{k_i·k_j} per entry. The DFT₆ handles the mixing.
 *
 * Storage: sparse (cvec, b, amp) entries.
 * Both cvec (base-3) and b (binary) dimensions are sparse.
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

static const double TCR_H2 = 0.70710678118654752440;
static const double TCR_W6_RE[6] = {1.0, 0.5, -0.5, -1.0, -0.5, 0.5};
static const double TCR_W6_IM[6] = {0.0, 0.86602540378443864676,
    0.86602540378443864676, 0.0, -0.86602540378443864676,
    -0.86602540378443864676};
static const double TCR_N3 = 0.57735026918962576451;
static const double TCR_W3R = -0.5;
static const double TCR_W3I = 0.86602540378443864676;

typedef struct {
    uint64_t cvec;
    uint64_t b;
    double   re, im;
} TCR_Entry;

typedef struct {
    TCR_Entry *entries;
    int        count;
    int        capacity;
    int        N;
} ThreeCh_Register;

/* ═══════════════ Base-3 ops ═══════════════ */

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

static inline int tcr_find(const ThreeCh_Register *r,
                           uint64_t cvec, uint64_t b) {
    for (int i = 0; i < r->count; i++)
        if (r->entries[i].cvec == cvec && r->entries[i].b == b)
            return i;
    return -1;
}

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

static inline void tcr_trim(ThreeCh_Register *r, double eps) {
    int j = 0;
    for (int i = 0; i < r->count; i++) {
        double m2 = r->entries[i].re * r->entries[i].re +
                    r->entries[i].im * r->entries[i].im;
        if (m2 > eps * eps) r->entries[j++] = r->entries[i];
    }
    r->count = j;
}

static inline void tcr_init_all_zero(ThreeCh_Register *r) {
    r->count = 0;
    tcr_add(r, 0ULL, 0ULL, 1.0, 0.0);
}

/* ═══════════════ Full Factored DFT₆ on quhit qi ═══════════════
 *
 * Stage 1: Hadamard on bit qi of b
 * Stage 2: Twiddle ω₆^(c_i · b_i) for entries with b_i=1
 * Stage 3: DFT₃ on c_i
 */
static inline void tcr_apply_dft6(ThreeCh_Register *r, int qi) {
    uint64_t mask = 1ULL << qi;

    /* Stage 1: Hadamard on b_i */
    int old_count = r->count;
    TCR_Entry *old = (TCR_Entry *)malloc(old_count * sizeof(TCR_Entry));
    memcpy(old, r->entries, old_count * sizeof(TCR_Entry));
    r->count = 0;

    for (int i = 0; i < old_count; i++) {
        uint64_t cv = old[i].cvec, b = old[i].b;
        double re = old[i].re, im = old[i].im;
        int bit = (b >> qi) & 1;
        uint64_t b0 = b & ~mask, b1 = b | mask;

        if (bit == 0) {
            tcr_add(r, cv, b0, TCR_H2 * re, TCR_H2 * im);
            tcr_add(r, cv, b1, TCR_H2 * re, TCR_H2 * im);
        } else {
            tcr_add(r, cv, b0,  TCR_H2 * re,  TCR_H2 * im);
            tcr_add(r, cv, b1, -TCR_H2 * re, -TCR_H2 * im);
        }
    }
    free(old);

    /* Stage 2: Twiddle — multiply channel c, b_i=1 entries by ω₆^c */
    for (int i = 0; i < r->count; i++) {
        if (!((r->entries[i].b >> qi) & 1)) continue;
        int c = tcr_get_c(r->entries[i].cvec, qi);
        if (c == 0) continue;
        double re = r->entries[i].re, im = r->entries[i].im;
        r->entries[i].re = TCR_W6_RE[c] * re - TCR_W6_IM[c] * im;
        r->entries[i].im = TCR_W6_RE[c] * im + TCR_W6_IM[c] * re;
    }

    /* Stage 3: DFT₃ on c_i */
    old_count = r->count;
    old = (TCR_Entry *)malloc(old_count * sizeof(TCR_Entry));
    memcpy(old, r->entries, old_count * sizeof(TCR_Entry));
    uint8_t *done = (uint8_t *)calloc(old_count + 1, 1);
    r->count = 0;

    for (int i = 0; i < old_count; i++) {
        if (done[i]) continue;
        uint64_t cv0 = tcr_set_c(old[i].cvec, qi, 0);
        uint64_t b = old[i].b;

        double a_re[3] = {0}, a_im[3] = {0};
        for (int j = i; j < old_count; j++) {
            if (done[j]) continue;
            if (old[j].b != b) continue;
            if (tcr_set_c(old[j].cvec, qi, 0) != cv0) continue;
            int c = tcr_get_c(old[j].cvec, qi);
            a_re[c] = old[j].re;
            a_im[c] = old[j].im;
            done[j] = 1;
        }

        double o_re[3], o_im[3];
        o_re[0] = TCR_N3 * (a_re[0] + a_re[1] + a_re[2]);
        o_im[0] = TCR_N3 * (a_im[0] + a_im[1] + a_im[2]);

        double wb_r = TCR_W3R*a_re[1] - TCR_W3I*a_im[1];
        double wb_i = TCR_W3R*a_im[1] + TCR_W3I*a_re[1];
        double wc_r = TCR_W3R*a_re[2] + TCR_W3I*a_im[2];
        double wc_i = TCR_W3R*a_im[2] - TCR_W3I*a_re[2];
        o_re[1] = TCR_N3 * (a_re[0] + wb_r + wc_r);
        o_im[1] = TCR_N3 * (a_im[0] + wb_i + wc_i);

        double w2b_r = TCR_W3R*a_re[1] + TCR_W3I*a_im[1];
        double w2b_i = TCR_W3R*a_im[1] - TCR_W3I*a_re[1];
        double w2c_r = TCR_W3R*a_re[2] - TCR_W3I*a_im[2];
        double w2c_i = TCR_W3R*a_im[2] + TCR_W3I*a_re[2];
        o_re[2] = TCR_N3 * (a_re[0] + w2b_r + w2c_r);
        o_im[2] = TCR_N3 * (a_im[0] + w2b_i + w2c_i);

        for (int c = 0; c < 3; c++) {
            double m2 = o_re[c]*o_re[c] + o_im[c]*o_im[c];
            if (m2 > 1e-30)
                tcr_add(r, tcr_set_c(cv0, qi, c), b, o_re[c], o_im[c]);
        }
    }
    free(old); free(done);
    tcr_trim(r, 1e-15);
}

/* ═══════════════ DFT₆⁻¹ = DFT₆³ (since DFT⁴ = I) ═══════════════ */

static inline void tcr_apply_dft6_inv(ThreeCh_Register *r, int qi) {
    tcr_apply_dft6(r, qi);
    tcr_apply_dft6(r, qi);
    tcr_apply_dft6(r, qi);
}

/* ═══════════════ CZ Gate ═══════════════
 *
 * CZ is diagonal in computational basis: ω₆^{k_i · k_j}
 * where k = 3*p + s = 3*b_i + c_i.
 *
 * Strategy using DFT⁴ = I:
 *   1. DFT₆⁻¹ on both quhits (→ computational basis)
 *   2. Apply CZ diagonal
 *   3. DFT₆ on both quhits (→ back to DFT basis)
 *
 * But actually CZ is ALREADY diagonal in the (cvec, b) representation!
 * k_i = 3*b_i + c_i. Just apply the phase directly.
 */
static inline void tcr_apply_cz(ThreeCh_Register *r, int qi, int qj) {
    for (int e = 0; e < r->count; e++) {
        int ci = tcr_get_c(r->entries[e].cvec, qi);
        int cj = tcr_get_c(r->entries[e].cvec, qj);
        int bi = (r->entries[e].b >> qi) & 1;
        int bj = (r->entries[e].b >> qj) & 1;
        int ki = 3 * bi + ci;
        int kj = 3 * bj + cj;
        int ph = (ki * kj) % 6;
        if (ph == 0) continue;
        double re = r->entries[e].re, im = r->entries[e].im;
        r->entries[e].re = TCR_W6_RE[ph] * re - TCR_W6_IM[ph] * im;
        r->entries[e].im = TCR_W6_RE[ph] * im + TCR_W6_IM[ph] * re;
    }
}

/* ═══════════════ Flat index for comparison ═══════════════ */

static inline uint64_t tcr_flat_index(int N, uint64_t cvec, uint64_t b) {
    uint64_t flat = 0, pow6 = 1;
    uint64_t cv = cvec;
    for (int i = 0; i < N; i++) {
        int c = cv % 3; cv /= 3;
        int p = (b >> i) & 1;
        flat += (3 * p + c) * pow6;
        pow6 *= 6;
    }
    return flat;
}

/* ═══════════════ Born Probabilities ═══════════════ */

static inline void tcr_born_probs(const ThreeCh_Register *r,
                                  int qi, double probs[6]) {
    memset(probs, 0, 6 * sizeof(double));
    for (int e = 0; e < r->count; e++) {
        int c = tcr_get_c(r->entries[e].cvec, qi);
        int p = (r->entries[e].b >> qi) & 1;
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
    for (int e = 0; e < r->count; e++) {
        int ki = 3 * ((r->entries[e].b >> qi) & 1) +
                 tcr_get_c(r->entries[e].cvec, qi);
        int kj = 3 * ((r->entries[e].b >> qj) & 1) +
                 tcr_get_c(r->entries[e].cvec, qj);
        double mag2 = r->entries[e].re * r->entries[e].re +
                      r->entries[e].im * r->entries[e].im;
        corr += (double)(ki * kj) * mag2;
    }
    return corr;
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
    printf("  %s: N=%d, %d entries, prob=%.6f\n",
           label, r->N, r->count, tcr_total_prob(r));
    printf("    Compression: %.0f → %d (%.1f×)\n",
           pow(6.0, r->N), r->count, pow(6.0, r->N) / r->count);
}

#endif /* QUHIT_3CH_REGISTER_H */
