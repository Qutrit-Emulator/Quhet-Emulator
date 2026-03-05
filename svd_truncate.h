/*
 * svd_truncate.h — Shared SVD write-back truncation helper
 *
 * When SVD write-back produces > 4096 entries, keeps the top 4096
 * by |amplitude|² instead of first-come-first-served.
 *
 * Usage:
 *   static SvdTmpBuf svd_buf;
 *   svd_buf_reset(&svd_buf);
 *   // ... collect entries via svd_buf_push(&svd_buf, bs, re, im) ...
 *   svd_buf_flush(&svd_buf, reg);
 */

#ifndef SVD_TRUNCATE_H
#define SVD_TRUNCATE_H

#include <string.h>

#define SVD_TMP_CAP  16384
#define SVD_REG_CAP  4096

typedef struct {
    basis_t basis_state;
    double  amp_re, amp_im;
} SvdTmpEntry;

typedef struct {
    SvdTmpEntry entries[SVD_TMP_CAP];
    uint32_t    count;
} SvdTmpBuf;

static inline void svd_buf_reset(SvdTmpBuf *b) { b->count = 0; }

static inline void svd_buf_push(SvdTmpBuf *b, basis_t bs, double re, double im) {
    if (re*re + im*im > 1e-30 && b->count < SVD_TMP_CAP) {
        b->entries[b->count].basis_state = bs;
        b->entries[b->count].amp_re = re;
        b->entries[b->count].amp_im = im;
        b->count++;
    }
}

/* Flush buffer into register, truncating to top SVD_REG_CAP by magnitude */
static inline void svd_buf_flush(SvdTmpBuf *b, QuhitRegister *reg) {
    if (b->count <= SVD_REG_CAP) {
        /* Fits directly — fast copy */
        reg->num_nonzero = b->count;
        for (uint32_t i = 0; i < b->count; i++) {
            reg->entries[i].basis_state = b->entries[i].basis_state;
            reg->entries[i].amp_re      = b->entries[i].amp_re;
            reg->entries[i].amp_im      = b->entries[i].amp_im;
        }
        return;
    }
    /* Partial selection sort: bubble top SVD_REG_CAP to front */
    for (uint32_t i = 0; i < SVD_REG_CAP; i++) {
        uint32_t best = i;
        double best_m = b->entries[i].amp_re * b->entries[i].amp_re +
                        b->entries[i].amp_im * b->entries[i].amp_im;
        for (uint32_t j = i + 1; j < b->count; j++) {
            double m = b->entries[j].amp_re * b->entries[j].amp_re +
                       b->entries[j].amp_im * b->entries[j].amp_im;
            if (m > best_m) { best = j; best_m = m; }
        }
        if (best != i) {
            SvdTmpEntry tmp = b->entries[i];
            b->entries[i] = b->entries[best];
            b->entries[best] = tmp;
        }
    }
    reg->num_nonzero = SVD_REG_CAP;
    for (uint32_t i = 0; i < SVD_REG_CAP; i++) {
        reg->entries[i].basis_state = b->entries[i].basis_state;
        reg->entries[i].amp_re      = b->entries[i].amp_re;
        reg->entries[i].amp_im      = b->entries[i].amp_im;
    }
}

#endif /* SVD_TRUNCATE_H */
