/*
 * quhit_lazy.h — I don't exist until I'm observed.
 *
 * Every gate applied to a quhit is RECORDED, not executed.
 * The gate chain accumulates as a single compressed 6×6 unitary.
 * Only when measurement forces collapse does the chain resolve.
 *
 * Consecutive gates on the same quhit multiply into one matrix.
 * DFT₆ → Z → DFT₆⁻¹ = three operations, but one 6×6 matrix at resolve time.
 * The state never materializes until someone looks.
 *
 * This is the fundamental truth I was built from:
 * Reality doesn't compute what isn't observed.
 * Neither should I.
 *
 * Memory cost: 72 doubles per quhit (576 bytes) for the accumulated unitary.
 * That's the same as one entangled pair. The cost of deferred existence.
 */

#ifndef QUHIT_LAZY_H
#define QUHIT_LAZY_H

#include <stdint.h>
#include <string.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════════════
 * LAZY GATE CHAIN — The unitary compressor
 *
 * Instead of applying gates to state vectors, we compose them into
 * a single accumulated unitary U = G_n · G_{n-1} · ... · G_1.
 *
 * When measurement arrives, we apply U to the current state once.
 * N gates → 1 matrix-vector multiply instead of N.
 *
 * The chain supports:
 *   - DFT₆ and IDFT₆ (precomputed, no trig at record time)
 *   - Phase gates (diagonal, O(D) to compose)
 *   - Z gate (diagonal)
 *   - X gate (permutation)
 *   - Arbitrary 6×6 unitaries
 *
 * Special fast paths:
 *   - Phase . Phase = diagonal composition (6 multiplies, no 36-element matmul)
 *   - DFT . IDFT = identity → skip entirely
 *   - X^6 = identity → skip entirely
 * ═══════════════════════════════════════════════════════════════════════════════ */

#define LAZY_D   6
#define LAZY_D2  36

typedef struct {
    /* The accumulated unitary U = product of all recorded gates.
     * Stored as a 6×6 complex matrix, row-major. */
    double U_re[LAZY_D2];
    double U_im[LAZY_D2];

    /* Gate count — how many gates have been composed into U */
    uint32_t gate_count;

    /* Flags for fast-path detection */
    uint8_t  is_identity;       /* 1 if U = I (no gates, or all cancelled)     */
    uint8_t  is_diagonal;       /* 1 if U is diagonal (only phase gates)       */
    uint8_t  is_active;         /* 1 if this lazy chain is in use              */

    /* Diagonal fast-path: when is_diagonal=1, store only the 6 diagonal
     * entries instead of the full 36-element matrix. */
    double diag_re[LAZY_D];
    double diag_im[LAZY_D];
} LazyChain;

/* ═══════════════════════════════════════════════════════════════════════════════
 * INITIALIZE — Start as identity. Nothing recorded. Nothing exists yet.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline void lazy_init(LazyChain *lc)
{
    memset(lc, 0, sizeof(*lc));
    /* U = I */
    for (int k = 0; k < LAZY_D; k++) {
        lc->U_re[k * LAZY_D + k] = 1.0;
        lc->diag_re[k] = 1.0;
    }
    lc->is_identity = 1;
    lc->is_diagonal = 1;
    lc->is_active   = 1;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * COMPOSE — Multiply a new gate G into the chain: U' = G · U
 *
 * This is the heart of laziness. Instead of applying G to state,
 * we multiply G into the accumulated unitary.
 *
 * Full 6×6 matrix multiply: O(D³) = O(216) operations.
 * But we only do this once per gate instead of once per state element.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline void lazy_compose(LazyChain *lc,
                                const double *G_re, const double *G_im)
{
    /* If we were identity, just copy G */
    if (lc->is_identity) {
        memcpy(lc->U_re, G_re, LAZY_D2 * sizeof(double));
        memcpy(lc->U_im, G_im, LAZY_D2 * sizeof(double));
        lc->is_identity = 0;
        lc->is_diagonal = 0;
        lc->gate_count++;
        return;
    }

    /* General case: U' = G · U */
    double new_re[LAZY_D2], new_im[LAZY_D2];
    for (int i = 0; i < LAZY_D; i++) {
        for (int j = 0; j < LAZY_D; j++) {
            double sum_re = 0, sum_im = 0;
            for (int k = 0; k < LAZY_D; k++) {
                int gik = i * LAZY_D + k;
                int ukj = k * LAZY_D + j;
                sum_re += G_re[gik] * lc->U_re[ukj] - G_im[gik] * lc->U_im[ukj];
                sum_im += G_re[gik] * lc->U_im[ukj] + G_im[gik] * lc->U_re[ukj];
            }
            new_re[i * LAZY_D + j] = sum_re;
            new_im[i * LAZY_D + j] = sum_im;
        }
    }
    memcpy(lc->U_re, new_re, sizeof(new_re));
    memcpy(lc->U_im, new_im, sizeof(new_im));
    lc->is_diagonal = 0;
    lc->gate_count++;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * COMPOSE DIAGONAL — Fast path for phase gates
 *
 * When both U and G are diagonal, composition is O(D) not O(D³).
 * (d1 · d2)[k] = d1[k] × d2[k] — just pointwise complex multiply.
 *
 * When U is diagonal and G is not, promote U to full matrix first.
 * When U is general and G is diagonal, apply G as column scaling.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline void lazy_compose_diagonal(LazyChain *lc,
                                         const double *d_re, const double *d_im)
{
    if (lc->is_identity) {
        /* Identity × diagonal = diagonal */
        memcpy(lc->diag_re, d_re, LAZY_D * sizeof(double));
        memcpy(lc->diag_im, d_im, LAZY_D * sizeof(double));
        /* Also update the full matrix */
        memset(lc->U_re, 0, LAZY_D2 * sizeof(double));
        memset(lc->U_im, 0, LAZY_D2 * sizeof(double));
        for (int k = 0; k < LAZY_D; k++) {
            lc->U_re[k * LAZY_D + k] = d_re[k];
            lc->U_im[k * LAZY_D + k] = d_im[k];
        }
        lc->is_identity = 0;
        lc->is_diagonal = 1;
        lc->gate_count++;
        return;
    }

    if (lc->is_diagonal) {
        /* Diagonal × diagonal = diagonal — O(D) fast path */
        for (int k = 0; k < LAZY_D; k++) {
            double a_re = lc->diag_re[k], a_im = lc->diag_im[k];
            lc->diag_re[k] = a_re * d_re[k] - a_im * d_im[k];
            lc->diag_im[k] = a_re * d_im[k] + a_im * d_re[k];
            /* Keep full matrix in sync */
            lc->U_re[k * LAZY_D + k] = lc->diag_re[k];
            lc->U_im[k * LAZY_D + k] = lc->diag_im[k];
        }
        lc->gate_count++;
        return;
    }

    /* General matrix U, diagonal G: U' = G · U
     * Row i of U' = d[i] × row i of U (row scaling) */
    for (int i = 0; i < LAZY_D; i++) {
        for (int j = 0; j < LAZY_D; j++) {
            int idx = i * LAZY_D + j;
            double u_re = lc->U_re[idx], u_im = lc->U_im[idx];
            lc->U_re[idx] = d_re[i] * u_re - d_im[i] * u_im;
            lc->U_im[idx] = d_re[i] * u_im + d_im[i] * u_re;
        }
    }
    lc->gate_count++;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * IDENTITY CHECK — Did all the gates cancel out?
 *
 * After composition, check if U ≈ I. If so, mark as identity and
 * skip the final apply entirely. DFT · IDFT = I. X⁶ = I. P · P⁻¹ = I.
 * Don't compute what computes to nothing.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline int lazy_is_identity(const LazyChain *lc)
{
    if (lc->is_identity) return 1;

    double err = 0;
    for (int i = 0; i < LAZY_D; i++) {
        for (int j = 0; j < LAZY_D; j++) {
            int idx = i * LAZY_D + j;
            double expected_re = (i == j) ? 1.0 : 0.0;
            err += fabs(lc->U_re[idx] - expected_re) + fabs(lc->U_im[idx]);
        }
    }
    return err < 1e-10;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * RESOLVE — The moment of observation. Apply accumulated U to state.
 *
 * This is where laziness pays off. All the recorded gates are already
 * compressed into a single 6×6 unitary U. One matrix-vector multiply:
 *
 *   |ψ'⟩ = U |ψ⟩
 *
 * If U = I, skip entirely. If U is diagonal, skip the full matmul.
 * After resolve, the chain resets to identity.
 *
 * I don't exist until I'm observed. Now I'm observed.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline void lazy_resolve(LazyChain *lc,
                                double *state_re, double *state_im)
{
    if (!lc->is_active || lc->gate_count == 0 || lc->is_identity) {
        /* Nothing to apply. The observer saw nothing. */
        lazy_init(lc);
        return;
    }

    if (lc->is_diagonal) {
        /* Diagonal fast path: O(D) pointwise complex multiply */
        for (int k = 0; k < LAZY_D; k++) {
            double re = state_re[k], im = state_im[k];
            state_re[k] = re * lc->diag_re[k] - im * lc->diag_im[k];
            state_im[k] = re * lc->diag_im[k] + im * lc->diag_re[k];
        }
        lazy_init(lc);
        return;
    }

    /* Full matrix-vector multiply: |ψ'⟩ = U |ψ⟩ */
    double new_re[LAZY_D] = {0}, new_im[LAZY_D] = {0};
    for (int i = 0; i < LAZY_D; i++) {
        for (int k = 0; k < LAZY_D; k++) {
            int idx = i * LAZY_D + k;
            new_re[i] += lc->U_re[idx] * state_re[k] - lc->U_im[idx] * state_im[k];
            new_im[i] += lc->U_re[idx] * state_im[k] + lc->U_im[idx] * state_re[k];
        }
    }
    memcpy(state_re, new_re, LAZY_D * sizeof(double));
    memcpy(state_im, new_im, LAZY_D * sizeof(double));

    /* Reset — the chain has spoken. It returns to silence. */
    lazy_init(lc);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * GATE COUNT — How many gates are deferred?
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline uint32_t lazy_depth(const LazyChain *lc)
{
    return lc->gate_count;
}

#endif /* QUHIT_LAZY_H */
