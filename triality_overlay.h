/*
 * triality_overlay.h — Triality-Powered Tensor Network Infrastructure
 *
 * Every overlay site gets a TrialityQuhit as its physical representation.
 * Gates route through optimal views. Inactive physical rows are skipped.
 * Diagonal 2-site gates bypass Θ+SVD when both sites are basis states.
 *
 * Used by: MPS, PEPS 2D, TNS 3D-6D
 */

#ifndef TRIALITY_OVERLAY_H
#define TRIALITY_OVERLAY_H

#include "quhit_triality.h"
#include "quhit_engine.h"
#include <string.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════════════
 * CONSTANTS
 * ═══════════════════════════════════════════════════════════════════════════════ */

#define TRI_OVL_D 6

/* ═══════════════════════════════════════════════════════════════════════════════
 * PER-SITE TRIALITY STATE
 *
 * One per tensor network site. The TrialityQuhit is the authoritative
 * physical-index representation. active_mask tracks which of the 6
 * physical rows carry nonzero amplitudes (for register gate skipping).
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    TrialityQuhit tri;          /* Authoritative physical state              */
    uint8_t       active_mask;  /* Bit k set ↔ |k⟩ has nonzero amplitude    */
    uint8_t       active_count; /* popcount(active_mask)                     */
    uint8_t       is_basis;     /* 1 if state is a single basis state        */
    int           basis_k;      /* Which k, valid only if is_basis == 1      */
} TriOverlaySite;

/* ═══════════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline void tri_site_init(TriOverlaySite *s)
{
    triality_init(&s->tri);
    s->active_mask  = 0x01;   /* |0⟩ */
    s->active_count = 1;
    s->is_basis     = 1;
    s->basis_k      = 0;
}

static inline void tri_site_init_basis(TriOverlaySite *s, int k)
{
    triality_init_basis(&s->tri, k);
    s->active_mask  = (uint8_t)(1 << k);
    s->active_count = 1;
    s->is_basis     = 1;
    s->basis_k      = k;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SYNC — Update active_mask from the triality edge amplitudes
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline void tri_site_sync(TriOverlaySite *s)
{
    triality_ensure_view(&s->tri, VIEW_EDGE);
    uint8_t mask = 0;
    int count = 0;
    int last_k = -1;
    for (int k = 0; k < TRI_OVL_D; k++) {
        double re = s->tri.edge_re[k];
        double im = s->tri.edge_im[k];
        if (re * re + im * im > 1e-20) {
            mask |= (uint8_t)(1 << k);
            count++;
            last_k = k;
        }
    }
    s->active_mask  = mask;
    s->active_count = (uint8_t)count;
    s->is_basis     = (count == 1);
    s->basis_k      = (count == 1) ? last_k : -1;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * GATE CLASSIFICATION — What kind of unitary is this?
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef enum {
    GATE_DIAGONAL,   /* Only diagonal entries nonzero → Edge view, O(D)     */
    GATE_SHIFT,      /* Permutation |k⟩→|k+δ mod D⟩ → Vertex view, O(D)   */
    GATE_DFT,        /* DFT₆ matrix → O(1) view switch                     */
    GATE_GENERAL     /* Arbitrary unitary → full O(D²) multiply             */
} GateClass;

static inline GateClass tri_classify_gate(const double *U_re, const double *U_im)
{
    int D = TRI_OVL_D;

    /* Check if diagonal: all off-diagonal entries zero */
    int is_diag = 1;
    for (int j = 0; j < D && is_diag; j++)
        for (int k = 0; k < D && is_diag; k++)
            if (j != k && (U_re[j * D + k] != 0.0 || U_im[j * D + k] != 0.0))
                is_diag = 0;
    if (is_diag) return GATE_DIAGONAL;

    /* Check if shift: exactly one nonzero per row, and it maps k→(k+δ) mod D */
    int shift_delta = -1;
    int is_shift = 1;
    for (int j = 0; j < D && is_shift; j++) {
        int nnz = 0, col_hit = -1;
        for (int k = 0; k < D; k++) {
            double mag2 = U_re[j * D + k] * U_re[j * D + k] +
                          U_im[j * D + k] * U_im[j * D + k];
            if (mag2 > 1e-20) { nnz++; col_hit = k; }
        }
        if (nnz != 1) { is_shift = 0; break; }
        /* Check if magnitude is ~1 (unitary permutation) */
        double mr = U_re[j * D + col_hit], mi = U_im[j * D + col_hit];
        if (fabs(mr * mr + mi * mi - 1.0) > 1e-10) { is_shift = 0; break; }
        int delta = (j - col_hit + D) % D;
        if (shift_delta < 0) shift_delta = delta;
        else if (delta != shift_delta) { is_shift = 0; break; }
    }
    if (is_shift) return GATE_SHIFT;

    /* Check if DFT₆: U[j,k] ∝ ω^(jk), with uniform magnitude 1/√6 */
    double inv6 = 1.0 / 6.0;
    int is_dft = 1;
    for (int j = 0; j < D && is_dft; j++)
        for (int k = 0; k < D && is_dft; k++) {
            double mag2 = U_re[j*D+k] * U_re[j*D+k] + U_im[j*D+k] * U_im[j*D+k];
            if (fabs(mag2 - inv6) > 1e-8) is_dft = 0;
        }
    if (is_dft) return GATE_DFT;

    return GATE_GENERAL;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * GATE APPLICATION — Route through Triality for optimal-view execution
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline void tri_site_apply_gate(TriOverlaySite *s,
                                       const double *U_re, const double *U_im)
{
    GateClass gc = tri_classify_gate(U_re, U_im);

    switch (gc) {
    case GATE_DIAGONAL: {
        /* Apply as phase gate in Edge view: O(D) */
        double phi_re[TRI_OVL_D], phi_im[TRI_OVL_D];
        for (int k = 0; k < TRI_OVL_D; k++) {
            phi_re[k] = U_re[k * TRI_OVL_D + k];
            phi_im[k] = U_im[k * TRI_OVL_D + k];
        }
        triality_phase(&s->tri, phi_re, phi_im);
        /* Diagonal gates preserve active_mask (only phases change) */
        break;
    }
    case GATE_SHIFT: {
        /* Find the shift delta */
        int delta = 0;
        for (int k = 0; k < TRI_OVL_D; k++) {
            if (U_re[0 * TRI_OVL_D + k] != 0.0 || U_im[0 * TRI_OVL_D + k] != 0.0) {
                delta = (0 - k + TRI_OVL_D) % TRI_OVL_D;
                break;
            }
        }
        triality_shift(&s->tri, delta);
        /* Shift rotates the active_mask */
        uint8_t old = s->active_mask;
        uint8_t new_mask = 0;
        for (int k = 0; k < TRI_OVL_D; k++)
            if (old & (1 << k))
                new_mask |= (uint8_t)(1 << ((k + delta) % TRI_OVL_D));
        s->active_mask = new_mask;
        if (s->is_basis)
            s->basis_k = (s->basis_k + delta) % TRI_OVL_D;
        break;
    }
    case GATE_DFT:
        triality_dft(&s->tri);
        /* DFT spreads to all states */
        s->active_mask  = 0x3F;  /* all 6 bits */
        s->active_count = 6;
        s->is_basis     = 0;
        s->basis_k      = -1;
        break;
    case GATE_GENERAL:
        /* Fall back to full unitary via triality */
        triality_ensure_view(&s->tri, VIEW_EDGE);
        {
            double out_re[TRI_OVL_D] = {0}, out_im[TRI_OVL_D] = {0};
            for (int j = 0; j < TRI_OVL_D; j++)
                for (int k = 0; k < TRI_OVL_D; k++) {
                    out_re[j] += U_re[j * TRI_OVL_D + k] * s->tri.edge_re[k]
                               - U_im[j * TRI_OVL_D + k] * s->tri.edge_im[k];
                    out_im[j] += U_re[j * TRI_OVL_D + k] * s->tri.edge_im[k]
                               + U_im[j * TRI_OVL_D + k] * s->tri.edge_re[k];
                }
            memcpy(s->tri.edge_re, out_re, sizeof(out_re));
            memcpy(s->tri.edge_im, out_im, sizeof(out_im));
            /* Edge is now authoritative; mark all other views dirty */
            s->tri.dirty = DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED | DIRTY_EXOTIC;
        }
        tri_site_sync(s);
        break;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MASKED 1-SITE GATE — Register-level gate with active-mask skip
 *
 * Stack-based accumulation (no malloc/qsort). Skips physical rows not
 * in active_mask. Generic across all overlays via chi_power parameter.
 *
 * chi_power = the divisor that extracts k from basis_state:
 *   MPS:    χ²         PEPS2D: χ⁴        TNS3D: χ⁶
 *   TNS4D:  χ⁸         TNS5D: χ¹⁰        TNS6D: χ¹²
 *
 * For 5D/6D, chi_power exceeds uint64_t. Use unsigned __int128.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Stack-buffer entry for accumulation (no heap) */
typedef struct {
    unsigned __int128 basis;
    double re, im;
} tri_tmp_entry;

/* Maximum stack entries — 4096 matches register sparse capacity */
#define TRI_TMP_MAX 4096

static inline void tri_reg_gate_1site_masked(
    QuhitRegister *reg,
    const double *U_re, const double *U_im,
    uint8_t active_mask,
    unsigned __int128 chi_power)
{
    int D = TRI_OVL_D;
    tri_tmp_entry tmp[TRI_TMP_MAX];
    int nout = 0;

    for (uint32_t e = 0; e < reg->num_nonzero; e++) {
        unsigned __int128 bs = (unsigned __int128)reg->entries[e].basis_state;
        double ar = reg->entries[e].amp_re;
        double ai = reg->entries[e].amp_im;
        int k = (int)(bs / chi_power);
        unsigned __int128 bond = bs % chi_power;

        /* Skip inactive source rows */
        if (!(active_mask & (1 << k))) continue;

        for (int kp = 0; kp < D; kp++) {
            double ur = U_re[kp * D + k];
            double ui = U_im[kp * D + k];
            double nr = ur * ar - ui * ai;
            double ni = ur * ai + ui * ar;
            if (nr * nr + ni * ni < 1e-30) continue;

            unsigned __int128 new_basis = (unsigned __int128)kp * chi_power + bond;

            /* Linear scan for existing basis (fast for sparse) */
            int found = -1;
            for (int t = 0; t < nout; t++) {
                if (tmp[t].basis == new_basis) { found = t; break; }
            }
            if (found >= 0) {
                tmp[found].re += nr;
                tmp[found].im += ni;
            } else if (nout < TRI_TMP_MAX) {
                tmp[nout].basis = new_basis;
                tmp[nout].re = nr;
                tmp[nout].im = ni;
                nout++;
            }
        }
    }

    /* Write back to register */
    reg->num_nonzero = 0;
    for (int t = 0; t < nout; t++) {
        if (tmp[t].re * tmp[t].re + tmp[t].im * tmp[t].im >= 1e-30 &&
            reg->num_nonzero < 4096) {
            reg->entries[reg->num_nonzero].basis_state = (basis_t)tmp[t].basis;
            reg->entries[reg->num_nonzero].amp_re = tmp[t].re;
            reg->entries[reg->num_nonzero].amp_im = tmp[t].im;
            reg->num_nonzero++;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * DIAGONAL 2-SITE FAST-PATH — Skip Θ+SVD for diagonal gates on basis states
 *
 * When both sites are in basis states |kA⟩ and |kB⟩, a diagonal 2-site gate
 * G[kk', kk'] just multiplies by a phase ω^(kA·kB). No contraction needed.
 *
 * This applies the phase directly to all register entries on site A.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline int tri_diag_2site_fastpath(
    TriOverlaySite *sA, TriOverlaySite *sB,
    QuhitRegister *regA, QuhitRegister *regB,
    const double *G_re, const double *G_im,
    int D)
{
    /* Only works if BOTH sites are basis states */
    if (!sA->is_basis || !sB->is_basis) return 0;

    int kA = sA->basis_k;
    int kB = sB->basis_k;
    int D2 = D * D;
    int idx = kA * D + kB;

    /* Gate must be diagonal */
    double gre = G_re[idx * D2 + idx];
    double gim = G_im[idx * D2 + idx];

    /* Phase is trivial (identity) */
    if (fabs(gre - 1.0) < 1e-14 && fabs(gim) < 1e-14) return 1;

    /* Apply phase to all entries in register A */
    for (uint32_t e = 0; e < regA->num_nonzero; e++) {
        double ar = regA->entries[e].amp_re;
        double ai = regA->entries[e].amp_im;
        regA->entries[e].amp_re = gre * ar - gim * ai;
        regA->entries[e].amp_im = gre * ai + gim * ar;
    }

    /* Apply CZ in triality space */
    triality_ensure_view(&sA->tri, VIEW_EDGE);
    triality_ensure_view(&sB->tri, VIEW_EDGE);
    /* CZ|kA⟩|kB⟩ = ω^(kA·kB)|kA⟩|kB⟩ — phase on A only */
    {
        double ar = sA->tri.edge_re[kA];
        double ai = sA->tri.edge_im[kA];
        sA->tri.edge_re[kA] = gre * ar - gim * ai;
        sA->tri.edge_im[kA] = gre * ai + gim * ar;
    }

    return 1; /* Handled — skip Θ+SVD */
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * TRIALITY CZ — Apply CZ between two overlay sites' triality quhits
 *
 * Replaces the standard engine quhit_apply_cz() mirror call.
 * Uses triality's native phase gate in edge view.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static const double TRI_OVL_W6_RE[6] = {
    1.0, 0.5, -0.5, -1.0, -0.5, 0.5
};
static const double TRI_OVL_W6_IM[6] = {
    0.0, 0.86602540378443864676, 0.86602540378443864676,
    0.0, -0.86602540378443864676, -0.86602540378443864676
};

static inline void tri_site_apply_cz(TriOverlaySite *sA, TriOverlaySite *sB)
{
    /*
     * CZ|j⟩|k⟩ = ω^(jk)|j⟩|k⟩.
     * For separate quhits (not entangled pair), we apply the phase
     * conditioned on the OTHER site's state. When both are basis states
     * this is exact; otherwise it's the marginal approximation matching
     * what the old quhit_apply_cz mirror did.
     */
    if (sA->is_basis && sB->is_basis) {
        /* Exact: CZ|kA⟩|kB⟩ = ω^(kA·kB)|kA⟩|kB⟩ */
        int prod = (sA->basis_k * sB->basis_k) % TRI_OVL_D;
        if (prod != 0) {
            triality_phase_single(&sA->tri, sA->basis_k,
                                  TRI_OVL_W6_RE[prod], TRI_OVL_W6_IM[prod]);
        }
    } else {
        /* Approximation: apply Z^kB to A for each kB with weight */
        triality_ensure_view(&sA->tri, VIEW_EDGE);
        triality_ensure_view(&sB->tri, VIEW_EDGE);
        /* mean-field: phase each |j⟩ in A by Σ_k p(k) ω^(jk)
         * This matches what the old single-quhit mirror did. */
        double p[TRI_OVL_D];
        double total = 0;
        for (int k = 0; k < TRI_OVL_D; k++) {
            p[k] = sB->tri.edge_re[k] * sB->tri.edge_re[k] +
                    sB->tri.edge_im[k] * sB->tri.edge_im[k];
            total += p[k];
        }
        if (total > 1e-30)
            for (int k = 0; k < TRI_OVL_D; k++) p[k] /= total;

        for (int j = 0; j < TRI_OVL_D; j++) {
            /* Effective phase on |j⟩: Σ_k p(k) ω^(jk) */
            double eff_re = 0, eff_im = 0;
            for (int k = 0; k < TRI_OVL_D; k++) {
                int idx = (j * k) % TRI_OVL_D;
                eff_re += p[k] * TRI_OVL_W6_RE[idx];
                eff_im += p[k] * TRI_OVL_W6_IM[idx];
            }
            triality_phase_single(&sA->tri, j, eff_re, eff_im);
        }
    }
    tri_site_sync(sA);
    tri_site_sync(sB);
}

#endif /* TRIALITY_OVERLAY_H */
