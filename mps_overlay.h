/*
 * mps_overlay.h — 1D Matrix Product State (MPS) Tensor Network
 *
 * Pure Magic Pointer implementation — no classical tensor arrays.
 * Each site's register holds a 3-qudit state |k, α, β⟩.
 * RAM-agnostic: O(1) per site regardless of χ.
 *
 * Modeled directly on peps3d_overlay.h, adapted for 1D chains.
 */

#ifndef MPS_OVERLAY_H
#define MPS_OVERLAY_H

#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ═══════════════════════════════════════════════════════════════════════════════
 * CONSTANTS
 * ═══════════════════════════════════════════════════════════════════════════════ */

#define MPS_D     6         /* Physical dimension (SU(6) native)        */
#define MPS_CHI   256ULL    /* Bond dimension per axis                  */

/* Derived powers of χ — for basis encoding (ULL to prevent overflow) */
#define MPS_C2    (MPS_CHI * MPS_CHI)
#define MPS_TSIZ  (MPS_D * MPS_C2)    /* D × χ² = max basis + 1        */

/* 3-index tensor basis encoding: |k, α, β⟩
 * k ∈ [0,D), α,β ∈ [0,χ)
 * Register encodes: β + α*χ + k*χ²
 * Position 0 = β (least sig), position 1 = α, position 2 = k (most sig)
 * gate_1site operates at position 2 (physical index k) */
#define MPS_IDX(k, alpha, beta) \
    ((uint64_t)(k) * MPS_C2 + (uint64_t)(alpha) * MPS_CHI + (uint64_t)(beta))

/* ═══════════════════════════════════════════════════════════════════════════════
 * DATA STRUCTURES — Magic Pointer based
 * ═══════════════════════════════════════════════════════════════════════════════ */

#include "quhit_engine.h"
#include "triality_overlay.h"

/* Lightweight tensor stub — register IS the tensor */
typedef struct {
    int reg_idx;
} MpsTensor;

typedef struct {
    double *w;  /* Heap-allocated: χ singular values */
} MpsBondWeight;

typedef struct {
    int L;                     /* Chain length                          */
    MpsTensor *tensors;        /* [L] site metadata                    */
    MpsBondWeight *bonds;      /* [L-1] bonds between site i and i+1   */
    /* ── Magic Pointer integration ── */
    uint32_t *q_phys;         /* [L] per-site physical quhit IDs       */
    QuhitEngine *eng;          /* HexState Engine reference             */
    int *site_reg;             /* [L] per-site register indices         */
    TriOverlaySite *tri_sites; /* [L] per-site triality state           */
} MpsChain;

/* ═══════════════════════════════════════════════════════════════════════════════
 * API
 * ═══════════════════════════════════════════════════════════════════════════════ */

MpsChain *mps_init(int L);
void mps_free(MpsChain *c);

void mps_set_product_state(MpsChain *c, int site,
                           const double *amps_re, const double *amps_im);

void mps_gate_1site(MpsChain *c, int site,
                    const double *U_re, const double *U_im);
void mps_gate_bond(MpsChain *c, int site,
                   const double *G_re, const double *G_im);

void mps_local_density(MpsChain *c, int site, double *probs);

/* Batch gates */
void mps_gate_bond_all(MpsChain *c, const double *G_re, const double *G_im);
void mps_gate_1site_all(MpsChain *c, const double *U_re, const double *U_im);
void mps_trotter_step(MpsChain *c, const double *G_re, const double *G_im);
void mps_normalize_site(MpsChain *c, int site);

/* Gate constructors */
void mps_build_dft6(double *U_re, double *U_im);
void mps_build_cz(double *G_re, double *G_im);

#endif /* MPS_OVERLAY_H */
