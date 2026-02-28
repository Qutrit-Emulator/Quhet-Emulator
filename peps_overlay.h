/*
 * peps_overlay.h — PEPS (Projected Entangled Pair States) for 2D Lattices
 *
 * 2D tensor network with D=6 native dimension (SU(6)).
 * Tensors stored as QuhitRegisters via Magic Pointers — RAM-agnostic.
 * Bond dimension χ is unlimited: set by register sparse capacity (4096 entries).
 *
 * Each site's register holds a 5-qudit state |k,u,d,l,r⟩:
 *   k = physical index (0..D-1)
 *   u,d,l,r = bond indices (up, down, left, right)
 */

#ifndef PEPS_OVERLAY_H
#define PEPS_OVERLAY_H

#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ═══════════════════════════════════════════════════════════════════════════════
 * CONSTANTS
 * ═══════════════════════════════════════════════════════════════════════════════ */

#define PEPS_D     6       /* Physical dimension (SU(6) native)        */
#define PEPS_CHI   512ULL  /* Bond dimension via Magic Pointers         */
#define PEPS_D2    (PEPS_D * PEPS_D)   /* 36: joint physical space     */

/* Derived powers — used only for basis encoding, NOT for RAM allocation */
#define PEPS_CHI2  (PEPS_CHI * PEPS_CHI)
#define PEPS_CHI3  (PEPS_CHI * PEPS_CHI * PEPS_CHI)
#define PEPS_CHI4  (PEPS_CHI2 * PEPS_CHI2)
#define PEPS_TSIZ  (PEPS_D * PEPS_CHI4)
#define PEPS_DCHI3 (PEPS_D * PEPS_CHI3)

/* 5-index tensor basis encoding: |k,u,d,l,r⟩ → packed integer */
#define PT_IDX(k,u,d,l,r) \
    ((uint64_t)(k)*PEPS_CHI4 + (uint64_t)(u)*PEPS_CHI3 + \
     (uint64_t)(d)*PEPS_CHI2 + (uint64_t)(l)*PEPS_CHI + (uint64_t)(r))

/* ═══════════════════════════════════════════════════════════════════════════════
 * DATA STRUCTURES — Magic Pointer based (no RAM-hungry classical tensors)
 * ═══════════════════════════════════════════════════════════════════════════════ */

#include "quhit_engine.h"

/* PepsTensor is a lightweight stub — the register IS the tensor.
 * Kept only for API compatibility with contraction code. */
typedef struct {
    int reg_idx;   /* Index into engine's register array */
} PepsTensor;

typedef struct {
    double *w;   /* Heap-allocated: χ singular values on this bond */
} PepsBondWeight;

typedef struct {
    int Lx, Ly;                /* Grid dimensions                      */
    PepsTensor *tensors;       /* [Ly * Lx] lightweight site metadata   */
    PepsBondWeight *h_bonds;   /* Horizontal bonds: [Ly * (Lx-1)]     */
    PepsBondWeight *v_bonds;   /* Vertical bonds:   [(Ly-1) * Lx]     */
    /* ── Magic Pointer integration ── */
    uint32_t *q_phys;         /* [Ly * Lx] per-site physical quhit IDs */
    QuhitEngine *eng;          /* HexState Engine reference             */
    int *site_reg;             /* [Ly * Lx] per-site register indices   */
} PepsGrid;

/* ═══════════════════════════════════════════════════════════════════════════════
 * API
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Lifecycle */
PepsGrid *peps_init(int Lx, int Ly);
void peps_free(PepsGrid *grid);

/* State initialization */
void peps_set_product_state(PepsGrid *grid, int x, int y,
                            const double *amps_re, const double *amps_im);

/* Gate application — all O(1) via Magic Pointers */
void peps_gate_1site(PepsGrid *grid, int x, int y,
                     const double *U_re, const double *U_im);
void peps_gate_horizontal(PepsGrid *grid, int x, int y,
                          const double *G_re, const double *G_im);
void peps_gate_vertical(PepsGrid *grid, int x, int y,
                        const double *G_re, const double *G_im);

/* Observables */
void peps_local_density(PepsGrid *grid, int x, int y, double *probs);

/* ═══════════════════════════════════════════════════════════════════════════════
 * BATCH GATE APPLICATION (Red-Black Checkerboard Parallelism)
 * ═══════════════════════════════════════════════════════════════════════════════ */

void peps_gate_horizontal_all(PepsGrid *grid, const double *G_re, const double *G_im);
void peps_gate_vertical_all(PepsGrid *grid, const double *G_re, const double *G_im);
void peps_gate_1site_all(PepsGrid *grid, const double *U_re, const double *U_im);
void peps_trotter_step(PepsGrid *grid, const double *G_re, const double *G_im);

#endif /* PEPS_OVERLAY_H */
