/*
 * peps_overlay.h — PEPS (Projected Entangled Pair States) for 2D Lattices
 *
 * 2D tensor network with D=6 native dimension (SU(6)).
 * Each site has a 5-index tensor T[k][u][d][l][r]:
 *   k = physical index (0..D-1)
 *   u,d,l,r = bond indices (up, down, left, right, 0..χ-1)
 *
 * Simple update algorithm for gate application.
 * Bond weights (singular values) stored on each bond.
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
#define PEPS_CHI   4       /* Bond dimension (χ⁵ scaling!)             */
#define PEPS_D2    (PEPS_D * PEPS_D)   /* 36: joint physical space     */

/* Derived powers of χ */
#define PEPS_CHI2  (PEPS_CHI * PEPS_CHI)                /* χ²  = 16    */
#define PEPS_CHI3  (PEPS_CHI * PEPS_CHI * PEPS_CHI)     /* χ³  = 64    */
#define PEPS_CHI4  (PEPS_CHI2 * PEPS_CHI2)              /* χ⁴  = 256   */
#define PEPS_TSIZ  (PEPS_D * PEPS_CHI4)                 /* entries/site */
#define PEPS_DCHI3 (PEPS_D * PEPS_CHI3)                 /* SVD matrix dim */

/* 5-index tensor access: T[k][u][d][l][r] */
#define PT_IDX(k,u,d,l,r) \
    ((k)*PEPS_CHI4 + (u)*PEPS_CHI3 + (d)*PEPS_CHI2 + (l)*PEPS_CHI + (r))

/* ═══════════════════════════════════════════════════════════════════════════════
 * DATA STRUCTURES
 * ═══════════════════════════════════════════════════════════════════════════════ */

#include "quhit_engine.h"

typedef struct {
    double re[PEPS_TSIZ];
    double im[PEPS_TSIZ];
} PepsTensor;

typedef struct {
    double w[PEPS_CHI];   /* Singular values on this bond */
} PepsBondWeight;

typedef struct {
    int Lx, Ly;                /* Grid dimensions                      */
    PepsTensor *tensors;       /* [Ly * Lx] site tensors               */
    PepsBondWeight *h_bonds;   /* Horizontal bonds: [Ly * (Lx-1)]     */
    PepsBondWeight *v_bonds;   /* Vertical bonds:   [(Ly-1) * Lx]     */
    /* ── Magic Pointer integration ── */
    uint32_t *q_phys;         /* [Ly * Lx] per-site physical quhit IDs */
    QuhitEngine *eng;          /* HexState Engine reference             */
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

/* Gate application */
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
 *
 * Apply a 2-site gate to ALL bonds along an axis using a checkerboard pattern.
 * Even-parity bonds first, then odd-parity. No two threads touch the same tensor.
 *
 * Compile with -fopenmp to enable. Without it, these run serially.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Sweep all horizontal bonds with gate G (Red-Black parallel) */
void peps_gate_horizontal_all(PepsGrid *grid, const double *G_re, const double *G_im);

/* Sweep all vertical bonds with gate G (Red-Black parallel) */
void peps_gate_vertical_all(PepsGrid *grid, const double *G_re, const double *G_im);

/* Apply 1-site gate to ALL sites (trivially parallel) */
void peps_gate_1site_all(PepsGrid *grid, const double *U_re, const double *U_im);

/* Full Trotter step: horizontal-all + vertical-all */
void peps_trotter_step(PepsGrid *grid, const double *G_re, const double *G_im);

#endif /* PEPS_OVERLAY_H */
