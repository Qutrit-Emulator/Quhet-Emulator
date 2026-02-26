/*
 * peps4d_overlay.h — 4D Tensor Network State (TNS) for Hypercubic Lattices
 *
 * Pure Magic Pointer implementation — no classical tensor arrays.
 * Each site's register holds a 9-qudit state |k,u,d,l,r,f,b,i,o⟩.
 * RAM-agnostic: O(1) per site regardless of χ.
 *
 * WORLD FIRST: 4-dimensional PEPS tensor network on consumer hardware.
 */

#ifndef PEPS4D_OVERLAY_H
#define PEPS4D_OVERLAY_H

#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ═══════════════════════════════════════════════════════════════════════════════
 * CONSTANTS
 * ═══════════════════════════════════════════════════════════════════════════════ */

#define TNS4D_D     6       /* Physical dimension (SU(6) native)          */
#define TNS4D_CHI   128ULL  /* Bond dimension per axis (8 bonds → χ³ feasible) */

/* Derived powers of χ — for basis encoding (ULL to prevent overflow) */
#define TNS4D_C2    (TNS4D_CHI * TNS4D_CHI)
#define TNS4D_C3    (TNS4D_CHI * TNS4D_CHI * TNS4D_CHI)
#define TNS4D_C4    (TNS4D_C2 * TNS4D_C2)
#define TNS4D_C5    (TNS4D_C2 * TNS4D_C3)
#define TNS4D_C6    (TNS4D_C3 * TNS4D_C3)
#define TNS4D_C7    (TNS4D_C3 * TNS4D_C4)
#define TNS4D_C8    (TNS4D_C4 * TNS4D_C4)
#define TNS4D_TSIZ  (TNS4D_D * TNS4D_C8)  /* D×χ⁸ max basis+1   */

/* 9-index tensor basis encoding: |k,u,d,l,r,f,b,i,o⟩
 *   k ∈ [0,D), bonds ∈ [0,χ)
 *
 * Register encodes (little-endian bond order):
 *   o + i*χ + b*χ² + f*χ³ + r*χ⁴ + l*χ⁵ + d*χ⁶ + u*χ⁷ + k*χ⁸
 *
 * Position 0 = o (least significant, W-out)
 * Position 1 = i (W-in)
 * Position 2 = b (Z-back)
 * Position 3 = f (Z-front)
 * Position 4 = r (X-right)
 * Position 5 = l (X-left)
 * Position 6 = d (Y-down)
 * Position 7 = u (Y-up)
 * Position 8 = k (physical, most significant)
 *
 * gate_1site operates at position 8 (physical index k)
 */
#define T4D_IDX(k,u,d,l,r,f,b,i,o) \
    ((uint64_t)(k)*TNS4D_C8 + (uint64_t)(u)*TNS4D_C7 + (uint64_t)(d)*TNS4D_C6 + \
     (uint64_t)(l)*TNS4D_C5 + (uint64_t)(r)*TNS4D_C4 + (uint64_t)(f)*TNS4D_C3 + \
     (uint64_t)(b)*TNS4D_C2 + (uint64_t)(i)*TNS4D_CHI + (uint64_t)(o))

/* ═══════════════════════════════════════════════════════════════════════════════
 * DATA STRUCTURES — Magic Pointer based
 * ═══════════════════════════════════════════════════════════════════════════════ */

#include "quhit_engine.h"

/* Lightweight tensor stub — register IS the tensor */
typedef struct {
    int reg_idx;
} Tns4dTensor;

typedef struct {
    double *w;  /* Heap-allocated: χ singular values on this bond */
} Tns4dBondWeight;

typedef struct {
    int Lx, Ly, Lz, Lw;
    Tns4dTensor *tensors;
    Tns4dBondWeight *x_bonds;
    Tns4dBondWeight *y_bonds;
    Tns4dBondWeight *z_bonds;
    Tns4dBondWeight *w_bonds;
    /* ── Magic Pointer integration ── */
    uint32_t *q_phys;
    QuhitEngine *eng;
    int *site_reg;
} Tns4dGrid;

/* ═══════════════════════════════════════════════════════════════════════════════
 * API
 * ═══════════════════════════════════════════════════════════════════════════════ */

Tns4dGrid *tns4d_init(int Lx, int Ly, int Lz, int Lw);
void tns4d_free(Tns4dGrid *grid);

void tns4d_set_product_state(Tns4dGrid *grid, int x, int y, int z, int w,
                             const double *amps_re, const double *amps_im);

void tns4d_gate_1site(Tns4dGrid *grid, int x, int y, int z, int w,
                      const double *U_re, const double *U_im);

void tns4d_gate_x(Tns4dGrid *grid, int x, int y, int z, int w,
                  const double *G_re, const double *G_im);
void tns4d_gate_y(Tns4dGrid *grid, int x, int y, int z, int w,
                  const double *G_re, const double *G_im);
void tns4d_gate_z(Tns4dGrid *grid, int x, int y, int z, int w,
                  const double *G_re, const double *G_im);
void tns4d_gate_w(Tns4dGrid *grid, int x, int y, int z, int w,
                  const double *G_re, const double *G_im);

void tns4d_local_density(Tns4dGrid *grid, int x, int y, int z, int w, double *probs);

/* Batch gates */
void tns4d_gate_x_all(Tns4dGrid *grid, const double *G_re, const double *G_im);
void tns4d_gate_y_all(Tns4dGrid *grid, const double *G_re, const double *G_im);
void tns4d_gate_z_all(Tns4dGrid *grid, const double *G_re, const double *G_im);
void tns4d_gate_w_all(Tns4dGrid *grid, const double *G_re, const double *G_im);
void tns4d_gate_1site_all(Tns4dGrid *grid, const double *U_re, const double *U_im);
void tns4d_normalize_site(Tns4dGrid *g, int x, int y, int z, int w);
void tns4d_trotter_step(Tns4dGrid *grid, const double *G_re, const double *G_im);

#endif /* PEPS4D_OVERLAY_H */
