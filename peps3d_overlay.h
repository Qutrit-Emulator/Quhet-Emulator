/*
 * peps3d_overlay.h — 3D Tensor Network State (TNS) for Cubic Lattices
 *
 * Pure Magic Pointer implementation — no classical tensor arrays.
 * Each site's register holds a 7-qudit state |k,u,d,l,r,f,b⟩.
 * RAM-agnostic: O(1) per site regardless of χ.
 */

#ifndef PEPS3D_OVERLAY_H
#define PEPS3D_OVERLAY_H

#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ═══════════════════════════════════════════════════════════════════════════════
 * CONSTANTS
 * ═══════════════════════════════════════════════════════════════════════════════ */

#define TNS3D_D     6       /* Physical dimension (SU(6) native)          */
#define TNS3D_CHI   12      /* Bond dimension per axis                    */

/* Derived powers of χ — for basis encoding (register dim=χ) */
#define TNS3D_C2    (TNS3D_CHI * TNS3D_CHI)                        /* 144    */
#define TNS3D_C3    (TNS3D_CHI * TNS3D_CHI * TNS3D_CHI)            /* 1728   */
#define TNS3D_C4    (TNS3D_C2 * TNS3D_C2)                          /* 20736  */
#define TNS3D_C5    (TNS3D_C2 * TNS3D_C3)                          /* 248832 */
#define TNS3D_C6    (TNS3D_C3 * TNS3D_C3)                          /* 2985984 */
#define TNS3D_TSIZ  (TNS3D_D * TNS3D_C6)  /* D×χ⁶ = max basis+1 */

/* 7-index tensor basis encoding: |k,u,d,l,r,f,b⟩
 * k ∈ [0,D), bonds ∈ [0,χ)
 * Register encodes: b + f*χ + r*χ² + l*χ³ + d*χ⁴ + u*χ⁵ + k*χ⁶
 * Position 0 = b (least sig), position 6 = k (most sig)
 * gate_1site operates at position 6 (physical index k) */
#define T3D_IDX(k,u,d,l,r,f,b) \
    ((k)*TNS3D_C6 + (u)*TNS3D_C5 + (d)*TNS3D_C4 + \
     (l)*TNS3D_C3 + (r)*TNS3D_C2 + (f)*TNS3D_CHI + (b))

/* ═══════════════════════════════════════════════════════════════════════════════
 * DATA STRUCTURES — Magic Pointer based
 * ═══════════════════════════════════════════════════════════════════════════════ */

#include "quhit_engine.h"

/* Lightweight tensor stub — register IS the tensor */
typedef struct {
    int reg_idx;
} Tns3dTensor;

typedef struct {
    double w[TNS3D_CHI];
} Tns3dBondWeight;

typedef struct {
    int Lx, Ly, Lz;
    Tns3dTensor *tensors;
    Tns3dBondWeight *x_bonds;
    Tns3dBondWeight *y_bonds;
    Tns3dBondWeight *z_bonds;
    /* ── Magic Pointer integration ── */
    uint32_t *q_phys;
    QuhitEngine *eng;
    int *site_reg;
} Tns3dGrid;

/* ═══════════════════════════════════════════════════════════════════════════════
 * API
 * ═══════════════════════════════════════════════════════════════════════════════ */

Tns3dGrid *tns3d_init(int Lx, int Ly, int Lz);
void tns3d_free(Tns3dGrid *grid);

void tns3d_set_product_state(Tns3dGrid *grid, int x, int y, int z,
                             const double *amps_re, const double *amps_im);

void tns3d_gate_x(Tns3dGrid *grid, int x, int y, int z,
                  const double *G_re, const double *G_im);
void tns3d_gate_y(Tns3dGrid *grid, int x, int y, int z,
                  const double *G_re, const double *G_im);
void tns3d_gate_z(Tns3dGrid *grid, int x, int y, int z,
                  const double *G_re, const double *G_im);
void tns3d_gate_1site(Tns3dGrid *grid, int x, int y, int z,
                      const double *U_re, const double *U_im);

void tns3d_local_density(Tns3dGrid *grid, int x, int y, int z, double *probs);

/* Batch gates */
void tns3d_gate_x_all(Tns3dGrid *grid, const double *G_re, const double *G_im);
void tns3d_gate_y_all(Tns3dGrid *grid, const double *G_re, const double *G_im);
void tns3d_gate_z_all(Tns3dGrid *grid, const double *G_re, const double *G_im);
void tns3d_gate_1site_all(Tns3dGrid *grid, const double *U_re, const double *U_im);
void tns3d_trotter_step(Tns3dGrid *grid, const double *G_re, const double *G_im);

#endif /* PEPS3D_OVERLAY_H */
