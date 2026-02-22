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
#define TNS3D_CHI   6       /* Bond dimension — matches D, full register usage */

/* Derived powers of D — for basis encoding (matches register dim=D) */
#define TNS3D_D2    (TNS3D_D * TNS3D_D)                    /* 36  */
#define TNS3D_D3    (TNS3D_D * TNS3D_D * TNS3D_D)          /* 216 */
#define TNS3D_D4    (TNS3D_D2 * TNS3D_D2)                  /* 1296 */
#define TNS3D_D5    (TNS3D_D2 * TNS3D_D3)                  /* 7776 */
#define TNS3D_D6    (TNS3D_D3 * TNS3D_D3)                  /* 46656 */
#define TNS3D_D7    (TNS3D_D6 * TNS3D_D)                   /* 279936 */
#define TNS3D_TSIZ  TNS3D_D7  /* Full register basis space */

/* 7-index tensor basis encoding: |k,u,d,l,r,f,b⟩
 * Register encodes: b + f*D + r*D² + l*D³ + d*D⁴ + u*D⁵ + k*D⁶
 * Position 0 = b (least sig), position 6 = k (most sig)
 * gate_1site operates at position 6 (physical index k) */
#define T3D_IDX(k,u,d,l,r,f,b) \
    ((k)*TNS3D_D6 + (u)*TNS3D_D5 + (d)*TNS3D_D4 + \
     (l)*TNS3D_D3 + (r)*TNS3D_D2 + (f)*TNS3D_D + (b))

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
