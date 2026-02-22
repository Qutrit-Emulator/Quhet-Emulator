/*
 * peps3d_overlay.h — 3D Tensor Network State (TNS) for Cubic Lattices
 *
 * Extends PEPS to three spatial dimensions.
 * Each site has a 7-index tensor T[k][u][d][l][r][f][b]:
 *   k = physical index (0..D-1)
 *   u,d = up/down        (y-axis, 0..χ-1)
 *   l,r = left/right     (x-axis, 0..χ-1)
 *   f,b = front/back     (z-axis, 0..χ-1)
 *
 * Simple update algorithm for 3D gate application.
 * Bond weights (singular values) stored on each bond.
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
#define TNS3D_CHI   3       /* Bond dimension (χ⁶ storage per site)       */
#define TNS3D_D2    (TNS3D_D * TNS3D_D)   /* 36: joint physical space   */

/* Derived powers of χ */
#define TNS3D_CHI2  (TNS3D_CHI * TNS3D_CHI)                          /*   4  */
#define TNS3D_CHI3  (TNS3D_CHI * TNS3D_CHI * TNS3D_CHI)              /*   8  */
#define TNS3D_CHI4  (TNS3D_CHI2 * TNS3D_CHI2)                        /*  16  */
#define TNS3D_CHI5  (TNS3D_CHI2 * TNS3D_CHI3)                        /*  32  */
#define TNS3D_CHI6  (TNS3D_CHI3 * TNS3D_CHI3)                        /*  64  */

#define TNS3D_TSIZ  (TNS3D_D * TNS3D_CHI6)     /* entries/site: 6×64=384    */
#define TNS3D_SVDDIM (TNS3D_D * TNS3D_CHI5)    /* SVD matrix dim: 6×32=192  */

/* 7-index tensor access: T[k][u][d][l][r][f][b] */
#define T3D_IDX(k,u,d,l,r,f,b) \
    ((k)*TNS3D_CHI6 + (u)*TNS3D_CHI5 + (d)*TNS3D_CHI4 + \
     (l)*TNS3D_CHI3 + (r)*TNS3D_CHI2 + (f)*TNS3D_CHI + (b))

/* ═══════════════════════════════════════════════════════════════════════════════
 * DATA STRUCTURES
 * ═══════════════════════════════════════════════════════════════════════════════ */

#include "quhit_engine.h"

typedef struct {
    double re[TNS3D_TSIZ];   /* 384 real parts    */
    double im[TNS3D_TSIZ];   /* 384 imaginary     */
} Tns3dTensor;               /* sizeof = 6144 bytes = 6 KB per site */

typedef struct {
    double w[TNS3D_CHI];     /* Singular values on this bond */
} Tns3dBondWeight;

typedef struct {
    int Lx, Ly, Lz;                  /* Grid dimensions                   */
    Tns3dTensor *tensors;             /* [Lz * Ly * Lx] site tensors       */
    Tns3dBondWeight *x_bonds;         /* X-bonds: [Lz * Ly * (Lx-1)]      */
    Tns3dBondWeight *y_bonds;         /* Y-bonds: [Lz * (Ly-1) * Lx]      */
    Tns3dBondWeight *z_bonds;         /* Z-bonds: [(Lz-1) * Ly * Lx]      */
    /* ── Magic Pointer integration ── */
    uint32_t *q_phys;                 /* [Lz*Ly*Lx] per-site quhit IDs    */
    QuhitEngine *eng;                 /* HexState Engine reference         */
} Tns3dGrid;

/* ═══════════════════════════════════════════════════════════════════════════════
 * API
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Lifecycle */
Tns3dGrid *tns3d_init(int Lx, int Ly, int Lz);
void tns3d_free(Tns3dGrid *grid);

/* State initialization */
void tns3d_set_product_state(Tns3dGrid *grid, int x, int y, int z,
                             const double *amps_re, const double *amps_im);

/* Gate application (2-site gates along each axis) */
void tns3d_gate_x(Tns3dGrid *grid, int x, int y, int z,
                  const double *G_re, const double *G_im);
void tns3d_gate_y(Tns3dGrid *grid, int x, int y, int z,
                  const double *G_re, const double *G_im);
void tns3d_gate_z(Tns3dGrid *grid, int x, int y, int z,
                  const double *G_re, const double *G_im);

/* 1-site gate */
void tns3d_gate_1site(Tns3dGrid *grid, int x, int y, int z,
                      const double *U_re, const double *U_im);

/* Observables */
void tns3d_local_density(Tns3dGrid *grid, int x, int y, int z, double *probs);

/* ═══════════════════════════════════════════════════════════════════════════════
 * BATCH GATE APPLICATION (Red-Black Checkerboard Parallelism)
 *
 * Apply a 2-site gate to ALL bonds along an axis using a checkerboard pattern.
 * Even-parity bonds (x%2==0) are applied first, then odd-parity (x%2==1).
 * No two threads ever touch the same tensor.
 *
 * Compile with -fopenmp to enable multi-threaded execution.
 * Without OpenMP, these run serially (still correct, just sequential).
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Sweep all X-bonds with gate G (Red-Black parallel) */
void tns3d_gate_x_all(Tns3dGrid *grid, const double *G_re, const double *G_im);

/* Sweep all Y-bonds with gate G (Red-Black parallel) */
void tns3d_gate_y_all(Tns3dGrid *grid, const double *G_re, const double *G_im);

/* Sweep all Z-bonds with gate G (Red-Black parallel) */
void tns3d_gate_z_all(Tns3dGrid *grid, const double *G_re, const double *G_im);

/* Apply 1-site gate to ALL sites (trivially parallel — no shared data) */
void tns3d_gate_1site_all(Tns3dGrid *grid, const double *U_re, const double *U_im);

/* Full Trotter step: X-all + Y-all + Z-all with the same gate */
void tns3d_trotter_step(Tns3dGrid *grid, const double *G_re, const double *G_im);

#endif /* PEPS3D_OVERLAY_H */
