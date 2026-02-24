/*
 * peps6d_overlay.h — 6D Tensor Network State
 *
 * D=6 in 6 spatial dimensions. The number mirrors itself.
 *
 * Each site: 13-index tensor |k, b0..b11⟩
 *   - k ∈ [0, D=6)         physical (SU(6))
 *   - 12 bond indices       2 per axis (X, Y, Z, W, V, U)
 *   - Each bond ∈ [0, χ=2)
 *
 * Per-site basis: D × χ¹² = 6 × 4096 = 24,576
 * Sparsity keeps NNZ ≪ 4,096 per site
 *
 * Grid sizes:
 *   2^6 =  64 quhits → 6^64  ≈ 10^50   (20 orders beyond Willow)
 *   3^6 = 729 quhits → 6^729 ≈ 10^567  (beyond all physical reference)
 */

#ifndef PEPS6D_OVERLAY_H
#define PEPS6D_OVERLAY_H

#include "quhit_engine.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TNS6D_D    6
#define TNS6D_CHI  2

/* Cumulative bond powers for 13-index encoding */
#define TNS6D_C1   2
#define TNS6D_C2   4
#define TNS6D_C3   8
#define TNS6D_C4   16
#define TNS6D_C5   32
#define TNS6D_C6   64
#define TNS6D_C7   128
#define TNS6D_C8   256
#define TNS6D_C9   512
#define TNS6D_C10  1024
#define TNS6D_C11  2048
#define TNS6D_C12  4096
#define TNS6D_TSIZ (TNS6D_D * TNS6D_C12)  /* 24,576 */

typedef struct { int reg_idx; } Tns6dTensor;
typedef struct { double w[TNS6D_CHI]; } Tns6dBondWeight;

typedef struct {
    int Lx, Ly, Lz, Lw, Lv, Lu;
    Tns6dTensor     *tensors;
    Tns6dBondWeight *x_bonds, *y_bonds, *z_bonds, *w_bonds, *v_bonds, *u_bonds;
    QuhitEngine     *eng;
    uint32_t        *q_phys;
    int             *site_reg;
} Tns6dGrid;

Tns6dGrid *tns6d_init(int Lx, int Ly, int Lz, int Lw, int Lv, int Lu);
void       tns6d_free(Tns6dGrid *g);

void tns6d_gate_1site(Tns6dGrid *g, int x, int y, int z, int w, int v, int u,
                      const double *U_re, const double *U_im);
void tns6d_gate_1site_all(Tns6dGrid *g, const double *U_re, const double *U_im);

void tns6d_gate_x(Tns6dGrid *g, int x, int y, int z, int w, int v, int u,
                  const double *G_re, const double *G_im);
void tns6d_gate_y(Tns6dGrid *g, int x, int y, int z, int w, int v, int u,
                  const double *G_re, const double *G_im);
void tns6d_gate_z(Tns6dGrid *g, int x, int y, int z, int w, int v, int u,
                  const double *G_re, const double *G_im);
void tns6d_gate_w(Tns6dGrid *g, int x, int y, int z, int w, int v, int u,
                  const double *G_re, const double *G_im);
void tns6d_gate_v(Tns6dGrid *g, int x, int y, int z, int w, int v, int u,
                  const double *G_re, const double *G_im);
void tns6d_gate_u(Tns6dGrid *g, int x, int y, int z, int w, int v, int u,
                  const double *G_re, const double *G_im);

void tns6d_trotter_step(Tns6dGrid *g, const double *G_re, const double *G_im);
void tns6d_normalize_site(Tns6dGrid *g, int x, int y, int z, int w, int v, int u);
void tns6d_local_density(Tns6dGrid *g, int x, int y, int z, int w, int v, int u,
                         double *probs);

#endif
