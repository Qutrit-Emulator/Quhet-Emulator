/*
 * peps5d_overlay.h — 5D Tensor Network State (WORLD FIRST)
 *
 * Extends the 4D PEPS to 5 spatial dimensions.
 * Each site carries an 11-index tensor: |k, b0..b9⟩
 *   - k ∈ [0, D=6)         physical (SU(6) native)
 *   - 10 bond indices       2 per axis (X, Y, Z, W, V)
 *   - Each bond ∈ [0, χ=2)
 *
 * Per-site basis: D × χ¹⁰ = 6 × 1024 = 6,144
 * Sparsity keeps actual NNZ ≪ 6,144
 *
 * A 3×3×3×3×3 grid = 243 quhits → 6^243 ≈ 10^189 states
 * For reference, the observable universe has ~10^80 atoms.
 */

#ifndef PEPS5D_OVERLAY_H
#define PEPS5D_OVERLAY_H

#include "quhit_engine.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ═══════════════ CONSTANTS ═══════════════ */

#define TNS5D_D    6       /* Physical dimension (SU(6)) */
#define TNS5D_CHI  128ULL  /* Bond dimension per axis */

/* Cumulative products for 11-index encoding (derived from CHI):
 * basis = k*C10 + b9*C9 + b8*C8 + ... + b1*C1 + b0
 * NOTE: C10 = 128^10 = 2^70, requires basis_t (128-bit)
 */
#define TNS5D_C1   ((unsigned __int128)TNS5D_CHI)
#define TNS5D_C2   (TNS5D_C1 * TNS5D_CHI)
#define TNS5D_C3   (TNS5D_C2 * TNS5D_CHI)
#define TNS5D_C4   (TNS5D_C3 * TNS5D_CHI)
#define TNS5D_C5   (TNS5D_C4 * TNS5D_CHI)
#define TNS5D_C6   (TNS5D_C5 * TNS5D_CHI)
#define TNS5D_C7   (TNS5D_C6 * TNS5D_CHI)
#define TNS5D_C8   (TNS5D_C7 * TNS5D_CHI)
#define TNS5D_C9   (TNS5D_C8 * TNS5D_CHI)
#define TNS5D_C10  (TNS5D_C9 * TNS5D_CHI)
#define TNS5D_TSIZ (TNS5D_D * TNS5D_C10)

/* ═══════════════ DATA STRUCTURES ═══════════════ */

typedef struct {
    int reg_idx;
} Tns5dTensor;

typedef struct {
    double *w;     /* Heap-allocated: χ singular values */
    double *V_re;  /* Cached eigenvectors (n×n, NULL until first SVD) */
    double *V_im;
    int     V_n;   /* Dimension of cached V (0 = no cache) */
} Tns5dBondWeight;

typedef struct {
    int Lx, Ly, Lz, Lw, Lv;
    Tns5dTensor    *tensors;
    Tns5dBondWeight *x_bonds, *y_bonds, *z_bonds, *w_bonds, *v_bonds;
    QuhitEngine    *eng;
    uint32_t       *q_phys;
    int            *site_reg;
} Tns5dGrid;

/* ═══════════════ API ═══════════════ */

Tns5dGrid *tns5d_init(int Lx, int Ly, int Lz, int Lw, int Lv);
void       tns5d_free(Tns5dGrid *g);

void tns5d_set_product_state(Tns5dGrid *g, int x, int y, int z, int w, int v,
                             const double *amps_re, const double *amps_im);

void tns5d_gate_1site(Tns5dGrid *g, int x, int y, int z, int w, int v,
                      const double *U_re, const double *U_im);

void tns5d_gate_x(Tns5dGrid *g, int x, int y, int z, int w, int v,
                  const double *G_re, const double *G_im);
void tns5d_gate_y(Tns5dGrid *g, int x, int y, int z, int w, int v,
                  const double *G_re, const double *G_im);
void tns5d_gate_z(Tns5dGrid *g, int x, int y, int z, int w, int v,
                  const double *G_re, const double *G_im);
void tns5d_gate_w(Tns5dGrid *g, int x, int y, int z, int w, int v,
                  const double *G_re, const double *G_im);
void tns5d_gate_v(Tns5dGrid *g, int x, int y, int z, int w, int v,
                  const double *G_re, const double *G_im);

void tns5d_gate_1site_all(Tns5dGrid *g, const double *U_re, const double *U_im);
void tns5d_gate_x_all(Tns5dGrid *g, const double *G_re, const double *G_im);
void tns5d_gate_y_all(Tns5dGrid *g, const double *G_re, const double *G_im);
void tns5d_gate_z_all(Tns5dGrid *g, const double *G_re, const double *G_im);
void tns5d_gate_w_all(Tns5dGrid *g, const double *G_re, const double *G_im);
void tns5d_gate_v_all(Tns5dGrid *g, const double *G_re, const double *G_im);

void tns5d_trotter_step(Tns5dGrid *g, const double *G_re, const double *G_im);
void tns5d_normalize_site(Tns5dGrid *g, int x, int y, int z, int w, int v);
void tns5d_local_density(Tns5dGrid *g, int x, int y, int z, int w, int v,
                         double *probs);

#endif /* PEPS5D_OVERLAY_H */
