/*
 * peps3d_overlay.c — 3D Tensor Network: Register-Based SVD Engine
 *
 * D=6 native (SU(6)), bond dimension χ=3 per axis (6 axes).
 * Simple-update with Jacobi SVD for proper 2-site gate application.
 * All tensor data stored in registers — temporary dense buffers for SVD.
 */

#include "peps3d_overlay.h"
#include "tensor_svd.h"
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define TNS3D_SVDDIM (TNS3D_D * TNS3D_CHI * TNS3D_CHI)  /* 54 at χ=3 */

/* ═══════════════ GRID ACCESS ═══════════════ */

static inline Tns3dTensor *tns3d_site(Tns3dGrid *g, int x, int y, int z)
{ return &g->tensors[(z * g->Ly + y) * g->Lx + x]; }

static inline Tns3dBondWeight *tns3d_xbond(Tns3dGrid *g, int x, int y, int z)
{ return &g->x_bonds[(z * g->Ly + y) * (g->Lx - 1) + x]; }

static inline Tns3dBondWeight *tns3d_ybond(Tns3dGrid *g, int x, int y, int z)
{ return &g->y_bonds[(z * (g->Ly - 1) + y) * g->Lx + x]; }

static inline Tns3dBondWeight *tns3d_zbond(Tns3dGrid *g, int x, int y, int z)
{ return &g->z_bonds[(z * g->Ly + y) * g->Lx + x]; }

static inline int tns3d_flat(Tns3dGrid *g, int x, int y, int z)
{ return (z * g->Ly + y) * g->Lx + x; }

/* ═══════════════ REGISTER DENSE I/O ═══════════════ */

static void tns3d_reg_read(Tns3dGrid *g, int site, double *T_re, double *T_im)
{
    int reg = g->site_reg[site];
    size_t tsz = (size_t)TNS3D_TSIZ;
    memset(T_re, 0, tsz * sizeof(double));
    memset(T_im, 0, tsz * sizeof(double));
    if (reg < 0 || !g->eng) return;

    QuhitRegister *r = &g->eng->registers[reg];
    for (uint32_t e = 0; e < r->num_nonzero; e++) {
        uint64_t bs = r->entries[e].basis_state;
        if (bs < tsz) {
            T_re[bs] = r->entries[e].amp_re;
            T_im[bs] = r->entries[e].amp_im;
        }
    }
}

static void tns3d_reg_write(Tns3dGrid *g, int site,
                            const double *T_re, const double *T_im)
{
    int reg = g->site_reg[site];
    if (reg < 0 || !g->eng) return;
    size_t tsz = (size_t)TNS3D_TSIZ;

    g->eng->registers[reg].num_nonzero = 0;
    for (size_t bs = 0; bs < tsz; bs++) {
        if (T_re[bs]*T_re[bs] + T_im[bs]*T_im[bs] > 1e-30)
            quhit_reg_sv_set(g->eng, reg, (uint64_t)bs, T_re[bs], T_im[bs]);
    }
}

/* ═══════════════ LIFECYCLE ═══════════════ */

Tns3dGrid *tns3d_init(int Lx, int Ly, int Lz)
{
    Tns3dGrid *g = (Tns3dGrid *)calloc(1, sizeof(Tns3dGrid));
    g->Lx = Lx; g->Ly = Ly; g->Lz = Lz;
    int N = Lx * Ly * Lz;

    g->tensors = (Tns3dTensor *)calloc(N, sizeof(Tns3dTensor));

    g->x_bonds = (Tns3dBondWeight *)calloc(Lz * Ly * (Lx - 1), sizeof(Tns3dBondWeight));
    g->y_bonds = (Tns3dBondWeight *)calloc(Lz * (Ly - 1) * Lx, sizeof(Tns3dBondWeight));
    g->z_bonds = (Tns3dBondWeight *)calloc((Lz - 1) * Ly * Lx, sizeof(Tns3dBondWeight));

    for (int i = 0; i < Lz * Ly * (Lx - 1); i++)
        for (int s = 0; s < TNS3D_CHI; s++) g->x_bonds[i].w[s] = 1.0;
    for (int i = 0; i < Lz * (Ly - 1) * Lx; i++)
        for (int s = 0; s < TNS3D_CHI; s++) g->y_bonds[i].w[s] = 1.0;
    for (int i = 0; i < (Lz - 1) * Ly * Lx; i++)
        for (int s = 0; s < TNS3D_CHI; s++) g->z_bonds[i].w[s] = 1.0;

    g->eng = (QuhitEngine *)calloc(1, sizeof(QuhitEngine));
    quhit_engine_init(g->eng);

    g->q_phys = (uint32_t *)calloc(N, sizeof(uint32_t));
    for (int i = 0; i < N; i++)
        g->q_phys[i] = quhit_init_basis(g->eng, 0);

    g->site_reg = (int *)calloc(N, sizeof(int));
    for (int i = 0; i < N; i++) {
        g->site_reg[i] = quhit_reg_init(g->eng, (uint64_t)i, 7, TNS3D_D);
        if (g->site_reg[i] >= 0) {
            g->eng->registers[g->site_reg[i]].bulk_rule = 0;
            quhit_reg_sv_set(g->eng, g->site_reg[i], 0, 1.0, 0.0);
        }
        g->tensors[i].reg_idx = g->site_reg[i];
    }

    return g;
}

void tns3d_free(Tns3dGrid *g)
{
    if (!g) return;
    free(g->tensors);
    free(g->x_bonds);
    free(g->y_bonds);
    free(g->z_bonds);
    if (g->eng) {
        quhit_engine_destroy(g->eng);
        free(g->eng);
    }
    free(g->q_phys);
    free(g->site_reg);
    free(g);
}

/* ═══════════════ STATE INITIALIZATION ═══════════════ */

void tns3d_set_product_state(Tns3dGrid *g, int x, int y, int z,
                             const double *amps_re, const double *amps_im)
{
    int site = tns3d_flat(g, x, y, z);
    int reg = g->site_reg[site];
    if (reg < 0) return;

    g->eng->registers[reg].num_nonzero = 0;
    for (int k = 0; k < TNS3D_D; k++) {
        if (amps_re[k] * amps_re[k] + amps_im[k] * amps_im[k] > 1e-30)
            quhit_reg_sv_set(g->eng, reg, (uint64_t)k, amps_re[k], amps_im[k]);
    }
}

/* ═══════════════ 1-SITE GATE ═══════════════ */

void tns3d_gate_1site(Tns3dGrid *g, int x, int y, int z,
                      const double *U_re, const double *U_im)
{
    int site = tns3d_flat(g, x, y, z);
    if (g->eng && g->site_reg)
        quhit_reg_apply_unitary_pos(g->eng, g->site_reg[site], 0, U_re, U_im);
    if (g->eng && g->q_phys)
        quhit_apply_unitary(g->eng, g->q_phys[site], U_re, U_im);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * GENERIC 3D 2-SITE GATE WITH SVD
 *
 * shared_axis: 0=x(r/l), 1=y(d/u), 2=z(b/f)
 * For each axis, the 6 remaining non-shared bonds are absorbed as environment.
 * SVD dim = D × χ² (physical × 2 kept free bonds)
 *
 * 7-index tensor: T[k, u, d, l, r, f, b]
 * Index positions: k=0, u=1, d=2, l=3, r=4, f=5, b=6
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void tns3d_gate_2site_generic(Tns3dGrid *g,
                                     int sA, int sB,
                                     Tns3dBondWeight *shared_bw,
                                     int shared_axis,
                                     const double *G_re, const double *G_im)
{
    int D = TNS3D_D, chi = TNS3D_CHI;
    int svddim = D * chi * chi;
    size_t tsz = (size_t)TNS3D_TSIZ;

    /* Read tensors */
    double *TA_re = (double *)calloc(tsz, sizeof(double));
    double *TA_im = (double *)calloc(tsz, sizeof(double));
    double *TB_re = (double *)calloc(tsz, sizeof(double));
    double *TB_im = (double *)calloc(tsz, sizeof(double));
    tns3d_reg_read(g, sA, TA_re, TA_im);
    tns3d_reg_read(g, sB, TB_re, TB_im);

    /* Build Θ matrix by contracting over shared bond + absorbing env bonds.
     * For axis X: shared = r_A = l_B. Row = (kA, u, f). Col = (kB, d, b).
     * For axis Y: shared = d_A = u_B. Row = (kA, l, f). Col = (kB, r, b).
     * For axis Z: shared = b_A = f_B. Row = (kA, u, l). Col = (kB, d, r).
     * The 4 non-shared, non-row/col bonds are summed over with env weights. */

    size_t svd2 = (size_t)svddim * svddim;
    double *Th_re = (double *)calloc(svd2, sizeof(double));
    double *Th_im = (double *)calloc(svd2, sizeof(double));

    /* For simplicity with χ=3, iterate over all index combinations.
     * Total iterations: D² × χ^12 = 36 × 531441 ≈ 19M — manageable for χ=3. */
    for (int kA = 0; kA < D; kA++)
     for (int kB = 0; kB < D; kB++)
      for (int uA = 0; uA < chi; uA++)
       for (int dA = 0; dA < chi; dA++)
        for (int lA = 0; lA < chi; lA++)
         for (int rA = 0; rA < chi; rA++)
          for (int fA = 0; fA < chi; fA++)
           for (int bA = 0; bA < chi; bA++) {
               /* Determine shared bond value and counterpart indices */
               int shared_val, uB_val, dB_val, lB_val, rB_val, fB_val, bB_val;

               if (shared_axis == 0) {
                   /* X-axis: r_A = l_B = rA */
                   shared_val = rA;
                   /* B indices: iterate over non-shared */
                   for (int dB = 0; dB < chi; dB++)
                    for (int rB = 0; rB < chi; rB++)
                     for (int uB = 0; uB < chi; uB++)
                      for (int fB = 0; fB < chi; fB++)
                       for (int bB = 0; bB < chi; bB++) {
                           int idxA = T3D_IDX(kA, uA, dA, lA, rA, fA, bA);
                           int idxB = T3D_IDX(kB, uB, dB, shared_val, rB, fB, bB);
                           double ar = TA_re[idxA], ai = TA_im[idxA];
                           double br = TB_re[idxB] * shared_bw->w[shared_val];
                           double bi = TB_im[idxB] * shared_bw->w[shared_val];
                           if ((ar*ar+ai*ai) < 1e-30 || (br*br+bi*bi) < 1e-30) continue;
                           /* Row = (kA, uA, fA), Col = (kB, dB, bB) */
                           int row = kA * chi * chi + uA * chi + fA;
                           int col = kB * chi * chi + dB * chi + bB;
                           Th_re[row * svddim + col] += ar*br - ai*bi;
                           Th_im[row * svddim + col] += ar*bi + ai*br;
                       }
               } else if (shared_axis == 1) {
                   /* Y-axis: d_A = u_B = dA */
                   shared_val = dA;
                   for (int uB_dummy = 0; uB_dummy < 1; uB_dummy++) {
                       for (int dB = 0; dB < chi; dB++)
                        for (int lB = 0; lB < chi; lB++)
                         for (int rB = 0; rB < chi; rB++)
                          for (int fB = 0; fB < chi; fB++)
                           for (int bB = 0; bB < chi; bB++) {
                               int idxA = T3D_IDX(kA, uA, dA, lA, rA, fA, bA);
                               int idxB = T3D_IDX(kB, shared_val, dB, lB, rB, fB, bB);
                               double ar = TA_re[idxA], ai = TA_im[idxA];
                               double br = TB_re[idxB] * shared_bw->w[shared_val];
                               double bi = TB_im[idxB] * shared_bw->w[shared_val];
                               if ((ar*ar+ai*ai) < 1e-30 || (br*br+bi*bi) < 1e-30) continue;
                               int row = kA * chi * chi + lA * chi + fA;
                               int col = kB * chi * chi + rB * chi + bB;
                               Th_re[row * svddim + col] += ar*br - ai*bi;
                               Th_im[row * svddim + col] += ar*bi + ai*br;
                           }
                   }
               } else {
                   /* Z-axis: b_A = f_B = bA */
                   shared_val = bA;
                   for (int fB_dummy = 0; fB_dummy < 1; fB_dummy++) {
                       for (int uB = 0; uB < chi; uB++)
                        for (int dB = 0; dB < chi; dB++)
                         for (int lB = 0; lB < chi; lB++)
                          for (int rB = 0; rB < chi; rB++)
                           for (int bB = 0; bB < chi; bB++) {
                               int idxA = T3D_IDX(kA, uA, dA, lA, rA, fA, bA);
                               int idxB = T3D_IDX(kB, uB, dB, lB, rB, shared_val, bB);
                               double ar = TA_re[idxA], ai = TA_im[idxA];
                               double br = TB_re[idxB] * shared_bw->w[shared_val];
                               double bi = TB_im[idxB] * shared_bw->w[shared_val];
                               if ((ar*ar+ai*ai) < 1e-30 || (br*br+bi*bi) < 1e-30) continue;
                               int row = kA * chi * chi + uA * chi + lA;
                               int col = kB * chi * chi + dB * chi + rB;
                               Th_re[row * svddim + col] += ar*br - ai*bi;
                               Th_im[row * svddim + col] += ar*bi + ai*br;
                           }
                   }
               }
           }

    /* Apply gate */
    double *Th2_re = (double *)calloc(svd2, sizeof(double));
    double *Th2_im = (double *)calloc(svd2, sizeof(double));
    int D2 = D * D;

    for (int kAp = 0; kAp < D; kAp++)
     for (int kBp = 0; kBp < D; kBp++) {
         int gr = kAp * D + kBp;
         for (int kA = 0; kA < D; kA++)
          for (int kB = 0; kB < D; kB++) {
              int gc = kA * D + kB;
              double gre = G_re[gr * D2 + gc];
              double gim = G_im[gr * D2 + gc];
              if (fabs(gre) < 1e-30 && fabs(gim) < 1e-30) continue;

              for (int i1 = 0; i1 < chi; i1++)
               for (int i2 = 0; i2 < chi; i2++) {
                   int dst_row = kAp * chi * chi + i1 * chi + i2;
                   int src_row = kA * chi * chi + i1 * chi + i2;
                   for (int j1 = 0; j1 < chi; j1++)
                    for (int j2 = 0; j2 < chi; j2++) {
                        int dst_col = kBp * chi * chi + j1 * chi + j2;
                        int src_col = kB * chi * chi + j1 * chi + j2;
                        double tr = Th_re[src_row * svddim + src_col];
                        double ti = Th_im[src_row * svddim + src_col];
                        Th2_re[dst_row * svddim + dst_col] += gre*tr - gim*ti;
                        Th2_im[dst_row * svddim + dst_col] += gre*ti + gim*tr;
                    }
               }
          }
     }

    free(Th_re); free(Th_im);

    /* SVD */
    double *U_re  = (double *)calloc((size_t)svddim * chi, sizeof(double));
    double *U_im  = (double *)calloc((size_t)svddim * chi, sizeof(double));
    double *sig   = (double *)calloc(chi, sizeof(double));
    double *Vc_re = (double *)calloc((size_t)chi * svddim, sizeof(double));
    double *Vc_im = (double *)calloc((size_t)chi * svddim, sizeof(double));

    tsvd_truncated(Th2_re, Th2_im, svddim, svddim, chi,
                   U_re, U_im, sig, Vc_re, Vc_im);

    free(Th2_re); free(Th2_im);

    /* Update shared bond weight */
    double sig_norm = 0;
    for (int s = 0; s < chi; s++) sig_norm += sig[s];
    if (sig_norm > 1e-30)
        for (int s = 0; s < chi; s++) shared_bw->w[s] = sig[s] / sig_norm;

    /* Write back: A' from U, B' from σ V† */
    memset(TA_re, 0, tsz * sizeof(double));
    memset(TA_im, 0, tsz * sizeof(double));
    memset(TB_re, 0, tsz * sizeof(double));
    memset(TB_im, 0, tsz * sizeof(double));

    for (int kA = 0; kA < D; kA++)
     for (int i1 = 0; i1 < chi; i1++)
      for (int i2 = 0; i2 < chi; i2++) {
          int row = kA * chi * chi + i1 * chi + i2;
          for (int g = 0; g < chi; g++) {
              double re = U_re[row * chi + g];
              double im = U_im[row * chi + g];
              if (re*re + im*im < 1e-30) continue;

              int idx;
              if (shared_axis == 0)      idx = T3D_IDX(kA, i1, 0, 0, g, i2, 0);
              else if (shared_axis == 1) idx = T3D_IDX(kA, 0, g, i1, 0, i2, 0);
              else                       idx = T3D_IDX(kA, i1, 0, i2, 0, 0, g);
              TA_re[idx] = re;
              TA_im[idx] = im;
          }
      }

    for (int kB = 0; kB < D; kB++)
     for (int j1 = 0; j1 < chi; j1++)
      for (int j2 = 0; j2 < chi; j2++) {
          int col = kB * chi * chi + j1 * chi + j2;
          for (int g = 0; g < chi; g++) {
              double s = sig[g];
              if (s < 1e-30) continue;
              double re = s * Vc_re[g * svddim + col];
              double im = s * Vc_im[g * svddim + col];
              if (re*re + im*im < 1e-30) continue;

              int idx;
              if (shared_axis == 0)      idx = T3D_IDX(kB, 0, j1, g, 0, 0, j2);
              else if (shared_axis == 1) idx = T3D_IDX(kB, g, j1, 0, j2, 0, 0);
              else                       idx = T3D_IDX(kB, j1, 0, 0, j2, g, 0);
              TB_re[idx] = re;
              TB_im[idx] = im;
          }
      }

    tns3d_reg_write(g, sA, TA_re, TA_im);
    tns3d_reg_write(g, sB, TB_re, TB_im);

    free(U_re); free(U_im);
    free(sig);
    free(Vc_re); free(Vc_im);
    free(TA_re); free(TA_im);
    free(TB_re); free(TB_im);

    /* Mirror to engine quhits */
    if (g->eng && g->q_phys)
        quhit_apply_cz(g->eng, g->q_phys[sA], g->q_phys[sB]);
}

/* ═══════════════ AXIS WRAPPERS ═══════════════ */

void tns3d_gate_x(Tns3dGrid *g, int x, int y, int z,
                  const double *G_re, const double *G_im)
{
    tns3d_gate_2site_generic(g,
        tns3d_flat(g, x, y, z), tns3d_flat(g, x+1, y, z),
        tns3d_xbond(g, x, y, z), 0, G_re, G_im);
}

void tns3d_gate_y(Tns3dGrid *g, int x, int y, int z,
                  const double *G_re, const double *G_im)
{
    tns3d_gate_2site_generic(g,
        tns3d_flat(g, x, y, z), tns3d_flat(g, x, y+1, z),
        tns3d_ybond(g, x, y, z), 1, G_re, G_im);
}

void tns3d_gate_z(Tns3dGrid *g, int x, int y, int z,
                  const double *G_re, const double *G_im)
{
    tns3d_gate_2site_generic(g,
        tns3d_flat(g, x, y, z), tns3d_flat(g, x, y, z+1),
        tns3d_zbond(g, x, y, z), 2, G_re, G_im);
}

/* ═══════════════ LOCAL DENSITY ═══════════════ */

void tns3d_local_density(Tns3dGrid *g, int x, int y, int z, double *probs)
{
    int site = tns3d_flat(g, x, y, z);
    int reg = g->site_reg[site];

    for (int k = 0; k < TNS3D_D; k++) probs[k] = 0;

    if (reg < 0 || !g->eng) { probs[0] = 1.0; return; }

    QuhitRegister *r = &g->eng->registers[reg];
    double total = 0;

    for (uint32_t e = 0; e < r->num_nonzero; e++) {
        /* Physical digit k is at position 0 in basis encoding */
        uint64_t bs = r->entries[e].basis_state;
        int k = (int)(bs / TNS3D_CHI6);  /* k = highest position */
        if (k >= TNS3D_D) continue;
        double re = r->entries[e].amp_re;
        double im = r->entries[e].amp_im;
        double p = re * re + im * im;
        probs[k] += p;
        total += p;
    }

    if (total > 1e-30)
        for (int k = 0; k < TNS3D_D; k++) probs[k] /= total;
    else
        probs[0] = 1.0;
}

/* ═══════════════ BATCH GATE APPLICATION ═══════════════ */

void tns3d_gate_x_all(Tns3dGrid *g, const double *G_re, const double *G_im)
{
    if (g->Lx < 2) return;

    for (int parity = 0; parity < 2; parity++) {
        #ifdef _OPENMP
        #pragma omp parallel for collapse(3) schedule(dynamic)
        #endif
        for (int z = 0; z < g->Lz; z++)
         for (int y = 0; y < g->Ly; y++)
          for (int xh = 0; xh < (g->Lx + 1) / 2; xh++) {
              int x = xh * 2 + parity;
              if (x < g->Lx - 1) tns3d_gate_x(g, x, y, z, G_re, G_im);
          }
    }
}

void tns3d_gate_y_all(Tns3dGrid *g, const double *G_re, const double *G_im)
{
    if (g->Ly < 2) return;

    for (int parity = 0; parity < 2; parity++) {
        #ifdef _OPENMP
        #pragma omp parallel for collapse(3) schedule(dynamic)
        #endif
        for (int z = 0; z < g->Lz; z++)
         for (int yh = 0; yh < (g->Ly + 1) / 2; yh++)
          for (int x = 0; x < g->Lx; x++) {
              int y = yh * 2 + parity;
              if (y < g->Ly - 1) tns3d_gate_y(g, x, y, z, G_re, G_im);
          }
    }
}

void tns3d_gate_z_all(Tns3dGrid *g, const double *G_re, const double *G_im)
{
    if (g->Lz < 2) return;

    for (int parity = 0; parity < 2; parity++) {
        #ifdef _OPENMP
        #pragma omp parallel for collapse(3) schedule(dynamic)
        #endif
        for (int zh = 0; zh < (g->Lz + 1) / 2; zh++)
         for (int y = 0; y < g->Ly; y++)
          for (int x = 0; x < g->Lx; x++) {
              int z = zh * 2 + parity;
              if (z < g->Lz - 1) tns3d_gate_z(g, x, y, z, G_re, G_im);
          }
    }
}

void tns3d_gate_1site_all(Tns3dGrid *g, const double *U_re, const double *U_im)
{
    #ifdef _OPENMP
    #pragma omp parallel for collapse(3) schedule(static)
    #endif
    for (int z = 0; z < g->Lz; z++)
     for (int y = 0; y < g->Ly; y++)
      for (int x = 0; x < g->Lx; x++)
          tns3d_gate_1site(g, x, y, z, U_re, U_im);
}

void tns3d_trotter_step(Tns3dGrid *g, const double *G_re, const double *G_im)
{
    tns3d_gate_x_all(g, G_re, G_im);
    tns3d_gate_y_all(g, G_re, G_im);
    tns3d_gate_z_all(g, G_re, G_im);
}
