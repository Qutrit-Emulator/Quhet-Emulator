/*
 * mps_overlay.h — MPS Side-Channel for N-Body States
 *
 * Pure Magic Pointer implementation — no classical tensor arrays.
 * Tensors stored as QuhitRegisters (3 qudits: k, α, β).
 * RAM-agnostic: O(1) per site regardless of χ.
 */

#ifndef MPS_OVERLAY_H
#define MPS_OVERLAY_H

#include "quhit_engine.h"
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════════════
 * CONSTANTS
 * ═══════════════════════════════════════════════════════════════════════════════ */

#define MPS_CHI  1024
#ifndef MPS_PHYS
#define MPS_PHYS 6
#endif

/* Basis encoding: |k, α, β⟩ → k * χ² + α * χ + β */
#define MPS_TENSOR_SIZE (MPS_PHYS * MPS_CHI * MPS_CHI)

/* ═══════════════════════════════════════════════════════════════════════════════
 * TENSOR STORE — Magic Pointer based (register IS the tensor)
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    int reg_idx;   /* Index into engine's register array */
} MpsTensor;

extern MpsTensor   *mps_store;
extern int          mps_store_n;
extern QuhitEngine *mps_eng;   /* Engine reference for register access */

/* ═══════════════════════════════════════════════════════════════════════════════
 * TENSOR ACCESS  (via register — O(1))
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline void mps_write_tensor(int site, int k, int alpha, int beta,
                                    double re, double im)
{
    uint64_t basis = (uint64_t)k * (MPS_CHI * MPS_CHI) + alpha * MPS_CHI + beta;
    if (mps_eng && mps_store && mps_store[site].reg_idx >= 0)
        quhit_reg_sv_set(mps_eng, mps_store[site].reg_idx, basis, re, im);
}

static inline void mps_read_tensor(int site, int k, int alpha, int beta,
                                   double *re, double *im)
{
    uint64_t basis = (uint64_t)k * (MPS_CHI * MPS_CHI) + alpha * MPS_CHI + beta;
    if (mps_eng && mps_store && mps_store[site].reg_idx >= 0) {
        SV_Amplitude a = quhit_reg_sv_get(mps_eng, mps_store[site].reg_idx, basis);
        *re = a.re; *im = a.im;
    } else {
        *re = 0; *im = 0;
    }
}

static inline void mps_zero_site(int site)
{
    if (mps_eng && mps_store && mps_store[site].reg_idx >= 0)
        mps_eng->registers[mps_store[site].reg_idx].num_nonzero = 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * STATE MANAGEMENT API
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_overlay_init(QuhitEngine *eng, uint32_t *quhits, int n);
void mps_overlay_free(void);

void mps_overlay_write_w_state(QuhitEngine *eng, uint32_t *quhits, int n);
void mps_overlay_write_zero(QuhitEngine *eng, uint32_t *quhits, int n);

uint32_t mps_overlay_measure(QuhitEngine *eng, uint32_t *quhits, int n, int target_idx);

void mps_overlay_amplitude(QuhitEngine *eng, uint32_t *quhits, int n,
                           const uint32_t *basis, double *out_re, double *out_im);

/* ═══════════════════════════════════════════════════════════════════════════════
 * GATE LAYER
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_gate_1site(QuhitEngine *eng, uint32_t *quhits, int n,
                    int site, const double *U_re, const double *U_im);

void mps_gate_2site(QuhitEngine *eng, uint32_t *quhits, int n,
                    int site, const double *G_re, const double *G_im);

/* ═══════════════════════════════════════════════════════════════════════════════
 * GATE CONSTRUCTORS
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_build_dft6(double *U_re, double *U_im);
void mps_build_cz(double *G_re, double *G_im);
void mps_build_controlled_phase(double *G_re, double *G_im, int power);
void mps_build_hadamard2(double *U_re, double *U_im);

double mps_overlay_norm(QuhitEngine *eng, uint32_t *quhits, int n);

/* ═══════════════════════════════════════════════════════════════════════════════
 * CANONICAL SWEEP DIRECTION
 * ═══════════════════════════════════════════════════════════════════════════════ */

extern int mps_sweep_right;

/* Legacy full-chain renormalization (O(N) — for verification only) */
extern int mps_defer_renorm;
void mps_renormalize_chain(QuhitEngine *eng, uint32_t *quhits, int n);

/* ═══════════════════════════════════════════════════════════════════════════════
 * LAZY EVALUATION LAYER
 * ═══════════════════════════════════════════════════════════════════════════════ */

#include "lazy_stats.h"

#define MAX_LAZY_GATES 65536

/* Deferred gate descriptor */
typedef struct {
    int     type;
    int     site;
    double  U_re[MPS_PHYS * MPS_PHYS];
    double  U_im[MPS_PHYS * MPS_PHYS];
    double *G_re;
    double *G_im;
    int     applied;
} MpsDeferredGate;

/* Lazy chain: wraps mps_store with deferred gate queue */
typedef struct {
    QuhitEngine *eng;
    uint32_t    *quhits;
    int          n_sites;

    MpsDeferredGate *queue;
    int              queue_len;
    int              queue_cap;

    uint8_t  *site_allocated;
    uint8_t  *site_dirty;

    LazyStats stats;
} MpsLazyChain;

/* ── Lifecycle ── */
MpsLazyChain *mps_lazy_init(QuhitEngine *eng, uint32_t *quhits, int n);
void          mps_lazy_free(MpsLazyChain *lc);

/* ── Gate queuing (NO immediate application) ── */
void mps_lazy_gate_1site(MpsLazyChain *lc, int site,
                         const double *U_re, const double *U_im);
void mps_lazy_gate_2site(MpsLazyChain *lc, int site,
                         const double *G_re, const double *G_im);

/* ── Measurement (TRIGGERS materialization) ── */
uint32_t mps_lazy_measure(MpsLazyChain *lc, int target_idx);

/* ── Force-apply all pending gates (escape hatch) ── */
void mps_lazy_flush(MpsLazyChain *lc);

/* ── Finalize stats (call before reading lc->stats) ── */
void mps_lazy_finalize_stats(MpsLazyChain *lc);

/* ── Write initial state (marks site as allocated) ── */
void mps_lazy_write_tensor(MpsLazyChain *lc, int site, int k,
                           int alpha, int beta, double re, double im);
void mps_lazy_zero_site(MpsLazyChain *lc, int site);

#endif /* MPS_OVERLAY_H */
