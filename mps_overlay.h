/*
 * mps_overlay.h — MPS Side-Channel for N-Body States
 *
 * χ=6 bond dimension, matching physical dimension D=6.
 * Every gate operation is EXACT — no SVD truncation loss.
 *
 * Storage: D×χ² = 216 complex entries = 3456 bytes per site.
 * Tensor store is dynamically allocated per chain.
 */

#ifndef MPS_OVERLAY_H
#define MPS_OVERLAY_H

#include "quhit_engine.h"
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════════════
 * CONSTANTS
 * ═══════════════════════════════════════════════════════════════════════════════ */

#define MPS_CHI  128
#define MPS_PHYS 6

/* Total entries per tensor: D × χ² = 216 complex numbers */
#define MPS_TENSOR_SIZE (MPS_PHYS * MPS_CHI * MPS_CHI)

/* ═══════════════════════════════════════════════════════════════════════════════
 * TENSOR STORE
 *
 * Dynamically allocated array of MpsTensor, one per MPS site.
 * Indexed by site position (0..N-1), NOT by engine pair_id.
 * mps_overlay_init allocates, mps_overlay_free releases.
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    double re[MPS_TENSOR_SIZE];
    double im[MPS_TENSOR_SIZE];
} MpsTensor;

extern MpsTensor *mps_store;
extern int         mps_store_n;

/* ═══════════════════════════════════════════════════════════════════════════════
 * TENSOR ACCESS  (by site index)
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline void mps_write_tensor(int site, int k, int alpha, int beta,
                                    double re, double im)
{
    int idx = k * (MPS_CHI * MPS_CHI) + alpha * MPS_CHI + beta;
    mps_store[site].re[idx] = re;
    mps_store[site].im[idx] = im;
}

static inline void mps_read_tensor(int site, int k, int alpha, int beta,
                                   double *re, double *im)
{
    int idx = k * (MPS_CHI * MPS_CHI) + alpha * MPS_CHI + beta;
    *re = mps_store[site].re[idx];
    *im = mps_store[site].im[idx];
}

static inline void mps_zero_site(int site)
{
    memset(mps_store[site].re, 0, sizeof(mps_store[site].re));
    memset(mps_store[site].im, 0, sizeof(mps_store[site].im));
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
 *
 * mps_sweep_right = 1 (default): L→R sweep.
 *   Left site  = U  (left-canonical).
 *   Right site = σ·V (gauge center moves right).
 *
 * mps_sweep_right = 0: R→L sweep.
 *   Left site  = U·σ (gauge center moves left).
 *   Right site = V   (right-canonical).
 *
 * O(1) renormalization: σ is rescaled so that Σσ²=1 at each SVD.
 * In canonical form this IS the global norm.  No O(N) trace needed.
 * ═══════════════════════════════════════════════════════════════════════════════ */

extern int mps_sweep_right;

/* Legacy full-chain renormalization (O(N) — for verification only) */
extern int mps_defer_renorm;
void mps_renormalize_chain(QuhitEngine *eng, uint32_t *quhits, int n);

/* ═══════════════════════════════════════════════════════════════════════════════
 * LAZY EVALUATION LAYER
 *
 * "Reality computes on demand."
 *
 * Gates are queued, not applied. Measurement triggers materialization
 * of only the gates needed for the measured site's environment.
 * Gates on unmeasured sites are NEVER applied.
 *
 * Gate fusion: consecutive 1-site gates on the same site are multiplied
 * into a single gate before application.
 *
 * Lazy allocation: sites that have never been written to stay as
 * implicit |0⟩ with no tensor allocated.
 * ═══════════════════════════════════════════════════════════════════════════════ */

#include "lazy_stats.h"

#define MAX_LAZY_GATES 65536

/* Deferred gate descriptor */
typedef struct {
    int     type;                   /* 0 = 1-site, 1 = 2-site                */
    int     site;                   /* Target site (for 2-site: left site)   */
    double  U_re[MPS_PHYS * MPS_PHYS];   /* 1-site: 6×6 matrix             */
    double  U_im[MPS_PHYS * MPS_PHYS];
    double *G_re;                   /* 2-site: 36×36 matrix (heap, owned)    */
    double *G_im;
    int     applied;                /* 1 = already materialized              */
} MpsDeferredGate;

/* Lazy chain: wraps mps_store with deferred gate queue */
typedef struct {
    QuhitEngine *eng;
    uint32_t    *quhits;
    int          n_sites;

    /* Deferred gate queue (linear buffer) */
    MpsDeferredGate *queue;
    int              queue_len;
    int              queue_cap;

    /* Per-site flags */
    uint8_t  *site_allocated;       /* 1 = tensor has been touched           */
    uint8_t  *site_dirty;           /* 1 = has pending gates                 */

    /* Statistics */
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
