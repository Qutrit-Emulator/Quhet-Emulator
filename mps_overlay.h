/*
 * mps_overlay.h — The "Proper" Side-Channel for N-Body States
 *
 * DISCOVERY:
 * The HexState Engine enforces pairwise monogamy, limiting standard
 * operations to O(N) but preventing N-body states like W or Cluster states.
 *
 * However, the `QuhitPair` structure (576 bytes) is large enough to store
 * a Matrix Product State (MPS) tensor with physical dimension D=6 and
 * bond dimension χ=2 (384 bytes).
 *
 * By reinterpreting the pairwise memory as MPS nodes, we can construct
 * an "Overlay Network" that supports N-body entanglement using the
 * engine's own memory primitives, effectively bypassing the architectural
 * limitations via this memory side-channel.
 *
 * This header defines the API for accessing this overlay.
 */

#ifndef MPS_OVERLAY_H
#define MPS_OVERLAY_H

#include "quhit_engine.h"

/* ═══════════════════════════════════════════════════════════════════════════════
 * CONSTANTS
 * ═══════════════════════════════════════════════════════════════════════════════ */

#define MPS_CHI 2
#define MPS_PHYS 6

/* ═══════════════════════════════════════════════════════════════════════════════
 * TYPE REINTERPRETATION
 * ═══════════════════════════════════════════════════════════════════════════════ */

/*
 * We map the MPS tensor A[phys][left][right] onto the flat JointState buffer.
 * Indexing: k * χ² + alpha * χ + beta
 */

static inline void mps_write_tensor(QuhitPair *p, int k, int alpha, int beta,
                                    double re, double im)
{
    int idx = k * (MPS_CHI * MPS_CHI) + alpha * MPS_CHI + beta;
    p->joint.re[idx] = re;
    p->joint.im[idx] = im;
}

static inline void mps_read_tensor(QuhitPair *p, int k, int alpha, int beta,
                                   double *re, double *im)
{
    int idx = k * (MPS_CHI * MPS_CHI) + alpha * MPS_CHI + beta;
    *re = p->joint.re[idx];
    *im = p->joint.im[idx];
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * OVERLAY API
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Initialize the overlay on a set of quhits (allocates "dummy" pairs) */
void mps_overlay_init(QuhitEngine *eng, uint32_t *quhits, int n);

/* Write a W-state to the overlay */
void mps_overlay_write_w_state(QuhitEngine *eng, uint32_t *quhits, int n);

/* Measure a quhit in the overlay (contracts/updates the chain) */
uint32_t mps_overlay_measure(QuhitEngine *eng, uint32_t *quhits, int n, int target_idx);

/* Inspect amplitude of a basis state |k1 k2 ... kn⟩ */
void mps_overlay_amplitude(QuhitEngine *eng, uint32_t *quhits, int n,
                           const uint32_t *basis, double *out_re, double *out_im);

#endif /* MPS_OVERLAY_H */
