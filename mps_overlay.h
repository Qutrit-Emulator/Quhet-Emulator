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
 * STATE MANAGEMENT API
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Initialize the overlay on a set of quhits (allocates "dummy" pairs) */
void mps_overlay_init(QuhitEngine *eng, uint32_t *quhits, int n);

/* Write a W-state to the overlay */
void mps_overlay_write_w_state(QuhitEngine *eng, uint32_t *quhits, int n);

/* Write |0⟩^⊗N product state (identity MPS, no entanglement) */
void mps_overlay_write_zero(QuhitEngine *eng, uint32_t *quhits, int n);

/* Measure a quhit in the overlay (contracts/updates the chain) */
uint32_t mps_overlay_measure(QuhitEngine *eng, uint32_t *quhits, int n, int target_idx);

/* Inspect amplitude of a basis state |k1 k2 ... kn⟩ */
void mps_overlay_amplitude(QuhitEngine *eng, uint32_t *quhits, int n,
                           const uint32_t *basis, double *out_re, double *out_im);

/* ═══════════════════════════════════════════════════════════════════════════════
 * GATE LAYER
 *
 * Single-site gate:
 *   A'[k'][α][β] = Σ_k U[k'][k] × A[k][α][β]
 *   Cost: O(D² × χ²) = O(36 × 4) = O(144)
 *
 * Two-site gate (adjacent sites i, i+1):
 *   1. Contract: Θ[k,l][α][γ] = Σ_β A_i[k][α][β] × A_{i+1}[l][β][γ]
 *   2. Apply:    Θ'[k',l'] = Σ_{k,l} G[k',l'][k,l] × Θ[k,l]
 *   3. Reshape:  Θ'[(k',α)][(l',γ)] = Dχ × Dχ matrix
 *   4. SVD:      Θ' = U Σ V†, truncate to χ singular values
 *   5. Split:    A'_i[k'][α][β'] = U, A'_{i+1}[l'][β'][γ] = Σ V†
 *   Cost: O(D² × χ³) + O((Dχ)³) for SVD
 *
 * ═══════════════════════════════════════════════════════════════════════════════ */

/*
 * Apply a D×D unitary U to one MPS site.
 * U is given as D×D matrices (row-major): U_re[k'*D+k], U_im[k'*D+k].
 */
void mps_gate_1site(QuhitEngine *eng, uint32_t *quhits, int n,
                    int site, const double *U_re, const double *U_im);

/*
 * Apply a D²×D² unitary gate G to two adjacent MPS sites (site, site+1).
 * G is given as D²×D² matrices (row-major):
 *   G_re[(k'*D+l')*D²+(k*D+l)], G_im[...].
 * Bond dimension is truncated back to χ after SVD.
 */
void mps_gate_2site(QuhitEngine *eng, uint32_t *quhits, int n,
                    int site, const double *G_re, const double *G_im);

/* ═══════════════════════════════════════════════════════════════════════════════
 * GATE CONSTRUCTORS
 *
 * Build specific gate matrices for use with mps_gate_1site / mps_gate_2site.
 * All matrices are D×D or D²×D² in row-major order.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* DFT₆ gate (Quantum Fourier Transform on one D=6 site) */
void mps_build_dft6(double *U_re, double *U_im);

/* CZ gate: diagonal D²×D², phases ω^{k·l} where ω=e^{2πi/D} */
void mps_build_cz(double *G_re, double *G_im);

/* Controlled phase rotation: phases ω^{k·l·2^power / D} on 2-site */
void mps_build_controlled_phase(double *G_re, double *G_im, int power);

/* Hadamard-like gate for qubit subspace (acts on k=0,1, identity on 2..5) */
void mps_build_hadamard2(double *U_re, double *U_im);

/* Compute total norm ⟨ψ|ψ⟩ of the MPS (should be 1 if normalized) */
double mps_overlay_norm(QuhitEngine *eng, uint32_t *quhits, int n);

#endif /* MPS_OVERLAY_H */
