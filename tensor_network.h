/*
 * tensor_network.h — Reality's Tensor Product Storage Format
 *
 * Reality stores D^N states not as a flat vector, but as a tensor network:
 * a chain of small local tensors connected by bond indices.
 *
 * Each quhit is a 3-index tensor: A[i][α][β]
 *   i = physical index   (0..D-1)   — the quhit's state
 *   α = left bond index  (0..χ_L-1) — entanglement with left neighbor
 *   β = right bond index (0..χ_R-1) — entanglement with right neighbor
 *
 * Storage per node: D × χ_L × χ_R complex doubles
 * Total network:    O(N × χ² × D) — LINEAR in N, not exponential
 *
 * Bond dimension χ controls entanglement capacity:
 *   χ = 1:  product state (no entanglement) — just QuhitState (96 bytes)
 *   χ = D:  full pairwise entanglement — equivalent to QuhitJoint (576 bytes)
 *   χ > D:  multi-party entanglement (W states, cluster states, etc.)
 *
 * Reality's constraints:
 *   - Area law: entropy across any cut ≤ log₂(χ).  Truncation enforces this.
 *   - Monogamy: total entanglement per node is bounded by bond dimension.
 *   - Decoherence: SVD truncation = garbage collection of weak entanglement.
 *   - Locality: two-site gates only on adjacent nodes (SWAP for non-adjacent).
 *
 * This is a Matrix Product State (MPS) — the 1D tensor network.
 * For our engine, 1D is sufficient because any circuit can be mapped to 1D
 * via SWAP networks, and all interactions are ultimately pairwise.
 */

#ifndef TENSOR_NETWORK_H
#define TENSOR_NETWORK_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ═══════════════════════════════════════════════════════════════════════════════
 * CONSTANTS
 * ═══════════════════════════════════════════════════════════════════════════════ */

#ifndef TN_D
#define TN_D 6   /* Physical dimension per site = QUHIT_D */
#endif

#define TN_MAX_CHI_DEFAULT 36  /* Default max bond dimension (D² = full 2-site) */
#define TN_SVD_EPSILON 1e-14   /* Singular values below this are decoherence   */

/* ═══════════════════════════════════════════════════════════════════════════════
 * TENSOR NODE — One site in the MPS
 *
 * Stores a rank-3 tensor A[i][α][β]:
 *   re[i * chi_left * chi_right + α * chi_right + β]
 *   im[i * chi_left * chi_right + α * chi_right + β]
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    uint32_t  chi_left;       /* Left bond dimension  (1 = no left neighbor)  */
    uint32_t  chi_right;      /* Right bond dimension (1 = no right neighbor) */
    double   *re;             /* Real part: D × chi_left × chi_right doubles  */
    double   *im;             /* Imag part: D × chi_left × chi_right doubles  */
} TensorNode;

/* ═══════════════════════════════════════════════════════════════════════════════
 * TENSOR NETWORK — The full MPS chain
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    uint64_t     n_nodes;     /* Number of quhits in network                  */
    TensorNode  *nodes;       /* Array of n_nodes tensor nodes                */
    uint32_t     max_chi;     /* Maximum bond dimension (entanglement budget) */
    uint32_t     d;           /* Physical dimension per site                  */
} TensorNetwork;

/* ═══════════════════════════════════════════════════════════════════════════════
 * INLINE HELPERS
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Number of complex amplitudes stored in a single node */
static inline uint64_t tn_node_size(const TensorNode *n)
{
    return (uint64_t)TN_D * n->chi_left * n->chi_right;
}

/* Byte cost of a single node */
static inline uint64_t tn_node_bytes(const TensorNode *n)
{
    return tn_node_size(n) * 2 * sizeof(double);  /* re + im */
}

/* Total bytes for entire network */
static inline uint64_t tn_total_bytes(const TensorNetwork *tn)
{
    uint64_t total = 0;
    for (uint64_t i = 0; i < tn->n_nodes; i++)
        total += tn_node_bytes(&tn->nodes[i]);
    return total;
}

/* Access element A[phys][alpha][beta] */
static inline int tn_idx(const TensorNode *n, int phys, int alpha, int beta)
{
    return phys * (int)(n->chi_left * n->chi_right)
         + alpha * (int)n->chi_right
         + beta;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * API — implemented in tensor_network.c
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Create N-node MPS in product state |0,0,...,0⟩. Each node is χ=1. */
TensorNetwork *tn_create(uint64_t n_nodes, uint32_t max_chi);

/* Destroy network and free all memory */
void tn_destroy(TensorNetwork *tn);

/* Initialize node i to a specific local state (D amplitudes, χ stays 1) */
void tn_set_local(TensorNetwork *tn, uint64_t node_i,
                  const double *re, const double *im);

/* Apply D×D local unitary to one site (no bond change) */
void tn_apply_local(TensorNetwork *tn, uint64_t node_i,
                    const double U_re[TN_D][TN_D],
                    const double U_im[TN_D][TN_D]);

/* Apply D²×D² two-site gate between adjacent sites i, i+1.
 * This is the key operation: contract, apply gate, SVD split, truncate.
 * Bond dimension grows up to max_chi, then area-law truncation kicks in. */
void tn_apply_two_site(TensorNetwork *tn, uint64_t node_i,
                       const double *gate_re,   /* D²×D² matrix, row-major */
                       const double *gate_im);

/* Measure site i. Returns outcome (0..D-1). Collapses bonds. */
uint32_t tn_measure(TensorNetwork *tn, uint64_t node_i, double random_01);

/* Get amplitude of a specific basis state |k₀ k₁ ... k_{N-1}⟩.
 * Contracts the entire chain. Cost: O(N × χ²). */
void tn_amplitude(const TensorNetwork *tn, const uint32_t *basis,
                  double *out_re, double *out_im);

/* Entanglement entropy across the cut between site i and site i+1.
 * Measured in bits (log₂). Returns 0 for product states. */
double tn_entropy(const TensorNetwork *tn, uint64_t cut_after);

/* Bond dimension between sites i and i+1 */
static inline uint32_t tn_bond_dim(const TensorNetwork *tn, uint64_t site_i)
{
    if (site_i >= tn->n_nodes - 1) return 0;
    return tn->nodes[site_i].chi_right;
}

/* Total max entanglement budget in bits */
static inline double tn_max_entropy(const TensorNetwork *tn)
{
    return log2((double)tn->max_chi);
}

#endif /* TENSOR_NETWORK_H */
