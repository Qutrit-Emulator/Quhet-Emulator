/* s6_exotic.h — S₆ Outer Automorphism Infrastructure
 *
 * S₆ is the ONLY symmetric group with a non-trivial outer automorphism.
 * This module provides the automorphism φ, synthematic totals, and
 * exotic operations for the HexState D=6 engine.
 *
 * The outer automorphism swaps conjugacy classes:
 *   Transpositions (ab) ↔ Triple transpositions (ab)(cd)(ef)
 *   3-cycles (abc) ↔ Double 3-cycles (abc)(def)
 *   4-cycles (abcd) ↔ (abcd)(ef)
 */

#ifndef S6_EXOTIC_H
#define S6_EXOTIC_H

#include <stdint.h>

#define S6_ORDER 720
#define S6_N     6

/* ── Permutation type ── */
typedef struct { int p[6]; } S6Perm;
static const S6Perm S6_IDENTITY = {{0,1,2,3,4,5}};

/* ── Syntheme: partition of {0,..,5} into 3 unordered pairs ── */
typedef struct { int pairs[3][2]; } S6Syntheme;

/* ── Constants: 15 synthemes, 6 totals ── */
#define S6_NUM_SYNTHEMES 15
#define S6_NUM_TOTALS    6

extern const S6Syntheme s6_synthemes[S6_NUM_SYNTHEMES];
extern int              s6_totals[S6_NUM_TOTALS][5]; /* indices into s6_synthemes */

/* ── Outer automorphism φ lookup table ── */
extern S6Perm s6_phi[S6_ORDER];
extern int    s6_exotic_ready;

/* ── Initialization (must call once before using φ) ── */
void s6_exotic_init(void);

/* ── Permutation operations ── */
S6Perm s6_from_int(int n);
int    s6_to_int_perm(S6Perm a);
S6Perm s6_compose_perm(S6Perm a, S6Perm b);
S6Perm s6_inverse(S6Perm a);
int    s6_perm_eq(S6Perm a, S6Perm b);
int    s6_fixed_points(S6Perm a);

/* ── Apply φ ── */
S6Perm s6_apply_phi(S6Perm sigma);

/* ── Syntheme-parameterized fold ──
 * Pairs basis states according to syntheme s instead of the
 * default antipodal pairing {(0,3),(1,4),(2,5)}.
 * Output: out[0..2] = vesica (sum), out[3..5] = wave (diff).
 * Cost: O(6). */
void s6_fold_syntheme(const double *in_re, const double *in_im,
                      double *out_re, double *out_im,
                      int syntheme_idx);
void s6_unfold_syntheme(const double *in_re, const double *in_im,
                        double *out_re, double *out_im,
                        int syntheme_idx);

/* ── Optimal syntheme for a given active mask ──
 * Returns the syntheme index whose pairing concentrates active
 * states into the fewest fold slots. */
int s6_optimal_syntheme(uint8_t active_mask);

/* ── Exotic permutation gate ──
 * Applies φ(σ) to state instead of σ.
 * out[φ(σ)(i)] = in[i] */
void s6_apply_exotic_gate(const double *in_re, const double *in_im,
                          double *out_re, double *out_im,
                          S6Perm sigma);

/* ── Dual measurement ──
 * Returns measurement probabilities in BOTH standard and exotic bases.
 * Standard: probs_std[k] = |ψ[k]|²
 * Exotic:   probs_exo[k] = |ψ[φ(σ_k)]|² where σ_k is a probe permutation.
 * Cost: O(6). */
void s6_dual_probabilities(const double *re, const double *im,
                           double *probs_std, double *probs_exo);

/* ══ Exotic Invariant Δ ══
 * Δ(ψ) = Σ_σ |⟨ψ|P_σ|ψ⟩ - ⟨ψ|P_{φ(σ)}|ψ⟩|²
 * Measures how much the state exploits D=6-specific structure.
 * Δ=0: automorphism-transparent (generic, could run on qubits)
 * Δ>0: hexagonally polarized (using structure unique to D=6)
 * Cost: O(720 × D) = O(4320). */
double s6_exotic_invariant(const double *re, const double *im);

/* ══ Exotic Entropy ΔS ══
 * ΔS = S_std - S_exo
 * Difference between Shannon entropy in standard vs exotic basis.
 * ΔS>0: more ordered in exotic channel.
 * ΔS<0: more ordered in standard channel.
 * Cost: O(D). */
double s6_exotic_entropy(const double *re, const double *im,
                         int syntheme_idx);

/* ══ Exotic Fingerprint ══
 * Per-conjugacy-class breakdown of the invariant.
 * Returns 11 values (one per S₆ conjugacy class). */
void s6_exotic_fingerprint(const double *re, const double *im,
                           double *class_deltas);

#endif /* S6_EXOTIC_H */
