/*
 * quhit_hexagram.h — The Hexagram Quhit
 *
 * A new quantum primitive: the EDGE DUAL of the triality quhit.
 *
 * The standard (triality) quhit stores amplitudes on 6 VERTICES of
 * the hexagon — the computational basis states |0⟩...|5⟩. 
 *
 * The hexagram quhit stores amplitudes on 6 LINE SEGMENTS of the
 * unicursal hexagram — the face diagonals of the cube projected along
 * its body diagonal (1,1,1).
 *
 * The 6 hexagram lines (unicursal traversal order):
 *
 *   ℓ₀: diameter  E—center—D   (cyan,    C face diagonals)
 *   ℓ₁: outer     D—C          (yellow,  Y face diagonal)
 *   ℓ₂: diameter  C—center—F   (magenta, M face diagonals)
 *   ℓ₃: outer     F—B          (cyan,    C face diagonal)
 *   ℓ₄: diameter  B—center—G   (yellow,  Y face diagonals)
 *   ℓ₅: outer     G—E          (magenta, M face diagonal)
 *
 * Key properties:
 *   - Chirality is intrinsic: the unicursal path has a direction.
 *     The two orientations correspond to the two mirror tetrahedra
 *     inscribed in the cube.
 *   - Δ=0 is the native ground state (hexagram states encode the
 *     exotic S₆ automorphism structure naturally).
 *   - The H₆ transform (vertex ↔ hexagram) is derived from the
 *     body-diagonal projection of face diagonals — NOT the DFT₆.
 *
 *   Vertex model:   TrialityQuhit  (amplitudes on points)
 *   Edge model:     HexagramQuhit  (amplitudes on paths)
 *   Duality:        Kramers-Wannier, mediated by S₆ outer automorphism
 */

#ifndef QUHIT_HEXAGRAM_H
#define QUHIT_HEXAGRAM_H

#include <stdint.h>

#define HEX_D 6

/* ═══════════════════════════════════════════════════════════════════════
 * CHIRALITY — Path orientation of the unicursal hexagram
 * ═══════════════════════════════════════════════════════════════════════ */

#define CHIRALITY_POS  (+1)   /* ℓ₀→ℓ₁→ℓ₂→ℓ₃→ℓ₄→ℓ₅ = tetrahedron A */
#define CHIRALITY_NEG  (-1)   /* ℓ₅→ℓ₄→ℓ₃→ℓ₂→ℓ₁→ℓ₀ = tetrahedron B (mirror) */

/* ═══════════════════════════════════════════════════════════════════════
 * LINE SEGMENT TYPES
 * ═══════════════════════════════════════════════════════════════════════ */

#define LINE_DIAMETER  0   /* Passes through center (2 face diagonals merged) */
#define LINE_OUTER     1   /* Outer edge connecting adjacent hex vertices */

/* ═══════════════════════════════════════════════════════════════════════
 * THE HEXAGRAM QUHIT
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    /* 6 complex amplitudes — one per hexagram line segment */
    double line_re[HEX_D];
    double line_im[HEX_D];

    /* Chirality: +1 (positive traversal) or -1 (mirror traversal) */
    int chirality;

    /* Cached vertex-basis representation (for interconversion) */
    double vertex_re[HEX_D];
    double vertex_im[HEX_D];
    uint8_t vertex_dirty;  /* 1 if vertex cache is stale */

    /* Line metadata (static, set at init) */
    /* line_type[k]: LINE_DIAMETER or LINE_OUTER */
    /* line_color[k]: 0=C(cyan), 1=M(magenta), 2=Y(yellow) */
} HexagramQuhit;

/* ═══════════════════════════════════════════════════════════════════════
 * H₆ TRANSFORM — The body-diagonal projection matrix
 *
 * H₆ converts vertex amplitudes → hexagram-line amplitudes.
 * H₆† converts hexagram-line amplitudes → vertex amplitudes.
 *
 * Derivation: each hexagram line ℓₖ is a specific combination of
 * vertex states determined by which cube face diagonals project
 * onto that line under the body-diagonal (1,1,1) projection.
 *
 * The matrix is syntheme-weighted: diameters combine antipodal
 * vertex pairs (both diagonals of a face), outer edges combine
 * adjacent vertex pairs (single diagonal connecting two faces).
 *
 * H₆ is UNITARY: H₆ · H₆† = I.
 * H₆ is NOT the DFT₆ — it encodes geometry, not Fourier analysis.
 * ═══════════════════════════════════════════════════════════════════════ */

/* The 6×6 H₆ transform matrices (precomputed at init) */
extern double H6_re[HEX_D][HEX_D];
extern double H6_im[HEX_D][HEX_D];
extern double H6_adj_re[HEX_D][HEX_D];  /* H₆† (adjoint) */
extern double H6_adj_im[HEX_D][HEX_D];

/* ═══════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 * ═══════════════════════════════════════════════════════════════════════ */

/* Initialize the H₆ transform tables. Call once at startup. */
void hexagram_init_tables(void);

/* Initialize to the "first line" state |ℓ₀⟩ with positive chirality */
void hexagram_init(HexagramQuhit *q);

/* Initialize from a standard-basis state vector via H₆ transform */
void hexagram_init_from_vertex(HexagramQuhit *q,
                               const double *vert_re, const double *vert_im,
                               int chirality);

/* Initialize to a specific hexagram line segment |ℓₖ⟩ */
void hexagram_init_line(HexagramQuhit *q, int k, int chirality);

/* ═══════════════════════════════════════════════════════════════════════
 * NATIVE HEXAGRAM GATES — O(D) operations
 * ═══════════════════════════════════════════════════════════════════════ */

/* Path shift: advance along the unicursal path by δ segments.
 * |ℓₖ⟩ → |ℓ_{(k+δ) mod 6}⟩
 * This is DIAGONAL in hexagram basis — O(D).
 * δ>0 = forward along chirality, δ<0 = backward. */
void hexagram_path_shift(HexagramQuhit *q, int delta);

/* Per-line phase gate: |ℓₖ⟩ → e^{iφₖ}|ℓₖ⟩
 * Diagonal in hexagram basis — O(D). */
void hexagram_phase(HexagramQuhit *q, const double *phi_re, const double *phi_im);

/* Diameter phase: apply phase only to diameter lines (ℓ₀,ℓ₂,ℓ₄).
 * This targets the "through-center" segments specifically. O(3). */
void hexagram_diameter_phase(HexagramQuhit *q, double phi_re, double phi_im);

/* Outer phase: apply phase only to outer lines (ℓ₁,ℓ₃,ℓ₅). O(3). */
void hexagram_outer_phase(HexagramQuhit *q, double phi_re, double phi_im);

/* Chirality flip: reverse the path orientation.
 * Corresponds to switching between the two mirror tetrahedra.
 * |ℓₖ, +⟩ → |ℓ_{5-k}, -⟩  (reversal + conjugation)
 * This is an INVOLUTION: flip ∘ flip = identity. O(D). */
void hexagram_flip(HexagramQuhit *q);

/* Triad gate: simultaneous rotation of all 3 diameters.
 * ℓ₀↔ℓ₂↔ℓ₄ (diameters cycle), ℓ₁↔ℓ₃↔ℓ₅ (outers cycle).
 * This is the φ-image of triality_rotate. O(D). */
void hexagram_triad(HexagramQuhit *q);

/* Inverse triad. O(D). */
void hexagram_triad_inv(HexagramQuhit *q);

/* ═══════════════════════════════════════════════════════════════════════
 * ENTANGLEMENT — Center-crossing interaction
 *
 * Two hexagram quhits can entangle through shared center crossings.
 * The 3 diameters all pass through the center point — when two
 * hexagram states have amplitude on overlapping diameters, they
 * interfere at the crossing.
 *
 * This is the hexagrammatic analog of CZ: it couples the diameter
 * amplitudes of both quhits while leaving outer amplitudes unchanged.
 * ═══════════════════════════════════════════════════════════════════════ */

void hexagram_cross(HexagramQuhit *a, HexagramQuhit *b);

/* ═══════════════════════════════════════════════════════════════════════
 * MEASUREMENT
 * ═══════════════════════════════════════════════════════════════════════ */

/* Measure which hexagram line the state occupies.
 * Returns outcome 0..5. Collapses state. */
int hexagram_measure(HexagramQuhit *q, uint64_t *rng_state);

/* Probability distribution over the 6 lines — no collapse. O(D). */
void hexagram_probabilities(const HexagramQuhit *q, double *probs);

/* ═══════════════════════════════════════════════════════════════════════
 * INTERCONVERSION — Vertex model ↔ Edge model
 *
 * These use the H₆ transform to convert between the two dual
 * representations. The conversion is exact (H₆ is unitary).
 * ═══════════════════════════════════════════════════════════════════════ */

/* Ensure vertex cache is up-to-date (applies H₆†) */
void hexagram_ensure_vertex(HexagramQuhit *q);

/* Get read-only vertex amplitudes (ensures first) */
const double *hexagram_vertex_re(HexagramQuhit *q);
const double *hexagram_vertex_im(HexagramQuhit *q);

/* ═══════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 * ═══════════════════════════════════════════════════════════════════════ */

/* Print hexagram state: line amplitudes + chirality */
void hexagram_print(const HexagramQuhit *q, const char *label);

/* Line metadata */
int  hexagram_line_type(int k);   /* LINE_DIAMETER or LINE_OUTER */
int  hexagram_line_color(int k);  /* 0=C, 1=M, 2=Y */
const char *hexagram_line_name(int k);  /* e.g. "ℓ₀ diam C" */

#endif /* QUHIT_HEXAGRAM_H */
