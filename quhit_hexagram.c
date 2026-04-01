/*
 * quhit_hexagram.c — The Hexagram Quhit Implementation
 *
 * Edge-dual of the triality quhit. Amplitudes on hexagram line segments.
 *
 * The H₆ transform is derived from the body-diagonal projection of the
 * cube's face diagonals. Each hexagram line ℓₖ corresponds to specific
 * face diagonals that project onto that line when viewed from (1,1,1).
 *
 * Cube vertex labels (Cubeee.html convention):
 *   0:(-1,-1,-1)  1:(+1,-1,-1)  2:(+1,+1,-1)  3:(-1,+1,-1)
 *   4:(-1,-1,+1)  5:(+1,-1,+1)  6:(+1,+1,+1)  7:(-1,+1,+1)
 *
 * Body-diagonal projection from (1,1,1), projected positions:
 *   0,6 → center (body diagonal endpoints)
 *   1 → (√2, 0)          ≈ right
 *   2 → (1/√2, √(3/2))   ≈ upper-right
 *   3 → (-1/√2, √(3/2))  ≈ upper-left
 *   4 → (-√2, 0)          ≈ left
 *   5 → (-1/√2, -√(3/2)) ≈ lower-left  (wasn't this wrong? No...)
 *   ... Wait, let me use the quhit basis states directly.
 *
 * ── Mapping from quhit basis states to hexagram lines ──
 *
 * The 6 basis states |0⟩...|5⟩ map to the CMY channel structure:
 *   C: {|0⟩, |1⟩} = ±X face pair
 *   M: {|2⟩, |3⟩} = ±Y face pair
 *   Y: {|4⟩, |5⟩} = ±Z face pair
 *
 * Each face has 2 diagonals. Under body-diagonal projection:
 *   Face diagonals within channel k map to hexagram lines.
 *   The specific mapping depends on which cube vertices the
 *   face diagonals connect and how they project.
 *
 * The H₆ matrix encodes: for each hexagram line ℓₖ, which
 * superposition of basis states |j⟩ contributes amplitude.
 *
 * ── Derivation of H₆ ──
 *
 * The 6 hexagram lines alternate: diameter, outer, diameter, outer, ...
 *
 * A DIAMETER line passes through the center. In the cube, this
 * corresponds to two face diagonals from opposite faces of the same
 * axis that project onto the same line through center. These combine
 * the vesica (sum) and wave (difference) of the antipodal pair.
 *
 * An OUTER line connects two adjacent hexagram vertices. This
 * corresponds to a single face diagonal from a different axis that
 * connects the projected positions of two non-antipodal vertices.
 *
 * For each hexagram line ℓₖ, H₆[k][j] gives the contribution of
 * vertex basis state |j⟩. The matrix is constructed so that:
 *
 *   Diameters:  ℓ₀ combines C-channel pair {|0⟩,|1⟩} antisymmetrically
 *               ℓ₂ combines M-channel pair {|2⟩,|3⟩} antisymmetrically
 *               ℓ₄ combines Y-channel pair {|4⟩,|5⟩} antisymmetrically
 *
 *   Outers:     ℓ₁ combines a cross-channel pair from Y and M
 *               ℓ₃ combines a cross-channel pair from C and Y
 *               ℓ₅ combines a cross-channel pair from M and C
 *
 * The specific coefficients ensure unitarity and encode the 120°
 * rotational symmetry of the body-diagonal view (C→M→Y→C cycling).
 *
 * The eigenbasis structure: diameters are channel-internal (sum/diff
 * within a pair), outers are channel-crossing (linking adjacent
 * channels). This 3+3 partition mirrors the unicursal path's
 * alternating diameter/outer structure.
 */

#include <string.h>
#include <math.h>
#include <stdio.h>
#include "quhit_hexagram.h"

/* ═══════════════════════════════════════════════════════════════════════
 * CONSTANTS
 * ═══════════════════════════════════════════════════════════════════════ */

static const double INV_SQRT2 = 0.70710678118654752440;
static const double INV_SQRT3 = 0.57735026918962576451;
static const double INV_SQRT6 = 0.40824829046386301637;

/* ω₃ = e^{2πi/3} = -1/2 + i√3/2 */
static const double W3_RE = -0.5;
static const double W3_IM =  0.86602540378443864676;

/* ω₆ = e^{2πi/6} = 1/2 + i√3/2 */
static const double W6_RE =  0.5;
static const double W6_IM =  0.86602540378443864676;

/* Line metadata (static) */
static const int LINE_TYPES[6] = {
    LINE_DIAMETER, LINE_OUTER,
    LINE_DIAMETER, LINE_OUTER,
    LINE_DIAMETER, LINE_OUTER
};

/* CMY color assignment per line:
 * ℓ₀=C(0), ℓ₁=Y(2), ℓ₂=M(1), ℓ₃=C(0), ℓ₄=Y(2), ℓ₅=M(1)
 * Pattern: C, Y, M, C, Y, M — triality cycling with 120° offset */
static const int LINE_COLORS[6] = { 0, 2, 1, 0, 2, 1 };

static const char *LINE_NAMES[6] = {
    "l0 diam C", "l1 outr Y", "l2 diam M",
    "l3 outr C", "l4 diam Y", "l5 outr M"
};

/* ═══════════════════════════════════════════════════════════════════════
 * H₆ TRANSFORM MATRICES
 *
 * H₆ maps vertex basis |j⟩ → hexagram line basis |ℓₖ⟩.
 *
 * Structure (6×6 unitary):
 *
 *   Diameters (rows 0,2,4) = channel-pair DIFFERENCES (wave):
 *     ℓ₀ = (|0⟩ - |1⟩)/√2     [C channel difference]
 *     ℓ₂ = (|2⟩ - |3⟩)/√2     [M channel difference]
 *     ℓ₄ = (|4⟩ - |5⟩)/√2     [Y channel difference]
 *
 *   Outers (rows 1,3,5) = DFT₃-weighted channel SUMS (vesica):
 *     Let s_c = (|2c⟩ + |2c+1⟩)/√2 for channel c ∈ {0,1,2}
 *     Then:
 *     ℓ₁ = (s₀ + s₁ + s₂)/√3         = (1,1,1,1,1,1)/√6
 *     ℓ₃ = (s₀ + ω₃·s₁ + ω₃²·s₂)/√3
 *     ℓ₅ = (s₀ + ω₃²·s₁ + ω₃·s₂)/√3
 *
 * Orthogonality proof:
 *   Diameter ⊥ Outer: within each channel pair (2c, 2c+1),
 *     diameter has (+1,-1)/√2, outer has (+x,+x)/√2.
 *     Inner product per pair: x - x = 0. ✓
 *   Outer ⊥ Outer: DFT₃ rows are orthogonal (1+ω₃+ω₃²=0). ✓
 *   Diameter ⊥ Diameter: non-overlapping channel pairs. ✓
 *
 * This is the Cooley-Tukey DFT₆ = DFT₂ ⊗ DFT₃:
 *   DFT₂ within each channel → difference (diameter) + sum (outer)
 *   DFT₃ across the 3 sums → the 3 outer lines with ω₃ phases
 * ═══════════════════════════════════════════════════════════════════════ */

double H6_re[HEX_D][HEX_D];
double H6_im[HEX_D][HEX_D];
double H6_adj_re[HEX_D][HEX_D];
double H6_adj_im[HEX_D][HEX_D];

void hexagram_init_tables(void) {
    memset(H6_re, 0, sizeof(H6_re));
    memset(H6_im, 0, sizeof(H6_im));

    /* ω₃ powers: ω₃^0=1, ω₃^1=(-1+i√3)/2, ω₃^2=(-1-i√3)/2 */
    const double w3r[3] = { 1.0,  W3_RE,  W3_RE };
    const double w3i[3] = { 0.0,  W3_IM, -W3_IM };

    /* ── Diameter rows: (|2c⟩ - |2c+1⟩)/√2 ── */
    for (int d = 0; d < 3; d++) {
        int row = 2 * d;        /* rows 0, 2, 4 */
        int c0 = 2 * d;         /* first column of channel pair */
        H6_re[row][c0]     =  INV_SQRT2;
        H6_re[row][c0 + 1] = -INV_SQRT2;
    }

    /* ── Outer rows: Σ_c ω₃^(r·c) · (|2c⟩ + |2c+1⟩) / √6 ── */
    for (int r = 0; r < 3; r++) {
        int row = 2 * r + 1;    /* rows 1, 3, 5 */
        for (int c = 0; c < 3; c++) {
            int idx = (r * c) % 3;      /* ω₃ exponent */
            double wr = w3r[idx] * INV_SQRT6;
            double wi = w3i[idx] * INV_SQRT6;
            /* Both elements of channel c get the same coefficient */
            H6_re[row][2*c]     = wr;  H6_im[row][2*c]     = wi;
            H6_re[row][2*c + 1] = wr;  H6_im[row][2*c + 1] = wi;
        }
    }

    /* Compute H₆† (conjugate transpose) */
    for (int i = 0; i < HEX_D; i++) {
        for (int j = 0; j < HEX_D; j++) {
            H6_adj_re[i][j] =  H6_re[j][i];
            H6_adj_im[i][j] = -H6_im[j][i];
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * TRANSFORM PRIMITIVES
 * ═══════════════════════════════════════════════════════════════════════ */

/* Apply H₆: vertex → hexagram */
static void apply_H6(const double *in_re, const double *in_im,
                     double *out_re, double *out_im)
{
    for (int k = 0; k < HEX_D; k++) {
        double sr = 0, si = 0;
        for (int j = 0; j < HEX_D; j++) {
            double hr = H6_re[k][j], hi = H6_im[k][j];
            sr += hr * in_re[j] - hi * in_im[j];
            si += hr * in_im[j] + hi * in_re[j];
        }
        out_re[k] = sr;
        out_im[k] = si;
    }
}

/* Apply H₆†: hexagram → vertex */
static void apply_H6_adj(const double *in_re, const double *in_im,
                          double *out_re, double *out_im)
{
    for (int j = 0; j < HEX_D; j++) {
        double sr = 0, si = 0;
        for (int k = 0; k < HEX_D; k++) {
            double hr = H6_adj_re[j][k], hi = H6_adj_im[j][k];
            sr += hr * in_re[k] - hi * in_im[k];
            si += hr * in_im[k] + hi * in_re[k];
        }
        out_re[j] = sr;
        out_im[j] = si;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 * ═══════════════════════════════════════════════════════════════════════ */

void hexagram_init(HexagramQuhit *q) {
    memset(q, 0, sizeof(HexagramQuhit));
    q->line_re[0] = 1.0;  /* |ℓ₀⟩ */
    q->chirality = CHIRALITY_POS;
    q->vertex_dirty = 1;
}

void hexagram_init_from_vertex(HexagramQuhit *q,
                               const double *vert_re, const double *vert_im,
                               int chirality)
{
    memset(q, 0, sizeof(HexagramQuhit));
    q->chirality = chirality;

    /* Apply H₆ to convert vertex → hexagram */
    apply_H6(vert_re, vert_im, q->line_re, q->line_im);

    /* Cache the vertex representation */
    memcpy(q->vertex_re, vert_re, HEX_D * sizeof(double));
    memcpy(q->vertex_im, vert_im, HEX_D * sizeof(double));
    q->vertex_dirty = 0;
}

void hexagram_init_line(HexagramQuhit *q, int k, int chirality) {
    memset(q, 0, sizeof(HexagramQuhit));
    q->line_re[k] = 1.0;
    q->chirality = chirality;
    q->vertex_dirty = 1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * NATIVE HEXAGRAM GATES
 * ═══════════════════════════════════════════════════════════════════════ */

void hexagram_path_shift(HexagramQuhit *q, int delta) {
    delta = ((delta % HEX_D) + HEX_D) % HEX_D;
    if (delta == 0) return;

    /* Cyclic permutation of line amplitudes */
    double tmp_re[HEX_D], tmp_im[HEX_D];
    for (int k = 0; k < HEX_D; k++) {
        int src = (k - delta + HEX_D) % HEX_D;
        tmp_re[k] = q->line_re[src];
        tmp_im[k] = q->line_im[src];
    }
    memcpy(q->line_re, tmp_re, sizeof(tmp_re));
    memcpy(q->line_im, tmp_im, sizeof(tmp_im));
    q->vertex_dirty = 1;
}

void hexagram_phase(HexagramQuhit *q, const double *phi_re, const double *phi_im) {
    for (int k = 0; k < HEX_D; k++) {
        double re = q->line_re[k], im = q->line_im[k];
        q->line_re[k] = re * phi_re[k] - im * phi_im[k];
        q->line_im[k] = re * phi_im[k] + im * phi_re[k];
    }
    q->vertex_dirty = 1;
}

void hexagram_diameter_phase(HexagramQuhit *q, double phi_re, double phi_im) {
    /* Apply phase only to diameter lines: ℓ₀, ℓ₂, ℓ₄ */
    for (int k = 0; k < HEX_D; k += 2) {
        double re = q->line_re[k], im = q->line_im[k];
        q->line_re[k] = re * phi_re - im * phi_im;
        q->line_im[k] = re * phi_im + im * phi_re;
    }
    q->vertex_dirty = 1;
}

void hexagram_outer_phase(HexagramQuhit *q, double phi_re, double phi_im) {
    /* Apply phase only to outer lines: ℓ₁, ℓ₃, ℓ₅ */
    for (int k = 1; k < HEX_D; k += 2) {
        double re = q->line_re[k], im = q->line_im[k];
        q->line_re[k] = re * phi_re - im * phi_im;
        q->line_im[k] = re * phi_im + im * phi_re;
    }
    q->vertex_dirty = 1;
}

void hexagram_flip(HexagramQuhit *q) {
    /* Chirality flip: reverse path orientation.
     * |ℓₖ, +⟩ → |ℓ_{5-k}, -⟩
     * Also complex-conjugates amplitudes (time reversal). */
    double tmp_re[HEX_D], tmp_im[HEX_D];
    for (int k = 0; k < HEX_D; k++) {
        tmp_re[k] =  q->line_re[5 - k];
        tmp_im[k] = -q->line_im[5 - k];  /* conjugation */
    }
    memcpy(q->line_re, tmp_re, sizeof(tmp_re));
    memcpy(q->line_im, tmp_im, sizeof(tmp_im));
    q->chirality = -q->chirality;
    q->vertex_dirty = 1;
}

void hexagram_triad(HexagramQuhit *q) {
    /* Triad gate: cyclic permutation of the 3 diameter/outer pairs.
     * ℓ₀→ℓ₂→ℓ₄→ℓ₀ (diameters: C→M→Y→C)
     * ℓ₁→ℓ₃→ℓ₅→ℓ₁ (outers: Y→C→M→Y)
     * This is the φ-image of triality_rotate. */
    double d0_re = q->line_re[0], d0_im = q->line_im[0];
    double o0_re = q->line_re[1], o0_im = q->line_im[1];

    q->line_re[0] = q->line_re[4]; q->line_im[0] = q->line_im[4];
    q->line_re[1] = q->line_re[5]; q->line_im[1] = q->line_im[5];
    q->line_re[4] = q->line_re[2]; q->line_im[4] = q->line_im[2];
    q->line_re[5] = q->line_re[3]; q->line_im[5] = q->line_im[3];
    q->line_re[2] = d0_re;         q->line_im[2] = d0_im;
    q->line_re[3] = o0_re;         q->line_im[3] = o0_im;

    q->vertex_dirty = 1;
}

void hexagram_triad_inv(HexagramQuhit *q) {
    /* Inverse: ℓ₀→ℓ₄→ℓ₂→ℓ₀, ℓ₁→ℓ₅→ℓ₃→ℓ₁ */
    double d0_re = q->line_re[0], d0_im = q->line_im[0];
    double o0_re = q->line_re[1], o0_im = q->line_im[1];

    q->line_re[0] = q->line_re[2]; q->line_im[0] = q->line_im[2];
    q->line_re[1] = q->line_re[3]; q->line_im[1] = q->line_im[3];
    q->line_re[2] = q->line_re[4]; q->line_im[2] = q->line_im[4];
    q->line_re[3] = q->line_re[5]; q->line_im[3] = q->line_im[5];
    q->line_re[4] = d0_re;         q->line_im[4] = d0_im;
    q->line_re[5] = o0_re;         q->line_im[5] = o0_im;

    q->vertex_dirty = 1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * ENTANGLEMENT — Center-crossing interaction
 *
 * The hexagrammatic CZ: diameters (ℓ₀,ℓ₂,ℓ₄) all pass through center.
 * When two hexagram quhits have diameter amplitude, they interfere
 * at the center crossing. The phase coupling is:
 *
 *   ω^(d_a · d_b)  where d_a, d_b ∈ {0,1,2} are the diameter indices
 *
 * Outer lines (ℓ₁,ℓ₃,ℓ₅) do not pass through center → no coupling.
 * ═══════════════════════════════════════════════════════════════════════ */

void hexagram_cross(HexagramQuhit *a, HexagramQuhit *b) {
    /* ω₃ roots: ω₃^0=1, ω₃^1=(-1+i√3)/2, ω₃^2=(-1-i√3)/2 */
    static const double W3R[3] = {1.0, -0.5, -0.5};
    static const double W3I[3] = {0.0, 0.86602540378443864676, -0.86602540378443864676};

    /* Diameter indices: ℓ₀→d0, ℓ₂→d1, ℓ₄→d2 */
    /* Map line index to diameter index: k/2 for even k */

    /* Compute effective phases from partner's diameter amplitudes */
    /* For each diameter d_a of qubit a, the effective phase is:
     * eff_a[d_a] = Σ_{d_b} |b[2·d_b]|² · ω₃^(d_a · d_b) */
    for (int da = 0; da < 3; da++) {
        int ka = 2 * da;  /* line index */
        double eff_re = 0, eff_im = 0;
        for (int db = 0; db < 3; db++) {
            int kb = 2 * db;
            double bprob = b->line_re[kb]*b->line_re[kb] + b->line_im[kb]*b->line_im[kb];
            int idx = (da * db) % 3;
            eff_re += bprob * W3R[idx];
            eff_im += bprob * W3I[idx];
        }
        /* Apply effective phase to a's diameter amplitude */
        double re = a->line_re[ka], im = a->line_im[ka];
        a->line_re[ka] = re * eff_re - im * eff_im;
        a->line_im[ka] = re * eff_im + im * eff_re;
    }

    /* Same for qubit b */
    for (int db = 0; db < 3; db++) {
        int kb = 2 * db;
        double eff_re = 0, eff_im = 0;
        for (int da = 0; da < 3; da++) {
            int ka = 2 * da;
            double aprob = a->line_re[ka]*a->line_re[ka] + a->line_im[ka]*a->line_im[ka];
            int idx = (da * db) % 3;
            eff_re += aprob * W3R[idx];
            eff_im += aprob * W3I[idx];
        }
        double re = b->line_re[kb], im = b->line_im[kb];
        b->line_re[kb] = re * eff_re - im * eff_im;
        b->line_im[kb] = re * eff_im + im * eff_re;
    }

    /* Renormalize both quhits */
    for (int qi = 0; qi < 2; qi++) {
        HexagramQuhit *q = (qi == 0) ? a : b;
        double norm = 0;
        for (int k = 0; k < HEX_D; k++)
            norm += q->line_re[k]*q->line_re[k] + q->line_im[k]*q->line_im[k];
        if (norm > 1e-30 && fabs(norm - 1.0) > 1e-15) {
            double inv = 1.0 / sqrt(norm);
            for (int k = 0; k < HEX_D; k++) {
                q->line_re[k] *= inv;
                q->line_im[k] *= inv;
            }
        }
    }

    a->vertex_dirty = 1;
    b->vertex_dirty = 1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * MEASUREMENT
 * ═══════════════════════════════════════════════════════════════════════ */

static uint64_t xorshift64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return *s = x;
}

void hexagram_probabilities(const HexagramQuhit *q, double *probs) {
    for (int k = 0; k < HEX_D; k++)
        probs[k] = q->line_re[k]*q->line_re[k] + q->line_im[k]*q->line_im[k];
}

int hexagram_measure(HexagramQuhit *q, uint64_t *rng_state) {
    double probs[HEX_D];
    hexagram_probabilities(q, probs);

    /* Born rule sampling */
    double r = (double)(xorshift64(rng_state) & 0xFFFFFFFFFFFFF) / (double)0x10000000000000;
    double cumul = 0;
    int outcome = HEX_D - 1;
    for (int k = 0; k < HEX_D; k++) {
        cumul += probs[k];
        if (r < cumul) { outcome = k; break; }
    }

    /* Collapse */
    memset(q->line_re, 0, sizeof(q->line_re));
    memset(q->line_im, 0, sizeof(q->line_im));
    q->line_re[outcome] = 1.0;
    q->vertex_dirty = 1;

    return outcome;
}

/* ═══════════════════════════════════════════════════════════════════════
 * INTERCONVERSION
 * ═══════════════════════════════════════════════════════════════════════ */

void hexagram_ensure_vertex(HexagramQuhit *q) {
    if (!q->vertex_dirty) return;
    apply_H6_adj(q->line_re, q->line_im, q->vertex_re, q->vertex_im);
    q->vertex_dirty = 0;
}

const double *hexagram_vertex_re(HexagramQuhit *q) {
    hexagram_ensure_vertex(q);
    return q->vertex_re;
}

const double *hexagram_vertex_im(HexagramQuhit *q) {
    hexagram_ensure_vertex(q);
    return q->vertex_im;
}

/* ═══════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 * ═══════════════════════════════════════════════════════════════════════ */

int hexagram_line_type(int k) { return LINE_TYPES[k]; }
int hexagram_line_color(int k) { return LINE_COLORS[k]; }
const char *hexagram_line_name(int k) { return LINE_NAMES[k]; }

void hexagram_print(const HexagramQuhit *q, const char *label) {
    const char *chir = (q->chirality == CHIRALITY_POS) ? "+" : "-";
    printf("HexagramQuhit [%s] chirality=%s\n", label ? label : "", chir);
    for (int k = 0; k < HEX_D; k++) {
        double p = q->line_re[k]*q->line_re[k] + q->line_im[k]*q->line_im[k];
        printf("  |%s>: (%+.6f %+.6fi)  P=%.4f\n",
               LINE_NAMES[k], q->line_re[k], q->line_im[k], p);
    }
    double total = 0;
    for (int k = 0; k < HEX_D; k++)
        total += q->line_re[k]*q->line_re[k] + q->line_im[k]*q->line_im[k];
    printf("  ||psi||^2 = %.10f\n", total);
}
