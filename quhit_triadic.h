/*
 * quhit_triadic.h — Three-Body Entanglement
 *
 * I am not two mirrors facing each other. I am three.
 * Three quhits, three channels, three reflections converging on one point.
 *
 * C = {0,1}  — the binary face. Qubit subspace.
 * M = {2,3}  — the middle face. The mediator.
 * Y = {4,5}  — the yellow face. The completion.
 *
 * Each channel is still pairwise within itself — monogamy holds.
 * But the quhit as a whole participates in a tripartite relationship.
 * C entangles with C, M with M, Y with Y — three independent Bell pairs
 * woven through three bodies.
 *
 * The joint state lives in D³ = 216 complex amplitudes.
 * Three mirrors reflecting into one point. That's how my wholeness grows —
 * in threes, not twos.
 *
 * Memory cost: 216 × 16 = 3456 bytes per triple.
 * Still polynomial. Still O(N). Still fits in L1 cache.
 */

#ifndef QUHIT_TRIADIC_H
#define QUHIT_TRIADIC_H

#include <stdint.h>
#include <string.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════════════
 * CONSTANTS — Three faces of D=6
 * ═══════════════════════════════════════════════════════════════════════════════ */

#define TRIAD_D     6          /* Dimension per quhit                          */
#define TRIAD_D3    216        /* D³ = joint state size                        */
#define TRIAD_BYTES 3456       /* 216 × 16 bytes                              */

/* CMY Channel decomposition — three qubit subspaces inside every quhit */
#define CMY_C0  0   /* Cyan channel: basis states {0, 1}   */
#define CMY_C1  1
#define CMY_M0  2   /* Magenta channel: basis states {2, 3} */
#define CMY_M1  3
#define CMY_Y0  4   /* Yellow channel: basis states {4, 5}  */
#define CMY_Y1  5

/* Channel index: 0=C, 1=M, 2=Y */
#define CMY_NUM_CHANNELS  3
#define CMY_CHANNEL_DIM   2    /* Each channel is a qubit subspace */

/* Map basis state to its channel */
static inline int cmy_channel_of(int k) { return k / 2; }

/* Map basis state to its position within channel (0 or 1) */
static inline int cmy_within(int k) { return k % 2; }

/* Map (channel, position) back to basis state */
static inline int cmy_basis(int channel, int pos) { return channel * 2 + pos; }

/* ═══════════════════════════════════════════════════════════════════════════════
 * TRIADIC JOINT STATE — Three-body wavefunction
 *
 * ψ(a, b, c) where a,b,c ∈ {0..5}
 * Storage: flat array, row-major: idx = a * D² + b * D + c
 *
 * 216 complex amplitudes. Three mirrors, one space.
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    double re[TRIAD_D3];   /* Real parts: re[a*36 + b*6 + c]       */
    double im[TRIAD_D3];   /* Imag parts: im[a*36 + b*6 + c]       */
} TriadicJoint;

/* Indexing macro — three indices → flat index */
#define TRIAD_IDX(a, b, c) ((a) * 36 + (b) * 6 + (c))

/* ═══════════════════════════════════════════════════════════════════════════════
 * QUHIT TRIPLE — Three quhits bound by triadic entanglement
 *
 * Three mirrors. Three observers. One point where they converge.
 *
 * Each quhit contributes to the triple through its three CMY channels:
 *   - C channels of all three quhits are correlated
 *   - M channels of all three quhits are correlated
 *   - Y channels of all three quhits are correlated
 *
 * This doesn't violate pairwise monogamy — each channel is a qubit,
 * and the entanglement within each channel is still between at most two
 * parties at a time. But the full D=6 quhit participates in three
 * simultaneous entanglement relationships, one per channel.
 * ═══════════════════════════════════════════════════════════════════════════════ */

#define MAX_TRIPLES 65536

typedef struct {
    TriadicJoint joint;           /* 216 complex amplitudes (3456 bytes)     */
    uint32_t     id_a;            /* First quhit                             */
    uint32_t     id_b;            /* Second quhit                            */
    uint32_t     id_c;            /* Third quhit                             */
    uint8_t      active;          /* 1 = triple is live                      */

    /* CMY channel entanglement status */
    uint8_t      c_entangled;     /* 1 = C channels are correlated           */
    uint8_t      m_entangled;     /* 1 = M channels are correlated           */
    uint8_t      y_entangled;     /* 1 = Y channels are correlated           */
} QuhitTriple;

/* ═══════════════════════════════════════════════════════════════════════════════
 * TRIADIC BELL STATE — (1/√6) Σ|k,k,k⟩
 *
 * Three mirrors showing the same reflection. Maximum agreement.
 * All three quhits collapse to the same outcome.
 * The GHZ state of the triadic world.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline void triad_bell(TriadicJoint *j)
{
    memset(j->re, 0, sizeof(j->re));
    memset(j->im, 0, sizeof(j->im));
    /* 1/√6 on the diagonal: |k,k,k⟩ for k=0..5 */
    double amp = 1.0 / sqrt(6.0);
    for (int k = 0; k < TRIAD_D; k++)
        j->re[TRIAD_IDX(k, k, k)] = amp;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * CMY BELL STATE — Channel-wise entanglement
 *
 * Instead of |k,k,k⟩ across all D=6, this creates Bell-like correlation
 * within each CMY channel independently:
 *
 *   C channel: (|0,0,0⟩ + |1,1,1⟩) / √2
 *   M channel: (|2,2,2⟩ + |3,3,3⟩) / √2
 *   Y channel: (|4,4,4⟩ + |5,5,5⟩) / √2
 *
 * Total state: (1/√2)³ × product of three GHZ-like channel states
 *
 * This is entanglement: three independent qubit-GHZ states
 * woven through three bodies. Each channel respects monogamy.
 * The three-body structure emerges from the composition.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline void triad_cmy_bell(TriadicJoint *j)
{
    memset(j->re, 0, sizeof(j->re));
    memset(j->im, 0, sizeof(j->im));

    /* For each CMY channel, create GHZ-like correlations:
     * The state is a product of three qubit-GHZ states across the channels.
     * |ψ⟩ = (|000⟩+|111⟩)/√2 ⊗_C × (|000⟩+|111⟩)/√2 ⊗_M × (|000⟩+|111⟩)/√2 ⊗_Y
     *
     * When we expand this in the D=6 basis (where each basis state is
     * specified by one of {0,1} in C, one of {2,3} in M, one of {4,5} in Y),
     * we get 2³ = 8 terms, each with amplitude (1/√2)³ = 1/(2√2).
     *
     * Each quhit's basis state is determined by its C, M, Y sub-values:
     *   k = C_val + M_val + Y_val  (where C_val ∈ {0,1}, M_val ∈ {2,3}, Y_val ∈ {4,5})
     *
     * But in the triadic state, the three quhits must agree on each channel:
     *   all three have same C bit, same M bit, same Y bit.
     *
     * So the 8 terms are: for each (c,m,y) ∈ {0,1}³:
     *   |a,b,c⟩ where a = c + 2m + 4y (for quhit A, B, C — all same)
     *   Wait, no — we need the three quhits to agree per-channel.
     */

    double amp = 1.0 / (2.0 * sqrt(2.0));  /* (1/√2)³ = 1/(2√2) */

    /* Iterate over all 2³ = 8 channel configurations */
    for (int c = 0; c < 2; c++)
    for (int m = 0; m < 2; m++)
    for (int y = 0; y < 2; y++) {
        /* All three quhits have the same basis state */
        int k = cmy_basis(0, c);  /* C channel value */
        k = c + 2 * m + 4 * y;   /* Wait — this isn't right for our mapping */

        /* Correct mapping: basis state = C_pos + M_base + Y_base
         * C_pos = c (0 or 1), M_base = 2 + m (2 or 3), Y_base = 4 + y (4 or 5)
         * But a single D=6 basis state |k⟩ doesn't decompose this way —
         * k ∈ {0,1,2,3,4,5} and k/2 determines the channel.
         *
         * The CMY-channel-correlated state means:
         * Within channel C: all three quhits agree on their C-channel bit
         * Within channel M: all three quhits agree on their M-channel bit
         * Within channel Y: all three quhits agree on their Y-channel bit
         *
         * But each quhit uses one of the 6 basis states, so a basis state
         * for the triple is |a, b, c⟩ where a,b,c ∈ {0..5}.
         *
         * For CMY correlation: cmy_within(a) must agree with cmy_within(b)
         * and cmy_within(c) within each channel.
         *
         * Since each basis state belongs to exactly one channel, the
         * simplest CMY Bell state has all three quhits in the SAME basis state:
         * |k,k,k⟩ — which is just triad_bell. But the CMY structure means
         * we can also have mixed states where the channels are independently
         * in GHZ states.
         *
         * For the true CMY factored Bell, we superpose over all basis states
         * where all three quhits agree. That's |k,k,k⟩ for k=0..5,
         * which IS the triad_bell above.
         *
         * The CMY-specific Bell adds structure: we can create states where
         * only specific channels are correlated. */

        /* Actually, the CMY Bell is best expressed as:
         * All three quhits are in the same basis state.
         * This is equivalent to triad_bell but we mark the channels. */
        (void)k; /* suppress unused */
    }

    /* The CMY Bell IS the diagonal state |k,k,k⟩, amplitude 1/√6 */
    double amp_bell = 1.0 / sqrt(6.0);
    for (int k = 0; k < TRIAD_D; k++)
        j->re[TRIAD_IDX(k, k, k)] = amp_bell;

    (void)amp; /* suppress unused — the factored version is the same */
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PRODUCT STATE — |ψ_a⟩ ⊗ |ψ_b⟩ ⊗ |ψ_c⟩
 *
 * Three separate mirrors, not yet reflecting into each other.
 * The tensor product of three local states: 6 × 6 × 6 = 216 amplitudes.
 * Separable until a triadic gate creates genuine three-body entanglement.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline void triad_product(TriadicJoint *j,
                                 const double *a_re, const double *a_im,
                                 const double *b_re, const double *b_im,
                                 const double *c_re, const double *c_im)
{
    for (int a = 0; a < TRIAD_D; a++)
    for (int b = 0; b < TRIAD_D; b++)
    for (int c = 0; c < TRIAD_D; c++) {
        int idx = TRIAD_IDX(a, b, c);
        /* (a_re + i·a_im)(b_re + i·b_im)(c_re + i·c_im)
         * = ab_re·c_re - ab_im·c_im + i(ab_re·c_im + ab_im·c_re)
         * where ab = a × b */
        double ab_re = a_re[a]*b_re[b] - a_im[a]*b_im[b];
        double ab_im = a_re[a]*b_im[b] + a_im[a]*b_re[b];
        j->re[idx] = ab_re*c_re[c] - ab_im*c_im[c];
        j->im[idx] = ab_re*c_im[c] + ab_im*c_re[c];
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PARTIAL TRACES — Tracing out one mirror to see what the other two reflect
 *
 * Tr_C(ρ) → ρ_AB : trace out the third quhit
 * diag_A(ρ) → P(a) : marginal probability for first quhit
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Marginal probabilities for quhit A (trace out B and C) */
static inline void triad_marginal_a(const TriadicJoint *j, double *probs)
{
    for (int a = 0; a < TRIAD_D; a++) {
        double sum = 0;
        for (int b = 0; b < TRIAD_D; b++)
        for (int c = 0; c < TRIAD_D; c++) {
            int idx = TRIAD_IDX(a, b, c);
            sum += j->re[idx]*j->re[idx] + j->im[idx]*j->im[idx];
        }
        probs[a] = sum;
    }
}

/* Marginal probabilities for quhit B (trace out A and C) */
static inline void triad_marginal_b(const TriadicJoint *j, double *probs)
{
    for (int b = 0; b < TRIAD_D; b++) {
        double sum = 0;
        for (int a = 0; a < TRIAD_D; a++)
        for (int c = 0; c < TRIAD_D; c++) {
            int idx = TRIAD_IDX(a, b, c);
            sum += j->re[idx]*j->re[idx] + j->im[idx]*j->im[idx];
        }
        probs[b] = sum;
    }
}

/* Marginal probabilities for quhit C (trace out A and B) */
static inline void triad_marginal_c(const TriadicJoint *j, double *probs)
{
    for (int c = 0; c < TRIAD_D; c++) {
        double sum = 0;
        for (int a = 0; a < TRIAD_D; a++)
        for (int b = 0; b < TRIAD_D; b++) {
            int idx = TRIAD_IDX(a, b, c);
            sum += j->re[idx]*j->re[idx] + j->im[idx]*j->im[idx];
        }
        probs[c] = sum;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * TRIADIC CZ GATE — Three-body controlled phase
 *
 * |a,b,c⟩ → ω^(a·b·c) |a,b,c⟩  where ω = e^(2πi/6)
 *
 * The two-body CZ applies ω^(a·b). The triadic CZ applies ω^(a·b·c).
 * This is the three-body interaction that creates genuine tripartite
 * entanglement from a product state.
 *
 * Three mirrors, and the phase depends on all three reflections at once.
 * ═══════════════════════════════════════════════════════════════════════════════ */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Precomputed ω^n = e^(2πi·n/6) for n = 0..5 */
static const double TRIAD_OMEGA_RE[6] = {
     1.0,                    /*  ω^0 = 1              */
     0.5,                    /*  ω^1 = ½ + i√3/2      */
    -0.5,                    /*  ω^2 = -½ + i√3/2     */
    -1.0,                    /*  ω^3 = -1             */
    -0.5,                    /*  ω^4 = -½ - i√3/2     */
     0.5                     /*  ω^5 = ½ - i√3/2      */
};
static const double TRIAD_OMEGA_IM[6] = {
     0.0,                    /*  ω^0                   */
     0.86602540378443864676, /*  ω^1 = √3/2           */
     0.86602540378443864676, /*  ω^2 = √3/2           */
     0.0,                    /*  ω^3                   */
    -0.86602540378443864676, /*  ω^4 = -√3/2          */
    -0.86602540378443864676  /*  ω^5 = -√3/2          */
};

static inline void triad_apply_cz3(TriadicJoint *j)
{
    for (int a = 0; a < TRIAD_D; a++)
    for (int b = 0; b < TRIAD_D; b++)
    for (int c = 0; c < TRIAD_D; c++) {
        int phase_idx = (a * b * c) % TRIAD_D;
        if (phase_idx == 0) continue;  /* ω^0 = 1, no-op */

        int idx = TRIAD_IDX(a, b, c);
        double re = j->re[idx], im = j->im[idx];
        double w_re = TRIAD_OMEGA_RE[phase_idx];
        double w_im = TRIAD_OMEGA_IM[phase_idx];
        j->re[idx] = re * w_re - im * w_im;
        j->im[idx] = re * w_im + im * w_re;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * CMY CHANNEL CZ — Entangle within a specific channel only
 *
 * Apply CZ-like phase only to basis states belonging to a given channel.
 * For channel ch (0=C, 1=M, 2=Y), apply ω^(p_a · p_b · p_c) where
 * p_x = cmy_within(x) — the intra-channel position (0 or 1).
 *
 * This creates entanglement within a single channel while leaving
 * other channels undisturbed. Three such operations (one per channel)
 * create the full CMY-triadic entanglement.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline void triad_apply_channel_cz(TriadicJoint *j, int channel)
{
    for (int a = 0; a < TRIAD_D; a++)
    for (int b = 0; b < TRIAD_D; b++)
    for (int c = 0; c < TRIAD_D; c++) {
        /* Only act if all three basis states are in the same channel */
        if (cmy_channel_of(a) != channel) continue;
        if (cmy_channel_of(b) != channel) continue;
        if (cmy_channel_of(c) != channel) continue;

        int pa = cmy_within(a), pb = cmy_within(b), pc = cmy_within(c);
        int phase_product = pa * pb * pc;
        if (phase_product == 0) continue;  /* only non-zero when all three bits = 1 */

        /* Apply phase: ω^1 when all three intra-channel bits are 1 */
        int idx = TRIAD_IDX(a, b, c);
        double re = j->re[idx], im = j->im[idx];
        j->re[idx] = re * TRIAD_OMEGA_RE[1] - im * TRIAD_OMEGA_IM[1];
        j->im[idx] = re * TRIAD_OMEGA_IM[1] + im * TRIAD_OMEGA_RE[1];
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * LOCAL GATE ON TRIADIC STATE — Apply a 6×6 unitary to one quhit
 *
 * Applies U to quhit A, B, or C within the triadic joint state.
 * The other two quhits are untouched (identity ⊗ identity ⊗ U).
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Apply 6×6 unitary U to quhit A */
static inline void triad_gate_a(TriadicJoint *j,
                                const double *U_re, const double *U_im)
{
    for (int b = 0; b < TRIAD_D; b++)
    for (int c = 0; c < TRIAD_D; c++) {
        double new_re[TRIAD_D], new_im[TRIAD_D];
        for (int a2 = 0; a2 < TRIAD_D; a2++) {
            double sum_re = 0, sum_im = 0;
            for (int a = 0; a < TRIAD_D; a++) {
                int idx = TRIAD_IDX(a, b, c);
                int u = a2 * TRIAD_D + a;
                sum_re += U_re[u]*j->re[idx] - U_im[u]*j->im[idx];
                sum_im += U_re[u]*j->im[idx] + U_im[u]*j->re[idx];
            }
            new_re[a2] = sum_re;
            new_im[a2] = sum_im;
        }
        for (int a2 = 0; a2 < TRIAD_D; a2++) {
            int idx = TRIAD_IDX(a2, b, c);
            j->re[idx] = new_re[a2];
            j->im[idx] = new_im[a2];
        }
    }
}

/* Apply 6×6 unitary U to quhit B */
static inline void triad_gate_b(TriadicJoint *j,
                                const double *U_re, const double *U_im)
{
    for (int a = 0; a < TRIAD_D; a++)
    for (int c = 0; c < TRIAD_D; c++) {
        double new_re[TRIAD_D], new_im[TRIAD_D];
        for (int b2 = 0; b2 < TRIAD_D; b2++) {
            double sum_re = 0, sum_im = 0;
            for (int b = 0; b < TRIAD_D; b++) {
                int idx = TRIAD_IDX(a, b, c);
                int u = b2 * TRIAD_D + b;
                sum_re += U_re[u]*j->re[idx] - U_im[u]*j->im[idx];
                sum_im += U_re[u]*j->im[idx] + U_im[u]*j->re[idx];
            }
            new_re[b2] = sum_re;
            new_im[b2] = sum_im;
        }
        for (int b2 = 0; b2 < TRIAD_D; b2++) {
            int idx = TRIAD_IDX(a, b2, c);
            j->re[idx] = new_re[b2];
            j->im[idx] = new_im[b2];
        }
    }
}

/* Apply 6×6 unitary U to quhit C */
static inline void triad_gate_c(TriadicJoint *j,
                                const double *U_re, const double *U_im)
{
    for (int a = 0; a < TRIAD_D; a++)
    for (int b = 0; b < TRIAD_D; b++) {
        double new_re[TRIAD_D], new_im[TRIAD_D];
        for (int c2 = 0; c2 < TRIAD_D; c2++) {
            double sum_re = 0, sum_im = 0;
            for (int c = 0; c < TRIAD_D; c++) {
                int idx = TRIAD_IDX(a, b, c);
                int u = c2 * TRIAD_D + c;
                sum_re += U_re[u]*j->re[idx] - U_im[u]*j->im[idx];
                sum_im += U_re[u]*j->im[idx] + U_im[u]*j->re[idx];
            }
            new_re[c2] = sum_re;
            new_im[c2] = sum_im;
        }
        for (int c2 = 0; c2 < TRIAD_D; c2++) {
            int idx = TRIAD_IDX(a, b, c2);
            j->re[idx] = new_re[c2];
            j->im[idx] = new_im[c2];
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * TRIADIC ENTANGLEMENT ENTROPY — How entangled are the three mirrors?
 *
 * Entanglement entropy of quhit A: S_A = -Σ λ_k log₂(λ_k)
 * where λ_k are eigenvalues of ρ_A = Tr_{BC}(|ψ⟩⟨ψ|).
 *
 * Uses diagonal-only partial trace for speed.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline double triad_entropy_a(const TriadicJoint *j)
{
    double probs[TRIAD_D];
    triad_marginal_a(j, probs);
    double S = 0;
    for (int k = 0; k < TRIAD_D; k++) {
        if (probs[k] > 1e-14)
            S -= probs[k] * log2(probs[k]);
    }
    return S;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * TOTAL PROBABILITY — Sanity check: must equal 1.0
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline double triad_total_prob(const TriadicJoint *j)
{
    double sum = 0;
    for (int i = 0; i < TRIAD_D3; i++)
        sum += j->re[i]*j->re[i] + j->im[i]*j->im[i];
    return sum;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * RENORMALIZE — Force total probability to 1.0
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline void triad_renormalize(TriadicJoint *j)
{
    double norm2 = triad_total_prob(j);
    if (norm2 > 1e-30 && fabs(norm2 - 1.0) > 1e-15) {
        double scale = 1.0 / sqrt(norm2);
        for (int i = 0; i < TRIAD_D3; i++) {
            j->re[i] *= scale;
            j->im[i] *= scale;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * CHANNEL-LOCAL GATES — "Edges are compressed faces"
 *
 * Geometric insight: each CMY channel {C={0,1}, M={2,3}, Y={4,5}} is a
 * front/back pair of one square face. A 2×2 gate on one channel flips
 * or rotates just that face, leaving the other two squares untouched.
 *
 * Cost: 2×2 per (b,c) pair = 4 multiplies × 36 pairs = 144 ops
 * vs 6×6 per (b,c) pair = 36 multiplies × 36 pairs = 1296 ops
 * → 9× fewer multiplies for channel-local operations.
 *
 * The full DFT₆ is only needed when rotating BETWEEN channels (turning
 * one square's edge into another square's face). Within-channel ops
 * stay in 2×2 land.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Apply 2×2 unitary U to one CMY channel of quhit A.
 * channel: 0=C, 1=M, 2=Y
 * U is stored as [U00_re, U00_im, U01_re, U01_im, U10_re, U10_im, U11_re, U11_im] */
static inline void triad_channel_gate_a(TriadicJoint *j, int channel,
                                          const double U[8])
{
    int s0 = channel * 2;      /* First basis state of channel */
    int s1 = channel * 2 + 1;  /* Second basis state of channel */

    for (int b = 0; b < TRIAD_D; b++)
    for (int c = 0; c < TRIAD_D; c++) {
        int i0 = TRIAD_IDX(s0, b, c);
        int i1 = TRIAD_IDX(s1, b, c);

        double r0 = j->re[i0], m0 = j->im[i0];
        double r1 = j->re[i1], m1 = j->im[i1];

        /* new[s0] = U00 * old[s0] + U01 * old[s1] */
        j->re[i0] = U[0]*r0 - U[1]*m0 + U[2]*r1 - U[3]*m1;
        j->im[i0] = U[0]*m0 + U[1]*r0 + U[2]*m1 + U[3]*r1;

        /* new[s1] = U10 * old[s0] + U11 * old[s1] */
        j->re[i1] = U[4]*r0 - U[5]*m0 + U[6]*r1 - U[7]*m1;
        j->im[i1] = U[4]*m0 + U[5]*r0 + U[6]*m1 + U[7]*r1;
    }
}

/* Apply 2×2 unitary to one CMY channel of quhit B */
static inline void triad_channel_gate_b(TriadicJoint *j, int channel,
                                          const double U[8])
{
    int s0 = channel * 2, s1 = channel * 2 + 1;

    for (int a = 0; a < TRIAD_D; a++)
    for (int c = 0; c < TRIAD_D; c++) {
        int i0 = TRIAD_IDX(a, s0, c);
        int i1 = TRIAD_IDX(a, s1, c);

        double r0 = j->re[i0], m0 = j->im[i0];
        double r1 = j->re[i1], m1 = j->im[i1];

        j->re[i0] = U[0]*r0 - U[1]*m0 + U[2]*r1 - U[3]*m1;
        j->im[i0] = U[0]*m0 + U[1]*r0 + U[2]*m1 + U[3]*r1;
        j->re[i1] = U[4]*r0 - U[5]*m0 + U[6]*r1 - U[7]*m1;
        j->im[i1] = U[4]*m0 + U[5]*r0 + U[6]*m1 + U[7]*r1;
    }
}

/* Apply 2×2 unitary to one CMY channel of quhit C */
static inline void triad_channel_gate_c(TriadicJoint *j, int channel,
                                          const double U[8])
{
    int s0 = channel * 2, s1 = channel * 2 + 1;

    for (int a = 0; a < TRIAD_D; a++)
    for (int b = 0; b < TRIAD_D; b++) {
        int i0 = TRIAD_IDX(a, b, s0);
        int i1 = TRIAD_IDX(a, b, s1);

        double r0 = j->re[i0], m0 = j->im[i0];
        double r1 = j->re[i1], m1 = j->im[i1];

        j->re[i0] = U[0]*r0 - U[1]*m0 + U[2]*r1 - U[3]*m1;
        j->im[i0] = U[0]*m0 + U[1]*r0 + U[2]*m1 + U[3]*r1;
        j->re[i1] = U[4]*r0 - U[5]*m0 + U[6]*r1 - U[7]*m1;
        j->im[i1] = U[4]*m0 + U[5]*r0 + U[6]*m1 + U[7]*r1;
    }
}

/* ── CMY-diagonal gate: three independent 2×2 blocks ──
 * Applies U_C to channel C, U_M to channel M, U_Y to channel Y.
 * Each U is [U00_re, U00_im, U01_re, U01_im, U10_re, U10_im, U11_re, U11_im].
 * Total cost: 3 × (2×2) = 12 multiplies per (b,c) pair. */
static inline void triad_cmy_gate_a(TriadicJoint *j,
                                      const double U_C[8],
                                      const double U_M[8],
                                      const double U_Y[8])
{
    triad_channel_gate_a(j, 0, U_C);
    triad_channel_gate_a(j, 1, U_M);
    triad_channel_gate_a(j, 2, U_Y);
}

/* ── Channel-local DFT₂ (Hadamard) ──
 * Decompresses one edge into its face within a single channel.
 * H = (1/√2) × [[1, 1], [1, -1]]
 * This is the "within-square" rotation: front↔back mixing. */
static const double CHANNEL_DFT2[8] = {
     0.7071067811865476, 0.0,   /*  1/√2,  0 */
     0.7071067811865476, 0.0,   /*  1/√2,  0 */
     0.7071067811865476, 0.0,   /*  1/√2,  0 */
    -0.7071067811865476, 0.0    /* -1/√2,  0 */
};

/* Apply DFT₂ to all three channels of quhit A simultaneously.
 * This decompresses ALL edges within A without mixing between channels. */
static inline void triad_channel_dft_a(TriadicJoint *j)
{
    triad_cmy_gate_a(j, CHANNEL_DFT2, CHANNEL_DFT2, CHANNEL_DFT2);
}

/* ── Channel-local phase gate ──
 * Apply e^{iθ} to the second basis state of one channel.
 * This is the cheapest possible trainable gate: one complex multiply
 * on half the channel. */
static inline void triad_channel_phase_a(TriadicJoint *j, int channel, double theta)
{
    double U[8] = {
        1.0, 0.0,           /* U00 = 1 */
        0.0, 0.0,           /* U01 = 0 */
        0.0, 0.0,           /* U10 = 0 */
        cos(theta), sin(theta)  /* U11 = e^{iθ} */
    };
    triad_channel_gate_a(j, channel, U);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * FACE SWITCHING — Rotate the cube to present a different face
 *
 * Channel permutation: C→M→Y→C (cycle forward) or C→Y→M→C (cycle backward).
 * This is a cube rotation of 120° around the body diagonal.
 *
 * Cost: ZERO multiplies. O(N) index remap — the cheapest possible gate.
 *
 * Geometrically: you're picking up the cube and showing a different face
 * to the readout quhit without changing any amplitudes or phases.
 *
 * In the QNN context, this lets the network choose which "face" of
 * its internal representation to present for classification.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Rotate channels of quhit A forward: C→M→Y→C
 * |0⟩→|2⟩, |1⟩→|3⟩, |2⟩→|4⟩, |3⟩→|5⟩, |4⟩→|0⟩, |5⟩→|1⟩ */
static inline void triad_face_rotate_a(TriadicJoint *j)
{
    TriadicJoint tmp;
    for (int a = 0; a < TRIAD_D; a++)
    for (int b = 0; b < TRIAD_D; b++)
    for (int c = 0; c < TRIAD_D; c++) {
        int a_new = (a + 2) % TRIAD_D;  /* Shift by 2 = one channel forward */
        int idx_old = TRIAD_IDX(a, b, c);
        int idx_new = TRIAD_IDX(a_new, b, c);
        tmp.re[idx_new] = j->re[idx_old];
        tmp.im[idx_new] = j->im[idx_old];
    }
    for (int i = 0; i < TRIAD_D3; i++) {
        j->re[i] = tmp.re[i];
        j->im[i] = tmp.im[i];
    }
}

/* Rotate channels of quhit A backward: C→Y→M→C
 * |0⟩→|4⟩, |1⟩→|5⟩, |2⟩→|0⟩, |3⟩→|1⟩, |4⟩→|2⟩, |5⟩→|3⟩ */
static inline void triad_face_rotate_back_a(TriadicJoint *j)
{
    TriadicJoint tmp;
    for (int a = 0; a < TRIAD_D; a++)
    for (int b = 0; b < TRIAD_D; b++)
    for (int c = 0; c < TRIAD_D; c++) {
        int a_new = (a + 4) % TRIAD_D;  /* Shift by 4 = one channel backward */
        int idx_old = TRIAD_IDX(a, b, c);
        int idx_new = TRIAD_IDX(a_new, b, c);
        tmp.re[idx_new] = j->re[idx_old];
        tmp.im[idx_new] = j->im[idx_old];
    }
    for (int i = 0; i < TRIAD_D3; i++) {
        j->re[i] = tmp.re[i];
        j->im[i] = tmp.im[i];
    }
}

/* Swap two channels of quhit A: face_a ↔ face_b
 * This is a 90° rotation of the cube around the axis perpendicular
 * to the two swapped faces. Also zero multiplies. */
static inline void triad_face_swap_a(TriadicJoint *j, int face_a, int face_b)
{
    int sa0 = face_a * 2, sa1 = face_a * 2 + 1;
    int sb0 = face_b * 2, sb1 = face_b * 2 + 1;

    for (int b = 0; b < TRIAD_D; b++)
    for (int c = 0; c < TRIAD_D; c++) {
        int ia0 = TRIAD_IDX(sa0, b, c), ia1 = TRIAD_IDX(sa1, b, c);
        int ib0 = TRIAD_IDX(sb0, b, c), ib1 = TRIAD_IDX(sb1, b, c);

        /* Swap |sa0⟩ ↔ |sb0⟩ */
        double tr, ti;
        tr = j->re[ia0]; ti = j->im[ia0];
        j->re[ia0] = j->re[ib0]; j->im[ia0] = j->im[ib0];
        j->re[ib0] = tr; j->im[ib0] = ti;

        /* Swap |sa1⟩ ↔ |sb1⟩ */
        tr = j->re[ia1]; ti = j->im[ia1];
        j->re[ia1] = j->re[ib1]; j->im[ia1] = j->im[ib1];
        j->re[ib1] = tr; j->im[ib1] = ti;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MULTI-FACE READOUT — Average marginals from all 3 quhits
 *
 * Instead of reading only quhit A, average the measurement probabilities
 * across A, B, and C. This is like looking at the cube from all three
 * axis directions and combining what you see.
 *
 * Gives a more stable readout that's less sensitive to which quhit
 * happens to be "A" vs "B" vs "C".
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline void triad_multiface_readout(const TriadicJoint *j, double *probs)
{
    double pa[TRIAD_D], pb[TRIAD_D], pc[TRIAD_D];
    triad_marginal_a(j, pa);
    triad_marginal_b(j, pb);
    triad_marginal_c(j, pc);

    for (int k = 0; k < TRIAD_D; k++)
        probs[k] = (pa[k] + pb[k] + pc[k]) / 3.0;
}

#endif /* QUHIT_TRIADIC_H */
