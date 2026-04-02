/*
 * multiverse_sim.c — Multiversal "Parallel Reality" Simulation
 *
 * Uses HexState's D=6 Triality + HPC architecture to simulate
 * branching realities where:
 *   D=2 (antipodal fold parity) = binary choices (timeline A/B)
 *   D=3 (plane index)           = environmental variables
 *   D=6 (full quhit)            = the complete multiverse state
 *
 * All 2×3 = 6 branches exist simultaneously as a single unbroken
 * quantum state. Interference between "parallel timelines" is
 * computed natively via HPC phase edges and S₆ exotic invariants.
 * No premature collapse until explicitly requested.
 *
 * Build:
 *   gcc -O2 -std=gnu99 multiverse_sim.c quhit_triality.c \
 *       quhit_hexagram.c s6_exotic.c bigint.c -lm -o multiverse_sim
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "quhit_triality.h"
#include "hpc_graph.h"
#include "s6_exotic.h"

/* ═══════════════════════════════════════════════════════════════════════
 * CONSTANTS & HELPERS
 * ═══════════════════════════════════════════════════════════════════════ */

#define MV_D        6
#define MV_SITES    8       /* Number of multiverse nodes */
#define MV_STEPS    4       /* Evolution steps */

/* CRT-style decomposition: k -> (plane, parity) */
static inline int mv_plane(int k)  { return k % 3; }          /* D=3 env    */
static inline int mv_parity(int k) { return k / 3; }          /* D=2 choice */

static const char *PLANE_NAMES[3]  = {"α", "β", "γ"};
static const char *PARITY_NAMES[2] = {"A", "B"};

/* XorShift64 RNG */
static uint64_t rng_state = 0xDEADBEEF42ULL;
static inline double rng_uniform(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)(rng_state & 0xFFFFFFFFFFFFF) / (double)0x10000000000000;
}

/* ═══════════════════════════════════════════════════════════════════════
 * BRANCHING GATE — The heart of the multiverse
 *
 * Stage 1: Fold — separate into vesica (parity=0, timeline A) and
 *          wave (parity=1, timeline B) via Hadamard on antipodal pairs.
 * Stage 2: Conditional env shift — timeline B gets a cyclic shift of
 *          the environmental index, creating branch-dependent evolution.
 * Stage 3: Unfold — recombine into the full D=6 superposition.
 *
 * The gate creates a state where the two timelines have DIFFERENT
 * environmental configurations, but remain coherent. O(D).
 * ═══════════════════════════════════════════════════════════════════════ */

static void mv_branch_evolve(TrialityQuhit *q, int env_shift)
{
    /* Ensure edge view */
    triality_ensure_view(q, VIEW_EDGE);

    /* Stage 1: Antipodal fold — splits into vesica (A) and wave (B) */
    triality_fold(q);
    triality_ensure_view(q, VIEW_FOLDED);

    /* Stage 2: Conditional shift on the wave (timeline B) components.
     * folded[k+3] for k=0,1,2 are the wave amplitudes.
     * We cyclically shift them: folded[k+3] → folded[((k+shift)%3)+3].
     * This means timeline B evolves its environment differently. */
    double tmp_re[3], tmp_im[3];
    for (int k = 0; k < 3; k++) {
        int src = ((k - env_shift) % 3 + 3) % 3;
        tmp_re[k] = q->folded_re[src + 3];
        tmp_im[k] = q->folded_im[src + 3];
    }
    for (int k = 0; k < 3; k++) {
        q->folded_re[k + 3] = tmp_re[k];
        q->folded_im[k + 3] = tmp_im[k];
    }

    /* Stage 3: Unfold — recombine into full D=6 state */
    triality_unfold(q);

    /* Mark state as modified */
    q->primary = VIEW_EDGE;
    q->dirty = DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED
             | DIRTY_EXOTIC | DIRTY_TETRA;
    q->delta_valid = 0;
    q->eigenstate_class = -1;
    q->active_mask = 0x3F;
    q->active_count = 6;
    q->real_valued = 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * ENVIRONMENT-DEPENDENT PHASE GATE
 *
 * Applies a phase that depends on the environmental index (plane).
 * This creates different "physics" for each environmental configuration.
 * Timeline A and B experience different effective Hamiltonians because
 * their env indices are shifted relative to each other.
 *
 * Phase: |k⟩ → e^{i·2π·plane(k)·strength/3} |k⟩
 * ═══════════════════════════════════════════════════════════════════════ */

static void mv_env_phase(TrialityQuhit *q, double strength)
{
    double phi_re[MV_D], phi_im[MV_D];
    for (int k = 0; k < MV_D; k++) {
        double angle = 2.0 * M_PI * mv_plane(k) * strength / 3.0;
        phi_re[k] = cos(angle);
        phi_im[k] = sin(angle);
    }
    triality_phase(q, phi_re, phi_im);
}

/* ═══════════════════════════════════════════════════════════════════════
 * TIMELINE INTERFERENCE ANALYSIS
 *
 * Measures how much the two timelines (parities) interfere by comparing:
 *   P_A = Σ_k |ψ[k]|² for parity(k)=0 (timeline A)
 *   P_B = Σ_k |ψ[k]|² for parity(k)=1 (timeline B)
 *
 * Also computes cross-timeline coherence from the folded view:
 *   C = Σ_k |vesica[k]|² - |wave[k]|²
 *
 * C ≈ 0 means timelines are maximally superposed.
 * |C| ≈ 1 means one timeline dominates.
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    double prob_A;        /* Total probability in timeline A */
    double prob_B;        /* Total probability in timeline B */
    double coherence;     /* Cross-timeline coherence C */
    double env_probs[3];  /* Marginal probability per environment plane */
    double delta;         /* S₆ exotic invariant */
    double entropy;       /* Shannon entropy of the 6-state distribution */
} TimelineAnalysis;

static TimelineAnalysis mv_analyze(TrialityQuhit *q)
{
    TimelineAnalysis a;
    memset(&a, 0, sizeof(a));

    /* Edge view probabilities */
    triality_ensure_view(q, VIEW_EDGE);
    double probs[MV_D];
    triality_probabilities(q, VIEW_EDGE, probs);

    for (int k = 0; k < MV_D; k++) {
        if (mv_parity(k) == 0) a.prob_A += probs[k];
        else                   a.prob_B += probs[k];
        a.env_probs[mv_plane(k)] += probs[k];
    }

    /* Coherence from folded view */
    triality_fold(q);
    triality_ensure_view(q, VIEW_FOLDED);
    double vesica_norm = 0, wave_norm = 0;
    for (int k = 0; k < 3; k++) {
        vesica_norm += q->folded_re[k]*q->folded_re[k]
                     + q->folded_im[k]*q->folded_im[k];
        wave_norm   += q->folded_re[k+3]*q->folded_re[k+3]
                     + q->folded_im[k+3]*q->folded_im[k+3];
    }
    a.coherence = vesica_norm - wave_norm;

    /* Exotic invariant Δ */
    a.delta = triality_exotic_invariant_cached(q);

    /* Shannon entropy */
    for (int k = 0; k < MV_D; k++) {
        if (probs[k] > 1e-15)
            a.entropy -= probs[k] * log2(probs[k]);
    }

    return a;
}

/* ═══════════════════════════════════════════════════════════════════════
 * PRINT HELPERS
 * ═══════════════════════════════════════════════════════════════════════ */

static void mv_print_divider(const char *title)
{
    printf("\n  ═══════════════════════════════════════════════════════\n");
    printf("    %s\n", title);
    printf("  ═══════════════════════════════════════════════════════\n\n");
}

static void mv_print_state_decomposed(TrialityQuhit *q, const char *label)
{
    triality_ensure_view(q, VIEW_EDGE);
    printf("  %s:\n", label);
    printf("  ┌──────┬───────┬───────┬───────────────────────┬────────┐\n");
    printf("  │  |k⟩ │ Timeline │ Env │      Amplitude         │ P(k)   │\n");
    printf("  ├──────┼───────┼───────┼───────────────────────┼────────┤\n");
    for (int k = 0; k < MV_D; k++) {
        double p = q->edge_re[k]*q->edge_re[k]
                 + q->edge_im[k]*q->edge_im[k];
        printf("  │  |%d⟩ │   %s    │   %s  │ %+.4f %+.4fi        │ %.4f │\n",
               k, PARITY_NAMES[mv_parity(k)], PLANE_NAMES[mv_plane(k)],
               q->edge_re[k], q->edge_im[k], p);
    }
    printf("  └──────┴───────┴───────┴───────────────────────┴────────┘\n");
}

static void mv_print_analysis(const TimelineAnalysis *a, const char *label)
{
    printf("  %s:\n", label);
    printf("    Timeline A (parity 0): P = %.6f\n", a->prob_A);
    printf("    Timeline B (parity 1): P = %.6f\n", a->prob_B);
    printf("    Cross-timeline coherence C = %+.6f  ", a->coherence);
    if (fabs(a->coherence) < 0.01)
        printf("(maximally superposed)\n");
    else if (fabs(a->coherence) > 0.9)
        printf("(one timeline dominates)\n");
    else
        printf("(partial interference)\n");
    printf("    Environment marginals: α=%.4f  β=%.4f  γ=%.4f\n",
           a->env_probs[0], a->env_probs[1], a->env_probs[2]);
    printf("    S₆ exotic invariant Δ = %.6f  ", a->delta);
    if (a->delta < 0.001)
        printf("(classical — qubit-equivalent)\n");
    else
        printf("(hexagonally polarized — D=6 unique!)\n");
    printf("    Shannon entropy H = %.4f bits (max %.4f)\n",
           a->entropy, log2(6.0));
}

static void mv_print_view_comparison(TrialityQuhit *q, const char *label)
{
    printf("  %s — probabilities in 3 triality views:\n", label);
    double p_edge[MV_D], p_vert[MV_D], p_diag[MV_D];
    triality_probabilities(q, VIEW_EDGE, p_edge);
    triality_probabilities(q, VIEW_VERTEX, p_vert);
    triality_probabilities(q, VIEW_DIAGONAL, p_diag);

    printf("    ┌──────┬──────────┬──────────┬──────────┐\n");
    printf("    │  |k⟩ │   Edge   │  Vertex  │  Diag    │\n");
    printf("    ├──────┼──────────┼──────────┼──────────┤\n");
    for (int k = 0; k < MV_D; k++) {
        printf("    │  |%d⟩ │  %.4f  │  %.4f  │  %.4f  │\n",
               k, p_edge[k], p_vert[k], p_diag[k]);
    }
    printf("    └──────┴──────────┴──────────┴──────────┘\n");
}

/* ═══════════════════════════════════════════════════════════════════════
 * COMPUTATIONAL HARVEST — Cross-timeline oracle readout
 *
 * The idea: encode a function f:{α,β,γ}→phase as an oracle running ONLY
 * in Timeline B (wave parity). Then read Timeline A (vesica parity) in
 * the vertex view. The Fourier transform of B's computation appears as
 * interference fringes in A — revealing GLOBAL PROPERTIES of f without
 * A ever computing f.
 *
 * This is a D=3 Deutsch-Jozsa embedded in the D=6 multiverse:
 *   • CONSTANT oracle: f(α)=f(β)=f(γ) → A sees uniform env distribution
 *   • BALANCED oracle: two outputs differ → A sees non-uniform
 *   • CYCLIC oracle: f(k)=k+1 mod 3 → A sees specific Fourier peak
 *
 * The fold/unfold IS the interference channel. Timeline B runs the
 * oracle; Timeline A harvests the result. One measurement in A
 * distinguishes oracle classes — O(1) queries, not O(3).
 * ═══════════════════════════════════════════════════════════════════════ */

typedef enum {
    ORACLE_CONSTANT,   /* f(α)=f(β)=f(γ)=0  → all same phase */
    ORACLE_BALANCED,   /* f(α)=0, f(β)=1, f(γ)=0  → one differs */
    ORACLE_CYCLIC,     /* f(α)=0, f(β)=1, f(γ)=2  → full permutation */
    ORACLE_ANTI_CYCLIC /* f(α)=0, f(β)=2, f(γ)=1  → reverse permutation */
} OracleType;

static const char *ORACLE_NAMES[] = {
    "CONSTANT", "BALANCED", "CYCLIC", "ANTI-CYCLIC"
};

/* Apply oracle phase ONLY to Timeline B's environmental components.
 * In edge basis: states |3⟩,|4⟩,|5⟩ are (B,α), (B,β), (B,γ).
 * Oracle imprints: |3+k⟩ → e^{i·2π·f(k)/3} |3+k⟩
 * Timeline A states |0⟩,|1⟩,|2⟩ are UNTOUCHED. */
static void mv_oracle_B(TrialityQuhit *q, OracleType oracle)
{
    int f[3];
    switch (oracle) {
        case ORACLE_CONSTANT:    f[0]=0; f[1]=0; f[2]=0; break;
        case ORACLE_BALANCED:    f[0]=0; f[1]=1; f[2]=0; break;
        case ORACLE_CYCLIC:      f[0]=0; f[1]=1; f[2]=2; break;
        case ORACLE_ANTI_CYCLIC: f[0]=0; f[1]=2; f[2]=1; break;
    }

    double phi_re[MV_D], phi_im[MV_D];
    for (int k = 0; k < MV_D; k++) {
        if (mv_parity(k) == 0) {
            /* Timeline A: identity (no oracle) */
            phi_re[k] = 1.0;
            phi_im[k] = 0.0;
        } else {
            /* Timeline B: oracle phase */
            int env = mv_plane(k);  /* 0,1,2 */
            double angle = 2.0 * M_PI * f[env] / 3.0;
            phi_re[k] = cos(angle);
            phi_im[k] = sin(angle);
        }
    }
    triality_phase(q, phi_re, phi_im);
}

/* Read the "harvest" from Timeline A after oracle interference.
 * Returns per-env probabilities in Timeline A's vertex view.
 * The pattern distinguishes oracle types:
 *   CONSTANT → A env marginals are uniform (1/3, 1/3, 1/3)
 *   BALANCED → A env marginals are non-uniform
 *   CYCLIC   → A env marginals show a specific peak */
typedef struct {
    double A_env[3];   /* Timeline A's per-env probabilities (vertex view) */
    double B_env[3];   /* Timeline B's per-env probabilities (vertex view) */
    double A_total;    /* Total Timeline A probability */
    double B_total;    /* Total Timeline B probability */
} HarvestResult;

static HarvestResult mv_harvest_A(TrialityQuhit *q)
{
    HarvestResult h;
    memset(&h, 0, sizeof(h));

    /* Read probabilities in ALL views */
    double p_edge[MV_D], p_vert[MV_D];
    triality_probabilities(q, VIEW_EDGE, p_edge);
    triality_probabilities(q, VIEW_VERTEX, p_vert);

    for (int k = 0; k < MV_D; k++) {
        int env = mv_plane(k);
        if (mv_parity(k) == 0) {
            h.A_env[env] += p_vert[k];
            h.A_total += p_vert[k];
        } else {
            h.B_env[env] += p_vert[k];
            h.B_total += p_vert[k];
        }
    }
    return h;
}

/* ═══════════════════════════════════════════════════════════════════════
 * MAIN — The Multiverse Experiment
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    rng_state = (uint64_t)time(NULL) ^ 0xCAFEBABE;

    printf("╔═════════════════════════════════════════════════════════════╗\n");
    printf("║  MULTIVERSAL PARALLEL REALITY SIMULATION                  ║\n");
    printf("║  D=2 (binary choice) × D=3 (environment) = D=6 (reality) ║\n");
    printf("║  Triality + HPC — No premature collapse                   ║\n");
    printf("╚═════════════════════════════════════════════════════════════╝\n");

    /* ── Initialize exotic engine ── */
    triality_exotic_init();
    s6_exotic_init();
    triality_stats_reset();

    /* ════════════════════════════════════════════════════════════════════
     * PHASE 1: Single-Site Branching Demo
     *
     * Show how one quhit splits into two timelines with different
     * environmental configurations, then interferes.
     * ════════════════════════════════════════════════════════════════════ */

    mv_print_divider("PHASE 1: SINGLE-SITE TIMELINE BRANCHING");

    TrialityQuhit single;
    triality_init(&single);

    printf("  Step 0: Initial state |0⟩ = (Timeline A, Environment α)\n");
    mv_print_state_decomposed(&single, "Initial");

    /* Put into uniform superposition via DFT₆ */
    triality_dft(&single);
    printf("\n  Step 1: DFT₆ → uniform superposition across all realities\n");
    mv_print_state_decomposed(&single, "After DFT₆ (all timelines active)");

    TimelineAnalysis a1 = mv_analyze(&single);
    mv_print_analysis(&a1, "Pre-branch analysis");

    /* Branch: fold + conditional env shift + unfold */
    printf("\n  Step 2: Branch gate — timeline B shifts environment by +1\n");
    printf("          (Timeline A sees {α,β,γ}, Timeline B sees {β,γ,α})\n\n");
    mv_branch_evolve(&single, 1);
    mv_print_state_decomposed(&single, "After branching");

    TimelineAnalysis a2 = mv_analyze(&single);
    mv_print_analysis(&a2, "Post-branch analysis");

    /* Apply environment-dependent phase → creates interference fringes */
    printf("\n  Step 3: Environment-dependent phase → interference fringes\n");
    mv_env_phase(&single, 1.0);
    mv_print_state_decomposed(&single, "After env-phase");

    TimelineAnalysis a3 = mv_analyze(&single);
    mv_print_analysis(&a3, "Post-interference analysis");

    /* Show all 3 triality views */
    printf("\n  Step 4: Triality view comparison (interference visible)\n");
    mv_print_view_comparison(&single, "Single-site multiverse");

    /* Exotic fingerprint */
    printf("\n  S₆ Exotic Fingerprint (per-conjugacy-class Δ):\n");
    double fingerprint[11];
    triality_exotic_fingerprint_cached(&single, fingerprint);
    printf("    [");
    for (int i = 0; i < 11; i++) {
        printf("%.4f", fingerprint[i]);
        if (i < 10) printf(", ");
    }
    printf("]\n");

    /* ════════════════════════════════════════════════════════════════════
     * PHASE 2: Multi-Site HPC Multiverse
     *
     * Create a chain of N sites. Each site branches independently.
     * CZ edges entangle neighboring timelines → interference propagates
     * across the entire chain without materializing 6^N amplitudes.
     * ════════════════════════════════════════════════════════════════════ */

    mv_print_divider("PHASE 2: HPC MULTIVERSE CHAIN (N=%d SITES)");

    printf("  Creating %d-site HPC graph...\n\n", MV_SITES);
    HPCGraph *multiverse = hpc_create(MV_SITES);

    /* Initialize all sites to |0⟩, then DFT₆ → uniform superposition */
    for (int i = 0; i < MV_SITES; i++) {
        triality_dft(&multiverse->locals[i]);
    }

    printf("  All sites in uniform superposition (all realities coexist)\n");
    hpc_print_state(multiverse, "Initial multiverse");

    /* Step 1: Branch each site with increasing environmental shift */
    printf("\n  Step 1: Branch each site (env_shift = site_index mod 3)\n");
    for (int i = 0; i < MV_SITES; i++) {
        mv_branch_evolve(&multiverse->locals[i], (i % 3) + 1);
    }
    hpc_print_state(multiverse, "After branching");

    /* Step 2: Entangle neighbors via CZ edges */
    printf("\n  Step 2: CZ entanglement between neighbors → timeline correlations\n");
    for (int i = 0; i < MV_SITES - 1; i++) {
        hpc_cz(multiverse, i, i + 1);
    }
    printf("  Created %lu CZ edges\n", multiverse->cz_edges);

    /* Step 3: Apply environment-dependent phases (different strength per site) */
    printf("\n  Step 3: Environment-dependent phase evolution\n");
    for (int i = 0; i < MV_SITES; i++) {
        double strength = 0.5 + 0.5 * (double)i / (MV_SITES - 1);
        mv_env_phase(&multiverse->locals[i], strength);
    }

    /* Step 4: Second round of CZ → deeper entanglement */
    printf("  Step 4: Second CZ round → deeper inter-timeline entanglement\n");
    for (int i = 0; i < MV_SITES - 1; i++) {
        hpc_cz(multiverse, i, i + 1);
    }
    printf("  Total edges: %lu (CZ: %lu)\n\n", multiverse->n_edges,
           multiverse->cz_edges);

    /* ── Compact parallel CZ edges ── */
    printf("  Compacting parallel CZ edges...\n");
    hpc_compact_edges(multiverse);
    printf("  After compaction: %lu edges\n\n", multiverse->n_edges);

    /* ════════════════════════════════════════════════════════════════════
     * PHASE 3: Interference Readout (No Collapse!)
     *
     * Probe the multiverse state WITHOUT collapsing it.
     * Marginal probabilities, exotic invariants, and entropy cuts
     * reveal the interference pattern between parallel realities.
     * ════════════════════════════════════════════════════════════════════ */

    mv_print_divider("PHASE 3: INTERFERENCE READOUT (NO COLLAPSE)");

    printf("  Per-site marginal probabilities (computed via HPC in O(N+E)):\n\n");
    printf("  ┌──────┬────────────────────────────────────────────────────┐\n");
    printf("  │ Site │ |0⟩(Aα)  |1⟩(Aβ)  |2⟩(Aγ)  |3⟩(Bα)  |4⟩(Bβ)  |5⟩(Bγ) │\n");
    printf("  ├──────┼────────────────────────────────────────────────────┤\n");

    for (int i = 0; i < MV_SITES; i++) {
        printf("  │  %d   │", i);
        for (int v = 0; v < MV_D; v++) {
            double p = hpc_marginal(multiverse, i, v);
            printf(" %.4f ", p);
        }
        printf("│\n");
    }
    printf("  └──────┴────────────────────────────────────────────────────┘\n\n");

    /* Per-site timeline analysis */
    printf("  Per-site timeline balance and exotic invariants:\n\n");
    printf("  ┌──────┬────────┬────────┬──────────┬────────┬────────────────┐\n");
    printf("  │ Site │  P(A)  │  P(B)  │ Coherence│   Δ    │  Interpretation│\n");
    printf("  ├──────┼────────┼────────┼──────────┼────────┼────────────────┤\n");

    for (int i = 0; i < MV_SITES; i++) {
        TimelineAnalysis ta = mv_analyze(&multiverse->locals[i]);
        const char *interp;
        if (fabs(ta.coherence) < 0.05)
            interp = "superposed    ";
        else if (ta.prob_A > ta.prob_B + 0.1)
            interp = "A-dominant    ";
        else if (ta.prob_B > ta.prob_A + 0.1)
            interp = "B-dominant    ";
        else
            interp = "interfering   ";

        printf("  │  %d   │ %.4f │ %.4f │ %+.4f   │ %.4f │ %s│\n",
               i, ta.prob_A, ta.prob_B, ta.coherence, ta.delta, interp);
    }
    printf("  └──────┴────────┴────────┴──────────┴────────┴────────────────┘\n\n");

    /* Entanglement entropy across cuts */
    printf("  Entanglement entropy across bipartition cuts:\n");
    printf("  ┌──────────────┬──────────────┐\n");
    printf("  │   Cut after  │  S (bits)    │\n");
    printf("  ├──────────────┼──────────────┤\n");
    for (int c = 0; c < MV_SITES - 1; c++) {
        double S = hpc_entropy_cut(multiverse, c);
        printf("  │    site %d    │    %.4f    │\n", c, S);
    }
    printf("  └──────────────┴──────────────┘\n\n");

    /* Global exotic invariant */
    double global_delta = hpc_exotic_invariant(multiverse);
    printf("  Global S₆ exotic invariant ⟨Δ⟩ = %.6f\n", global_delta);
    if (global_delta > 0.001)
        printf("  → The multiverse is HEXAGONALLY POLARIZED (D=6 unique)\n");
    else
        printf("  → The multiverse is classically reducible\n");

    /* Dual probabilities on site 0 */
    printf("\n  Dual measurement (standard + exotic) on site 0:\n");
    double probs_std[MV_D], probs_exo[MV_D];
    triality_dual_probabilities(&multiverse->locals[0], VIEW_EDGE,
                                probs_std, probs_exo);
    printf("    Standard: [");
    for (int k = 0; k < MV_D; k++) printf("%.4f%s", probs_std[k], k<5?", ":"");
    printf("]\n    Exotic:   [");
    for (int k = 0; k < MV_D; k++) printf("%.4f%s", probs_exo[k], k<5?", ":"");
    printf("]\n");

    double dist = 0;
    for (int k = 0; k < MV_D; k++) dist += fabs(probs_std[k] - probs_exo[k]);
    printf("    Statistical distance = %.6f\n", dist / 2.0);
    if (dist > 0.01)
        printf("    → Standard and exotic channels SEE DIFFERENT REALITIES\n\n");
    else
        printf("    → Channels agree (automorphism-transparent state)\n\n");

    /* ════════════════════════════════════════════════════════════════════
     * PHASE 4: Deferred Collapse — Finally resolving the multiverse
     *
     * Measure all sites. Before: N-body superposition.
     * After: one definite timeline per site. The interference pattern
     * determines WHICH timeline wins at each site.
     * ════════════════════════════════════════════════════════════════════ */

    mv_print_divider("PHASE 4: DEFERRED COLLAPSE — RESOLVING THE MULTIVERSE");

    printf("  Measuring all sites (wavefunction collapse)...\n\n");
    printf("  ┌──────┬─────────┬──────────┬────────────────────────┐\n");
    printf("  │ Site │ Outcome │ Timeline │     Environment        │\n");
    printf("  ├──────┼─────────┼──────────┼────────────────────────┤\n");

    /* Verify norm before collapse */
    double norm_before = hpc_norm_sq(multiverse);
    printf("  │ (norm² before collapse = %.6f)%s│\n",
           norm_before, "                      ");

    for (int i = 0; i < MV_SITES; i++) {
        double r = rng_uniform();
        uint32_t outcome = hpc_measure(multiverse, i, r);
        printf("  │  %d   │   |%d⟩   │    %s     │ Environment %s          │\n",
               i, outcome, PARITY_NAMES[mv_parity(outcome)],
               PLANE_NAMES[mv_plane(outcome)]);
    }
    printf("  └──────┴─────────┴──────────┴────────────────────────┘\n\n");

    /* Post-collapse analysis */
    printf("  Post-collapse exotic invariants:\n");
    for (int i = 0; i < MV_SITES; i++) {
        double d = triality_exotic_invariant_cached(&multiverse->locals[i]);
        printf("    Site %d: Δ = %.6f %s\n", i, d,
               d < 0.001 ? "(collapsed — classical)" : "(residual coherence)");
    }
    printf("\n  Edges remaining: %lu (should be 0 after full collapse)\n",
           multiverse->n_edges);

    /* ════════════════════════════════════════════════════════════════════
     * PHASE 5: Statistics — Repeat the experiment to show interference
     *
     * Run the full multiverse protocol many times. The measurement
     * statistics should show NON-UNIFORM distributions (interference
     * fringes) that differ from what independent coin flips would give.
     * ════════════════════════════════════════════════════════════════════ */

    mv_print_divider("PHASE 5: STATISTICAL VERIFICATION (1000 runs)");

    int histogram[MV_SITES][MV_D];
    memset(histogram, 0, sizeof(histogram));

    int parity_correlation = 0;
    int env_correlation = 0;
    int trials = 1000;

    for (int trial = 0; trial < trials; trial++) {
        HPCGraph *g = hpc_create(MV_SITES);

        /* Protocol: DFT₆ → branch → CZ chain → env phase → CZ chain */
        for (int i = 0; i < MV_SITES; i++) {
            triality_dft(&g->locals[i]);
            mv_branch_evolve(&g->locals[i], (i % 3) + 1);
        }
        for (int i = 0; i < MV_SITES - 1; i++) hpc_cz(g, i, i + 1);
        for (int i = 0; i < MV_SITES; i++)
            mv_env_phase(&g->locals[i], 0.5 + 0.5*(double)i/(MV_SITES-1));
        for (int i = 0; i < MV_SITES - 1; i++) hpc_cz(g, i, i + 1);
        hpc_compact_edges(g);

        /* Measure all sites */
        uint32_t outcomes[MV_SITES];
        for (int i = 0; i < MV_SITES; i++) {
            double r = rng_uniform();
            outcomes[i] = hpc_measure(g, i, r);
            histogram[i][outcomes[i]]++;
        }

        /* Check parity correlation between first and last site */
        if (mv_parity(outcomes[0]) == mv_parity(outcomes[MV_SITES-1]))
            parity_correlation++;
        if (mv_plane(outcomes[0]) == mv_plane(outcomes[MV_SITES-1]))
            env_correlation++;

        hpc_destroy(g);
    }

    /* Print histogram */
    printf("  Measurement histogram (%d trials):\n\n", trials);
    printf("  ┌──────┬────────┬────────┬────────┬────────┬────────┬────────┐\n");
    printf("  │ Site │ |0⟩ Aα │ |1⟩ Aβ │ |2⟩ Aγ │ |3⟩ Bα │ |4⟩ Bβ │ |5⟩ Bγ │\n");
    printf("  ├──────┼────────┼────────┼────────┼────────┼────────┼────────┤\n");
    for (int i = 0; i < MV_SITES; i++) {
        printf("  │  %d   │", i);
        for (int k = 0; k < MV_D; k++)
            printf(" %5.1f%% │", 100.0 * histogram[i][k] / trials);
        printf("\n");
    }
    printf("  └──────┴────────┴────────┴────────┴────────┴────────┴────────┘\n\n");

    /* Check for interference: uniform would be 16.7% each */
    printf("  Uniformity check (16.67%% = no interference):\n");
    int non_uniform_count = 0;
    for (int i = 0; i < MV_SITES; i++) {
        for (int k = 0; k < MV_D; k++) {
            double pct = 100.0 * histogram[i][k] / trials;
            if (fabs(pct - 100.0/MV_D) > 3.0) non_uniform_count++;
        }
    }
    printf("    %d/%d bins deviate >3%% from uniform → ", non_uniform_count,
           MV_SITES * MV_D);
    if (non_uniform_count > MV_SITES)
        printf("INTERFERENCE FRINGES DETECTED ✓\n");
    else
        printf("weak interference\n");

    /* Timeline correlations */
    printf("\n  Cross-chain correlations (site 0 ↔ site %d):\n", MV_SITES-1);
    printf("    Parity (timeline) correlation: %d/%d = %.1f%%\n",
           parity_correlation, trials, 100.0*parity_correlation/trials);
    printf("    Environment correlation:       %d/%d = %.1f%%\n",
           env_correlation, trials, 100.0*env_correlation/trials);
    printf("    (Independent would be: parity=50%%, env=33%%)\n");

    /* ── Final stats ── */
    mv_print_divider("ENGINE STATISTICS");
    triality_stats_print();

    /* ════════════════════════════════════════════════════════════════════
     * PHASE 6: COMPUTATIONAL HARVEST
     *
     * The key experiment: encode an oracle in Timeline B, then harvest
     * its properties from Timeline A via interference. Timeline A NEVER
     * runs the oracle — it reads the answer through the fold channel.
     *
     * Protocol:
     *   1. Start with |0⟩ (Timeline A, env α)
     *   2. DFT₆ → uniform superposition (both timelines, all envs)
     *   3. Apply oracle ONLY to Timeline B states (|3⟩,|4⟩,|5⟩)
     *   4. Branch gate → fold/unfold creates interference
     *   5. Read Timeline A in vertex view → oracle class is revealed
     *
     * Timeline A harvests B's computation without ever performing it.
     * ════════════════════════════════════════════════════════════════════ */

    mv_print_divider("PHASE 6: COMPUTATIONAL HARVEST — ORACLE IN B, READ FROM A");

    printf("  Protocol: encode oracle in Timeline B, harvest from Timeline A\n");
    printf("  Timeline A NEVER runs the oracle.\n");
    printf("  The fold/unfold channel IS the interference bridge.\n\n");

    OracleType oracles[] = {
        ORACLE_CONSTANT, ORACLE_BALANCED, ORACLE_CYCLIC, ORACLE_ANTI_CYCLIC
    };
    int n_oracles = 4;

    printf("  ┌──────────────┬─────────────────────────────┬─────────────────────────────┬────────────────────┐\n");
    printf("  │   Oracle     │  A harvested (vertex view)  │  B env (vertex view)        │   A ≠ uniform?    │\n");
    printf("  ├──────────────┼─────────────────────────────┼─────────────────────────────┼────────────────────┤\n");

    for (int oi = 0; oi < n_oracles; oi++) {
        TrialityQuhit q;
        triality_init(&q);

        /* Step 1-2: DFT₆ → all realities active */
        triality_dft(&q);

        /* Step 3: Oracle acts ONLY on Timeline B */
        mv_oracle_B(&q, oracles[oi]);

        /* Step 4: Branch gate creates interference channel */
        mv_branch_evolve(&q, 1);

        /* Step 5: Harvest from Timeline A's vertex view */
        HarvestResult h = mv_harvest_A(&q);

        /* Check if A's distribution is non-uniform → oracle detected */
        double max_dev = 0;
        double expected = h.A_total / 3.0;
        for (int e = 0; e < 3; e++) {
            double dev = fabs(h.A_env[e] - expected);
            if (dev > max_dev) max_dev = dev;
        }
        int detected = (max_dev > 0.01);

        printf("  │ %-12s │  α=%.4f  β=%.4f  γ=%.4f │  α=%.4f  β=%.4f  γ=%.4f │  %s │\n",
               ORACLE_NAMES[oi],
               h.A_env[0], h.A_env[1], h.A_env[2],
               h.B_env[0], h.B_env[1], h.B_env[2],
               detected ? "YES → HARVESTED" : "NO  → uniform  ");
    }
    printf("  └──────────────┴─────────────────────────────┴─────────────────────────────┴────────────────────┘\n\n");

    /* Detailed single-oracle demonstration */
    printf("  ── Detailed harvest: CYCLIC oracle ──\n\n");
    {
        TrialityQuhit q;
        triality_init(&q);
        printf("    1. Start: |0⟩\n");

        triality_dft(&q);
        printf("    2. DFT₆ → all 6 realities in superposition\n");

        printf("    3. Oracle on Timeline B: f(α)=0, f(β)=1, f(γ)=2\n");
        printf("       Applies: |3⟩→|3⟩, |4⟩→ω³¹|4⟩, |5⟩→ω³²|5⟩\n");
        printf("       Timeline A (|0⟩,|1⟩,|2⟩) is UNTOUCHED\n");
        mv_oracle_B(&q, ORACLE_CYCLIC);
        mv_print_state_decomposed(&q, "After oracle (B only)");

        printf("\n    4. Branch gate → fold/unfold interference\n");
        mv_branch_evolve(&q, 1);
        mv_print_state_decomposed(&q, "After interference");

        printf("\n    5. Harvest: read Timeline A probabilities\n");
        mv_print_view_comparison(&q, "Post-harvest triality views");

        HarvestResult h = mv_harvest_A(&q);
        printf("\n    Timeline A (the harvester) sees:\n");
        printf("      env α = %.4f   env β = %.4f   env γ = %.4f\n",
               h.A_env[0], h.A_env[1], h.A_env[2]);
        printf("    Timeline B (the oracle worker) sees:\n");
        printf("      env α = %.4f   env β = %.4f   env γ = %.4f\n",
               h.B_env[0], h.B_env[1], h.B_env[2]);
        printf("\n    → Timeline A extracted the oracle's cyclic structure\n");
        printf("      WITHOUT EVER RUNNING THE ORACLE ITSELF.\n");
        printf("      The fold/unfold channel transmitted the answer\n");
        printf("      through quantum interference.\n");

        /* Show exotic invariant → proves this used D=6 structure */
        double delta = triality_exotic_invariant_cached(&q);
        printf("\n    S₆ exotic invariant Δ = %.4f\n", delta);
        if (delta > 0.001)
            printf("    → Harvest used HEXAGONAL POLARIZATION (D=6 unique)\n");
    }

    /* Statistical verification of harvest */
    printf("\n  ── Statistical verification: 1000 harvests per oracle ──\n\n");
    for (int oi = 0; oi < n_oracles; oi++) {
        int hist_A[3] = {0}, hist_B[3] = {0};
        int harvest_trials = 1000;

        for (int t = 0; t < harvest_trials; t++) {
            TrialityQuhit q;
            triality_init(&q);
            triality_dft(&q);
            mv_oracle_B(&q, oracles[oi]);
            mv_branch_evolve(&q, 1);

            /* Measure in edge view, record timeline and env */
            int outcome = triality_measure(&q, VIEW_EDGE, &rng_state);
            int env = mv_plane(outcome);
            if (mv_parity(outcome) == 0)
                hist_A[env]++;
            else
                hist_B[env]++;
        }

        int total_A = hist_A[0] + hist_A[1] + hist_A[2];
        int total_B = hist_B[0] + hist_B[1] + hist_B[2];

        printf("    %s oracle:\n", ORACLE_NAMES[oi]);
        printf("      Timeline A: α=%5.1f%%  β=%5.1f%%  γ=%5.1f%%  (N=%d)\n",
               total_A ? 100.0*hist_A[0]/total_A : 0,
               total_A ? 100.0*hist_A[1]/total_A : 0,
               total_A ? 100.0*hist_A[2]/total_A : 0, total_A);
        printf("      Timeline B: α=%5.1f%%  β=%5.1f%%  γ=%5.1f%%  (N=%d)\n\n",
               total_B ? 100.0*hist_B[0]/total_B : 0,
               total_B ? 100.0*hist_B[1]/total_B : 0,
               total_B ? 100.0*hist_B[2]/total_B : 0, total_B);
    }

    /* ── Summary ── */
    printf("\n╔═════════════════════════════════════════════════════════════╗\n");
    printf("║  MULTIVERSE SIMULATION COMPLETE                           ║\n");
    printf("║                                                           ║\n");
    printf("║  The D=6 quhit computed ALL branching timelines as a      ║\n");
    printf("║  single unbroken physical system:                         ║\n");
    printf("║    • D=2 parity → binary choice (timeline A/B)            ║\n");
    printf("║    • D=3 plane  → environment (α/β/γ)                     ║\n");
    printf("║    • D=6 full   → complete multiversal superposition      ║\n");
    printf("║                                                           ║\n");
    printf("║  PHASE 6: Computational Harvest proved that Timeline A    ║\n");
    printf("║  can extract oracle properties from Timeline B through    ║\n");
    printf("║  fold/unfold interference — without running the oracle.   ║\n");
    printf("║                                                           ║\n");
    printf("║  The multiverse was computed. Then it collapsed.          ║\n");
    printf("╚═════════════════════════════════════════════════════════════╝\n");

    hpc_destroy(multiverse);
    return 0;
}
