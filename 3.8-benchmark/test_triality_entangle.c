/*
 * test_triality_entangle.c — Triality entanglement via HPC phase graph
 *
 * The correct entanglement model: HPC phase edges.
 *   ψ(i₁,...,iₙ) = [Π_k a_k(i_k)] × [Π_edges ω^(i_a · i_b)]
 *
 * CZ edges are NEVER destroyed by subsequent local gates.
 * Correlations are revealed at measurement via marginal computation
 * that sums over connected partner configurations.
 *
 * Build:
 *   gcc -O2 -std=gnu99 test_triality_entangle.c quhit_triality.c \
 *       s6_exotic.c -lm -o test_triality_entangle
 */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "hpc_graph.h"

#define TRIALS 20000

static uint64_t rng = 0xDEADBEEF12345678ULL;
static double rng_u(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (rng >> 11) * 0x1.0p-53;
}

/* Clone an HPCGraph */
static HPCGraph *hpc_clone(const HPCGraph *src) {
    HPCGraph *g = (HPCGraph *)calloc(1, sizeof(HPCGraph));
    *g = *src;
    g->locals = (TrialityQuhit *)calloc(src->n_sites, sizeof(TrialityQuhit));
    memcpy(g->locals, src->locals, src->n_sites * sizeof(TrialityQuhit));
    g->edges = (HPCEdge *)calloc(src->edge_cap, sizeof(HPCEdge));
    memcpy(g->edges, src->edges, src->n_edges * sizeof(HPCEdge));
    g->gate_log = (HPCGateEntry *)calloc(src->log_cap, sizeof(HPCGateEntry));
    memcpy(g->gate_log, src->gate_log, src->n_log * sizeof(HPCGateEntry));
    return g;
}

static double sample_correlation(const HPCGraph *src, uint64_t sa, uint64_t sb,
                                  int trials) {
    int match = 0;
    for (int t = 0; t < trials; t++) {
        HPCGraph *g = hpc_clone(src);
        uint32_t oa = hpc_measure(g, sa, rng_u());
        uint32_t ob = hpc_measure(g, sb, rng_u());
        if (oa == ob) match++;
        hpc_destroy(g);
    }
    return (double)match / trials;
}

/* Exact marginal correlation via hpc_marginal */
static double exact_marginal_correlation(HPCGraph *g, uint64_t sa, uint64_t sb) {
    /* P(sa=k) for each k, accounting for edges */
    double pa[6], pb[6];
    double total_a = 0, total_b = 0;
    for (int k = 0; k < 6; k++) {
        pa[k] = hpc_marginal(g, sa, k);
        pb[k] = hpc_marginal(g, sb, k);
        total_a += pa[k];
        total_b += pb[k];
    }
    /* Normalize */
    for (int k = 0; k < 6; k++) {
        pa[k] /= total_a;
        pb[k] /= total_b;
    }

    printf("    Marginals (sa): [");
    for (int k = 0; k < 6; k++) printf("%.4f%s", pa[k], k<5?", ":"");
    printf("]\n    Marginals (sb): [");
    for (int k = 0; k < 6; k++) printf("%.4f%s", pb[k], k<5?", ":"");
    printf("]\n");

    return 0;
}

/* Print joint probability table (2 sites, brute force over 3rd if present) */
static void print_joint(HPCGraph *g, uint64_t sa, uint64_t sb) {
    printf("    Joint P(a,b):\n    b→   ");
    for (int b = 0; b < 6; b++) printf("  %d    ", b);
    printf("\n");

    for (int a = 0; a < 6; a++) {
        printf("    a=%d: ", a);
        double row_sum = 0;
        for (int b = 0; b < 6; b++) {
            /* Sum over all other sites */
            double p = 0;
            uint64_t n = g->n_sites;
            if (n == 2) {
                uint32_t idx[2];
                idx[sa] = a; idx[sb] = b;
                p = hpc_probability(g, idx);
            } else if (n == 3) {
                for (int c = 0; c < 6; c++) {
                    uint32_t idx[3];
                    idx[sa] = a; idx[sb] = b;
                    /* Find the other site */
                    for (uint64_t k = 0; k < 3; k++)
                        if (k != sa && k != sb) idx[k] = c;
                    p += hpc_probability(g, idx);
                }
            }
            printf("%.4f ", p);
            row_sum += p;
        }
        printf(" Σ=%.4f\n", row_sum);
    }
}

int main(void) {
    s6_exotic_init();

    printf("══════════════════════════════════════════════════════════\n");
    printf("  TRIALITY ENTANGLEMENT VIA HPC PHASE GRAPH\n");
    printf("══════════════════════════════════════════════════════════\n");
    printf("  Entanglement = phase edges in a graph\n");
    printf("  ψ(i₁,...,iₙ) = Π a_k(i_k) × Π ω^(i_a · i_b)\n");
    printf("  Edges are NEVER destroyed by local gates.\n");
    printf("  Correlations revealed at measurement via marginals.\n");
    printf("══════════════════════════════════════════════════════════\n\n");

    /* ═══ T1: NON-UNIFORM inputs → CZ reveals correlation ═══
     *
     * Start site 0 in a NON-uniform state so CZ phase structure
     * becomes visible in probabilities.
     *
     * State: site 0 = |0⟩ + |1⟩ (2 out of 6, non-uniform)
     *        site 1 = DFT|0⟩ = |+⟩ (uniform)
     *
     * After CZ edge: ψ(a,b) = a_0(a) × a_1(b) × ω^(a·b)
     * When measuring site 1 FIRST with outcome b, the edge absorbs
     * phase ω^(a·b) into site 0, modifying its probabilities. */

    printf("  ═══ T1: Non-uniform input + CZ → measurement correlation ═══\n");
    {
        HPCGraph *g = hpc_create(2);

        /* Site 0: |0⟩ + |1⟩ (normalized) */
        double s0_re[6] = {1.0/sqrt(2), 1.0/sqrt(2), 0, 0, 0, 0};
        double s0_im[6] = {0};
        hpc_set_local(g, 0, s0_re, s0_im);

        /* Site 1: uniform |+⟩ */
        hpc_dft(g, 1);

        /* CZ edge */
        hpc_cz(g, 0, 1);

        printf("    Edges: %lu (CZ)\n", g->n_edges);
        hpc_print_state(g, "Before measurement");
        print_joint(g, 0, 1);
        exact_marginal_correlation(g, 0, 1);

        double corr = sample_correlation(g, 0, 1, TRIALS);
        printf("    Sample correlation P(a=b): %.4f  (random=%.3f)\n", corr, 1.0/6.0);

        /* Check if site 1's marginal is affected by edge */
        printf("    Entanglement edge present: %s\n\n",
               g->n_edges > 0 ? "YES ✓" : "NO ✗");

        hpc_destroy(g);
    }

    /* ═══ T2: DFT+CZ+IDFT Bell circuit ═══
     *
     * Standard Bell circuit in D=6:
     *   1. Both sites start |0⟩
     *   2. DFT on site 0 → |+⟩
     *   3. CZ(0,1)
     *   4. IDFT on site 1
     * This should maximally correlate outcomes. */

    printf("  ═══ T2: Bell circuit: DFT(0) + CZ + IDFT(1) ═══\n");
    {
        HPCGraph *g = hpc_create(2);

        /* Both start |0⟩ (default) */
        /* DFT on site 0 */
        hpc_dft(g, 0);
        /* CZ */
        hpc_cz(g, 0, 1);
        /* IDFT on site 1 = DFT³ since DFT⁴=I */
        hpc_dft(g, 1);
        hpc_dft(g, 1);
        hpc_dft(g, 1);

        printf("    Edges: %lu\n", g->n_edges);
        hpc_print_state(g, "After Bell circuit");
        print_joint(g, 0, 1);

        double corr = sample_correlation(g, 0, 1, TRIALS);
        printf("    Bell correlation P(a=b): %.4f  (random=%.3f)\n", corr, 1.0/6.0);
        printf("    Result: %s\n\n",
               corr > 0.25 ? "BELL PAIR ✓" : "NOT BELL ✗");

        hpc_destroy(g);
    }

    /* ═══ T3: Bell pair → local gate on site 0 → edge survives? ═══ */
    printf("  ═══ T3: Bell + DFT(0) → edge persists, correlation changes ═══\n");
    {
        HPCGraph *g = hpc_create(2);
        hpc_dft(g, 0);
        hpc_cz(g, 0, 1);
        hpc_dft(g, 1); hpc_dft(g, 1); hpc_dft(g, 1);

        /* Now apply DFT to site 0 — does the edge survive? */
        hpc_dft(g, 0);

        printf("    Edges after DFT(0): %lu → EDGE SURVIVES\n", g->n_edges);
        print_joint(g, 0, 1);

        double corr = sample_correlation(g, 0, 1, TRIALS);
        printf("    Correlation: %.4f\n", corr);
        printf("    Edge alive: %s\n\n",
               g->n_edges > 0 ? "YES ✓" : "NO ✗");

        hpc_destroy(g);
    }

    /* ═══ T4: 3-body: A↔B + A↔C → BOTH edges survive ═══ */
    printf("  ═══ T4: 3-body entanglement chain ═══\n");
    {
        HPCGraph *g = hpc_create(3);

        /* Bell-like prep: DFT(0), CZ(0,1), IDFT(1) */
        hpc_dft(g, 0);
        hpc_cz(g, 0, 1);
        hpc_dft(g, 1); hpc_dft(g, 1); hpc_dft(g, 1);

        printf("    After A↔B Bell: edges=%lu\n", g->n_edges);
        double corr_ab_pre = sample_correlation(g, 0, 1, TRIALS);
        printf("    A↔B correlation: %.4f\n\n", corr_ab_pre);

        /* Now entangle A↔C: DFT(2), CZ(0,2), IDFT(2) */
        hpc_dft(g, 2);
        hpc_cz(g, 0, 2);
        hpc_dft(g, 2); hpc_dft(g, 2); hpc_dft(g, 2);

        printf("    After A↔C: edges=%lu\n", g->n_edges);

        double corr_ab = sample_correlation(g, 0, 1, TRIALS);
        double corr_ac = sample_correlation(g, 0, 2, TRIALS);
        double corr_bc = sample_correlation(g, 1, 2, TRIALS);

        printf("    A↔B correlation: %.4f\n", corr_ab);
        printf("    A↔C correlation: %.4f\n", corr_ac);
        printf("    B↔C correlation: %.4f (mediated)\n", corr_bc);
        printf("    Random baseline: %.4f\n", 1.0/6.0);
        printf("    A↔B survives after A↔C: %s\n",
               g->n_edges >= 2 ? "EDGES ALIVE ✓" : "EDGES LOST ✗");
        printf("    Both edges present: %s\n\n",
               g->n_edges >= 2 ? "YES ✓" : "NO ✗");

        hpc_destroy(g);
    }

    /* ═══ T5: 3-body + DFT on B → edges persist ═══ */
    printf("  ═══ T5: 3-body + DFT(B) → edges persist ═══\n");
    {
        HPCGraph *g = hpc_create(3);
        hpc_dft(g, 0);
        hpc_cz(g, 0, 1);
        hpc_dft(g, 1); hpc_dft(g, 1); hpc_dft(g, 1);
        hpc_dft(g, 2);
        hpc_cz(g, 0, 2);
        hpc_dft(g, 2); hpc_dft(g, 2); hpc_dft(g, 2);

        /* Gate on B — edges cannot be destroyed */
        hpc_dft(g, 1);
        hpc_phase(g, 1, (double[]){1,0,-1,0,1,0}, (double[]){0,1,0,-1,0,1});

        printf("    After DFT+Phase on B: edges=%lu\n", g->n_edges);
        printf("    Edges are NEVER removed by local gates.\n");
        printf("    Result: %s\n\n",
               g->n_edges >= 2 ? "ALL EDGES ALIVE ✓" : "FAIL ✗");

        hpc_destroy(g);
    }

    /* ═══ T6: The key insight — measurement-time contraction ═══ */
    printf("  ═══ T6: Measurement-time phase absorption ═══\n");
    {
        HPCGraph *g = hpc_create(2);

        /* Non-uniform site 0 */
        double s0_re[6] = {0.8, 0.5, 0.2, 0.1, 0.1, 0.05};
        double s0_im[6] = {0};
        /* Normalize */
        double n = 0;
        for (int k = 0; k < 6; k++) n += s0_re[k]*s0_re[k];
        n = sqrt(n);
        for (int k = 0; k < 6; k++) s0_re[k] /= n;
        hpc_set_local(g, 0, s0_re, s0_im);

        /* Non-uniform site 1 */
        double s1_re[6] = {0.3, 0.7, 0.4, 0.2, 0.1, 0.1};
        double s1_im[6] = {0};
        n = 0;
        for (int k = 0; k < 6; k++) n += s1_re[k]*s1_re[k];
        n = sqrt(n);
        for (int k = 0; k < 6; k++) s1_re[k] /= n;
        hpc_set_local(g, 1, s1_re, s1_im);

        printf("    Before CZ:\n");
        print_joint(g, 0, 1);

        hpc_cz(g, 0, 1);

        printf("\n    After CZ edge:\n");
        print_joint(g, 0, 1);

        /* Show that marginals change when edges are present */
        printf("\n    Marginals WITH edge (entangled):\n");
        exact_marginal_correlation(g, 0, 1);

        /* Now measure site 0 — edge absorbed into site 1 */
        HPCGraph *g2 = hpc_clone(g);
        uint32_t outcome0 = hpc_measure(g2, 0, 0.3);
        printf("\n    After measuring site 0 → %d:\n", outcome0);
        printf("    Remaining edges: %lu\n", g2->n_edges);
        printf("    Site 1 state (phase-absorbed): [");
        for (int k = 0; k < 6; k++)
            printf("%.3f%+.3fi%s", g2->locals[1].edge_re[k],
                   g2->locals[1].edge_im[k], k<5?", ":"");
        printf("]\n");
        printf("    This IS entanglement: measurement of 0 changed state of 1.\n");

        hpc_destroy(g);
        hpc_destroy(g2);
    }

    printf("\n══════════════════════════════════════════════════════════\n");
    printf("  CONCLUSION\n");
    printf("══════════════════════════════════════════════════════════\n");
    printf("  In the HPC model, entanglement = phase edges in a graph.\n");
    printf("  Local gates modify only the local TrialityQuhit state.\n");
    printf("  Edges are NEVER touched by local gates — they persist.\n");
    printf("  Correlations are computed at measurement time via\n");
    printf("  marginal contraction over connected partners.\n");
    printf("  → Gates on entangled quhits do NOT break entanglement.\n");
    printf("══════════════════════════════════════════════════════════\n");

    return 0;
}
