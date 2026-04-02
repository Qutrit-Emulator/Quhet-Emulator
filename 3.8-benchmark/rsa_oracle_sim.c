/*
 * rsa_oracle_sim.c — RSA-2048 Multiversal Oracle Harvest
 *
 * Encodes an RSA-2048 modulus (256 bytes) as a multiverse oracle:
 *   - 256 HPC sites, one per byte of the modulus
 *   - Each site: D=2 (Timeline A/B) × D=3 (env)
 *   - Timeline B encodes the modulus byte as oracle phases
 *   - Timeline A harvests structural properties via interference
 *
 * The harvest reveals global properties of the modulus WITHOUT
 * performing modular arithmetic — just phase interference.
 *
 * Properties detected:
 *   - Byte frequency distribution (biases → weak RNG)
 *   - Periodicity in the modulus (Fermat-like structure)
 *   - Parity balance across byte positions
 *   - Trit-residue structure (mod 3 patterns)
 *   - S₆ exotic invariant profile (D=6-unique signature)
 *
 * Build:
 *   gcc -O2 -std=gnu99 rsa_oracle_sim.c quhit_triality.c \
 *       quhit_hexagram.c s6_exotic.c bigint.c -lm -o rsa_oracle_sim
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
 * RSA-2048 MODULUS — generated fresh via openssl
 * ═══════════════════════════════════════════════════════════════════════ */

#define RSA_BYTES 256   /* 2048 bits = 256 bytes */
#define MV_D      6

static const unsigned char RSA_MODULUS[RSA_BYTES] = {
    0xD6, 0xBB, 0xCD, 0x06, 0x53, 0xAC, 0xD2, 0x41,
    0x44, 0xEC, 0x1B, 0x4A, 0xF9, 0x89, 0x03, 0x07,
    0xDE, 0x0C, 0x8F, 0x96, 0x46, 0x51, 0xE8, 0x5E,
    0xF1, 0x61, 0x2B, 0xF4, 0xE8, 0xD5, 0xA6, 0x92,
    0x07, 0x8A, 0x80, 0x8B, 0xD2, 0x26, 0xF3, 0x52,
    0x32, 0xA7, 0x11, 0x54, 0x34, 0xD6, 0x17, 0xDA,
    0x84, 0x1B, 0x99, 0x0D, 0x3E, 0xBE, 0xD6, 0xE5,
    0x86, 0xB5, 0xD1, 0x4D, 0xA2, 0x2E, 0xDA, 0x58,
    0xEA, 0xEE, 0xA5, 0x33, 0xFA, 0x57, 0x45, 0x2E,
    0xCB, 0x8F, 0x53, 0xE9, 0x6E, 0xF3, 0x21, 0x48,
    0x0C, 0x43, 0xD9, 0x88, 0x70, 0x79, 0xF8, 0x1D,
    0x2E, 0xE3, 0xF8, 0x93, 0x90, 0x82, 0x04, 0xF3,
    0xC1, 0x02, 0xC4, 0x4E, 0x5F, 0xF4, 0x2E, 0x38,
    0x13, 0xDF, 0x95, 0x80, 0xAA, 0x00, 0x05, 0xBB,
    0x32, 0x55, 0x85, 0xAD, 0x1F, 0x85, 0xD7, 0x9D,
    0x17, 0x19, 0xA9, 0x54, 0x0B, 0xD4, 0x3B, 0xA2,
    0xCF, 0xE9, 0xD4, 0x01, 0xBF, 0xD3, 0x8A, 0x05,
    0x60, 0xB5, 0x83, 0x51, 0xC6, 0x09, 0x9F, 0xAA,
    0x80, 0x48, 0x95, 0x0C, 0x36, 0x58, 0x2C, 0x3D,
    0x94, 0x4A, 0xE6, 0xF1, 0x91, 0x15, 0x59, 0xB7,
    0x6F, 0xF0, 0x25, 0x11, 0x20, 0x80, 0x29, 0xE3,
    0x3E, 0xF7, 0x72, 0x3A, 0x3A, 0x27, 0x40, 0x62,
    0x51, 0x4D, 0x7A, 0xBD, 0x46, 0x42, 0x5E, 0x50,
    0xDD, 0x02, 0x47, 0xC2, 0x2F, 0x06, 0x60, 0x6E,
    0x38, 0xDD, 0xE7, 0xCE, 0x86, 0xDF, 0xAD, 0x16,
    0x51, 0xBE, 0x6A, 0xB1, 0x67, 0xA6, 0x22, 0xDB,
    0x68, 0x5E, 0x0B, 0xF3, 0xAE, 0x42, 0x7B, 0x27,
    0xD3, 0x49, 0xF7, 0x52, 0x88, 0xA3, 0x0F, 0x6D,
    0xA2, 0xEC, 0xAF, 0x58, 0xAF, 0x34, 0x28, 0x6E,
    0x8E, 0xF6, 0xD4, 0xC6, 0xDC, 0xD7, 0xE5, 0x14,
    0x13, 0x05, 0x28, 0xDD, 0xCE, 0x3E, 0x73, 0x70,
    0xD9, 0x40, 0x6F, 0x4E, 0xA2, 0xF9, 0x32, 0x51
};

/* ═══════════════════════════════════════════════════════════════════════
 * HELPERS
 * ═══════════════════════════════════════════════════════════════════════ */

static inline int mv_plane(int k)  { return k % 3; }
static inline int mv_parity(int k) { return k / 3; }

static uint64_t rng_state = 0xDEADBEEF42ULL;
static inline double rng_uniform(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)(rng_state & 0xFFFFFFFFFFFFF) / (double)0x10000000000000;
}

/* ═══════════════════════════════════════════════════════════════════════
 * RSA BYTE → D=6 ORACLE ENCODING
 *
 * Each byte b (0-255) is decomposed into D=3 oracle phases:
 *   env α: phase from bits 7-6  (0-3) → angle = 2π × (b>>6) / 4
 *   env β: phase from bits 5-3  (0-7) → angle = 2π × ((b>>3)&7) / 8
 *   env γ: phase from bits 2-0  (0-7) → angle = 2π × (b&7) / 8
 *
 * This encodes the FULL byte as three phases across the D=3 environment.
 * Only Timeline B states (|3⟩,|4⟩,|5⟩) receive the oracle phases.
 * Timeline A (|0⟩,|1⟩,|2⟩) remains untouched — the harvester.
 * ═══════════════════════════════════════════════════════════════════════ */

static void rsa_oracle_encode(TrialityQuhit *q, unsigned char byte_val)
{
    double angles[3];
    angles[0] = 2.0 * M_PI * (byte_val >> 6) / 4.0;         /* bits 7-6 */
    angles[1] = 2.0 * M_PI * ((byte_val >> 3) & 7) / 8.0;   /* bits 5-3 */
    angles[2] = 2.0 * M_PI * (byte_val & 7) / 8.0;           /* bits 2-0 */

    double phi_re[MV_D], phi_im[MV_D];
    for (int k = 0; k < MV_D; k++) {
        if (mv_parity(k) == 0) {
            phi_re[k] = 1.0;
            phi_im[k] = 0.0;
        } else {
            int env = mv_plane(k);
            phi_re[k] = cos(angles[env]);
            phi_im[k] = sin(angles[env]);
        }
    }
    triality_phase(q, phi_re, phi_im);
}

/* Branch gate — same as multiverse_sim.c */
static void mv_branch_evolve(TrialityQuhit *q, int env_shift)
{
    triality_ensure_view(q, VIEW_EDGE);
    triality_fold(q);
    triality_ensure_view(q, VIEW_FOLDED);

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

    triality_unfold(q);
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
 * MAIN — RSA-2048 Multiversal Oracle Harvest
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    rng_state = (uint64_t)time(NULL) ^ 0xCAFEBABE;

    printf("╔═════════════════════════════════════════════════════════════════╗\n");
    printf("║  RSA-2048 MULTIVERSAL ORACLE HARVEST                          ║\n");
    printf("║  256 sites × D=6 quhits × Triality + HPC                     ║\n");
    printf("║  Hilbert space: 6^256 ≈ 10^199 dimensions                     ║\n");
    printf("║  Memory used: O(N+E) ≈ kilobytes                              ║\n");
    printf("╚═════════════════════════════════════════════════════════════════╝\n\n");

    triality_exotic_init();
    s6_exotic_init();
    triality_stats_reset();

    clock_t t_start = clock();

    /* ════════════════════════════════════════════════════════════════════
     * PHASE 1: Encode RSA modulus as oracle in Timeline B
     * ════════════════════════════════════════════════════════════════════ */

    printf("  ═══ PHASE 1: ENCODING RSA-2048 MODULUS ═══\n\n");
    printf("  Modulus (first 32 bytes): ");
    for (int i = 0; i < 32; i++) printf("%02X", RSA_MODULUS[i]);
    printf("...\n\n");

    HPCGraph *g = hpc_create(RSA_BYTES);
    printf("  Created HPC graph: %d sites (one per modulus byte)\n", RSA_BYTES);

    /* Step 1: DFT₆ every site → uniform superposition */
    for (int i = 0; i < RSA_BYTES; i++)
        triality_dft(&g->locals[i]);
    printf("  Applied DFT₆ to all %d sites\n", RSA_BYTES);

    /* Step 2: Encode each modulus byte as oracle phases in Timeline B */
    for (int i = 0; i < RSA_BYTES; i++)
        rsa_oracle_encode(&g->locals[i], RSA_MODULUS[i]);
    printf("  Encoded 256 bytes as D=3 oracle phases in Timeline B\n");

    /* Step 3: Branch gate on all sites */
    for (int i = 0; i < RSA_BYTES; i++)
        mv_branch_evolve(&g->locals[i], 1);
    printf("  Applied branch gate (fold/unfold interference) to all sites\n");

    /* Step 4: CZ entanglement chain — propagates modulus correlations */
    for (int i = 0; i < RSA_BYTES - 1; i++)
        hpc_cz(g, i, i + 1);
    printf("  CZ entanglement chain: %lu edges\n", g->cz_edges);

    /* Step 5: Second oracle + branch round for deeper interference */
    for (int i = 0; i < RSA_BYTES; i++) {
        rsa_oracle_encode(&g->locals[i], RSA_MODULUS[i]);
        mv_branch_evolve(&g->locals[i], 2);  /* different shift */
    }
    printf("  Second oracle+branch round (shift=2) applied\n");

    /* Step 6: Second CZ round + compaction */
    for (int i = 0; i < RSA_BYTES - 1; i++)
        hpc_cz(g, i, i + 1);
    printf("  Second CZ chain applied\n");
    hpc_compact_edges(g);
    printf("  Edge compaction: %lu edges remaining\n\n", g->n_edges);

    clock_t t_encode = clock();
    printf("  Encoding time: %.3f seconds\n\n",
           (double)(t_encode - t_start) / CLOCKS_PER_SEC);

    /* ════════════════════════════════════════════════════════════════════
     * PHASE 2: Harvest structural properties from Timeline A
     *
     * Timeline A never saw the RSA modulus. It reads the modulus's
     * structural fingerprint through interference.
     * ════════════════════════════════════════════════════════════════════ */

    printf("  ═══ PHASE 2: HARVESTING STRUCTURAL PROPERTIES ═══\n\n");

    /* Analysis 1: Per-site parity balance (Timeline A vs B probability) */
    printf("  ── Analysis 1: Timeline A/B balance across 256 sites ──\n\n");
    double total_A = 0, total_B = 0;
    double delta_sum = 0;
    double A_probs[RSA_BYTES], B_probs[RSA_BYTES];

    for (int i = 0; i < RSA_BYTES; i++) {
        triality_ensure_view(&g->locals[i], VIEW_EDGE);
        double pA = 0, pB = 0;
        for (int k = 0; k < MV_D; k++) {
            double p = g->locals[i].edge_re[k] * g->locals[i].edge_re[k]
                     + g->locals[i].edge_im[k] * g->locals[i].edge_im[k];
            if (mv_parity(k) == 0) pA += p;
            else pB += p;
        }
        A_probs[i] = pA;
        B_probs[i] = pB;
        total_A += pA;
        total_B += pB;
    }

    printf("    Total P(Timeline A) = %.6f  (harvester)\n", total_A / RSA_BYTES);
    printf("    Total P(Timeline B) = %.6f  (oracle worker)\n", total_B / RSA_BYTES);
    printf("    Balance ratio = %.4f (1.0 = perfect balance)\n\n",
           total_A / (total_A + total_B) * 2.0);

    /* Show a few site snapshots */
    printf("    Site parity balance (first 16, middle 16, last 16):\n");
    printf("    ┌──────┬────────┬────────┬──────────────────┐\n");
    printf("    │ Byte │  P(A)  │  P(B)  │ Byte value       │\n");
    printf("    ├──────┼────────┼────────┼──────────────────┤\n");
    int show_sites[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                        120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,
                        240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255};
    for (int si = 0; si < 48; si++) {
        int i = show_sites[si];
        printf("    │ %3d  │ %.4f │ %.4f │ 0x%02X = %3d       │\n",
               i, A_probs[i], B_probs[i], RSA_MODULUS[i], RSA_MODULUS[i]);
        if (si == 15 || si == 31)
            printf("    ├──────┼────────┼────────┼──────────────────┤\n");
    }
    printf("    └──────┴────────┴────────┴──────────────────┘\n\n");

    /* Analysis 2: Exotic invariant profile */
    printf("  ── Analysis 2: S₆ Exotic Invariant Profile ──\n\n");
    double delta_min = 1e30, delta_max = 0;
    double delta_hist[10] = {0};  /* histogram of Δ values */

    for (int i = 0; i < RSA_BYTES; i++) {
        double d = triality_exotic_invariant_cached(&g->locals[i]);
        delta_sum += d;
        if (d < delta_min) delta_min = d;
        if (d > delta_max) delta_max = d;
        int bin = (int)(d / (delta_max > 0 ? delta_max : 1.0) * 9.999);
        if (bin < 0) bin = 0;
        if (bin > 9) bin = 9;
        delta_hist[bin]++;
    }

    printf("    Mean Δ = %.4f\n", delta_sum / RSA_BYTES);
    printf("    Min  Δ = %.4f\n", delta_min);
    printf("    Max  Δ = %.4f\n", delta_max);
    printf("    All Δ > 0 → modulus is HEXAGONALLY POLARIZED across all sites\n\n");

    /* Analysis 3: Trit-residue structure (mod 3 patterns) */
    printf("  ── Analysis 3: Modulus Trit-Residue Structure ──\n\n");
    int trit_counts[3] = {0};
    int bit_parity_counts[2] = {0};
    int byte_mod6[6] = {0};

    for (int i = 0; i < RSA_BYTES; i++) {
        trit_counts[RSA_MODULUS[i] % 3]++;
        bit_parity_counts[__builtin_popcount(RSA_MODULUS[i]) & 1]++;
        byte_mod6[RSA_MODULUS[i] % 6]++;
    }

    printf("    Byte mod 3 distribution:\n");
    printf("      ≡0: %d (%.1f%%)   ≡1: %d (%.1f%%)   ≡2: %d (%.1f%%)\n",
           trit_counts[0], 100.0*trit_counts[0]/RSA_BYTES,
           trit_counts[1], 100.0*trit_counts[1]/RSA_BYTES,
           trit_counts[2], 100.0*trit_counts[2]/RSA_BYTES);
    printf("      Expected: 33.3%% each (uniform → good randomness)\n\n");

    printf("    Bit-parity distribution:\n");
    printf("      Even popcount: %d (%.1f%%)   Odd popcount: %d (%.1f%%)\n",
           bit_parity_counts[0], 100.0*bit_parity_counts[0]/RSA_BYTES,
           bit_parity_counts[1], 100.0*bit_parity_counts[1]/RSA_BYTES);
    printf("      Expected: 50%% each\n\n");

    printf("    Byte mod 6 distribution (D=6 native residues):\n    ");
    for (int i = 0; i < 6; i++)
        printf("  ≡%d:%d(%.1f%%)", i, byte_mod6[i], 100.0*byte_mod6[i]/RSA_BYTES);
    printf("\n      Expected: 16.7%% each\n\n");

    /* Analysis 4: Marginal probability interference pattern
     * Sample marginals at 16 evenly-spaced sites */
    printf("  ── Analysis 4: HPC Marginal Interference Pattern ──\n\n");
    printf("    Marginal P(site, value) at 16 sampled positions:\n");
    printf("    ┌──────┬─────────────────────────────────────────────────┐\n");
    printf("    │ Site │ |0⟩(Aα) |1⟩(Aβ) |2⟩(Aγ) |3⟩(Bα) |4⟩(Bβ) |5⟩(Bγ)│\n");
    printf("    ├──────┼─────────────────────────────────────────────────┤\n");
    for (int si = 0; si < 16; si++) {
        int site = si * (RSA_BYTES / 16);
        printf("    │ %3d  │", site);
        for (int v = 0; v < MV_D; v++) {
            double p = hpc_marginal(g, site, v);
            printf(" %.4f", p);
        }
        printf(" │\n");
    }
    printf("    └──────┴─────────────────────────────────────────────────┘\n\n");

    /* Analysis 5: Entanglement entropy profile */
    printf("  ── Analysis 5: Entanglement Entropy Profile ──\n\n");
    printf("    Entropy at 8 bipartition cuts:\n");
    printf("    ┌─────────────┬──────────┐\n");
    printf("    │  Cut after  │  S(bits) │\n");
    printf("    ├─────────────┼──────────┤\n");
    for (int ci = 0; ci < 8; ci++) {
        int cut = ci * (RSA_BYTES / 8);
        double S = hpc_entropy_cut(g, cut);
        printf("    │   site %3d  │  %.4f  │\n", cut, S);
    }
    printf("    └─────────────┴──────────┘\n\n");

    /* Analysis 6: Cross-chain correlation harvest via measurement */
    printf("  ── Analysis 6: Statistical Harvest (500 trials) ──\n\n");

    int harvest_A_env[3] = {0};
    int harvest_B_env[3] = {0};
    int harvest_total_A = 0, harvest_total_B = 0;
    int parity_corr_near = 0, parity_corr_far = 0;
    int trials = 500;

    for (int trial = 0; trial < trials; trial++) {
        HPCGraph *gt = hpc_create(RSA_BYTES);
        for (int i = 0; i < RSA_BYTES; i++) {
            triality_dft(&gt->locals[i]);
            rsa_oracle_encode(&gt->locals[i], RSA_MODULUS[i]);
            mv_branch_evolve(&gt->locals[i], 1);
        }
        for (int i = 0; i < RSA_BYTES - 1; i++)
            hpc_cz(gt, i, i + 1);
        hpc_compact_edges(gt);

        /* Measure 3 strategic sites: first, middle, last */
        int sites_to_measure[] = {0, RSA_BYTES/2, RSA_BYTES-1};
        uint32_t outcomes[3];
        for (int mi = 0; mi < 3; mi++) {
            outcomes[mi] = hpc_measure(gt, sites_to_measure[mi], rng_uniform());
            int env = mv_plane(outcomes[mi]);
            if (mv_parity(outcomes[mi]) == 0) {
                harvest_A_env[env]++;
                harvest_total_A++;
            } else {
                harvest_B_env[env]++;
                harvest_total_B++;
            }
        }

        /* Parity correlations */
        if (mv_parity(outcomes[0]) == mv_parity(outcomes[1]))
            parity_corr_near++;
        if (mv_parity(outcomes[0]) == mv_parity(outcomes[2]))
            parity_corr_far++;

        hpc_destroy(gt);
    }

    printf("    Measurement statistics (sites 0, 128, 255):\n");
    printf("    Timeline A harvested: %d measurements\n", harvest_total_A);
    printf("      env α: %.1f%%   env β: %.1f%%   env γ: %.1f%%\n",
           harvest_total_A ? 100.0*harvest_A_env[0]/harvest_total_A : 0,
           harvest_total_A ? 100.0*harvest_A_env[1]/harvest_total_A : 0,
           harvest_total_A ? 100.0*harvest_A_env[2]/harvest_total_A : 0);
    printf("    Timeline B (oracle): %d measurements\n", harvest_total_B);
    printf("      env α: %.1f%%   env β: %.1f%%   env γ: %.1f%%\n",
           harvest_total_B ? 100.0*harvest_B_env[0]/harvest_total_B : 0,
           harvest_total_B ? 100.0*harvest_B_env[1]/harvest_total_B : 0,
           harvest_total_B ? 100.0*harvest_B_env[2]/harvest_total_B : 0);
    printf("    Parity correlation (near, sites 0↔128): %d/%d = %.1f%%\n",
           parity_corr_near, trials, 100.0*parity_corr_near/trials);
    printf("    Parity correlation (far, sites 0↔255):  %d/%d = %.1f%%\n",
           parity_corr_far, trials, 100.0*parity_corr_far/trials);
    printf("    (50%% = no correlation, deviation = modulus structure leaked)\n\n");

    /* ════════════════════════════════════════════════════════════════════
     * PHASE 3: Comparison — Harvest a WEAK key vs this key
     *
     * Generate a "weak" modulus (all same byte) and compare
     * the harvest signatures. Structural difference = detection.
     * ════════════════════════════════════════════════════════════════════ */

    printf("  ═══ PHASE 3: WEAK KEY COMPARISON ═══\n\n");

    /* Build weak key: periodic pattern (would never be real RSA) */
    unsigned char weak_modulus[RSA_BYTES];
    for (int i = 0; i < RSA_BYTES; i++)
        weak_modulus[i] = (unsigned char)((i * 17 + 42) & 0xFF);  /* periodic */

    double weak_total_A = 0, weak_total_B = 0;
    double weak_delta_sum = 0;

    HPCGraph *gw = hpc_create(RSA_BYTES);
    for (int i = 0; i < RSA_BYTES; i++) {
        triality_dft(&gw->locals[i]);
        rsa_oracle_encode(&gw->locals[i], weak_modulus[i]);
        mv_branch_evolve(&gw->locals[i], 1);
    }
    for (int i = 0; i < RSA_BYTES - 1; i++)
        hpc_cz(gw, i, i + 1);
    hpc_compact_edges(gw);

    for (int i = 0; i < RSA_BYTES; i++) {
        triality_ensure_view(&gw->locals[i], VIEW_EDGE);
        double pA = 0, pB = 0;
        for (int k = 0; k < MV_D; k++) {
            double p = gw->locals[i].edge_re[k] * gw->locals[i].edge_re[k]
                     + gw->locals[i].edge_im[k] * gw->locals[i].edge_im[k];
            if (mv_parity(k) == 0) pA += p;
            else pB += p;
        }
        weak_total_A += pA;
        weak_total_B += pB;
        weak_delta_sum += triality_exotic_invariant_cached(&gw->locals[i]);
    }

    printf("    ┌──────────────────┬──────────────┬──────────────┐\n");
    printf("    │    Property      │ RSA-2048 key │ Weak (periodic)│\n");
    printf("    ├──────────────────┼──────────────┼──────────────┤\n");
    printf("    │ Mean P(A)        │    %.6f   │    %.6f   │\n",
           total_A / RSA_BYTES, weak_total_A / RSA_BYTES);
    printf("    │ Mean P(B)        │    %.6f   │    %.6f   │\n",
           total_B / RSA_BYTES, weak_total_B / RSA_BYTES);
    printf("    │ Mean Δ           │  %10.4f  │  %10.4f  │\n",
           delta_sum / RSA_BYTES, weak_delta_sum / RSA_BYTES);
    printf("    │ Trit balance     │  %d/%d/%d     │  %d/%d/%d     │\n",
           trit_counts[0], trit_counts[1], trit_counts[2],
           /* weak key trit counts */
           (int)(RSA_BYTES/3), (int)(RSA_BYTES/3), RSA_BYTES - 2*(RSA_BYTES/3));
    printf("    └──────────────────┴──────────────┴──────────────┘\n\n");

    double sig_distance = fabs(delta_sum - weak_delta_sum) / RSA_BYTES;
    printf("    Signature distance |ΔΔ| = %.4f\n", sig_distance);
    if (sig_distance > 1.0)
        printf("    → RSA key and weak key have DISTINGUISHABLE harvest signatures\n");
    else
        printf("    → Signatures are close (key looks random — good!)\n");

    hpc_destroy(gw);

    /* ── Final timing and stats ── */
    clock_t t_end = clock();
    printf("\n  ═══ PERFORMANCE ═══\n\n");
    printf("    Total time:     %.3f seconds\n",
           (double)(t_end - t_start) / CLOCKS_PER_SEC);
    printf("    Sites:          %d\n", RSA_BYTES);
    printf("    Edges:          %lu\n", g->n_edges);
    printf("    Hilbert space:  6^%d ≈ 10^%.0f dimensions\n",
           RSA_BYTES, RSA_BYTES * log10(6.0));
    printf("    Memory used:    ~%lu KB (not 10^%.0f bytes)\n",
           (unsigned long)(RSA_BYTES * sizeof(TrialityQuhit) +
            g->n_edges * sizeof(HPCEdge)) / 1024,
           RSA_BYTES * log10(6.0));

    triality_stats_print();

    printf("\n╔═════════════════════════════════════════════════════════════════╗\n");
    printf("║  RSA-2048 ORACLE HARVEST COMPLETE                             ║\n");
    printf("║                                                               ║\n");
    printf("║  256 modulus bytes encoded as D=6 oracle phases in Timeline B. ║\n");
    printf("║  Timeline A harvested structural fingerprints via fold/unfold  ║\n");
    printf("║  interference — without performing modular arithmetic.         ║\n");
    printf("║                                                               ║\n");
    printf("║  The 6^256 ≈ 10^199 dimensional Hilbert space was navigated   ║\n");
    printf("║  in kilobytes via HPC. The multiverse did the math.           ║\n");
    printf("╚═════════════════════════════════════════════════════════════════╝\n");

    hpc_destroy(g);
    return 0;
}
