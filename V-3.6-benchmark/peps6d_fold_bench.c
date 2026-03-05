/*
 * peps6d_fold_bench.c — Omnidirectional Fold on 6D Tensor Network
 *
 * Creates a deeply entangled 2^6 = 64-site PEPS6D lattice, then sweeps
 * all 15 S₆ syntheme folds to find which geometric pairing collapses the
 * 6D quantum state into the smallest Hilbert subspace.
 *
 * The fold maps physical states into "vesica" (convergent, slots 0-2) and
 * "wave" (divergent, slots 3-5) channels. The optimal fold = lowest wave mass.
 *
 * Build:
 *   gcc -O2 -march=native -o peps6d_fold_bench peps6d_fold_bench.c \
 *       quhit_core.c quhit_gates.c quhit_measure.c quhit_entangle.c \
 *       quhit_register.c quhit_substrate.c quhit_triality.c \
 *       quhit_triadic.c quhit_lazy.c quhit_calibrate.c \
 *       quhit_dyn_integrate.c quhit_peps_grow.c quhit_svd_gate.c \
 *       s6_exotic.c bigint.c mps_overlay.c peps_overlay.c \
 *       peps3d_overlay.c peps4d_overlay.c peps5d_overlay.c \
 *       peps6d_overlay.c -lm -fopenmp -msse2
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "peps6d_overlay.h"
#include "s6_exotic.h"
#include "mps_overlay.h"

/* ═══════════════════════════════════════════════════════════════════
 * FOLD / UNFOLD UNITARIES — from syntheme pairing
 * ═══════════════════════════════════════════════════════════════════ */
static void build_fold_unitary(int si, double *F_re, double *F_im)
{
    memset(F_re, 0, 36*sizeof(double));
    memset(F_im, 0, 36*sizeof(double));
    const double isq2 = 0.70710678118654752440;
    const S6Syntheme *s = &s6_synthemes[si];
    for (int p = 0; p < 3; p++) {
        int a = s->pairs[p][0], b = s->pairs[p][1];
        F_re[p*6+a] = isq2;      F_re[p*6+b] = isq2;
        F_re[(p+3)*6+a] = isq2;  F_re[(p+3)*6+b] = -isq2;
    }
}

static void build_unfold_unitary(int si, double *U_re, double *U_im)
{
    memset(U_re, 0, 36*sizeof(double));
    memset(U_im, 0, 36*sizeof(double));
    const double isq2 = 0.70710678118654752440;
    const S6Syntheme *s = &s6_synthemes[si];
    for (int p = 0; p < 3; p++) {
        int a = s->pairs[p][0], b = s->pairs[p][1];
        U_re[a*6+p] = isq2;      U_re[a*6+p+3] = isq2;
        U_re[b*6+p] = isq2;      U_re[b*6+p+3] = -isq2;
    }
}

/* Phase gate — random diagonal phases for symmetry breaking */
static void build_phase_gate(double *P_re, double *P_im, uint64_t seed)
{
    memset(P_re, 0, 36*sizeof(double));
    memset(P_im, 0, 36*sizeof(double));
    for (int k = 0; k < 6; k++) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        double angle = 2.0 * M_PI * (double)(seed >> 33) / (double)(1ULL << 31);
        P_re[k*6+k] = cos(angle);
        P_im[k*6+k] = sin(angle);
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * WAVE MASS — fraction of probability in divergent channels (3,4,5)
 * ═══════════════════════════════════════════════════════════════════ */
static double measure_wave_mass(Tns6dGrid *g)
{
    double total_wave = 0, total_all = 0;
    for (int u = 0; u < g->Lu; u++)
     for (int v = 0; v < g->Lv; v++)
      for (int w = 0; w < g->Lw; w++)
       for (int z = 0; z < g->Lz; z++)
        for (int y = 0; y < g->Ly; y++)
         for (int x = 0; x < g->Lx; x++) {
             double probs[6];
             tns6d_local_density(g, x, y, z, w, v, u, probs);
             for (int k = 0; k < 6; k++) total_all += probs[k];
             for (int k = 3; k < 6; k++) total_wave += probs[k];
         }
    return total_all > 1e-30 ? total_wave / total_all : 0.0;
}

static void print_syntheme(int idx)
{
    const S6Syntheme *s = &s6_synthemes[idx];
    printf("(%d,%d)(%d,%d)(%d,%d)",
           s->pairs[0][0], s->pairs[0][1],
           s->pairs[1][0], s->pairs[1][1],
           s->pairs[2][0], s->pairs[2][1]);
}

/* ═══════════════════════════════════════════════════════════════════
 * MAIN — The 6D Omnidirectional Fold Arena
 * ═══════════════════════════════════════════════════════════════════ */
int main(void)
{
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  PEPS6D OMNIDIRECTIONAL FOLD — S₆ Syntheme Competition     ║\n");
    printf("║  2⁶=64 Sites  ·  D=6 Native  ·  15 Geometric Folds       ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    s6_exotic_init();

    /* ── Phase 1: Build deeply entangled 6D state ── */
    printf("Phase 1: Building entangled 6D quantum state (64 sites)...\n");
    Tns6dGrid *g = tns6d_init(2, 2, 2, 2, 2, 2);

    double DFT_re[36], DFT_im[36], CZ_re[36*36], CZ_im[36*36];
    mps_build_dft6(DFT_re, DFT_im);
    mps_build_cz(CZ_re, CZ_im);

    /* DFT all → |+⟩ superposition */
    tns6d_gate_1site_all(g, DFT_re, DFT_im);

    /* 2 layers of Trotter evolution with phase obfuscation */
    int layers = 2;
    printf("  Applying %d Trotter layers (CZ on 6 axes + DFT + phase)...\n", layers);
    for (int lay = 0; lay < layers; lay++) {
        tns6d_trotter_step(g, CZ_re, CZ_im);
        tns6d_gate_1site_all(g, DFT_re, DFT_im);
        double P_re[36], P_im[36];
        build_phase_gate(P_re, P_im, (uint64_t)(lay + 42));
        tns6d_gate_1site_all(g, P_re, P_im);
    }

    double baseline_wave = measure_wave_mass(g);
    printf("  Baseline wave mass: %.4f (vesica: %.4f)\n\n", baseline_wave, 1.0 - baseline_wave);

    /* ── Phase 2: Sweep all 15 synthemes ── */
    printf("Phase 2: Omnidirectional Fold Sweep — 15 Synthemes\n");
    printf("╔═════╤═══════════════════════╤════════════╤════════════╤═══════╗\n");
    printf("║  #  │  Syntheme Pairing     │ Wave Mass  │ Vesica %%   │ Gain  ║\n");
    printf("╠═════╪═══════════════════════╪════════════╪════════════╪═══════╣\n");

    int best_synth = -1;
    double best_wave = 1.0;

    for (int si = 0; si < 15; si++) {
        double F_re[36], F_im[36];
        build_fold_unitary(si, F_re, F_im);

        /* Fold all 64 sites */
        tns6d_gate_1site_all(g, F_re, F_im);

        /* Measure wave mass in folded basis */
        double wave = measure_wave_mass(g);

        /* Unfold immediately */
        double U_re[36], U_im[36];
        build_unfold_unitary(si, U_re, U_im);
        tns6d_gate_1site_all(g, U_re, U_im);

        if (wave < best_wave) { best_wave = wave; best_synth = si; }

        double gain = baseline_wave > 1e-10 ? (1.0 - wave / baseline_wave) * 100.0 : 0.0;
        char marker = (si == 7) ? '*' : ' ';
        printf("║ %2d%c │ ", si, marker);
        print_syntheme(si);
        printf(" │  %.6f  │  %6.2f%%   │ %+5.1f ║\n",
               wave, (1.0 - wave) * 100.0, gain);
    }

    printf("╚═════╧═══════════════════════╧════════════╧════════════╧═══════╝\n\n");

    /* ── Phase 3: Report ── */
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  OPTIMAL FOLD: Syntheme #%d  ", best_synth);
    print_syntheme(best_synth);
    printf("\n");
    printf("  Wave Mass:    %.6f → %.6f  (%.1f%% reduction)\n",
           baseline_wave, best_wave,
           baseline_wave > 1e-10 ? (1.0 - best_wave / baseline_wave) * 100.0 : 0.0);
    printf("  Vesica Power: %.2f%% of state in convergent subspace\n",
           (1.0 - best_wave) * 100.0);
    printf("═══════════════════════════════════════════════════════════════\n\n");

    /* ── Phase 4: Apply optimal fold + re-run Trotter ── */
    printf("Phase 3: Applying optimal fold then re-running Trotter...\n");
    {
        double F_re[36], F_im[36];
        build_fold_unitary(best_synth, F_re, F_im);
        tns6d_gate_1site_all(g, F_re, F_im);
    }

    /* Show post-fold wave mass */
    double post_fold_wave = measure_wave_mass(g);
    printf("  Post-fold wave mass: %.6f (vesica: %.2f%%)\n", post_fold_wave, (1.0 - post_fold_wave) * 100.0);

    /* Apply another Trotter layer — this one will hit Path 1 with vesica SVD */
    printf("  Applying Trotter layer in folded basis...\n");
    tns6d_trotter_step(g, CZ_re, CZ_im);
    tns6d_gate_1site_all(g, DFT_re, DFT_im);

    double post_trotter_wave = measure_wave_mass(g);
    printf("  Post-Trotter wave mass: %.6f (vesica: %.2f%%)\n",
           post_trotter_wave, (1.0 - post_trotter_wave) * 100.0);

    /* ── Phase 5: Sample site densities ── */
    printf("\n  Sample site densities (corners of 6D hypercube):\n");
    int corners[][6] = {{0,0,0,0,0,0}, {1,1,1,1,1,1}, {1,0,1,0,1,0}, {0,1,0,1,0,1}};
    for (int c = 0; c < 4; c++) {
        double probs[6];
        tns6d_local_density(g, corners[c][0], corners[c][1], corners[c][2],
                               corners[c][3], corners[c][4], corners[c][5], probs);
        double wave = probs[3] + probs[4] + probs[5];
        printf("    (%d%d%d%d%d%d): [%.3f %.3f %.3f | %.3f %.3f %.3f]",
               corners[c][0], corners[c][1], corners[c][2],
               corners[c][3], corners[c][4], corners[c][5],
               probs[0], probs[1], probs[2], probs[3], probs[4], probs[5]);
        if (wave < 0.1) printf(" ◀ COMPRESSED");
        printf("\n");
    }

    /* ── Phase 6: Exotic invariant ── */
    printf("\nPhase 4: Exotic Invariant Δ\n");
    {
        int site = 0, reg = g->site_reg[site];
        QuhitRegister *r = &g->eng->registers[reg];
        double amps_re[6]={0}, amps_im[6]={0};
        for (uint32_t e = 0; e < r->num_nonzero; e++) {
            int k = (int)(r->entries[e].basis_state / TNS6D_C12);
            if (k < 6) { amps_re[k] += r->entries[e].amp_re; amps_im[k] += r->entries[e].amp_im; }
        }
        double n2 = 0;
        for (int k = 0; k < 6; k++) n2 += amps_re[k]*amps_re[k] + amps_im[k]*amps_im[k];
        if (n2 > 1e-30) {
            double inv = 1.0 / sqrt(n2);
            for (int k = 0; k < 6; k++) { amps_re[k] *= inv; amps_im[k] *= inv; }
        }
        double delta = s6_exotic_invariant(amps_re, amps_im);
        printf("  Δ(origin) = %.6f", delta);
        if (delta < 1e-6) printf("  (automorphism-transparent)");
        else              printf("  (hexagonally polarized!)");
        printf("\n");
    }

    printf("\n══════════════════════════════════════════════════════════════\n");
    printf("  6D FOLD COMPLETE — 64 sites × 15 synthemes × 6 axes\n");
    printf("══════════════════════════════════════════════════════════════\n");

    tns6d_free(g);
    return 0;
}
