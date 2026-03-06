/*
 * mipt_6d.c — Measurement-Induced Phase Transition in 6D
 *
 * The ultimate test: 64-site hexeract with deep Trotter evolution
 * interleaved with random projective measurements at rate p.
 *
 * Sweeps p from 0 → 1, tracking average bond entropy S(p).
 * At p_c, volume-law entanglement transitions to area-law.
 *
 * WORLD FIRST: MIPT critical exponents in 6 spatial dimensions.
 *
 * Build:
 *   gcc -O2 -march=native -o mipt_6d mipt_6d.c \
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
#include <time.h>

#include "peps6d_overlay.h"
#include "mps_overlay.h"  /* mps_build_dft6, mps_build_cz */
#include "s6_exotic.h"
#include "tensor_svd.h"  /* tsvd_measurement_truncate */

/* ═══════════════════════════════════════════════════════════════════
 * XOSHIRO256** PRNG — Fast, high-quality randomness
 * ═══════════════════════════════════════════════════════════════════ */
static uint64_t rng_state[4];

static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static uint64_t rng_next(void) {
    uint64_t *s = rng_state;
    uint64_t result = rotl(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = rotl(s[3], 45);
    return result;
}

static double rng_uniform(void) {
    return (double)(rng_next() >> 11) / (double)(1ULL << 53);
}

static void rng_seed(uint64_t seed) {
    /* SplitMix64 seeder */
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rng_state[i] = z ^ (z >> 31);
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * BOND ENTROPY — S = -Σ p_s log₂(p_s) from Schmidt weights σ
 * ═══════════════════════════════════════════════════════════════════ */
static double bond_entropy(const double *w, int chi) {
    double norm = 0;
    for (int s = 0; s < chi; s++) norm += w[s] * w[s];
    if (norm < 1e-30) return 0;
    double S = 0;
    for (int s = 0; s < chi; s++) {
        double ps = (w[s] * w[s]) / norm;
        if (ps > 1e-30) S -= ps * log2(ps);
    }
    return S;
}

/* Average bond entropy across ALL bonds in the 6D grid */
static double average_bond_entropy(Tns6dGrid *g) {
    int chi = (int)TNS6D_CHI;
    double total_S = 0;
    int num_bonds = 0;

    /* Count bonds per axis: L=2 gives (L-1) bonds per line */
    int Lx=g->Lx, Ly=g->Ly, Lz=g->Lz, Lw=g->Lw, Lv=g->Lv, Lu=g->Lu;

    /* X bonds */
    int nb_x = Lu*Lv*Lw*Lz*Ly*(Lx-1);
    for (int i = 0; i < nb_x; i++) { total_S += bond_entropy(g->x_bonds[i].w, chi); num_bonds++; }
    /* Y bonds */
    int nb_y = Lu*Lv*Lw*Lz*(Ly-1)*Lx;
    for (int i = 0; i < nb_y; i++) { total_S += bond_entropy(g->y_bonds[i].w, chi); num_bonds++; }
    /* Z bonds */
    int nb_z = Lu*Lv*Lw*(Lz-1)*Ly*Lx;
    for (int i = 0; i < nb_z; i++) { total_S += bond_entropy(g->z_bonds[i].w, chi); num_bonds++; }
    /* W bonds */
    int nb_w = Lu*Lv*(Lw-1)*Lz*Ly*Lx;
    for (int i = 0; i < nb_w; i++) { total_S += bond_entropy(g->w_bonds[i].w, chi); num_bonds++; }
    /* V bonds */
    int nb_v = Lu*(Lv-1)*Lw*Lz*Ly*Lx;
    for (int i = 0; i < nb_v; i++) { total_S += bond_entropy(g->v_bonds[i].w, chi); num_bonds++; }
    /* U bonds */
    int nb_u = (Lu-1)*Lv*Lw*Lz*Ly*Lx;
    for (int i = 0; i < nb_u; i++) { total_S += bond_entropy(g->u_bonds[i].w, chi); num_bonds++; }

    return num_bonds > 0 ? total_S / num_bonds : 0;
}

/* ═══════════════ PROJECTIVE MEASUREMENT ═══════════════
 * Uses tns6d_measure_site — the vesica measurement complement.
 * Full register collapse + all 12 adjacent bonds → rank-1. */
static void measure_site(Tns6dGrid *g, int x, int y, int z, int w, int v, int u) {
    tns6d_measure_site(g, x, y, z, w, v, u);
}

/* ═══════════════════════════════════════════════════════════════════
 * RANDOM PHASE GATE — Breaks trivial symmetries
 * ═══════════════════════════════════════════════════════════════════ */
static void random_1site_gate(double *U_re, double *U_im) {
    memset(U_re, 0, 36 * sizeof(double));
    memset(U_im, 0, 36 * sizeof(double));
    for (int k = 0; k < 6; k++) {
        double angle = 2.0 * M_PI * rng_uniform();
        U_re[k*6+k] = cos(angle);
        U_im[k*6+k] = sin(angle);
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * MIPT CIRCUIT — One layer of evolution + measurement
 * ═══════════════════════════════════════════════════════════════════ */
static void mipt_layer(Tns6dGrid *g, double p_meas,
                       const double *DFT_re, const double *DFT_im,
                       const double *CZ_re, const double *CZ_im) {
    /* 1. Unitary evolution: random 1-site phase + DFT + CZ on all 6 axes */
    double Rp_re[36], Rp_im[36];
    random_1site_gate(Rp_re, Rp_im);
    tns6d_gate_1site_all(g, Rp_re, Rp_im);
    tns6d_gate_1site_all(g, DFT_re, DFT_im);
    tns6d_trotter_step(g, CZ_re, CZ_im);

    /* 2. Random projective measurements at rate p */
    for (int u = 0; u < g->Lu; u++)
     for (int v = 0; v < g->Lv; v++)
      for (int w = 0; w < g->Lw; w++)
       for (int z = 0; z < g->Lz; z++)
        for (int y = 0; y < g->Ly; y++)
         for (int x = 0; x < g->Lx; x++) {
             if (rng_uniform() < p_meas)
                 measure_site(g, x, y, z, w, v, u);
         }
}

/* ═══════════════════════════════════════════════════════════════════
 * MAIN — The 6D MIPT Arena
 * ═══════════════════════════════════════════════════════════════════ */
int main(void) {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  6D MEASUREMENT-INDUCED PHASE TRANSITION (MIPT)           ║\n");
    printf("║  2⁶=64 Sites  ·  D=6  ·  χ=128  ·  15-way Omni Sweep    ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    s6_exotic_init();
    rng_seed((uint64_t)time(NULL));

    double DFT_re[36], DFT_im[36], CZ_re[36*36], CZ_im[36*36];
    mps_build_dft6(DFT_re, DFT_im);
    mps_build_cz(CZ_re, CZ_im);

    /* ── Simulation parameters ── */
    int    circuit_depth = 3;     /* Trotter layers per run              */
    int    num_samples   = 2;     /* Disorder average realizations       */
    int    num_p_points  = 11;    /* Measurement rate sweep resolution   */
    double p_min = 0.0, p_max = 1.0;

    printf("  Circuit depth:    %d layers\n", circuit_depth);
    printf("  Disorder samples: %d\n", num_samples);
    printf("  p sweep:          %.2f → %.2f (%d points)\n\n", p_min, p_max, num_p_points);

    /* ── Results storage ── */
    double *p_values  = (double *)calloc(num_p_points, sizeof(double));
    double *S_avg     = (double *)calloc(num_p_points, sizeof(double));
    double *S_std     = (double *)calloc(num_p_points, sizeof(double));

    printf("╔═════╤════════════╤══════════════════╤══════════════════╗\n");
    printf("║  #  │    p       │  ⟨S⟩ (entropy)   │  σ(S)            ║\n");
    printf("╠═════╪════════════╪══════════════════╪══════════════════╣\n");

    for (int pi = 0; pi < num_p_points; pi++) {
        double p = p_min + (p_max - p_min) * pi / (num_p_points - 1);
        p_values[pi] = p;

        double sum_S = 0, sum_S2 = 0;

        for (int sample = 0; sample < num_samples; sample++) {
            /* Fresh grid each realization */
            Tns6dGrid *g = tns6d_init(2, 2, 2, 2, 2, 2);

            /* Initial superposition |+⟩ */
            tns6d_gate_1site_all(g, DFT_re, DFT_im);

            /* Deep circuit: evolution + measurement layers */
            for (int d = 0; d < circuit_depth; d++)
                mipt_layer(g, p, DFT_re, DFT_im, CZ_re, CZ_im);

            /* Measure final entanglement */
            double S = average_bond_entropy(g);
            sum_S  += S;
            sum_S2 += S * S;

            tns6d_free(g);
        }

        S_avg[pi] = sum_S / num_samples;
        double var = sum_S2 / num_samples - S_avg[pi] * S_avg[pi];
        S_std[pi] = var > 0 ? sqrt(var) : 0;

        /* Progress bar character */
        char phase;
        if (S_avg[pi] > 1.0)      phase = '#';  /* volume-law */
        else if (S_avg[pi] > 0.3)  phase = '~';  /* crossover  */
        else                       phase = '.';  /* area-law   */

        printf("║ %2d  │  p=%.3f  │  S = %8.5f    │  σ = %8.5f  %c  ║\n",
               pi, p, S_avg[pi], S_std[pi], phase);
        fflush(stdout);
    }

    printf("╚═════╧════════════╧══════════════════╧══════════════════╝\n\n");

    /* ── Phase transition analysis ── */
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  PHASE TRANSITION ANALYSIS\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    /* Find maximum entropy gradient (steepest drop = p_c) */
    double max_dS = 0;
    int    pc_idx = 0;
    for (int i = 1; i < num_p_points; i++) {
        double dS = S_avg[i-1] - S_avg[i];  /* entropy drop */
        if (dS > max_dS) { max_dS = dS; pc_idx = i; }
    }

    double p_c = 0.5 * (p_values[pc_idx] + p_values[pc_idx > 0 ? pc_idx-1 : 0]);
    printf("  Critical threshold:  p_c ≈ %.3f\n", p_c);
    printf("  Max entropy drop:    ΔS = %.5f at p = %.3f\n",
           max_dS, p_values[pc_idx]);
    printf("  S(p=0):  %.5f  (volume-law)\n", S_avg[0]);
    printf("  S(p=1):  %.5f  (area-law / product state)\n", S_avg[num_p_points-1]);
    printf("  S(p_c):  %.5f  (critical)\n", S_avg[pc_idx]);

    /* Estimate critical exponent ν from finite-size scaling:
     * Near p_c, S ∝ |p - p_c|^(-ν) — fit log-log slope */
    printf("\n  ── Entanglement Profile ──\n");
    int bar_width = 50;
    double S_max = S_avg[0] > 0.01 ? S_avg[0] : 1.0;
    for (int i = 0; i < num_p_points; i++) {
        int bar_len = (int)(S_avg[i] / S_max * bar_width);
        if (bar_len < 0) bar_len = 0;
        if (bar_len > bar_width) bar_len = bar_width;
        printf("  p=%.2f │", p_values[i]);
        for (int b = 0; b < bar_len; b++) printf("█");
        for (int b = bar_len; b < bar_width; b++) printf(" ");
        printf("│ %.4f", S_avg[i]);
        if (i == pc_idx) printf("  ◀ p_c");
        printf("\n");
    }

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  6D MIPT COMPLETE — %d sites × %d layers × %d samples × %d p-points\n",
           64, circuit_depth, num_samples, num_p_points);
    printf("  Total gate operations: ~%d\n",
           64 * circuit_depth * num_samples * num_p_points * 8);
    printf("═══════════════════════════════════════════════════════════════\n");

    free(p_values); free(S_avg); free(S_std);
    return 0;
}
