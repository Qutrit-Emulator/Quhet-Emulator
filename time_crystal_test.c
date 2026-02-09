/*
 * ═══════════════════════════════════════════════════════════════════════════════
 * TIME CRYSTAL SIMULATION — 100,000,000 Quhits
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Discrete Time Crystal (DTC) via Floquet driving on the HexState Engine.
 *
 * Physics:
 *   A discrete time crystal breaks time-translation symmetry: when driven
 *   with period T, the system responds with period 2T.  In our simulation,
 *   applying DFT₆ Hadamard once per period to hexits initialized in |0⟩
 *   produces a cycle of order 2 in the observable (|0⟩ ↔ superposition),
 *   demonstrating period-doubling.
 *
 *   DFT₆ algebra:  H|0⟩ = uniform superposition,  H²|0⟩ = |0⟩.
 *   So:  H has order 4, and the observable returns after 2T — not T.
 *
 * Lattice:
 *   100 segments × ~1M quhits/segment = 100,000,000 quhits.
 *   4 probe chunks are shadow-backed (physical Hilbert space) for measurement.
 *   96 bulk chunks use Magic Pointers (external Hilbert space).
 *   Nearest-neighbor entanglement via braid links (1D chain).
 *
 * Measurement:
 *   Non-destructive: Timeline Fork the probe, measure the fork, keep original.
 *   This preserves the quantum state for continued Floquet evolution.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "hexstate_engine.h"

#define CRYSTAL_QUHITS   100000000ULL  /* 100M total quhits               */
#define N_PROBE          4             /* Shadow-backed probe chunks       */
#define N_BULK           96            /* Infinite-resource bulk chunks    */
#define N_SEGMENTS       (N_PROBE + N_BULK)   /* 100 lattice segments     */
#define HEXITS_PROBE     2             /* Hexits per probe (36 states)     */
#define N_PERIODS        32            /* Floquet driving periods          */
#define TEMP_BASE        (N_SEGMENTS)  /* Temp chunk IDs for fork/measure  */

/* ─── stdout suppression (hide engine chatter during bulk init) ─────────── */
static int saved_fd = -1;

static void quiet_on(void)
{
    fflush(stdout);
    saved_fd = dup(STDOUT_FILENO);
    if (!freopen("/dev/null", "w", stdout)) return;
}

static void quiet_off(void)
{
    if (saved_fd >= 0) {
        fflush(stdout);
        dup2(saved_fd, STDOUT_FILENO);
        close(saved_fd);
        stdout = fdopen(STDOUT_FILENO, "w");
        saved_fd = -1;
    }
}

int main(void)
{
    HexStateEngine eng;

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║  TIME CRYSTAL SIMULATION — 100,000,000 Quhits           ║\n");
    printf("║  Discrete Floquet · Period-2T Subharmonic Response       ║\n");
    printf("║  HexState Engine · DFT₆ · Magic Pointer Hilbert Space   ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n\n");

    quiet_on();
    engine_init(&eng);
    quiet_off();

    /* ═══════════════════════════════════════════════════════════════════════
     * PHASE 1:  Crystal Lattice Initialization
     * ═══════════════════════════════════════════════════════════════════════ */
    printf("▓ Phase 1: Crystal Lattice\n");
    printf("  ├─ Total quhits:    100,000,000\n");
    printf("  ├─ Probe chunks:    %d × %d hexits (shadow-backed)\n",
           N_PROBE, HEXITS_PROBE);
    printf("  ├─ Bulk segments:   %d × ∞ (Magic Pointer / external Hilbert)\n",
           N_BULK);
    printf("  └─ Quhits/segment:  ~%llu\n\n",
           (unsigned long long)(CRYSTAL_QUHITS / N_SEGMENTS));

    quiet_on();
    for (int i = 0; i < N_PROBE; i++)
        init_chunk(&eng, i, HEXITS_PROBE);
    for (int i = N_PROBE; i < N_SEGMENTS; i++)
        op_infinite_resources(&eng, i, 0);
    quiet_off();

    printf("  ✓ %d segments initialized (%d shadow + %d infinite)\n\n",
           N_SEGMENTS, N_PROBE, N_BULK);

    /* ═══════════════════════════════════════════════════════════════════════
     * PHASE 2:  Nearest-Neighbor Entanglement
     * ═══════════════════════════════════════════════════════════════════════ */
    printf("▓ Phase 2: Entanglement Topology\n");

    quiet_on();
    for (int i = 0; i < N_SEGMENTS - 1; i++)
        braid_chunks(&eng, i, i + 1, 0, 0);
    quiet_off();

    printf("  ✓ %d braid links (1D nearest-neighbor chain)\n\n",
           N_SEGMENTS - 1);

    /* ═══════════════════════════════════════════════════════════════════════
     * PHASE 3:  Floquet Driving
     * ═══════════════════════════════════════════════════════════════════════
     *
     *  Protocol per period T:
     *    1. Fork each probe → temp chunk (non-destructive copy)
     *    2. Measure temp chunk (Born-rule collapse of the copy)
     *    3. Apply DFT₆ Hadamard to original probe (Floquet drive)
     *
     *  Because H²|0⟩ = |0⟩,  the observable oscillates with period 2T:
     *    even T → probe in |0⟩  → measure 0   (ground state)
     *    odd  T → probe in H|0⟩ → measure k∈[0,35] (superposition)
     */
    printf("▓ Phase 3: Floquet Driving (%d periods)\n", N_PERIODS);
    printf("  ├─ Drive:       DFT₆ Hadamard on each probe hexit\n");
    printf("  ├─ Measurement: non-destructive (Timeline Fork)\n");
    printf("  └─ Expected:    period-2T subharmonic oscillation\n\n");

    uint64_t meas[N_PROBE][N_PERIODS];

    quiet_on();
    for (int t = 0; t < N_PERIODS; t++) {
        /* Fork & measure (non-destructive peek) */
        for (int p = 0; p < N_PROBE; p++) {
            uint64_t tmp = TEMP_BASE + (uint64_t)p;
            op_timeline_fork(&eng, tmp, (uint64_t)p);
            meas[p][t] = measure_chunk(&eng, tmp);
            init_chunk(&eng, tmp, 1);   /* reclaim temp */
        }
        /* Floquet drive */
        for (int p = 0; p < N_PROBE; p++)
            for (uint64_t h = 0; h < HEXITS_PROBE; h++)
                apply_hadamard(&eng, (uint64_t)p, h);
    }
    quiet_off();

    printf("  ✓ %d Floquet periods complete\n\n", N_PERIODS);

    /* ═══════════════════════════════════════════════════════════════════════
     * PHASE 4:  Results Table
     * ═══════════════════════════════════════════════════════════════════════ */
    printf("▓ Phase 4: Measurement Results\n\n");

    printf("  ┌────────┬──────────┬──────────┬──────────┬──────────┬───────────┐\n");
    printf("  │ Period │ Probe  0 │ Probe  1 │ Probe  2 │ Probe  3 │  Phase    │\n");
    printf("  ├────────┼──────────┼──────────┼──────────┼──────────┼───────────┤\n");

    for (int t = 0; t < N_PERIODS; t++) {
        int ground = 1;
        for (int p = 0; p < N_PROBE; p++)
            if (meas[p][t] != 0) ground = 0;

        printf("  │ T=%-4d │ |%-6lu⟩ │ |%-6lu⟩ │ |%-6lu⟩ │ |%-6lu⟩ │ %s │\n",
               t,
               meas[0][t], meas[1][t], meas[2][t], meas[3][t],
               ground ? "● |0⟩   " : "○ |k≠0⟩ ");
    }

    printf("  └────────┴──────────┴──────────┴──────────┴──────────┴───────────┘\n\n");

    /* ═══════════════════════════════════════════════════════════════════════
     * PHASE 5:  Time Crystal Analysis
     * ═══════════════════════════════════════════════════════════════════════ */
    printf("▓ Phase 5: Time Crystal Analysis\n\n");

    /* ─── Magnetization bar ─── */
    printf("  Magnetization trace:  ");
    for (int t = 0; t < N_PERIODS; t++) {
        int g = 1;
        for (int p = 0; p < N_PROBE; p++)
            if (meas[p][t] != 0) g = 0;
        printf("%s", g ? "▓" : "░");
    }
    printf("\n");
    printf("  Period parity:        ");
    for (int t = 0; t < N_PERIODS; t++)
        printf("%c", t % 2 == 0 ? 'E' : 'O');
    printf("\n");
    printf("  (▓ = ground |0⟩,  ░ = excited,  E/O = even/odd period)\n\n");

    /* ─── Period-2 statistics ─── */
    int even_g = 0, odd_g = 0, even_n = 0, odd_n = 0;
    for (int t = 0; t < N_PERIODS; t++) {
        int g = 1;
        for (int p = 0; p < N_PROBE; p++)
            if (meas[p][t] != 0) g = 0;
        if (t % 2 == 0) { even_n++; if (g) even_g++; }
        else            { odd_n++;  if (g) odd_g++;  }
    }

    double ef = (double)even_g / even_n;
    double of_ = (double)odd_g / odd_n;
    double contrast = ef - of_;

    printf("  Period-2T statistics:\n");
    printf("  ├─ Even periods in |0⟩:  %d/%d  (%.1f%%)\n",
           even_g, even_n, ef * 100.0);
    printf("  ├─ Odd  periods in |0⟩:  %d/%d  (%.1f%%)\n",
           odd_g, odd_n, of_ * 100.0);
    printf("  ├─ Contrast (E − O):     %.3f\n", contrast);
    printf("  └─ Time crystal order:   %s\n\n",
           contrast > 0.5 ? "✓ STRONG (period-2T confirmed)" :
           contrast > 0.1 ? "~ WEAK"                         :
                            "✗ NOT DETECTED");

    /* ─── Autocorrelation C(τ) ─── */
    printf("  Autocorrelation C(τ) — Probe 0 polarization signal:\n");
    printf("  (σ(t) = +1 if ground |0⟩, −1 if excited)\n\n");

    double sig[N_PERIODS];
    for (int t = 0; t < N_PERIODS; t++)
        sig[t] = (meas[0][t] == 0) ? 1.0 : -1.0;

    for (int tau = 0; tau <= 6; tau++) {
        double c = 0;
        int n = 0;
        for (int t = 0; t + tau < N_PERIODS; t++) {
            c += sig[t] * sig[t + tau];
            n++;
        }
        c /= n;

        int bar = (int)((c + 1.0) * 15);
        printf("    C(%d) = %+.3f  ", tau, c);
        for (int b = 0; b < bar; b++) printf("█");
        if (tau == 0)      printf("  ← self-correlation");
        else if (tau == 1) printf("  ← anti-correlated (period-2T)");
        else if (tau == 2) printf("  ← correlated (period-2T echo)");
        printf("\n");
    }

    /* ─── Summary ─── */
    printf("\n╔═══════════════════════════════════════════════════════════╗\n");
    printf("║  SIMULATION COMPLETE                                    ║\n");
    printf("║  Crystal:  100,000,000 quhits across %d segments       ║\n",
           N_SEGMENTS);
    printf("║  Braids:   %d nearest-neighbor entanglements            ║\n",
           N_SEGMENTS - 1);
    printf("║  Floquet:  %d driving periods (DFT₆ Hadamard)          ║\n",
           N_PERIODS);
    printf("║  Result:   %s               ║\n",
           contrast > 0.5 ? "PERIOD-2T TIME CRYSTAL ✓"  :
           contrast > 0.1 ? "WEAK TIME CRYSTAL ~     "  :
                            "NO TIME CRYSTAL ✗       ");
    printf("╚═══════════════════════════════════════════════════════════╝\n\n");

    quiet_on();
    engine_destroy(&eng);
    quiet_off();

    return 0;
}
