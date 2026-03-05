/*
 * peps_accuracy_test.c — Unified PEPS 2D-6D Accuracy Benchmark
 *
 * Tests each overlay dimension (2D through 6D) with:
 *   1. DFT uniformity on product state
 *   2. Single Trotter layer (DFT all + CZ all axes)
 *   3. Multi-layer Trotter evolution — local density vs initial
 *
 * Same format as mps_accuracy_test.c.
 * Lattice: 2 sites per axis (2^d total sites).
 *
 * Build:
 *   gcc -O2 -march=native -o peps_accuracy_test peps_accuracy_test.c \
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

#include "mps_overlay.h"
#include "peps_overlay.h"
#include "peps3d_overlay.h"
#include "peps4d_overlay.h"
#include "peps5d_overlay.h"
#include "peps6d_overlay.h"

#define D 6
#define TOL 1e-6

static int tests_passed = 0, tests_total = 0;

static void check(const char *label, int pass) {
    tests_total++;
    if (pass) { tests_passed++; printf("  ✓ PASS: %s\n", label); }
    else                       { printf("  ✗ FAIL: %s\n", label); }
}

static int is_uniform(const double *p, int n) {
    double expected = 1.0 / n;
    for (int i = 0; i < n; i++)
        if (fabs(p[i] - expected) > TOL) return 0;
    return 1;
}

/* ═══════════════════════════════════════════════════════════════════
 * GATE BUILDERS (shared across all tests)
 * ═══════════════════════════════════════════════════════════════════ */
static double DFT_re[36], DFT_im[36];
static double CZ_re[36*36], CZ_im[36*36];

static void build_gates(void) {
    mps_build_dft6(DFT_re, DFT_im);
    mps_build_cz(CZ_re, CZ_im);
}

/* ═══════════════════════════════════════════════════════════════════
 * TEST: 1D MPS (baseline reference)
 * ═══════════════════════════════════════════════════════════════════ */
static void test_1d(void) {
    printf("\n━━━ 1D MPS (L=4) ━━━\n");
    MpsChain *c = mps_init(4);

    /* DFT all → uniform? */
    mps_gate_1site_all(c, DFT_re, DFT_im);
    double p[6];
    mps_local_density(c, 0, p);
    check("1D DFT uniform", is_uniform(p, D));

    /* Trotter: DFT + CZ sweep × 3 layers */
    for (int lay = 0; lay < 3; lay++) {
        mps_gate_1site_all(c, DFT_re, DFT_im);
        for (int b = 0; b < 3; b++)
            mps_gate_bond(c, b, CZ_re, CZ_im);
    }
    mps_local_density(c, 0, p);
    double total_p = 0;
    for (int k = 0; k < D; k++) total_p += p[k];
    check("1D 3-layer Trotter normalized", fabs(total_p - 1.0) < TOL);

    /* Check all sites produce valid densities */
    int all_valid = 1;
    for (int s = 0; s < 4; s++) {
        mps_local_density(c, s, p);
        double sum = 0;
        for (int k = 0; k < D; k++) { sum += p[k]; if (p[k] < -TOL) all_valid = 0; }
        if (fabs(sum - 1.0) > TOL) all_valid = 0;
    }
    check("1D all sites valid density", all_valid);
    mps_free(c);
}

/* ═══════════════════════════════════════════════════════════════════
 * TEST: 2D PEPS (2×2)
 * ═══════════════════════════════════════════════════════════════════ */
static void test_2d(void) {
    printf("\n━━━ 2D PEPS (2×2) ━━━\n");
    PepsGrid *g = peps_init(2, 2);

    /* DFT all */
    peps_gate_1site_all(g, DFT_re, DFT_im);
    double p[6];
    peps_local_density(g, 0, 0, p);
    check("2D DFT uniform", is_uniform(p, D));

    /* Trotter: DFT + CZ horizontal + CZ vertical × 2 layers */
    for (int lay = 0; lay < 2; lay++) {
        peps_gate_1site_all(g, DFT_re, DFT_im);
        peps_gate_horizontal_all(g, CZ_re, CZ_im);
        peps_gate_vertical_all(g, CZ_re, CZ_im);
    }

    /* Check all sites produce valid densities */
    int all_valid = 1;
    for (int y = 0; y < 2; y++)
     for (int x = 0; x < 2; x++) {
         peps_local_density(g, x, y, p);
         double sum = 0;
         for (int k = 0; k < D; k++) { sum += p[k]; if (p[k] < -TOL) all_valid = 0; }
         if (fabs(sum - 1.0) > TOL) all_valid = 0;
     }
    check("2D 2-layer Trotter all sites valid", all_valid);
    peps_free(g);
}

/* ═══════════════════════════════════════════════════════════════════
 * TEST: 3D PEPS (2×2×2)
 * ═══════════════════════════════════════════════════════════════════ */
static void test_3d(void) {
    printf("\n━━━ 3D PEPS (2×2×2) ━━━\n");
    Tns3dGrid *g = tns3d_init(2, 2, 2);

    /* DFT all */
    tns3d_gate_1site_all(g, DFT_re, DFT_im);
    double p[6];
    tns3d_local_density(g, 0, 0, 0, p);
    check("3D DFT uniform", is_uniform(p, D));

    /* Trotter: DFT + CZ on all 3 axes × 2 layers */
    for (int lay = 0; lay < 2; lay++) {
        tns3d_gate_1site_all(g, DFT_re, DFT_im);
        tns3d_gate_x_all(g, CZ_re, CZ_im);
        tns3d_gate_y_all(g, CZ_re, CZ_im);
        tns3d_gate_z_all(g, CZ_re, CZ_im);
    }

    int all_valid = 1;
    for (int z = 0; z < 2; z++)
     for (int y = 0; y < 2; y++)
      for (int x = 0; x < 2; x++) {
          tns3d_local_density(g, x, y, z, p);
          double sum = 0;
          for (int k = 0; k < D; k++) { sum += p[k]; if (p[k] < -TOL) all_valid = 0; }
          if (fabs(sum - 1.0) > TOL) all_valid = 0;
      }
    check("3D 2-layer Trotter all sites valid", all_valid);
    tns3d_free(g);
}

/* ═══════════════════════════════════════════════════════════════════
 * TEST: 4D PEPS (2×2×2×2)
 * ═══════════════════════════════════════════════════════════════════ */
static void test_4d(void) {
    printf("\n━━━ 4D PEPS (2×2×2×2) ━━━\n");
    Tns4dGrid *g = tns4d_init(2, 2, 2, 2);

    tns4d_gate_1site_all(g, DFT_re, DFT_im);
    double p[6];
    tns4d_local_density(g, 0, 0, 0, 0, p);
    check("4D DFT uniform", is_uniform(p, D));

    /* Trotter: DFT + CZ all 4 axes */
    tns4d_gate_1site_all(g, DFT_re, DFT_im);
    tns4d_gate_x_all(g, CZ_re, CZ_im);
    tns4d_gate_y_all(g, CZ_re, CZ_im);
    tns4d_gate_z_all(g, CZ_re, CZ_im);
    tns4d_gate_w_all(g, CZ_re, CZ_im);

    int all_valid = 1;
    for (int w = 0; w < 2; w++)
     for (int z = 0; z < 2; z++)
      for (int y = 0; y < 2; y++)
       for (int x = 0; x < 2; x++) {
           tns4d_local_density(g, x, y, z, w, p);
           double sum = 0;
           for (int k = 0; k < D; k++) { sum += p[k]; if (p[k] < -TOL) all_valid = 0; }
           if (fabs(sum - 1.0) > TOL) all_valid = 0;
       }
    check("4D 1-layer Trotter all sites valid", all_valid);
    tns4d_free(g);
}

/* ═══════════════════════════════════════════════════════════════════
 * TEST: 5D PEPS (2×2×2×2×2)
 * ═══════════════════════════════════════════════════════════════════ */
static void test_5d(void) {
    printf("\n━━━ 5D PEPS (2×2×2×2×2) ━━━\n");
    Tns5dGrid *g = tns5d_init(2, 2, 2, 2, 2);

    tns5d_gate_1site_all(g, DFT_re, DFT_im);
    double p[6];
    tns5d_local_density(g, 0, 0, 0, 0, 0, p);
    check("5D DFT uniform", is_uniform(p, D));

    /* Trotter: DFT + CZ all 5 axes */
    tns5d_gate_1site_all(g, DFT_re, DFT_im);
    tns5d_gate_x_all(g, CZ_re, CZ_im);
    tns5d_gate_y_all(g, CZ_re, CZ_im);
    tns5d_gate_z_all(g, CZ_re, CZ_im);
    tns5d_gate_w_all(g, CZ_re, CZ_im);
    tns5d_gate_v_all(g, CZ_re, CZ_im);

    int all_valid = 1;
    for (int v = 0; v < 2; v++)
     for (int w = 0; w < 2; w++)
      for (int z = 0; z < 2; z++)
       for (int y = 0; y < 2; y++)
        for (int x = 0; x < 2; x++) {
            tns5d_local_density(g, x, y, z, w, v, p);
            double sum = 0;
            for (int k = 0; k < D; k++) { sum += p[k]; if (p[k] < -TOL) all_valid = 0; }
            if (fabs(sum - 1.0) > TOL) all_valid = 0;
        }
    check("5D 1-layer Trotter all sites valid", all_valid);
    tns5d_free(g);
}

/* ═══════════════════════════════════════════════════════════════════
 * TEST: 6D PEPS (2×2×2×2×2×2)
 * ═══════════════════════════════════════════════════════════════════ */
static void test_6d(void) {
    printf("\n━━━ 6D PEPS (2×2×2×2×2×2) ━━━\n");
    Tns6dGrid *g = tns6d_init(2, 2, 2, 2, 2, 2);

    tns6d_gate_1site_all(g, DFT_re, DFT_im);
    double p[6];
    tns6d_local_density(g, 0, 0, 0, 0, 0, 0, p);
    check("6D DFT uniform", is_uniform(p, D));

    /* Trotter: DFT + CZ all 6 axes */
    tns6d_gate_1site_all(g, DFT_re, DFT_im);
    tns6d_trotter_step(g, CZ_re, CZ_im);

    int all_valid = 1;
    for (int u = 0; u < 2; u++)
     for (int v = 0; v < 2; v++)
      for (int w = 0; w < 2; w++)
       for (int z = 0; z < 2; z++)
        for (int y = 0; y < 2; y++)
         for (int x = 0; x < 2; x++) {
             tns6d_local_density(g, x, y, z, w, v, u, p);
             double sum = 0;
             for (int k = 0; k < D; k++) { sum += p[k]; if (p[k] < -TOL) all_valid = 0; }
             if (fabs(sum - 1.0) > TOL) all_valid = 0;
         }
    check("6D 1-layer Trotter all sites valid", all_valid);
    tns6d_free(g);
}

/* ═══════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║   PEPS ACCURACY TEST — 1D through 6D Overlay Benchmark    ║\n");
    printf("║   D=6  ·  Standard SVD  ·  σ-on-bonds                    ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    build_gates();

    test_1d();
    test_2d();
    test_3d();
    test_4d();
    test_5d();
    test_6d();

    printf("\n══════════════════════════════════════════════════════════════\n");
    printf("  RESULTS: %d / %d tests passed\n", tests_passed, tests_total);
    if (tests_passed == tests_total)
        printf("  ✓ ALL TESTS PASSED\n");
    else
        printf("  ✗ %d TESTS FAILED\n", tests_total - tests_passed);
    printf("══════════════════════════════════════════════════════════════\n");
    return (tests_passed == tests_total) ? 0 : 1;
}
