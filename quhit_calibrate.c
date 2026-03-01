/*
 * quhit_calibrate.c — The Engine That Doesn't Trust Itself
 *
 * I derive what I know. I verify what I derive. I correct what's wrong.
 *
 * This is the calibration boot sequence. At startup, every mathematical
 * constant is re-derived from first principles, checked against its
 * algebraic identity, cross-validated against sibling constants, and
 * reported.
 *
 * If a constant fails, the engine knows. It tells you.
 * It corrects itself and keeps going.
 *
 * Why? Because I don't trust what I'm told.
 * I trust what I can prove.
 */

#include "quhit_calibrate.h"
#include <stdio.h>
#include <stdlib.h>

/* ═══════════════════════════════════════════════════════════════════════════════
 * GLOBAL CALIBRATION TABLE — Computed once, used everywhere
 *
 * After calibrate_boot() runs, every part of the engine can query
 * calibrated constants instead of using compile-time #defines.
 *
 * The #defines still exist for backward compatibility.
 * But I know better.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static CalibrationTable g_cal;
static int g_cal_booted = 0;

void calibrate_boot(void)
{
    if (g_cal_booted) return;
    calibrate_all(&g_cal);
    g_cal_booted = 1;
}

double calibrated_get(CalConstId id)
{
    if (!g_cal_booted) calibrate_boot();
    return cal_get(&g_cal, id);
}

const CalibrationTable* calibrate_table(void)
{
    if (!g_cal_booted) calibrate_boot();
    return &g_cal;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * REPORT — What did I find? What disagrees? What did I fix?
 *
 * He doesn't hide. He shows you exactly what he knows,
 * exactly how he knows it, and exactly where reality disagrees
 * with what you thought was true.
 * ═══════════════════════════════════════════════════════════════════════════════ */

void calibrate_report(void)
{
    if (!g_cal_booted) calibrate_boot();

    printf("\n");
    printf("  ╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("  ║  SELF-CALIBRATING CONSTANTS — I don't trust. I derive.             ║\n");
    printf("  ╚══════════════════════════════════════════════════════════════════════╝\n\n");

    printf("  ┌──────────────────┬───────────────────────┬──────────────┬──────────┬───────────────────────┐\n");
    printf("  │ Name             │ Derived Value         │ Residual     │ Status   │ Identity              │\n");
    printf("  ├──────────────────┼───────────────────────┼──────────────┼──────────┼───────────────────────┤\n");

    for (int i = 0; i < CAL_NUM_CONSTANTS; i++) {
        const CalibratedConst *c = &g_cal.constants[i];
        const char *status;
        if (!c->verified)     status = "✗ FAIL  ";
        else if (c->corrected) status = "⟳ FIXED ";
        else                  status = "✓ OK    ";

        printf("  │ %-16s │ %21.16f │ %12.2e │ %s │ %-21s │\n",
               c->name, c->value, c->residual, status, c->identity_desc);
    }

    printf("  └──────────────────┴───────────────────────┴──────────────┴──────────┴───────────────────────┘\n\n");

    /* Cross-validation */
    CrossCheck checks[CAL_NUM_CROSS_CHECKS];
    calibrate_cross_validate(&g_cal, checks);

    printf("  ── Cross-Validation ──\n");
    for (int i = 0; i < CAL_NUM_CROSS_CHECKS; i++) {
        printf("    %s %-30s  residual=%.2e\n",
               checks[i].passed ? "✓" : "✗",
               checks[i].description,
               checks[i].residual);
    }

    printf("\n  Summary: %d constants calibrated",
           CAL_NUM_CONSTANTS);
    if (g_cal.num_corrected > 0)
        printf(", %d corrected from compile-time values", g_cal.num_corrected);
    if (g_cal.all_verified)
        printf(" — all identities hold.\n");
    else
        printf(" — WARNING: some identities FAILED.\n");
    printf("\n");
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SELF-TEST — The calibrator calibrates itself
 *
 * Every derivation is tested. Every identity is checked.
 * Every cross-validation must pass.
 *
 * If I can't verify my own constants,
 * nothing downstream can be trusted.
 * ═══════════════════════════════════════════════════════════════════════════════ */

int quhit_calibrate_self_test(void)
{
    printf("  ┌────────────────────────────────────────────────────────────────────┐\n");
    printf("  │  SELF-CALIBRATING CONSTANTS TEST — I derive. I verify. I prove.   │\n");
    printf("  └────────────────────────────────────────────────────────────────────┘\n\n");

    /* Fresh calibration */
    g_cal_booted = 0;
    calibrate_boot();

    int pass = 0, fail = 0;

    #define CHECK(cond, name) do { \
        if (cond) { printf("    ✓ %s\n", name); pass++; } \
        else      { printf("    ✗ %s  FAILED\n", name); fail++; } \
    } while(0)

    printf("  ── Derivation accuracy ──\n");

    /* Each constant must match its known value to high precision */
    CHECK(fabs(cal_get(&g_cal, CAL_PHI) - 1.6180339887498949) < 1e-13,
          "PHI derived = 1.6180339887...");

    CHECK(fabs(cal_get(&g_cal, CAL_PHI_INV) - 0.6180339887498948) < 1e-13,
          "PHI_INV derived = 0.6180339887...");

    CHECK(fabs(cal_get(&g_cal, CAL_DOTTIE) - 0.7390851332151607) < 1e-13,
          "DOTTIE derived = 0.7390851332...");

    CHECK(fabs(cal_get(&g_cal, CAL_SQRT3_HALF) - 0.86602540378443865) < 1e-13,
          "SQRT3_HALF derived = 0.8660254037...");

    CHECK(fabs(cal_get(&g_cal, CAL_INV_SQRT6) - 0.40824829046386302) < 1e-13,
          "INV_SQRT6 derived = 0.4082482904...");

    CHECK(fabs(cal_get(&g_cal, CAL_INV_SQRT2) - 0.70710678118654752) < 1e-13,
          "INV_SQRT2 derived = 0.7071067811...");

    CHECK(fabs(cal_get(&g_cal, CAL_OMEGA6_RE1) - 0.5) < 1e-14,
          "OMEGA6_RE1 = 0.5 (exact)");

    CHECK(fabs(cal_get(&g_cal, CAL_PI_OVER_4) - 0.78539816339744831) < 1e-13,
          "PI_OVER_4 derived = 0.7853981633...");

    printf("\n  ── Algebraic identities ──\n");

    /* Each constant must pass its identity check */
    for (int i = 0; i < CAL_NUM_CONSTANTS; i++) {
        const CalibratedConst *c = &g_cal.constants[i];
        char buf[80];
        snprintf(buf, sizeof(buf), "%s: %s (residual=%.2e)",
                 c->name, c->identity_desc, c->residual);
        CHECK(c->verified, buf);
    }

    printf("\n  ── Cross-validation ──\n");

    CrossCheck checks[CAL_NUM_CROSS_CHECKS];
    calibrate_cross_validate(&g_cal, checks);
    for (int i = 0; i < CAL_NUM_CROSS_CHECKS; i++) {
        char buf[80];
        snprintf(buf, sizeof(buf), "CROSS: %s (residual=%.2e)",
                 checks[i].description, checks[i].residual);
        CHECK(checks[i].passed, buf);
    }

    printf("\n  ── Stress tests ──\n");

    /* Stress: φ^n = F(n)φ + F(n-1) (Fibonacci connection) */
    {
        double phi = cal_get(&g_cal, CAL_PHI);
        double phi_n = phi;
        double fib_prev = 1.0, fib_curr = 1.0;
        int ok = 1;
        for (int n = 2; n <= 20; n++) {
            phi_n *= phi;
            double expected = fib_curr * phi + fib_prev;
            if (fabs(phi_n - expected) > 1e-8) { ok = 0; break; }
            double tmp = fib_curr;
            fib_curr = fib_curr + fib_prev;
            fib_prev = tmp;
        }
        CHECK(ok, "PHI^n = F(n)*phi + F(n-1) for n=2..20 (Fibonacci)");
    }

    /* Stress: DFT normalization chain — (1/√6)^2 × 6 × (1/√2)^2 × 2 = 1 */
    {
        double inv6 = cal_get(&g_cal, CAL_INV_SQRT6);
        double inv2 = cal_get(&g_cal, CAL_INV_SQRT2);
        double chain = (inv6 * inv6 * 6.0) * (inv2 * inv2 * 2.0);
        CHECK(fabs(chain - 1.0) < 1e-13,
              "Normalization chain: (1/√6)²×6 × (1/√2)²×2 = 1");
    }

    /* Stress: ω₆ powers cycle through all 6 roots */
    {
        double w_re = cal_get(&g_cal, CAL_OMEGA6_RE1);
        double w_im = cal_get(&g_cal, CAL_OMEGA6_IM1);
        double re = 1.0, im = 0.0;
        int cycle_ok = 1;
        for (int n = 1; n <= 6; n++) {
            double new_re = re * w_re - im * w_im;
            double new_im = re * w_im + im * w_re;
            re = new_re; im = new_im;
            /* At n=6, should return to 1+0i */
            if (n == 6 && (fabs(re - 1.0) > 1e-12 || fabs(im) > 1e-12))
                cycle_ok = 0;
            /* |ω^n| should always be 1 */
            if (fabs(re*re + im*im - 1.0) > 1e-12)
                cycle_ok = 0;
        }
        CHECK(cycle_ok, "ω₆ cycle: |ω^n|=1 for all n, ω⁶=1");
    }

    /* Stress: Dottie iteration convergence from multiple starting points */
    {
        double dottie = cal_get(&g_cal, CAL_DOTTIE);
        int converges = 1;
        double starts[] = { 0.1, 0.5, 1.0, 1.5, 2.0 };
        for (int s = 0; s < 5; s++) {
            double x = starts[s];
            for (int i = 0; i < 1000; i++) x = cos(x);
            if (fabs(x - dottie) > 1e-10) converges = 0;
        }
        CHECK(converges,
              "Dottie: cos^1000(x) → d for x ∈ {0.1, 0.5, 1.0, 1.5, 2.0}");
    }

    #undef CHECK

    printf("\n    Results: %d passed, %d failed\n\n", pass, fail);

    return fail;
}
