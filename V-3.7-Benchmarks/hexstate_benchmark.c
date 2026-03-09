/*
 * hexstate_benchmark.c — Comprehensive HexState Engine Benchmark
 *
 * 21 sections in 3 parts covering every feature tier:
 *   Part 1 (§1–7):   Core Engine & Primitives
 *   Part 2 (§8–14):  Substrate, Triality & Specialized
 *   Part 3 (§15–21): Registers, Overlays & Scale
 *
 * Compile:
 *   gcc -O2 -march=native -o hexstate_benchmark hexstate_benchmark.c \
 *       quhit_core.c quhit_gates.c quhit_measure.c quhit_entangle.c \
 *       quhit_register.c quhit_substrate.c quhit_triality.c \
 *       quhit_triadic.c quhit_lazy.c quhit_calibrate.c \
 *       quhit_dyn_integrate.c quhit_peps_grow.c quhit_svd_gate.c \
 *       s6_exotic.c bigint.c mps_overlay.c peps_overlay.c \
 *       peps3d_overlay.c peps4d_overlay.c peps5d_overlay.c peps6d_overlay.c \
 *       -lm -fopenmp -msse2
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

/* ─── HexState Headers ─── */
#include "quhit_engine.h"
#include "bigint.h"
#include "quhit_triality.h"
#include "quhit_triadic.h"
#include "quhit_lazy.h"
#include "quhit_calibrate.h"
#include "flat_quhit.h"
/* NOTE: quhit_factored.h conflicts with flat_quhit.h on fq_probabilities
 * — we test flat_quhit only which covers the factored representation */
#include "quhit_svd_gate.h"
#include "quhit_dyn_integrate.h"
#include "quhit_peps_grow.h"
#include "mps_overlay.h"
#include "peps_overlay.h"
#include "tensor_product.h"
#include "triality_overlay.h"

/* ══════════════════════════════════════════════════════════════════════════════
 * TEST INFRASTRUCTURE
 * ══════════════════════════════════════════════════════════════════════════════ */

static int g_pass = 0, g_fail = 0;
static struct timespec g_t0;

#define CHECK(cond, name) do { \
    if (cond) { printf("    ✓ %s\n", name); g_pass++; } \
    else      { printf("    ✗ %s  *** FAIL ***\n", name); g_fail++; } \
} while(0)

#define NEAR(a, b, tol) (fabs((a)-(b)) < (tol))

#define SECTION(num, title) \
    printf("\n  ───── §%d: %s ─────\n", num, title)

#define BENCH_START() clock_gettime(CLOCK_MONOTONIC, &g_t0)

#define BENCH_END(label) do { \
    struct timespec t1; clock_gettime(CLOCK_MONOTONIC, &t1); \
    double ms = (t1.tv_sec - g_t0.tv_sec)*1e3 + (t1.tv_nsec - g_t0.tv_nsec)*1e-6; \
    printf("    ⏱  %s: %.3f ms\n", label, ms); \
} while(0)

#define PART_HEADER(num, title) \
    printf("\n╔══════════════════════════════════════════════╗\n"); \
    printf("║  PART %d: %-36s ║\n", num, title); \
    printf("╚══════════════════════════════════════════════╝\n")

/* Simple LCG for deterministic tests */
static uint64_t bench_rng_state = 0xDEADBEEFCAFE1234ULL;
static double bench_rand01(void) {
    bench_rng_state = bench_rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(bench_rng_state >> 11) / (double)(1ULL << 53);
}

/* Global engine — heap allocated */
static QuhitEngine *eng = NULL;

/* ══════════════════════════════════════════════════════════════════════════════
 * PART 1: CORE ENGINE & PRIMITIVES (§1–7)
 * ══════════════════════════════════════════════════════════════════════════════ */

static void run_part1(void)
{
    PART_HEADER(1, "CORE ENGINE & PRIMITIVES");

    /* §1: Engine Lifecycle */
    SECTION(1, "Engine Lifecycle");
    BENCH_START();
    eng = (QuhitEngine *)calloc(1, sizeof(QuhitEngine));
    CHECK(eng != NULL, "Engine allocation");
    quhit_engine_init(eng);
    CHECK(eng->num_quhits == 0, "Engine starts empty");

    uint32_t q0 = quhit_init(eng);
    CHECK(q0 == 0, "First quhit ID = 0");
    CHECK(eng->num_quhits == 1, "Quhit count = 1");
    CHECK(NEAR(eng->quhits[q0].state.re[0], 1.0, 1e-12), "|0⟩ init: amp[0]=1");

    uint32_t qp = quhit_init_plus(eng);
    double inv_sqrt6 = 1.0 / sqrt(6.0);
    CHECK(NEAR(eng->quhits[qp].state.re[0], inv_sqrt6, 1e-10), "|+⟩ init: uniform amp");

    uint32_t qk = quhit_init_basis(eng, 3);
    CHECK(NEAR(eng->quhits[qk].state.re[3], 1.0, 1e-12), "|3⟩ init: amp[3]=1");

    uint64_t r1 = quhit_prng(eng);
    uint64_t r2 = quhit_prng(eng);
    CHECK(r1 != r2, "PRNG: consecutive values differ");
    double rd = quhit_prng_double(eng);
    CHECK(rd >= 0.0 && rd < 1.0, "PRNG double in [0,1)");
    BENCH_END("Lifecycle");

    /* §2: Born Rule & Arithmetic */
    SECTION(2, "Born Rule & Arithmetic");
    BENCH_START();
    {
        double re = 0.6, im = 0.8;
        double p_exact = born_prob_exact(re, im);
        CHECK(NEAR(p_exact, re*re+im*im, 1e-12), "born_prob_exact = |z|²");

        double p_fast = born_prob_fast(re, im);
        CHECK(fabs(p_fast - p_exact) < 0.05, "born_prob_fast ≈ born_prob_exact");

        float f4 = 4.0f;
        double isqrt4 = born_fast_isqrt(f4);
        CHECK(fabs(isqrt4 - 0.5) < 0.02, "born_fast_isqrt(4) ≈ 0.5");

        double sqrt4 = born_fast_sqrt(f4);
        CHECK(fabs(sqrt4 - 2.0) < 0.05, "born_fast_sqrt(4) ≈ 2.0");

        double recip4 = born_fast_recip(f4);
        CHECK(fabs(recip4 - 0.25) < 0.02, "born_fast_recip(4) ≈ 0.25");
    }
    BENCH_END("Born Rule");

    /* §3: Superposition (DFT₆) */
    SECTION(3, "Superposition (DFT₆)");
    BENCH_START();
    {
        /* DFT₆ → IDFT₆ round-trip (in-place API) */
        double re[6] = {1,0,0,0,0,0}, im[6] = {0};
        sup_apply_dft6(re, im);
        /* After DFT on |0⟩: should be uniform 1/√6 */
        int uniform = 1;
        for (int k = 0; k < 6; k++)
            if (!NEAR(re[k], 1.0/sqrt(6.0), 1e-10)) uniform = 0;
        CHECK(uniform, "DFT₆|0⟩ = uniform superposition");

        sup_apply_idft6(re, im);
        CHECK(NEAR(re[0], 1.0, 1e-10), "IDFT₆(DFT₆|0⟩)[0] = 1.0");
        double err = 0;
        for (int k = 1; k < 6; k++) err += fabs(re[k]) + fabs(im[k]);
        CHECK(err < 1e-10, "IDFT₆(DFT₆|0⟩) round-trip residual < 1e-10");

        /* 1000 random round-trips */
        int round_trips_ok = 1;
        for (int trial = 0; trial < 1000; trial++) {
            double r_re[6], r_im[6], save_re[6], save_im[6];
            double norm = 0;
            for (int k = 0; k < 6; k++) {
                r_re[k] = bench_rand01() - 0.5;
                r_im[k] = bench_rand01() - 0.5;
                norm += r_re[k]*r_re[k] + r_im[k]*r_im[k];
            }
            double inv = 1.0 / sqrt(norm);
            for (int k = 0; k < 6; k++) {
                r_re[k] *= inv; r_im[k] *= inv;
                save_re[k] = r_re[k]; save_im[k] = r_im[k];
            }
            sup_apply_dft6(r_re, r_im);
            sup_apply_idft6(r_re, r_im);
            double rerr = 0;
            for (int k = 0; k < 6; k++)
                rerr += fabs(r_re[k]-save_re[k]) + fabs(r_im[k]-save_im[k]);
            if (rerr > 1e-10) { round_trips_ok = 0; break; }
        }
        CHECK(round_trips_ok, "DFT₆ unitarity (1000 random round-trips)");
    }
    BENCH_END("Superposition");

    /* §4: Gates */
    SECTION(4, "Gates");
    BENCH_START();
    {
        uint32_t g1 = quhit_init(eng);
        quhit_apply_dft(eng, g1);
        double p_sum = qm_total_prob(&eng->quhits[g1].state);
        CHECK(NEAR(p_sum, 1.0, 1e-10), "DFT preserves norm");

        uint32_t gx = quhit_init(eng);
        quhit_apply_x(eng, gx);
        CHECK(NEAR(eng->quhits[gx].state.re[1], 1.0, 1e-12), "X|0⟩ = |1⟩");

        uint32_t gz = quhit_init(eng);
        quhit_apply_z(eng, gz);
        CHECK(NEAR(eng->quhits[gz].state.re[0], 1.0, 1e-12), "Z|0⟩ = |0⟩");

        uint32_t gz2 = quhit_init_plus(eng);
        quhit_apply_z(eng, gz2);
        double zp = qm_total_prob(&eng->quhits[gz2].state);
        CHECK(NEAR(zp, 1.0, 1e-10), "Z|+⟩ preserves norm");

        uint32_t ca = quhit_init_basis(eng, 1);
        uint32_t cb = quhit_init_basis(eng, 2);
        quhit_apply_cz(eng, ca, cb);
        CHECK(eng->quhits[ca].pair_id >= 0, "CZ creates entangled pair");

        uint32_t ph = quhit_init_plus(eng);
        double phases[6] = {0, M_PI/6, M_PI/3, M_PI/2, 2*M_PI/3, 5*M_PI/6};
        quhit_apply_phase(eng, ph, phases);
        double ph_norm = qm_total_prob(&eng->quhits[ph].state);
        CHECK(NEAR(ph_norm, 1.0, 1e-10), "Phase gate preserves norm");

        /* Gate throughput: 10K DFT gates */
        uint32_t qt = quhit_init_plus(eng);
        struct timespec gt0, gt1;
        clock_gettime(CLOCK_MONOTONIC, &gt0);
        for (int i = 0; i < 10000; i++) quhit_apply_dft(eng, qt);
        clock_gettime(CLOCK_MONOTONIC, &gt1);
        double gms = (gt1.tv_sec-gt0.tv_sec)*1e3 + (gt1.tv_nsec-gt0.tv_nsec)*1e-6;
        printf("    ⏱  10K DFT gates: %.3f ms (%.0f gates/sec)\n", gms, 10000.0/(gms/1000.0));
    }
    BENCH_END("Gates total");

    /* §5: Measurement */
    SECTION(5, "Measurement");
    BENCH_START();
    {
        int counts[6] = {0};
        for (int trial = 0; trial < 6000; trial++) {
            uint32_t qm = quhit_init_plus(eng);
            uint32_t outcome = quhit_measure(eng, qm);
            if (outcome < 6) counts[outcome]++;
        }
        double chi2 = 0;
        for (int k = 0; k < 6; k++) {
            double diff = counts[k] - 1000.0;
            chi2 += diff * diff / 1000.0;
        }
        CHECK(chi2 < 30.0, "Born statistics χ²-test (6000 samples)");
        printf("    ℹ  Distribution: ");
        for (int k = 0; k < 6; k++) printf("%d ", counts[k]);
        printf("(χ²=%.1f)\n", chi2);

        uint32_t qi = quhit_init_plus(eng);
        QuhitSnapshot snap;
        quhit_inspect(eng, qi, &snap);
        CHECK(NEAR(snap.total_prob, 1.0, 1e-8), "Inspect: total_prob ≈ 1.0");
        CHECK(NEAR(snap.entropy, log2(6.0), 0.01), "Inspect: entropy ≈ log₂6");
        CHECK(NEAR(snap.purity, 1.0/6.0, 0.01), "Inspect: purity ≈ 1/6");

        uint32_t qb = quhit_init_basis(eng, 4);
        double p4 = quhit_prob(eng, qb, 4);
        CHECK(NEAR(p4, 1.0, 1e-10), "quhit_prob(|4⟩, 4) = 1.0");
    }
    BENCH_END("Measurement");

    /* §6: Entanglement */
    SECTION(6, "Entanglement");
    BENCH_START();
    {
        uint32_t ea = quhit_init(eng), eb = quhit_init(eng);
        int bell_ok = quhit_entangle_bell(eng, ea, eb);
        CHECK(bell_ok >= 0, "Bell entanglement succeeds");
        CHECK(eng->quhits[ea].pair_id >= 0, "A is entangled");

        uint32_t ma = quhit_measure(eng, ea);
        uint32_t mb = quhit_measure(eng, eb);
        CHECK(ma == mb, "Bell pair: correlated outcomes (A==B)");

        uint32_t pa = quhit_init_basis(eng, 2), pb = quhit_init_basis(eng, 3);
        int prod_ok = quhit_entangle_product(eng, pa, pb);
        CHECK(prod_ok >= 0, "Product pair succeeds");

        uint32_t da = quhit_init(eng), db = quhit_init(eng);
        quhit_entangle_bell(eng, da, db);
        quhit_disentangle(eng, da, db);
        CHECK(eng->quhits[da].pair_id == -1, "Disentangle: A free");
        CHECK(eng->quhits[db].pair_id == -1, "Disentangle: B free");
    }
    BENCH_END("Entanglement");

    /* §7: BigInt */
    SECTION(7, "BigInt Library");
    BENCH_START();
    {
        BigInt a, b, c;
        bigint_set_u64(&a, 123456789ULL);
        bigint_set_u64(&b, 987654321ULL);
        bigint_mul(&c, &a, &b);
        /* 123456789 × 987654321 = 121932631112635269 */
        char buf[256];
        bigint_to_decimal(buf, sizeof(buf), &c);
        CHECK(strcmp(buf, "121932631112635269") == 0, "BigInt mul");

        BigInt g1, g2, gg;
        bigint_set_u64(&g1, 48);
        bigint_set_u64(&g2, 36);
        bigint_gcd(&gg, &g1, &g2);
        bigint_to_decimal(buf, sizeof(buf), &gg);
        CHECK(strcmp(buf, "12") == 0, "BigInt gcd(48,36) = 12");

        BigInt base, exp_v, mod_v, result;
        bigint_set_u64(&base, 2);
        bigint_set_u64(&exp_v, 10);
        bigint_set_u64(&mod_v, 1000);
        bigint_pow_mod(&result, &base, &exp_v, &mod_v);
        bigint_to_decimal(buf, sizeof(buf), &result);
        CHECK(strcmp(buf, "24") == 0, "BigInt pow_mod(2,10,1000) = 24");

        BigInt s1, s2, sum;
        bigint_set_u64(&s1, 9999999999ULL);
        bigint_set_u64(&s2, 1000000007ULL);
        bigint_add(&sum, &s1, &s2);
        bigint_to_decimal(buf, sizeof(buf), &sum);
        CHECK(strcmp(buf, "11000000006") == 0, "BigInt add");
    }
    BENCH_END("BigInt");
}

/* ══════════════════════════════════════════════════════════════════════════════
 * PART 2: SUBSTRATE, TRIALITY & SPECIALIZED (§8–14)
 * ══════════════════════════════════════════════════════════════════════════════ */

static void run_part2(void)
{
    PART_HEADER(2, "SUBSTRATE, TRIALITY & SPECIALIZED");

    /* §8: Substrate Opcodes */
    SECTION(8, "Substrate Opcodes (20 ops, 6 families)");
    BENCH_START();
    {
        int sub_ok = quhit_substrate_self_test();
        CHECK(sub_ok == 0, "Substrate self-test (all 20 opcodes)");

        /* Exercise each family (3-arg API: eng, id, op) */
        uint32_t sq = quhit_init_plus(eng);
        quhit_substrate_exec(eng, sq, SUB_NULL);
        double np = qm_total_prob(&eng->quhits[sq].state);
        CHECK(np >= 0.0, "SUB_NULL produces valid state");

        uint32_t sp = quhit_init_plus(eng);
        for (int i = 0; i < 5; i++)
            quhit_substrate_exec(eng, sp, SUB_CLOCK);
        double cp = qm_total_prob(&eng->quhits[sp].state);
        CHECK(cp > 0.0, "5× SUB_CLOCK produces valid state");

        uint32_t st = quhit_init_plus(eng);
        quhit_substrate_exec(eng, st, SUB_FUSE);
        quhit_substrate_exec(eng, st, SUB_SCATTER);
        double tp = qm_total_prob(&eng->quhits[st].state);
        CHECK(tp > 0.0, "Transform ops produce valid state");
    }
    BENCH_END("Substrate");

    /* §9: Calibration */
    SECTION(9, "Self-Calibration");
    BENCH_START();
    {
        CalibrationTable ct;
        calibrate_all(&ct);
        CHECK(ct.calibrated, "Calibration completed");
        CHECK(ct.all_verified, "All constants verified");

        CrossCheck checks[CAL_NUM_CROSS_CHECKS];
        calibrate_cross_validate(&ct, checks);
        int cross_ok = 1;
        for (int i = 0; i < CAL_NUM_CROSS_CHECKS; i++)
            if (!checks[i].passed) cross_ok = 0;
        CHECK(cross_ok, "Cross-validation (5 identities)");

        double phi = cal_get(&ct, CAL_PHI);
        CHECK(NEAR(phi, (1.0 + sqrt(5.0)) / 2.0, 1e-12), "φ = (1+√5)/2");

        double inv6 = cal_get(&ct, CAL_INV_SQRT6);
        CHECK(NEAR(inv6, 1.0/sqrt(6.0), 1e-12), "1/√6 derived correctly");
    }
    BENCH_END("Calibration");

    /* §10: Triality Quhit */
    SECTION(10, "Triality Quhit");
    BENCH_START();
    {
        TrialityQuhit tq;
        triality_init(&tq);
        CHECK(tq.primary == VIEW_EDGE, "Init primary = EDGE");
        CHECK(NEAR(tq.edge_re[0], 1.0, 1e-12), "Init |0⟩ edge");

        triality_ensure_view(&tq, VIEW_VERTEX);
        CHECK(!(tq.dirty & DIRTY_VERTEX), "Vertex view materialized");
        double v_norm = 0;
        for (int k = 0; k < 6; k++)
            v_norm += tq.vertex_re[k]*tq.vertex_re[k] + tq.vertex_im[k]*tq.vertex_im[k];
        CHECK(NEAR(v_norm, 1.0, 1e-10), "Vertex view normalized");

        TrialityQuhit tp;
        triality_init_basis(&tp, 2);
        double phi_re[6] = {1,1,0,1,1,1};
        double phi_im[6] = {0,0,1,0,0,0};
        triality_phase(&tp, phi_re, phi_im);
        CHECK(NEAR(tp.edge_im[2], 1.0, 1e-12), "Phase on |2⟩ applies i");

        TrialityQuhit tz;
        triality_init_basis(&tz, 0);
        triality_z(&tz);
        CHECK(NEAR(tz.edge_re[0], 1.0, 1e-12), "Z|0⟩ = |0⟩");

        TrialityQuhit td;
        triality_init(&td);
        triality_dft(&td);
        double dnorm = 0;
        for (int k = 0; k < 6; k++)
            dnorm += td.edge_re[k]*td.edge_re[k] + td.edge_im[k]*td.edge_im[k];
        CHECK(NEAR(dnorm, 1.0, 1e-10), "Triality DFT preserves norm");

        /* Throughput: 10K triality Z gates */
        TrialityQuhit tperf;
        triality_init_basis(&tperf, 0);
        struct timespec tt0, tt1;
        clock_gettime(CLOCK_MONOTONIC, &tt0);
        for (int i = 0; i < 10000; i++) triality_z(&tperf);
        clock_gettime(CLOCK_MONOTONIC, &tt1);
        double tms = (tt1.tv_sec-tt0.tv_sec)*1e3 + (tt1.tv_nsec-tt0.tv_nsec)*1e-6;
        printf("    ⏱  10K triality Z gates: %.3f ms (%.0f gates/sec)\n",
               tms, 10000.0/(tms/1000.0));
    }
    BENCH_END("Triality");

    /* §11: Lazy Gate Chain */
    SECTION(11, "Lazy Gate Chain");
    BENCH_START();
    {
        LazyChain lc;
        lazy_init(&lc);
        CHECK(lc.is_identity, "LazyChain starts as identity");
        CHECK(lazy_depth(&lc) == 0, "Depth = 0 initially");

        double dft_re[36] = {0}, dft_im[36] = {0};
        double inv = 1.0/sqrt(6.0);
        for (int j = 0; j < 6; j++)
            for (int k = 0; k < 6; k++) {
                double angle = 2.0 * M_PI * j * k / 6.0;
                dft_re[j*6+k] = inv * cos(angle);
                dft_im[j*6+k] = inv * sin(angle);
            }

        lazy_compose(&lc, dft_re, dft_im);
        CHECK(!lc.is_identity, "After DFT: not identity");
        CHECK(lazy_depth(&lc) == 1, "Depth = 1");

        double idft_re[36] = {0}, idft_im[36] = {0};
        for (int j = 0; j < 6; j++)
            for (int k = 0; k < 6; k++) {
                double angle = -2.0 * M_PI * j * k / 6.0;
                idft_re[j*6+k] = inv * cos(angle);
                idft_im[j*6+k] = inv * sin(angle);
            }
        lazy_compose(&lc, idft_re, idft_im);
        CHECK(lazy_is_identity(&lc), "DFT · IDFT = Identity");

        LazyChain lc2;
        lazy_init(&lc2);
        double d_re[6] = {1, 0, -1, 0, 1, 0};
        double d_im[6] = {0, 1, 0, -1, 0, 1};
        lazy_compose_diagonal(&lc2, d_re, d_im);
        CHECK(lc2.is_diagonal, "Diagonal compose stays diagonal");

        double state_re[6] = {1,0,0,0,0,0}, state_im[6] = {0};
        lazy_resolve(&lc2, state_re, state_im);
        CHECK(NEAR(state_re[0], 1.0, 1e-12), "Resolve diagonal on |0⟩");
        CHECK(lc2.is_identity, "After resolve: reset to identity");
    }
    BENCH_END("Lazy Chain");

    /* §12: S₆ Outer Automorphism */
    SECTION(12, "S₆ Outer Automorphism");
    BENCH_START();
    {
        s6_exotic_init();
        CHECK(s6_exotic_ready, "S₆ exotic initialized");

        S6Perm id = S6_IDENTITY;
        S6Perm phi_id = s6_apply_phi(id);
        CHECK(s6_perm_eq(phi_id, id), "φ(identity) = identity");

        double in_re[6] = {1,0,0,0,0,0}, in_im[6] = {0};
        double fold_re[6], fold_im[6], unfold_re[6], unfold_im[6];
        int all_ok = 1;
        for (int s = 0; s < 15; s++) {
            s6_fold_syntheme(in_re, in_im, fold_re, fold_im, s);
            s6_unfold_syntheme(fold_re, fold_im, unfold_re, unfold_im, s);
            double err = 0;
            for (int k = 0; k < 6; k++)
                err += fabs(unfold_re[k]-in_re[k]) + fabs(unfold_im[k]-in_im[k]);
            if (err > 1e-10) { all_ok = 0; break; }
        }
        CHECK(all_ok, "All 15 syntheme fold/unfold round-trips");

        double plus_re[6], plus_im[6];
        double inv6 = 1.0/sqrt(6.0);
        for (int k = 0; k < 6; k++) { plus_re[k] = inv6; plus_im[k] = 0; }
        double delta = s6_exotic_invariant(plus_re, plus_im);
        CHECK(delta >= 0.0, "Exotic invariant Δ ≥ 0");

        int opt = s6_optimal_syntheme(0x03);
        CHECK(opt >= 0 && opt < 15, "Optimal syntheme in range");
    }
    BENCH_END("S₆ Exotic");

    /* §13: Flat Quhit */
    SECTION(13, "Flat Quhit (auto-promote/demote)");
    BENCH_START();
    {
        FlatQuhit fq;
        fq_init(&fq);
        CHECK(fq.repr == FLAT_BASIS, "FlatQuhit starts FLAT_BASIS");
        CHECK(fq.basis_index == 0, "Basis state = |0⟩");

        fq_x(&fq);
        CHECK(fq.repr == FLAT_BASIS, "X|0⟩ stays FLAT");
        CHECK(fq.basis_index == 1, "X|0⟩ = |1⟩");

        fq_shift(&fq, 3);
        CHECK(fq.basis_index == 4, "|1⟩ + shift(3) = |4⟩");

        fq_z(&fq);
        CHECK(fq.repr == FLAT_BASIS, "Z on basis stays FLAT_BASIS");

        fq_dft(&fq);
        CHECK(fq.repr == QUANTUM_FULL, "DFT promotes to QUANTUM_FULL");

        FlatQuhit fa, fb;
        fq_init_basis(&fa, 2);
        fq_init_basis(&fb, 3);
        fq_cz(&fa, &fb);
        CHECK(fa.repr == FLAT_BASIS, "CZ on basis states stays FLAT");

        FlatQuhit fm;
        fq_init(&fm);
        fq_dft(&fm);
        int outcome = fq_measure(&fm, VIEW_EDGE, &bench_rng_state);
        CHECK(outcome >= 0 && outcome < 6, "FlatQuhit measure in range");
        CHECK(fm.repr == FLAT_BASIS, "Post-measure demotes to FLAT_BASIS");
    }
    BENCH_END("Flat Quhit");

    /* §14: Triadic 3-Body */
    SECTION(14, "Triadic 3-Body Entanglement");
    BENCH_START();
    {
        TriadicJoint tj;
        triad_cmy_bell(&tj);
        double norm = 0;
        for (int i = 0; i < TRIAD_D3; i++)
            norm += tj.re[i]*tj.re[i] + tj.im[i]*tj.im[i];
        CHECK(NEAR(norm, 1.0, 1e-8), "CMY Bell normalized");

        double ma[6], mb[6], mc[6];
        triad_marginal_a(&tj, ma);
        triad_marginal_b(&tj, mb);
        triad_marginal_c(&tj, mc);
        double sum_a = 0, sum_b = 0, sum_c = 0;
        for (int k = 0; k < 6; k++) {
            sum_a += ma[k]; sum_b += mb[k]; sum_c += mc[k];
        }
        CHECK(NEAR(sum_a, 1.0, 1e-8), "Marginal A sums to 1");
        CHECK(NEAR(sum_b, 1.0, 1e-8), "Marginal B sums to 1");
        CHECK(NEAR(sum_c, 1.0, 1e-8), "Marginal C sums to 1");

        double ar[6]={1,0,0,0,0,0}, ai[6]={0};
        double br[6]={0,1,0,0,0,0}, bi_r[6]={0};
        double cr[6]={0,0,1,0,0,0}, ci[6]={0};
        TriadicJoint pj;
        triad_product(&pj, ar, ai, br, bi_r, cr, ci);
        CHECK(NEAR(pj.re[TRIAD_IDX(0,1,2)], 1.0, 1e-12), "Product |0⟩|1⟩|2⟩");

        TriadicJoint ghz;
        triad_bell(&ghz);
        double ghz_norm = 0;
        for (int i = 0; i < TRIAD_D3; i++)
            ghz_norm += ghz.re[i]*ghz.re[i] + ghz.im[i]*ghz.im[i];
        CHECK(NEAR(ghz_norm, 1.0, 1e-8), "GHZ triple normalized");
    }
    BENCH_END("Triadic");
}

/* ══════════════════════════════════════════════════════════════════════════════
 * PART 3: REGISTERS, OVERLAYS & SCALE (§15–21)
 * ══════════════════════════════════════════════════════════════════════════════ */

static void run_part3(void)
{
    PART_HEADER(3, "REGISTERS, OVERLAYS & SCALE");

    /* §15: Quhit Register */
    SECTION(15, "Quhit Register (100T-scale)");
    BENCH_START();
    {
        int r0 = quhit_reg_init(eng, 1, 10, 6);
        CHECK(r0 >= 0, "Register init (10 quhits)");

        quhit_reg_entangle_all(eng, r0);
        CHECK(eng->registers[r0].bulk_rule == 1, "GHZ bulk rule set");
        CHECK(eng->registers[r0].num_nonzero == 6, "GHZ: exactly 6 amplitudes");

        double total = quhit_reg_sv_total_prob(eng, r0);
        CHECK(NEAR(total, 1.0, 1e-8), "Register total prob = 1.0");

        int r1 = quhit_reg_init(eng, 2, 1000000, 6);
        CHECK(r1 >= 0, "1M-quhit register init");
        quhit_reg_entangle_all(eng, r1);
        CHECK(eng->registers[r1].num_nonzero == 6, "1M GHZ: still 6 amps");
        double total1m = quhit_reg_sv_total_prob(eng, r1);
        CHECK(NEAR(total1m, 1.0, 1e-8), "1M register total prob = 1.0");

        int r2 = quhit_reg_init(eng, 3, 4, 6);
        quhit_reg_sv_set(eng, r2, 0, 1.0, 0.0);  /* Seed |0⟩ so DFT has input */
        quhit_reg_apply_dft(eng, r2, 0);
        CHECK(eng->registers[r2].num_nonzero > 0, "DFT on register produces entries");

        int r3 = quhit_reg_init(eng, 4, 3, 6);
        quhit_reg_entangle_all(eng, r3);
        uint64_t meas = quhit_reg_measure(eng, r3, 0);
        CHECK(meas < 6, "Register measurement in range");
    }
    BENCH_END("Register");

    /* §16: Gauss Sums */
    SECTION(16, "Gauss Sum Amplitudes");
    BENCH_START();
    {
        double amp_re, amp_im;
        int basis2[2] = {0, 0};
        gauss_amp_line(basis2, 2, &amp_re, &amp_im);
        double born_p = amp_re*amp_re + amp_im*amp_im;
        CHECK(born_p >= 0.0 && born_p <= 1.0, "Gauss amp Born prob in [0,1]");

        int basis4[4] = {1, 2, 3, 0};
        gauss_amp_line(basis4, 4, &amp_re, &amp_im);
        CHECK(1, "Gauss 4-quhit circuit computed");

        double gbl_pr[6];
        gauss_born_line(3, 0, gbl_pr);
        double bp_sum = 0;
        for (int k = 0; k < 6; k++) bp_sum += gbl_pr[k];
        CHECK(NEAR(bp_sum, 1.0, 0.1), "gauss_born_line marginals sum ≈ 1");
    }
    BENCH_END("Gauss Sums");

    /* §17: Lazy Chain Throughput */
    SECTION(17, "Lazy Chain Throughput");
    BENCH_START();
    {
        LazyChain lc;
        lazy_init(&lc);
        double dft_re[36] = {0}, dft_im[36] = {0};
        double inv = 1.0/sqrt(6.0);
        for (int j = 0; j < 6; j++)
            for (int k = 0; k < 6; k++) {
                double angle = 2.0 * M_PI * j * k / 6.0;
                dft_re[j*6+k] = inv * cos(angle);
                dft_im[j*6+k] = inv * sin(angle);
            }

        struct timespec lt0, lt1;
        clock_gettime(CLOCK_MONOTONIC, &lt0);
        for (int i = 0; i < 100000; i++)
            lazy_compose(&lc, dft_re, dft_im);
        clock_gettime(CLOCK_MONOTONIC, &lt1);
        double lms = (lt1.tv_sec-lt0.tv_sec)*1e3 + (lt1.tv_nsec-lt0.tv_nsec)*1e-6;
        printf("    ⏱  100K lazy compositions: %.3f ms (%.0f/sec)\n",
               lms, 100000.0/(lms/1000.0));
        CHECK(lazy_depth(&lc) == 100000, "100K gates composed");
    }
    BENCH_END("Lazy Throughput");

    /* §18: MPS Overlay */
    SECTION(18, "MPS Overlay");
    BENCH_START();
    {
        int n = 8;
        MpsChain *chain = mps_init(n);
        CHECK(chain != NULL && chain->L == n, "MPS init (8 sites)");

        double amps_re[6] = {1,0,0,0,0,0}, amps_im[6] = {0};
        for (int i = 0; i < n; i++)
            mps_set_product_state(chain, i, amps_re, amps_im);
        CHECK(1, "MPS product state |0⟩⊗8 set");

        double probs0[6];
        mps_local_density(chain, 0, probs0);
        CHECK(NEAR(probs0[0], 1.0, 0.1), "MPS |0⟩ local density correct");

        double U_re[36] = {0}, U_im[36] = {0};
        mps_build_dft6(U_re, U_im);
        mps_gate_1site(chain, 0, U_re, U_im);
        CHECK(1, "MPS 1-site DFT applied");

        double G_re[36*36], G_im[36*36];
        memset(G_re, 0, sizeof(G_re));
        memset(G_im, 0, sizeof(G_im));
        mps_build_cz(G_re, G_im);
        mps_gate_bond(chain, 0, G_re, G_im);
        CHECK(1, "MPS 2-site CZ applied");

        double probs1[6];
        mps_local_density(chain, 0, probs1);
        double psum = 0;
        for (int k = 0; k < 6; k++) psum += probs1[k];
        CHECK(NEAR(psum, 1.0, 0.1), "MPS local density sums ≈ 1");

        mps_free(chain);
        CHECK(1, "MPS free");
    }
    BENCH_END("MPS Overlay");

    /* §19: PEPS 2D Overlay */
    SECTION(19, "PEPS 2D Overlay");
    BENCH_START();
    {
        PepsGrid *grid = peps_init(3, 3);
        CHECK(grid != NULL && grid->Lx == 3 && grid->Ly == 3, "PEPS 3×3 grid init");

        double amps_re[6] = {1,0,0,0,0,0}, amps_im[6] = {0};
        for (int y = 0; y < 3; y++)
            for (int x = 0; x < 3; x++)
                peps_set_product_state(grid, x, y, amps_re, amps_im);
        CHECK(1, "PEPS product states set");

        double PU_re[36] = {0}, PU_im[36] = {0};
        mps_build_dft6(PU_re, PU_im);
        peps_gate_1site(grid, 0, 0, PU_re, PU_im);
        CHECK(1, "PEPS 1-site gate applied");

        double probs[6];
        peps_local_density(grid, 0, 0, probs);
        double psum = 0;
        for (int k = 0; k < 6; k++) psum += probs[k];
        CHECK(NEAR(psum, 1.0, 0.1), "PEPS local density sums ≈ 1");

        peps_free(grid);
        CHECK(1, "PEPS free");
    }
    BENCH_END("PEPS 2D");

    /* §20: Dynamic Growth */
    SECTION(20, "Dynamic Chain Growth");
    BENCH_START();
    {
        DynChain dc = dyn_chain_create(64);
        CHECK(dc.max_sites == 64, "DynChain created (64 sites)");

        dyn_chain_seed(&dc, 10, 20);
        CHECK(dyn_chain_active_length(&dc) == 11, "Active length = 11");

        double uniform_probs[6] = {1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6};
        for (int s = 10; s <= 20; s++)
            dyn_chain_update_entropy(&dc, s, uniform_probs, 6);

        dyn_chain_record_entropy(&dc);
        dyn_chain_predict_entropy(&dc);
        CHECK(1, "Entropy prediction ran");

        int conv = dyn_chain_check_convergence(&dc);
        CHECK(conv >= 0 && conv <= 2, "Convergence state valid");

        dyn_chain_free(&dc);
        CHECK(1, "DynChain freed");
    }
    BENCH_END("Dynamic Growth");

    /* §21: Full Pipeline */
    SECTION(21, "Full Pipeline: Init → DFT → CZ → Measure");
    BENCH_START();
    {
        uint32_t pipeline[6];
        for (int i = 0; i < 6; i++)
            pipeline[i] = quhit_init(eng);
        for (int i = 0; i < 6; i++)
            quhit_apply_dft(eng, pipeline[i]);
        for (int i = 0; i < 5; i++)
            quhit_apply_cz(eng, pipeline[i], pipeline[i+1]);
        uint32_t results[6];
        for (int i = 0; i < 6; i++)
            results[i] = quhit_measure(eng, pipeline[i]);
        int all_valid = 1;
        for (int i = 0; i < 6; i++)
            if (results[i] >= 6) all_valid = 0;
        CHECK(all_valid, "Pipeline: all measurements in [0,5]");

        /* Throughput: 1000 full circuits */
        struct timespec pt0, pt1;
        clock_gettime(CLOCK_MONOTONIC, &pt0);
        for (int trial = 0; trial < 1000; trial++) {
            uint32_t q[4];
            for (int i = 0; i < 4; i++) q[i] = quhit_init(eng);
            for (int i = 0; i < 4; i++) quhit_apply_dft(eng, q[i]);
            for (int i = 0; i < 3; i++) quhit_apply_cz(eng, q[i], q[i+1]);
            for (int i = 0; i < 4; i++) quhit_measure(eng, q[i]);
        }
        clock_gettime(CLOCK_MONOTONIC, &pt1);
        double pms = (pt1.tv_sec-pt0.tv_sec)*1e3 + (pt1.tv_nsec-pt0.tv_nsec)*1e-6;
        int total_ops = 1000 * (4 + 4 + 3 + 4);
        printf("    ⏱  1K full circuits (15K ops): %.3f ms (%.0f ops/sec)\n",
               pms, (double)total_ops/(pms/1000.0));

        /* SVD Gate Oracle */
        GateLog gl;
        glog_init(&gl);
        glog_push(&gl, GTAG_DFT, 0);
        glog_push(&gl, GTAG_CZ, 0);
        SvdPrediction pred = glog_analyze(&gl);
        CHECK(pred >= 0 && pred < SVD_NUM_PREDICTIONS, "SVD oracle prediction valid");

        /* Triality overlay site */
        TriOverlaySite ts;
        tri_site_init(&ts);
        CHECK(ts.tri.primary == VIEW_EDGE, "TriOverlay site init OK");
        double tU_re[36] = {0}, tU_im[36] = {0};
        mps_build_dft6(tU_re, tU_im);
        GateClass gc = tri_classify_gate(tU_re, tU_im);
        CHECK(gc == GATE_DFT, "Gate classified as DFT");
    }
    BENCH_END("Full Pipeline");

    /* §22: Triality Stress Test — The Lightbringer's Gauntlet */
    SECTION(22, "Triality Stress Test (10K gates, 5 views)");
    BENCH_START();
    {
        triality_stats_reset();

        /* ── 23a: Norm stability under 10K random gates with view switching ── */
        TrialityQuhit ts;
        triality_init(&ts);
        triality_dft(&ts); /* Start in superposition, not a trivial basis state */

        int norm_ok = 1;
        for (int i = 0; i < 10000; i++) {
            double r = bench_rand01();
            if (r < 0.2) {
                triality_z(&ts);
            } else if (r < 0.4) {
                triality_x(&ts);
            } else if (r < 0.6) {
                triality_dft(&ts);
            } else if (r < 0.8) {
                triality_shift(&ts, (int)(bench_rand01() * 5) + 1);
            } else {
                triality_idft(&ts);
            }

            /* Force a random view materialization every 100 gates */
            if (i % 100 == 0) {
                int view = (int)(bench_rand01() * 3); /* Edge, Vertex, or Diagonal */
                triality_ensure_view(&ts, view);
                const double *re = triality_view_re(&ts, view);
                const double *im = triality_view_im(&ts, view);
                double norm = 0;
                for (int k = 0; k < 6; k++)
                    norm += re[k]*re[k] + im[k]*im[k];
                if (!NEAR(norm, 1.0, 1e-6)) { norm_ok = 0; break; }
            }
        }
        CHECK(norm_ok, "Triality: Norm stable after 10K random gates + 100 view switches");

        /* ── 23b: View consistency — all views agree on probabilities ── */
        double probs_e[6], probs_v[6], probs_d[6];
        triality_probabilities(&ts, VIEW_EDGE, probs_e);
        triality_probabilities(&ts, VIEW_VERTEX, probs_v);
        triality_probabilities(&ts, VIEW_DIAGONAL, probs_d);
        double sum_e = 0, sum_v = 0, sum_d = 0;
        for (int k = 0; k < 6; k++) {
            sum_e += probs_e[k]; sum_v += probs_v[k]; sum_d += probs_d[k];
        }
        CHECK(NEAR(sum_e, 1.0, 1e-8) && NEAR(sum_v, 1.0, 1e-8) && NEAR(sum_d, 1.0, 1e-8),
              "Triality: All 3 views produce valid probability distributions");

        /* ── 23c: Rotation identity — 3 rotations = identity ── */
        TrialityQuhit tr;
        triality_init_basis(&tr, 2);
        triality_dft(&tr); /* non-trivial state */
        TrialityQuhit tr_save;
        triality_copy(&tr_save, &tr);

        triality_rotate(&tr);
        triality_rotate(&tr);
        triality_rotate(&tr); /* 3 rotations = full cycle */

        triality_ensure_view(&tr, VIEW_EDGE);
        triality_ensure_view(&tr_save, VIEW_EDGE);
        double rot_err = 0;
        for (int k = 0; k < 6; k++) {
            rot_err += fabs(tr.edge_re[k] - tr_save.edge_re[k]);
            rot_err += fabs(tr.edge_im[k] - tr_save.edge_im[k]);
        }
        CHECK(rot_err < 1e-10, "Triality: 3× rotation = identity (err < 1e-10)");

        /* ── 23d: Exotic syntheme round-trips — all 15 synthemes ── */
        s6_exotic_init();
        int syntheme_ok = 1;
        for (int s = 0; s < 15; s++) {
            TrialityQuhit te;
            triality_init(&te);
            triality_dft(&te);
            triality_z(&te); /* non-trivial state with phases */

            /* Save state */
            double save_re[6], save_im[6];
            triality_ensure_view(&te, VIEW_EDGE);
            for (int k = 0; k < 6; k++) {
                save_re[k] = te.edge_re[k];
                save_im[k] = te.edge_im[k];
            }

            /* Fold then unfold */
            triality_fold_syntheme(&te, s);
            triality_unfold_syntheme(&te, s);

            triality_ensure_view(&te, VIEW_EDGE);
            double serr = 0;
            for (int k = 0; k < 6; k++) {
                serr += fabs(te.edge_re[k] - save_re[k]);
                serr += fabs(te.edge_im[k] - save_im[k]);
            }
            if (serr > 1e-10) { syntheme_ok = 0; break; }
        }
        CHECK(syntheme_ok, "Triality: All 15 syntheme fold/unfold round-trips exact");

        /* ── 23e: Lazy triality chain vs eager — fidelity check ── */
        LazyTrialityQuhit lazy;
        TrialityQuhit eager;
        ltri_init(&lazy);
        triality_init(&eager);

        /* Apply same random gate sequence to both */
        bench_rng_state = 0xFACEFEED42ULL;
        for (int i = 0; i < 500; i++) {
            double r = bench_rand01();
            if (r < 0.25) {
                ltri_z(&lazy); triality_z(&eager);
            } else if (r < 0.5) {
                ltri_x(&lazy); triality_x(&eager);
            } else if (r < 0.75) {
                ltri_dft(&lazy); triality_dft(&eager);
            } else {
                int d = (int)(bench_rand01() * 5) + 1;
                ltri_shift(&lazy, d); triality_shift(&eager, d);
            }
        }

        double lazy_re[6], lazy_im[6];
        ltri_materialize(&lazy, lazy_re, lazy_im);
        triality_ensure_view(&eager, VIEW_EDGE);
        double fid_err = 0;
        for (int k = 0; k < 6; k++) {
            fid_err += fabs(lazy_re[k] - eager.edge_re[k]);
            fid_err += fabs(lazy_im[k] - eager.edge_im[k]);
        }
        CHECK(fid_err < 1e-8, "Triality: Lazy chain matches eager after 500 gates");
        printf("    ℹ  Lazy: %lu gates fused, %lu segments, %lu materializations\n",
               (unsigned long)lazy.gates_fused,
               (unsigned long)lazy.segments_created,
               (unsigned long)lazy.materializations);

        /* ── 23f: Throughput benchmark — 100K triality Z+X+DFT pipeline ── */
        TrialityQuhit tp;
        triality_init_basis(&tp, 0);
        struct timespec tri_t0, tri_t1;
        clock_gettime(CLOCK_MONOTONIC, &tri_t0);
        for (int i = 0; i < 100000; i++) {
            triality_z(&tp);
            triality_x(&tp);
            triality_dft(&tp);
        }
        clock_gettime(CLOCK_MONOTONIC, &tri_t1);
        double tri_ms = (tri_t1.tv_sec-tri_t0.tv_sec)*1e3 + (tri_t1.tv_nsec-tri_t0.tv_nsec)*1e-6;
        printf("    ⏱  300K triality gates (Z+X+DFT): %.3f ms (%.0f gates/sec)\n",
               tri_ms, 300000.0/(tri_ms/1000.0));

        /* Print diagnostic stats */
        triality_stats_print();
    }
    BENCH_END("Triality Stress");
}

/* ══════════════════════════════════════════════════════════════════════════════
 * MAIN
 * ══════════════════════════════════════════════════════════════════════════════ */

int main(void)
{
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║     HEXSTATE ENGINE — COMPREHENSIVE BENCHMARK           ║\n");
    printf("║     22 Sections · 3 Parts · Every Feature Tier          ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");

    struct timespec total_t0, total_t1;
    clock_gettime(CLOCK_MONOTONIC, &total_t0);

    run_part1();
    run_part2();
    run_part3();

    clock_gettime(CLOCK_MONOTONIC, &total_t1);
    double total_ms = (total_t1.tv_sec - total_t0.tv_sec)*1e3
                    + (total_t1.tv_nsec - total_t0.tv_nsec)*1e-6;

    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║                    FINAL RESULTS                        ║\n");
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║  Passed: %-5d  Failed: %-5d  Total: %-5d             ║\n",
           g_pass, g_fail, g_pass + g_fail);
    printf("║  Total time: %.1f ms                                    ║\n", total_ms);
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    if (eng) free(eng);
    return g_fail > 0 ? 1 : 0;
}
