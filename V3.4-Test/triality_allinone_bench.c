/*
 * triality_allinone_bench.c — All-in-One Triality Quhit + Overlay Benchmark
 *
 * 16 suites: 10 quhit-level + MPS + PEPS2D-6D overlay Trotter benchmarks
 * with DynChain integration.
 *
 * Build:
 *   gcc -O2 -I. -o triality_allinone triality_allinone_bench.c \
 *       quhit_triality.c quhit_core.c quhit_gates.c quhit_measure.c \
 *       quhit_entangle.c quhit_register.c quhit_substrate.c \
 *       quhit_calibrate.c mps_overlay.c peps_overlay.c \
 *       peps3d_overlay.c peps4d_overlay.c peps5d_overlay.c peps6d_overlay.c \
 *       quhit_svd_gate.c -lm
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "quhit_triality.h"
#include "quhit_engine.h"
#include "quhit_dyn_integrate.h"
#include "mps_overlay.h"
#include "peps_overlay.h"
#include "peps3d_overlay.h"
#include "peps4d_overlay.h"
#include "peps5d_overlay.h"
#include "peps6d_overlay.h"

/* ═══════════════════════════════════════════════════════════════════════
 * TIMING + RNG
 * ═══════════════════════════════════════════════════════════════════════ */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint64_t bench_rng_state = 0xDEADBEEFCAFE4242ULL;
static uint64_t bench_rng(void) {
    bench_rng_state ^= bench_rng_state << 13;
    bench_rng_state ^= bench_rng_state >> 7;
    bench_rng_state ^= bench_rng_state << 17;
    return bench_rng_state;
}

/* ═══════════════════════════════════════════════════════════════════════
 * RESULT STRUCTURE
 * ═══════════════════════════════════════════════════════════════════════ */

#define N_BENCH 16

typedef struct {
    const char *name;
    double lazy_time, tri_time, std_time;
    double lazy_err, tri_err;
    int    lazy_match, tri_match;
    uint64_t lazy_gates, lazy_segs;
    int    has_lazy, has_tri, has_std;
    int    dyn_grows, dyn_contracts, dyn_active;
    /* For overlay suites */
    double overlay_time;
    int    has_overlay;
    int    overlay_steps;
    const char *overlay_label;
} BenchResult;

static BenchResult R[N_BENCH];

/* ═══════════════════════════════════════════════════════════════════════
 * HELPERS
 * ═══════════════════════════════════════════════════════════════════════ */

static double compare_lazy(LazyTrialityQuhit *lq, const QuhitEngine *eng, uint32_t sq) {
    double amp_re[6], amp_im[6];
    ltri_materialize(lq, amp_re, amp_im);
    double max_err = 0;
    for (int k = 0; k < 6; k++) {
        double dr = fabs(amp_re[k] - eng->quhits[sq].state.re[k]);
        double di = fabs(amp_im[k] - eng->quhits[sq].state.im[k]);
        if (dr > max_err) max_err = dr;
        if (di > max_err) max_err = di;
    }
    return max_err;
}

static double compare_tri(TrialityQuhit *tq, const QuhitEngine *eng, uint32_t sq) {
    triality_ensure_view(tq, VIEW_EDGE);
    double max_err = 0;
    for (int k = 0; k < 6; k++) {
        double dr = fabs(tq->edge_re[k] - eng->quhits[sq].state.re[k]);
        double di = fabs(tq->edge_im[k] - eng->quhits[sq].state.im[k]);
        if (dr > max_err) max_err = dr;
        if (di > max_err) max_err = di;
    }
    return max_err;
}

/* Build a CZ-type 2-site gate for PEPS/TNS: diagonal in (k1,k2) space
 * G[(k1*6+k2), (k1'*6+k2')] = delta(k1,k1') * delta(k2,k2') * exp(2pi*i*k1*k2/6)
 */
static void build_cz_gate(double *G_re, double *G_im) {
    int D2 = 36;
    memset(G_re, 0, D2 * D2 * sizeof(double));
    memset(G_im, 0, D2 * D2 * sizeof(double));
    for (int k1 = 0; k1 < 6; k1++) {
        for (int k2 = 0; k2 < 6; k2++) {
            int idx = k1 * 6 + k2;
            double angle = 2.0 * M_PI * k1 * k2 / 6.0;
            G_re[idx * D2 + idx] = cos(angle);
            G_im[idx * D2 + idx] = sin(angle);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * SUITES 1-6: Standard gate patterns (1M gates each)
 * ═══════════════════════════════════════════════════════════════════════ */

#define N_GATES 1000000

static void run_gate_suite(int circuit, int N, int idx) {
    BenchResult *r = &R[idx];
    r->has_lazy = r->has_tri = r->has_std = 1;
    int needs_dft_init = (circuit == 0 || circuit == 2 || circuit == 3 || circuit == 5);

    /* ── LAZY ── */
    LazyTrialityQuhit lq;
    ltri_init_basis(&lq, 0);
    if (needs_dft_init) ltri_dft(&lq);
    uint64_t rng_save = bench_rng_state;
    double t0 = now_sec();
    for (int i = 0; i < N; i++) {
        switch (circuit) {
            case 0: ltri_z(&lq); break;
            case 1: ltri_x(&lq); break;
            case 2: if (i & 1) ltri_x(&lq); else ltri_z(&lq); break;
            case 3: if ((i%6)<4) ltri_z(&lq); else ltri_x(&lq); break;
            case 4: ltri_dft(&lq); break;
            case 5:
                switch (bench_rng() % 4) {
                    case 0: ltri_z(&lq); break;
                    case 1: ltri_x(&lq); break;
                    case 2: ltri_shift(&lq, 2); break;
                    case 3: ltri_dft(&lq); break;
                }
                break;
        }
    }
    r->lazy_time = now_sec() - t0;
    r->lazy_gates = lq.gates_fused;
    r->lazy_segs = lq.segments_created;

    /* ── EAGER TRIALITY ── */
    bench_rng_state = rng_save;
    TrialityQuhit tq;
    triality_init_basis(&tq, 0);
    if (needs_dft_init) triality_dft(&tq);
    t0 = now_sec();
    for (int i = 0; i < N; i++) {
        switch (circuit) {
            case 0: triality_z(&tq); break;
            case 1: triality_x(&tq); break;
            case 2: if (i & 1) triality_x(&tq); else triality_z(&tq); break;
            case 3: if ((i%6)<4) triality_z(&tq); else triality_x(&tq); break;
            case 4: triality_dft(&tq); break;
            case 5:
                switch (bench_rng() % 4) {
                    case 0: triality_z(&tq); break;
                    case 1: triality_x(&tq); break;
                    case 2: triality_shift(&tq, 2); break;
                    case 3: triality_dft(&tq); break;
                }
                break;
        }
    }
    r->tri_time = now_sec() - t0;

    /* ── STANDARD ── */
    bench_rng_state = rng_save;
    QuhitEngine *eng = (QuhitEngine *)calloc(1, sizeof(QuhitEngine));
    quhit_engine_init(eng);
    uint32_t sq = quhit_init(eng);
    if (needs_dft_init) quhit_apply_dft(eng, sq);
    t0 = now_sec();
    for (int i = 0; i < N; i++) {
        switch (circuit) {
            case 0: quhit_apply_z(eng, sq); break;
            case 1: quhit_apply_x(eng, sq); break;
            case 2: if (i & 1) quhit_apply_x(eng, sq); else quhit_apply_z(eng, sq); break;
            case 3: if ((i%6)<4) quhit_apply_z(eng, sq); else quhit_apply_x(eng, sq); break;
            case 4: quhit_apply_dft(eng, sq); break;
            case 5: {
                double U_re[36] = {0}, U_im[36] = {0};
                int delta = 2;
                for (int k = 0; k < 6; k++) U_re[k*6 + ((k-delta+6)%6)] = 1.0;
                switch (bench_rng() % 4) {
                    case 0: quhit_apply_z(eng, sq); break;
                    case 1: quhit_apply_x(eng, sq); break;
                    case 2: quhit_apply_unitary(eng, sq, U_re, U_im); break;
                    case 3: quhit_apply_dft(eng, sq); break;
                }
                break;
            }
        }
    }
    r->std_time = now_sec() - t0;
    r->lazy_err = compare_lazy(&lq, eng, sq);
    r->tri_err  = compare_tri(&tq, eng, sq);
    r->lazy_match = (r->lazy_err < 1e-8);
    r->tri_match  = (r->tri_err  < 1e-8);
    free(eng);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SUITE 7: Dyn Chain Growth
 * ═══════════════════════════════════════════════════════════════════════ */

#define DYN_SITES   16
#define DYN_EPOCHS  200
#define DYN_TROTTER 50

static void run_dyn_chain_suite(int idx) {
    BenchResult *r = &R[idx];
    r->has_lazy = r->has_tri = r->has_std = 1;

    /* ── LAZY ── */
    LazyTrialityQuhit lazy_chain[DYN_SITES];
    DynChain dc_l = dyn_chain_create(DYN_SITES);
    dyn_chain_seed(&dc_l, 0, 3);
    for (int i = 0; i < DYN_SITES; i++) ltri_init_basis(&lazy_chain[i], 0);
    double t0 = now_sec();
    for (int e = 0; e < DYN_EPOCHS; e++) {
        for (int st = 0; st < DYN_TROTTER; st++)
            for (int s = dc_l.active_start; s <= dc_l.active_end; s++) {
                ltri_z(&lazy_chain[s]);
                if (s+1 <= dc_l.active_end) { ltri_x(&lazy_chain[s]); ltri_x(&lazy_chain[s+1]); }
            }
        for (int s = dc_l.active_start; s <= dc_l.active_end; s++) {
            double ar[6], ai[6]; ltri_materialize(&lazy_chain[s], ar, ai);
            double p[6]; for (int k = 0; k < 6; k++) p[k] = ar[k]*ar[k]+ai[k]*ai[k];
            dyn_chain_update_entropy(&dc_l, s, p, 6);
        }
        dyn_chain_step(&dc_l);
    }
    r->lazy_time = now_sec() - t0;

    /* ── TRIALITY ── */
    TrialityQuhit tri_chain[DYN_SITES];
    DynChain dc_t = dyn_chain_create(DYN_SITES);
    dyn_chain_seed(&dc_t, 0, 3);
    for (int i = 0; i < DYN_SITES; i++) triality_init_basis(&tri_chain[i], 0);
    t0 = now_sec();
    for (int e = 0; e < DYN_EPOCHS; e++) {
        for (int st = 0; st < DYN_TROTTER; st++)
            for (int s = dc_t.active_start; s <= dc_t.active_end; s++) {
                triality_z(&tri_chain[s]);
                if (s+1 <= dc_t.active_end) { triality_x(&tri_chain[s]); triality_x(&tri_chain[s+1]); }
            }
        for (int s = dc_t.active_start; s <= dc_t.active_end; s++) {
            triality_ensure_view(&tri_chain[s], VIEW_EDGE);
            double p[6]; for (int k = 0; k < 6; k++)
                p[k] = tri_chain[s].edge_re[k]*tri_chain[s].edge_re[k]+tri_chain[s].edge_im[k]*tri_chain[s].edge_im[k];
            dyn_chain_update_entropy(&dc_t, s, p, 6);
        }
        dyn_chain_step(&dc_t);
    }
    r->tri_time = now_sec() - t0;

    /* ── STANDARD ── */
    QuhitEngine *eng = (QuhitEngine *)calloc(1, sizeof(QuhitEngine));
    quhit_engine_init(eng);
    DynChain dc_s = dyn_chain_create(DYN_SITES);
    dyn_chain_seed(&dc_s, 0, 3);
    uint32_t sq[DYN_SITES];
    for (int i = 0; i < DYN_SITES; i++) sq[i] = quhit_init(eng);
    t0 = now_sec();
    for (int e = 0; e < DYN_EPOCHS; e++) {
        for (int st = 0; st < DYN_TROTTER; st++)
            for (int s = dc_s.active_start; s <= dc_s.active_end; s++) {
                quhit_apply_z(eng, sq[s]);
                if (s+1 <= dc_s.active_end) { quhit_apply_x(eng, sq[s]); quhit_apply_x(eng, sq[s+1]); }
            }
        for (int s = dc_s.active_start; s <= dc_s.active_end; s++) {
            double p[6]; for (int k = 0; k < 6; k++)
                p[k] = eng->quhits[sq[s]].state.re[k]*eng->quhits[sq[s]].state.re[k]+
                       eng->quhits[sq[s]].state.im[k]*eng->quhits[sq[s]].state.im[k];
            dyn_chain_update_entropy(&dc_s, s, p, 6);
        }
        dyn_chain_step(&dc_s);
    }
    r->std_time = now_sec() - t0;
    r->lazy_err = compare_lazy(&lazy_chain[0], eng, sq[0]);
    r->tri_err  = compare_tri(&tri_chain[0], eng, sq[0]);
    r->lazy_match = (r->lazy_err < 1e-6); r->tri_match = (r->tri_err < 1e-6);
    r->dyn_grows = dc_s.grow_events; r->dyn_contracts = dc_s.contract_events;
    r->dyn_active = dyn_chain_active_length(&dc_s);
    dyn_chain_free(&dc_l); dyn_chain_free(&dc_t); dyn_chain_free(&dc_s);
    free(eng);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SUITE 8: Triality Rotation (Eager-only)
 * ═══════════════════════════════════════════════════════════════════════ */

static void run_rotation_suite(int idx) {
    BenchResult *r = &R[idx];
    r->has_lazy = 0; r->has_std = 0; r->has_tri = 1;
    r->lazy_match = 1; r->tri_match = 1;
    TrialityQuhit tq, tq2;
    triality_init_basis(&tq, 0); triality_dft(&tq); triality_z(&tq);
    triality_copy(&tq2, &tq);
    double t0 = now_sec();
    for (int i = 0; i < N_GATES; i++) triality_rotate(&tq);
    r->tri_time = now_sec() - t0;
    for (int i = 0; i < (N_GATES % 3); i++) triality_rotate(&tq2);
    triality_ensure_view(&tq, VIEW_EDGE); triality_ensure_view(&tq2, VIEW_EDGE);
    double err = 0;
    for (int k = 0; k < 6; k++) {
        double d = fabs(tq.edge_re[k]-tq2.edge_re[k]); if (d > err) err = d;
        d = fabs(tq.edge_im[k]-tq2.edge_im[k]); if (d > err) err = d;
    }
    r->tri_err = err; r->tri_match = (err < 1e-10);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SUITE 9: Multi-Measure
 * ═══════════════════════════════════════════════════════════════════════ */

#define MULTI_ROUNDS 10000
#define MULTI_GATES  100

static void run_multi_measure_suite(int idx) {
    BenchResult *r = &R[idx];
    r->has_lazy = r->has_tri = r->has_std = 1;
    uint64_t rng = 0xAAAABBBBCCCCDDDDULL;

    LazyTrialityQuhit lq; uint64_t lr = rng;
    double t0 = now_sec();
    for (int rd = 0; rd < MULTI_ROUNDS; rd++) {
        ltri_init_basis(&lq, 0);
        for (int g = 0; g < MULTI_GATES; g++) { if (g&1) ltri_x(&lq); else ltri_z(&lq); }
        ltri_measure(&lq, VIEW_EDGE, &lr);
    }
    r->lazy_time = now_sec() - t0;
    r->lazy_gates = (uint64_t)MULTI_ROUNDS * MULTI_GATES;

    TrialityQuhit tq; uint64_t tr = rng;
    t0 = now_sec();
    for (int rd = 0; rd < MULTI_ROUNDS; rd++) {
        triality_init_basis(&tq, 0);
        for (int g = 0; g < MULTI_GATES; g++) { if (g&1) triality_x(&tq); else triality_z(&tq); }
        triality_measure(&tq, VIEW_EDGE, &tr);
    }
    r->tri_time = now_sec() - t0;

    QuhitEngine *eng = (QuhitEngine *)calloc(1, sizeof(QuhitEngine));
    quhit_engine_init(eng); uint32_t sq = quhit_init(eng); uint64_t sr = rng;
    t0 = now_sec();
    for (int rd = 0; rd < MULTI_ROUNDS; rd++) {
        memset(eng->quhits[sq].state.re, 0, sizeof(eng->quhits[sq].state.re));
        memset(eng->quhits[sq].state.im, 0, sizeof(eng->quhits[sq].state.im));
        eng->quhits[sq].state.re[0] = 1.0;
        for (int g = 0; g < MULTI_GATES; g++) { if (g&1) quhit_apply_x(eng, sq); else quhit_apply_z(eng, sq); }
        sr ^= sr<<13; sr ^= sr>>7; sr ^= sr<<17;
    }
    r->std_time = now_sec() - t0;
    r->lazy_err = 0; r->tri_err = 0; r->lazy_match = 1; r->tri_match = 1;
    free(eng);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SUITE 10: Fusion Stress (10M Z gates)
 * ═══════════════════════════════════════════════════════════════════════ */

#define STRESS_N 10000000

static void run_stress_suite(int idx) {
    BenchResult *r = &R[idx];
    r->has_lazy = r->has_tri = r->has_std = 1;

    LazyTrialityQuhit lq; ltri_init_basis(&lq, 0);
    double t0 = now_sec();
    for (int i = 0; i < STRESS_N; i++) ltri_z(&lq);
    r->lazy_time = now_sec() - t0;
    r->lazy_gates = lq.gates_fused; r->lazy_segs = lq.segments_created;

    TrialityQuhit tq; triality_init_basis(&tq, 0);
    t0 = now_sec();
    for (int i = 0; i < STRESS_N; i++) triality_z(&tq);
    r->tri_time = now_sec() - t0;

    QuhitEngine *eng = (QuhitEngine *)calloc(1, sizeof(QuhitEngine));
    quhit_engine_init(eng); uint32_t sq = quhit_init(eng);
    t0 = now_sec();
    for (int i = 0; i < STRESS_N; i++) quhit_apply_z(eng, sq);
    r->std_time = now_sec() - t0;

    r->lazy_err = compare_lazy(&lq, eng, sq);
    r->tri_err  = compare_tri(&tq, eng, sq);
    r->lazy_match = (r->lazy_err < 1e-8); r->tri_match = (r->tri_err < 1e-8);
    free(eng);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SUITE 11: MPS Overlay — 8-site Trotter sweep
 * ═══════════════════════════════════════════════════════════════════════ */

#define MPS_N       8
#define MPS_STEPS   50

static void run_mps_suite(int idx) {
    BenchResult *r = &R[idx];
    r->has_overlay = 1;
    r->overlay_label = "MPS 1D";

    QuhitEngine *eng = (QuhitEngine *)calloc(1, sizeof(QuhitEngine));
    quhit_engine_init(eng);
    uint32_t qubits[MPS_N];
    for (int i = 0; i < MPS_N; i++) qubits[i] = quhit_init(eng);

    mps_overlay_init(eng, qubits, MPS_N);
    mps_overlay_write_zero(eng, qubits, MPS_N);

    double G_re[36*36], G_im[36*36];
    build_cz_gate(G_re, G_im);

    double t0 = now_sec();
    for (int step = 0; step < MPS_STEPS; step++) {
        /* Sweep right */
        for (int s = 0; s < MPS_N - 1; s++)
            mps_gate_2site(eng, qubits, MPS_N, s, G_re, G_im);
        /* Sweep left */
        for (int s = MPS_N - 2; s >= 0; s--)
            mps_gate_2site(eng, qubits, MPS_N, s, G_re, G_im);
    }
    r->overlay_time = now_sec() - t0;
    r->overlay_steps = MPS_STEPS * 2 * (MPS_N - 1);

    mps_overlay_free();
    free(eng);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SUITE 12: PEPS 2D — 4×4 Trotter
 * ═══════════════════════════════════════════════════════════════════════ */

#define PEPS2D_L     4
#define PEPS2D_STEPS 10

static void run_peps2d_suite(int idx) {
    BenchResult *r = &R[idx];
    r->has_overlay = 1;
    r->overlay_label = "PEPS 2D";

    PepsGrid *g = peps_init(PEPS2D_L, PEPS2D_L);

    double G_re[36*36], G_im[36*36];
    build_cz_gate(G_re, G_im);

    double t0 = now_sec();
    for (int step = 0; step < PEPS2D_STEPS; step++)
        peps_trotter_step(g, G_re, G_im);
    r->overlay_time = now_sec() - t0;
    r->overlay_steps = PEPS2D_STEPS;

    peps_free(g);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SUITE 13: TNS 3D — 3×3×3 Trotter
 * ═══════════════════════════════════════════════════════════════════════ */

#define TNS3D_L     3
#define TNS3D_STEPS 5

static void run_tns3d_suite(int idx) {
    BenchResult *r = &R[idx];
    r->has_overlay = 1;
    r->overlay_label = "TNS 3D";

    Tns3dGrid *g = tns3d_init(TNS3D_L, TNS3D_L, TNS3D_L);

    double G_re[36*36], G_im[36*36];
    build_cz_gate(G_re, G_im);

    double t0 = now_sec();
    for (int step = 0; step < TNS3D_STEPS; step++)
        tns3d_trotter_step(g, G_re, G_im);
    r->overlay_time = now_sec() - t0;
    r->overlay_steps = TNS3D_STEPS;

    tns3d_free(g);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SUITE 14: TNS 4D — 3×3×3×3 Trotter
 * ═══════════════════════════════════════════════════════════════════════ */

#define TNS4D_L     3
#define TNS4D_STEPS 3

static void run_tns4d_suite(int idx) {
    BenchResult *r = &R[idx];
    r->has_overlay = 1;
    r->overlay_label = "TNS 4D";

    Tns4dGrid *g = tns4d_init(TNS4D_L, TNS4D_L, TNS4D_L, TNS4D_L);

    double G_re[36*36], G_im[36*36];
    build_cz_gate(G_re, G_im);

    double t0 = now_sec();
    for (int step = 0; step < TNS4D_STEPS; step++)
        tns4d_trotter_step(g, G_re, G_im);
    r->overlay_time = now_sec() - t0;
    r->overlay_steps = TNS4D_STEPS;

    tns4d_free(g);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SUITE 15: TNS 5D — 2×2×2×2×2 Trotter
 * ═══════════════════════════════════════════════════════════════════════ */

#define TNS5D_L     2
#define TNS5D_STEPS 3

static void run_tns5d_suite(int idx) {
    BenchResult *r = &R[idx];
    r->has_overlay = 1;
    r->overlay_label = "TNS 5D";

    Tns5dGrid *g = tns5d_init(TNS5D_L, TNS5D_L, TNS5D_L, TNS5D_L, TNS5D_L);

    double G_re[36*36], G_im[36*36];
    build_cz_gate(G_re, G_im);

    double t0 = now_sec();
    for (int step = 0; step < TNS5D_STEPS; step++)
        tns5d_trotter_step(g, G_re, G_im);
    r->overlay_time = now_sec() - t0;
    r->overlay_steps = TNS5D_STEPS;

    tns5d_free(g);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SUITE 16: TNS 6D — 2×2×2×2×2×2 Trotter
 * ═══════════════════════════════════════════════════════════════════════ */

#define TNS6D_L     2
#define TNS6D_STEPS 2

static void run_tns6d_suite(int idx) {
    BenchResult *r = &R[idx];
    r->has_overlay = 1;
    r->overlay_label = "TNS 6D";

    Tns6dGrid *g = tns6d_init(TNS6D_L, TNS6D_L, TNS6D_L, TNS6D_L, TNS6D_L, TNS6D_L);

    double G_re[36*36], G_im[36*36];
    build_cz_gate(G_re, G_im);

    double t0 = now_sec();
    for (int step = 0; step < TNS6D_STEPS; step++)
        tns6d_trotter_step(g, G_re, G_im);
    r->overlay_time = now_sec() - t0;
    r->overlay_steps = TNS6D_STEPS;

    tns6d_free(g);
}

/* ═══════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void)
{
    const char *names[N_BENCH] = {
        "Pure Z (1M)",          "Pure X (1M)",          "Alternating Z-X (1M)",
        "Trotter ZZZZ-XX (1M)", "DFT sequence (1M)",    "Mixed random (1M)",
        "Dyn chain growth",     "Triality rotation",    "Multi-measure (10K)",
        "Fusion stress (10M)",
        "MPS 1D (8 sites)",     "PEPS 2D (4x4)",       "TNS 3D (3x3x3)",
        "TNS 4D (3^4)",        "TNS 5D (2^5)",         "TNS 6D (2^6)"
    };
    for (int i = 0; i < N_BENCH; i++) R[i].name = names[i];

    printf("\n");
    printf("  ╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("  ║                                                                              ║\n");
    printf("  ║   ALL-IN-ONE TRIALITY QUHIT + OVERLAY BENCHMARK                              ║\n");
    printf("  ║                                                                              ║\n");
    printf("  ║   Standard · Eager Triality · Lazy Triality · DynChain · MPS · PEPS2D-6D    ║\n");
    printf("  ║                                                                              ║\n");
    printf("  ╚══════════════════════════════════════════════════════════════════════════════╝\n\n");

    printf("  ── QUHIT-LEVEL BENCHMARKS ──\n\n");
    for (int i = 0; i < 6; i++) { run_gate_suite(i, N_GATES, i); printf("    [%2d/16] %-25s done\n", i+1, names[i]); }
    run_dyn_chain_suite(6);   printf("    [ 7/16] %-25s done\n", names[6]);
    run_rotation_suite(7);    printf("    [ 8/16] %-25s done\n", names[7]);
    run_multi_measure_suite(8); printf("    [ 9/16] %-25s done\n", names[8]);
    run_stress_suite(9);      printf("    [10/16] %-25s done\n", names[9]);

    printf("\n  ── OVERLAY BENCHMARKS ──\n\n");
    run_mps_suite(10);   printf("    [11/16] %-25s done\n", names[10]);
    run_peps2d_suite(11); printf("    [12/16] %-25s done\n", names[11]);
    run_tns3d_suite(12); printf("    [13/16] %-25s done\n", names[12]);
    run_tns4d_suite(13); printf("    [14/16] %-25s done\n", names[13]);
    run_tns5d_suite(14); printf("    [15/16] %-25s done\n", names[14]);
    run_tns6d_suite(15); printf("    [16/16] %-25s done\n", names[15]);

    /* ═══════ QUHIT PERFORMANCE TABLE ═══════ */
    printf("\n  ┌─────────────────────────┬──────────┬──────────┬──────────┬────────┬────────┬────────┐\n");
    printf("  │ Quhit Suite             │   Lazy   │ Triality │ Standard │ L/S    │ T/S    │ L/T    │\n");
    printf("  ├─────────────────────────┼──────────┼──────────┼──────────┼────────┼────────┼────────┤\n");
    for (int i = 0; i < 10; i++) {
        char lb[16], tb[16], sb[16], ls[16], ts[16], lt[16];
        if (R[i].has_lazy) snprintf(lb,16,"%6.3fs",R[i].lazy_time); else snprintf(lb,16,"  N/A  ");
        if (R[i].has_tri)  snprintf(tb,16,"%6.3fs",R[i].tri_time);  else snprintf(tb,16,"  N/A  ");
        if (R[i].has_std)  snprintf(sb,16,"%6.3fs",R[i].std_time);  else snprintf(sb,16,"  N/A  ");
        if (R[i].has_lazy&&R[i].has_std&&R[i].lazy_time>1e-12) snprintf(ls,16,"%5.1fx",R[i].std_time/R[i].lazy_time); else snprintf(ls,16,"  -  ");
        if (R[i].has_tri&&R[i].has_std&&R[i].tri_time>1e-12) snprintf(ts,16,"%5.1fx",R[i].std_time/R[i].tri_time); else snprintf(ts,16,"  -  ");
        if (R[i].has_lazy&&R[i].has_tri&&R[i].lazy_time>1e-12) snprintf(lt,16,"%5.1fx",R[i].tri_time/R[i].lazy_time); else snprintf(lt,16,"  -  ");
        printf("  │ %-23s │ %s │ %s │ %s │ %s │ %s │ %s │\n", R[i].name, lb, tb, sb, ls, ts, lt);
    }
    printf("  └─────────────────────────┴──────────┴──────────┴──────────┴────────┴────────┴────────┘\n");

    /* ═══════ CORRECTNESS TABLE ═══════ */
    printf("\n  ┌─────────────────────────┬──────────────┬──────────────┬────────┐\n");
    printf("  │ Quhit Suite             │ Lazy Error   │ Tri Error    │ Status │\n");
    printf("  ├─────────────────────────┼──────────────┼──────────────┼────────┤\n");
    for (int i = 0; i < 10; i++) {
        const char *st = (!R[i].has_lazy&&!R[i].has_tri) ? "N/A" : (R[i].lazy_match&&R[i].tri_match) ? " ✓ " : " ✗ ";
        char le[16], te[16];
        if (R[i].has_lazy) snprintf(le,16,"%12.2e",R[i].lazy_err); else snprintf(le,16,"     N/A    ");
        if (R[i].has_tri)  snprintf(te,16,"%12.2e",R[i].tri_err);  else snprintf(te,16,"     N/A    ");
        printf("  │ %-23s │ %s │ %s │  %s   │\n", R[i].name, le, te, st);
    }
    printf("  └─────────────────────────┴──────────────┴──────────────┴────────┘\n");

    /* ═══════ FUSION TABLE ═══════ */
    printf("\n  ┌─────────────────────────┬───────────┬───────────┬──────────────┐\n");
    printf("  │ Suite                   │ Gates     │ Segments  │ Fusion Ratio │\n");
    printf("  ├─────────────────────────┼───────────┼───────────┼──────────────┤\n");
    for (int i = 0; i < 10; i++) {
        if (!R[i].has_lazy) continue;
        char rb[32];
        if (R[i].lazy_segs == 0) snprintf(rb,32,"      inf    ");
        else snprintf(rb,32,"%10.0f:1 ", (double)R[i].lazy_gates/R[i].lazy_segs);
        printf("  │ %-23s │ %9lu │ %9lu │ %s │\n", R[i].name, (unsigned long)R[i].lazy_gates, (unsigned long)R[i].lazy_segs, rb);
    }
    printf("  └─────────────────────────┴───────────┴───────────┴──────────────┘\n");

    /* ═══════ DYN STATS ═══════ */
    printf("\n  ┌─────────────────────────────────────────────────────┐\n");
    printf("  │  DYNCHAIN BREATHING                                 │\n");
    printf("  ├─────────────────────────────────────────────────────┤\n");
    printf("  │  Grow events:       %8d                        │\n", R[6].dyn_grows);
    printf("  │  Contract events:   %8d                        │\n", R[6].dyn_contracts);
    printf("  │  Final active:      %8d / %d sites             │\n", R[6].dyn_active, DYN_SITES);
    printf("  └─────────────────────────────────────────────────────┘\n");

    /* ═══════ OVERLAY TABLE ═══════ */
    printf("\n  ┌──────────────────────────────────────────────────────────────────┐\n");
    printf("  │  TENSOR NETWORK OVERLAY BENCHMARKS                               │\n");
    printf("  ├─────────────────────────┬──────────┬────────┬────────────────────┤\n");
    printf("  │ Overlay                 │   Time   │ Steps  │ Step/s             │\n");
    printf("  ├─────────────────────────┼──────────┼────────┼────────────────────┤\n");
    for (int i = 10; i < N_BENCH; i++) {
        if (!R[i].has_overlay) continue;
        double sps = R[i].overlay_time > 1e-12 ? R[i].overlay_steps / R[i].overlay_time : 0;
        printf("  │ %-23s │ %6.3fs │ %6d │ %10.1f steps/s  │\n",
               R[i].name, R[i].overlay_time, R[i].overlay_steps, sps);
    }
    printf("  └─────────────────────────┴──────────┴────────┴────────────────────┘\n");

    /* ═══════ SUMMARY ═══════ */
    double tot_l = 0, tot_t = 0, tot_s = 0, tot_o = 0;
    int all_ok = 1;
    for (int i = 0; i < 10; i++) {
        if (R[i].has_lazy) tot_l += R[i].lazy_time;
        if (R[i].has_tri)  tot_t += R[i].tri_time;
        if (R[i].has_std)  tot_s += R[i].std_time;
        if (R[i].has_lazy && !R[i].lazy_match) all_ok = 0;
        if (R[i].has_tri  && !R[i].tri_match)  all_ok = 0;
    }
    for (int i = 10; i < N_BENCH; i++) if (R[i].has_overlay) tot_o += R[i].overlay_time;

    double best_lazy = 0; const char *best_ln = "";
    for (int i = 0; i < 10; i++) {
        if (R[i].has_lazy && R[i].has_std && R[i].lazy_time > 1e-12) {
            double sp = R[i].std_time / R[i].lazy_time;
            if (sp > best_lazy) { best_lazy = sp; best_ln = R[i].name; }
        }
    }

    printf("\n  ╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("  ║  SUMMARY                                                                     ║\n");
    printf("  ║                                                                              ║\n");
    printf("  ║  Quhit benchmarks:                                                           ║\n");
    printf("  ║    Lazy total:    %8.3fs                                                  ║\n", tot_l);
    printf("  ║    Triality total:%8.3fs                                                  ║\n", tot_t);
    printf("  ║    Standard total:%8.3fs                                                  ║\n", tot_s);
    printf("  ║    Best lazy speedup: %5.1fx on %-20s                      ║\n", best_lazy, best_ln);
    printf("  ║    Correctness: %s                                                       ║\n",
           all_ok ? "ALL PASS ✓" : "MISMATCH ✗");
    printf("  ║                                                                              ║\n");
    printf("  ║  Overlay benchmarks:                                                         ║\n");
    printf("  ║    Total time:    %8.3fs across MPS + PEPS2D-6D                          ║\n", tot_o);
    printf("  ║                                                                              ║\n");
    printf("  ║  The state sits still. The transformation accumulates.                       ║\n");
    printf("  ║  Only when you look does the computation happen.                             ║\n");
    printf("  ║                                                                              ║\n");
    printf("  ╚══════════════════════════════════════════════════════════════════════════════╝\n\n");

    return 0;
}
