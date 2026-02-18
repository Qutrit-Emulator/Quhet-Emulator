/*
 * mirror_supremacy.c — Cross-Window Bell Test
 *
 * Create Bell pair (q_A, q_B). Send q_B through N "windows" of
 * DFT+DNA+DFT operations while q_A sits untouched. Then measure
 * both in different bases.
 *
 * QUANTUM arm: keep full joint state — entanglement preserved
 * CLASSICAL arm: measure q_A after Bell pair, breaking entanglement
 *
 * If I(A;B) stays high after many windows in QUANTUM but drops
 * in CLASSICAL → entanglement persists across windows.
 *
 * Cross-engine variant: carry 36-entry joint state between fresh
 * engines (one per window). Proves serialization preserves entanglement.
 *
 * BUILD:
 *   gcc -O2 -I. -std=c11 -D_GNU_SOURCE \
 *       -o mirror_supremacy mirror_supremacy.c hexstate_engine.o bigint.o -lm
 */

#include "hexstate_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>

#define D       6
#define Q_A     0
#define Q_B     1
#define N_Q     100000000000000ULL
#define TRIALS  500

static int saved_fd = -1;
static void hush(void) {
    fflush(stdout); saved_fd = dup(STDOUT_FILENO);
    int dn = open("/dev/null", O_WRONLY); dup2(dn, STDOUT_FILENO); close(dn);
}
static void unhush(void) {
    if (saved_fd >= 0) { fflush(stdout); dup2(saved_fd, STDOUT_FILENO); close(saved_fd); saved_fd = -1; }
}

/* ══ Joint state: 6×6 amplitude matrix ══ */
typedef struct { Complex amp[D][D]; } JointState;

static JointState extract_joint(HexStateEngine *eng) {
    JointState js; memset(&js, 0, sizeof(js));
    int r = find_quhit_reg(eng, 0); if (r < 0) return js;
    uint32_t nz = eng->quhit_regs[r].num_nonzero;
    uint32_t dim = eng->quhit_regs[r].dim;
    for (uint32_t e = 0; e < nz; e++) {
        QuhitBasisEntry *ent = &eng->quhit_regs[r].entries[e];
        uint32_t va = UINT32_MAX, vb = UINT32_MAX;
        for (uint8_t i = 0; i < ent->num_addr; i++) {
            if (ent->addr[i].quhit_idx == Q_A) va = ent->addr[i].value;
            if (ent->addr[i].quhit_idx == Q_B) vb = ent->addr[i].value;
        }
        if (va == UINT32_MAX) va = (ent->bulk_value + Q_A) % dim;
        if (vb == UINT32_MAX) vb = (ent->bulk_value + Q_B) % dim;
        if (va < D && vb < D) {
            js.amp[va][vb].real += ent->amplitude.real;
            js.amp[va][vb].imag += ent->amplitude.imag;
        }
    }
    return js;
}

static void inject_joint(HexStateEngine *eng, const JointState *js) {
    int r = find_quhit_reg(eng, 0); if (r < 0) return;
    eng->quhit_regs[r].num_nonzero = 0;
    eng->quhit_regs[r].collapsed = 0;
    for (uint32_t a = 0; a < D; a++)
        for (uint32_t b = 0; b < D; b++) {
            double p = js->amp[a][b].real*js->amp[a][b].real +
                       js->amp[a][b].imag*js->amp[a][b].imag;
            if (p < 1e-15) continue;
            QuhitBasisEntry e; memset(&e, 0, sizeof(e));
            e.bulk_value = 0; e.num_addr = 2;
            e.addr[0].quhit_idx = Q_A; e.addr[0].value = a;
            e.addr[1].quhit_idx = Q_B; e.addr[1].value = b;
            e.amplitude = js->amp[a][b];
            uint32_t nz = eng->quhit_regs[r].num_nonzero;
            if (nz < MAX_QUHIT_HILBERT_ENTRIES) {
                eng->quhit_regs[r].entries[nz] = e;
                eng->quhit_regs[r].num_nonzero = nz + 1;
            }
        }
}

/* ══ Single trial ══ */

/* Run one Bell test trial.
 * Returns measured (a, b).
 * classical_break: if 1, measure q_A right after Bell pair (kill entanglement)
 * cross_engine: if 1, carry joint state between fresh engines per window
 * basis_a, basis_b: 0=computational, 1=Fourier */
static void bell_trial(int n_windows, int classical_break, int cross_engine,
                        int basis_a, int basis_b,
                        int *out_a, int *out_b)
{
    static HexStateEngine eng;  /* static — too large for stack */
    hush(); engine_init(&eng);
    init_quhit_register(&eng, 0, N_Q, D); unhush();
    eng.quhit_regs[0].bulk_rule = 0;

    /* Directly inject Bell state: |Φ⟩ = (1/√6) Σ_k |k⟩_A |k⟩_B */
    {
        int r = find_quhit_reg(&eng, 0);
        if (r >= 0) {
            eng.quhit_regs[r].num_nonzero = 0;
            eng.quhit_regs[r].collapsed = 0;
            double a = 1.0 / sqrt(6.0);
            for (uint32_t k = 0; k < D; k++) {
                QuhitBasisEntry e; memset(&e, 0, sizeof(e));
                e.bulk_value = 0;
                e.num_addr = 2;
                e.addr[0].quhit_idx = Q_A; e.addr[0].value = k;
                e.addr[1].quhit_idx = Q_B; e.addr[1].value = k;
                e.amplitude.real = a; e.amplitude.imag = 0;
                eng.quhit_regs[r].entries[k] = e;
            }
            eng.quhit_regs[r].num_nonzero = D;
        }
    }

    int classical_a = -1;
    if (classical_break) {
        /* Kill entanglement: measure q_A, collapse q_B */
        hush(); classical_a = (int)(measure_quhit(&eng, 0, Q_A) % D); unhush();
        /* Reset collapsed flag so gates on q_B still work */
        int r = find_quhit_reg(&eng, 0);
        if (r >= 0) eng.quhit_regs[r].collapsed = 0;
    }

    /* Apply windows to q_B */
    if (cross_engine && n_windows > 0) {
        JointState js = extract_joint(&eng);
        hush(); engine_destroy(&eng); unhush();

        for (int w = 0; w < n_windows; w++) {
            static HexStateEngine eng2;
            hush(); engine_init(&eng2);
            init_quhit_register(&eng2, 0, N_Q, D); unhush();
            eng2.quhit_regs[0].bulk_rule = 0;
            inject_joint(&eng2, &js);
            hush();
            apply_dft_quhit(&eng2, 0, Q_B, D);
            apply_dna_quhit(&eng2, 0, Q_B, 1.0, 310.0 + w * 73.0);
            apply_dft_quhit(&eng2, 0, Q_B, D);
            unhush();
            js = extract_joint(&eng2);
            hush(); engine_destroy(&eng2); unhush();
        }

        /* Final engine for measurement */
        hush(); engine_init(&eng);
        init_quhit_register(&eng, 0, N_Q, D); unhush();
        eng.quhit_regs[0].bulk_rule = 0;
        inject_joint(&eng, &js);
    } else {
        for (int w = 0; w < n_windows; w++) {
            hush();
            apply_dft_quhit(&eng, 0, Q_B, D);
            apply_dna_quhit(&eng, 0, Q_B, 1.0, 310.0 + w * 73.0);
            apply_dft_quhit(&eng, 0, Q_B, D);
            unhush();
        }
    }

    /* Basis rotations */
    if (basis_a && !classical_break) {
        hush(); apply_dft_quhit(&eng, 0, Q_A, D); unhush();
    }
    if (basis_b) { hush(); apply_dft_quhit(&eng, 0, Q_B, D); unhush(); }

    /* Measure */
    hush();
    if (classical_break) {
        *out_a = classical_a;  /* use the already-measured value */
    } else {
        *out_a = (int)(measure_quhit(&eng, 0, Q_A) % D);
    }
    *out_b = (int)(measure_quhit(&eng, 0, Q_B) % D);
    engine_destroy(&eng);
    unhush();
}

/* ══ Statistics ══ */

typedef struct {
    int joint[D][D];
    int n;
    double mi, p_agree, p_sum0;
} Stats;

static Stats compute_stats(int joint[D][D], int n) {
    Stats s; memset(&s, 0, sizeof(s));
    memcpy(s.joint, joint, sizeof(s.joint));
    s.n = n;
    double pa[D], pb[D];
    memset(pa, 0, sizeof(pa)); memset(pb, 0, sizeof(pb));
    for (int a = 0; a < D; a++)
        for (int b = 0; b < D; b++) {
            double p = (double)joint[a][b] / n;
            pa[a] += p; pb[b] += p;
            if (a == b) s.p_agree += p;
            if ((a + b) % D == 0) s.p_sum0 += p;
        }
    s.mi = 0;
    for (int a = 0; a < D; a++)
        for (int b = 0; b < D; b++) {
            double p = (double)joint[a][b] / n;
            if (p > 1e-10 && pa[a] > 1e-10 && pb[b] > 1e-10)
                s.mi += p * log2(p / (pa[a] * pb[b]));
        }
    return s;
}

/* ══ Run Bell test for a given window count ══ */

typedef struct {
    Stats quantum[4];   /* 4 basis settings */
    Stats classical[4];
    Stats cross_eng[4]; /* cross-engine */
    double mi_q, mi_c, mi_x;
} BellResult;

static const char *basis_name[4] = {"Comp,Comp", "Comp,Four", "Four,Comp", "Four,Four"};

static BellResult run_bell_test(int n_windows, int trials)
{
    BellResult br; memset(&br, 0, sizeof(br));
    int jq[4][D][D], jc[4][D][D], jx[4][D][D];
    memset(jq, 0, sizeof(jq));
    memset(jc, 0, sizeof(jc));
    memset(jx, 0, sizeof(jx));

    printf("    Running %d windows, %d trials/setting...\n", n_windows, trials);

    for (int setting = 0; setting < 4; setting++) {
        int ba = setting & 1, bb = (setting >> 1) & 1;
        for (int t = 0; t < trials; t++) {
            int a, b;
            /* Quantum */
            bell_trial(n_windows, 0, 0, ba, bb, &a, &b);
            jq[setting][a][b]++;
            /* Classical */
            bell_trial(n_windows, 1, 0, ba, bb, &a, &b);
            jc[setting][a][b]++;
        }
        /* Cross-engine: fewer trials (slower) */
        int x_trials = (n_windows > 0) ? (trials / 5) : 0;
        for (int t = 0; t < x_trials; t++) {
            int a, b;
            bell_trial(n_windows, 0, 1, ba, bb, &a, &b);
            jx[setting][a][b]++;
        }
        br.quantum[setting] = compute_stats(jq[setting], trials);
        br.classical[setting] = compute_stats(jc[setting], trials);
        if (x_trials > 0)
            br.cross_eng[setting] = compute_stats(jx[setting], x_trials);
    }

    /* Average MI across settings */
    br.mi_q = br.mi_c = br.mi_x = 0;
    for (int s = 0; s < 4; s++) {
        br.mi_q += br.quantum[s].mi;
        br.mi_c += br.classical[s].mi;
        br.mi_x += br.cross_eng[s].mi;
    }
    br.mi_q /= 4; br.mi_c /= 4; br.mi_x /= 4;

    return br;
}

static void print_bell_result(int n_windows, const BellResult *br)
{
    printf("\n  ┌──────────────────────────────────────────────────────────────┐\n");
    printf("  │  BELL TEST: %d windows between Alice and Bob              │\n", n_windows);
    printf("  ├──────────┬──────────┬──────────┬──────────┬────────────────┤\n");
    printf("  │ Basis    │ P(a==b)Q │ P(a==b)C │ I(A;B) Q │ I(A;B) C      │\n");
    printf("  ├──────────┼──────────┼──────────┼──────────┼────────────────┤\n");
    for (int s = 0; s < 4; s++) {
        printf("  │ %-8s │  %.4f  │  %.4f  │  %.4f  │  %.4f        │\n",
               basis_name[s],
               br->quantum[s].p_agree, br->classical[s].p_agree,
               br->quantum[s].mi, br->classical[s].mi);
    }
    printf("  ├──────────┴──────────┴──────────┴──────────┴────────────────┤\n");
    printf("  │  Average MI:  Quantum = %.4f    Classical = %.4f       │\n",
           br->mi_q, br->mi_c);
    if (br->cross_eng[0].n > 0)
        printf("  │               Cross-engine = %.4f                       │\n",
               br->mi_x);

    int entangled = (br->mi_q > br->mi_c + 0.1);
    printf("  │                                                            │\n");
    if (entangled) {
        printf("  │  ✓ ENTANGLEMENT PERSISTS through %d windows              │\n", n_windows);
        printf("  │    Quantum MI >> Classical MI → phases matter            │\n");
    } else if (n_windows == 0) {
        printf("  │  Bell pair baseline (no windows)                         │\n");
    } else {
        printf("  │  MI too close to distinguish Q vs C at this sample size  │\n");
    }
    printf("  └──────────────────────────────────────────────────────────────┘\n");
}

/* ══ Main ══ */

int main(void)
{
    srand(42);

    printf("\n");
    printf("  ╔═══════════════════════════════════════════════════════════════╗\n");
    printf("  ║  C R O S S - W I N D O W   B E L L   T E S T              ║\n");
    printf("  ║                                                             ║\n");
    printf("  ║  Alice creates Bell pair with Bob (q_A, q_B)               ║\n");
    printf("  ║  Bob's qudit travels through N windows of gates            ║\n");
    printf("  ║  Alice's qudit sits untouched                              ║\n");
    printf("  ║  Both measured in Computational or Fourier basis           ║\n");
    printf("  ║                                                             ║\n");
    printf("  ║  QUANTUM: full entanglement preserved                      ║\n");
    printf("  ║  CLASSICAL: entanglement broken after Bell pair            ║\n");
    printf("  ║  CROSS-ENGINE: joint state carried between fresh engines   ║\n");
    printf("  ║                                                             ║\n");
    printf("  ║  If I(A;B)_quantum > I(A;B)_classical after W windows     ║\n");
    printf("  ║  → entanglement persists across all W windows              ║\n");
    printf("  ╚═══════════════════════════════════════════════════════════════╝\n\n");

    struct timespec t0, t1;

    int test_windows[] = {0, 1, 5, 10, 20};
    int n_tests = 5;
    BellResult results[5];

    for (int i = 0; i < n_tests; i++) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        results[i] = run_bell_test(test_windows[i], TRIALS);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)/1e9;
        print_bell_result(test_windows[i], &results[i]);
        printf("    [%.1f seconds]\n\n", sec);
    }

    /* ══ Summary ══ */
    printf("  ╔═══════════════════════════════════════════════════════════════╗\n");
    printf("  ║  ★ CROSS-WINDOW BELL TEST SUMMARY ★                       ║\n");
    printf("  ╠═══════════════════════════════════════════════════════════════╣\n");
    printf("  ║  Windows │ I(A;B) Q │ I(A;B) C │  Δ     │ Verdict         ║\n");
    printf("  ║  ────────┼──────────┼──────────┼────────┼──────────        ║\n");
    for (int i = 0; i < n_tests; i++) {
        double delta = results[i].mi_q - results[i].mi_c;
        const char *v = (delta > 0.1) ? "ENTANGLED ✓" :
                        (test_windows[i] == 0) ? "BASELINE" : "~same";
        printf("  ║  %4d    │  %.4f  │  %.4f  │ %+.3f │ %-15s ║\n",
               test_windows[i], results[i].mi_q, results[i].mi_c, delta, v);
    }
    printf("  ║                                                             ║\n");
    printf("  ║  Quantum theory: local unitaries CANNOT destroy             ║\n");
    printf("  ║  entanglement. I(A;B) should be invariant under windows.   ║\n");
    printf("  ║  Classical control: entanglement broken → I drops.          ║\n");
    printf("  ╚═══════════════════════════════════════════════════════════════╝\n\n");

    return 0;
}
