/* exotic_invariant_test.c — Exotic Invariant Δ Diagnostic
 *
 * Measures the D=6 hexagonal quantum number Δ(ψ) across:
 *   - Basis states |k⟩
 *   - Uniform superposition DFT|0⟩
 *   - States after various gates (Z, X, DFT, CZ)
 *   - The factoring oracle state
 * Shows WHERE D=6 is working and where it's just along for the ride.
 */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "quhit_triality.h"
#include "s6_exotic.h"

static const double W6_RE[6] = {1.0, 0.5, -0.5, -1.0, -0.5, 0.5};
static const double W6_IM[6] = {0.0, 0.866025, 0.866025, 0.0, -0.866025, -0.866025};

static void print_bar(double val, double max_val) {
    int len = (max_val > 0) ? (int)(40.0 * val / max_val) : 0;
    if (len < 0) len = 0;
    if (len > 40) len = 40;
    for (int i = 0; i < len; i++) printf("█");
    for (int i = len; i < 40; i++) printf("░");
}

static void print_delta(const char *label, double re[6], double im[6]) {
    double delta = s6_exotic_invariant(re, im);
    double dS = s6_exotic_entropy(re, im, 0);
    printf("  %-28s Δ=%8.4f  ΔS=%+.4f  ", label, delta, dS);
    print_bar(delta, 10.0);
    printf("\n");
}

int main(void) {
    printf("\n");
    printf("  ╔══════════════════════════════════════════════════════════════╗\n");
    printf("  ║                                                              ║\n");
    printf("  ║   THE EXOTIC INVARIANT Δ — Where D=6 Bites                   ║\n");
    printf("  ║                                                              ║\n");
    printf("  ║   Δ=0: generic (could run on qubits)                         ║\n");
    printf("  ║   Δ>0: hexagonally polarized (D=6 advantage)                 ║\n");
    printf("  ║                                                              ║\n");
    printf("  ╚══════════════════════════════════════════════════════════════╝\n\n");

    triality_exotic_init();

    /* ═══ §1: Basis states ═══ */
    printf("  ═══ §1: Basis States ═══\n\n");
    for (int k = 0; k < 6; k++) {
        double re[6] = {0}, im[6] = {0};
        re[k] = 1.0;
        char label[32]; sprintf(label, "|%d⟩", k);
        print_delta(label, re, im);
    }
    printf("\n");

    /* ═══ §2: Superpositions ═══ */
    printf("  ═══ §2: Superpositions ═══\n\n");
    {
        double n6 = 1.0/sqrt(6.0);
        double re[6], im[6] = {0};
        for (int k = 0; k < 6; k++) re[k] = n6;
        print_delta("DFT|0⟩ (uniform)", re, im);
    }
    {
        double n2 = 1.0/sqrt(2.0);
        double re[6] = {n2, n2, 0, 0, 0, 0}, im[6] = {0};
        print_delta("|0⟩+|1⟩", re, im);
    }
    {
        double n2 = 1.0/sqrt(2.0);
        double re[6] = {n2, 0, 0, n2, 0, 0}, im[6] = {0};
        print_delta("|0⟩+|3⟩ (antipodal)", re, im);
    }
    {
        double n3 = 1.0/sqrt(3.0);
        double re[6] = {n3, 0, n3, 0, n3, 0}, im[6] = {0};
        print_delta("|0⟩+|2⟩+|4⟩ (even)", re, im);
    }
    {
        double n3 = 1.0/sqrt(3.0);
        double re[6] = {0, n3, 0, n3, 0, n3}, im[6] = {0};
        print_delta("|1⟩+|3⟩+|5⟩ (odd)", re, im);
    }
    printf("\n");

    /* ═══ §3: After gates ═══ */
    printf("  ═══ §3: Gate Evolution — Δ after each operation ═══\n\n");
    {
        TrialityQuhit q;
        triality_init(&q);
        print_delta("Init |0⟩", q.edge_re, q.edge_im);

        triality_dft(&q);
        print_delta("After DFT₆", q.edge_re, q.edge_im);

        triality_z(&q);
        print_delta("After DFT₆ → Z", q.edge_re, q.edge_im);

        triality_dft(&q);
        print_delta("After DFT₆ → Z → DFT₆", q.edge_re, q.edge_im);

        triality_x(&q);
        print_delta("After DFT → Z → DFT → X", q.edge_re, q.edge_im);
    }
    printf("\n");

    /* ═══ §4: Factoring oracle states ═══ */
    printf("  ═══ §4: Factoring Oracle States ═══\n\n");
    {
        /* State encoding a^x mod 7 for a=3 */
        /* Powers: 3^0=1, 3^1=3, 3^2=2, 3^3=6, 3^4=4, 3^5=5 mod 7
         * Map to indices: x → (a^x mod 7) - 1 */
        double re[6] = {0}, im[6] = {0};
        int vals[6] = {0, 2, 1, 5, 3, 4}; /* 3^x mod 7's permutation */
        double n = 1.0/sqrt(6.0);
        for (int x = 0; x < 6; x++) {
            re[vals[x]] += n;
        }
        /* Normalize */
        double total = 0;
        for (int k = 0; k < 6; k++) total += re[k]*re[k] + im[k]*im[k];
        double sc = 1.0/sqrt(total);
        for (int k = 0; k < 6; k++) { re[k] *= sc; im[k] *= sc; }
        print_delta("Oracle: 3^x mod 7", re, im);
    }
    {
        /* After DFT, the period structure should emerge */
        TrialityQuhit q; triality_init(&q);
        double n6 = 1.0/sqrt(6.0);
        for (int k = 0; k < 6; k++) q.edge_re[k] = n6;
        q.active_mask = 0x3F; q.active_count = 6;
        q.dirty = DIRTY_VERTEX|DIRTY_DIAGONAL|DIRTY_FOLDED|DIRTY_EXOTIC;
        /* Apply Z gate (phase imprint) */
        triality_z(&q);
        print_delta("Uniform + Z phase", q.edge_re, q.edge_im);
        /* Apply DFT to extract period */
        triality_dft(&q);
        print_delta("Uniform + Z → DFT₆", q.edge_re, q.edge_im);
    }
    printf("\n");

    /* ═══ §5: Exotic gate effect on Δ ═══ */
    printf("  ═══ §5: Standard vs Exotic Gate on Same State ═══\n\n");
    {
        S6Perm sigma = {{1,0,2,3,4,5}}; /* transposition (01) */
        double n2 = 1.0/sqrt(2.0);

        /* Start state: |0⟩+|3⟩ */
        TrialityQuhit q_std, q_exo;
        triality_init(&q_std); triality_init(&q_exo);
        q_std.edge_re[0] = n2; q_std.edge_re[3] = n2;
        q_exo.edge_re[0] = n2; q_exo.edge_re[3] = n2;
        q_std.active_mask = q_exo.active_mask = 0x09;
        q_std.active_count = q_exo.active_count = 2;
        q_std.dirty = q_exo.dirty = DIRTY_VERTEX|DIRTY_DIAGONAL|DIRTY_FOLDED|DIRTY_EXOTIC;

        print_delta("Before gate: |0⟩+|3⟩", q_std.edge_re, q_std.edge_im);

        /* Standard gate */
        double tmp_re[6] = {0}, tmp_im[6] = {0};
        for (int i = 0; i < 6; i++) {
            tmp_re[sigma.p[i]] = q_std.edge_re[i];
            tmp_im[sigma.p[i]] = q_std.edge_im[i];
        }
        memcpy(q_std.edge_re, tmp_re, sizeof(tmp_re));
        memcpy(q_std.edge_im, tmp_im, sizeof(tmp_im));
        print_delta("After standard (01)", q_std.edge_re, q_std.edge_im);

        /* Exotic gate */
        triality_exotic_gate(&q_exo, sigma);
        print_delta("After exotic φ((01))", q_exo.edge_re, q_exo.edge_im);
    }
    printf("\n");

    /* ═══ §6: Fingerprint — per-class breakdown ═══ */
    printf("  ═══ §6: Exotic Fingerprint (per conjugacy class) ═══\n\n");
    {
        /* Most interesting state: DFT|0⟩ + Z phase */
        TrialityQuhit q; triality_init(&q);
        triality_dft(&q);
        triality_z(&q);

        double fp[11];
        s6_exotic_fingerprint(q.edge_re, q.edge_im, fp);

        const char *class_names[11] = {
            "1⁶ (id)", "2·1⁴ (transp)", "2²·1²", "2³ (triple-tr)",
            "3·1³ (3-cycle)", "3·2·1", "4·1² (4-cycle)", "4·2",
            "5·1 (5-cycle)", "3² (dbl 3-cyc)", "6 (6-cycle)"
        };
        const char *phi_swaps[11] = {
            "fixed", "↔ 2³", "fixed", "↔ 2·1⁴",
            "↔ 3²", "fixed", "↔ 4·2", "↔ 4·1²",
            "fixed", "↔ 3·1³", "fixed"
        };
        int sizes[11] = {1, 15, 45, 15, 40, 120, 90, 90, 144, 40, 120};

        double max_fp = 0;
        for (int c = 0; c < 11; c++) if (fp[c] > max_fp) max_fp = fp[c];

        printf("  %-18s  %-10s %-11s  %-10s\n",
               "Class", "φ swaps?", "|C|", "Avg Δ");
        printf("  ────────────────── ────────── ─────────── ──────────\n");
        for (int c = 0; c < 11; c++) {
            printf("  %-18s  %-10s %-11d  %-10.6f  ",
                   class_names[c], phi_swaps[c], sizes[c], fp[c]);
            print_bar(fp[c], max_fp);
            printf("%s\n", fp[c] > 0.001 ? " ◄" : "");
        }
    }
    printf("\n");

    /* ═══ §7: Δ sweep — apply DFT repeatedly ═══ */
    printf("  ═══ §7: Δ Evolution Under Repeated DFT₆ ═══\n\n");
    {
        TrialityQuhit q; triality_init(&q);
        /* Start: |0⟩ + |2⟩ + |4⟩ (even subspace) */
        double n3 = 1.0/sqrt(3.0);
        q.edge_re[0] = n3; q.edge_re[2] = n3; q.edge_re[4] = n3;
        q.active_mask = 0x15; q.active_count = 3;
        q.dirty = DIRTY_VERTEX|DIRTY_DIAGONAL|DIRTY_FOLDED|DIRTY_EXOTIC;

        printf("  Step  State                          Δ         ΔS\n");
        printf("  ──── ─────────────────────────────── ───────── ─────────\n");
        for (int step = 0; step < 8; step++) {
            double delta = s6_exotic_invariant(q.edge_re, q.edge_im);
            double dS = s6_exotic_entropy(q.edge_re, q.edge_im, 0);
            printf("  %4d  [", step);
            for (int k = 0; k < 6; k++)
                printf("%+.3f%+.3fi%s", q.edge_re[k], q.edge_im[k],
                       k < 5 ? " " : "");
            printf("]  %8.4f  %+.4f\n", delta, dS);

            /* Apply Z then DFT */
            triality_z(&q);
            triality_dft(&q);
            triality_ensure_view(&q, VIEW_EDGE);
        }
    }

    printf("\n  ═══════════════════════════════════════════════════════════════\n\n");
    return 0;
}
