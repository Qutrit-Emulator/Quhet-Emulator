/*
 * test_mps_svd.c — Verify MPS at χ=128 produces valid tensor network data
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mps_overlay.c"

static void build_hadamard6(double *U_re, double *U_im)
{
    int D = 6;
    double norm = 1.0 / sqrt((double)D);
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++) {
            double angle = 2.0 * M_PI * i * j / D;
            U_re[i*D+j] = norm * cos(angle);
            U_im[i*D+j] = norm * sin(angle);
        }
}

static void build_cz6(double *G_re, double *G_im)
{
    int D = 6, D2 = 36;
    memset(G_re, 0, D2*D2*sizeof(double));
    memset(G_im, 0, D2*D2*sizeof(double));
    double omega = 2.0 * M_PI / 6.0;
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++) {
            int idx = i*D + j;
            double phase = omega * i * j;
            G_re[idx * D2 + idx] = cos(phase);
            G_im[idx * D2 + idx] = sin(phase);
        }
}

static int check_density(const char *label, double *probs, int D)
{
    double sum = 0;
    int valid = 1;
    for (int k = 0; k < D; k++) {
        sum += probs[k];
        if (probs[k] < -1e-10 || probs[k] > 1.0 + 1e-10 || isnan(probs[k]))
            valid = 0;
    }
    printf("  %s: [", label);
    for (int k = 0; k < D; k++) printf("%.4f%s", probs[k], k<D-1?", ":"");
    printf("]  sum=%.6f  %s\n", sum, (fabs(sum-1.0)<1e-6 && valid) ? "✓" : "✗ FAIL");
    return (fabs(sum-1.0)<1e-6 && valid);
}

/* Compute local density from MPS register */
static void mps_local_density(int site, double *probs)
{
    int D = MPS_PHYS, chi = MPS_CHI;
    for (int k = 0; k < D; k++) probs[k] = 0;

    if (!mps_store || !mps_eng || mps_store[site].reg_idx < 0) {
        probs[0] = 1.0;
        return;
    }

    QuhitRegister *r = &mps_eng->registers[mps_store[site].reg_idx];
    double total = 0;
    for (uint32_t e = 0; e < r->num_nonzero; e++) {
        uint64_t bs = r->entries[e].basis_state;
        int k = (int)(bs / (chi * chi));  /* k is highest position */
        if (k >= D) continue;
        double re = r->entries[e].amp_re;
        double im = r->entries[e].amp_im;
        double p = re*re + im*im;
        probs[k] += p;
        total += p;
    }
    if (total > 1e-30)
        for (int k = 0; k < D; k++) probs[k] /= total;
    else
        probs[0] = 1.0;
}

int main(void)
{
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  MPS VERIFICATION — χ=%d, D=%d                  ║\n", MPS_CHI, MPS_PHYS);
    printf("║  SVD dim: %d × %d                              ║\n", MPS_PHYS*MPS_CHI, MPS_PHYS*MPS_CHI);
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    int N = 8;  /* 8-site chain */
    QuhitEngine *eng = (QuhitEngine *)calloc(1, sizeof(QuhitEngine));
    quhit_engine_init(eng);
    uint32_t quhits[8];
    for (int i = 0; i < N; i++) quhits[i] = quhit_init_basis(eng, 0);

    mps_overlay_init(eng, quhits, N);

    int pass = 1;
    double probs[6];

    /* Test 1: Product state */
    printf("[1] Product state |0...0⟩\n");
    mps_local_density(0, probs);
    pass &= check_density("Site 0", probs, 6);
    mps_local_density(N-1, probs);
    pass &= check_density("Site 7", probs, 6);

    /* Test 2: Hadamard on site 0 */
    printf("[2] Hadamard on site 0\n");
    double U_re[36], U_im[36];
    build_hadamard6(U_re, U_im);
    mps_gate_1site(eng, quhits, N, 0, U_re, U_im);
    mps_local_density(0, probs);
    pass &= check_density("Site 0", probs, 6);
    int uniform = 1;
    for (int k = 0; k < 6; k++)
        if (fabs(probs[k] - 1.0/6.0) > 0.05) uniform = 0;
    printf("  → Uniform? %s\n", uniform ? "YES ✓" : "NO ✗");
    pass &= uniform;

    /* Test 3: CZ₆ 2-site gate with SVD */
    printf("[3] CZ₆ between sites 0-1 (SVD at %d×%d)\n", MPS_PHYS*MPS_CHI, MPS_PHYS*MPS_CHI);
    double G_re[36*36], G_im[36*36];
    build_cz6(G_re, G_im);
    mps_gate_1site(eng, quhits, N, 1, U_re, U_im);  /* H on site 1 first */

    clock_t t0 = clock();
    mps_gate_2site(eng, quhits, N, 0, G_re, G_im);
    clock_t t1 = clock();
    double svd_time = (double)(t1 - t0) / CLOCKS_PER_SEC;

    mps_local_density(0, probs);
    pass &= check_density("Site 0 post-CZ", probs, 6);
    mps_local_density(1, probs);
    pass &= check_density("Site 1 post-CZ", probs, 6);
    printf("  → SVD time: %.3f s\n", svd_time);

    /* Test 4: Chain of CZ₆ along entire MPS */
    printf("[4] H on all sites + CZ₆ chain (7 SVDs)\n");
    for (int i = 2; i < N; i++)
        mps_gate_1site(eng, quhits, N, i, U_re, U_im);

    t0 = clock();
    for (int i = 0; i < N - 1; i++)
        mps_gate_2site(eng, quhits, N, i, G_re, G_im);
    t1 = clock();
    svd_time = (double)(t1 - t0) / CLOCKS_PER_SEC;

    int all_valid = 1;
    for (int i = 0; i < N; i++) {
        char label[32];
        snprintf(label, sizeof(label), "Site %d chain", i);
        mps_local_density(i, probs);
        all_valid &= check_density(label, probs, 6);
    }
    pass &= all_valid;
    printf("  → 7 SVDs in: %.3f s (%.3f s/gate)\n", svd_time, svd_time/7.0);

    /* Test 5: 3 full sweeps */
    printf("[5] 3 full sweeps (H + CZ chain × 3)\n");
    t0 = clock();
    for (int sweep = 0; sweep < 3; sweep++) {
        for (int i = 0; i < N; i++)
            mps_gate_1site(eng, quhits, N, i, U_re, U_im);
        for (int i = 0; i < N - 1; i++)
            mps_gate_2site(eng, quhits, N, i, G_re, G_im);
    }
    t1 = clock();
    svd_time = (double)(t1 - t0) / CLOCKS_PER_SEC;

    all_valid = 1;
    for (int i = 0; i < N; i++) {
        char label[32];
        snprintf(label, sizeof(label), "Site %d s3", i);
        mps_local_density(i, probs);
        all_valid &= check_density(label, probs, 6);
    }
    pass &= all_valid;
    printf("  → 3 sweeps (%d SVDs) in: %.3f s\n", 3 * (N-1), svd_time);

    mps_overlay_free();

    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  MPS χ=%d: %s                              ║\n",
           MPS_CHI, pass ? "ALL PASSED ✓" : "SOME FAILED ✗");
    printf("╚══════════════════════════════════════════════════════╝\n");

    return pass ? 0 : 1;
}
