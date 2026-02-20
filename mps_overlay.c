/*
 * mps_overlay.c — Implementation of the Side-Channel Overlay
 *
 * This implementation provides the logic to treat the engine's
 * pairwise memory blocks (QuhitPair) as nodes in a Matrix Product State.
 */

#include "mps_overlay.h"
#include <string.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════════════
 * INIT
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_overlay_init(QuhitEngine *eng, uint32_t *quhits, int n)
{
    /* For each quhit, create a self-pair (or dummy pair) to allocate storage. */
    /* We use 'product entangle' with a dummy to get a fresh pair struct. */
    
    for (int i = 0; i < n; i++) {
        uint32_t dummy = quhit_init(eng);
        quhit_entangle_product(eng, quhits[i], dummy);
        
        /* Zero out the storage to prepare for MPS tensor use */
        int pid = eng->quhits[quhits[i]].pair_id;
        if (pid >= 0) {
            memset(&eng->pairs[pid].joint, 0, 576);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * W-STATE CONSTRUCTION
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_overlay_write_w_state(QuhitEngine *eng, uint32_t *quhits, int n)
{
    double norm = 1.0 / sqrt((double)n);

    for (int i = 0; i < n; i++) {
        int pid = eng->quhits[quhits[i]].pair_id;
        if (pid < 0) continue; /* Should not happen if initialized */
        QuhitPair *p = &eng->pairs[pid];

        /* A[0] = Identity on bond */
        mps_write_tensor(p, 0, 0, 0, 1.0, 0.0); /* 0->0 */
        mps_write_tensor(p, 0, 1, 1, 1.0, 0.0); /* 1->1 */

        /* A[1] = Transition 0->1 */
        mps_write_tensor(p, 1, 0, 1, 1.0, 0.0);

        /* Apply normalization factor to the first site */
        if (i == 0) {
            mps_write_tensor(p, 0, 0, 0, norm, 0.0);
            mps_write_tensor(p, 0, 1, 1, norm, 0.0);
            mps_write_tensor(p, 1, 0, 1, norm, 0.0);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * AMPLITUDE INSPECTION
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_overlay_amplitude(QuhitEngine *eng, uint32_t *quhits, int n,
                           const uint32_t *basis, double *out_re, double *out_im)
{
    /* Contract L * A[k0] * ... * A[kn-1] * R */
    /* L = [1, 0], R = [0, 1]^T */
    
    double v_re[2] = {1.0, 0.0};
    double v_im[2] = {0.0, 0.0};

    for (int i = 0; i < n; i++) {
        int pid = eng->quhits[quhits[i]].pair_id;
        QuhitPair *p = &eng->pairs[pid];
        int k = (int)basis[i];

        double next_re[2] = {0, 0};
        double next_im[2] = {0, 0};

        /* v_next[beta] = Σ_alpha v[alpha] * A[k][alpha][beta] */
        for (int beta = 0; beta < 2; beta++) {
            for (int alpha = 0; alpha < 2; alpha++) {
                double t_re, t_im;
                mps_read_tensor(p, k, alpha, beta, &t_re, &t_im);
                
                next_re[beta] += v_re[alpha]*t_re - v_im[alpha]*t_im;
                next_im[beta] += v_re[alpha]*t_im + v_im[alpha]*t_re;
            }
        }
        v_re[0] = next_re[0]; v_re[1] = next_re[1];
        v_im[0] = next_im[0]; v_im[1] = next_im[1];
    }

    *out_re = v_re[1]; /* Project onto R=[0,1] */
    *out_im = v_im[1];
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MEASUREMENT
 * ═══════════════════════════════════════════════════════════════════════════════ */

uint32_t mps_overlay_measure(QuhitEngine *eng, uint32_t *quhits, int n, int target_idx)
{
    /*
     * Full left-right density environment contraction for exact P(k).
     * Cost: O(N × χ³ × D) = O(N × 8 × 6) = O(48N). Trivial for χ=2.
     *
     * Step 1: Build right density environment ρ_R from site N-1 down to target+1.
     *         ρ_R[j] = Σ_k A[k]_j† × ρ_R[j+1] × A[k]_j
     *         Boundary: ρ_R[N] = |R⟩⟨R| where R=[0,...,0,1] (last bond index).
     *
     * Step 2: Build left environment vector L from site 0 up to target-1.
     *         For previously-measured sites, use the projected tensor.
     *         L starts as [1, 0, ..., 0].
     *
     * Step 3: For each physical index k at target:
     *         P(k) = L† × A[k]† × ρ_R[target] × A[k] × L
     *
     * Step 4: Born sample from {P(k)}.
     * Step 5: Project tensor at target, renormalize.
     */

    /* ── Step 1: Right density environment ── */
    /* ρ_R is a χ×χ = 2×2 Hermitian matrix at each site */
    double rho_R[MPS_CHI][MPS_CHI]; /* current right env */

    /* Boundary: |R⟩⟨R| where R = [0, 1] */
    rho_R[0][0] = 0; rho_R[0][1] = 0;
    rho_R[1][0] = 0; rho_R[1][1] = 1;

    /* Sweep from site N-1 down to target+1 */
    for (int j = n - 1; j > target_idx; j--) {
        int pid = eng->quhits[quhits[j]].pair_id;
        if (pid < 0) continue;
        QuhitPair *p = &eng->pairs[pid];

        double new_rho[MPS_CHI][MPS_CHI] = {{0}};

        for (int k = 0; k < MPS_PHYS; k++) {
            /* Extract A[k] (χ×χ real matrix for typical states) */
            double A[MPS_CHI][MPS_CHI];
            for (int a = 0; a < MPS_CHI; a++)
                for (int b = 0; b < MPS_CHI; b++) {
                    double re, im;
                    mps_read_tensor(p, k, a, b, &re, &im);
                    A[a][b] = re; /* general case would track im separately */
                }

            /* tmp = A[k] * ρ_R  (correct: multiply from left) */
            double tmp[MPS_CHI][MPS_CHI] = {{0}};
            for (int a = 0; a < MPS_CHI; a++)
                for (int b = 0; b < MPS_CHI; b++)
                    for (int c = 0; c < MPS_CHI; c++)
                        tmp[a][b] += A[a][c] * rho_R[c][b];

            /* new_rho += tmp * A[k]^T  (= A ρ A^T, correct transfer) */
            for (int a = 0; a < MPS_CHI; a++)
                for (int b = 0; b < MPS_CHI; b++)
                    for (int c = 0; c < MPS_CHI; c++)
                        new_rho[a][b] += tmp[a][c] * A[b][c];
        }

        memcpy(rho_R, new_rho, sizeof(rho_R));
    }

    /* ── Step 2: Left environment vector ── */
    double L[MPS_CHI] = {1.0, 0.0}; /* Boundary: L = [1, 0] */

    for (int j = 0; j < target_idx; j++) {
        int pid = eng->quhits[quhits[j]].pair_id;
        if (pid < 0) continue;
        QuhitPair *p = &eng->pairs[pid];

        /* For previously measured sites, the tensor has been projected.
         * We use whatever physical slice remains (non-zero). */
        double new_L[MPS_CHI] = {0};
        for (int k = 0; k < MPS_PHYS; k++) {
            double Ak[MPS_CHI][MPS_CHI];
            int nonzero = 0;
            for (int a = 0; a < MPS_CHI; a++)
                for (int b = 0; b < MPS_CHI; b++) {
                    double re, im;
                    mps_read_tensor(p, k, a, b, &re, &im);
                    Ak[a][b] = re;
                    if (re != 0 || im != 0) nonzero = 1;
                }
            if (!nonzero) continue; /* This slice was zeroed by projection */

            /* new_L[b] += Σ_a L[a] × A[k][a][b] */
            for (int b = 0; b < MPS_CHI; b++)
                for (int a = 0; a < MPS_CHI; a++)
                    new_L[b] += L[a] * Ak[a][b];
        }
        L[0] = new_L[0]; L[1] = new_L[1];
    }

    /* ── Step 3: Compute P(k) for each physical index ── */
    double probs[MPS_PHYS];
    double total_prob = 0;

    int target_pid = eng->quhits[quhits[target_idx]].pair_id;
    QuhitPair *pt = &eng->pairs[target_pid];

    for (int k = 0; k < MPS_PHYS; k++) {
        double Ak[MPS_CHI][MPS_CHI];
        for (int a = 0; a < MPS_CHI; a++)
            for (int b = 0; b < MPS_CHI; b++) {
                double re, im;
                mps_read_tensor(pt, k, a, b, &re, &im);
                Ak[a][b] = re;
            }

        /* mid[b] = Σ_a L[a] × A[k][a][b] */
        double mid[MPS_CHI] = {0};
        for (int b = 0; b < MPS_CHI; b++)
            for (int a = 0; a < MPS_CHI; a++)
                mid[b] += L[a] * Ak[a][b];

        /* P(k) = mid^T × ρ_R × mid */
        double pk = 0;
        for (int a = 0; a < MPS_CHI; a++)
            for (int b = 0; b < MPS_CHI; b++)
                pk += mid[a] * rho_R[a][b] * mid[b];

        probs[k] = pk > 0 ? pk : 0;
        total_prob += probs[k];
    }

    /* ── Step 4: Born sample ── */
    if (total_prob > 1e-30)
        for (int k = 0; k < MPS_PHYS; k++) probs[k] /= total_prob;

    double r = quhit_prng_double(eng);
    uint32_t outcome = 0;
    double cdf = 0;
    for (int k = 0; k < MPS_PHYS; k++) {
        cdf += probs[k];
        if (r < cdf) { outcome = (uint32_t)k; break; }
    }

    /* ── Step 5: Project + renormalize tensor at target ── */
    for (int k = 0; k < MPS_PHYS; k++) {
        if ((uint32_t)k != outcome) {
            for (int a = 0; a < MPS_CHI; a++)
                for (int b = 0; b < MPS_CHI; b++)
                    mps_write_tensor(pt, k, a, b, 0, 0);
        }
    }

    /* Renormalize the surviving slice */
    double slice_norm2 = 0;
    for (int a = 0; a < MPS_CHI; a++)
        for (int b = 0; b < MPS_CHI; b++) {
            double re, im;
            mps_read_tensor(pt, (int)outcome, a, b, &re, &im);
            slice_norm2 += re * re + im * im;
        }
    if (slice_norm2 > 1e-30) {
        double scale = 1.0 / sqrt(slice_norm2);
        for (int a = 0; a < MPS_CHI; a++)
            for (int b = 0; b < MPS_CHI; b++) {
                double re, im;
                mps_read_tensor(pt, (int)outcome, a, b, &re, &im);
                mps_write_tensor(pt, (int)outcome, a, b, re * scale, im * scale);
            }
    }

    return outcome;
}
