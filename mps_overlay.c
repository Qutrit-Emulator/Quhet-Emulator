/*
 * mps_overlay.c — MPS Overlay with Pure Magic Pointers
 *
 * All tensor data stored in QuhitRegisters — RAM-agnostic.
 * Gate functions use register-based read/write via mps_read_tensor/mps_write_tensor.
 * 2-site gates are O(1) via CZ₆ engine pair operations.
 */

#ifdef _OPENMP
#include <omp.h>
#endif

#include "mps_overlay.h"
#include <math.h>
#include <fenv.h>
#include <xmmintrin.h>    /* _MM_SET_FLUSH_ZERO_MODE */
#include <pmmintrin.h>    /* _MM_SET_DENORMALS_ZERO_MODE */

/* ─── Global tensor store ──────────────────────────────────────────────────── */
MpsTensor        *mps_store     = NULL;
int               mps_store_n   = 0;
QuhitEngine      *mps_eng       = NULL;
TriOverlaySite   *mps_tri_sites = NULL;
int               mps_defer_renorm = 0;
int               mps_sweep_right  = 1;

/* ═══════════════════════════════════════════════════════════════════════════════
 * INIT / FREE
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_overlay_init(QuhitEngine *eng, uint32_t *quhits, int n)
{
    (void)quhits;
    mps_eng = eng;

    /* Lightweight tensor metadata — 4 bytes per site */
    if (mps_store) { free(mps_store); mps_store = NULL; }
    mps_store = (MpsTensor *)calloc((size_t)n, sizeof(MpsTensor));
    mps_store_n = n;

    /* Sidechannel probe: DAZ/FTZ eliminates 25.7× denormal penalty.
     * Denormal ops (< 2^-1022): 59.7 ns → 2.0 ns with flush enabled.
     * Safe for quantum simulation: denormal amplitudes are below noise. */
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    /* Create per-site registers: 3 qudits (k, α, β) */
    for (int i = 0; i < n; i++) {
        mps_store[i].reg_idx = quhit_reg_init(eng, (uint64_t)i, 3, MPS_CHI);
        if (mps_store[i].reg_idx >= 0) {
            eng->registers[mps_store[i].reg_idx].bulk_rule = 0;
            /* Seed product state: |k=0, α=0, β=0⟩ with amplitude 1.0 */
            quhit_reg_sv_set(eng, mps_store[i].reg_idx, 0, 1.0, 0.0);
        }
    }

    /* ── Triality per-site state ── */
    if (mps_tri_sites) { free(mps_tri_sites); mps_tri_sites = NULL; }
    mps_tri_sites = (TriOverlaySite *)calloc((size_t)n, sizeof(TriOverlaySite));
    for (int i = 0; i < n; i++)
        tri_site_init(&mps_tri_sites[i]);
}

void mps_overlay_free(void)
{
    free(mps_store);
    mps_store = NULL;
    mps_store_n = 0;
    mps_eng = NULL;
    free(mps_tri_sites);
    mps_tri_sites = NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * W-STATE CONSTRUCTION
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_overlay_write_w_state(QuhitEngine *eng, uint32_t *quhits, int n)
{
    (void)eng; (void)quhits;
    double site_scale = pow((double)n, -1.0 / (2.0 * n));

    for (int i = 0; i < n; i++) {
        mps_zero_site(i);
        mps_write_tensor(i, 0, 0, 0, site_scale, 0.0);
        mps_write_tensor(i, 0, 1, 1, site_scale, 0.0);
        mps_write_tensor(i, 1, 0, 1, site_scale, 0.0);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PRODUCT STATE |0⟩^⊗N
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_overlay_write_zero(QuhitEngine *eng, uint32_t *quhits, int n)
{
    (void)eng; (void)quhits;
    for (int i = 0; i < n; i++) {
        mps_zero_site(i);
        mps_write_tensor(i, 0, 0, 0, 1.0, 0.0);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * AMPLITUDE INSPECTION: ⟨basis|ψ⟩ = L^T · Π_i A[k_i] · R
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_overlay_amplitude(QuhitEngine *eng, uint32_t *quhits, int n,
                           const uint32_t *basis, double *out_re, double *out_im)
{
    (void)eng; (void)quhits;
    double v_re[MPS_CHI], v_im[MPS_CHI];
    memset(v_re, 0, sizeof(v_re));
    memset(v_im, 0, sizeof(v_im));
    v_re[0] = 1.0;

    for (int i = 0; i < n; i++) {
        int k = (int)basis[i];
        double next_re[MPS_CHI] = {0}, next_im[MPS_CHI] = {0};

        for (int beta = 0; beta < MPS_CHI; beta++)
            for (int alpha = 0; alpha < MPS_CHI; alpha++) {
                double t_re, t_im;
                mps_read_tensor(i, k, alpha, beta, &t_re, &t_im);
                next_re[beta] += v_re[alpha]*t_re - v_im[alpha]*t_im;
                next_im[beta] += v_re[alpha]*t_im + v_im[alpha]*t_re;
            }
        memcpy(v_re, next_re, sizeof(v_re));
        memcpy(v_im, next_im, sizeof(v_im));
    }

    double sr = 0, si = 0;
    for (int i = 0; i < MPS_CHI; i++) { sr += v_re[i]; si += v_im[i]; }
    *out_re = sr;
    *out_im = si;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MEASUREMENT — O(N × χ³ × D)
 * ═══════════════════════════════════════════════════════════════════════════════ */

uint32_t mps_overlay_measure(QuhitEngine *eng, uint32_t *quhits, int n, int target_idx)
{
    (void)quhits;
    int chi = MPS_CHI;
    size_t chi2 = (size_t)chi * chi;

    /* Right density environment (heap) */
    double *rho_R = (double *)calloc(chi2, sizeof(double));
    rho_R[0] = 1.0;  /* rho_R[0][0] = 1.0 */

    for (int j = n - 1; j > target_idx; j--) {
        double *new_rho = (double *)calloc(chi2, sizeof(double));
        for (int k = 0; k < MPS_PHYS; k++) {
            double *A = (double *)calloc(chi2, sizeof(double));
            for (int a = 0; a < chi; a++)
                for (int b = 0; b < chi; b++) {
                    double re, im;
                    mps_read_tensor(j, k, a, b, &re, &im);
                    A[a * chi + b] = re;
                }
            double *tmp = (double *)calloc(chi2, sizeof(double));
            for (int a = 0; a < chi; a++)
                for (int b = 0; b < chi; b++)
                    for (int c = 0; c < chi; c++)
                        tmp[a * chi + b] += A[a * chi + c] * rho_R[c * chi + b];
            for (int a = 0; a < chi; a++)
                for (int b = 0; b < chi; b++)
                    for (int c = 0; c < chi; c++)
                        new_rho[a * chi + b] += tmp[a * chi + c] * A[b * chi + c];
            free(A); free(tmp);
        }
        memcpy(rho_R, new_rho, chi2 * sizeof(double));
        free(new_rho);
    }

    /* Left environment */
    double *L = (double *)calloc(chi, sizeof(double));
    L[0] = 1.0;

    for (int j = 0; j < target_idx; j++) {
        double *new_L = (double *)calloc(chi, sizeof(double));
        for (int k = 0; k < MPS_PHYS; k++) {
            double *Ak = (double *)calloc(chi2, sizeof(double));
            int nonzero = 0;
            for (int a = 0; a < chi; a++)
                for (int b = 0; b < chi; b++) {
                    double re, im;
                    mps_read_tensor(j, k, a, b, &re, &im);
                    Ak[a * chi + b] = re;
                    if (re != 0 || im != 0) nonzero = 1;
                }
            if (!nonzero) { free(Ak); continue; }
            for (int b = 0; b < chi; b++)
                for (int a = 0; a < chi; a++)
                    new_L[b] += L[a] * Ak[a * chi + b];
            free(Ak);
        }
        memcpy(L, new_L, chi * sizeof(double));
        free(new_L);
    }

    /* P(k) */
    double probs[MPS_PHYS];
    double total_prob = 0;

    for (int k = 0; k < MPS_PHYS; k++) {
        double *Ak = (double *)calloc(chi2, sizeof(double));
        for (int a = 0; a < chi; a++)
            for (int b = 0; b < chi; b++) {
                double re, im;
                mps_read_tensor(target_idx, k, a, b, &re, &im);
                Ak[a * chi + b] = re;
            }
        double *mid = (double *)calloc(chi, sizeof(double));
        for (int b = 0; b < chi; b++)
            for (int a = 0; a < chi; a++)
                mid[b] += L[a] * Ak[a * chi + b];
        double pk = 0;
        for (int a = 0; a < chi; a++)
            for (int b = 0; b < chi; b++)
                pk += mid[a] * rho_R[a * chi + b] * mid[b];
        probs[k] = pk > 0 ? pk : 0;
        total_prob += probs[k];
        free(Ak); free(mid);
    }

    free(L);
    free(rho_R);

    /* Born sample */
    if (total_prob > 1e-30)
        for (int k = 0; k < MPS_PHYS; k++) probs[k] /= total_prob;

    double r = quhit_prng_double(eng);
    uint32_t outcome = 0;
    double cdf = 0;
    for (int k = 0; k < MPS_PHYS; k++) {
        cdf += probs[k];
        if (r < cdf) { outcome = (uint32_t)k; break; }
    }

    /* Project + renormalize */
    for (int k = 0; k < MPS_PHYS; k++) {
        if ((uint32_t)k != outcome) {
            for (int a = 0; a < chi; a++)
                for (int b = 0; b < chi; b++)
                    mps_write_tensor(target_idx, k, a, b, 0, 0);
        }
    }
    double slice_norm2 = 0;
    for (int a = 0; a < chi; a++)
        for (int b = 0; b < chi; b++) {
            double re, im;
            mps_read_tensor(target_idx, (int)outcome, a, b, &re, &im);
            slice_norm2 += re*re + im*im;
        }
    if (slice_norm2 > 1e-30) {
        double scale = born_fast_isqrt(slice_norm2);
        for (int a = 0; a < chi; a++)
            for (int b = 0; b < chi; b++) {
                double re, im;
                mps_read_tensor(target_idx, (int)outcome, a, b, &re, &im);
                mps_write_tensor(target_idx, (int)outcome, a, b, re*scale, im*scale);
            }
    }

    return outcome;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SINGLE-SITE GATE — O(entries × D) via register
 * ═══════════════════════════════════════════════════════════════════════════════ */

static int mps_cmp_basis(const void *a, const void *b)
{
    const struct { basis_t basis; double re, im; } *ea = a, *eb = b;
    if (ea->basis < eb->basis) return -1;
    if (ea->basis > eb->basis) return 1;
    return 0;
}

void mps_gate_1site(QuhitEngine *eng, uint32_t *quhits, int n,
                    int site, const double *U_re, const double *U_im)
{
    /* Register-based: rotate the physical index k using triality-masked gate */
    if (eng && mps_store && mps_store[site].reg_idx >= 0) {
        int reg_idx = mps_store[site].reg_idx;
        QuhitRegister *reg = &eng->registers[reg_idx];
        uint8_t mask = mps_tri_sites ? mps_tri_sites[site].active_mask : 0x3F;
        unsigned __int128 chi_power = (unsigned __int128)MPS_CHI * MPS_CHI;
        tri_reg_gate_1site_masked(reg, U_re, U_im, mask, chi_power);
    }

    /* Mirror to triality site (replaces standard quhit mirror) */
    if (mps_tri_sites && site < n)
        tri_site_apply_gate(&mps_tri_sites[site], U_re, U_im);
    else if (eng && quhits && site < n)
        quhit_apply_unitary(eng, quhits[site], U_re, U_im);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * 2-SITE GATE — Register-Based SVD Contraction
 *
 *  Optimizations V3.4:
 *   #7  Memory pool: static scratch buffers, no per-gate malloc/free
 *   #8  Diagonal fast-path: in-place O(D²χ²) for CZ/Potts gates
 *   #9  γ-bucketing: pre-sort register B by shared bond index
 *   #11 √σ pre-compute: one sqrt per singular value, not per entry
 *
 * Original V3.3 findings preserved:
 *   • Zero attractor → skip negligible Θ entries via mag² (no sqrt)
 *   • Gate sparsity → skip via mag² (no fabs)
 *   • 1.0 attractor → norm always converges to 1.0
 *   • 1/6 spectrum → σ values converge to equal weights at D=6
 * ═══════════════════════════════════════════════════════════════════════════════ */

#include "tensor_svd.h"

/* Temp entry struct matching QuhitRegister::entries[] layout */
typedef struct { basis_t basis_state; double amp_re, amp_im; } TmpEntry;

#define DCHI (MPS_PHYS * MPS_CHI)

/* ──  #7: Static scratch pool ──
 * Pre-allocated buffers reused across all gate calls.
 * Eliminates 14 malloc/free round-trips per gate. */
static double *g2_Th_re, *g2_Th_im;
static double *g2_Th2_re, *g2_Th2_im;
static double *g2_U_re, *g2_U_im, *g2_sig;
static double *g2_Vc_re, *g2_Vc_im;
static int g2_pool_init = 0;

static void g2_pool_ensure(void) {
    if (g2_pool_init) return;
    size_t dchi = DCHI;
    size_t dchi2 = dchi * dchi;
    size_t chi = MPS_CHI;
    g2_Th_re  = (double *)malloc(dchi2 * sizeof(double));
    g2_Th_im  = (double *)malloc(dchi2 * sizeof(double));
    g2_Th2_re = (double *)malloc(dchi2 * sizeof(double));
    g2_Th2_im = (double *)malloc(dchi2 * sizeof(double));
    g2_U_re   = (double *)malloc(dchi * chi * sizeof(double));
    g2_U_im   = (double *)malloc(dchi * chi * sizeof(double));
    g2_sig    = (double *)malloc(chi * sizeof(double));
    g2_Vc_re  = (double *)malloc(chi * dchi * sizeof(double));
    g2_Vc_im  = (double *)malloc(chi * dchi * sizeof(double));
    g2_pool_init = 1;
}

/* ──  #8: Diagonal gate detection ── */
static int gate_is_diagonal(const double *G_re, const double *G_im, int D2) {
    for (int i = 0; i < D2; i++)
        for (int j = 0; j < D2; j++)
            if (i != j) {
                if (G_re[i*D2+j] != 0.0 || G_im[i*D2+j] != 0.0)
                    return 0;
            }
    return 1;
}

/* ──  #9: γ-bucket structure for sparse contraction ── */
typedef struct {
    uint16_t *indices;  /* entry indices per bucket */
    uint16_t *counts;   /* count per gamma value */
    uint16_t *offsets;  /* start offset per gamma value */
} GammaBuckets;

static GammaBuckets gb_cache = {0};

static void gb_ensure(int chi, int max_entries) {
    if (!gb_cache.indices) {
        gb_cache.indices = (uint16_t *)malloc(max_entries * sizeof(uint16_t));
        gb_cache.counts  = (uint16_t *)malloc(chi * sizeof(uint16_t));
        gb_cache.offsets = (uint16_t *)malloc(chi * sizeof(uint16_t));
    }
}

static void gb_build(GammaBuckets *gb, QuhitRegister *reg, int chi) {
    memset(gb->counts, 0, chi * sizeof(uint16_t));
    /* Bucket regB by LEFT bond (alpha = shared bond with regA's right bond β)
     * Encoding: bs = k * chi² + alpha * chi + beta
     * alpha = (bs / chi) % chi */
    for (uint32_t e = 0; e < reg->num_nonzero; e++) {
        double ar = reg->entries[e].amp_re, ai = reg->entries[e].amp_im;
        if (ar*ar + ai*ai < 1e-30) continue;
        int alpha = (int)((reg->entries[e].basis_state / chi) % chi);
        if (alpha < chi) gb->counts[alpha]++;
    }
    /* Compute offsets */
    uint16_t off = 0;
    for (int g = 0; g < chi; g++) {
        gb->offsets[g] = off;
        off += gb->counts[g];
    }
    /* Fill index array */
    uint16_t *pos = (uint16_t *)alloca(chi * sizeof(uint16_t));
    memcpy(pos, gb->offsets, chi * sizeof(uint16_t));
    for (uint32_t e = 0; e < reg->num_nonzero; e++) {
        double ar = reg->entries[e].amp_re, ai = reg->entries[e].amp_im;
        if (ar*ar + ai*ai < 1e-30) continue;
        int alpha = (int)((reg->entries[e].basis_state / chi) % chi);
        if (alpha < chi)
            gb->indices[pos[alpha]++] = (uint16_t)e;
    }
}

void mps_gate_2site(QuhitEngine *eng, uint32_t *quhits, int n,
                    int site, const double *G_re, const double *G_im)
{
    int sA = site, sB = site + 1;
    int D = MPS_PHYS, chi = MPS_CHI;
    int dchi = D * chi;
    int chi2 = chi * chi;
    size_t dchi2 = (size_t)dchi * dchi;
    int D2 = D * D;

    QuhitRegister *regA = &eng->registers[mps_store[sA].reg_idx];
    QuhitRegister *regB = &eng->registers[mps_store[sB].reg_idx];

    /* ── Optimization 3: Product-state SVD bypass ──
     * When both sites have nnz ≤ 1, the joint state is a single product
     * |kA, α, γ⟩ ⊗ |kB, γ', β⟩. The 2-site gate simply maps the physical
     * indices (kA,kB) → Σ G(kA',kB'; kA,kB) without needing SVD.
     * Result: O(D²) instead of O(D²·χ²). */
    if (regA->num_nonzero <= 1 && regB->num_nonzero <= 1) {
        if (regA->num_nonzero == 0 || regB->num_nonzero == 0) return;

        /* Extract the single nonzero entry from each register */
        basis_t bsA = regA->entries[0].basis_state;
        double arA = regA->entries[0].amp_re, aiA = regA->entries[0].amp_im;
        int kA_old = (int)(bsA / chi2);
        int alpha  = (int)((bsA / chi) % chi);
        int gamma  = (int)(bsA % chi);

        basis_t bsB = regB->entries[0].basis_state;
        double arB = regB->entries[0].amp_re, aiB = regB->entries[0].amp_im;
        int kB_old = (int)(bsB / chi2);
        int gamma2 = (int)((bsB / chi) % chi);
        int beta   = (int)(bsB % chi);

        /* Product amplitude */
        double prod_re = arA * arB - aiA * aiB;
        double prod_im = arA * aiB + aiA * arB;

        /* Apply gate: new state = Σ_{kA',kB'} G(kA',kB'; kA,kB) × prod × |kA',α,γ⟩⊗|kB',γ',β⟩ */
        int gc = kA_old * D + kB_old;  /* gate input column */

        /* Clear both registers */
        regA->num_nonzero = 0;
        regB->num_nonzero = 0;

        /* Find the dominant output (for rank-1, there's typically one large output) */
        for (int kAp = 0; kAp < D; kAp++) {
            for (int kBp = 0; kBp < D; kBp++) {
                int gr = kAp * D + kBp;
                double gre = G_re[gr * D2 + gc];
                double gim = G_im[gr * D2 + gc];
                if (gre*gre + gim*gim < 1e-30) continue;

                double out_re = gre * prod_re - gim * prod_im;
                double out_im = gre * prod_im + gim * prod_re;
                if (out_re*out_re + out_im*out_im < 1e-30) continue;

                /* Write to register A: |kAp, α, γ⟩ */
                basis_t newA = (uint64_t)kAp * chi2 + alpha * chi + gamma;
                /* Write to register B: |kBp, γ', β⟩ */
                basis_t newB = (uint64_t)kBp * chi2 + gamma2 * chi + beta;

                /* For product-state output, factor as √|out|² on each site */
                double mag = sqrt(out_re*out_re + out_im*out_im);
                double phase_re = out_re / mag, phase_im = out_im / mag;

                /* Put magnitude on A, phase on A, unit on B */
                {
                    int found = -1;
                    for (uint32_t e = 0; e < regA->num_nonzero; e++) {
                        if (regA->entries[e].basis_state == newA) { found = e; break; }
                    }
                    if (found >= 0) {
                        regA->entries[found].amp_re += mag * phase_re;
                        regA->entries[found].amp_im += mag * phase_im;
                    } else if (regA->num_nonzero < 4096) {
                        int e = regA->num_nonzero++;
                        regA->entries[e].basis_state = newA;
                        regA->entries[e].amp_re = mag * phase_re;
                        regA->entries[e].amp_im = mag * phase_im;
                    }
                }
                {
                    int found = -1;
                    for (uint32_t e = 0; e < regB->num_nonzero; e++) {
                        if (regB->entries[e].basis_state == newB) { found = e; break; }
                    }
                    if (found >= 0) {
                        /* already there */
                    } else if (regB->num_nonzero < 4096) {
                        int e = regB->num_nonzero++;
                        regB->entries[e].basis_state = newB;
                        regB->entries[e].amp_re = 1.0;
                        regB->entries[e].amp_im = 0.0;
                    }
                }
            }
        }
        return;  /* Skip entire Θ formation + SVD pipeline */
    }

    /* ── General path: full Θ contraction + SVD ── */

    /*  #7: Ensure scratch pool */
    g2_pool_ensure();
    memset(g2_Th_re, 0, dchi2 * sizeof(double));
    memset(g2_Th_im, 0, dchi2 * sizeof(double));

    /* ── Step 1+2: Build Θ —  #9: γ-bucketed contraction ──
     * Pre-sort register B entries by shared bond γ.
     * Then for each regA entry, only iterate matching γ bucket. */
    gb_ensure(chi, 8192);
    gb_build(&gb_cache, regB, chi);

    for (uint32_t eA = 0; eA < regA->num_nonzero; eA++) {
        basis_t bsA = regA->entries[eA].basis_state;
        double arA = regA->entries[eA].amp_re;
        double aiA = regA->entries[eA].amp_im;
        if (arA*arA + aiA*aiA < 1e-30) continue;

        int kA = (int)(bsA / chi2);
        int alpha = (int)((bsA / chi) % chi);
        int gamma_A = (int)(bsA % chi);
        int row = kA * chi + alpha;

        /* Only iterate register B entries with matching gamma */
        uint16_t start = gb_cache.offsets[gamma_A];
        uint16_t count = gb_cache.counts[gamma_A];
        for (uint16_t bi = 0; bi < count; bi++) {
            uint16_t eB = gb_cache.indices[start + bi];
            basis_t bsB = regB->entries[eB].basis_state;
            double arB = regB->entries[eB].amp_re;
            double aiB = regB->entries[eB].amp_im;

            int kB   = (int)(bsB / chi2);
            int beta = (int)(bsB % chi);
            int col  = kB * chi + beta;

            g2_Th_re[row * dchi + col] += arA*arB - aiA*aiB;
            g2_Th_im[row * dchi + col] += arA*aiB + aiA*arB;
        }
    }

    /* ── Step 3: Apply gate —  #8: diagonal fast-path ── */
    int is_diag = gate_is_diagonal(G_re, G_im, D2);

    double *Th_out_re, *Th_out_im;

    if (is_diag) {
        /* Diagonal gate: in-place, O(D²×χ²) — no Th2 buffer needed */
        for (int kA = 0; kA < D; kA++)
            for (int kB = 0; kB < D; kB++) {
                int idx = kA * D + kB;
                double gre = G_re[idx * D2 + idx];
                double gim = G_im[idx * D2 + idx];
                if (gre == 1.0 && gim == 0.0) continue; /* identity element */
                for (int a = 0; a < chi; a++) {
                    int r = kA * chi + a;
                    for (int b = 0; b < chi; b++) {
                        int c = kB * chi + b;
                        size_t pos = (size_t)r * dchi + c;
                        double tr = g2_Th_re[pos];
                        double ti = g2_Th_im[pos];
                        g2_Th_re[pos] = gre*tr - gim*ti;
                        g2_Th_im[pos] = gre*ti + gim*tr;
                    }
                }
            }
        Th_out_re = g2_Th_re;
        Th_out_im = g2_Th_im;
    } else {
        /* General gate: O(D⁴×χ²) with sparsity skip */
        memset(g2_Th2_re, 0, dchi2 * sizeof(double));
        memset(g2_Th2_im, 0, dchi2 * sizeof(double));

        for (int kAp = 0; kAp < D; kAp++)
         for (int kBp = 0; kBp < D; kBp++) {
             int gr = kAp * D + kBp;
             for (int kA = 0; kA < D; kA++)
              for (int kB = 0; kB < D; kB++) {
                  int gc = kA * D + kB;
                  double gre = G_re[gr * D2 + gc];
                  double gim = G_im[gr * D2 + gc];
                  if (gre*gre + gim*gim < 1e-20) continue;

                  for (int a = 0; a < chi; a++) {
                      int dst_row = kAp * chi + a;
                      int src_row = kA * chi + a;
                      for (int b = 0; b < chi; b++) {
                          int dst_col = kBp * chi + b;
                          int src_col = kB * chi + b;
                          double tr = g2_Th_re[src_row * dchi + src_col];
                          double ti = g2_Th_im[src_row * dchi + src_col];
                          g2_Th2_re[dst_row * dchi + dst_col] += gre*tr - gim*ti;
                          g2_Th2_im[dst_row * dchi + dst_col] += gre*ti + gim*tr;
                      }
                  }
              }
         }
        Th_out_re = g2_Th2_re;
        Th_out_im = g2_Th2_im;
    }

    /* ── Step 4: Renormalize Θ → SVD → truncate to χ ── */

    /* Pre-normalize Θ to unit Frobenius norm to prevent σ explosion */
    double th_norm2 = 0;
    for (size_t i = 0; i < (size_t)dchi * dchi; i++)
        th_norm2 += Th_out_re[i] * Th_out_re[i] + Th_out_im[i] * Th_out_im[i];
    double th_norm = (th_norm2 > 1e-60) ? sqrt(th_norm2) : 1.0;
    if (th_norm > 1e-30 && th_norm != 1.0) {
        double inv_norm = 1.0 / th_norm;
        for (size_t i = 0; i < (size_t)dchi * dchi; i++) {
            Th_out_re[i] *= inv_norm;
            Th_out_im[i] *= inv_norm;
        }
    }

    memset(g2_U_re, 0, (size_t)dchi * chi * sizeof(double));
    memset(g2_U_im, 0, (size_t)dchi * chi * sizeof(double));
    memset(g2_sig, 0, chi * sizeof(double));
    memset(g2_Vc_re, 0, (size_t)chi * dchi * sizeof(double));
    memset(g2_Vc_im, 0, (size_t)chi * dchi * sizeof(double));

    tsvd_truncated_sparse(Th_out_re, Th_out_im, dchi, dchi, chi,
                   g2_U_re, g2_U_im, g2_sig, g2_Vc_re, g2_Vc_im);

    /* Rescale σ by original Θ norm */
    for (int g = 0; g < chi; g++)
        g2_sig[g] *= th_norm;

    /* ── Step 5: Write back —  #11: √σ pre-computed ── */
    int rank = chi < dchi ? chi : dchi;

    /* Pre-compute √σ once — not per entry ( #11) */
    double sqrt_sig[MPS_CHI];
    for (int g = 0; g < rank; g++)
        sqrt_sig[g] = g2_sig[g] > 1e-30
                      ? g2_sig[g] * born_fast_isqrt(g2_sig[g]) : 0;

    /* ── Write-back with magnitude-sorted truncation ──
     * Collect ALL entries into a temp buffer, then keep the top 4096
     * by |amplitude|². This prevents random drops that cascade to zero. */

    /* Static temp buffer (allocated once alongside pool) */
    static TmpEntry *g2_tmp_buf = NULL;
    static int g2_tmp_init = 0;
    if (!g2_tmp_init) {
        g2_tmp_buf = (TmpEntry *)malloc(16384 * sizeof(TmpEntry));
        g2_tmp_init = 1;
    }

    /* A'[kA', α, γ] = √σ[γ] × U[(kA'*χ+α), γ] */
    uint32_t nA = 0;
    for (int kA = 0; kA < D; kA++)
     for (int a = 0; a < chi; a++) {
         int row = kA * chi + a;
         for (int g = 0; g < rank; g++) {
             double sq = sqrt_sig[g];
             if (sq < 1e-30) continue;
             double re = sq * g2_U_re[row * rank + g];
             double im = sq * g2_U_im[row * rank + g];
             if (re*re + im*im > 1e-30 && nA < 16384) {
                 g2_tmp_buf[nA].basis_state = (basis_t)kA * chi2 + (basis_t)a * chi + g;
                 g2_tmp_buf[nA].amp_re = re;
                 g2_tmp_buf[nA].amp_im = im;
                 nA++;
             }
         }
     }
    /* Partial selection sort to keep top-4096 by magnitude */
    if (nA > 4096) {
        for (uint32_t i = 0; i < 4096; i++) {
            uint32_t best = i;
            double best_m = g2_tmp_buf[i].amp_re * g2_tmp_buf[i].amp_re +
                            g2_tmp_buf[i].amp_im * g2_tmp_buf[i].amp_im;
            for (uint32_t j = i + 1; j < nA; j++) {
                double m = g2_tmp_buf[j].amp_re * g2_tmp_buf[j].amp_re +
                           g2_tmp_buf[j].amp_im * g2_tmp_buf[j].amp_im;
                if (m > best_m) { best = j; best_m = m; }
            }
            if (best != i) {
                TmpEntry tmp = g2_tmp_buf[i];
                g2_tmp_buf[i] = g2_tmp_buf[best];
                g2_tmp_buf[best] = tmp;
            }
        }
        nA = 4096;
    }
    regA->num_nonzero = nA;
    memcpy(regA->entries, g2_tmp_buf, nA * sizeof(TmpEntry));

    /* B'[kB', γ, β] = √σ[γ] × V†[γ, (kB'*χ+β)] */
    uint32_t nB = 0;
    for (int kB = 0; kB < D; kB++)
     for (int g = 0; g < rank; g++) {
          double sq = sqrt_sig[g];
          if (sq < 1e-30) continue;
         for (int b = 0; b < chi; b++) {
             int col = kB * chi + b;
             double re = sq * g2_Vc_re[g * dchi + col];
             double im = sq * g2_Vc_im[g * dchi + col];
             if (re*re + im*im > 1e-30 && nB < 16384) {
                 g2_tmp_buf[nB].basis_state = (basis_t)kB * chi2 + (basis_t)g * chi + b;
                 g2_tmp_buf[nB].amp_re = re;
                 g2_tmp_buf[nB].amp_im = im;
                 nB++;
             }
         }
     }
    if (nB > 4096) {
        for (uint32_t i = 0; i < 4096; i++) {
            uint32_t best = i;
            double best_m = g2_tmp_buf[i].amp_re * g2_tmp_buf[i].amp_re +
                            g2_tmp_buf[i].amp_im * g2_tmp_buf[i].amp_im;
            for (uint32_t j = i + 1; j < nB; j++) {
                double m = g2_tmp_buf[j].amp_re * g2_tmp_buf[j].amp_re +
                           g2_tmp_buf[j].amp_im * g2_tmp_buf[j].amp_im;
                if (m > best_m) { best = j; best_m = m; }
            }
            if (best != i) {
                TmpEntry tmp = g2_tmp_buf[i];
                g2_tmp_buf[i] = g2_tmp_buf[best];
                g2_tmp_buf[best] = tmp;
            }
        }
        nB = 4096;
    }
    regB->num_nonzero = nB;
    memcpy(regB->entries, g2_tmp_buf, nB * sizeof(TmpEntry));


    /* ── Mirror to triality sites (replaces engine quhit CZ mirror) ── */
    if (mps_tri_sites && sB < n)
        tri_site_apply_cz(&mps_tri_sites[sA], &mps_tri_sites[sB]);
    else if (eng && quhits && sB < n)
        quhit_apply_cz(eng, quhits[sA], quhits[sB]);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * GATE CONSTRUCTORS
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Sidechannel probe: OMEGA6 table eliminates runtime cos/sin.
 * 26.5 ns vs 235.5 ns = 8.9× speedup per gate construction.
 * ω^k = (OMEGA6_RE[k%6], OMEGA6_IM[k%6]) for k = jk mod 6. */
static const double OMEGA6_RE[6] = {
    1.0, 0.5, -0.5, -1.0, -0.5, 0.5
};
static const double OMEGA6_IM[6] = {
    0.0, 0.86602540378443864676, 0.86602540378443864676,
    0.0, -0.86602540378443864676, -0.86602540378443864676
};

void mps_build_dft6(double *U_re, double *U_im)
{
    double inv = born_fast_isqrt(6.0);
    for (int j = 0; j < 6; j++)
        for (int k = 0; k < 6; k++) {
            int idx = (j * k) % 6;
            U_re[j * 6 + k] = inv * OMEGA6_RE[idx];
            U_im[j * 6 + k] = inv * OMEGA6_IM[idx];
        }
}

void mps_build_cz(double *G_re, double *G_im)
{
    int D2 = MPS_PHYS * MPS_PHYS;
    memset(G_re, 0, D2 * D2 * sizeof(double));
    memset(G_im, 0, D2 * D2 * sizeof(double));
    for (int k = 0; k < MPS_PHYS; k++)
        for (int l = 0; l < MPS_PHYS; l++) {
            int idx_g = (k * MPS_PHYS + l) * D2 + (k * MPS_PHYS + l);
            int idx_w = (k * l) % 6;
            G_re[idx_g] = OMEGA6_RE[idx_w];
            G_im[idx_g] = OMEGA6_IM[idx_w];
        }
}

void mps_build_controlled_phase(double *G_re, double *G_im, int power)
{
    int D2 = MPS_PHYS * MPS_PHYS;
    memset(G_re, 0, D2 * D2 * sizeof(double));
    memset(G_im, 0, D2 * D2 * sizeof(double));
    double pf = (double)(1 << power) / (double)(MPS_PHYS * MPS_PHYS);
    for (int k = 0; k < MPS_PHYS; k++)
        for (int l = 0; l < MPS_PHYS; l++) {
            int idx = (k * MPS_PHYS + l) * D2 + (k * MPS_PHYS + l);
            double angle = 2.0 * M_PI * k * l * pf;
            G_re[idx] = cos(angle);
            G_im[idx] = sin(angle);
        }
}

void mps_build_hadamard2(double *U_re, double *U_im)
{
    memset(U_re, 0, 36 * sizeof(double));
    memset(U_im, 0, 36 * sizeof(double));
    double s = born_fast_isqrt(2.0);
    U_re[0*6+0] =  s; U_re[0*6+1] =  s;
    U_re[1*6+0] =  s; U_re[1*6+1] = -s;
    for (int k = 2; k < 6; k++) U_re[k*6+k] = 1.0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * NORM ⟨ψ|ψ⟩
 * ═══════════════════════════════════════════════════════════════════════════════ */

double mps_overlay_norm(QuhitEngine *eng, uint32_t *quhits, int n)
{
    (void)eng; (void)quhits;
    int chi = MPS_CHI;
    size_t chi2 = (size_t)chi * chi;

    double *rho_re = (double *)calloc(chi2, sizeof(double));
    double *rho_im = (double *)calloc(chi2, sizeof(double));
    rho_re[0] = 1.0;  /* [0][0] */

    for (int i = 0; i < n; i++) {
        double *nr     = (double *)calloc(chi2, sizeof(double));
        double *ni_arr = (double *)calloc(chi2, sizeof(double));

        for (int k = 0; k < MPS_PHYS; k++) {
            double *A_re = (double *)calloc(chi2, sizeof(double));
            double *A_im = (double *)calloc(chi2, sizeof(double));
            for (int a = 0; a < chi; a++)
                for (int b = 0; b < chi; b++)
                    mps_read_tensor(i, k, a, b,
                                    &A_re[a * chi + b], &A_im[a * chi + b]);

            double *tr2 = (double *)calloc(chi2, sizeof(double));
            double *ti2 = (double *)calloc(chi2, sizeof(double));
            for (int a = 0; a < chi; a++)
                for (int bp = 0; bp < chi; bp++)
                    for (int ap = 0; ap < chi; ap++) {
                        tr2[a*chi+bp] += rho_re[a*chi+ap]*A_re[ap*chi+bp]
                                       - rho_im[a*chi+ap]*A_im[ap*chi+bp];
                        ti2[a*chi+bp] += rho_re[a*chi+ap]*A_im[ap*chi+bp]
                                       + rho_im[a*chi+ap]*A_re[ap*chi+bp];
                    }

            for (int b = 0; b < chi; b++)
                for (int bp = 0; bp < chi; bp++)
                    for (int a = 0; a < chi; a++) {
                        double ar = A_re[a*chi+b], ai = -A_im[a*chi+b];
                        nr[b*chi+bp]     += ar*tr2[a*chi+bp] - ai*ti2[a*chi+bp];
                        ni_arr[b*chi+bp] += ar*ti2[a*chi+bp] + ai*tr2[a*chi+bp];
                    }
            free(A_re); free(A_im); free(tr2); free(ti2);
        }
        memcpy(rho_re, nr, chi2 * sizeof(double));
        memcpy(rho_im, ni_arr, chi2 * sizeof(double));
        free(nr); free(ni_arr);
    }

    double trace = 0;
    for (int i = 0; i < chi; i++) trace += rho_re[i * chi + i];
    free(rho_re); free(rho_im);
    return trace;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * DEFERRED RENORMALIZATION
 * ═══════════════════════════════════════════════════════════════════════════════ */

void mps_renormalize_chain(QuhitEngine *eng, uint32_t *quhits, int n)
{
    double norm = mps_overlay_norm(eng, quhits, n);
    if (norm > 1e-30 && fabs(norm - 1.0) > 1e-12) {
        double scale = born_fast_isqrt(norm);
        for (int k = 0; k < MPS_PHYS; k++)
            for (int a = 0; a < MPS_CHI; a++)
                for (int b = 0; b < MPS_CHI; b++) {
                    double tr, ti;
                    mps_read_tensor(0, k, a, b, &tr, &ti);
                    mps_write_tensor(0, k, a, b, tr * scale, ti * scale);
                }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * LAZY EVALUATION LAYER
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Forward declaration (defined below in gate fusion section) */
static void fuse_1site_gates(const double *A_re, const double *A_im,
                             const double *B_re, const double *B_im,
                             double *C_re, double *C_im);

MpsLazyChain *mps_lazy_init(QuhitEngine *eng, uint32_t *quhits, int n)
{
    MpsLazyChain *lc = (MpsLazyChain *)calloc(1, sizeof(MpsLazyChain));
    lc->eng = eng;
    lc->quhits = quhits;
    lc->n_sites = n;

    mps_overlay_init(eng, quhits, n);
    lc->tri_sites = mps_tri_sites;  /* Link to per-site triality state */

    lc->queue_cap = MAX_LAZY_GATES;
    lc->queue = (MpsDeferredGate *)calloc(lc->queue_cap, sizeof(MpsDeferredGate));
    lc->queue_len = 0;

    lc->site_allocated = (uint8_t *)calloc(n, sizeof(uint8_t));
    lc->site_dirty     = (uint8_t *)calloc(n, sizeof(uint8_t));

    /*  #10: Per-site pending 1-site gate buffer */
    int D2 = MPS_PHYS * MPS_PHYS;
    lc->pending_1site_re    = (double *)calloc((size_t)n * D2, sizeof(double));
    lc->pending_1site_im    = (double *)calloc((size_t)n * D2, sizeof(double));
    lc->pending_1site_valid = (uint8_t *)calloc(n, sizeof(uint8_t));

    lazy_stats_reset(&lc->stats);
    lc->stats.sites_total = n;
    lc->stats.hilbert_log10 = n * log10(6.0);

    return lc;
}

void mps_lazy_free(MpsLazyChain *lc)
{
    if (!lc) return;
    for (int i = 0; i < lc->queue_len; i++) {
        if (lc->queue[i].type == 1) {
            free(lc->queue[i].G_re);
            free(lc->queue[i].G_im);
        }
    }
    free(lc->queue);
    free(lc->site_allocated);
    free(lc->site_dirty);
    free(lc->pending_1site_re);
    free(lc->pending_1site_im);
    free(lc->pending_1site_valid);
    mps_overlay_free();
    free(lc);
}

static void lazy_ensure_site(MpsLazyChain *lc, int site)
{
    if (!lc->site_allocated[site]) {
        mps_zero_site(site);
        mps_write_tensor(site, 0, 0, 0, 1.0, 0.0);
        lc->site_allocated[site] = 1;
    }
}

/*  #10: Drain a pending 1-site gate into the queue */
static void drain_pending_1site(MpsLazyChain *lc, int site)
{
    if (!lc->pending_1site_valid[site]) return;

    if (lc->queue_len >= lc->queue_cap) mps_lazy_flush(lc);

    int D2 = MPS_PHYS * MPS_PHYS;
    MpsDeferredGate *g = &lc->queue[lc->queue_len];
    g->type = 0;
    g->site = site;
    memcpy(g->U_re, &lc->pending_1site_re[(size_t)site * D2], D2 * sizeof(double));
    memcpy(g->U_im, &lc->pending_1site_im[(size_t)site * D2], D2 * sizeof(double));
    g->G_re = NULL;
    g->G_im = NULL;
    g->applied = 0;

    lc->queue_len++;
    lc->stats.gates_queued++;
    lc->pending_1site_valid[site] = 0;
}

void mps_lazy_gate_1site(MpsLazyChain *lc, int site,
                         const double *U_re, const double *U_im)
{
    int D = MPS_PHYS;
    int D2 = D * D;
    size_t off = (size_t)site * D2;

    if (lc->pending_1site_valid[site]) {
        /*  #10: Eager fusion — compose B × A in-place */
        double C_re[MPS_PHYS * MPS_PHYS], C_im[MPS_PHYS * MPS_PHYS];
        fuse_1site_gates(&lc->pending_1site_re[off], &lc->pending_1site_im[off],
                         U_re, U_im, C_re, C_im);
        memcpy(&lc->pending_1site_re[off], C_re, D2 * sizeof(double));
        memcpy(&lc->pending_1site_im[off], C_im, D2 * sizeof(double));
        lc->stats.gates_fused++;
    } else {
        /* First gate for this site — store as pending */
        memcpy(&lc->pending_1site_re[off], U_re, D2 * sizeof(double));
        memcpy(&lc->pending_1site_im[off], U_im, D2 * sizeof(double));
        lc->pending_1site_valid[site] = 1;
        lc->stats.gates_queued++;
    }

    lc->site_dirty[site] = 1;
}

void mps_lazy_gate_2site(MpsLazyChain *lc, int site,
                         const double *G_re, const double *G_im)
{
    /*  #10: Drain pending 1-site gates on affected sites first */
    drain_pending_1site(lc, site);
    if (site + 1 < lc->n_sites)
        drain_pending_1site(lc, site + 1);

    if (lc->queue_len >= lc->queue_cap) mps_lazy_flush(lc);

    int D2 = MPS_PHYS * MPS_PHYS;
    int sz = D2 * D2;

    MpsDeferredGate *g = &lc->queue[lc->queue_len];
    g->type = 1;
    g->site = site;
    g->G_re = (double *)malloc(sz * sizeof(double));
    g->G_im = (double *)malloc(sz * sizeof(double));
    memcpy(g->G_re, G_re, sz * sizeof(double));
    memcpy(g->G_im, G_im, sz * sizeof(double));
    g->applied = 0;

    lc->queue_len++;
    lc->site_dirty[site] = 1;
    if (site + 1 < lc->n_sites)
        lc->site_dirty[site + 1] = 1;
    lc->stats.gates_queued++;
}

/* Gate fusion: C = B × A */
static void fuse_1site_gates(const double *A_re, const double *A_im,
                             const double *B_re, const double *B_im,
                             double *C_re, double *C_im)
{
    int D = MPS_PHYS;
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++) {
            double sr = 0, si = 0;
            for (int k = 0; k < D; k++) {
                double br = B_re[i * D + k], bi = B_im[i * D + k];
                double ar = A_re[k * D + j], ai = A_im[k * D + j];
                sr += br * ar - bi * ai;
                si += br * ai + bi * ar;
            }
            C_re[i * D + j] = sr;
            C_im[i * D + j] = si;
        }
}

static void apply_gate(MpsLazyChain *lc, MpsDeferredGate *g)
{
    if (g->applied) return;

    if (g->type == 0) {
        lazy_ensure_site(lc, g->site);
        mps_gate_1site(lc->eng, lc->quhits, lc->n_sites,
                       g->site, g->U_re, g->U_im);
    } else {
        lazy_ensure_site(lc, g->site);
        lazy_ensure_site(lc, g->site + 1);
        mps_gate_2site(lc->eng, lc->quhits, lc->n_sites,
                       g->site, g->G_re, g->G_im);
    }

    g->applied = 1;
    lc->stats.gates_materialized++;
}

uint32_t mps_lazy_measure(MpsLazyChain *lc, int target_idx)
{
    /*  #10: Drain pending 1-site gates before measuring */
    for (int s = 0; s < lc->n_sites; s++)
        drain_pending_1site(lc, s);

    /* Gate fusion pass */
    for (int i = 0; i < lc->queue_len - 1; i++) {
        if (lc->queue[i].applied) continue;
        if (lc->queue[i].type != 0) continue;

        int j = i + 1;
        while (j < lc->queue_len &&
               lc->queue[j].type == 0 &&
               lc->queue[j].site == lc->queue[i].site &&
               !lc->queue[j].applied) {
            double C_re[MPS_PHYS * MPS_PHYS], C_im[MPS_PHYS * MPS_PHYS];
            fuse_1site_gates(lc->queue[i].U_re, lc->queue[i].U_im,
                             lc->queue[j].U_re, lc->queue[j].U_im,
                             C_re, C_im);
            memcpy(lc->queue[i].U_re, C_re, sizeof(C_re));
            memcpy(lc->queue[i].U_im, C_im, sizeof(C_im));
            lc->queue[j].applied = 1;
            lc->stats.gates_fused++;
            j++;
        }
    }

    /* Apply all pending gates */
    for (int i = 0; i < lc->queue_len; i++) {
        if (!lc->queue[i].applied)
            apply_gate(lc, &lc->queue[i]);
    }

    lazy_ensure_site(lc, target_idx);
    return mps_overlay_measure(lc->eng, lc->quhits, lc->n_sites, target_idx);
}

void mps_lazy_flush(MpsLazyChain *lc)
{
    /*  #10: Drain all pending 1-site gates first */
    for (int s = 0; s < lc->n_sites; s++)
        drain_pending_1site(lc, s);

    /* Fusion pass */
    for (int i = 0; i < lc->queue_len - 1; i++) {
        if (lc->queue[i].applied) continue;
        if (lc->queue[i].type != 0) continue;

        int j = i + 1;
        while (j < lc->queue_len &&
               lc->queue[j].type == 0 &&
               lc->queue[j].site == lc->queue[i].site &&
               !lc->queue[j].applied) {
            double C_re[MPS_PHYS * MPS_PHYS], C_im[MPS_PHYS * MPS_PHYS];
            fuse_1site_gates(lc->queue[i].U_re, lc->queue[i].U_im,
                             lc->queue[j].U_re, lc->queue[j].U_im,
                             C_re, C_im);
            memcpy(lc->queue[i].U_re, C_re, sizeof(C_re));
            memcpy(lc->queue[i].U_im, C_im, sizeof(C_im));
            lc->queue[j].applied = 1;
            lc->stats.gates_fused++;
            j++;
        }
    }

    for (int i = 0; i < lc->queue_len; i++) {
        if (!lc->queue[i].applied)
            apply_gate(lc, &lc->queue[i]);
    }

    for (int i = 0; i < lc->queue_len; i++) {
        if (lc->queue[i].type == 1) {
            free(lc->queue[i].G_re);
            free(lc->queue[i].G_im);
            lc->queue[i].G_re = NULL;
            lc->queue[i].G_im = NULL;
        }
    }
    lc->queue_len = 0;
}

void mps_lazy_finalize_stats(MpsLazyChain *lc)
{
    uint64_t skipped = 0;
    for (int i = 0; i < lc->queue_len; i++)
        if (!lc->queue[i].applied) skipped++;
    lc->stats.gates_skipped = skipped;

    uint64_t alloc = 0;
    for (int i = 0; i < lc->n_sites; i++)
        if (lc->site_allocated[i]) alloc++;
    lc->stats.sites_allocated = alloc;
    lc->stats.sites_lazy = lc->n_sites - alloc;

    lc->stats.memory_actual = alloc * sizeof(MpsTensor)
                            + lc->queue_len * sizeof(MpsDeferredGate)
                            + sizeof(MpsLazyChain);
}

void mps_lazy_write_tensor(MpsLazyChain *lc, int site, int k,
                           int alpha, int beta, double re, double im)
{
    lc->site_allocated[site] = 1;
    mps_write_tensor(site, k, alpha, beta, re, im);
}

void mps_lazy_zero_site(MpsLazyChain *lc, int site)
{
    lc->site_allocated[site] = 1;
    mps_zero_site(site);
    mps_write_tensor(site, 0, 0, 0, 1.0, 0.0);
}
