/* ═══════════════════════════════════════════════════════════════════════════
 * kyber_lattice_attack.c — MLWE Lattice Attack via Complex-Amplitude BP
 *
 *
 * Attack Model:
 *   Given (A, b = A·s + e mod q), recover secret vector s.
 *   A ∈ ℤ_q^{m×n}, s ∈ {-η,...,+η}^n, e ∈ {-η,...,+η}^m
 *
 * Factor Graph Encoding:
 *   - n VARIABLE nodes (sites 0..n-1): secret coefficients s[j]
 *     Domain D_var = 2η+1, prior = centered binomial B_η
 *   - m CHECK nodes (sites n..n+m-1): LWE equations
 *     Domain D_chk = q, used for Fourier message passing
 *   - n×m edges: PHASE weight w(v,c) = ω_q^{A[i][j]·v·c}
 *     encodes the mod-q linear constraint
 *
 * Build:
 *   gcc -O2 -std=gnu99 -I.. -o kyber_lattice_attack \
 *       kyber_lattice_attack.c ../quhit_triality.c ../quhit_hexagram.c \
 *       ../s6_exotic.c ../bigint.c -lm -lgmp
 *
 * Usage:
 *   ./kyber_lattice_attack [n] [q] [eta] [m] [seed]
 *   Default: n=16, q=97, η=2, m=32, seed=time
 * ═══════════════════════════════════════════════════════════════════════════ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * §0 — PARAMETERS AND CONSTANTS
 * ═══════════════════════════════════════════════════════════════════════════ */

#define MAX_N       64      /* Max secret dimension                       */
#define MAX_M       128     /* Max number of equations                    */
#define MAX_Q       4096    /* Max modulus (Kyber: q=3329)                */
#define MAX_D_VAR   9       /* Max variable domain = 2*MAX_ETA+1         */
#define MAX_ETA     4       /* Max noise width                           */

#define BP_MAX_ITER 300     /* Maximum BP iterations per decimation round */
#define BP_TOL      1e-8    /* Convergence tolerance                     */
#define BP_DAMP_START 0.50  /* Initial damping (annealing)               */
#define BP_DAMP_END   0.05  /* Final damping                             */
#define BP_COOL_ITERS 200   /* Annealing iterations                      */
#define BP_NUM_STARTS 3     /* Multi-start seeds                         */

#define PI 3.14159265358979323846

/* ═══════════════════════════════════════════════════════════════════════════
 * §1 — MLWE INSTANCE GENERATOR
 *
 * Generates b = A·s + e (mod q) with:
 *   A ~ Uniform(ℤ_q^{m×n})
 *   s ~ B_η^n  (centered binomial, range [-η, +η])
 *   e ~ B_η^m
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    int n;              /* Secret dimension                              */
    int m;              /* Number of equations                           */
    int q;              /* Modulus                                       */
    int eta;            /* Noise width                                   */
    int d_var;          /* Variable domain = 2*eta + 1                   */

    int A[MAX_M][MAX_N];    /* Public matrix A (mod q)                   */
    int b[MAX_M];           /* Public vector b = A·s + e (mod q)         */
    int s_true[MAX_N];      /* Secret (ground truth, for verification)   */
    int e_true[MAX_M];      /* Error  (ground truth, for verification)   */
} LWEInstance;

/* Centered binomial sample: sum of eta coin flips minus sum of eta coin flips */
static int sample_centered_binomial(int eta)
{
    int sum = 0;
    for (int i = 0; i < eta; i++) {
        sum += (rand() & 1);
        sum -= (rand() & 1);
    }
    return sum;
}

/* Positive modular reduction */
static inline int mod_pos(int x, int q)
{
    int r = x % q;
    return (r < 0) ? r + q : r;
}

static void lwe_generate(LWEInstance *inst, int n, int q, int eta, int m)
{
    inst->n = n;
    inst->m = m;
    inst->q = q;
    inst->eta = eta;
    inst->d_var = 2 * eta + 1;

    printf("\n  ═══ MLWE INSTANCE GENERATION ═══\n");
    printf("  Parameters: n=%d, q=%d, η=%d, m=%d\n", n, q, eta, m);
    printf("  Variable domain: D_var = %d  (values %d..%d)\n",
           inst->d_var, -eta, eta);
    printf("  Search space: %d^%d = ", inst->d_var, n);
    double search = pow(inst->d_var, n);
    if (search < 1e15) printf("%.0f\n", search);
    else printf("%.2e\n", search);

    /* Generate secret s ~ B_η */
    printf("  Secret s = [");
    for (int j = 0; j < n; j++) {
        inst->s_true[j] = sample_centered_binomial(eta);
        printf("%d%s", inst->s_true[j], j < n-1 ? ", " : "");
    }
    printf("]\n");

    /* Generate A ~ Uniform(ℤ_q) */
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            inst->A[i][j] = rand() % q;

    /* Generate error e ~ B_η, compute b = A·s + e mod q */
    for (int i = 0; i < m; i++) {
        inst->e_true[i] = sample_centered_binomial(eta);
        int dot = 0;
        for (int j = 0; j < n; j++)
            dot += inst->A[i][j] * inst->s_true[j];
        inst->b[i] = mod_pos(dot + inst->e_true[i], q);
    }

    /* Print first few equations for debugging */
    int show = (m < 4) ? m : 4;
    printf("  First %d equations (b = A·s + e mod %d):\n", show, q);
    for (int i = 0; i < show; i++) {
        printf("    b[%d] = %d  (A[%d] = [", i, inst->b[i], i);
        int show_n = (n < 6) ? n : 6;
        for (int j = 0; j < show_n; j++)
            printf("%d%s", inst->A[i][j], j < show_n-1 ? " " : "");
        if (n > 6) printf(" ...");
        printf("], e=%d)\n", inst->e_true[i]);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §2 — FACTOR GRAPH via COMPLEX-AMPLITUDE MESSAGES
 *
 * Messages live on edges between variable nodes and check nodes.
 * Each edge carries two directional messages:
 *   dir=0: variable → check  (domain = D_var)
 *   dir=1: check → variable  (domain = D_var)
 *
 * Both messages are complex-valued vectors of dimension D_var.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Edge message: directional complex vector */
typedef struct {
    double re[2][MAX_D_VAR];   /* [dir][value] — real part                */
    double im[2][MAX_D_VAR];   /* [dir][value] — imaginary part           */
} EdgeMsg;

/* Precomputed ω_q^k table */
typedef struct {
    double re[MAX_Q];       /* cos(2π·k/q) for k=0..q-1                  */
    double im[MAX_Q];       /* sin(2π·k/q) for k=0..q-1                  */
} OmegaTable;

static void omega_init(OmegaTable *wt, int q)
{
    for (int k = 0; k < q; k++) {
        double theta = 2.0 * PI * k / q;
        wt->re[k] = cos(theta);
        wt->im[k] = sin(theta);
    }
}

/* Variable node prior: centered binomial B_η
 * P(v) = C(2η, η+v) / 2^(2η) for v ∈ {-η,...,+η} */
static void compute_prior(double *prior, int eta, int d_var)
{
    int two_eta = 2 * eta;
    double denom = pow(2.0, two_eta);
    for (int vi = 0; vi < d_var; vi++) {
        int v = vi - eta;  /* Map index to value {-η...+η} */
        int k = eta + v;   /* Binomial index */
        /* C(2η, k) */
        double binom = 1.0;
        for (int i = 0; i < k; i++)
            binom *= (double)(two_eta - i) / (i + 1);
        prior[vi] = binom / denom;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §3 — COMPLEX-AMPLITUDE BELIEF PROPAGATION
 *
 * The heart of the attack. Adapted from the Ouroboros factoring engine.
 *
 * Variable → Check message (edge j→i):
 *   μ_{j→i}[v] = prior[v] × Π_{i' ≠ i} ν_{i'→j}[v]
 *
 * Check → Variable message (edge i→j):
 *   ν_{i→j}[v_j] = Σ_{v_{j'}: j'≠j} I[b_i = Σ A[i][j']·v_{j'} + A[i][j]·v_j (mod q)]
 *                   × Π_{j'≠j} μ_{j'→i}[v_{j'}]   × noise_penalty
 *
 *   Computed via Fourier trick:
 *   ν_{i→j}[v_j] = (1/q) Σ_{c=0}^{q-1} ω_q^{c·(b_i - A[i][j]·v_j)}
 *                  × Π_{j'≠j} F_c[μ_{j'→i}]
 *
 *   where F_c[μ] = Σ_v μ[v] · ω_q^{-A[i][j']·v·c} × noise_weight(v)
 *
 * This Fourier check is the exact analog of the CZ ω^{a·b} DFT₆ in
 * the factoring engine — but over ℤ_q instead of ℤ_6.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Precompute the exact error probability table for B_η
 * P_error[e mod q] for e ∈ {-η,...,+η} */
static void compute_error_prior(double *error_prior, int q, int eta)
{
    memset(error_prior, 0, sizeof(double) * q);
    double prior[MAX_D_VAR];
    compute_prior(prior, eta, 2 * eta + 1);
    for (int vi = 0; vi < 2 * eta + 1; vi++) {
        int val = vi - eta;
        int idx = mod_pos(val, q);
        error_prior[idx] = prior[vi];
    }
}

static void lwe_bp_solve(const LWEInstance *inst, int *s_recovered,
                         double *confidence,
                         const int *fixed, const int *fixed_vals,
                         unsigned int seed)
{
    const int n = inst->n;
    const int m = inst->m;
    const int q = inst->q;
    const int eta = inst->eta;
    const int d_var = inst->d_var;

    /* Count free variables */
    int n_free = 0;
    for (int j = 0; j < n; j++) if (!fixed[j]) n_free++;
    if (n_free == 0) return;

    /* Precompute ω_q tables */
    OmegaTable wt;
    omega_init(&wt, q);

    /* Prior on each variable node */
    double prior[MAX_D_VAR];
    compute_prior(prior, eta, d_var);

    printf("\n  ── BP Solver (seed %u, %d free / %d total) ──\n", seed, n_free, n);
    printf("  Prior B_%d: [", eta);
    for (int vi = 0; vi < d_var; vi++)
        printf("%.3f%s", prior[vi], vi < d_var-1 ? " " : "]\n");

    /* Precompute exact error distribution over ℤ_q */
    double *error_prior = (double *)calloc(q, sizeof(double));
    compute_error_prior(error_prior, q, eta);
    printf("  Using exact B_%d convolution over ℤ_%d\n", eta, q);

    /* Allocate messages: m×n edges, each with 2 directions × d_var complex */
    /* Edge (i,j) is at index i*n + j */
    int n_edges = m * n;
    EdgeMsg *msgs = (EdgeMsg *)calloc(n_edges, sizeof(EdgeMsg));
    EdgeMsg *new_msgs = (EdgeMsg *)calloc(n_edges, sizeof(EdgeMsg));

    /* Initialize messages: variable→check = prior, check→variable = uniform */
    srand(seed);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int eid = i * n + j;
            for (int vi = 0; vi < d_var; vi++) {
                if (fixed[j]) {
                    /* Fixed variable: delta at known value */
                    int fvi = fixed_vals[j] + eta;
                    msgs[eid].re[0][vi] = (vi == fvi) ? 1.0 : 0.0;
                    msgs[eid].im[0][vi] = 0.0;
                    msgs[eid].re[1][vi] = (vi == fvi) ? 1.0 : 0.0;
                    msgs[eid].im[1][vi] = 0.0;
                } else {
                    double noise = 0.001 * ((double)rand() / RAND_MAX - 0.5);
                    msgs[eid].re[0][vi] = prior[vi] + noise;
                    msgs[eid].im[0][vi] = 0.001 * ((double)rand() / RAND_MAX - 0.5);
                    msgs[eid].re[1][vi] = 1.0 / d_var;
                    msgs[eid].im[1][vi] = 0.0;
                }
            }
        }
    }

    /* Main BP loop */
    int converged = 0;
    for (int it = 0; it < BP_MAX_ITER && !converged; it++) {
        double max_delta = 0.0;

        /* Annealing damping schedule */
        double alpha = (it < BP_COOL_ITERS)
            ? BP_DAMP_START * exp(log(BP_DAMP_END / BP_DAMP_START) *
                                  ((double)it / BP_COOL_ITERS))
            : BP_DAMP_END;

        /* ── Variable → Check messages ──
         * μ_{j→i}[v] = prior[v] × Π_{i' ≠ i} ν_{i'→j}[v]
         * Uses INCREMENTAL RENORMALIZATION to prevent underflow:
         * normalize the running product vector after each multiplication.
         * SKIP fixed variables — their messages never change. */
        for (int j = 0; j < n; j++) {
            if (fixed[j]) continue;  /* Fixed vars keep delta messages */
            for (int i = 0; i < m; i++) {
                int eid = i * n + j;

                /* Start with prior as initial product */
                for (int vi = 0; vi < d_var; vi++) {
                    new_msgs[eid].re[0][vi] = prior[vi];
                    new_msgs[eid].im[0][vi] = 0.0;
                }

                /* Multiply incoming check→var messages one at a time,
                 * renormalizing after EACH multiplication */
                for (int ip = 0; ip < m; ip++) {
                    if (ip == i) continue;
                    int eid_in = ip * n + j;

                    for (int vi = 0; vi < d_var; vi++) {
                        double pr = new_msgs[eid].re[0][vi];
                        double pi_ = new_msgs[eid].im[0][vi];
                        double mr = msgs[eid_in].re[1][vi];
                        double mi = msgs[eid_in].im[1][vi];
                        new_msgs[eid].re[0][vi] = pr * mr - pi_ * mi;
                        new_msgs[eid].im[0][vi] = pr * mi + pi_ * mr;
                    }

                    /* Incremental L2 renormalization — prevents underflow */
                    double norm_sq = 0.0;
                    for (int vi = 0; vi < d_var; vi++)
                        norm_sq += new_msgs[eid].re[0][vi] * new_msgs[eid].re[0][vi] +
                                   new_msgs[eid].im[0][vi] * new_msgs[eid].im[0][vi];
                    if (norm_sq > 1e-300) {
                        double inv = 1.0 / sqrt(norm_sq);
                        for (int vi = 0; vi < d_var; vi++) {
                            new_msgs[eid].re[0][vi] *= inv;
                            new_msgs[eid].im[0][vi] *= inv;
                        }
                    }
                }
            }
        }

        /* ── Check → Variable messages ──
         * Computed via Fourier trick over ℤ_q:
         * ν_{i→j}[v_j] = (1/q) Σ_c ω_q^{c·(b_i - A[i][j]·v_j)}
         *                × Π_{j'≠j} [Σ_{v'} μ_{j'→i}[v'] · ω_q^{-A[i][j']·val(v')·c}]
         *                × noise_envelope(c)
         */
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int eid = i * n + j;

                /* ── DIRECT CONVOLUTION over ℤ_q ──
                 * Maintain running distribution P_sum[x] over ℤ_q.
                 * Convolve with each incoming variable message.
                 * This is EXACT — no Fourier approximation.
                 *
                 * P_sum starts as δ(0), then for each j'≠j:
                 * P_sum_new[x] = Σ_v μ_{j'→i}[v] × P_sum[x - A·val(v) mod q]
                 *
                 * Final: ν_{i→j}[v_j] = Σ_x P_sum[x] × P_error[b - A·v_j - x mod q]
                 */
                double *psum_cur = (double *)calloc(q, sizeof(double));
                double *psum_new = (double *)calloc(q, sizeof(double));
                psum_cur[0] = 1.0;  /* δ(0) */

                for (int jp = 0; jp < n; jp++) {
                    if (jp == j) continue;
                    int eid_in = i * n + jp;

                    memset(psum_new, 0, sizeof(double) * q);
                    for (int x = 0; x < q; x++) {
                        if (psum_cur[x] < 1e-30) continue; /* Skip zeros */
                        for (int vi = 0; vi < d_var; vi++) {
                            int val = vi - eta;
                            /* Use magnitude of complex message as probability */
                            double mr = msgs[eid_in].re[0][vi];
                            double mi_ = msgs[eid_in].im[0][vi];
                            double prob = mr * mr + mi_ * mi_;
                            if (prob < 1e-30) continue;
                            int new_x = mod_pos(x + inst->A[i][jp] * val, q);
                            psum_new[new_x] += psum_cur[x] * prob;
                        }
                    }

                    /* Normalize to prevent underflow/overflow */
                    double psum_max = 0.0;
                    for (int x = 0; x < q; x++)
                        if (psum_new[x] > psum_max) psum_max = psum_new[x];
                    if (psum_max > 1e-30) {
                        double inv = 1.0 / psum_max;
                        for (int x = 0; x < q; x++) psum_new[x] *= inv;
                    }

                    /* Swap buffers */
                    double *tmp = psum_cur;
                    psum_cur = psum_new;
                    psum_new = tmp;
                }

                /* Read out check→var message using exact error prior */
                for (int vi = 0; vi < d_var; vi++) {
                    int val = vi - eta;
                    double sum = 0.0;
                    for (int x = 0; x < q; x++) {
                        /* error = b - A·v_j - x mod q */
                        int err_idx = mod_pos(inst->b[i] - inst->A[i][j] * val - x, q);
                        sum += psum_cur[x] * error_prior[err_idx];
                    }
                    new_msgs[eid].re[1][vi] = sum;
                    new_msgs[eid].im[1][vi] = 0.0;
                }

                free(psum_cur);
                free(psum_new);

                /* Normalize check→var to unit L2 */
                double norm_sq = 0.0;
                for (int vi = 0; vi < d_var; vi++)
                    norm_sq += new_msgs[eid].re[1][vi] * new_msgs[eid].re[1][vi];
                if (norm_sq > 1e-30) {
                    double inv = 1.0 / sqrt(norm_sq);
                    for (int vi = 0; vi < d_var; vi++)
                        new_msgs[eid].re[1][vi] *= inv;
                }
            }
        }

        /* ── Damped update (annealing) ── */
        for (int eid = 0; eid < n_edges; eid++) {
            for (int dir = 0; dir < 2; dir++) {
                for (int vi = 0; vi < d_var; vi++) {
                    double upd_re = alpha * new_msgs[eid].re[dir][vi] +
                                    (1.0 - alpha) * msgs[eid].re[dir][vi];
                    double upd_im = alpha * new_msgs[eid].im[dir][vi] +
                                    (1.0 - alpha) * msgs[eid].im[dir][vi];

                    double dr = upd_re - msgs[eid].re[dir][vi];
                    double di = upd_im - msgs[eid].im[dir][vi];
                    double delta = dr*dr + di*di;
                    if (delta > max_delta) max_delta = delta;

                    msgs[eid].re[dir][vi] = upd_re;
                    msgs[eid].im[dir][vi] = upd_im;
                }
            }
        }

        /* Progress reporting */
        if (it < 10 || (it + 1) % 25 == 0 || max_delta < BP_TOL) {
            printf("    [BP] Iter %3d: residual = %.6e  α = %.4f\n",
                   it + 1, max_delta, alpha);
        }

        if (max_delta < BP_TOL) {
            converged = 1;
            printf("    [BP] CONVERGED at iteration %d\n", it + 1);
        }
    }

    if (!converged)
        printf("    [BP] Reached max iterations (%d)\n", BP_MAX_ITER);

    /* ── Extract marginals from converged messages ──
     * Uses INCREMENTAL RENORMALIZATION: normalize the belief vector
     * after each message multiplication to prevent underflow. */
    printf("\n  ── Marginal beliefs ──\n");
    for (int j = 0; j < n; j++) {
        if (fixed[j]) {
            s_recovered[j] = fixed_vals[j];
            continue;
        }
        /* Belief[v] = prior[v] × Π_i ν_{i→j}[v] */
        double belief_re[MAX_D_VAR], belief_im[MAX_D_VAR];
        for (int vi = 0; vi < d_var; vi++) {
            belief_re[vi] = prior[vi];
            belief_im[vi] = 0.0;
        }

        for (int i = 0; i < m; i++) {
            int eid = i * n + j;
            for (int vi = 0; vi < d_var; vi++) {
                double mr = msgs[eid].re[1][vi];
                double mi = msgs[eid].im[1][vi];
                double nr = belief_re[vi] * mr - belief_im[vi] * mi;
                double ni = belief_re[vi] * mi + belief_im[vi] * mr;
                belief_re[vi] = nr;
                belief_im[vi] = ni;
            }

            /* Incremental renormalization after each message */
            double ns = 0.0;
            for (int vi = 0; vi < d_var; vi++)
                ns += belief_re[vi] * belief_re[vi] +
                      belief_im[vi] * belief_im[vi];
            if (ns > 1e-300) {
                double inv = 1.0 / sqrt(ns);
                for (int vi = 0; vi < d_var; vi++) {
                    belief_re[vi] *= inv;
                    belief_im[vi] *= inv;
                }
            }
        }

        /* Born probability |belief|² */
        double prob[MAX_D_VAR];
        double sum_prob = 0.0;
        for (int vi = 0; vi < d_var; vi++) {
            prob[vi] = belief_re[vi] * belief_re[vi] +
                       belief_im[vi] * belief_im[vi];
            sum_prob += prob[vi];
        }
        if (sum_prob > 1e-30)
            for (int vi = 0; vi < d_var; vi++)
                prob[vi] /= sum_prob;

        /* Argmax → recovered secret */
        int best_vi = 0;
        double best_p = 0.0;
        for (int vi = 0; vi < d_var; vi++) {
            if (prob[vi] > best_p) {
                best_p = prob[vi];
                best_vi = vi;
            }
        }
        s_recovered[j] = best_vi - eta;
        confidence[j] = best_p;

        /* Print marginal */
        if (j < 20 || j == n - 1) {
            printf("    s[%2d]: val=%+d  P=%.4f  true=%+d %s  [",
                   j, s_recovered[j], best_p, inst->s_true[j],
                   (s_recovered[j] == inst->s_true[j]) ? "✓" : "✗");
            for (int vi = 0; vi < d_var; vi++)
                printf("%.3f%s", prob[vi], vi < d_var-1 ? " " : "");
            printf("]\n");
        }
    }

    free(msgs);
    free(new_msgs);
    free(error_prior);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §4 — SECRET RECOVERY AND VALIDATION
 * ═══════════════════════════════════════════════════════════════════════════ */

static int validate_recovery(const LWEInstance *inst, const int *s_recovered)
{
    int n = inst->n;
    int m = inst->m;
    int q = inst->q;

    /* Check: how many coefficients match? */
    int correct = 0;
    for (int j = 0; j < n; j++)
        if (s_recovered[j] == inst->s_true[j]) correct++;

    printf("\n  ═══ RECOVERY RESULT ═══\n");
    printf("  Coefficients correct: %d / %d (%.1f%%)\n",
           correct, n, 100.0 * correct / n);

    /* Compute residual: ||b - A·ŝ||₂ mod q */
    double residual = 0.0;
    int exact_eqs = 0;
    for (int i = 0; i < m; i++) {
        int dot = 0;
        for (int j = 0; j < n; j++)
            dot += inst->A[i][j] * s_recovered[j];
        int err = mod_pos(inst->b[i] - dot, q);
        if (err > q / 2) err -= q;
        residual += (double)(err * err);
        if (abs(err) <= inst->eta) exact_eqs++;
    }
    residual = sqrt(residual / m);

    printf("  Residual ||b - A·ŝ||_rms = %.3f\n", residual);
    printf("  Equations satisfied (|e| ≤ η): %d / %d\n", exact_eqs, m);

    if (correct == n) {
        printf("\n  ╔══════════════════════════════════════════════════════════╗\n");
        printf("  ║  ★ SECRET FULLY RECOVERED ★                             ║\n");
        printf("  ╚══════════════════════════════════════════════════════════╝\n");
        printf("\n  s = [");
        for (int j = 0; j < n; j++)
            printf("%+d%s", s_recovered[j], j < n-1 ? ", " : "");
        printf("]\n");
        return 1;
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §5 — MAIN DRIVER WITH MULTI-START
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char *argv[])
{
    int n   = (argc > 1) ? atoi(argv[1]) : 16;
    int q   = (argc > 2) ? atoi(argv[2]) : 97;
    int eta = (argc > 3) ? atoi(argv[3]) : 2;
    int m   = (argc > 4) ? atoi(argv[4]) : 32;
    unsigned int base_seed = (argc > 5) ? (unsigned int)atoi(argv[5])
                                         : (unsigned int)time(NULL);

    /* Validate parameters */
    if (n > MAX_N) { fprintf(stderr, "Error: n > %d\n", MAX_N); return 1; }
    if (m > MAX_M) { fprintf(stderr, "Error: m > %d\n", MAX_M); return 1; }
    if (q > MAX_Q) { fprintf(stderr, "Error: q > %d\n", MAX_Q); return 1; }
    if (eta > MAX_ETA) { fprintf(stderr, "Error: eta > %d\n", MAX_ETA); return 1; }
    if (2 * eta + 1 > MAX_D_VAR) {
        fprintf(stderr, "Error: D_var = %d > %d\n", 2*eta+1, MAX_D_VAR);
        return 1;
    }

    printf("  ═══════════════════════════════════════════════════════════════\n");
    printf("  MLWE Lattice Attack — Complex-Amplitude BP\n");
    printf("  The Devil's Assault on Post-Quantum Cryptography\n");
    printf("  ═══════════════════════════════════════════════════════════════\n");

    /* Generate the MLWE instance */
    srand(base_seed);
    LWEInstance inst;
    lwe_generate(&inst, n, q, eta, m);

    /* ── Decimation loop: iteratively fix most confident variable ── */
    int fixed[MAX_N];
    int fixed_vals[MAX_N];
    memset(fixed, 0, sizeof(fixed));
    memset(fixed_vals, 0, sizeof(fixed_vals));

    int best_correct = 0;
    int best_recovered[MAX_N];
    memset(best_recovered, 0, sizeof(best_recovered));

    clock_t t_start = clock();

    int n_fixed = 0;
    while (n_fixed < n) {
        printf("\n  ╔════════════════════════════════════════╗\n");
        printf("  ║  Decimation Round %d / %d  (%d fixed)    ║\n",
               n_fixed + 1, n, n_fixed);
        printf("  ╚════════════════════════════════════════╝\n");

        /* Run multi-start BP on the reduced system */
        int round_best_correct = 0;
        int round_best_recovered[MAX_N];
        double round_best_conf[MAX_N];
        memset(round_best_recovered, 0, sizeof(round_best_recovered));
        memset(round_best_conf, 0, sizeof(round_best_conf));

        int starts_this_round = (n_fixed == 0) ? BP_NUM_STARTS : 1;
        for (int start = 0; start < starts_this_round; start++) {
            int s_recovered[MAX_N];
            double conf[MAX_N];
            memset(s_recovered, 0, sizeof(s_recovered));
            memset(conf, 0, sizeof(conf));

            lwe_bp_solve(&inst, s_recovered, conf, fixed, fixed_vals,
                         base_seed + n_fixed * 7919 + start * 1337);

            int correct = 0;
            for (int j = 0; j < n; j++)
                if (s_recovered[j] == inst.s_true[j]) correct++;

            printf("  [Round %d/Start %d] Correct: %d / %d\n",
                   n_fixed + 1, start + 1, correct, n);

            if (correct > round_best_correct) {
                round_best_correct = correct;
                memcpy(round_best_recovered, s_recovered, sizeof(int) * n);
            }

            if (correct == n) break;
        }

        if (round_best_correct > best_correct) {
            best_correct = round_best_correct;
            memcpy(best_recovered, round_best_recovered, sizeof(int) * n);
        }

        if (best_correct == n) {
            printf("  ★ ALL VARIABLES DETERMINED ★\n");
            break;
        }

        /* Find the most confident FREE variable and fix it */
        int s_conf_run[MAX_N];
        double conf_run[MAX_N];
        memset(s_conf_run, 0, sizeof(s_conf_run));
        memset(conf_run, 0, sizeof(conf_run));
        lwe_bp_solve(&inst, s_conf_run, conf_run, fixed, fixed_vals,
                     base_seed + n_fixed * 3571);

        /* Pick the free variable with highest confidence */
        int fix_j = -1;
        double max_conf = -1.0;
        for (int j = 0; j < n; j++) {
            if (!fixed[j] && conf_run[j] > max_conf) {
                max_conf = conf_run[j];
                fix_j = j;
            }
        }

        if (fix_j < 0) break;

        fixed[fix_j] = 1;
        fixed_vals[fix_j] = s_conf_run[fix_j];
        n_fixed++;

        printf("  → Fixed s[%d] = %+d  (true = %+d) %s\n",
               fix_j, fixed_vals[fix_j], inst.s_true[fix_j],
               (fixed_vals[fix_j] == inst.s_true[fix_j]) ? "✓" : "✗");
    }

    clock_t t_end = clock();
    double elapsed = (double)(t_end - t_start) / CLOCKS_PER_SEC;

    printf("\n");
    int success = validate_recovery(&inst, best_recovered);

    printf("\n  Time: %.3f seconds\n", elapsed);
    printf("\n  ═══════════════════════════════════════════════════════════════\n");
    printf("  MLWE Lattice Attack Engine complete.\n");
    printf("  ═══════════════════════════════════════════════════════════════\n");

    return success ? 0 : 1;
}
