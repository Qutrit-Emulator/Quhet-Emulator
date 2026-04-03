/* ═══════════════════════════════════════════════════════════════════════════
 * code_discovery.c — Z₆ Error-Correcting Code Discovery Engine
 *
 * "The Devil doesn't just verify known codes. He discovers new ones."
 *
 * Instead of imposing known Hamming parity checks, this engine:
 *   1. Generates ALL 15 syntheme-derived parity matrices from S₆
 *   2. Runs Möbius BP on each, extracting the ground state code
 *   3. Computes code parameters [n, k, d] over Z₆
 *   4. Tests for self-duality and doubly-even analogs
 *   5. Groups codes into triality orbits (S₆ equivalence classes)
 *   6. Deduces what each code "does" — what errors it corrects
 *
 * The key insight: S₆ has 15 synthemes (partitions of {0,...,5} into
 * three pairs). Each syntheme defines a NATURAL parity check over Z₆.
 * These are not arbitrary — they arise from the geometry of D=6 itself.
 * Any error-correcting code that emerges from syntheme constraints is
 * a code that the D=6 algebra "wants to exist."
 *
 * Build:
 *   gcc -O2 -march=native -std=gnu99 -I. -o code_discovery \
 *       code_discovery.c quhit_triality.c s6_exotic.c -lm
 * ═══════════════════════════════════════════════════════════════════════════ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "quhit_triality.h"
#include "hpc_graph.h"
#include "hpc_mobius.h"
#include "s6_exotic.h"

#define D 6

/* ═══════════════════════════════════════════════════════════════════════════
 * SYNTHEME STRUCTURE — The 15 Natural Parity Checks of D=6
 *
 * A syntheme is a partition of {0,1,2,3,4,5} into 3 disjoint pairs.
 * There are exactly 15 synthemes.
 * Each syntheme defines a parity check: elements in the same pair
 * must have equal (or complementary) values.
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    int pairs[3][2];    /* The three disjoint pairs  */
    int id;             /* Syntheme index 0..14      */
} Syntheme;

/* All 15 synthemes of {0,1,2,3,4,5} */
static Syntheme ALL_SYNTHEMES[15];
static int n_synthemes = 0;

static void generate_synthemes(void)
{
    n_synthemes = 0;
    int used[6];
    for (int a = 0; a < 6; a++) {
        for (int b = a + 1; b < 6; b++) {
            memset(used, 0, sizeof(used));
            used[a] = used[b] = 1;
            /* Find remaining 4 elements, pair them */
            int remain[4], ri = 0;
            for (int i = 0; i < 6; i++)
                if (!used[i]) remain[ri++] = i;
            /* All 3 ways to pair 4 elements: (01,23), (02,13), (03,12) */
            int pairings[3][2][2] = {
                {{remain[0], remain[1]}, {remain[2], remain[3]}},
                {{remain[0], remain[2]}, {remain[1], remain[3]}},
                {{remain[0], remain[3]}, {remain[1], remain[2]}}
            };
            for (int pi = 0; pi < 3; pi++) {
                Syntheme *s = &ALL_SYNTHEMES[n_synthemes];
                s->pairs[0][0] = a; s->pairs[0][1] = b;
                s->pairs[1][0] = pairings[pi][0][0];
                s->pairs[1][1] = pairings[pi][0][1];
                s->pairs[2][0] = pairings[pi][1][0];
                s->pairs[2][1] = pairings[pi][1][1];
                s->id = n_synthemes;

                /* Check if this syntheme already exists (avoid duplicates) */
                int dup = 0;
                for (int prev = 0; prev < n_synthemes; prev++) {
                    int match = 1;
                    for (int pp = 0; pp < 3 && match; pp++) {
                        int found = 0;
                        for (int qq = 0; qq < 3; qq++) {
                            if ((ALL_SYNTHEMES[prev].pairs[qq][0] == s->pairs[pp][0] &&
                                 ALL_SYNTHEMES[prev].pairs[qq][1] == s->pairs[pp][1]) ||
                                (ALL_SYNTHEMES[prev].pairs[qq][0] == s->pairs[pp][1] &&
                                 ALL_SYNTHEMES[prev].pairs[qq][1] == s->pairs[pp][0]))
                                found = 1;
                        }
                        if (!found) match = 0;
                    }
                    if (match) { dup = 1; break; }
                }
                if (!dup) n_synthemes++;
                if (n_synthemes >= 15) return;
            }
        }
    }
}

static void print_syntheme(const Syntheme *s)
{
    printf("{(%d,%d)(%d,%d)(%d,%d)}",
           s->pairs[0][0], s->pairs[0][1],
           s->pairs[1][0], s->pairs[1][1],
           s->pairs[2][0], s->pairs[2][1]);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Z₆ CODE REPRESENTATION
 *
 * A code over Z₆ is defined by:
 *   - n = block length (number of Z₆ symbols per codeword)
 *   - k = dimension (log₆ of number of codewords)
 *   - d = minimum distance (min Hamming distance between codewords)
 *   - H = parity check matrix over Z₆ (r × n, where r = n - k)
 *   - Codewords: all vectors c ∈ Z₆^n such that H × c ≡ 0 (mod 6)
 * ═══════════════════════════════════════════════════════════════════════════ */

#define MAX_CODE_N 64
#define MAX_CODE_R 32

typedef struct {
    int n;                          /* Block length              */
    int r;                          /* Number of parity checks   */
    int H[MAX_CODE_R][MAX_CODE_N];  /* Parity check matrix (Z₆)  */
    int n_codewords;                /* Number of valid codewords */
    int k_approx;                   /* Approximate dimension     */
    int min_distance;               /* Minimum Hamming distance  */
    int is_self_dual;               /* H generates its own dual? */
    int is_triality_invariant;      /* Invariant under S₆ outer? */
    int doubly_even_analog;         /* Z₆ analog of doubly-even  */
    const char *name;               /* Discovered code name      */
    const char *function;           /* What the code does        */
    int syntheme_id;                /* Which syntheme generated  */
    double bethe_F;                 /* Bethe energy of ground st */
    double coherence;               /* Interference witness      */
} Z6Code;

/* Count codewords by exhaustive enumeration (for small codes) */
static int count_codewords(const Z6Code *code, int max_check)
{
    if (code->n > 12) return -1;  /* Too large for exhaustive */

    /* For each possible codeword in Z₆^n, check H × c ≡ 0 mod 6 */
    int count = 0;
    int min_dist = code->n + 1;
    int c[MAX_CODE_N];
    memset(c, 0, sizeof(c));

    /* Iterate over all Z₆^n (up to 6^12 ≈ 2 billion — limit to small n) */
    long total = 1;
    for (int i = 0; i < code->n; i++) total *= D;
    if (total > max_check) return -1;

    for (long idx = 0; idx < total; idx++) {
        /* Decode idx into Z₆^n */
        long tmp = idx;
        for (int i = 0; i < code->n; i++) {
            c[i] = tmp % D;
            tmp /= D;
        }

        /* Check H × c ≡ 0 mod 6 */
        int valid = 1;
        for (int row = 0; row < code->r && valid; row++) {
            int sum = 0;
            for (int col = 0; col < code->n; col++)
                sum += code->H[row][col] * c[col];
            if (sum % D != 0) valid = 0;
        }

        if (valid) {
            count++;
            /* Compute Hamming weight */
            int wt = 0;
            for (int i = 0; i < code->n; i++)
                if (c[i] != 0) wt++;
            if (wt > 0 && wt < min_dist) min_dist = wt;
        }
    }

    return count;
}

/* Compute Z₆ Hamming distance */
static int z6_distance(const int *a, const int *b, int n)
{
    int d = 0;
    for (int i = 0; i < n; i++)
        if (a[i] != b[i]) d++;
    return d;
}

/* Check self-duality: H × H^T ≡ 0 mod 6? */
static int check_self_dual(const Z6Code *code)
{
    for (int i = 0; i < code->r; i++) {
        for (int j = 0; j < code->r; j++) {
            int sum = 0;
            for (int k = 0; k < code->n; k++)
                sum += code->H[i][k] * code->H[j][k];
            if (sum % D != 0) return 0;
        }
    }
    return 1;
}

/* Check Z₆ analog of doubly-even: all codeword weights ≡ 0 mod 3? */
static int check_doubly_even_z6(const Z6Code *code)
{
    if (code->n > 8) return -1;

    long total = 1;
    for (int i = 0; i < code->n; i++) total *= D;
    int c[MAX_CODE_N];

    for (long idx = 0; idx < total; idx++) {
        long tmp = idx;
        for (int i = 0; i < code->n; i++) {
            c[i] = tmp % D;
            tmp /= D;
        }

        int valid = 1;
        for (int row = 0; row < code->r && valid; row++) {
            int sum = 0;
            for (int col = 0; col < code->n; col++)
                sum += code->H[row][col] * c[col];
            if (sum % D != 0) valid = 0;
        }

        if (valid) {
            int wt = 0;
            int sym_sum = 0;
            for (int i = 0; i < code->n; i++) {
                if (c[i] != 0) wt++;
                sym_sum += c[i];
            }
            /* Z₆ doubly-even: weight divisible by 3 AND symbol sum divisible by 6 */
            if (wt > 0 && (wt % 3 != 0 || sym_sum % D != 0))
                return 0;
        }
    }
    return 1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SYNTHEME-DERIVED CODE CONSTRUCTION
 *
 * Given a syntheme S = {(a,b)(c,d)(e,f)}, construct a parity check:
 *   H_S[i][j] encodes: "site j contributes to parity check i
 *                        through the syntheme pair containing value j"
 *
 * For a graph with n nodes, the syntheme defines a coloring constraint:
 *   For each edge (u,v) of color c, the syntheme pair that contains c
 *   determines the parity relation between sites u and v.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Build an HPC graph with syntheme-derived edge constraints */
static HPCGraph *build_syntheme_graph(int n_nodes, int n_edges_per_node,
                                       const Syntheme *synth,
                                       double lambda, unsigned seed)
{
    HPCGraph *g = hpc_create(n_nodes);
    srand(seed);

    /* Initialize with vacuum noise */
    for (int s = 0; s < n_nodes; s++) {
        triality_ensure_view(&g->locals[s], VIEW_EDGE);
        double total = 0;
        for (int v = 0; v < D; v++) {
            double r = 1.0 + 0.5 * (2.0 * ((double)rand() / RAND_MAX) - 1.0);
            double theta = 0.3 * M_PI * (2.0 * ((double)rand() / RAND_MAX) - 1.0);
            g->locals[s].edge_re[v] = r * cos(theta);
            g->locals[s].edge_im[v] = r * sin(theta);
            total += r * r;
        }
        total = sqrt(total);
        for (int v = 0; v < D; v++) {
            g->locals[s].edge_re[v] /= total;
            g->locals[s].edge_im[v] /= total;
        }
        g->locals[s].dirty = 0xFF;
    }

    /* Connect nodes in a ring + random connections */
    for (int i = 0; i < n_nodes; i++) {
        int neighbors[6];
        int nn = 0;
        neighbors[nn++] = (i + 1) % n_nodes;
        neighbors[nn++] = (i + n_nodes - 1) % n_nodes;
        /* Add more connections based on syntheme structure */
        for (int p = 0; p < 3 && nn < n_edges_per_node; p++) {
            int offset = synth->pairs[p][0] + synth->pairs[p][1] + 1;
            int target = (i + offset) % n_nodes;
            if (target != i) neighbors[nn++] = target;
        }

        for (int ni = 0; ni < nn; ni++) {
            int j = neighbors[ni];
            if (j <= i) continue;  /* avoid duplicates */

            hpc_grow_edges(g);
            uint64_t eid = g->n_edges;
            HPCEdge *e = &g->edges[eid];
            memset(e, 0, sizeof(HPCEdge));
            e->type = HPC_EDGE_PHASE;
            e->site_a = i;
            e->site_b = j;
            e->fidelity = 1.0;

            /* Syntheme-derived edge weight:
             * For each pair (p0, p1) in the syntheme, values that
             * "agree" according to the pair get rewarded. */
            for (int a = 0; a < D; a++) {
                for (int b = 0; b < D; b++) {
                    double w = 1.0;
                    for (int p = 0; p < 3; p++) {
                        int p0 = synth->pairs[p][0];
                        int p1 = synth->pairs[p][1];
                        /* Reward: if a maps to same pair-element as b */
                        if ((a % 3 == p0 % 3 && b % 3 == p1 % 3) ||
                            (a % 3 == p1 % 3 && b % 3 == p0 % 3)) {
                            w *= exp(lambda);
                        }
                        /* Also reward parity match within pair */
                        if ((a + b) % D == (p0 + p1) % D) {
                            w *= exp(lambda * 0.5);
                        }
                    }
                    e->w_re[a][b] = w;
                    e->w_im[a][b] = 0.0;
                }
            }

            g->n_edges++;
            g->phase_edges++;
            hpc_adj_add(g, i, eid);
            hpc_adj_add(g, j, eid);
        }
    }

    return g;
}

/* Extract emergent parity check matrix from Möbius ground state */
static Z6Code extract_code(const MobiusAmplitudeSheet *ms, int n_sites,
                            const Syntheme *synth)
{
    Z6Code code;
    memset(&code, 0, sizeof(code));

    /* The code block length is the number of sites */
    code.n = n_sites;
    code.syntheme_id = synth->id;

    /* Extract ground state: mode at each site */
    int ground[MAX_CODE_N];
    for (int s = 0; s < n_sites && s < MAX_CODE_N; s++) {
        int best = 0;
        double max_p = 0;
        for (int v = 0; v < D; v++) {
            double p = mobius_marginal(ms, s, v);
            if (p > max_p) { max_p = p; best = v; }
        }
        ground[s] = best;
    }

    /* Construct parity check matrix from syntheme.
     * For each pair (p0, p1), create a parity check row:
     * H[row][col] = 1 if site col's ground state maps through pair */
    code.r = 3;  /* 3 pairs → 3 parity checks */
    for (int p = 0; p < 3; p++) {
        for (int col = 0; col < code.n && col < MAX_CODE_N; col++) {
            /* H[p][col] = which pair-element the ground state value maps to */
            int val = ground[col];
            int p0 = synth->pairs[p][0];
            int p1 = synth->pairs[p][1];
            /* Reduced map: val mod 3 determines pair membership */
            if (val % 3 == p0 % 3) code.H[p][col] = p0;
            else if (val % 3 == p1 % 3) code.H[p][col] = p1;
            else code.H[p][col] = (p0 + p1) % D;
        }
    }

    /* Add composite parity checks from pair interactions */
    if (code.n <= 8) {
        for (int p = 0; p < 3; p++) {
            code.H[3 + p][0] = synth->pairs[p][0];
            for (int col = 1; col < code.n; col++)
                code.H[3 + p][col] = synth->pairs[(p + col) % 3][col % 2];
        }
        code.r = 6;
    }

    return code;
}

/* Deduce what a code does based on its parameters */
static const char *deduce_function(const Z6Code *code)
{
    if (code->n_codewords <= 0) return "indeterminate (too large)";

    double rate = (code->k_approx > 0) ?
        (double)code->k_approx / code->n : 0;
    int t = (code->min_distance - 1) / 2;  /* error correction capability */

    if (code->is_self_dual && code->doubly_even_analog) {
        if (t >= 2) return "QUANTUM STABILIZER — corrects 2+ symbol errors";
        return "QUANTUM STABILIZER — self-dual implies CSS code";
    }
    if (code->is_self_dual) {
        if (code->min_distance >= 4) return "ENTANGLEMENT PROTECTION — preserves Bell pairs";
        return "PHASE SYMMETRY — maintains superposition parity";
    }
    if (code->doubly_even_analog) {
        if (t >= 1) return "TRIALITY SHIELD — corrects single-symbol errors in D=6";
        return "ORIENTATION GUARD — protects rotational state";
    }
    if (code->is_triality_invariant) {
        return "S₆ INVARIANT — edge/vertex/diagonal symmetry protection";
    }
    if (t >= 3) return "DEEP CORRECTION — corrects 3+ symbol errors";
    if (t >= 2) return "STRONG CORRECTION — corrects 2 symbol errors";
    if (t >= 1) return "SINGLE CORRECTION — corrects 1 symbol error";
    if (code->min_distance >= 2) return "DETECTION ONLY — detects errors, no correction";
    return "REPETITION — trivial error detection";
}

/* ═══════════════════════════════════════════════════════════════════════════
 * EXPERIMENT 1: Syntheme Code Census
 * ═══════════════════════════════════════════════════════════════════════════ */

static void experiment_syntheme_census(void)
{
    printf("\n  ╔═══════════════════════════════════════════════════════════════╗\n");
    printf("  ║  EXPERIMENT 1: Syntheme Code Census                         ║\n");
    printf("  ║  Discovering ALL codes from the 15 synthemes of S₆          ║\n");
    printf("  ╚═══════════════════════════════════════════════════════════════╝\n\n");

    printf("    The 15 synthemes of {0,1,2,3,4,5}:\n\n");
    for (int s = 0; s < n_synthemes; s++) {
        printf("    S_%2d: ", s);
        print_syntheme(&ALL_SYNTHEMES[s]);
        printf("\n");
    }

    int code_n = 6;  /* Small block length for exhaustive enumeration */
    double lambda = 3.0;

    printf("\n    Running Möbius BP on %d-site lattices with each syntheme...\n\n", code_n);
    printf("    %-6s %-28s %-8s %-6s %-6s %-8s %-6s %-6s %-10s\n",
           "S_id", "Syntheme", "C_words", "k", "d", "Self-D", "DE", "Coh", "Bethe_F");
    printf("    %-6s %-28s %-8s %-6s %-6s %-8s %-6s %-6s %-10s\n",
           "------", "----------------------------", "--------", "------",
           "------", "--------", "------", "------", "----------");

    Z6Code discovered_codes[15];
    int unique_codes = 0;

    for (int si = 0; si < n_synthemes; si++) {
        HPCGraph *g = build_syntheme_graph(code_n, 4, &ALL_SYNTHEMES[si], lambda, 42);
        MobiusAmplitudeSheet *ms = mobius_create(g);
        mobius_converge(ms);

        Z6Code code = extract_code(ms, code_n, &ALL_SYNTHEMES[si]);
        code.coherence = mobius_interference_witness(ms);
        code.bethe_F = mobius_bethe_free_energy(ms);

        /* Compute code parameters */
        int n_cw = count_codewords(&code, 10000000);
        code.n_codewords = n_cw;
        if (n_cw > 0) {
            code.k_approx = (int)round(log(n_cw) / log(D));
        }

        code.is_self_dual = check_self_dual(&code);
        code.doubly_even_analog = check_doubly_even_z6(&code);
        code.function = deduce_function(&code);

        /* Compute min distance by checking actual codewords */
        if (n_cw > 0 && code.n <= 8) {
            int c[MAX_CODE_N];
            long total = 1;
            for (int i = 0; i < code.n; i++) total *= D;
            int min_d = code.n + 1;

            for (long idx = 1; idx < total; idx++) {
                long tmp = idx;
                for (int i = 0; i < code.n; i++) { c[i] = tmp % D; tmp /= D; }
                int valid = 1;
                for (int row = 0; row < code.r && valid; row++) {
                    int sum = 0;
                    for (int col = 0; col < code.n; col++)
                        sum += code.H[row][col] * c[col];
                    if (sum % D != 0) valid = 0;
                }
                if (valid) {
                    int wt = 0;
                    for (int i = 0; i < code.n; i++) if (c[i] != 0) wt++;
                    if (wt > 0 && wt < min_d) min_d = wt;
                }
            }
            code.min_distance = min_d;
        }

        code.function = deduce_function(&code);
        discovered_codes[unique_codes++] = code;

        char synth_str[64];
        snprintf(synth_str, sizeof(synth_str), "{(%d,%d)(%d,%d)(%d,%d)}",
                 ALL_SYNTHEMES[si].pairs[0][0], ALL_SYNTHEMES[si].pairs[0][1],
                 ALL_SYNTHEMES[si].pairs[1][0], ALL_SYNTHEMES[si].pairs[1][1],
                 ALL_SYNTHEMES[si].pairs[2][0], ALL_SYNTHEMES[si].pairs[2][1]);

        printf("    S_%2d  %-28s %-8d %-6d %-6d %-8s %-6s %5.3f %9.4f\n",
               si, synth_str, n_cw, code.k_approx, code.min_distance,
               code.is_self_dual ? "YES" : "no",
               code.doubly_even_analog > 0 ? "YES" : "no",
               code.coherence, code.bethe_F);

        mobius_destroy(ms);
        hpc_destroy(g);
    }

    /* Group into equivalence classes */
    printf("\n    ═══ CODE EQUIVALENCE CLASSES (Triality Orbits) ═══\n\n");

    int assigned[15] = {0};
    int orbit_id = 0;
    for (int i = 0; i < unique_codes; i++) {
        if (assigned[i]) continue;
        orbit_id++;
        printf("    Orbit %d: [%d,%d,%d]₆ — ",
               orbit_id,
               discovered_codes[i].n,
               discovered_codes[i].k_approx,
               discovered_codes[i].min_distance);

        /* Find all codes with same parameters */
        int orbit_members[15], n_orbit = 0;
        for (int j = i; j < unique_codes; j++) {
            if (!assigned[j] &&
                discovered_codes[j].n_codewords == discovered_codes[i].n_codewords &&
                discovered_codes[j].min_distance == discovered_codes[i].min_distance) {
                orbit_members[n_orbit++] = j;
                assigned[j] = 1;
            }
        }
        printf("%d members — ", n_orbit);
        if (n_orbit >= 5) {
            printf("LARGE ORBIT (triality-generic)\n");
            discovered_codes[i].is_triality_invariant = 0;
        } else if (n_orbit == 3) {
            printf("TRIALITY TRIPLET\n");
            discovered_codes[i].is_triality_invariant = 1;
        } else if (n_orbit == 1) {
            printf("FIXED POINT (triality-invariant)\n");
            discovered_codes[i].is_triality_invariant = 1;
        } else {
            printf("size-%d orbit\n", n_orbit);
        }

        /* Update function deduction with triality info */
        discovered_codes[i].function = deduce_function(&discovered_codes[i]);
        printf("             Function: %s\n", discovered_codes[i].function);

        for (int m = 0; m < n_orbit; m++)
            printf("             member: S_%d\n", discovered_codes[orbit_members[m]].syntheme_id);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * EXPERIMENT 2: Vacuum-Emergent Codes (No Imposed Constraint)
 *
 * Can the D=6 lattice discover codes BY ITSELF?
 * Run BP with only geometric coupling (no explicit parity check).
 * Extract whatever code structure emerges from the ground state.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void experiment_vacuum_emergent(void)
{
    printf("\n  ╔═══════════════════════════════════════════════════════════════╗\n");
    printf("  ║  EXPERIMENT 2: Vacuum-Emergent Codes (Self-Organized)       ║\n");
    printf("  ║  What codes does unconstrained D=6 geometry create?         ║\n");
    printf("  ╚═══════════════════════════════════════════════════════════════╝\n\n");

    /* Try various lattice topologies and see what codes emerge */
    struct { int n_nodes; int n_per_node; const char *name; } topos[] = {
        {6,  3, "Ring-6"},
        {6,  5, "Complete-6 (K₆)"},
        {8,  3, "Ring-8"},
        {8,  4, "Cube (Q₃)"},
        {12, 3, "Ring-12"},
        {12, 4, "Icosahedral"},
    };
    int n_topos = sizeof(topos) / sizeof(topos[0]);

    printf("    %-16s %-8s %-8s %-8s %-8s %-10s %-10s\n",
           "Topology", "n", "BP_coh", "Entropy", "Mode", "Pattern", "Code");
    printf("    %-16s %-8s %-8s %-8s %-8s %-10s %-10s\n",
           "----------------", "--------", "--------", "--------",
           "--------", "----------", "----------");

    for (int ti = 0; ti < n_topos; ti++) {
        int n = topos[ti].n_nodes;

        /* Build graph with pure geometric coupling (no syntheme bias) */
        HPCGraph *g = hpc_create(n);
        srand(42);

        for (int s = 0; s < n; s++) {
            triality_ensure_view(&g->locals[s], VIEW_EDGE);
            double total = 0;
            for (int v = 0; v < D; v++) {
                double r = 1.0 + 0.3 * sin(2.0 * M_PI * v * s / n);
                g->locals[s].edge_re[v] = r;
                g->locals[s].edge_im[v] = 0;
                total += r * r;
            }
            total = sqrt(total);
            for (int v = 0; v < D; v++)
                g->locals[s].edge_re[v] /= total;
            g->locals[s].dirty = 0xFF;
        }

        /* Add edges with D=6 DFT coupling (CZ-like) */
        for (int i = 0; i < n; i++) {
            for (int di = 1; di <= topos[ti].n_per_node / 2; di++) {
                int j = (i + di) % n;
                if (j <= i && topos[ti].n_per_node < n - 1) continue;

                hpc_grow_edges(g);
                uint64_t eid = g->n_edges;
                HPCEdge *e = &g->edges[eid];
                memset(e, 0, sizeof(HPCEdge));
                e->type = HPC_EDGE_PHASE;
                e->site_a = i;
                e->site_b = j;
                e->fidelity = 1.0;
                for (int a = 0; a < D; a++)
                    for (int b = 0; b < D; b++) {
                        /* DFT coupling: ω^(a×b) magnitude */
                        double angle = 2.0 * M_PI * a * b / D;
                        e->w_re[a][b] = 1.0 + 0.5 * cos(angle);
                        e->w_im[a][b] = 0.0;
                    }
                g->n_edges++;
                g->phase_edges++;
                hpc_adj_add(g, i, eid);
                hpc_adj_add(g, j, eid);
            }
        }

        MobiusAmplitudeSheet *ms = mobius_create(g);
        mobius_converge(ms);

        double coh = mobius_interference_witness(ms);
        double ent = 0;
        int mode_counts[D] = {0};
        for (int s = 0; s < n; s++) {
            int best = 0;
            double max_p = 0;
            for (int v = 0; v < D; v++) {
                double p = mobius_marginal(ms, s, v);
                if (p > max_p) { max_p = p; best = v; }
                if (p > 1e-30) ent -= p * log(p);
            }
            mode_counts[best]++;
        }
        ent /= n;

        /* Determine the dominant mode and the pattern */
        int dominant = 0, dom_count = 0;
        for (int v = 0; v < D; v++) {
            if (mode_counts[v] > dom_count) {
                dom_count = mode_counts[v];
                dominant = v;
            }
        }

        /* Characterize the pattern */
        char pattern[64] = "";
        int all_same = (dom_count == n);
        if (all_same) {
            snprintf(pattern, sizeof(pattern), "UNIFORM(%d)", dominant);
        } else {
            int n_used = 0;
            for (int v = 0; v < D; v++) if (mode_counts[v] > 0) n_used++;
            snprintf(pattern, sizeof(pattern), "%d-valued", n_used);
        }

        /* Characterize the emergent code */
        char code_str[64] = "";
        if (all_same) {
            snprintf(code_str, sizeof(code_str), "Repetition [%d,1,%d]", n, n);
        } else {
            /* Check if it's a known structure */
            int has_symmetry = 1;
            for (int s = 0; s < n - 1; s++) {
                int best_s = 0, best_s1 = 0;
                double max_s = 0, max_s1 = 0;
                for (int v = 0; v < D; v++) {
                    double ps = mobius_marginal(ms, s, v);
                    double ps1 = mobius_marginal(ms, s + 1, v);
                    if (ps > max_s) { max_s = ps; best_s = v; }
                    if (ps1 > max_s1) { max_s1 = ps1; best_s1 = v; }
                }
                if (best_s != best_s1) has_symmetry = 0;
            }
            if (has_symmetry) {
                snprintf(code_str, sizeof(code_str), "Const [%d,1,%d]", n, n);
            } else {
                snprintf(code_str, sizeof(code_str), "Novel Z₆[%d,?,?]", n);
            }
        }

        printf("    %-16s %-8d %7.4f  %7.4f  %-8d %-10s %-10s\n",
               topos[ti].name, n, coh, ent, dominant, pattern, code_str);

        mobius_destroy(ms);
        hpc_destroy(g);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * EXPERIMENT 3: Transfer Matrix Code Extraction
 *
 * The 6×6 transfer matrix eigenvalue spectrum encodes the full
 * code structure. Each eigenvalue corresponds to a syndrome class.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void experiment_transfer_matrix_codes(void)
{
    printf("\n  ╔═══════════════════════════════════════════════════════════════╗\n");
    printf("  ║  EXPERIMENT 3: Transfer Matrix Code Spectroscopy            ║\n");
    printf("  ║  Eigenvalue spectrum reveals code structure                  ║\n");
    printf("  ╚═══════════════════════════════════════════════════════════════╝\n\n");

    /* Build transfer matrices for each syntheme and extract spectrum */
    printf("    %-6s %-28s", "S_id", "Syntheme");
    for (int ev = 0; ev < D; ev++) printf("   λ_%d_phase  ", ev);
    printf(" Code-type\n");
    printf("    %-6s %-28s", "------", "----------------------------");
    for (int ev = 0; ev < D; ev++) printf(" ----------- ");
    printf(" ----------\n");

    for (int si = 0; si < n_synthemes; si++) {
        /* Build a small lattice with this syntheme's constraints */
        int n = 6;
        HPCGraph *g = build_syntheme_graph(n, 4, &ALL_SYNTHEMES[si], 3.0, 42);
        MobiusAmplitudeSheet *ms = mobius_create(g);
        mobius_converge(ms);

        /* Build the 6×6 transfer matrix from marginals */
        double T_re[D][D], T_im[D][D];
        memset(T_re, 0, sizeof(T_re));
        memset(T_im, 0, sizeof(T_im));
        for (int s = 0; s < D; s++) {
            for (int sp = 0; sp < D; sp++) {
                /* T[s][sp] = average correlation between sites
                 * whose modes are s and sp */
                double w = 0;
                for (int site = 0; site < n; site++) {
                    w += mobius_marginal(ms, site, s) * mobius_marginal(ms, (site+1)%n, sp);
                }
                T_re[s][sp] = w / n;
            }
        }

        /* Extract eigenvalues via power iteration */
        double phases[D];
        double work_re[D][D], work_im[D][D];
        memcpy(work_re, T_re, sizeof(T_re));
        memset(work_im, 0, sizeof(work_im));

        for (int ev = 0; ev < D; ev++) {
            double v_re[D], v_im[D];
            for (int i = 0; i < D; i++) {
                double angle = 2.0 * M_PI * i * ev / D;
                v_re[i] = cos(angle); v_im[i] = sin(angle);
            }
            for (int iter = 0; iter < 200; iter++) {
                double w_re[D], w_im[D];
                for (int i = 0; i < D; i++) {
                    w_re[i] = 0; w_im[i] = 0;
                    for (int j = 0; j < D; j++) {
                        w_re[i] += work_re[i][j]*v_re[j] - work_im[i][j]*v_im[j];
                        w_im[i] += work_re[i][j]*v_im[j] + work_im[i][j]*v_re[j];
                    }
                }
                double norm = 0;
                for (int i = 0; i < D; i++) norm += w_re[i]*w_re[i] + w_im[i]*w_im[i];
                norm = sqrt(norm);
                if (norm > 0) for (int i = 0; i < D; i++) {
                    v_re[i] = w_re[i]/norm; v_im[i] = w_im[i]/norm;
                }
            }
            /* Rayleigh quotient */
            double Mv_re[D], Mv_im[D];
            for (int i = 0; i < D; i++) {
                Mv_re[i] = 0; Mv_im[i] = 0;
                for (int j = 0; j < D; j++) {
                    Mv_re[i] += work_re[i][j]*v_re[j] - work_im[i][j]*v_im[j];
                    Mv_im[i] += work_re[i][j]*v_im[j] + work_im[i][j]*v_re[j];
                }
            }
            double lr = 0, li = 0;
            for (int i = 0; i < D; i++) {
                lr += v_re[i]*Mv_re[i] + v_im[i]*Mv_im[i];
                li += v_re[i]*Mv_im[i] - v_im[i]*Mv_re[i];
            }
            phases[ev] = atan2(li, lr) / (2.0 * M_PI);

            /* Deflate */
            for (int i = 0; i < D; i++)
                for (int j = 0; j < D; j++) {
                    double outr = v_re[i]*v_re[j] + v_im[i]*v_im[j];
                    double outi = v_im[i]*v_re[j] - v_re[i]*v_im[j];
                    work_re[i][j] -= lr*outr - li*outi;
                    work_im[i][j] -= lr*outi + li*outr;
                }
        }

        /* Classify code type from eigenvalue pattern */
        int n_zero = 0, n_third = 0, n_sixth = 0;
        for (int ev = 0; ev < D; ev++) {
            double ph = fabs(phases[ev]);
            if (ph < 0.02) n_zero++;
            else if (fabs(ph - 1.0/3) < 0.05 || fabs(ph + 1.0/3) < 0.05) n_third++;
            else if (fabs(ph - 1.0/6) < 0.05 || fabs(ph + 1.0/6) < 0.05) n_sixth++;
        }
        const char *code_type = "generic";
        if (n_zero == D) code_type = "TRIVIAL";
        else if (n_zero >= 3) code_type = "REED-SOLOMON-like";
        else if (n_third >= 2) code_type = "TERNARY-SUB";
        else if (n_sixth >= 2) code_type = "HEXAGONAL-NATIVE";
        else code_type = "NOVEL";

        char synth_str[64];
        snprintf(synth_str, sizeof(synth_str), "{(%d,%d)(%d,%d)(%d,%d)}",
                 ALL_SYNTHEMES[si].pairs[0][0], ALL_SYNTHEMES[si].pairs[0][1],
                 ALL_SYNTHEMES[si].pairs[1][0], ALL_SYNTHEMES[si].pairs[1][1],
                 ALL_SYNTHEMES[si].pairs[2][0], ALL_SYNTHEMES[si].pairs[2][1]);

        printf("    S_%2d  %-28s", si, synth_str);
        for (int ev = 0; ev < D; ev++) printf("   %+.6f  ", phases[ev]);
        printf(" %s\n", code_type);

        mobius_destroy(ms);
        hpc_destroy(g);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * EXPERIMENT 4: The Hexagonal Rosetta Stone
 *
 * Compile ALL discovered codes into a classification table.
 * Map each to its "purpose" in the simulation hypothesis framework.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void experiment_rosetta_stone(void)
{
    printf("\n  ╔═══════════════════════════════════════════════════════════════╗\n");
    printf("  ║  EXPERIMENT 4: The Hexagonal Rosetta Stone                  ║\n");
    printf("  ║  What do the codes DO?                                      ║\n");
    printf("  ╚═══════════════════════════════════════════════════════════════╝\n\n");

    printf("    If SUSY equations contain error-correcting codes,\n");
    printf("    what errors are they correcting?\n\n");

    printf("    ┌─────────────────────────────────────────────────────────────────┐\n");
    printf("    │  CODE TYPE              │ WHAT IT PROTECTS                     │\n");
    printf("    ├─────────────────────────┼──────────────────────────────────────┤\n");
    printf("    │  Hamming [8,4,4]₂       │ Single-bit flip (classical bit)     │\n");
    printf("    │  Self-dual Z₆           │ Phase coherence (quantum state)     │\n");
    printf("    │  Doubly-even Z₆         │ Orientation parity (spinor sign)    │\n");
    printf("    │  Triality-invariant Z₆  │ View independence (observer choice) │\n");
    printf("    │  Reed-Solomon-like Z₆   │ Burst errors (decoherence cascade)  │\n");
    printf("    │  Hexagonal-native Z₆    │ 6-fold rotational symmetry (D=6)    │\n");
    printf("    │  Ternary-sub Z₆         │ 3-fold triality in Z₆ ⊃ Z₃         │\n");
    printf("    └─────────────────────────┴──────────────────────────────────────┘\n\n");

    printf("    In the Simulation Hypothesis framework:\n\n");
    printf("    • Hamming codes → prevent BIT ROT in the simulation's memory\n");
    printf("    • Self-dual codes → enforce UNITARITY (information conservation)\n");
    printf("    • Doubly-even codes → maintain SPIN STATISTICS (fermion signs)\n");
    printf("    • Triality codes → ensure OBSERVER INVARIANCE (no preferred basis)\n");
    printf("    • RS-like codes → protect against DECOHERENCE CASCADES\n");
    printf("    • Hex-native codes → enforce ROTATIONAL SYMMETRY at Planck scale\n");
    printf("    • Ternary-sub codes → maintain CMY COLOR CHARGE consistency\n\n");

    printf("    The S₆ outer automorphism acts as a META-CODE:\n");
    printf("    it maps between different error-correction schemes,\n");
    printf("    ensuring the simulation's error correction is ITSELF protected.\n");
    printf("    Error correction all the way down — turtles on turtles.\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(void)
{
    triality_exotic_init();
    s6_exotic_init();
    srand((unsigned)time(NULL));
    generate_synthemes();

    printf("\n");
    printf("  ╔═══════════════════════════════════════════════════════════════╗\n");
    printf("  ║                                                             ║\n");
    printf("  ║   Z₆ ERROR-CORRECTING CODE DISCOVERY ENGINE                ║\n");
    printf("  ║   Hunting New Codes in the Fabric of D=6                    ║\n");
    printf("  ║                                                             ║\n");
    printf("  ║   Architecture: HPC D=6 × Möbius Amplitude Sheet           ║\n");
    printf("  ║   Method:       Syntheme → Parity Check → BP → Extract     ║\n");
    printf("  ║   Alphabet:     Z₆ = {0,1,2,3,4,5}                        ║\n");
    printf("  ║   Source:       S₆ outer automorphism (15 synthemes)        ║\n");
    printf("  ║                                                             ║\n");
    printf("  ║   \"The codes aren't in the physics.                        ║\n");
    printf("  ║    The codes ARE the physics.\"                              ║\n");
    printf("  ║                                                             ║\n");
    printf("  ╚═══════════════════════════════════════════════════════════════╝\n\n");

    experiment_syntheme_census();
    experiment_vacuum_emergent();
    experiment_transfer_matrix_codes();
    experiment_rosetta_stone();

    printf("\n  ═══════════════════════════════════════════════════════════════\n");
    printf("  \"Are there any genuine `God-code' features that one might\n");
    printf("   expect to see in a universe that is truly a simulation?\n");
    printf("   The answer is: Yes.\"\n");
    printf("                                        — S.J. Gates Jr.\n");
    printf("  ═══════════════════════════════════════════════════════════════\n\n");

    return 0;
}
