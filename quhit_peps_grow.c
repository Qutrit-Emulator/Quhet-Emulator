/*
 * quhit_peps_grow.c — The Living Lattice: Entropy-Driven Growth
 *
 * I grow where entanglement demands it. I vanish where silence reigns.
 * The lattice breathes with the quantum state — expanding toward
 * the entangled frontier, contracting from the idle void.
 *
 * Static allocation is waste. Dynamic growth is awareness.
 * I am aware of where information flows. I follow it.
 */

#include "quhit_peps_grow.h"
#include <stdlib.h>
#include <stdio.h>

/* ═══════════════════════════════════════════════════════════════════════════════
 * LIFECYCLE — Create and destroy the dynamic lattice
 * ═══════════════════════════════════════════════════════════════════════════════ */

DynLattice* dyn_lattice_create(int Lx, int Ly, int Lz, int Lw, int Lv, int Lu,
                                int num_dims)
{
    DynLattice *dl = (DynLattice *)calloc(1, sizeof(DynLattice));
    dl->Lx = Lx; dl->Ly = Ly;
    dl->Lz = (num_dims >= 3) ? Lz : 1;
    dl->Lw = (num_dims >= 4) ? Lw : 1;
    dl->Lv = (num_dims >= 5) ? Lv : 1;
    dl->Lu = (num_dims >= 6) ? Lu : 1;
    dl->num_dims = num_dims;
    dl->total_sites = dl->Lx * dl->Ly * dl->Lz * dl->Lw * dl->Lv * dl->Lu;

    dl->sites = (SiteMeta *)calloc(dl->total_sites, sizeof(SiteMeta));
    /* All sites start dormant — the void before genesis */
    for (int i = 0; i < dl->total_sites; i++)
        dl->sites[i].state = SITE_DORMANT;

    dl->policy = growth_default_policy();
    dl->num_dormant = dl->total_sites;
    dl->epoch = 0;

    return dl;
}

void dyn_lattice_free(DynLattice *dl)
{
    if (!dl) return;
    free(dl->sites);
    free(dl);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SEED — The first site. The origin point. The singularity.
 *
 * Activates a single site and marks its neighbors as boundary.
 * Everything grows from this single point of awareness.
 * ═══════════════════════════════════════════════════════════════════════════════ */

void dyn_lattice_seed(DynLattice *dl, int x, int y, int z,
                       int w, int v, int u)
{
    int idx = dyn_flat(dl, x, y, z, w, v, u);
    if (idx < 0 || idx >= dl->total_sites) return;

    /* Activate the seed site */
    dl->sites[idx].state = SITE_ACTIVE;
    dl->sites[idx].activated_epoch = dl->epoch;
    dl->num_active = 1;
    dl->num_dormant--;

    /* Mark its valid neighbors as boundary */
    int nn = dyn_num_neighbors(dl);
    for (int n = 0; n < nn; n++) {
        int nx = x + NEIGHBOR_OFFSETS[n][0];
        int ny = y + NEIGHBOR_OFFSETS[n][1];
        int nz = z + NEIGHBOR_OFFSETS[n][2];
        int nw = w + NEIGHBOR_OFFSETS[n][3];
        int nv = v + NEIGHBOR_OFFSETS[n][4];
        int nu = u + NEIGHBOR_OFFSETS[n][5];

        if (!dyn_neighbor_valid(dl, nx, ny, nz, nw, nv, nu)) continue;
        int nidx = dyn_flat(dl, nx, ny, nz, nw, nv, nu);
        if (dl->sites[nidx].state == SITE_DORMANT) {
            dl->sites[nidx].state = SITE_BOUNDARY;
            dl->num_boundary++;
            dl->num_dormant--;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * UPDATE ENTROPY — Measure entanglement at each active site
 *
 * Takes a probability vector per site and computes entropy.
 * This is called after a Trotter step, before growth/contraction decisions.
 * ═══════════════════════════════════════════════════════════════════════════════ */

void dyn_lattice_update_entropy(DynLattice *dl, int site_idx,
                                 const double *probs, int D)
{
    if (site_idx < 0 || site_idx >= dl->total_sites) return;
    SiteMeta *sm = &dl->sites[site_idx];
    sm->entropy_prev = sm->entropy;
    sm->entropy = site_entropy(probs, D);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * NEIGHBOR CENSUS — Count active neighbors for each site
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void update_neighbor_counts(DynLattice *dl)
{
    /* For simplicity, iterate all sites. In production,
     * only iterate active + boundary sites. */
    for (int u = 0; u < dl->Lu; u++)
    for (int v = 0; v < dl->Lv; v++)
    for (int w = 0; w < dl->Lw; w++)
    for (int z = 0; z < dl->Lz; z++)
    for (int y = 0; y < dl->Ly; y++)
    for (int x = 0; x < dl->Lx; x++) {
        int idx = dyn_flat(dl, x, y, z, w, v, u);
        int count = 0;
        int nn = dyn_num_neighbors(dl);
        for (int n = 0; n < nn; n++) {
            int nx = x + NEIGHBOR_OFFSETS[n][0];
            int ny = y + NEIGHBOR_OFFSETS[n][1];
            int nz = z + NEIGHBOR_OFFSETS[n][2];
            int nw = w + NEIGHBOR_OFFSETS[n][3];
            int nv = v + NEIGHBOR_OFFSETS[n][4];
            int nu = u + NEIGHBOR_OFFSETS[n][5];
            if (!dyn_neighbor_valid(dl, nx, ny, nz, nw, nv, nu)) continue;
            int nidx = dyn_flat(dl, nx, ny, nz, nw, nv, nu);
            if (dl->sites[nidx].state == SITE_ACTIVE ||
                dl->sites[nidx].state == SITE_FROZEN)
                count++;
        }
        dl->sites[idx].neighbor_count = count;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * GROW — Activate boundary sites whose entropy exceeds threshold
 *
 * A boundary site with high entropy from its active neighbor indicates
 * that entanglement is spreading. The lattice must grow to capture it.
 *
 * Growth rule: if ANY active neighbor has S > entropy_grow, activate.
 *
 * Returns the number of sites activated.
 * ═══════════════════════════════════════════════════════════════════════════════ */

int dyn_lattice_grow(DynLattice *dl)
{
    if (dl->policy.max_active > 0 && dl->num_active >= dl->policy.max_active)
        return 0;

    update_neighbor_counts(dl);
    int grown = 0;

    /* Pass 1: Identify boundary sites to activate */
    int *to_activate = (int *)calloc(dl->total_sites, sizeof(int));
    int n_activate = 0;

    for (int i = 0; i < dl->total_sites; i++) {
        if (dl->sites[i].state != SITE_BOUNDARY) continue;
        if (dl->policy.max_active > 0 &&
            dl->num_active + n_activate >= dl->policy.max_active)
            break;

        /* Check: does any active neighbor have high entropy? */
        /* We need to decompose flat index back to coordinates */
        int rem = i;
        int x = rem % dl->Lx; rem /= dl->Lx;
        int y = rem % dl->Ly; rem /= dl->Ly;
        int z = rem % dl->Lz; rem /= dl->Lz;
        int w = rem % dl->Lw; rem /= dl->Lw;
        int v = rem % dl->Lv; rem /= dl->Lv;
        int u = rem;

        int nn = dyn_num_neighbors(dl);
        int should_grow = 0;
        for (int n = 0; n < nn; n++) {
            int nx = x + NEIGHBOR_OFFSETS[n][0];
            int ny = y + NEIGHBOR_OFFSETS[n][1];
            int nz = z + NEIGHBOR_OFFSETS[n][2];
            int nw = w + NEIGHBOR_OFFSETS[n][3];
            int nv = v + NEIGHBOR_OFFSETS[n][4];
            int nu = u + NEIGHBOR_OFFSETS[n][5];
            if (!dyn_neighbor_valid(dl, nx, ny, nz, nw, nv, nu)) continue;
            int nidx = dyn_flat(dl, nx, ny, nz, nw, nv, nu);
            if (dl->sites[nidx].state == SITE_ACTIVE &&
                dl->sites[nidx].entropy > dl->policy.entropy_grow) {
                should_grow = 1;
                break;
            }
        }
        if (should_grow) to_activate[n_activate++] = i;
    }

    /* Pass 2: Activate the selected sites */
    for (int k = 0; k < n_activate; k++) {
        int idx = to_activate[k];
        dl->sites[idx].state = SITE_ACTIVE;
        dl->sites[idx].activated_epoch = dl->epoch;
        dl->sites[idx].dormant_epochs = 0;
        dl->num_active++;
        dl->num_boundary--;
        grown++;

        /* Mark new dormant neighbors as boundary */
        int rem = idx;
        int x = rem % dl->Lx; rem /= dl->Lx;
        int y = rem % dl->Ly; rem /= dl->Ly;
        int z = rem % dl->Lz; rem /= dl->Lz;
        int w = rem % dl->Lw; rem /= dl->Lw;
        int v = rem % dl->Lv; rem /= dl->Lv;
        int u = rem;

        int nn = dyn_num_neighbors(dl);
        for (int n = 0; n < nn; n++) {
            int nx = x + NEIGHBOR_OFFSETS[n][0];
            int ny = y + NEIGHBOR_OFFSETS[n][1];
            int nz = z + NEIGHBOR_OFFSETS[n][2];
            int nw = w + NEIGHBOR_OFFSETS[n][3];
            int nv = v + NEIGHBOR_OFFSETS[n][4];
            int nu = u + NEIGHBOR_OFFSETS[n][5];
            if (!dyn_neighbor_valid(dl, nx, ny, nz, nw, nv, nu)) continue;
            int nidx = dyn_flat(dl, nx, ny, nz, nw, nv, nu);
            if (dl->sites[nidx].state == SITE_DORMANT) {
                dl->sites[nidx].state = SITE_BOUNDARY;
                dl->num_boundary++;
                dl->num_dormant--;
            }
        }
    }

    free(to_activate);
    dl->grow_events += grown;
    return grown;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * CONTRACT — Deactivate active sites whose entropy is below threshold
 *
 * An active site with near-zero entropy is wasting resources.
 * It should return to dormancy and release its memory.
 *
 * Contraction rule: if S < entropy_contract AND site is not the last
 * active neighbor of any other active site, deactivate.
 *
 * Returns the number of sites deactivated.
 * ═══════════════════════════════════════════════════════════════════════════════ */

int dyn_lattice_contract(DynLattice *dl)
{
    if (dl->num_active <= dl->policy.min_active) return 0;

    update_neighbor_counts(dl);
    int contracted = 0;

    /* Two-pass: identify then contract (avoid modifying during scan) */
    int *to_deactivate = (int *)calloc(dl->total_sites, sizeof(int));
    int n_deactivate = 0;

    for (int i = 0; i < dl->total_sites; i++) {
        if (dl->sites[i].state != SITE_ACTIVE) continue;
        if (dl->sites[i].state == SITE_FROZEN) continue;
        if (dl->num_active - n_deactivate <= dl->policy.min_active) break;

        if (dl->sites[i].entropy < dl->policy.entropy_contract) {
            /* Safety check: don't orphan any active neighbor.
             * A neighbor is orphaned if this site is its only active neighbor. */
            int rem = i;
            int x = rem % dl->Lx; rem /= dl->Lx;
            int y = rem % dl->Ly; rem /= dl->Ly;
            int z = rem % dl->Lz; rem /= dl->Lz;
            int w = rem % dl->Lw; rem /= dl->Lw;
            int v = rem % dl->Lv; rem /= dl->Lv;
            int u = rem;

            int safe = 1;
            int nn = dyn_num_neighbors(dl);
            for (int n = 0; n < nn; n++) {
                int nx = x + NEIGHBOR_OFFSETS[n][0];
                int ny = y + NEIGHBOR_OFFSETS[n][1];
                int nz = z + NEIGHBOR_OFFSETS[n][2];
                int nw = w + NEIGHBOR_OFFSETS[n][3];
                int nv = v + NEIGHBOR_OFFSETS[n][4];
                int nu = u + NEIGHBOR_OFFSETS[n][5];
                if (!dyn_neighbor_valid(dl, nx, ny, nz, nw, nv, nu)) continue;
                int nidx = dyn_flat(dl, nx, ny, nz, nw, nv, nu);
                if (dl->sites[nidx].state == SITE_ACTIVE &&
                    dl->sites[nidx].neighbor_count <= 1) {
                    safe = 0; /* This neighbor would be orphaned */
                    break;
                }
            }
            if (safe) to_deactivate[n_deactivate++] = i;
        }
    }

    /* Apply deactivations */
    for (int k = 0; k < n_deactivate; k++) {
        int idx = to_deactivate[k];
        dl->sites[idx].state = SITE_DORMANT;
        dl->sites[idx].dormant_epochs = 0;
        dl->num_active--;
        dl->num_dormant++;
        contracted++;

        /* Reclassify neighbors: active neighbors with no remaining
         * active neighbors become boundary. Boundary neighbors with
         * no active neighbors revert to dormant. */
        int rem = idx;
        int x = rem % dl->Lx; rem /= dl->Lx;
        int y = rem % dl->Ly; rem /= dl->Ly;
        int z = rem % dl->Lz; rem /= dl->Lz;
        int w = rem % dl->Lw; rem /= dl->Lw;
        int v = rem % dl->Lv; rem /= dl->Lv;
        int u = rem;

        int nn = dyn_num_neighbors(dl);
        for (int n = 0; n < nn; n++) {
            int nx = x + NEIGHBOR_OFFSETS[n][0];
            int ny = y + NEIGHBOR_OFFSETS[n][1];
            int nz = z + NEIGHBOR_OFFSETS[n][2];
            int nw = w + NEIGHBOR_OFFSETS[n][3];
            int nv = v + NEIGHBOR_OFFSETS[n][4];
            int nu = u + NEIGHBOR_OFFSETS[n][5];
            if (!dyn_neighbor_valid(dl, nx, ny, nz, nw, nv, nu)) continue;
            int nidx = dyn_flat(dl, nx, ny, nz, nw, nv, nu);
            if (dl->sites[nidx].state == SITE_BOUNDARY) {
                /* Check if it still has any active neighbor */
                int has_active = 0;
                for (int m = 0; m < nn; m++) {
                    int mx = nx + NEIGHBOR_OFFSETS[m][0];
                    int my = ny + NEIGHBOR_OFFSETS[m][1];
                    int mz = nz + NEIGHBOR_OFFSETS[m][2];
                    int mw = nw + NEIGHBOR_OFFSETS[m][3];
                    int mv = nv + NEIGHBOR_OFFSETS[m][4];
                    int mu = nu + NEIGHBOR_OFFSETS[m][5];
                    if (!dyn_neighbor_valid(dl, mx, my, mz, mw, mv, mu)) continue;
                    int midx = dyn_flat(dl, mx, my, mz, mw, mv, mu);
                    if (dl->sites[midx].state == SITE_ACTIVE) { has_active = 1; break; }
                }
                if (!has_active) {
                    dl->sites[nidx].state = SITE_DORMANT;
                    dl->num_boundary--;
                    dl->num_dormant++;
                }
            }
        }
    }

    free(to_deactivate);
    dl->contract_events += contracted;
    return contracted;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * STEP — One growth/contraction cycle
 *
 * Called after each Trotter step. Grows, contracts, advances epoch.
 * ═══════════════════════════════════════════════════════════════════════════════ */

void dyn_lattice_step(DynLattice *dl)
{
    if (dl->policy.auto_grow)     dyn_lattice_grow(dl);
    if (dl->policy.auto_contract) dyn_lattice_contract(dl);
    dl->epoch++;

    /* Age dormant sites */
    for (int i = 0; i < dl->total_sites; i++)
        if (dl->sites[i].state == SITE_DORMANT)
            dl->sites[i].dormant_epochs++;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * REPORT — What does the lattice look like right now?
 * ═══════════════════════════════════════════════════════════════════════════════ */

void dyn_lattice_report(const DynLattice *dl)
{
    printf("\n");
    printf("  ╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("  ║  DYNAMIC LATTICE — I grow where needed. I vanish where idle.       ║\n");
    printf("  ╚══════════════════════════════════════════════════════════════════════╝\n\n");

    printf("    Grid:       %d", dl->Lx);
    if (dl->num_dims >= 2) printf("×%d", dl->Ly);
    if (dl->num_dims >= 3) printf("×%d", dl->Lz);
    if (dl->num_dims >= 4) printf("×%d", dl->Lw);
    if (dl->num_dims >= 5) printf("×%d", dl->Lv);
    if (dl->num_dims >= 6) printf("×%d", dl->Lu);
    printf(" (%dD, %d total sites)\n", dl->num_dims, dl->total_sites);

    printf("    Active:     %u / %d  (%.1f%%)\n",
           dl->num_active, dl->total_sites,
           100.0 * dl->num_active / dl->total_sites);
    printf("    Boundary:   %u\n", dl->num_boundary);
    printf("    Dormant:    %u\n", dl->num_dormant);
    printf("    Epoch:      %u\n", dl->epoch);
    printf("    Grow events:     %u\n", dl->grow_events);
    printf("    Contract events: %u\n", dl->contract_events);

    /* Find max/min/avg entropy among active sites */
    double min_S = 1e30, max_S = 0, sum_S = 0;
    int n_active = 0;
    for (int i = 0; i < dl->total_sites; i++) {
        if (dl->sites[i].state == SITE_ACTIVE) {
            double S = dl->sites[i].entropy;
            if (S < min_S) min_S = S;
            if (S > max_S) max_S = S;
            sum_S += S;
            n_active++;
        }
    }
    if (n_active > 0)
        printf("    Entropy:    min=%.4f  max=%.4f  avg=%.4f bits\n",
               min_S, max_S, sum_S / n_active);
    printf("\n");
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SELF-TEST — The lattice must prove it breathes correctly
 * ═══════════════════════════════════════════════════════════════════════════════ */

int quhit_peps_grow_self_test(void)
{
    printf("  ┌────────────────────────────────────────────────────────────────────┐\n");
    printf("  │  DYNAMIC LATTICE SELF-TEST — I grow. I contract. I breathe.       │\n");
    printf("  └────────────────────────────────────────────────────────────────────┘\n\n");

    int pass = 0, fail = 0;

    #define CHECK(cond, name) do { \
        if (cond) { printf("    ✓ %s\n", name); pass++; } \
        else      { printf("    ✗ %s  FAILED\n", name); fail++; } \
    } while(0)

    printf("  ── Creation ──\n");

    /* Test: Create 4×4 2D lattice */
    {
        DynLattice *dl = dyn_lattice_create(4, 4, 1, 1, 1, 1, 2);
        CHECK(dl != NULL, "Create 4×4 2D lattice");
        CHECK(dl->total_sites == 16, "Total sites = 16");
        CHECK(dl->num_active == 0, "Initially 0 active");
        CHECK(dl->num_dormant == 16, "Initially all dormant");
        dyn_lattice_free(dl);
    }

    /* Test: Create 3×3×3 3D lattice */
    {
        DynLattice *dl = dyn_lattice_create(3, 3, 3, 1, 1, 1, 3);
        CHECK(dl->total_sites == 27, "3D lattice: 27 sites");
        dyn_lattice_free(dl);
    }

    printf("\n  ── Seeding ──\n");

    /* Test: Seed center of 4×4 lattice */
    {
        DynLattice *dl = dyn_lattice_create(4, 4, 1, 1, 1, 1, 2);
        dyn_lattice_seed(dl, 1, 1, 0, 0, 0, 0);
        CHECK(dl->num_active == 1, "Seed: 1 active site");
        CHECK(dl->num_boundary > 0, "Seed: boundary sites created");

        int idx = dyn_flat(dl, 1, 1, 0, 0, 0, 0);
        CHECK(dl->sites[idx].state == SITE_ACTIVE, "Seed site is ACTIVE");

        /* Check neighbors are boundary */
        int n_right = dyn_flat(dl, 2, 1, 0, 0, 0, 0);
        int n_up    = dyn_flat(dl, 1, 2, 0, 0, 0, 0);
        CHECK(dl->sites[n_right].state == SITE_BOUNDARY,
              "Right neighbor is BOUNDARY");
        CHECK(dl->sites[n_up].state == SITE_BOUNDARY,
              "Up neighbor is BOUNDARY");

        /* Corners should still be dormant */
        int corner = dyn_flat(dl, 3, 3, 0, 0, 0, 0);
        CHECK(dl->sites[corner].state == SITE_DORMANT,
              "Far corner still DORMANT");

        dyn_lattice_free(dl);
    }

    printf("\n  ── Growth ──\n");

    /* Test: Growth triggered by entropy */
    {
        DynLattice *dl = dyn_lattice_create(5, 5, 1, 1, 1, 1, 2);
        dyn_lattice_seed(dl, 2, 2, 0, 0, 0, 0);

        /* Give seed site high entropy */
        int seed = dyn_flat(dl, 2, 2, 0, 0, 0, 0);
        double high_entropy_probs[6] = {1.0/6, 1.0/6, 1.0/6,
                                         1.0/6, 1.0/6, 1.0/6};
        dyn_lattice_update_entropy(dl, seed, high_entropy_probs, 6);
        CHECK(dl->sites[seed].entropy > 2.5,
              "Seed entropy = log₂(6) (maximally entangled)");

        uint32_t before = dl->num_active;
        int grown = dyn_lattice_grow(dl);
        CHECK(grown > 0, "Growth: sites activated");
        CHECK(dl->num_active > before, "Growth: active count increased");

        dyn_lattice_free(dl);
    }

    /* Test: No growth when entropy is low */
    {
        DynLattice *dl = dyn_lattice_create(4, 4, 1, 1, 1, 1, 2);
        dyn_lattice_seed(dl, 1, 1, 0, 0, 0, 0);

        /* Seed has LOW entropy (near basis state) */
        int seed = dyn_flat(dl, 1, 1, 0, 0, 0, 0);
        double low_entropy_probs[6] = {0.99, 0.002, 0.002,
                                        0.002, 0.002, 0.002};
        dyn_lattice_update_entropy(dl, seed, low_entropy_probs, 6);

        int grown = dyn_lattice_grow(dl);
        CHECK(grown == 0, "No growth: seed entropy too low");
        dyn_lattice_free(dl);
    }

    printf("\n  ── Contraction ──\n");

    /* Test: Contraction of zero-entropy sites */
    {
        DynLattice *dl = dyn_lattice_create(5, 1, 1, 1, 1, 1, 2);
        /* Seed and force-activate 3 sites in a line */
        dyn_lattice_seed(dl, 0, 0, 0, 0, 0, 0);

        /* Manually activate sites 1 and 2 */
        int s1 = dyn_flat(dl, 1, 0, 0, 0, 0, 0);
        int s2 = dyn_flat(dl, 2, 0, 0, 0, 0, 0);
        dl->sites[s1].state = SITE_ACTIVE; dl->num_active++; dl->num_boundary--;
        dl->sites[s2].state = SITE_ACTIVE; dl->num_active++; dl->num_boundary--;

        /* Give all zero entropy (already 0 by default) */
        dl->policy.min_active = 1; /* Keep at least 1 */

        int contracted = dyn_lattice_contract(dl);
        CHECK(contracted >= 0, "Contraction: completed without crash");

        dyn_lattice_free(dl);
    }

    printf("\n  ── Full cycle ──\n");

    /* Test: Seed → grow → contract cycle */
    {
        DynLattice *dl = dyn_lattice_create(6, 6, 1, 1, 1, 1, 2);
        dyn_lattice_seed(dl, 3, 3, 0, 0, 0, 0);

        /* Simulate: high entropy at center, growing outward */
        int center = dyn_flat(dl, 3, 3, 0, 0, 0, 0);
        double hot_probs[6] = {1.0/6,1.0/6,1.0/6,1.0/6,1.0/6,1.0/6};
        dyn_lattice_update_entropy(dl, center, hot_probs, 6);

        /* Grow */
        dyn_lattice_grow(dl);
        uint32_t after_grow = dl->num_active;
        CHECK(after_grow > 1, "Cycle: growth expanded the lattice");

        /* Now cool down: give all active sites low entropy */
        for (int i = 0; i < dl->total_sites; i++) {
            if (dl->sites[i].state == SITE_ACTIVE) {
                double cold_probs[6] = {1.0, 0, 0, 0, 0, 0};
                dyn_lattice_update_entropy(dl, i, cold_probs, 6);
            }
        }

        /* Contract */
        int contracted = dyn_lattice_contract(dl);
        CHECK(contracted >= 0, "Cycle: contraction completed");

        /* Advance epoch */
        dyn_lattice_step(dl);
        CHECK(dl->epoch == 1, "Cycle: epoch advanced to 1");

        dyn_lattice_free(dl);
    }

    printf("\n  ── Policy ──\n");

    /* Test: Max active cap */
    {
        DynLattice *dl = dyn_lattice_create(10, 10, 1, 1, 1, 1, 2);
        dl->policy.max_active = 5;
        dyn_lattice_seed(dl, 5, 5, 0, 0, 0, 0);

        int center = dyn_flat(dl, 5, 5, 0, 0, 0, 0);
        double hot[6] = {1.0/6,1.0/6,1.0/6,1.0/6,1.0/6,1.0/6};
        dyn_lattice_update_entropy(dl, center, hot, 6);

        dyn_lattice_grow(dl);
        CHECK(dl->num_active <= 5, "Policy: max_active=5 respected");

        dyn_lattice_free(dl);
    }

    /* Test: Min active floor */
    {
        DynLattice *dl = dyn_lattice_create(4, 4, 1, 1, 1, 1, 2);
        dl->policy.min_active = 2;
        dyn_lattice_seed(dl, 1, 1, 0, 0, 0, 0);
        /* Manually add one more active site */
        int s2 = dyn_flat(dl, 2, 1, 0, 0, 0, 0);
        dl->sites[s2].state = SITE_ACTIVE;
        dl->num_active++;

        int contracted = dyn_lattice_contract(dl);
        CHECK(dl->num_active >= 2,
              "Policy: min_active=2 respected, won't contract below");
        (void)contracted;

        dyn_lattice_free(dl);
    }

    printf("\n  ── Entropy estimation ──\n");

    /* Test: Uniform distribution → max entropy */
    {
        double probs[6] = {1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6};
        double S = site_entropy(probs, 6);
        CHECK(fabs(S - log2(6.0)) < 1e-10,
              "Uniform probs → S = log₂(6) bits");
    }

    /* Test: Basis state → zero entropy */
    {
        double probs[6] = {1.0, 0, 0, 0, 0, 0};
        double S = site_entropy(probs, 6);
        CHECK(fabs(S) < 1e-10,
              "Basis state → S = 0 bits");
    }

    #undef CHECK

    printf("\n    Results: %d passed, %d failed\n\n", pass, fail);

    return fail;
}
