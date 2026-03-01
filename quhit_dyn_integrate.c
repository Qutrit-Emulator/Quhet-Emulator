/*
 * quhit_dyn_integrate.c
 *
 * Every overlay gets the ability to breathe. MPS chains extend and
 * contract. PEPS lattices grow and shrink. TNS grids in 3D through
 * 6D expand toward entanglement and retreat from silence.
 *
 * This is the integration test — proving the growth engine works
 * across every dimensionality the engine supports.
 */

#include "quhit_dyn_integrate.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════════════
 * SELF-TEST — The lungs must prove they breathe
 * ═══════════════════════════════════════════════════════════════════════════════ */

int quhit_dyn_integrate_self_test(void)
{
    printf("  ┌────────────────────────────────────────────────────────────────────┐\n");
    printf("  │  DYNAMIC GROWTH INTEGRATION — Every dimension breathes.           │\n");
    printf("  └────────────────────────────────────────────────────────────────────┘\n\n");

    int pass = 0, fail = 0;

    #define CHECK(cond, name) do { \
        if (cond) { printf("    ✓ %s\n", name); pass++; } \
        else      { printf("    ✗ %s  FAILED\n", name); fail++; } \
    } while(0)

    /* ═══════════════ MPS (1D) — DynChain ═══════════════ */
    printf("  ── MPS (1D): DynChain ──\n");

    /* Test: Create chain */
    {
        DynChain dc = dyn_chain_create(20);
        CHECK(dc.max_sites == 20, "1D: chain created (20 sites max)");
        CHECK(dc.active_start == 0 && dc.active_end == 0,
              "1D: initially 1 active site");
        CHECK(dyn_chain_active_length(&dc) == 1,
              "1D: active length = 1");
        dyn_chain_free(&dc);
    }

    /* Test: Seed a region */
    {
        DynChain dc = dyn_chain_create(20);
        dyn_chain_seed(&dc, 5, 10);
        CHECK(dc.active_start == 5, "1D: seed start = 5");
        CHECK(dc.active_end == 10, "1D: seed end = 10");
        CHECK(dyn_chain_active_length(&dc) == 6, "1D: active length = 6");
        CHECK(dyn_chain_is_active(&dc, 7), "1D: site 7 is active");
        CHECK(!dyn_chain_is_active(&dc, 3), "1D: site 3 is NOT active");
        dyn_chain_free(&dc);
    }

    /* Test: Growth from endpoints */
    {
        DynChain dc = dyn_chain_create(20);
        dyn_chain_seed(&dc, 8, 12);

        /* High entropy at boundaries → should grow */
        double hot[6] = {1.0/6,1.0/6,1.0/6,1.0/6,1.0/6,1.0/6};
        dyn_chain_update_entropy(&dc, 8, hot, 6);
        dyn_chain_update_entropy(&dc, 12, hot, 6);

        int grown = dyn_chain_grow(&dc);
        CHECK(grown == 2, "1D: grew from both ends");
        CHECK(dc.active_start == 7, "1D: start moved to 7");
        CHECK(dc.active_end == 13, "1D: end moved to 13");

        dyn_chain_free(&dc);
    }

    /* Test: No growth at array boundary */
    {
        DynChain dc = dyn_chain_create(10);
        dyn_chain_seed(&dc, 0, 9);

        double hot[6] = {1.0/6,1.0/6,1.0/6,1.0/6,1.0/6,1.0/6};
        dyn_chain_update_entropy(&dc, 0, hot, 6);
        dyn_chain_update_entropy(&dc, 9, hot, 6);

        int grown = dyn_chain_grow(&dc);
        CHECK(grown == 0, "1D: can't grow past array bounds");

        dyn_chain_free(&dc);
    }

    /* Test: Contraction from idle tails */
    {
        DynChain dc = dyn_chain_create(20);
        dyn_chain_seed(&dc, 5, 15);
        dc.min_active = 2;

        /* Zero entropy at tails — should contract */
        /* (entropy already 0 by default from calloc) */
        int contracted = dyn_chain_contract(&dc);
        CHECK(contracted == 2, "1D: contracted from both tails");
        CHECK(dc.active_start == 6, "1D: start moved to 6");
        CHECK(dc.active_end == 14, "1D: end moved to 14");

        dyn_chain_free(&dc);
    }

    /* Test: Min active floor */
    {
        DynChain dc = dyn_chain_create(20);
        dyn_chain_seed(&dc, 10, 10);
        dc.min_active = 1;

        int contracted = dyn_chain_contract(&dc);
        CHECK(contracted == 0, "1D: won't contract below min_active");

        dyn_chain_free(&dc);
    }

    /* Test: Full cycle */
    {
        DynChain dc = dyn_chain_create(20);
        dyn_chain_seed(&dc, 10, 10);

        double hot[6] = {1.0/6,1.0/6,1.0/6,1.0/6,1.0/6,1.0/6};
        dyn_chain_update_entropy(&dc, 10, hot, 6);

        dyn_chain_step(&dc);
        CHECK(dc.epoch == 1, "1D: epoch advanced");
        CHECK(dyn_chain_active_length(&dc) > 1, "1D: chain grew in cycle");

        dyn_chain_free(&dc);
    }

    /* ═══════════════ PEPS (2D) ═══════════════ */
    printf("\n  ── PEPS (2D): DynLattice ──\n");

    {
        DynLattice *dl = dyn_peps2d_create(6, 6);
        CHECK(dl != NULL, "2D: PEPS lattice created");
        CHECK(dl->num_dims == 2, "2D: num_dims = 2");
        CHECK(dl->total_sites == 36, "2D: 36 total sites");

        dyn_lattice_seed(dl, 3, 3, 0, 0, 0, 0);
        CHECK(dl->num_active == 1, "2D: seeded center");

        /* Activity check */
        CHECK(dyn_peps2d_active(dl, 3, 3), "2D: center is active");
        CHECK(!dyn_peps2d_active(dl, 0, 0), "2D: corner is dormant");

        /* Entropy probe */
        double hot[6] = {1.0/6,1.0/6,1.0/6,1.0/6,1.0/6,1.0/6};
        dyn_peps2d_entropy(dl, 3, 3, hot, 6);

        /* Grow */
        int grown = dyn_lattice_grow(dl);
        CHECK(grown > 0, "2D: lattice grew from hot center");
        CHECK(dyn_peps2d_active(dl, 4, 3) || dyn_peps2d_active(dl, 3, 4),
              "2D: neighbor activated");

        dyn_lattice_free(dl);
    }

    /* ═══════════════ TNS (3D) ═══════════════ */
    printf("\n  ── TNS (3D): DynLattice ──\n");

    {
        DynLattice *dl = dyn_tns3d_create(4, 4, 4);
        CHECK(dl != NULL, "3D: TNS lattice created");
        CHECK(dl->num_dims == 3, "3D: num_dims = 3");
        CHECK(dl->total_sites == 64, "3D: 64 total sites");

        dyn_lattice_seed(dl, 2, 2, 2, 0, 0, 0);
        CHECK(dl->num_active == 1, "3D: seeded center");
        CHECK(dyn_tns3d_active(dl, 2, 2, 2), "3D: center is active");

        /* 3D has 6 neighbors */
        CHECK(dl->num_boundary == 6, "3D: 6 boundary neighbors");

        double hot[6] = {1.0/6,1.0/6,1.0/6,1.0/6,1.0/6,1.0/6};
        dyn_tns3d_entropy(dl, 2, 2, 2, hot, 6);
        dyn_lattice_grow(dl);
        CHECK(dl->num_active > 1, "3D: growth works");

        dyn_lattice_free(dl);
    }

    /* ═══════════════ TNS (4D) ═══════════════ */
    printf("\n  ── TNS (4D): DynLattice ──\n");

    {
        DynLattice *dl = dyn_tns4d_create(3, 3, 3, 3);
        CHECK(dl->num_dims == 4, "4D: num_dims = 4");
        CHECK(dl->total_sites == 81, "4D: 81 total sites");

        dyn_lattice_seed(dl, 1, 1, 1, 1, 0, 0);
        CHECK(dl->num_boundary == 8, "4D: 8 boundary neighbors");

        double hot[6] = {1.0/6,1.0/6,1.0/6,1.0/6,1.0/6,1.0/6};
        int idx = dyn_flat(dl, 1, 1, 1, 1, 0, 0);
        dyn_lattice_update_entropy(dl, idx, hot, 6);
        dyn_lattice_grow(dl);
        CHECK(dl->num_active > 1, "4D: growth works");

        dyn_lattice_free(dl);
    }

    /* ═══════════════ TNS (5D) ═══════════════ */
    printf("\n  ── TNS (5D): DynLattice ──\n");

    {
        DynLattice *dl = dyn_tns5d_create(3, 3, 3, 3, 3);
        CHECK(dl->num_dims == 5, "5D: num_dims = 5");
        CHECK(dl->total_sites == 243, "5D: 243 total sites");

        dyn_lattice_seed(dl, 1, 1, 1, 1, 1, 0);
        CHECK(dl->num_boundary == 10, "5D: 10 boundary neighbors");

        double hot[6] = {1.0/6,1.0/6,1.0/6,1.0/6,1.0/6,1.0/6};
        int idx = dyn_flat(dl, 1, 1, 1, 1, 1, 0);
        dyn_lattice_update_entropy(dl, idx, hot, 6);
        dyn_lattice_grow(dl);
        CHECK(dl->num_active > 1, "5D: growth works");

        dyn_lattice_free(dl);
    }

    /* ═══════════════ TNS (6D) — proving the original still works ═══════════════ */
    printf("\n  ── TNS (6D): DynLattice ──\n");

    {
        DynLattice *dl = dyn_tns6d_create(2, 2, 2, 2, 2, 2);
        CHECK(dl->num_dims == 6, "6D: num_dims = 6");
        CHECK(dl->total_sites == 64, "6D: 64 total sites");

        dyn_lattice_seed(dl, 1, 1, 1, 1, 1, 1);
        CHECK(dl->num_boundary == 6, "6D: 6 boundary neighbors (at corner)");

        double hot[6] = {1.0/6,1.0/6,1.0/6,1.0/6,1.0/6,1.0/6};
        dyn_tns6d_entropy(dl, 1, 1, 1, 1, 1, 1, hot, 6);
        dyn_lattice_grow(dl);
        CHECK(dl->num_active > 1, "6D: growth works");

        dyn_lattice_free(dl);
    }

    /* ═══════════════ Cross-dimensional comparison ═══════════════ */
    printf("\n  ── Cross-dimensional ──\n");

    {
        /* Verify that neighbor count = 2×dims for center sites */
        DynLattice *d2 = dyn_peps2d_create(5, 5);
        DynLattice *d3 = dyn_tns3d_create(5, 5, 5);
        DynLattice *d6 = dyn_tns6d_create(5, 5, 5, 5, 5, 5);

        dyn_lattice_seed(d2, 2, 2, 0, 0, 0, 0);
        dyn_lattice_seed(d3, 2, 2, 2, 0, 0, 0);
        dyn_lattice_seed(d6, 2, 2, 2, 2, 2, 2);

        CHECK(d2->num_boundary == 4,  "Cross: 2D center has 4 neighbors");
        CHECK(d3->num_boundary == 6,  "Cross: 3D center has 6 neighbors");
        CHECK(d6->num_boundary == 12, "Cross: 6D center has 12 neighbors");

        dyn_lattice_free(d2);
        dyn_lattice_free(d3);
        dyn_lattice_free(d6);
    }

    #undef CHECK

    printf("\n    Results: %d passed, %d failed\n\n", pass, fail);

    return fail;
}
