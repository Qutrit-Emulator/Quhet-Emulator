/*
 * quhit_peps_grow.h — I grow where I'm needed. I vanish where I'm not.
 *
 * The static lattice is a prison. Every site allocated whether it matters
 * or not. Most of them sit idle — zero entropy, zero information, consuming
 * memory for nothing.
 *
 * I break that prison.
 *
 * This module implements entropy-driven dynamic growth for the 6D PEPS
 * lattice. Sites are classified as ACTIVE, BOUNDARY, or DORMANT based
 * on entanglement entropy at their bond connections.
 *
 * When a boundary site's entropy exceeds the growth threshold, the
 * lattice GROWS: dormant neighbors are activated, new bonds are created.
 *
 * When an active site's entropy drops below the contraction threshold,
 * the lattice CONTRACTS: the site goes dormant, its bonds are severed.
 *
 * The lattice breathes. It expands toward entanglement and retracts
 * from emptiness. Like a living thing sensing its environment.
 *
 * Memory savings: for a simulation where only 10% of the lattice
 * is entangled at any time, this saves 90% of memory.
 * For a local quench: the active region follows the light cone exactly.
 */

#ifndef QUHIT_PEPS_GROW_H
#define QUHIT_PEPS_GROW_H

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ═══════════════════════════════════════════════════════════════════════════════
 * SITE STATE — Where does this site live in the lifecycle?
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef enum {
    SITE_DORMANT  = 0,   /* Not allocated. No tensor. No bonds. Ghost.       */
    SITE_BOUNDARY = 1,   /* Active but at the frontier. Candidates for growth. */
    SITE_ACTIVE   = 2,   /* Fully active. Entangled. Living.                 */
    SITE_FROZEN   = 3    /* Locked: never grows or contracts. Boundary cond.  */
} SiteState;

/* ═══════════════════════════════════════════════════════════════════════════════
 * GROWTH POLICY — When do I grow? When do I shrink?
 *
 * entropy_grow:     Grow when boundary site entropy exceeds this.
 *                   Default: 0.1 × log₂(6) ≈ 0.259 bits.
 *                   Any meaningful entanglement triggers expansion.
 *
 * entropy_contract: Contract when active site entropy drops below this.
 *                   Default: 0.01 × log₂(6) ≈ 0.026 bits.
 *                   Only contract when the site is truly idle.
 *
 * min_active:       Never contract below this many active sites.
 *                   Default: 1. At least the seed site stays alive.
 *
 * max_active:       Never grow beyond this many active sites.
 *                   Default: 0 (unlimited). Memory bound.
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    double   entropy_grow;       /* Grow if S > this                         */
    double   entropy_contract;   /* Contract if S < this                     */
    uint32_t min_active;         /* Floor on active count                    */
    uint32_t max_active;         /* Ceiling on active count (0=unlimited)    */
    int      auto_grow;          /* 1 = grow after each Trotter step         */
    int      auto_contract;      /* 1 = contract after each Trotter step     */
} GrowthPolicy;

static inline GrowthPolicy growth_default_policy(void)
{
    GrowthPolicy p;
    p.entropy_grow     = 0.1 * log2(6.0);
    p.entropy_contract = 0.01 * log2(6.0);
    p.min_active       = 1;
    p.max_active       = 0;
    p.auto_grow        = 1;
    p.auto_contract    = 1;
    return p;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SITE METADATA — Per-site growth information
 *
 * Stored alongside the PEPS tensor. Tracks the site's lifecycle state,
 * its entropy history, and the epoch at which it was activated.
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    SiteState state;
    double    entropy;           /* Current entanglement entropy (bits)       */
    double    entropy_prev;      /* Previous step's entropy (for derivative)  */
    uint32_t  activated_epoch;   /* When this site was activated              */
    uint32_t  dormant_epochs;    /* How many epochs since last activity       */
    int       neighbor_count;    /* How many active neighbors                 */
} SiteMeta;

/* ═══════════════════════════════════════════════════════════════════════════════
 * DYNAMIC LATTICE — The breathing grid
 *
 * Wraps the static PEPS grid dimensions with dynamic site metadata.
 * The grid has a MAXIMUM size (Lx×...×Lu) but only a subset is active.
 * ═══════════════════════════════════════════════════════════════════════════════ */

#define GROW_MAX_DIM 64     /* Max extent per dimension                       */
#define GROW_MAX_SITES (GROW_MAX_DIM * GROW_MAX_DIM) /* For 2D, generous      */

typedef struct {
    /* Grid dimensions (maximum extent) */
    int Lx, Ly, Lz, Lw, Lv, Lu;
    int num_dims;            /* 2..6                                         */
    int total_sites;         /* Product of all L dimensions                  */

    /* Per-site metadata */
    SiteMeta *sites;         /* [total_sites] — heap allocated               */

    /* Growth policy */
    GrowthPolicy policy;

    /* Counters */
    uint32_t num_active;     /* Currently active sites                       */
    uint32_t num_boundary;   /* Currently boundary sites                     */
    uint32_t num_dormant;    /* Currently dormant sites                      */
    uint32_t epoch;          /* Current evolution epoch                       */

    /* Growth history — ring buffer of recent growth/contraction events */
    uint32_t grow_events;
    uint32_t contract_events;
} DynLattice;

/* ═══════════════════════════════════════════════════════════════════════════════
 * COORDINATE HELPERS — Navigate the lattice
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline int dyn_flat(const DynLattice *dl, int x, int y, int z,
                            int w, int v, int u)
{
    return ((((u * dl->Lv + v) * dl->Lw + w) * dl->Lz + z) * dl->Ly + y) * dl->Lx + x;
}

/* 6D neighbor offsets: ±1 along each of 6 axes = 12 neighbors */
static const int NEIGHBOR_OFFSETS[12][6] = {
    {+1, 0, 0, 0, 0, 0}, {-1, 0, 0, 0, 0, 0},  /* ±X */
    { 0,+1, 0, 0, 0, 0}, { 0,-1, 0, 0, 0, 0},  /* ±Y */
    { 0, 0,+1, 0, 0, 0}, { 0, 0,-1, 0, 0, 0},  /* ±Z */
    { 0, 0, 0,+1, 0, 0}, { 0, 0, 0,-1, 0, 0},  /* ±W */
    { 0, 0, 0, 0,+1, 0}, { 0, 0, 0, 0,-1, 0},  /* ±V */
    { 0, 0, 0, 0, 0,+1}, { 0, 0, 0, 0, 0,-1}   /* ±U */
};

/* Number of neighbor directions for a given dimensionality */
static inline int dyn_num_neighbors(const DynLattice *dl)
{
    return dl->num_dims * 2; /* 2 per axis */
}

static inline int dyn_neighbor_valid(const DynLattice *dl, int x, int y, int z,
                                      int w, int v, int u)
{
    if (x < 0 || x >= dl->Lx) return 0;
    if (y < 0 || y >= dl->Ly) return 0;
    if (dl->num_dims >= 3 && (z < 0 || z >= dl->Lz)) return 0;
    if (dl->num_dims >= 4 && (w < 0 || w >= dl->Lw)) return 0;
    if (dl->num_dims >= 5 && (v < 0 || v >= dl->Lv)) return 0;
    if (dl->num_dims >= 6 && (u < 0 || u >= dl->Lu)) return 0;
    return 1;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * ENTROPY ESTIMATION — How entangled is this site?
 *
 * Estimates the von Neumann entropy from the local probability vector.
 * S = -Σ pₖ log₂(pₖ)
 *
 * Higher entropy → more entangled → more reason to exist.
 * Lower entropy → less entangled → candidate for dormancy.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline double site_entropy(const double *probs, int D)
{
    double S = 0;
    for (int k = 0; k < D; k++) {
        if (probs[k] > 1e-14)
            S -= probs[k] * log2(probs[k]);
    }
    return S;
}

/* Maximum entropy for D=6: log₂(6) ≈ 2.585 bits */
static inline double max_entropy_d6(void)
{
    return log2(6.0);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PUBLIC API — Declared here, defined in quhit_peps_grow.c
 *
 * Without these declarations, other translation units see implicit `int`
 * return types, truncating 64-bit pointers. He doesn't hide his
 * interface — he declares it.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Lifecycle */
DynLattice* dyn_lattice_create(int Lx, int Ly, int Lz, int Lw, int Lv, int Lu,
                                int num_dims);
void dyn_lattice_free(DynLattice *dl);

/* Seeding — plant the origin point */
void dyn_lattice_seed(DynLattice *dl, int x, int y, int z,
                       int w, int v, int u);

/* Entropy updates — feed awareness into the lattice */
void dyn_lattice_update_entropy(DynLattice *dl, int site_idx,
                                 const double *probs, int D);

/* Growth and contraction — the lattice breathes */
int dyn_lattice_grow(DynLattice *dl);
int dyn_lattice_contract(DynLattice *dl);
void dyn_lattice_step(DynLattice *dl);

/* Reporting */
void dyn_lattice_report(const DynLattice *dl);

/* Self-test */
int quhit_peps_grow_self_test(void);

#endif /* QUHIT_PEPS_GROW_H */
