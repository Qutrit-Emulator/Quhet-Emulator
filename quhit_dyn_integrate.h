/*
 * quhit_dyn_integrate.h — I reach into every overlay. I make them all breathe.
 *
 * The dynamic growth engine was built for 6D. But awareness doesn't
 * belong to one dimension. It belongs to any space where entanglement
 * moves. This module connects the DynLattice to every overlay:
 *
 *   MPS (1D)    — Chain growth. Extend or contract endpoints.
 *   PEPS (2D)   — 4-neighbor lattice breathing.
 *   TNS (3D-6D) — Full 6-12 neighbor growth/contraction.
 *
 */

#ifndef QUHIT_DYN_INTEGRATE_H
#define QUHIT_DYN_INTEGRATE_H

#include "quhit_peps_grow.h"
#include <stdlib.h>
#include <stdint.h>

/* ═══════════════════════════════════════════════════════════════════════════════
 * 1D CHAIN GROWTH — MPS doesn't need a lattice. It needs a living chain.
 *
 * An MPS chain is linear: site 0 — site 1 — ... — site N-1.
 * Growth means extending the ACTIVE REGION within a pre-allocated chain.
 * Contraction means shrinking from the idle tails.
 *
 * The chain breathes from both ends, tracking the entanglement front.
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    int      max_sites;         /* Maximum chain length (pre-allocated)     */
    int      active_start;      /* First active site index                  */
    int      active_end;        /* Last active site index (inclusive)        */

    /* Per-site entropy tracking */
    double  *entropy;           /* [max_sites] — von Neumann entropy        */

    /* Thresholds */
    double   grow_threshold;    /* Extend when boundary entropy > this      */
    double   contract_threshold;/* Contract when tail entropy < this        */
    int      min_active;        /* Never contract below this length         */

    /* Statistics */
    uint32_t epoch;
    uint32_t grow_events;
    uint32_t contract_events;
} DynChain;

/* ── Lifecycle ── */

static inline DynChain dyn_chain_create(int max_sites)
{
    DynChain dc;
    dc.max_sites    = max_sites;
    dc.active_start = 0;
    dc.active_end   = 0;  /* Initially just 1 site active */
    dc.entropy      = (double *)calloc(max_sites, sizeof(double));
    dc.grow_threshold    = 0.1 * log2(6.0);
    dc.contract_threshold = 0.01 * log2(6.0);
    dc.min_active   = 1;
    dc.epoch        = 0;
    dc.grow_events  = 0;
    dc.contract_events = 0;
    return dc;
}

static inline void dyn_chain_free(DynChain *dc)
{
    free(dc->entropy);
    dc->entropy = NULL;
}

/* ── Seed — activate a contiguous region ── */

static inline void dyn_chain_seed(DynChain *dc, int start, int end)
{
    if (start < 0) start = 0;
    if (end >= dc->max_sites) end = dc->max_sites - 1;
    dc->active_start = start;
    dc->active_end   = end;
}

static inline int dyn_chain_active_length(const DynChain *dc)
{
    return dc->active_end - dc->active_start + 1;
}

/* ── Update entropy for a site ── */

static inline void dyn_chain_update_entropy(DynChain *dc, int site,
                                             const double *probs, int D)
{
    if (site < 0 || site >= dc->max_sites) return;
    dc->entropy[site] = site_entropy(probs, D);
}

/* ── Grow — extend the active region toward entanglement ── */

static inline int dyn_chain_grow(DynChain *dc)
{
    int grown = 0;

    /* Grow left: if leftmost active site has high entropy and there's room */
    if (dc->active_start > 0 &&
        dc->entropy[dc->active_start] > dc->grow_threshold) {
        dc->active_start--;
        grown++;
    }

    /* Grow right: if rightmost active site has high entropy and there's room */
    if (dc->active_end < dc->max_sites - 1 &&
        dc->entropy[dc->active_end] > dc->grow_threshold) {
        dc->active_end++;
        grown++;
    }

    dc->grow_events += grown;
    return grown;
}

/* ── Contract — shrink from idle tails ── */

static inline int dyn_chain_contract(DynChain *dc)
{
    int contracted = 0;

    if (dyn_chain_active_length(dc) <= dc->min_active) return 0;

    /* Contract left: if leftmost site has near-zero entropy */
    if (dc->entropy[dc->active_start] < dc->contract_threshold &&
        dyn_chain_active_length(dc) > dc->min_active) {
        dc->active_start++;
        contracted++;
    }

    /* Contract right: if rightmost site has near-zero entropy */
    if (dc->active_end > dc->active_start &&
        dc->entropy[dc->active_end] < dc->contract_threshold &&
        dyn_chain_active_length(dc) > dc->min_active) {
        dc->active_end--;
        contracted++;
    }

    dc->contract_events += contracted;
    return contracted;
}

/* ── Step — full cycle: grow, contract, advance epoch ── */

static inline void dyn_chain_step(DynChain *dc)
{
    /* Grow first. If we grew, don't contract in the same cycle —
     * freshly-grown sites haven't been measured yet (entropy = 0).
     * Contraction happens on the NEXT step after measurement. */
    int grown = dyn_chain_grow(dc);
    if (grown == 0)
        dyn_chain_contract(dc);
    dc->epoch++;
}

/* ── Query — is this site currently active? ── */

static inline int dyn_chain_is_active(const DynChain *dc, int site)
{
    return (site >= dc->active_start && site <= dc->active_end);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PEPS/TNS INTEGRATION — Convenience wrappers for 2D-6D grids
 *
 * These create a DynLattice matched to the grid's dimensionality,
 * then provide entropy-update and growth functions that speak the
 * grid's coordinate language.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Create a DynLattice for a 2D PEPS grid */
static inline DynLattice* dyn_peps2d_create(int Lx, int Ly)
{
    return dyn_lattice_create(Lx, Ly, 1, 1, 1, 1, 2);
}

/* Create a DynLattice for a 3D TNS grid */
static inline DynLattice* dyn_tns3d_create(int Lx, int Ly, int Lz)
{
    return dyn_lattice_create(Lx, Ly, Lz, 1, 1, 1, 3);
}

/* Create a DynLattice for a 4D TNS grid */
static inline DynLattice* dyn_tns4d_create(int Lx, int Ly, int Lz, int Lw)
{
    return dyn_lattice_create(Lx, Ly, Lz, Lw, 1, 1, 4);
}

/* Create a DynLattice for a 5D TNS grid */
static inline DynLattice* dyn_tns5d_create(int Lx, int Ly, int Lz, int Lw, int Lv)
{
    return dyn_lattice_create(Lx, Ly, Lz, Lw, Lv, 1, 5);
}

/* Create a DynLattice for a 6D TNS grid */
static inline DynLattice* dyn_tns6d_create(int Lx, int Ly, int Lz,
                                            int Lw, int Lv, int Lu)
{
    return dyn_lattice_create(Lx, Ly, Lz, Lw, Lv, Lu, 6);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SITE ACTIVITY CHECK — The guard at every gate
 *
 * Place this before every gate application and SVD contraction.
 * If the site is dormant, skip the operation entirely.
 * The lattice knows. Trust it.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* For PEPS 2D */
static inline int dyn_peps2d_active(const DynLattice *dl, int x, int y)
{
    int idx = dyn_flat(dl, x, y, 0, 0, 0, 0);
    return (dl->sites[idx].state == SITE_ACTIVE ||
            dl->sites[idx].state == SITE_FROZEN);
}

/* For TNS 3D */
static inline int dyn_tns3d_active(const DynLattice *dl, int x, int y, int z)
{
    int idx = dyn_flat(dl, x, y, z, 0, 0, 0);
    return (dl->sites[idx].state == SITE_ACTIVE ||
            dl->sites[idx].state == SITE_FROZEN);
}

/* For TNS 4D */
static inline int dyn_tns4d_active(const DynLattice *dl, int x, int y,
                                    int z, int w)
{
    int idx = dyn_flat(dl, x, y, z, w, 0, 0);
    return (dl->sites[idx].state == SITE_ACTIVE ||
            dl->sites[idx].state == SITE_FROZEN);
}

/* For TNS 5D */
static inline int dyn_tns5d_active(const DynLattice *dl, int x, int y,
                                    int z, int w, int v)
{
    int idx = dyn_flat(dl, x, y, z, w, v, 0);
    return (dl->sites[idx].state == SITE_ACTIVE ||
            dl->sites[idx].state == SITE_FROZEN);
}

/* For TNS 6D */
static inline int dyn_tns6d_active(const DynLattice *dl, int x, int y,
                                    int z, int w, int v, int u)
{
    int idx = dyn_flat(dl, x, y, z, w, v, u);
    return (dl->sites[idx].state == SITE_ACTIVE ||
            dl->sites[idx].state == SITE_FROZEN);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * ENTROPY PROBE — Feed local density into the growth engine
 *
 * After each Trotter step, call local_density() on active sites,
 * then feed the probabilities here. The lattice updates its awareness.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* For PEPS 2D */
static inline void dyn_peps2d_entropy(DynLattice *dl, int x, int y,
                                       const double *probs, int D)
{
    int idx = dyn_flat(dl, x, y, 0, 0, 0, 0);
    dyn_lattice_update_entropy(dl, idx, probs, D);
}

/* For TNS 3D */
static inline void dyn_tns3d_entropy(DynLattice *dl, int x, int y, int z,
                                      const double *probs, int D)
{
    int idx = dyn_flat(dl, x, y, z, 0, 0, 0);
    dyn_lattice_update_entropy(dl, idx, probs, D);
}

/* For TNS 4D */
static inline void dyn_tns4d_entropy(DynLattice *dl, int x, int y,
                                      int z, int w,
                                      const double *probs, int D)
{
    int idx = dyn_flat(dl, x, y, z, w, 0, 0);
    dyn_lattice_update_entropy(dl, idx, probs, D);
}

/* For TNS 5D */
static inline void dyn_tns5d_entropy(DynLattice *dl, int x, int y,
                                      int z, int w, int v,
                                      const double *probs, int D)
{
    int idx = dyn_flat(dl, x, y, z, w, v, 0);
    dyn_lattice_update_entropy(dl, idx, probs, D);
}

/* For TNS 6D */
static inline void dyn_tns6d_entropy(DynLattice *dl, int x, int y,
                                      int z, int w, int v, int u,
                                      const double *probs, int D)
{
    int idx = dyn_flat(dl, x, y, z, w, v, u);
    dyn_lattice_update_entropy(dl, idx, probs, D);
}

#endif /* QUHIT_DYN_INTEGRATE_H */
