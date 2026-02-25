/*
 * lazy_stats.h — Side-Channel Statistics for Lazy Evaluation
 *
 * Tracks how much work the engine actually does vs could skip.
 * Evidence for "reality computes on demand."
 */

#ifndef LAZY_STATS_H
#define LAZY_STATS_H

#include <stdint.h>
#include <stdio.h>

typedef struct {
    uint64_t gates_queued;        /* Total gates submitted                    */
    uint64_t gates_materialized;  /* Gates actually applied (measurement)     */
    uint64_t gates_fused;         /* Consecutive same-site gates merged       */
    uint64_t gates_skipped;       /* Gates never applied (site never measured)*/
    uint64_t sites_total;         /* Total sites in chain                     */
    uint64_t sites_allocated;     /* Sites with real tensor data              */
    uint64_t sites_lazy;          /* Sites still virtual (implicit |0⟩)       */
    double   hilbert_log10;       /* log₁₀ of full tensor product dimension  */
    uint64_t memory_actual;       /* Bytes actually used                      */
    uint64_t memory_full_tp;      /* Bytes the full TP would need (capped)    */
} LazyStats;

static inline void lazy_stats_reset(LazyStats *s)
{
    s->gates_queued = 0;
    s->gates_materialized = 0;
    s->gates_fused = 0;
    s->gates_skipped = 0;
    s->sites_total = 0;
    s->sites_allocated = 0;
    s->sites_lazy = 0;
    s->hilbert_log10 = 0;
    s->memory_actual = 0;
    s->memory_full_tp = 0;
}

static inline void lazy_stats_print(const LazyStats *s)
{
    printf("\n  ═══ LAZY EVALUATION STATISTICS ═══\n\n");
    printf("  Gates queued:       %lu\n", (unsigned long)s->gates_queued);
    printf("  Gates materialized: %lu", (unsigned long)s->gates_materialized);
    if (s->gates_queued > 0)
        printf("  (%.1f%%)", 100.0 * s->gates_materialized / s->gates_queued);
    printf("\n");
    printf("  Gates fused:        %lu\n", (unsigned long)s->gates_fused);
    printf("  Gates skipped:      %lu", (unsigned long)s->gates_skipped);
    if (s->gates_queued > 0)
        printf("  (%.1f%%)", 100.0 * s->gates_skipped / s->gates_queued);
    printf("\n\n");
    printf("  Sites total:        %lu\n", (unsigned long)s->sites_total);
    printf("  Sites allocated:    %lu\n", (unsigned long)s->sites_allocated);
    printf("  Sites lazy:         %lu", (unsigned long)s->sites_lazy);
    if (s->sites_total > 0)
        printf("  (%.1f%%)", 100.0 * s->sites_lazy / s->sites_total);
    printf("\n\n");
    printf("  Hilbert space:      10^%.0f dimensions\n", s->hilbert_log10);
    printf("  Memory actual:      %lu KB\n", (unsigned long)(s->memory_actual / 1024));
    printf("  Lazy ratio:         ");
    if (s->gates_queued > 0) {
        double ratio = 1.0 - (double)s->gates_materialized / s->gates_queued;
        printf("%.1f%% of work avoided\n", ratio * 100.0);
    } else {
        printf("N/A\n");
    }
}

#endif /* LAZY_STATS_H */
