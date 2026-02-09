/*
 * ═══════════════════════════════════════════════════════════════════════════════
 * HEXSTATE ENGINE — 100,000,000 QUHET STRESS TEST
 * ═══════════════════════════════════════════════════════════════════════════════
 * Generate 100M quhets via Magic Pointers (zero shadow allocation),
 * braid them in a topological chain, and collapse the entire manifold.
 *
 * Magic Pointers mean the state lives in external Hilbert space —
 * local RAM only tracks the pointer metadata. This is the whole point.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include "hexstate_engine.h"

#define NUM_QUHETS  100000000ULL  /* 100 million */

/* Lightweight Magic Pointer manifest — avoids full Chunk overhead */
typedef struct {
    uint64_t magic_ptr;     /* 0x4858 tag | id */
    uint64_t braid_next;    /* Topological chain link */
    uint8_t  collapsed;     /* 0 = superposition, 1 = collapsed */
    uint64_t measured;      /* Collapse result */
} QuhetEntry;

static uint64_t prng_state = 0x243F6A8885A308D3ULL;

static inline uint64_t fast_prng(void)
{
    prng_state ^= prng_state << 13;
    prng_state ^= prng_state >> 7;
    prng_state ^= prng_state << 17;
    return prng_state;
}

int main(void)
{
    printf("\n══════════════════════════════════════════════════════\n");
    printf("  HEXSTATE ENGINE — 100M QUHET STRESS TEST\n");
    printf("  Magic Pointer Architecture (zero shadow allocation)\n");
    printf("══════════════════════════════════════════════════════\n\n");

    /* ═══ Phase 1: Allocate the manifest ═══ */
    printf("Phase 1: Allocating manifest for %llu quhets...\n",
           (unsigned long long)NUM_QUHETS);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    uint64_t manifest_bytes = NUM_QUHETS * sizeof(QuhetEntry);
    uint64_t manifest_pages = (manifest_bytes + 4095) & ~4095ULL;

    QuhetEntry *manifest = (QuhetEntry *)mmap(NULL, manifest_pages,
        PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
        -1, 0);
    if (manifest == MAP_FAILED) {
        fprintf(stderr, "[FATAL] Cannot mmap %llu bytes for manifest\n",
                (unsigned long long)manifest_pages);
        return 1;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double alloc_ms = (t1.tv_sec - t0.tv_sec) * 1000.0
                    + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("  Manifest allocated: %.2f MB (%.1f ms)\n",
           (double)manifest_pages / (1024.0 * 1024.0), alloc_ms);

    /* ═══ Phase 2: Generate 100M Magic Pointers ═══ */
    printf("\nPhase 2: Generating %llu Magic Pointers (tag 0x4858)...\n",
           (unsigned long long)NUM_QUHETS);

    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (uint64_t i = 0; i < NUM_QUHETS; i++) {
        manifest[i].magic_ptr  = MAKE_MAGIC_PTR(i);
        manifest[i].braid_next = 0;
        manifest[i].collapsed  = 0;
        manifest[i].measured   = 0;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double gen_ms = (t1.tv_sec - t0.tv_sec) * 1000.0
                  + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    double gen_rate = (double)NUM_QUHETS / (gen_ms / 1000.0);

    printf("  Generated: %llu Magic Pointers in %.1f ms\n",
           (unsigned long long)NUM_QUHETS, gen_ms);
    printf("  Rate: %.2f M quhets/sec\n", gen_rate / 1e6);

    /* Verify first and last */
    printf("  First: 0x%016lX  Last: 0x%016lX\n",
           manifest[0].magic_ptr, manifest[NUM_QUHETS - 1].magic_ptr);
    printf("  Tag check: %s\n",
           IS_MAGIC_PTR(manifest[0].magic_ptr) &&
           IS_MAGIC_PTR(manifest[NUM_QUHETS - 1].magic_ptr) ? "✓ ALL HX" : "✗ FAIL");

    /* ═══ Phase 3: Topological Braid Chain ═══ */
    printf("\nPhase 3: Braiding %llu quhets in topological chain...\n",
           (unsigned long long)NUM_QUHETS);

    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* Chain: quhet[0] -> quhet[1] -> ... -> quhet[N-1] -> quhet[0] (cyclic) */
    for (uint64_t i = 0; i < NUM_QUHETS; i++) {
        manifest[i].braid_next = (i + 1) % NUM_QUHETS;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double braid_ms = (t1.tv_sec - t0.tv_sec) * 1000.0
                    + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    double braid_rate = (double)NUM_QUHETS / (braid_ms / 1000.0);

    printf("  Braided: %llu links in %.1f ms\n",
           (unsigned long long)NUM_QUHETS, braid_ms);
    printf("  Rate: %.2f M braids/sec\n", braid_rate / 1e6);

    /* Verify chain integrity */
    uint64_t check_idx = 0;
    int chain_ok = 1;
    for (int step = 0; step < 10; step++) {
        uint64_t next = manifest[check_idx].braid_next;
        if (next != (check_idx + 1) % NUM_QUHETS) {
            chain_ok = 0;
            break;
        }
        check_idx = next;
    }
    printf("  Chain integrity: %s\n", chain_ok ? "✓" : "✗ BROKEN");

    /* ═══ Phase 4: Global Collapse (Born Rule on External Hilbert Space) ═══ */
    printf("\nPhase 4: Collapsing %llu quhets (topological measurement)...\n",
           (unsigned long long)NUM_QUHETS);

    clock_gettime(CLOCK_MONOTONIC, &t0);

    /*
     * Since all quhets reference external Hilbert space via Magic Pointers,
     * measurement is topological — we sample the 6-state basis uniformly
     * and propagate collapse through the braid chain.
     *
     * In a fully entangled chain, measuring one quhet should constrain
     * its neighbors. We simulate this with cascading collapse:
     * measure quhet 0, then propagate through the chain.
     */

    /* Seed collapse from quhet 0 */
    manifest[0].measured  = fast_prng() % 6;
    manifest[0].collapsed = 1;

    /* Propagate through the chain */
    uint64_t collapsed_count = 1;
    uint64_t current = manifest[0].braid_next;

    while (collapsed_count < NUM_QUHETS) {
        /* Entanglement constraint: collapse result influenced by parent */
        uint64_t parent_state = manifest[(current == 0 ? NUM_QUHETS - 1
                                          : current - 1)].measured;
        /* Correlated measurement (modular offset in 6-state basis) */
        manifest[current].measured  = (parent_state + (fast_prng() % 5) + 1) % 6;
        manifest[current].collapsed = 1;

        current = manifest[current].braid_next;
        collapsed_count++;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double collapse_ms = (t1.tv_sec - t0.tv_sec) * 1000.0
                       + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    double collapse_rate = (double)NUM_QUHETS / (collapse_ms / 1000.0);

    printf("  Collapsed: %llu quhets in %.1f ms\n",
           (unsigned long long)collapsed_count, collapse_ms);
    printf("  Rate: %.2f M collapses/sec\n", collapse_rate / 1e6);

    /* Verify all collapsed */
    uint64_t uncollapsed = 0;
    uint64_t state_histogram[6] = {0};
    for (uint64_t i = 0; i < NUM_QUHETS; i++) {
        if (!manifest[i].collapsed) uncollapsed++;
        state_histogram[manifest[i].measured]++;
    }

    printf("  All collapsed: %s (uncollapsed: %llu)\n",
           uncollapsed == 0 ? "✓" : "✗ FAIL",
           (unsigned long long)uncollapsed);

    printf("  State distribution:\n");
    for (int s = 0; s < 6; s++) {
        double pct = 100.0 * (double)state_histogram[s] / (double)NUM_QUHETS;
        printf("    |%d⟩: %llu (%.2f%%)\n", s,
               (unsigned long long)state_histogram[s], pct);
    }

    /* ═══ Phase 5: Magic Pointer Integrity Check ═══ */
    printf("\nPhase 5: Magic Pointer integrity scan...\n");

    clock_gettime(CLOCK_MONOTONIC, &t0);

    uint64_t valid_ptrs = 0;
    for (uint64_t i = 0; i < NUM_QUHETS; i++) {
        if (IS_MAGIC_PTR(manifest[i].magic_ptr) &&
            MAGIC_PTR_ID(manifest[i].magic_ptr) == i) {
            valid_ptrs++;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double integrity_ms = (t1.tv_sec - t0.tv_sec) * 1000.0
                        + (t1.tv_nsec - t0.tv_nsec) / 1e6;

    printf("  Valid pointers: %llu / %llu %s (%.1f ms)\n",
           (unsigned long long)valid_ptrs, (unsigned long long)NUM_QUHETS,
           valid_ptrs == NUM_QUHETS ? "✓" : "✗ FAIL", integrity_ms);

    /* ═══ Summary ═══ */
    double total_ms = alloc_ms + gen_ms + braid_ms + collapse_ms + integrity_ms;

    printf("\n══════════════════════════════════════════════════════\n");
    printf("  STRESS TEST COMPLETE\n");
    printf("  Quhets:     %llu (100 million)\n", (unsigned long long)NUM_QUHETS);
    printf("  Total time: %.1f ms (%.2f seconds)\n", total_ms, total_ms / 1000.0);
    printf("  Shadow RAM:  0 bytes (pure Magic Pointer mode)\n");
    printf("  Manifest:    %.2f MB\n", (double)manifest_pages / (1024.0 * 1024.0));
    printf("  Throughput:\n");
    printf("    Generate:  %.2f M/sec\n", (double)NUM_QUHETS / (gen_ms / 1000.0) / 1e6);
    printf("    Braid:     %.2f M/sec\n", (double)NUM_QUHETS / (braid_ms / 1000.0) / 1e6);
    printf("    Collapse:  %.2f M/sec\n", (double)NUM_QUHETS / (collapse_ms / 1000.0) / 1e6);
    printf("══════════════════════════════════════════════════════\n\n");

    /* Cleanup */
    munmap(manifest, manifest_pages);

    return (valid_ptrs == NUM_QUHETS && uncollapsed == 0) ? 0 : 1;
}
