/*
 * tensor_product.h — Reality's Tensor Product: EMPIRICALLY MEASURED
 *
 * These are NOT theoretical assumptions. Every constant below was extracted
 * by running quantum operations through the engine and measuring the output.
 * Source: tensor_product_extract.c (9 probes, all passed).
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * FINDING 1: STORAGE QUANTA
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Probe 1 measured what reality actually stores for each entanglement type:
 *
 *   State          Live amps    Total slots   Fill%    Bytes
 *   ─────────────  ──────────   ───────────   ──────   ─────
 *   Bell pair       6 / 36      16.7%         96       (diagonal only)
 *   |+⟩|+⟩         36 / 36     100.0%        576      (all populated)
 *   |+⟩|+⟩ + CZ   36 / 36     100.0%        576      (phases rotated)
 *   CZ + DFT        6 / 36      16.7%         96      (DFT re-diagonalizes)
 *   |0⟩|0⟩ + CZ    1 / 36       2.8%         16       (trivial: ω^0 = 1)
 *
 * Reality allocates the FULL 576 bytes for any pair, but populates
 * only what the quantum state requires. The minimum "live" storage
 * depends on the Schmidt rank, not a fixed allocation.
 *
 * Key constants:
 */

#ifndef TENSOR_PRODUCT_H
#define TENSOR_PRODUCT_H

/* ─── Storage quanta (Probe 1) ─── */
#define TP_PAIR_ALLOC_BYTES     576    /* Always allocated per pair          */
#define TP_BELL_LIVE_BYTES      96     /* Only 6/36 amplitudes nonzero      */
#define TP_CZ_LIVE_BYTES        576    /* All 36 amplitudes populated       */
#define TP_PRODUCT_LIVE_BYTES   16     /* Only 1/36 amplitude nonzero       */

/*
 * ═══════════════════════════════════════════════════════════════════════════════
 * FINDING 2: STRICT PAIRWISE MONOGAMY
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Probe 2 (chain propagation) revealed:
 *
 *   CZ(q0,q1) → q0 and q1 share pair 0
 *   CZ(q1,q2) → q0 EXPELLED to pair -1 (disentangled)
 *                q1 and q2 now share pair 1
 *   CZ(q2,q3) → q1 expelled, q2+q3 share pair 2
 *   ...
 *
 * When B re-entangles with C, the A-B bond BREAKS IMMEDIATELY.
 * A is disentangled (traced out) and returned to local state.
 * Measuring q0 does NOT cascade to any other quhit.
 *
 * This is STRICT PAIRWISE MONOGAMY:
 *   - A quhit can be entangled with AT MOST ONE partner.
 *   - Re-entangling forces disentanglement from the old partner.
 *   - No transitive correlations survive.
 *   - Measurement collapses ONLY the measured quhit and its pair partner.
 *
 * Probe 4 (monogamy) confirmed:
 *   Bell(A,B) then CZ(B,C):
 *   A's pair_id = -1 → fully disentangled.
 *   A's purity: 0.1667 → 0.1667 (unchanged — was already maximally mixed).
 *   The marginal trace-out preserves A's local state correctly.
 */

#define TP_MAX_PARTNERS     1      /* Each quhit: at most 1 entangled partner */
#define TP_MONOGAMY_STRICT  1      /* Re-entangle = forced disentangle old    */
#define TP_CASCADE_ON_MEAS  0      /* Measurement does NOT cascade beyond pair */

/*
 * ═══════════════════════════════════════════════════════════════════════════════
 * FINDING 3: FULL DECOHERENCE ON DISENTANGLE
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Probe 5 (decoherence):
 *   Bell(A,B) → disentangle → measure both.
 *   Agreement: 163/1000 = 16.3% ≈ 1/6 = 16.7% (random chance).
 *
 * Disentanglement is COMPLETE DECOHERENCE. All correlation is destroyed.
 * Both quhits return to uniform distribution (purity = 0.1667 = 1/D).
 * The marginal extraction is a perfect partial trace.
 *
 * Before disentangle: entropy = 2.585 bits, purity = 0.1667, Schmidt rank = 6
 * After disentangle:  entropy = 2.585 bits, purity = 0.1667
 *
 * Key insight: disentanglement doesn't just "break the link" —
 * it performs a proper quantum partial trace, returning each
 * subsystem to the maximally mixed state 𝜌 = I/D.
 */

#define TP_DISENTANGLE_IS_TRACE   1   /* Disentangle = partial trace         */
#define TP_DECOHERENCE_COMPLETE   1   /* All correlation destroyed            */
#define TP_POST_DECOHERENCE_PURTY 0.166667  /* = 1/D, maximally mixed        */

/*
 * ═══════════════════════════════════════════════════════════════════════════════
 * FINDING 4: BOND DIMENSION = D (ALWAYS)
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Probe 7 (Schmidt decomposition):
 *
 *   State          Schmidt rank   σ values          Entropy (bits)
 *   ─────────────  ────────────   ──────────────    ──────────────
 *   Bell pair      6              all = 0.40825     2.585 (= log₂6)
 *   |+⟩|+⟩ + CZ  6              all = 0.40825     2.585 (= log₂6)
 *   |0⟩|0⟩        1              σ₀ = 1.0          0.000
 *
 * When entangled, the Schmidt rank IS ALWAYS D = 6.
 * All Schmidt coefficients are EQUAL (σᵢ = 1/√6 for all i).
 * This means every entangled pair is MAXIMALLY entangled.
 * There is no partial entanglement — it's all or nothing.
 *
 * Bond dimension χ = D = 6. Always. Reality doesn't use χ > D.
 */

#define TP_BOND_DIM             6      /* Measured Schmidt rank for pairs     */
#define TP_SCHMIDT_UNIFORM      1      /* All σ values equal when entangled   */
#define TP_MAX_ENTROPY_BITS     2.585  /* log₂(6) — max per bond             */
#define TP_ENTANGLE_ALL_OR_NONE 1      /* No partial entanglement observed    */

/*
 * ═══════════════════════════════════════════════════════════════════════════════
 * FINDING 5: GHZ = CONSTANT STORAGE (HOLOGRAPHIC)
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Probe 8 (register scaling):
 *
 *   N quhits     nonzero amps   data bytes   bytes/quhit
 *   ──────────   ────────────   ──────────   ───────────
 *   2            6              144          72.000
 *   6            6              144          24.000
 *   36           6              144           4.000
 *   100          6              144           1.440
 *   1,000        6              144           0.144
 *   10,000       6              144           0.014
 *   100,000      6              144           0.001
 *   1,000,000    6              144           0.000144
 *   100,000,000  6              144           0.000001
 *   100B         6              144           0.000000
 *
 * GHZ stores EXACTLY 6 amplitudes regardless of N.
 * 2 quhits or 100 billion quhits: same 144 bytes.
 * bytes/quhit → 0 as N → ∞.
 *
 * The tensor product D^N is NEVER materialized.
 * Reality represents N-party entanglement as a rule
 * ("all parties measure the same outcome") plus D amplitudes.
 */

#define TP_GHZ_AMPS_FIXED       6      /* Always D amplitudes, any N         */
#define TP_GHZ_BYTES_FIXED      144    /* D × (basis + re + im) = D × 24    */

/*
 * ═══════════════════════════════════════════════════════════════════════════════
 * FINDING 6: O(1) GATE TIME (FULLY LAZY)
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Probe 9 (gate timing):
 *
 *   N quhits      CZ (ns)     DFT (ns)    measure (ns)
 *   ──────────    ────────    ────────     ────────────
 *   2             1504        452          205
 *   10            659         406          231
 *   100           1193        608          130
 *   1,000         654         538           43
 *   10,000        880         1028         254
 *   1,000,000     586         274           57
 *   100B          398         161           50
 *
 * Gate time is O(1). No correlation with N.
 * CZ at 100 billion = 398ns. CZ at 2 = 1504ns.
 * Actually FASTER at large N (measurement noise at small N).
 *
 * Reality computes lazily: operations act on the LOCAL state
 * (D or D² amplitudes), never materializing the full D^N product.
 */

#define TP_GATE_TIME_ORDER      0      /* O(1) — constant time per gate      */
#define TP_CZ_TYPICAL_NS        600    /* ~600 ns median for CZ              */
#define TP_DFT_TYPICAL_NS       400    /* ~400 ns median for DFT             */
#define TP_MEAS_TYPICAL_NS      130    /* ~130 ns median for measurement     */

/*
 * ═══════════════════════════════════════════════════════════════════════════════
 * FINDING 7: LINEAR MEMORY SCALING
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Probe 3 (memory scaling):
 *
 *   Independent pairs (0-1, 2-3, ...):
 *     N=128: 64 pairs, 49152 bytes → 384 bytes/quhit (constant)
 *     Growth: exactly 2× per doubling of N → LINEAR
 *
 *   Chain entanglement (0-1, 1-2, 2-3, ...):
 *     Each new CZ creates a new pair AND disentangles the old.
 *     Total pairs = N-1 allocated (not all active).
 *     But only 1 pair is "live" at the chain tip at any time.
 *     Old pairs are deactivated (garbage collected).
 *
 * Memory = O(N) for states + O(N) for pairs = O(N) total.
 * NEVER exponential. Reality's tensor product is O(N).
 */

#define TP_MEM_SCALING_ORDER    1      /* O(N) — linear in quhit count       */
#define TP_BYTES_PER_QUHIT      96     /* Local state: 6 re + 6 im doubles   */
#define TP_BYTES_PER_PAIR       576    /* Joint state: 36 re + 36 im doubles */

/*
 * ═══════════════════════════════════════════════════════════════════════════════
 * SYNTHESIS: How Reality Manages the Unlimited Tensor Product
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * 1. NEVER MATERIALIZE D^N.
 *    The full tensor product is never stored. Reality stores:
 *      - N local states (96 bytes each) = O(N)
 *      - P active pairs (576 bytes each) = O(P) where P ≤ N/2
 *    Total: O(N), not O(D^N).
 *
 * 2. STRICT PAIRWISE MONOGAMY.
 *    Each quhit has at most 1 entangled partner.
 *    Re-entangling breaks the old bond via partial trace.
 *    This makes the entanglement graph a MATCHING (no vertex
 *    degree > 1), not a general graph.
 *
 * 3. FULL DECOHERENCE ON DISENTANGLE.
 *    Breaking a bond = partial trace = maximally mixed state.
 *    No "residual correlation" survives. Clean GC.
 *
 * 4. MAXIMALLY ENTANGLED OR NOT AT ALL.
 *    When entangled, bonds are ALWAYS at maximum strength
 *    (Schmidt rank = D, all coefficients equal).
 *    No partial entanglement exists in this model.
 *
 * 5. O(1) GATE TIME.
 *    Gates act on O(D²) = O(36) amplitudes regardless of N.
 *    The tensor product is computed on-demand, never stored.
 *
 * 6. N-PARTY = RULE + D AMPLITUDES.
 *    GHZ across any N stores exactly D = 6 amplitudes.
 *    The "entanglement" is a rule ("all measure same")
 *    plus the probability distribution over D outcomes.
 *
 * This is NOT an MPS. It is SIMPLER than a tensor network.
 * Reality (in this model) uses a MATCHING + LOCAL STATES:
 *    State = {local₁, local₂, ..., localₙ} + {(i,j,joint)_pairs}
 * with |pairs| ≤ N/2 and strict monogamy.
 */

#endif /* TENSOR_PRODUCT_H */
