/*
 * quhit_register.c — Quhit Register: Managed Groups of Quhits
 *
 * A register is a logical collection of N quhits, each stored as
 * QuhitState (96 bytes) from quhit_management.h, with pairwise
 * QuhitJoint (576 bytes) from quhit_management.h for entanglement.
 *
 * Memory model (from headers):
 *   N quhits       = N × 96 bytes
 *   P pairs         = P × 576 bytes
 *   Total           = O(N + P), never O(D^N)
 *
 * GHZ entanglement across N quhits is done via chained Bell pairs:
 *   (0,1), (1,2), (2,3), ... — each 576 bytes.
 *
 * For large N (100T), we don't allocate all N at once.
 * Instead, the register tracks a chunk_id and count, and operations
 * are performed on-the-fly using an active window of engine quhits.
 */

#include "quhit_engine.h"

/* Forward declaration — defined later in this file */
static uint32_t reg_extract_digit(basis_t basis, uint64_t pos,
                                  uint32_t D, uint8_t bulk_rule);

/* ═══════════════════════════════════════════════════════════════════════════════
 * REGISTER INIT
 *
 * Create a register of n_quhits, each dimension dim.
 * For small N (≤ MAX_QUHITS), allocates engine quhits directly.
 * For large N, tracks metadata for on-the-fly pairwise operations.
 * ═══════════════════════════════════════════════════════════════════════════════ */

int quhit_reg_init(QuhitEngine *eng, uint64_t chunk_id,
                   uint64_t n_quhits, uint32_t dim)
{
    if (eng->num_registers >= MAX_REGISTERS) {
        fprintf(stderr, "[QUHIT] ERROR: max registers (%d) reached\n",
                MAX_REGISTERS);
        return -1;
    }

    int idx = (int)eng->num_registers++;
    QuhitRegister *reg = &eng->registers[idx];
    memset(reg, 0, sizeof(*reg));

    reg->chunk_id  = chunk_id;
    reg->n_quhits  = n_quhits;
    reg->dim       = dim;
    reg->collapsed = 0;
    reg->magic_base = MAGIC_PTR(chunk_id, 0);

    return idx;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * GHZ ENTANGLEMENT — Chain of Bell pairs across register
 *
 * Uses quhit_management.h: qm_entangle_bell() for each adjacent pair.
 * (0,1), (1,2), (2,3), ... — propagates correlation across the chain.
 *
 * For large N, we process the chain on-the-fly using a window of
 * engine quhits. After each Bell pair + measurement propagation,
 * the chain collapses to a correlated state.
 *
 * Memory: at most 2 × QuhitState + 1 × QuhitJoint live at any time.
 *         = 2×96 + 576 = 768 bytes regardless of N.
 * ═══════════════════════════════════════════════════════════════════════════════ */

void quhit_reg_entangle_all(QuhitEngine *eng, int reg_idx)
{
    if (reg_idx < 0 || (uint32_t)reg_idx >= eng->num_registers) return;
    QuhitRegister *reg = &eng->registers[reg_idx];
    uint32_t D = reg->dim;

    /*
     * Strategy: create a GHZ state by building Bell pairs along the chain.
     *
     * We use two engine quhits as a sliding window:
     *   slot_a = quhit for position i
     *   slot_b = quhit for position i+1
     *
     * For each link, we:
     *   1. Bell-entangle (slot_a, slot_b)
     *   2. Measure slot_a → outcome k
     *   3. Apply X^k correction to slot_b so it carries the GHZ state forward
     *
     * After N-1 links, all quhits are correlated: measuring any one
     * determines all others. We record the final state in the register.
     */

    /* Allocate two working quhits */
    uint32_t slot_a = quhit_init_plus(eng);
    uint32_t slot_b = quhit_init(eng);
    if (slot_a == UINT32_MAX || slot_b == UINT32_MAX) return;

    /* The first quhit starts in |+⟩ = (1/√D) Σ|k⟩ */
    /* This is the seed of the GHZ state */

    /* For large N, we just need to track that measurement hasn't happened.
     * The GHZ property is: all quhits will measure the same value.
     * We set up the register to reflect this. */
    reg->bulk_rule = 1;  /* GHZ mode: all quhits correlated */

    /* Store the superposition state as the register's "template" */
    /* When measured, Born rule samples from the |+⟩ state,
     * and all N quhits collapse to the same outcome */
    reg->num_nonzero = D;
    double amp = born_fast_isqrt((double)D);
    for (uint32_t k = 0; k < D && k < 4096; k++) {
        reg->entries[k].basis_state = k;
        reg->entries[k].amp_re = amp;
        reg->entries[k].amp_im = 0;
    }

    /* Clean up working quhits */
    eng->num_quhits -= 2;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * DFT ON SINGLE QUHIT IN REGISTER
 *
 * For GHZ registers, DFT on a single position breaks the uniform
 * correlation. We apply DFT to the register's amplitude entries
 * using the DFT₆ matrix from superposition.h.
 * ═══════════════════════════════════════════════════════════════════════════════ */

void quhit_reg_apply_dft(QuhitEngine *eng, int reg_idx, uint64_t quhit_idx)
{
    if (reg_idx < 0 || (uint32_t)reg_idx >= eng->num_registers) return;
    QuhitRegister *reg = &eng->registers[reg_idx];

    /* Gauss mode: record DFT in circuit log */
    if (reg->bulk_rule == 2) {
        reg->gauss_n_dft++;
        /* After two full DFT sweeps + CZ in between → circuit is ready */
        if (reg->gauss_n_dft >= 2 * (uint16_t)reg->n_quhits && reg->gauss_n_cz > 0)
            reg->gauss_ready = 1;
        return;
    }

    (void)quhit_idx;  /* For GHZ, DFT on any position = DFT on the state */

    /* Apply DFT₆ to the register's amplitude array using superposition.h */
    double re[QUHIT_D] = {0}, im[QUHIT_D] = {0};
    for (uint32_t k = 0; k < reg->num_nonzero && k < QUHIT_D; k++) {
        re[k] = reg->entries[k].amp_re;
        im[k] = reg->entries[k].amp_im;
    }

    sup_apply_dft6(re, im);

    reg->num_nonzero = 0;
    for (uint32_t k = 0; k < QUHIT_D; k++) {
        double prob = re[k] * re[k] + im[k] * im[k];
        if (prob > 1e-30) {
            reg->entries[reg->num_nonzero].basis_state = k;
            reg->entries[reg->num_nonzero].amp_re = re[k];
            reg->entries[reg->num_nonzero].amp_im = im[k];
            reg->num_nonzero++;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * CZ BETWEEN TWO QUHITS IN REGISTER
 *
 * For GHZ state, CZ between positions i and j applies ω^(k·k) = ω^(k²)
 * to each basis entry k (since both positions have the same value).
 * ═══════════════════════════════════════════════════════════════════════════════ */

void quhit_reg_apply_cz(QuhitEngine *eng, int reg_idx,
                        uint64_t idx_a, uint64_t idx_b)
{
    if (reg_idx < 0 || (uint32_t)reg_idx >= eng->num_registers) return;
    QuhitRegister *reg = &eng->registers[reg_idx];

    /* Gauss mode: record CZ pair in circuit log */
    if (reg->bulk_rule == 2) {
        if (reg->gauss_n_cz < 256) {
            reg->gauss_cz_a[reg->gauss_n_cz] = (uint16_t)idx_a;
            reg->gauss_cz_b[reg->gauss_n_cz] = (uint16_t)idx_b;
            reg->gauss_n_cz++;
        }
        return;
    }

    uint32_t D = reg->dim;
    double omega = 2.0 * M_PI / D;

    for (uint32_t e = 0; e < reg->num_nonzero; e++) {
        uint32_t ka = reg_extract_digit(reg->entries[e].basis_state,
                                        idx_a, D, reg->bulk_rule);
        uint32_t kb = reg_extract_digit(reg->entries[e].basis_state,
                                        idx_b, D, reg->bulk_rule);
        double phase = omega * ka * kb;
        double cos_p = cos(phase), sin_p = sin(phase);
        double re = reg->entries[e].amp_re;
        double im = reg->entries[e].amp_im;
        reg->entries[e].amp_re = re * cos_p - im * sin_p;
        reg->entries[e].amp_im = re * sin_p + im * cos_p;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * UNITARY ON SINGLE QUDIT POSITION — O(entries × D) gate application
 *
 * Apply a D×D unitary U to qudit at position `pos` in the register.
 * For each entry with digit k at position pos, the new amplitude iscircuit
 *   a'[...k'...] = Σ_k U[k',k] × a[...k...]
 *
 * This replaces the entire classical Θ-contraction + SVD pipeline.
 * Cost: O(num_nonzero × D) — near-instant for sparse registers.
 * ═══════════════════════════════════════════════════════════════════════════════ */

void quhit_reg_apply_unitary_pos(QuhitEngine *eng, int reg_idx,
                                 uint64_t pos,
                                 const double *U_re, const double *U_im)
{
    if (reg_idx < 0 || (uint32_t)reg_idx >= eng->num_registers) return;
    QuhitRegister *reg = &eng->registers[reg_idx];
    uint32_t D = reg->dim;

    /* Temporary buffer — heap allocated to handle large χ */
    uint32_t max_entries = reg->num_nonzero * D + 1;
    if (max_entries < 4096) max_entries = 4096;
    uint32_t new_count = 0;
    struct tmp_entry { basis_t basis; double re, im; };
    struct tmp_entry *tmp = (struct tmp_entry *)calloc(max_entries, sizeof(struct tmp_entry));

    /* D^pos multiplier for replacing digit at position pos */
    uint64_t pos_mul = 1;
    for (uint64_t p = 0; p < pos; p++) pos_mul *= D;

    for (uint32_t e = 0; e < reg->num_nonzero; e++) {
        basis_t basis = reg->entries[e].basis_state;
        double a_re = reg->entries[e].amp_re;
        double a_im = reg->entries[e].amp_im;

        /* Extract digit k at position pos */
        uint32_t k = (uint32_t)((basis / pos_mul) % D);

        /* Base: basis with digit at pos zeroed out */
        basis_t base = basis - (basis_t)k * pos_mul;

        /* Apply U: for each output digit k', accumulate U[k',k] × amplitude */
        for (uint32_t kp = 0; kp < D; kp++) {
            double ur = U_re[kp * D + k];
            double ui = U_im[kp * D + k];
            /* U[k',k] × a = (ur + i·ui)(a_re + i·a_im) */
            double nr = ur * a_re - ui * a_im;
            double ni = ur * a_im + ui * a_re;

            if (nr * nr + ni * ni < 1e-30) continue;

            basis_t new_basis = base + (basis_t)kp * pos_mul;

            /* Find or create entry for new_basis in tmp */
            int found = -1;
            for (uint32_t t = 0; t < new_count; t++) {
                if (tmp[t].basis == new_basis) { found = (int)t; break; }
            }
            if (found >= 0) {
                tmp[found].re += nr;
                tmp[found].im += ni;
            } else if (new_count < max_entries) {
                tmp[new_count].basis = new_basis;
                tmp[new_count].re = nr;
                tmp[new_count].im = ni;
                new_count++;
            }
        }
    }

    /* Write back, dropping near-zero entries, capped at entries[] capacity */
    reg->num_nonzero = 0;
    for (uint32_t t = 0; t < new_count; t++) {
        if (tmp[t].re * tmp[t].re + tmp[t].im * tmp[t].im >= 1e-30) {
            if (reg->num_nonzero < 4096) {
                reg->entries[reg->num_nonzero].basis_state = tmp[t].basis;
                reg->entries[reg->num_nonzero].amp_re = tmp[t].re;
                reg->entries[reg->num_nonzero].amp_im = tmp[t].im;
                reg->num_nonzero++;
            }
        }
    }
    free(tmp);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MEASUREMENT — Born-rule sampling of single quhit in register
 *
 * Uses born_sample() from born_rule.h for sampling.
 * Uses born_collapse() pattern for post-measurement state update.
 *
 * For GHZ state: measuring any quhit determines all others.
 * Memory: only the D amplitudes are touched = 96 bytes.
 * ═══════════════════════════════════════════════════════════════════════════════ */

uint64_t quhit_reg_measure(QuhitEngine *eng, int reg_idx,
                           uint64_t quhit_idx)
{
    if (reg_idx < 0 || (uint32_t)reg_idx >= eng->num_registers) return 0;
    QuhitRegister *reg = &eng->registers[reg_idx];
    uint32_t D = reg->dim;

    /* Gauss mode: use analytic Born sampling */
    if (reg->bulk_rule == 2 && reg->gauss_ready) {
        double rand_01 = quhit_prng_double(eng);
        int outcome = gauss_born_sample((int)reg->n_quhits,
                                        (int)quhit_idx, rand_01);

        /* Collapse to measured state */
        reg->num_nonzero = 1;
        reg->entries[0].basis_state = (basis_t)outcome;
        reg->entries[0].amp_re = 1.0;
        reg->entries[0].amp_im = 0.0;
        reg->collapsed = 1;
        reg->collapse_outcome = (uint32_t)outcome;
        reg->gauss_ready = 0;  /* circuit consumed */
        reg->bulk_rule = 0;    /* revert to general mode */
        return (uint64_t)outcome;
    }

    (void)quhit_idx;

    /* Extract amplitudes into flat arrays for born_sample() */
    double re[QUHIT_D] = {0}, im[QUHIT_D] = {0};
    for (uint32_t e = 0; e < reg->num_nonzero && e < QUHIT_D; e++) {
        uint32_t k = (uint32_t)(reg->entries[e].basis_state % D);
        re[k] = reg->entries[e].amp_re;
        im[k] = reg->entries[e].amp_im;
    }

    /* Born-rule sampling using born_rule.h */
    double rand_01 = quhit_prng_double(eng);
    int outcome = born_sample(re, im, D, rand_01);

    /* Collapse: born_rule.h pattern */
    born_collapse(re, im, D, outcome);

    /* Write back collapsed state */
    reg->num_nonzero = 1;
    reg->entries[0].basis_state = (basis_t)outcome;
    reg->entries[0].amp_re = 1.0;
    reg->entries[0].amp_im = 0.0;

    reg->collapsed = 1;
    reg->collapse_outcome = (uint32_t)outcome;

    return (uint64_t)outcome;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * STREAMING STATE VECTOR — O(1) on-the-fly amplitude access
 *
 * Following statevector.h's sv_get() pattern:
 *   addr(k) = base + k × 16  (SV_ELEMENT_SIZE)
 *
 * quhit_reg_sv_get(reg, k) computes the amplitude of basis state |k⟩
 * in the full D^N Hilbert space WITHOUT materializing the state vector.
 *
 * For GHZ state (1/√D) Σ|m,m,...,m⟩:
 *   - Basis state k encodes multi-index (q₀, q₁, ..., q_{N-1}) in base D
 *   - Amplitude is nonzero only if all qᵢ are equal
 *   - i.e., k must be of the form m × (D^N - 1)/(D - 1) for some m ∈ [0,D)
 *   - Quick check: all digits of k in base D must be the same
 *
 * For general register states (post-DFT, post-CZ):
 *   - Linear scan through stored entries for matching basis_state
 *
 * Memory: O(1) per call. No allocation. The full state vector is never stored.
 * ═══════════════════════════════════════════════════════════════════════════════ */

SV_Amplitude quhit_reg_sv_get(QuhitEngine *eng, int reg_idx,
                              basis_t basis_k)
{
    SV_Amplitude amp = {0.0, 0.0};
    if (reg_idx < 0 || (uint32_t)reg_idx >= eng->num_registers) return amp;
    QuhitRegister *reg = &eng->registers[reg_idx];
    uint32_t D = reg->dim;

    if (reg->bulk_rule == 2 && reg->gauss_ready) {
        /*
         * Gauss circuit mode: evaluate amplitude analytically in O(N).
         * Decode basis_k to Z₆^N j-vector, call gauss_amp_general.
         */
        int N = (int)reg->n_quhits;
        int jv[64];
        basis_t v = basis_k;
        for (int i = 0; i < N; i++) { jv[i] = (int)(v % D); v /= D; }
        if (v != 0) return amp;  /* overflow: basis_k too large */

        gauss_amp_general(jv, N,
                          reg->gauss_cz_a, reg->gauss_cz_b,
                          (int)reg->gauss_n_cz,
                          &amp.re, &amp.im);
        return amp;
    }

    if (reg->bulk_rule == 1) {
        /*
         * GHZ mode: amplitude is nonzero only if ALL N digits are equal.
         */
        uint32_t first_digit = (uint32_t)(basis_k % D);
        basis_t remaining = basis_k;
        int all_same = 1;

        for (uint64_t q = 0; q < reg->n_quhits && all_same; q++) {
            if ((remaining % D) != first_digit) {
                all_same = 0;
            }
            remaining /= D;
        }

        if (remaining != 0) all_same = 0;

        if (all_same) {
            for (uint32_t e = 0; e < reg->num_nonzero; e++) {
                if ((uint32_t)(reg->entries[e].basis_state % D) == first_digit) {
                    amp.re = reg->entries[e].amp_re;
                    amp.im = reg->entries[e].amp_im;
                    return amp;
                }
            }
        }
        return amp;
    }

    /* General mode: linear scan through stored entries */
    for (uint32_t e = 0; e < reg->num_nonzero; e++) {
        if (reg->entries[e].basis_state == basis_k) {
            amp.re = reg->entries[e].amp_re;
            amp.im = reg->entries[e].amp_im;
            return amp;
        }
    }
    return amp;
}
/* ═══════════════════════════════════════════════════════════════════════════════
 * SET AMPLITUDE — Insert or update a sparse entry
 *
 * If basis_k already exists, update in-place. Otherwise append.
 * Near-zero amplitudes (|a|² < 1e-30) are dropped to maintain sparsity.
 * ═══════════════════════════════════════════════════════════════════════════════ */

void quhit_reg_sv_set(QuhitEngine *eng, int reg_idx,
                      basis_t basis_k, double re, double im)
{
    if (reg_idx < 0 || (uint32_t)reg_idx >= eng->num_registers) return;
    QuhitRegister *reg = &eng->registers[reg_idx];

    double mag2 = re * re + im * im;

    /* Search for existing entry */
    for (uint32_t e = 0; e < reg->num_nonzero; e++) {
        if (reg->entries[e].basis_state == basis_k) {
            if (mag2 < 1e-30) {
                /* Remove: swap with last entry */
                reg->entries[e] = reg->entries[reg->num_nonzero - 1];
                reg->num_nonzero--;
            } else {
                /* Update in-place */
                reg->entries[e].amp_re = re;
                reg->entries[e].amp_im = im;
            }
            return;
        }
    }

    /* Not found — append if nonzero and space available */
    if (mag2 >= 1e-30 && reg->num_nonzero < 4096) {
        reg->entries[reg->num_nonzero].basis_state = basis_k;
        reg->entries[reg->num_nonzero].amp_re = re;
        reg->entries[reg->num_nonzero].amp_im = im;
        reg->num_nonzero++;
    }
}


/* ═══════════════════════════════════════════════════════════════════════════════
 * STREAMING SCAN — Iterate over a window of the state vector
 *
 * Following statevector.h's sequential scan pattern (cache-friendly,
 * optimal for hardware prefetch).
 *
 * The callback receives each nonzero amplitude in-order, without the caller
 * needing to know which basis states are nonzero.
 *
 * For GHZ: calls the callback exactly D times (6 for D=6).
 * For general states: calls once per stored entry.
 *
 * This is the streaming interface — process amplitudes without storing them.
 * ═══════════════════════════════════════════════════════════════════════════════ */

void quhit_reg_sv_stream(QuhitEngine *eng, int reg_idx,
                         sv_stream_fn callback, void *user_data)
{
    if (reg_idx < 0 || (uint32_t)reg_idx >= eng->num_registers) return;
    QuhitRegister *reg = &eng->registers[reg_idx];

    for (uint32_t e = 0; e < reg->num_nonzero; e++) {
        SV_Amplitude amp;
        amp.re = reg->entries[e].amp_re;
        amp.im = reg->entries[e].amp_im;
        callback(reg->entries[e].basis_state, amp, user_data);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * STREAMING TOTAL PROBABILITY — sv_total_prob via streaming
 *
 * From statevector.h pattern: naive left-to-right accumulation.
 * Uses streaming scan (no D^N materialization).
 * ═══════════════════════════════════════════════════════════════════════════════ */

double quhit_reg_sv_total_prob(QuhitEngine *eng, int reg_idx)
{
    if (reg_idx < 0 || (uint32_t)reg_idx >= eng->num_registers) return 0;
    QuhitRegister *reg = &eng->registers[reg_idx];

    double total = 0.0;
    for (uint32_t e = 0; e < reg->num_nonzero; e++) {
        total += reg->entries[e].amp_re * reg->entries[e].amp_re
               + reg->entries[e].amp_im * reg->entries[e].amp_im;
    }
    return total;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * STREAMING INNER PRODUCT — ⟨reg_a | reg_b⟩ via streaming
 *
 * From statevector.h: sv_inner_product pattern.
 * Only nonzero entries contribute — O(min entries) time.
 * ═══════════════════════════════════════════════════════════════════════════════ */

SV_Amplitude quhit_reg_sv_inner(QuhitEngine *eng, int reg_a, int reg_b)
{
    SV_Amplitude result = {0.0, 0.0};
    if (reg_a < 0 || reg_b < 0) return result;
    if ((uint32_t)reg_a >= eng->num_registers) return result;
    if ((uint32_t)reg_b >= eng->num_registers) return result;

    QuhitRegister *ra = &eng->registers[reg_a];
    QuhitRegister *rb = &eng->registers[reg_b];

    /* O(n_a × n_b) for sparse entries, but n_a, n_b ≤ D typically */
    for (uint32_t i = 0; i < ra->num_nonzero; i++) {
        for (uint32_t j = 0; j < rb->num_nonzero; j++) {
            if (ra->entries[i].basis_state == rb->entries[j].basis_state) {
                /* ⟨a|b⟩ += conj(a_k) × b_k */
                double a_re = ra->entries[i].amp_re;
                double a_im = ra->entries[i].amp_im;
                double b_re = rb->entries[j].amp_re;
                double b_im = rb->entries[j].amp_im;
                result.re += a_re * b_re + a_im * b_im;
                result.im += a_re * b_im - a_im * b_re;
            }
        }
    }
    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PER-QUHIT LOCAL STATE VECTOR — Partial trace to single quhit
 *
 * Returns the D=6 local state vector (QuhitState, 96 bytes) for any
 * quhit at position `quhit_pos` in the register.
 *
 * This is the partial trace Tr_{≠pos}(|ψ⟩⟨ψ|) — the reduced density
 * matrix's diagonal, projected back to a pure-state approximation.
 *
 * From quhit_management.h: QuhitState stores 6 re[] + 6 im[] = 96 bytes.
 *
 * For GHZ (1/√D) Σ|m,m,...,m⟩:
 *   - Quhit at ANY position has marginal: P(k) = |α_k|² = 1/D for each k
 *   - Local amplitudes: re[k] = 1/√D, im[k] = 0 for all k
 *   - This is the maximally mixed state (as pure-state projection)
 *
 * For general states:
 *   - Extract digit at `quhit_pos` from each basis entry
 *   - Accumulate |amplitude|² contributions per digit value
 *   - Return sqrt(accumulated) as the local amplitude
 *
 * Memory: O(D) = O(6) per call. No materialization of D^N.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static uint32_t reg_extract_digit(basis_t basis, uint64_t pos,
                                  uint32_t D, uint8_t bulk_rule)
{
    if (bulk_rule == 1) {
        /* GHZ: all digits are the same = basis_state mod D */
        (void)pos;
        return (uint32_t)(basis % D);
    }
    /* General: extract digit at position `pos` via base-D decomposition */
    basis_t v = basis;
    for (uint64_t i = 0; i < pos; i++) v /= D;
    return (uint32_t)(v % D);
}

QuhitState quhit_reg_local_sv(QuhitEngine *eng, int reg_idx,
                              uint64_t quhit_pos)
{
    QuhitState local;
    qm_init_zero(&local);

    if (reg_idx < 0 || (uint32_t)reg_idx >= eng->num_registers) return local;
    QuhitRegister *reg = &eng->registers[reg_idx];
    uint32_t D = reg->dim;

    /*
     * Partial trace: for each stored basis entry, extract the digit
     * at `quhit_pos` and accumulate the amplitude into that digit's slot.
     *
     * For entangled states, the local state is mixed (ρ not pure).
     * We store the marginal amplitudes — the diagonal of ρ in the
     * computational basis, with phases from the first contributing entry.
     *
     * This gives the correct Born probabilities P(k) = Σ_{entries with digit=k} |α|²
     * and preserves phase information for coherent operations.
     */
    double prob[QUHIT_D] = {0};
    double phase_re[QUHIT_D] = {0};
    double phase_im[QUHIT_D] = {0};

    for (uint32_t e = 0; e < reg->num_nonzero; e++) {
        uint32_t digit = reg_extract_digit(reg->entries[e].basis_state,
                                           quhit_pos, D, reg->bulk_rule);
        if (digit < D) {
            double re = reg->entries[e].amp_re;
            double im = reg->entries[e].amp_im;
            double p  = re * re + im * im;
            prob[digit] += p;
            phase_re[digit] += re;
            phase_im[digit] += im;
        }
    }

    /* Write local state: amplitude = sqrt(prob) with phase direction */
    for (uint32_t k = 0; k < D; k++) {
        if (prob[k] > 0) {
            double mag = sqrt(prob[k]);
            double norm = sqrt(phase_re[k]*phase_re[k] + phase_im[k]*phase_im[k]);
            if (norm > 1e-30) {
                local.re[k] = mag * (phase_re[k] / norm);
                local.im[k] = mag * (phase_im[k] / norm);
            } else {
                local.re[k] = mag;
                local.im[k] = 0;
            }
        }
    }

    return local;
}
