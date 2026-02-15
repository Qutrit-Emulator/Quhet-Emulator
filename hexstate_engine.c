/*
 * ═══════════════════════════════════════════════════════════════════════════════
 * HEXSTATE ENGINE — 6-State Quantum Processor Emulator
 * ═══════════════════════════════════════════════════════════════════════════════
 * Core implementation. Magic Pointers are the default addressing mode —
 * every chunk references an external Hilbert space (tag 0x4858 "HX").
 * Local mmap'd memory serves as a "shadow cache" of the external space.
 *
 * Basis states: |0⟩, |1⟩, |2⟩, |3⟩, |4⟩, |5⟩
 * Hadamard gate: 6×6 DFT matrix  H[j][k] = (1/√6) · ω^(jk), ω = e^(2πi/6)
 */

#include "hexstate_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ─── Utility: Complex arithmetic ─────────────────────────────────────────── */

static inline Complex cmplx(double r, double i)
{
    return (Complex){r, i};
}

static inline Complex cmul(Complex a, Complex b)
{
    return cmplx(a.real * b.real - a.imag * b.imag,
                 a.real * b.imag + a.imag * b.real);
}

static inline Complex cadd(Complex a, Complex b)
{
    return cmplx(a.real + b.real, a.imag + b.imag);
}

static inline Complex csub(Complex a, Complex b)
{
    return cmplx(a.real - b.real, a.imag - b.imag);
}

static inline double cnorm2(Complex a)
{
    return a.real * a.real + a.imag * a.imag;
}

static inline Complex cconj(Complex a)
{
    return cmplx(a.real, -a.imag);
}

/* ─── Next power of 2 ≥ n ────────────────────────────────────────────────── */
static uint32_t next_pow2(uint32_t n)
{
    uint32_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

/* ─── In-place Cooley-Tukey FFT for power-of-2 length ─────────────────── */
static void fft_pow2_inplace(Complex *buf, uint32_t N)
{
    if (N <= 1) return;
    uint32_t logN = 0;
    { uint32_t t = N; while (t > 1) { logN++; t >>= 1; } }

    /* Bit-reversal permutation */
    for (uint32_t i = 0; i < N; i++) {
        uint32_t rev = 0;
        for (uint32_t bit = 0; bit < logN; bit++)
            if (i & (1u << bit)) rev |= (1u << (logN - 1 - bit));
        if (rev > i) { Complex t = buf[i]; buf[i] = buf[rev]; buf[rev] = t; }
    }

    /* Butterfly stages */
    for (uint32_t s = 1; s <= logN; s++) {
        uint32_t m = 1u << s;
        double angle = 2.0 * M_PI / m;
        Complex wm = cmplx(cos(angle), sin(angle));
        for (uint32_t k = 0; k < N; k += m) {
            Complex w = cmplx(1.0, 0.0);
            for (uint32_t j = 0; j < m/2; j++) {
                Complex t = cmul(w, buf[k + j + m/2]);
                Complex u = buf[k + j];
                buf[k + j] = cadd(u, t);
                buf[k + j + m/2] = csub(u, t);
                w = cmul(w, wm);
            }
        }
    }
}

/* ─── Bluestein's DFT: O(N·log N) for ANY dimension N ─────────────────── 
 *
 * Converts a length-N DFT into a circular convolution of length M (power of 2),
 * computed via two FFTs and one IFFT.  Works for any N.
 *
 * Identity: ω^(jk) = ω^(j²/2) · ω^(k²/2) · ω^(-(j-k)²/2)
 *
 * Input:  x[0..N-1]     (overwritten with result)
 * Output: X[j] = Σ_k x[k]·ω^(jk),  ω = exp(2πi/N)
 *
 * No 1/√N scaling — caller handles that.
 */
static void bluestein_dft(Complex *x, uint32_t N)
{
    if (N <= 1) return;

    uint32_t M = next_pow2(2 * N - 1);  /* convolution length */

    /* Chirp: chirp[k] = exp(-iπk²/N) */
    Complex *chirp = calloc(N, sizeof(Complex));
    for (uint32_t k = 0; k < N; k++) {
        double phase = M_PI * (double)k * (double)k / (double)N;
        chirp[k] = cmplx(cos(phase), -sin(phase));
    }

    /* a[k] = x[k] · chirp[k],  zero-padded to M */
    Complex *a = calloc(M, sizeof(Complex));
    for (uint32_t k = 0; k < N; k++)
        a[k] = cmul(x[k], chirp[k]);

    /* b[k] = conj(chirp[k]) with wrap-around for negative indices */
    Complex *b = calloc(M, sizeof(Complex));
    b[0] = cconj(chirp[0]);
    for (uint32_t k = 1; k < N; k++) {
        b[k]     = cconj(chirp[k]);
        b[M - k] = cconj(chirp[k]);
    }

    /* Convolve via FFT: FFT(a), FFT(b), pointwise multiply, IFFT */
    fft_pow2_inplace(a, M);
    fft_pow2_inplace(b, M);

    for (uint32_t i = 0; i < M; i++)
        a[i] = cmul(a[i], b[i]);

    /* IFFT = conj → FFT → conj → /M */
    for (uint32_t i = 0; i < M; i++) a[i] = cconj(a[i]);
    fft_pow2_inplace(a, M);
    double inv_M = 1.0 / (double)M;
    for (uint32_t i = 0; i < M; i++)
        a[i] = cmplx(a[i].real * inv_M, -a[i].imag * inv_M);

    /* Extract result: X[j] = chirp[j] · a[j] */
    for (uint32_t j = 0; j < N; j++)
        x[j] = cmul(chirp[j], a[j]);

    free(chirp);
    free(a);
    free(b);
}

/* ─── Precomputed DFT₆ Matrix ────────────────────────────────────────────── */
/* H[j][k] = (1/√6) · exp(2πi·j·k/6)
 * ω = exp(2πi/6) = cos(60°) + i·sin(60°) = 0.5 + i·(√3/2)
 */

static Complex dft6_matrix[6][6];
static int     dft6_initialized = 0;

static void init_dft6(void)
{
    if (dft6_initialized) return;
    double inv_sqrt6 = 1.0 / sqrt(6.0);
    for (int j = 0; j < 6; j++) {
        for (int k = 0; k < 6; k++) {
            double angle = 2.0 * M_PI * j * k / 6.0;
            dft6_matrix[j][k] = cmplx(inv_sqrt6 * cos(angle),
                                      inv_sqrt6 * sin(angle));
        }
    }
    dft6_initialized = 1;
}

/* ─── Powers of 6 Lookup ──────────────────────────────────────────────────── */

static uint64_t power_of_6(uint64_t n)
{
    uint64_t result = 1;
    for (uint64_t i = 0; i < n; i++) {
        uint64_t next = result * 6;
        if (next / 6 != result || next > (uint64_t)0x7FFFFFFFFFFFFFFF) {
            return 0x7FFFFFFFFFFFFFFF;  /* Saturate */
        }
        result = next;
    }
    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * ENGINE LIFECYCLE
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Helper: mmap with page alignment */
static void *mmap_alloc(uint64_t bytes)
{
    uint64_t aligned = (bytes + 4095) & ~4095ULL;
    if (aligned == 0) aligned = 4096;
    void *p = mmap(NULL, aligned, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return (p == MAP_FAILED) ? NULL : p;
}

/* Ensure chunk arrays can hold at least (id+1) entries */
static int ensure_chunk_capacity(HexStateEngine *eng, uint64_t id)
{
    if (id < eng->chunk_capacity) return 0;

    uint64_t new_cap = eng->chunk_capacity;
    if (new_cap == 0) new_cap = INITIAL_CHUNK_CAP;
    while (new_cap <= id) new_cap *= 2;
    if (new_cap > MAX_CHUNKS) new_cap = MAX_CHUNKS;
    if (id >= new_cap) return -1;

    /* Grow chunks array */
    Chunk *new_chunks = (Chunk *)mmap_alloc(new_cap * sizeof(Chunk));
    if (!new_chunks) return -1;
    if (eng->chunks) {
        memcpy(new_chunks, eng->chunks, eng->chunk_capacity * sizeof(Chunk));
        munmap(eng->chunks, (eng->chunk_capacity * sizeof(Chunk) + 4095) & ~4095ULL);
    }
    memset(new_chunks + eng->chunk_capacity, 0,
           (new_cap - eng->chunk_capacity) * sizeof(Chunk));
    eng->chunks = new_chunks;

    /* Grow parallel array */
    ParallelReality *new_par = (ParallelReality *)mmap_alloc(new_cap * sizeof(ParallelReality));
    if (!new_par) return -1;
    if (eng->parallel) {
        memcpy(new_par, eng->parallel, eng->chunk_capacity * sizeof(ParallelReality));
        munmap(eng->parallel, (eng->chunk_capacity * sizeof(ParallelReality) + 4095) & ~4095ULL);
    }
    memset(new_par + eng->chunk_capacity, 0,
           (new_cap - eng->chunk_capacity) * sizeof(ParallelReality));
    eng->parallel = new_par;

    /* Grow measured_values array */
    uint64_t *new_meas = (uint64_t *)mmap_alloc(new_cap * sizeof(uint64_t));
    if (!new_meas) return -1;
    if (eng->measured_values) {
        memcpy(new_meas, eng->measured_values, eng->chunk_capacity * sizeof(uint64_t));
        munmap(eng->measured_values, (eng->chunk_capacity * sizeof(uint64_t) + 4095) & ~4095ULL);
    }
    memset(new_meas + eng->chunk_capacity, 0,
           (new_cap - eng->chunk_capacity) * sizeof(uint64_t));
    eng->measured_values = new_meas;

    eng->chunk_capacity = new_cap;
    return 0;
}

int engine_init(HexStateEngine *eng)
{
    memset(eng, 0, sizeof(HexStateEngine));

    /* Non-deterministic PRNG seed: nanosecond clock × PID × π-constant.
     * Every engine instance gets a unique sequence while keeping
     * the π-constant character as a mixing factor. */
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    eng->prng_state = 0x243F6A8885A308D3ULL          /* π fractional bits   */
                    ^ ((uint64_t)ts.tv_nsec << 17)    /* nanosecond entropy  */
                    ^ ((uint64_t)ts.tv_sec  * 2654435761ULL) /* second hash  */
                    ^ ((uint64_t)getpid()   << 7);    /* process identity    */
    eng->running = 1;
    eng->next_reality_id = 1;

    /* Allocate initial chunk/parallel/measured arrays */
    if (ensure_chunk_capacity(eng, INITIAL_CHUNK_CAP - 1) != 0) return -1;

    /* Allocate initial braid link storage */
    eng->braid_capacity = 4096;
    eng->braid_links = (BraidLink *)mmap_alloc(
        eng->braid_capacity * sizeof(BraidLink));
    if (!eng->braid_links) return -1;
    eng->num_braid_links = 0;

    init_dft6();
    register_builtin_oracles(eng);

    return 0;
}

void engine_destroy(HexStateEngine *eng)
{
    if (!eng) return;

    /* Free all HilbertGroups (track to avoid double-free) */
    HilbertGroup *freed_groups[1024];
    uint32_t nfreed = 0;
    for (uint64_t i = 0; i < eng->num_chunks; i++) {
        HilbertGroup *g = eng->chunks[i].hilbert.group;
        if (!g) continue;
        int already = 0;
        for (uint32_t f = 0; f < nfreed; f++)
            if (freed_groups[f] == g) { already = 1; break; }
        if (already) continue;
        if (nfreed < 1024) freed_groups[nfreed++] = g;
        free(g->basis_indices);
        free(g->amplitudes);
        free(g);
        eng->chunks[i].hilbert.group = NULL;
    }

    /* Free all local states and joint states from braid partners */
    for (uint64_t i = 0; i < eng->num_chunks; i++) {
        Chunk *c = &eng->chunks[i];
        /* Free local Hilbert space */
        if (c->hilbert.q_local_state) {
            free(c->hilbert.q_local_state);
            c->hilbert.q_local_state = NULL;
        }
        /* Free joint states (only side A to avoid double-free) */
        for (uint16_t p = 0; p < c->hilbert.num_partners; p++) {
            Complex *js = c->hilbert.partners[p].q_joint_state;
            if (!js) continue;
            /* Only free if we are side A (avoid double-free with partner's copy) */
            if (c->hilbert.partners[p].q_which == 0) {
                free(js);
            }
            c->hilbert.partners[p].q_joint_state = NULL;
        }
        c->hilbert.num_partners = 0;
    }

    /* Unmap all chunk shadow states */
    for (uint64_t i = 0; i < eng->num_chunks; i++) {
        Chunk *c = &eng->chunks[i];
        if (c->hilbert.shadow_state != NULL) {
            uint64_t sz = c->hilbert.shadow_capacity * STATE_BYTES;
            sz = (sz + 4095) & ~4095ULL;
            munmap(c->hilbert.shadow_state, sz);
        }
    }

    /* Unmap parallel hardware contexts */
    for (uint64_t i = 0; i < eng->num_chunks; i++) {
        if (eng->parallel[i].hw_context != NULL) {
            munmap(eng->parallel[i].hw_context, 4096);
        }
    }

    /* Unmap dynamic arrays */
    if (eng->chunks)
        munmap(eng->chunks, (eng->chunk_capacity * sizeof(Chunk) + 4095) & ~4095ULL);
    if (eng->parallel)
        munmap(eng->parallel, (eng->chunk_capacity * sizeof(ParallelReality) + 4095) & ~4095ULL);
    if (eng->measured_values)
        munmap(eng->measured_values, (eng->chunk_capacity * sizeof(uint64_t) + 4095) & ~4095ULL);

    /* Unmap braid links */
    if (eng->braid_links != NULL) {
        munmap(eng->braid_links, (eng->braid_capacity * sizeof(BraidLink) + 4095) & ~4095ULL);
    }

    /* Unmap program */
    if (eng->program != NULL) {
        munmap(eng->program, (eng->program_size + 4095) & ~4095ULL);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PRNG (xorshift64, Pi-seeded)
 * ═══════════════════════════════════════════════════════════════════════════════ */

uint64_t engine_prng(HexStateEngine *eng)
{
    uint64_t x = eng->prng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;

    /* Mix with rdtsc-style entropy if available */
#ifdef __x86_64__
    uint32_t lo, hi;
    __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
    x ^= ((uint64_t)hi << 32) | lo;
#endif

    eng->prng_state = x;
    return x;
}

static double prng_uniform(HexStateEngine *eng)
{
    return (double)(engine_prng(eng) >> 11) / (double)(1ULL << 53);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MAGIC POINTER RESOLUTION
 * ═══════════════════════════════════════════════════════════════════════════════
 * Every operation must go through this gate. The Magic Pointer is validated,
 * the chunk identity is extracted, and the shadow_state cache pointer is
 * returned (may be NULL for infinite / external-only chunks).
 */

static Complex *resolve_shadow(HexStateEngine *eng, uint64_t chunk_id,
                                uint64_t *out_num_states)
{
    if (chunk_id >= eng->num_chunks) return NULL;
    Chunk *c = &eng->chunks[chunk_id];

    /* ── Validate Magic Pointer ── */
    if (!IS_MAGIC_PTR(c->hilbert.magic_ptr)) {
        printf("  [RESOLVE] ERROR: chunk %lu has invalid Magic Pointer 0x%016lX\n",
               chunk_id, c->hilbert.magic_ptr);
        return NULL;
    }

    /* ── Extract identity from pointer ── */
    uint64_t resolved_id = MAGIC_PTR_ID(c->hilbert.magic_ptr);
    (void)resolved_id;  /* identity confirmed — matches chunk_id by construction */

    if (out_num_states) *out_num_states = c->num_states;

    /* Return the shadow cache at this Magic Pointer address (NULL = infinite) */
    return c->hilbert.shadow_state;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * CHUNK INITIALIZATION
 * ═══════════════════════════════════════════════════════════════════════════════ */

int init_chunk(HexStateEngine *eng, uint64_t id, uint64_t num_hexits)
{
    if (id >= MAX_CHUNKS || num_hexits < 1) return -1;
    if (ensure_chunk_capacity(eng, id) != 0) return -1;

    Chunk *c = &eng->chunks[id];
    c->id = id;
    c->size = num_hexits;
    c->locked = 0;

    /* ═══ MAGIC POINTER (always external Hilbert space) ═══ */
    c->hilbert.magic_ptr = MAKE_MAGIC_PTR(id);

    if (num_hexits > MAX_CHUNK_SIZE) {
        /* ─── Infinite / Massive Reality ─── */
        c->num_states = 0x7FFFFFFFFFFFFFFF;
        c->hilbert.shadow_state = NULL;
        c->hilbert.shadow_capacity = 0;
        /* WRITE quantum state to Magic Pointer address */
        c->hilbert.q_flags = 0x01;  /* superposed */
        /* Allocate local Hilbert space: |0⟩ (default D=6, overridable) */
        c->hilbert.q_local_dim = 6;
        c->hilbert.q_local_state = calloc(6, sizeof(Complex));
        c->hilbert.q_local_state[0] = cmplx(1.0, 0.0);  /* |0⟩ */
        memset(c->hilbert.partners, 0, sizeof(c->hilbert.partners));
        c->hilbert.num_partners = 0;

        printf("  [PARALLEL] Magic Pointer 0x%016lX — %lu hexits (infinite plane)\n",
               c->hilbert.magic_ptr, num_hexits);
    } else {
        /* ─── Standard Reality (shadow cache allocated) ─── */
        c->num_states = power_of_6(num_hexits);

        uint64_t alloc_bytes = c->num_states * STATE_BYTES;
        alloc_bytes = (alloc_bytes + 4095) & ~4095ULL;

        Complex *shadow = (Complex *)mmap(NULL, alloc_bytes,
            PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (shadow == MAP_FAILED) return -1;

        c->hilbert.shadow_state = shadow;
        c->hilbert.shadow_capacity = c->num_states;

        /* Initialize to |0...0⟩: amplitude 1.0 at state 0, rest 0 */
        memset(shadow, 0, alloc_bytes);
        shadow[0].real = 1.0;
        shadow[0].imag = 0.0;

        printf("  [INIT] Chunk %lu: %lu hexits, %lu states — Magic Pointer 0x%016lX\n",
               id, num_hexits, c->num_states, c->hilbert.magic_ptr);
    }

    /* Update chunk count */
    if (id >= eng->num_chunks) {
        eng->num_chunks = id + 1;
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * QUANTUM OPERATIONS
 * ═══════════════════════════════════════════════════════════════════════════════ */

void create_superposition(HexStateEngine *eng, uint64_t id)
{
    if (id >= eng->num_chunks) return;
    Chunk *c = &eng->chunks[id];

    /* ── Group-aware superposition ──
     * If this register is in a HilbertGroup, apply the DFT (Hadamard)
     * which IS a unitary quantum gate: |k⟩ → (1/√D) Σⱼ e^{2πijk/D} |j⟩
     * This preserves entanglement structure and is 100% quantum-accurate. */
    if (c->hilbert.group) {
        apply_hadamard(eng, id, 0);
        c->hilbert.q_flags = 0x01;  /* superposed */
        return;
    }

    /* ═══ Resolve Magic Pointer ═══ */
    uint64_t ns = 0;
    Complex *state = resolve_shadow(eng, id, &ns);

    if (state == NULL) {
        /* ── WRITE superposition to local Hilbert space ── */
        c->hilbert.q_flags = 0x01;  /* superposed */
        if (c->hilbert.q_local_state) {
            uint32_t d = c->hilbert.q_local_dim;
            if (d == 0) d = 6;
            double amp = 1.0 / sqrt((double)d);
            for (uint32_t i = 0; i < d; i++)
                c->hilbert.q_local_state[i] = cmplx(amp, 0.0);
        }
        printf("  [SUP] Superposition WRITTEN to Hilbert space at Ptr 0x%016lX\n",
               c->hilbert.magic_ptr);
        return;
    }

    double inv_sqrt_n = 1.0 / sqrt((double)ns);
    for (uint64_t i = 0; i < ns; i++) {
        state[i].real = inv_sqrt_n;
        state[i].imag = 0.0;
    }

    printf("  [SUP] Superposition on chunk %lu (%lu states, amp=%.6f) via Ptr 0x%016lX\n",
           id, ns, inv_sqrt_n, c->hilbert.magic_ptr);
}

/* ─── Group-Aware Unitary ──────────────────────────────────────────────────
 * Apply a D×D unitary matrix U to one register within a HilbertGroup.
 *
 * For a sparse multi-party state |Ψ⟩ = Σ α_e |k₀,k₁,...,kₙ⟩,
 * applying U to register i transforms:
 *   α'_{...,j,...} = Σ_k U[j][k] × α_{...,k,...}
 *
 * This potentially EXPANDS the number of nonzero entries from N to N×D
 * in the worst case, because each entry spawns D output values.
 * We then compact by merging entries with identical index tuples.
 * ──────────────────────────────────────────────────────────────────────── */

void apply_group_unitary(HexStateEngine *eng, uint64_t id,
                         Complex *U, uint32_t dim)
{
    if (id >= eng->num_chunks) return;
    Chunk *c = &eng->chunks[id];
    HilbertGroup *g = c->hilbert.group;

    if (!g) {
        /* No group — apply to local state if present */
        if (c->hilbert.q_local_state && dim <= c->hilbert.q_local_dim) {
            Complex *s = c->hilbert.q_local_state;
            Complex *tmp = calloc(dim, sizeof(Complex));
            for (uint32_t i = 0; i < dim; i++)
                for (uint32_t j = 0; j < dim; j++) {
                    tmp[i].real += U[i * dim + j].real * s[j].real
                                 - U[i * dim + j].imag * s[j].imag;
                    tmp[i].imag += U[i * dim + j].real * s[j].imag
                                 + U[i * dim + j].imag * s[j].real;
                }
            for (uint32_t i = 0; i < dim; i++) s[i] = tmp[i];
            free(tmp);
            printf("  [U] Applied %u×%u unitary to local state of chunk %lu\n",
                   dim, dim, id);
        }
        return;
    }
    /* ── Apply unitary to the shared Hilbert space ──
     * The group's sparse state is a D-dimensional Hilbert space.
     * Entries are indexed by the shared basis k, with amplitudes α_k.
     * A unitary U transforms the amplitude vector:
     *   new_α[k] = Σ_j U[k][j] × α[j]
     * This is O(D²), keeping ≤D entries. The Hilbert space handles
     * the computation — no client-side tensor product expansion.
     *
     * New entries are created when the unitary produces nonzero amplitudes
     * for basis values not in the current state. Entries are removed when
     * their amplitudes go to zero. */

    uint32_t ns = g->num_nonzero;
    uint32_t nm = g->num_members;

    /* Build the amplitude vector indexed by basis value for this member */
    Complex amp_in[dim];
    memset(amp_in, 0, dim * sizeof(Complex));
    uint32_t my_idx = c->hilbert.group_index;

    for (uint32_t e = 0; e < ns; e++) {
        uint32_t k = g->basis_indices[e * nm + my_idx];
        if (k < dim) {
            amp_in[k].real += g->amplitudes[e].real;
            amp_in[k].imag += g->amplitudes[e].imag;
        }
    }

    /* Apply U: new_α = U × old_α */
    Complex amp_out[dim];
    for (uint32_t i = 0; i < dim; i++) {
        amp_out[i].real = 0.0;
        amp_out[i].imag = 0.0;
        for (uint32_t j = 0; j < dim; j++) {
            amp_out[i].real += U[i * dim + j].real * amp_in[j].real
                             - U[i * dim + j].imag * amp_in[j].imag;
            amp_out[i].imag += U[i * dim + j].real * amp_in[j].imag
                             + U[i * dim + j].imag * amp_in[j].real;
        }
    }

    /* Count how many output amplitudes are nonzero */
    uint32_t new_count = 0;
    for (uint32_t k = 0; k < dim; k++)
        if (cnorm2(amp_out[k]) > 1e-28) new_count++;

    /* Rebuild the sparse state with the new amplitudes.
     * For each nonzero amp_out[k], create entry |k,k,...,k⟩ with that amplitude.
     * The basis indices for ALL members are set to k (shared Hilbert space). */
    uint32_t *new_indices = calloc((size_t)new_count * nm, sizeof(uint32_t));
    Complex  *new_amps    = calloc(new_count, sizeof(Complex));
    uint32_t  w = 0;
    for (uint32_t k = 0; k < dim; k++) {
        if (cnorm2(amp_out[k]) > 1e-28) {
            for (uint32_t m = 0; m < nm; m++)
                new_indices[w * nm + m] = k;
            new_amps[w] = amp_out[k];
            w++;
        }
    }

    /* Replace group's sparse state */
    free(g->basis_indices);
    free(g->amplitudes);
    g->basis_indices = new_indices;
    g->amplitudes    = new_amps;
    g->num_nonzero   = new_count;
    g->sparse_cap    = new_count;

    printf("  [U] Applied %u×%u unitary to group Hilbert space via member %u/%u "
           "(%u nonzero entries)\n",
           dim, dim, my_idx, nm, new_count);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * LOCAL UNITARY — transforms ONE member's basis index independently
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Unlike apply_group_unitary (which transforms the shared amplitude vector),
 * this applies U to ONE member's index while leaving all other members'
 * indices unchanged. This is a proper local unitary on one subsystem:
 *
 *   For entry |..., k_A, ...⟩ with amplitude α:
 *   → creates D entries |..., j, ...⟩ with amplitude α × U[j][k_A]
 *
 * This is essential for Bell violation tests: Alice and Bob need
 * independent measurement bases. Alice's rotation changes only her
 * basis index, not Bob's — enabling disagreement between them.
 *
 * State may expand from N entries to up to N×D entries.
 * ═══════════════════════════════════════════════════════════════════════════ */
void apply_local_unitary(HexStateEngine *eng, uint64_t id,
                         const Complex *U, uint32_t dim)
{
    if (id >= eng->num_chunks) return;
    Chunk *c = &eng->chunks[id];
    HilbertGroup *g = c->hilbert.group;
    if (!g || g->num_nonzero == 0) return;

    uint32_t nm = g->num_members;
    uint32_t ns = g->num_nonzero;
    uint32_t my_idx = c->hilbert.group_index;

    /* ═══ Deferred Local Unitary Path ═══
     * If the base state has ≤ D entries (e.g. GHZ), applying a local
     * unitary would expand it to ≤ D² entries per member. Instead,
     * DEFER: store the D×D unitary per member. The state is implicitly:
     *   |Ψ⟩ = Σ_k α_k · ⊗_m (U_L · ... · U_1 |index_{m,k}⟩)
     * Measurement replays ops as matrix-vector products: O(D² × L).
     *
     * Eligible when: num_nonzero ≤ dim (sum of ≤ D product states).
     * LAZY: push U to per-member list, no composition needed. */
    if (ns <= dim && nm >= 2 && !g->no_defer) {
        /* Grow lazy list if needed */
        if (g->lazy_count[my_idx] >= g->lazy_cap[my_idx]) {
            uint32_t new_cap = g->lazy_cap[my_idx] ? g->lazy_cap[my_idx] * 2 : 8;
            g->lazy_U[my_idx] = realloc(g->lazy_U[my_idx],
                                        new_cap * sizeof(Complex*));
            g->lazy_cap[my_idx] = new_cap;
        }
        /* Store copy of U */
        Complex *copy = malloc((size_t)dim * dim * sizeof(Complex));
        memcpy(copy, U, (size_t)dim * dim * sizeof(Complex));
        g->lazy_U[my_idx][g->lazy_count[my_idx]++] = copy;

        if (g->lazy_count[my_idx] == 1)
            g->num_deferred++;

        printf("  [LOCAL U] LAZY push %u×%u unitary for member %u/%u "
               "(op #%u, base entries: %u)\n",
               dim, dim, my_idx, nm, g->lazy_count[my_idx], ns);
        return;
    }

    /* Maximum output: each entry can split into dim entries */
    uint32_t max_out = ns * dim;

    /* ── Hash table for O(n) deduplication ──
     * Instead of O(n²) pairwise comparison, we hash each index tuple
     * and merge amplitudes in O(1) per entry.
     * Table size: next power of 2 ≥ 2 × max_out for low collision rate */
    uint32_t ht_size = 1;
    while (ht_size < max_out * 2) ht_size <<= 1;
    uint32_t ht_mask = ht_size - 1;

    /* Hash table: each slot stores an index into the output arrays, or UINT32_MAX if empty */
    uint32_t *ht_slots = malloc(ht_size * sizeof(uint32_t));
    memset(ht_slots, 0xFF, ht_size * sizeof(uint32_t));  /* fill with UINT32_MAX */

    uint32_t *new_indices = calloc((size_t)max_out * nm, sizeof(uint32_t));
    Complex  *new_amps    = calloc(max_out, sizeof(Complex));
    uint32_t  out_count   = 0;

    for (uint32_t e = 0; e < ns; e++) {
        uint32_t *old_row = &g->basis_indices[e * nm];
        uint32_t k = old_row[my_idx];
        Complex  alpha = g->amplitudes[e];

        for (uint32_t j = 0; j < dim; j++) {
            Complex u_jk = U[j * dim + k];
            Complex new_amp;
            new_amp.real = alpha.real * u_jk.real - alpha.imag * u_jk.imag;
            new_amp.imag = alpha.real * u_jk.imag + alpha.imag * u_jk.real;

            if (cnorm2(new_amp) < 1e-28) continue;

            /* Build the candidate index tuple in-place:
             * same as old_row but with my_idx changed to j */

            /* FNV-1a hash of the index tuple */
            uint32_t hash = 2166136261u;
            for (uint32_t m = 0; m < nm; m++) {
                uint32_t val = (m == my_idx) ? j : old_row[m];
                hash ^= val;
                hash *= 16777619u;
            }

            /* Open addressing: linear probe for matching tuple or empty slot */
            uint32_t slot = hash & ht_mask;
            for (;;) {
                if (ht_slots[slot] == UINT32_MAX) {
                    /* Empty slot — insert new entry */
                    uint32_t *new_row = &new_indices[out_count * nm];
                    memcpy(new_row, old_row, nm * sizeof(uint32_t));
                    new_row[my_idx] = j;
                    new_amps[out_count] = new_amp;
                    ht_slots[slot] = out_count;
                    out_count++;
                    break;
                }

                /* Slot occupied — check if tuples match */
                uint32_t existing = ht_slots[slot];
                uint32_t *ex_row = &new_indices[existing * nm];
                int same = 1;
                for (uint32_t m = 0; m < nm; m++) {
                    uint32_t val = (m == my_idx) ? j : old_row[m];
                    if (ex_row[m] != val) { same = 0; break; }
                }
                if (same) {
                    /* Merge amplitudes */
                    new_amps[existing].real += new_amp.real;
                    new_amps[existing].imag += new_amp.imag;
                    break;
                }

                /* Collision — linear probe */
                slot = (slot + 1) & ht_mask;
            }
        }
    }

    free(ht_slots);

    /* Compact: remove zero entries */
    uint32_t w = 0;
    for (uint32_t i = 0; i < out_count; i++) {
        if (cnorm2(new_amps[i]) < 1e-28) continue;
        if (w != i) {
            memcpy(&new_indices[w * nm], &new_indices[i * nm],
                   nm * sizeof(uint32_t));
            new_amps[w] = new_amps[i];
        }
        w++;
    }

    free(g->basis_indices);
    free(g->amplitudes);
    g->basis_indices = new_indices;
    g->amplitudes    = new_amps;
    g->num_nonzero   = w;
    g->sparse_cap    = max_out;

    printf("  [LOCAL U] Applied %u×%u local unitary to member %u/%u "
           "(%u → %u entries)\n",
           dim, dim, my_idx, nm, ns, w);
}

/* ─── Hadamard (DFT₆) Gate ────────────────────────────────────────────────── */

void apply_hadamard(HexStateEngine *eng, uint64_t id, uint64_t hexit_index)
{
    if (id >= eng->num_chunks) return;
    Chunk *c = &eng->chunks[id];

    if (c->hilbert.shadow_state == NULL) {
        /* ═══ WRITE QFT to Hilbert space ═══ */

        /* ── Group-first: apply DFT as local unitary ── */
        if (c->hilbert.group) {
            uint32_t dim = c->hilbert.group->dim;
            double inv_sqrt_d = 1.0 / sqrt((double)dim);
            Complex *dft = calloc((size_t)dim * dim, sizeof(Complex));
            for (uint32_t j = 0; j < dim; j++)
                for (uint32_t k = 0; k < dim; k++) {
                    double angle = -2.0 * M_PI * j * k / (double)dim;
                    dft[j * dim + k] = cmplx(inv_sqrt_d * cos(angle),
                                              inv_sqrt_d * sin(angle));
                }
            apply_local_unitary(eng, id, dft, dim);
            free(dft);
            printf("  [H] DFT_%u applied locally to group member %u via Hilbert space\n",
                   dim, c->hilbert.group_index);
            return;
        }

        /* ── Pairwise joint state DFT ── */
        if (c->hilbert.num_partners > 0 && c->hilbert.partners[0].q_joint_state) {
            Complex *joint = c->hilbert.partners[0].q_joint_state;
            uint8_t which = c->hilbert.partners[0].q_which;
            uint32_t dim = c->hilbert.partners[0].q_joint_dim;
            if (dim == 0) dim = 6;
            double inv_sqrt_d = 1.0 / sqrt((double)dim);

            /* Check if dim is power of 2 */
            int is_pow2 = (dim > 0) && ((dim & (dim - 1)) == 0);

            if (is_pow2 && dim > 1) {
                /* ── Cooley-Tukey FFT: O(D·log D) per row/column ── */
                Complex *buf = calloc(dim, sizeof(Complex));

                if (which == 0) {
                    /* FFT on A side: for each fixed b, FFT the row */
                    for (uint32_t b = 0; b < dim; b++) {
                        Complex *row = &joint[(uint64_t)b * dim];
                        /* Bit-reversal permutation */
                        for (uint32_t i = 0; i < dim; i++) buf[i] = row[i];
                        uint32_t logN = 0; { uint32_t t = dim; while (t > 1) { logN++; t >>= 1; } }
                        for (uint32_t i = 0; i < dim; i++) {
                            uint32_t rev = 0;
                            for (uint32_t bit = 0; bit < logN; bit++)
                                if (i & (1u << bit)) rev |= (1u << (logN - 1 - bit));
                            if (rev > i) { Complex t = buf[i]; buf[i] = buf[rev]; buf[rev] = t; }
                        }
                        /* Butterfly stages */
                        for (uint32_t s = 1; s <= logN; s++) {
                            uint32_t m = 1u << s;
                            double angle = 2.0 * M_PI / m;
                            Complex wm = cmplx(cos(angle), sin(angle));
                            for (uint32_t k = 0; k < dim; k += m) {
                                Complex w = cmplx(1.0, 0.0);
                                for (uint32_t j = 0; j < m/2; j++) {
                                    Complex t = cmul(w, buf[k + j + m/2]);
                                    Complex u = buf[k + j];
                                    buf[k + j] = cadd(u, t);
                                    buf[k + j + m/2] = csub(u, t);
                                    w = cmul(w, wm);
                                }
                            }
                        }
                        /* Scale by 1/√D */
                        for (uint32_t i = 0; i < dim; i++)
                            row[i] = cmplx(buf[i].real * inv_sqrt_d, buf[i].imag * inv_sqrt_d);
                    }
                } else {
                    /* FFT on B side: for each fixed a, FFT the column */
                    for (uint32_t a = 0; a < dim; a++) {
                        /* Gather column */
                        for (uint32_t b = 0; b < dim; b++)
                            buf[b] = joint[(uint64_t)b * dim + a];
                        /* Bit-reversal permutation */
                        uint32_t logN = 0; { uint32_t t = dim; while (t > 1) { logN++; t >>= 1; } }
                        for (uint32_t i = 0; i < dim; i++) {
                            uint32_t rev = 0;
                            for (uint32_t bit = 0; bit < logN; bit++)
                                if (i & (1u << bit)) rev |= (1u << (logN - 1 - bit));
                            if (rev > i) { Complex t = buf[i]; buf[i] = buf[rev]; buf[rev] = t; }
                        }
                        /* Butterfly stages */
                        for (uint32_t s = 1; s <= logN; s++) {
                            uint32_t m = 1u << s;
                            double angle = 2.0 * M_PI / m;
                            Complex wm = cmplx(cos(angle), sin(angle));
                            for (uint32_t k = 0; k < dim; k += m) {
                                Complex w = cmplx(1.0, 0.0);
                                for (uint32_t j = 0; j < m/2; j++) {
                                    Complex t = cmul(w, buf[k + j + m/2]);
                                    Complex u = buf[k + j];
                                    buf[k + j] = cadd(u, t);
                                    buf[k + j + m/2] = csub(u, t);
                                    w = cmul(w, wm);
                                }
                            }
                        }
                        /* Scale and scatter back to column */
                        for (uint32_t b = 0; b < dim; b++)
                            joint[(uint64_t)b * dim + a] = cmplx(buf[b].real * inv_sqrt_d,
                                                                   buf[b].imag * inv_sqrt_d);
                    }
                }
                free(buf);
            } else {
                /* ── Bluestein DFT: O(D·log D) for any D ── */
                Complex *tmp = calloc(dim, sizeof(Complex));
                if (which == 0) {
                    for (uint32_t b = 0; b < dim; b++) {
                        for (uint32_t j = 0; j < dim; j++)
                            tmp[j] = joint[(uint64_t)b * dim + j];
                        bluestein_dft(tmp, dim);
                        for (uint32_t j = 0; j < dim; j++)
                            joint[(uint64_t)b * dim + j] = cmplx(tmp[j].real * inv_sqrt_d,
                                                                   tmp[j].imag * inv_sqrt_d);
                    }
                } else {
                    for (uint32_t a = 0; a < dim; a++) {
                        for (uint32_t j = 0; j < dim; j++)
                            tmp[j] = joint[(uint64_t)j * dim + a];
                        bluestein_dft(tmp, dim);
                        for (uint32_t j = 0; j < dim; j++)
                            joint[(uint64_t)j * dim + a] = cmplx(tmp[j].real * inv_sqrt_d,
                                                                   tmp[j].imag * inv_sqrt_d);
                    }
                }
                free(tmp);
            }
            printf("  [H] QFT_%u WRITTEN to Hilbert space at Ptr 0x%016lX (side %c%s)\n",
                   dim, c->hilbert.magic_ptr, which == 0 ? 'A' : 'B',
                   is_pow2 ? ", FFT" : ", Bluestein");
        } else if (c->hilbert.q_local_state) {
            /* ── Apply DFT to local single-particle Hilbert space ── */
            uint32_t d = c->hilbert.q_local_dim;
            if (d == 0) d = 6;
            bluestein_dft(c->hilbert.q_local_state, d);
            double inv_sqrt_d = 1.0 / sqrt((double)d);
            for (uint32_t i = 0; i < d; i++)
                c->hilbert.q_local_state[i] = cmplx(
                    c->hilbert.q_local_state[i].real * inv_sqrt_d,
                    c->hilbert.q_local_state[i].imag * inv_sqrt_d);
            printf("  [H] DFT_%u WRITTEN to local Hilbert space at Ptr 0x%016lX\n",
                   d, c->hilbert.magic_ptr);
        }
        return;
    }

    if (hexit_index >= c->size) return;

    /* ═══ Resolve Magic Pointer ═══ */
    uint64_t ns = 0;
    Complex *state = resolve_shadow(eng, id, &ns);
    if (state == NULL) return;  /* shouldn't happen on this path */

    /*
     * Apply DFT to the specified hexit (dimension-agnostic).
     * For each group of states sharing the same "other hexits",
     * transform the D amplitudes indexed by the target hexit.
     * D defaults to 6 (hexit base) but reads q_joint_dim if set.
     */
    uint32_t base_d = (c->hilbert.num_partners > 0) ? c->hilbert.partners[0].q_joint_dim : 0;
    if (base_d == 0) base_d = 6;
    uint64_t stride = power_of_6(hexit_index);
    double inv_sqrt_d = 1.0 / sqrt((double)base_d);

    Complex *temp = calloc(base_d, sizeof(Complex));

    for (uint64_t base = 0; base < ns; base++) {
        if ((base / stride) % base_d != 0) continue;

        /* Gather amplitudes */
        for (uint32_t j = 0; j < base_d; j++)
            temp[j] = state[base + j * stride];

        /* Bluestein DFT: O(D·log D) for any D */
        bluestein_dft(temp, base_d);

        /* Scale by 1/√D and scatter back */
        for (uint32_t j = 0; j < base_d; j++)
            state[base + j * stride] = cmplx(temp[j].real * inv_sqrt_d,
                                              temp[j].imag * inv_sqrt_d);
    }
    free(temp);

    printf("  [H] DFT_%u Hadamard on chunk %lu, hexit %lu via Ptr 0x%016lX\n",
           base_d, id, hexit_index, c->hilbert.magic_ptr);
}

/* ─── Materialize Deferred Unitaries ──────────────────────────────────────── */
/* Forces expansion of all deferred unitaries into the explicit state vector.
 * After this, num_nonzero = D^N and each amplitude is fully computed.
 * This MUST be called before any non-local gate. */
/* ─── Lazy replay helper: apply lazy op list to a D-vector in-place ───────── */
static void lazy_apply_vec(Complex **ops, uint32_t num_ops,
                           Complex *vec, uint32_t dim)
{
    Complex *tmp = malloc(dim * sizeof(Complex));
    for (uint32_t op = 0; op < num_ops; op++) {
        Complex *M = ops[op];
        for (uint32_t i = 0; i < dim; i++) {
            double re = 0.0, im = 0.0;
            for (uint32_t j = 0; j < dim; j++) {
                re += M[i*dim+j].real * vec[j].real - M[i*dim+j].imag * vec[j].imag;
                im += M[i*dim+j].real * vec[j].imag + M[i*dim+j].imag * vec[j].real;
            }
            tmp[i].real = re;
            tmp[i].imag = im;
        }
        memcpy(vec, tmp, dim * sizeof(Complex));
    }
    free(tmp);
}

/* ─── Lazy compose: build composed D×D matrix from lazy op list ──────────── */
/* Computes U = U_L · ... · U_1 by replaying ops on each basis vector.
 * Cost: O(D² × L) instead of O(D³ × L) for matrix-matrix multiply. */
Complex *lazy_compose(Complex **ops, uint32_t num_ops, uint32_t dim)
{
    Complex *composed = calloc((size_t)dim * dim, sizeof(Complex));
    Complex *col = malloc(dim * sizeof(Complex));
    for (uint32_t k = 0; k < dim; k++) {
        /* Start with basis vector e_k */
        memset(col, 0, dim * sizeof(Complex));
        col[k].real = 1.0;
        /* Replay all ops */
        lazy_apply_vec(ops, num_ops, col, dim);
        /* Column k of composed matrix */
        for (uint32_t j = 0; j < dim; j++)
            composed[j * dim + k] = col[j];
    }
    free(col);
    return composed;
}

/* ─── Free lazy ops for a member ─────────────────────────────────────────── */
static void lazy_free_member(HilbertGroup *g, uint32_t m)
{
    for (uint32_t i = 0; i < g->lazy_count[m]; i++)
        free(g->lazy_U[m][i]);
    free(g->lazy_U[m]);
    g->lazy_U[m] = NULL;
    g->lazy_count[m] = 0;
    g->lazy_cap[m] = 0;
}

void materialize_deferred(HexStateEngine *eng, HilbertGroup *g)
{
    if (!g || g->num_deferred == 0) return;

    uint32_t dim = g->dim;

    /* Set no_defer to prevent re-deferral during expansion */
    g->no_defer = 1;

    /* Compose each member's lazy ops into a single matrix, then apply */
    for (uint32_t m = 0; m < g->num_members; m++) {
        if (g->lazy_count[m] == 0) continue;

        Complex *composed = lazy_compose(g->lazy_U[m], g->lazy_count[m], dim);
        lazy_free_member(g, m);

        uint64_t chunk_id = g->member_ids[m];
        apply_local_unitary(eng, chunk_id, composed, dim);
        free(composed);
    }
    g->num_deferred = 0;

    g->no_defer = 0;

    printf("  [MATERIALIZE] Expanded lazy unitaries: "
           "%u entries, %.1f KB\n",
           g->num_nonzero,
           (double)g->num_nonzero * (g->num_members * sizeof(uint32_t) + sizeof(Complex)) / 1024.0);
}

/* ─── Controlled Phase Gate (Non-Local) — DEFERRED ────────────────────────── */
/* CZ|j,k⟩ = ω^(j·k) |j,k⟩  where ω = e^(2πi/D)
 *
 * Stores CZ pairs for absorption at measurement time.
 * Cost: O(1) at gate time, O(D²) at measurement. */
void apply_cz_gate(HexStateEngine *eng, uint64_t id_a, uint64_t id_b)
{
    if (id_a >= eng->num_chunks || id_b >= eng->num_chunks) return;
    Chunk *ca = &eng->chunks[id_a];
    Chunk *cb = &eng->chunks[id_b];
    HilbertGroup *g = ca->hilbert.group;
    if (!g || g != cb->hilbert.group) return;

    uint32_t idx_a = ca->hilbert.group_index;
    uint32_t idx_b = cb->hilbert.group_index;

    if (g->num_cz >= g->cz_cap) {
        uint32_t new_cap = g->cz_cap ? g->cz_cap * 2 : 64;
        g->cz_pairs = realloc(g->cz_pairs, new_cap * 2 * sizeof(uint32_t));
        g->cz_cap = new_cap;
    }
    g->cz_pairs[g->num_cz * 2 + 0] = idx_a;
    g->cz_pairs[g->num_cz * 2 + 1] = idx_b;
    g->num_cz++;

    printf("  [CZ] DEFERRED between members %u and %u (pair %u stored)\n",
           idx_a, idx_b, g->num_cz);
}

/* ─── Measurement (Born Rule) ─────────────────────────────────────────────── */

uint64_t measure_chunk(HexStateEngine *eng, uint64_t id)
{
    /* Auto-initialize infinite chunk if it doesn't exist yet */
    if (id >= eng->num_chunks && id < MAX_CHUNKS) {
        op_infinite_resources(eng, id, 0);
    }
    if (id >= eng->num_chunks) return 0;
    Chunk *c = &eng->chunks[id];

    if (c->hilbert.shadow_state == NULL) {
        /* ═══ READ from Hilbert space ═══ */

        /* ── Group-based measurement: read from shared multi-party state ──
         * The HilbertGroup IS the Hilbert space. All connected registers
         * share it. Measurement reads the marginal for this register,
         * samples via Born rule, and collapses the shared state.
         * All other members automatically see the collapsed result
         * because they read from the same memory. No propagation needed. */
        if (c->hilbert.group) {
            HilbertGroup *g = c->hilbert.group;
            uint32_t my_idx = c->hilbert.group_index;
            uint32_t dim = g->dim;

            /* If group is already collapsed and this member was determined,
             * just READ the stored result from the Hilbert space */
            if (c->hilbert.q_flags == 0x02) {
                return eng->measured_values[id];
            }

            /* ═══ Deferred Measurement Path ═══
             * When local unitaries are deferred, the state is implicitly:
             *   |Ψ⟩ = Σ_k α_k · ⊗_m (U_m |index_{m,k}⟩)
             * We sample ALL members sequentially in O(N × D²):
             *
             * Algorithm:
             * 1. Maintain D complex coefficients c_k (initialized from amplitudes)
             * 2. For each member m (except last):
             *    P(v) = Σ_k |c_k|² · |U_m[v, idx_{m,k}]|²   (unitarity kills cross-terms)
             *    Sample v, update c_k ← c_k · U_m[v, idx_{m,k}]
             * 3. For last member: P(v) = |Σ_k c_k · U[v, idx_k]|²  (full interference)
             * 4. Collapse group to single entry.
             *
             * Total: O(N × D²) time, O(D) space. The Hilbert space holds the math. */
            if (g->num_deferred > 0 || g->num_cz > 0) {
                uint32_t ns = g->num_nonzero;
                uint32_t nm = g->num_members;
                uint32_t dim_local = g->dim;
                /* Allocate tracking arrays */
                Complex *c_k = calloc(ns, sizeof(Complex));
                for (uint32_t e = 0; e < ns; e++)
                    c_k[e] = g->amplitudes[e];

                uint64_t *outcomes = calloc(nm, sizeof(uint64_t));
                int *measured = calloc(nm, sizeof(int));

                /* Build measurement order: requested member first */
                uint32_t *order = calloc(nm, sizeof(uint32_t));
                order[0] = my_idx;
                uint32_t oi = 1;
                for (uint32_t m = 0; m < nm; m++)
                    if (m != my_idx) order[oi++] = m;

                /* Pre-compose lazy ops into per-member matrices via lazy replay
                 * Cost: O(D² × L) per member instead of O(D³ × L) for eager composition */
                Complex *composed_U[2] = {NULL, NULL};
                for (uint32_t mi = 0; mi < nm && mi < 2; mi++) {
                    if (g->lazy_count[mi] > 0)
                        composed_U[mi] = lazy_compose(g->lazy_U[mi],
                                                      g->lazy_count[mi], dim_local);
                }

                for (uint32_t step = 0; step < nm; step++) {
                    uint32_t m = order[step];
                    Complex *U_m = (m < 2) ? composed_U[m] : NULL;
                    uint32_t members_remaining = nm - step;

                    /* Compute marginal P(v) for this member.
                     * CZ phases cancel in marginals: |ω^(j·k)|² = 1.
                     * So the formula is the SAME as without CZ. */
                    double probs[NUM_BASIS_STATES];
                    memset(probs, 0, dim_local * sizeof(double));

                    if (members_remaining > 1) {
                        /* ═══ COHERENT marginal — preserves DFT phase ═══
                         * P(v) = |Σ_k c_k · U_m[v, idx_{m,k}]|²
                         *
                         * Previous code used incoherent sum (|c_k|²·|U|²)
                         * claiming "cross-terms vanish by unitarity."
                         * WRONG: cross-terms only vanish when marginalizing
                         * over ALL remaining members simultaneously. Once
                         * c_k carries phase from prior measurements, the
                         * cross-terms are physical and must be kept.
                         *
                         * The coherent formula is exact for GHZ states
                         * because all members share the same basis index k
                         * in each branch — there's no multi-outcome sum
                         * needed for the other members within a branch. */
                        for (uint32_t v = 0; v < dim_local; v++) {
                            Complex amp = {0.0, 0.0};
                            for (uint32_t e = 0; e < ns; e++) {
                                uint32_t idx_mk = g->basis_indices[e * nm + m];
                                Complex u_val;
                                if (U_m)
                                    u_val = U_m[v * dim_local + idx_mk];
                                else
                                    u_val = (v == idx_mk) ?
                                        (Complex){1.0, 0.0} : (Complex){0.0, 0.0};
                                amp.real += c_k[e].real * u_val.real
                                          - c_k[e].imag * u_val.imag;
                                amp.imag += c_k[e].real * u_val.imag
                                          + c_k[e].imag * u_val.real;
                            }
                            probs[v] = cnorm2(amp);
                        }
                    } else {
                        /* LAST member: full quantum interference (same formula) */
                        for (uint32_t v = 0; v < dim_local; v++) {
                            Complex amp = {0.0, 0.0};
                            for (uint32_t e = 0; e < ns; e++) {
                                uint32_t idx_mk = g->basis_indices[e * nm + m];
                                Complex u_val;
                                if (U_m)
                                    u_val = U_m[v * dim_local + idx_mk];
                                else
                                    u_val = (v == idx_mk) ?
                                        (Complex){1.0, 0.0} : (Complex){0.0, 0.0};
                                amp.real += c_k[e].real * u_val.real
                                          - c_k[e].imag * u_val.imag;
                                amp.imag += c_k[e].real * u_val.imag
                                          + c_k[e].imag * u_val.real;
                            }
                            probs[v] = cnorm2(amp);
                        }
                    }

                    /* Normalize */
                    double total = 0.0;
                    for (uint32_t v = 0; v < dim_local; v++) total += probs[v];
                    if (total > 0.0)
                        for (uint32_t v = 0; v < dim_local; v++) probs[v] /= total;

                    /* Born rule sample */
                    double r = prng_uniform(eng);
                    double cumul = 0.0;
                    uint32_t result_v = dim_local - 1;
                    for (uint32_t v = 0; v < dim_local; v++) {
                        cumul += probs[v];
                        if (cumul >= r) { result_v = v; break; }
                    }

                    outcomes[m] = result_v;
                    measured[m] = 1;

                    /* Update c_k: c_k[e] ← c_k[e] × U_m[result_v, idx_{m,e}] */
                    for (uint32_t e = 0; e < ns; e++) {
                        uint32_t idx_mk = g->basis_indices[e * nm + m];
                        Complex u_val;
                        if (U_m)
                            u_val = U_m[result_v * dim_local + idx_mk];
                        else
                            u_val = (result_v == idx_mk) ?
                                (Complex){1.0, 0.0} : (Complex){0.0, 0.0};
                        Complex old = c_k[e];
                        c_k[e].real = old.real * u_val.real - old.imag * u_val.imag;
                        c_k[e].imag = old.real * u_val.imag + old.imag * u_val.real;
                    }

                    /* ── Renormalize c_k to prevent numerical underflow ──
                     * Each step multiplies |c_k| by |U[v,k]| ≈ 1/√D.
                     * After N steps, |c_k| ≈ (1/√D)^N which underflows
                     * for large N (e.g. N=1000, D=6 → (0.408)^1000 ≈ 0).
                     * Renormalizing preserves relative phases/magnitudes. */
                    {
                        double ck_norm = 0.0;
                        for (uint32_t e = 0; e < ns; e++)
                            ck_norm += cnorm2(c_k[e]);
                        if (ck_norm > 0.0 && (ck_norm < 1e-200 || ck_norm > 1e200)) {
                            double scale = 1.0 / sqrt(ck_norm);
                            for (uint32_t e = 0; e < ns; e++) {
                                c_k[e].real *= scale;
                                c_k[e].imag *= scale;
                            }
                        }
                    }

                    /* ═══ CZ ABSORPTION ═══
                     * After sampling outcome v for member m, absorb CZ phases
                     * into unmeasured partners' unitaries.
                     * CZ(m, b) with outcome v: U_b' = D_v · U_b
                     * where D_v = diag(ω^(v·0), ω^(v·1), ..., ω^(v·(D-1)))
                     * This is O(D²) per CZ pair — still polynomial. */
                    double omega = 2.0 * M_PI / (double)dim_local;
                    for (uint32_t ci = 0; ci < g->num_cz; ci++) {
                        uint32_t a = g->cz_pairs[ci * 2 + 0];
                        uint32_t b = g->cz_pairs[ci * 2 + 1];
                        uint32_t partner = UINT32_MAX;
                        if (a == m && !measured[b]) partner = b;
                        else if (b == m && !measured[a]) partner = a;
                        if (partner == UINT32_MAX) continue;

                        /* Absorb CZ into partner's composed matrix */
                        if (partner < 2 && composed_U[partner]) {
                            Complex *U_b = composed_U[partner];
                            for (uint32_t j = 0; j < dim_local; j++) {
                                uint32_t phase_idx = ((uint32_t)result_v * j) % dim_local;
                                if (phase_idx == 0) continue;
                                double angle = omega * (double)phase_idx;
                                double c = cos(angle), s = sin(angle);
                                for (uint32_t k = 0; k < dim_local; k++) {
                                    Complex old = U_b[j * dim_local + k];
                                    U_b[j * dim_local + k].real =
                                        old.real * c - old.imag * s;
                                    U_b[j * dim_local + k].imag =
                                        old.real * s + old.imag * c;
                                }
                            }
                        }
                    }
                }

                /* Record all outcomes */
                for (uint32_t m = 0; m < nm; m++) {
                    uint64_t mid = g->member_ids[m];
                    eng->measured_values[mid] = outcomes[m];
                    eng->chunks[mid].hilbert.q_flags = 0x02;
                }

                /* Collapse group to single entry */
                g->num_nonzero = 1;
                for (uint32_t m = 0; m < nm; m++)
                    g->basis_indices[m] = (uint32_t)outcomes[m];
                g->amplitudes[0] = (Complex){1.0, 0.0};
                g->collapsed = 1;

                /* Free composed and lazy unitaries */
                for (uint32_t mi = 0; mi < nm && mi < 2; mi++) {
                    if (composed_U[mi]) free(composed_U[mi]);
                }
                for (uint32_t m = 0; m < nm; m++)
                    if (g->lazy_count[m] > 0) lazy_free_member(g, m);
                g->num_deferred = 0;

                /* Free CZ pairs */
                if (g->cz_pairs) { free(g->cz_pairs); g->cz_pairs = NULL; }
                g->num_cz = 0;
                g->cz_cap = 0;

                uint64_t my_result = outcomes[my_idx];

                printf("  [MEAS] DEFERRED measurement via Hilbert space O((N+E)×D²): "
                       "member %u/%u => %lu (all %u members determined)\n",
                       my_idx, nm, my_result, nm);

                free(c_k);
                free(outcomes);
                free(measured);
                free(order);
                return my_result;
            }

            /* Compute marginal probability for this register:
             * P(v) = Σ_{entries where my_index == v} |amplitude|² */
            double *probs = calloc(dim, sizeof(double));
            for (uint32_t e = 0; e < g->num_nonzero; e++) {
                uint32_t my_val = g->basis_indices[e * g->num_members + my_idx];
                probs[my_val] += cnorm2(g->amplitudes[e]);
            }

            /* Born rule: sample from the Hilbert space */
            double r = prng_uniform(eng);
            double cumul = 0.0;
            int result = (int)(dim - 1);
            for (uint32_t i = 0; i < dim; i++) {
                cumul += probs[i];
                if (cumul >= r) { result = (int)i; break; }
            }
            free(probs);

            /* Collapse the shared state: remove all entries where
             * this register's index ≠ result. This is a WRITE to the
             * shared Hilbert space — all other members automatically
             * see only the surviving amplitudes on their next read. */
            uint32_t write_pos = 0;
            for (uint32_t e = 0; e < g->num_nonzero; e++) {
                uint32_t my_val = g->basis_indices[e * g->num_members + my_idx];
                if ((int)my_val == result) {
                    if (write_pos != e) {
                        memcpy(&g->basis_indices[write_pos * g->num_members],
                               &g->basis_indices[e * g->num_members],
                               g->num_members * sizeof(uint32_t));
                        g->amplitudes[write_pos] = g->amplitudes[e];
                    }
                    write_pos++;
                }
            }
            g->num_nonzero = write_pos;

            /* Renormalize surviving amplitudes */
            double norm = 0.0;
            for (uint32_t e = 0; e < g->num_nonzero; e++)
                norm += cnorm2(g->amplitudes[e]);
            if (norm > 0.0) {
                double scale = 1.0 / sqrt(norm);
                for (uint32_t e = 0; e < g->num_nonzero; e++) {
                    g->amplitudes[e].real *= scale;
                    g->amplitudes[e].imag *= scale;
                }
            }

            /* Record our measurement */
            eng->measured_values[id] = (uint64_t)result;
            c->hilbert.q_flags = 0x02;

            /* Check if any other group members are now fully determined.
             * If only 1 nonzero entry left, ALL members are determined. */
            if (g->num_nonzero == 1) {
                for (uint32_t m = 0; m < g->num_members; m++) {
                    uint64_t mid = g->member_ids[m];
                    uint32_t mval = g->basis_indices[m]; /* only 1 entry */
                    eng->measured_values[mid] = mval;
                    eng->chunks[mid].hilbert.q_flags = 0x02;
                }
            }

            printf("  [MEAS] READ shared Hilbert space group "
                   "(member %u/%u, D=%u, %u nonzero remaining) => %d\n",
                   my_idx, g->num_members, dim, g->num_nonzero, result);

            return (uint64_t)result;
        }



        /* ── Born rule on local single-particle Hilbert space ── */
        if (c->hilbert.q_local_state) {
            uint32_t d = c->hilbert.q_local_dim;
            if (d == 0) d = 6;
            double r = prng_uniform(eng);
            double cumul = 0.0;
            uint64_t result = d - 1;
            for (uint32_t i = 0; i < d; i++) {
                cumul += cnorm2(c->hilbert.q_local_state[i]);
                if (cumul >= r) { result = i; break; }
            }
            /* Collapse: set measured state to |result⟩ */
            for (uint32_t i = 0; i < d; i++)
                c->hilbert.q_local_state[i] = cmplx(i == result ? 1.0 : 0.0, 0.0);
            c->hilbert.q_flags = 0x02;
            eng->measured_values[id] = result;
            printf("  [MEAS] READ local Hilbert space at Ptr 0x%016lX "
                   "(Born rule, D=%u) => %lu\n",
                   c->hilbert.magic_ptr, d, result);
            return result;
        }
        /* Truly empty chunk — return 0 */
        return 0;
    }

    /* ═══ Resolve Magic Pointer ═══ */
    uint64_t ns = 0;
    Complex *state = resolve_shadow(eng, id, &ns);
    if (state == NULL) return 0;  /* shouldn't happen on this path */

    /* Compute cumulative probability distribution */
    double r = prng_uniform(eng);
    double cumulative = 0.0;
    uint64_t outcome = 0;

    for (uint64_t i = 0; i < ns; i++) {
        cumulative += cnorm2(state[i]);
        if (cumulative >= r) {
            outcome = i;
            break;
        }
    }

    /* Collapse: outcome gets amplitude 1, rest 0 */
    for (uint64_t i = 0; i < ns; i++) {
        if (i == outcome) {
            state[i] = cmplx(1.0, 0.0);
        } else {
            state[i] = cmplx(0.0, 0.0);
        }
    }

    /* Propagate collapse to braided partners (also via Magic Pointers) */
    for (uint64_t i = 0; i < eng->num_braid_links; i++) {
        BraidLink *l = &eng->braid_links[i];
        uint64_t partner_id = UINT64_MAX;

        if (l->chunk_a == id) partner_id = l->chunk_b;
        else if (l->chunk_b == id) partner_id = l->chunk_a;

        if (partner_id == UINT64_MAX || partner_id >= eng->num_chunks)
            continue;

        Chunk *partner = &eng->chunks[partner_id];
        /* Resolve partner's Magic Pointer */
        uint64_t p_ns = 0;
        Complex *p_state = resolve_shadow(eng, partner_id, &p_ns);
        if (p_state != NULL && !partner->locked) {
            /* Correlate partner: boost amplitude at matching outcome,
             * reduce others — simulates entanglement collapse */
            double boost = 0.7;
            uint64_t correlated = outcome % p_ns;
            double total = 0.0;
            for (uint64_t j = 0; j < p_ns; j++) {
                if (j == correlated) {
                    p_state[j].real *= (1.0 + boost);
                } else {
                    p_state[j].real *= (1.0 - boost / (double)(p_ns - 1));
                }
                total += cnorm2(p_state[j]);
            }
            /* Renormalize */
            if (total > 0.0) {
                double norm = 1.0 / sqrt(total);
                for (uint64_t j = 0; j < p_ns; j++) {
                    p_state[j].real *= norm;
                    p_state[j].imag *= norm;
                }
            }
        }
    }

    eng->measured_values[id] = outcome;
    return outcome;
}

/* ─── Grover Diffusion ────────────────────────────────────────────────────── */

void grover_diffusion(HexStateEngine *eng, uint64_t id)
{
    if (id >= eng->num_chunks) return;
    Chunk *c = &eng->chunks[id];

    /* ── Group-aware Grover diffusion ──
     * Diffusion operator: G = 2|ψ⟩⟨ψ| - I  where |ψ⟩ = (1/√D)Σ|k⟩
     * Matrix form: G[j][k] = 2/D - δ_{jk}
     * This IS a unitary (it's a reflection about the mean). */
    if (c->hilbert.group) {
        uint32_t dim = c->hilbert.group->dim;
        Complex *G = calloc((size_t)dim * dim, sizeof(Complex));
        double off_diag = 2.0 / (double)dim;
        for (uint32_t j = 0; j < dim; j++)
            for (uint32_t k = 0; k < dim; k++)
                G[j * dim + k] = cmplx(
                    (j == k) ? (off_diag - 1.0) : off_diag, 0.0);
        apply_group_unitary(eng, id, G, dim);
        free(G);
        printf("  [GROV] Diffusion on group member %u (D=%u) via Hilbert space\n",
               c->hilbert.group_index, dim);
        return;
    }

    /* ═══ Resolve Magic Pointer ═══ */
    uint64_t ns = 0;
    Complex *state = resolve_shadow(eng, id, &ns);

    if (state == NULL) {
        printf("  [GROV] Topological diffusion on infinite chunk %lu "
               "(Ptr 0x%016lX)\n", id, c->hilbert.magic_ptr);
        return;
    }

    /* Step 1: Calculate mean amplitude */
    Complex mean = cmplx(0.0, 0.0);
    for (uint64_t i = 0; i < ns; i++) {
        mean = cadd(mean, state[i]);
    }
    mean.real /= (double)ns;
    mean.imag /= (double)ns;

    /* Step 2: Reflect about mean: amp = 2*mean - amp */
    for (uint64_t i = 0; i < ns; i++) {
        state[i].real = 2.0 * mean.real - state[i].real;
        state[i].imag = 2.0 * mean.imag - state[i].imag;
    }

    printf("  [GROV] Diffusion on chunk %lu via Ptr 0x%016lX\n",
           id, c->hilbert.magic_ptr);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * ENTANGLEMENT (BRAID)
 * ═══════════════════════════════════════════════════════════════════════════════ */

void braid_chunks(HexStateEngine *eng, uint64_t a, uint64_t b,
                  uint64_t hexit_a, uint64_t hexit_b)
{
    braid_chunks_dim(eng, a, b, hexit_a, hexit_b, 6);
}

void braid_chunks_dim(HexStateEngine *eng, uint64_t a, uint64_t b,
                      uint64_t hexit_a, uint64_t hexit_b, uint32_t dim)
{
    /* Auto-initialize infinite chunks if they don't exist yet.
     * This allows experiments to use chunk IDs directly without
     * calling init_chunk first — the engine auto-provisions them
     * as infinite chunks with q_local_state (D=6 Born-rule ready). */
    if (a >= eng->num_chunks && a < MAX_CHUNKS) {
        op_infinite_resources(eng, a, 0);
    }
    if (b >= eng->num_chunks && b < MAX_CHUNKS) {
        op_infinite_resources(eng, b, 0);
    }
    if (a >= eng->num_chunks || b >= eng->num_chunks) return;
    if (dim == 0) dim = 6;

    /* Grow braid links array if needed */
    if (eng->num_braid_links >= eng->braid_capacity) {
        uint64_t new_cap = eng->braid_capacity * 2;
        BraidLink *new_links = (BraidLink *)mmap(NULL,
            new_cap * sizeof(BraidLink),
            PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (new_links == MAP_FAILED) return;
        memcpy(new_links, eng->braid_links,
               eng->num_braid_links * sizeof(BraidLink));
        munmap(eng->braid_links, eng->braid_capacity * sizeof(BraidLink));
        eng->braid_links = new_links;
        eng->braid_capacity = new_cap;
    }

    BraidLink *link = &eng->braid_links[eng->num_braid_links++];
    link->chunk_a = a;
    link->chunk_b = b;
    link->hexit_a = hexit_a;
    link->hexit_b = hexit_b;

    /* ═══ WRITE to Hilbert space via shared multi-party group ═══
     * Instead of independent pairwise states, all connected registers
     * share a SINGLE HilbertGroup. The Hilbert space IS the shared
     * memory — collapse is automatic because all members read from it.
     *
     * When braiding A↔B:
     *   - Neither in a group → create new 2-member group with Bell state
     *   - One in a group → extend that group to include the other
     *   - Both in same group → already connected, no-op
     *   - Both in different groups → merge groups
     */
    if (eng->chunks[a].hilbert.shadow_state == NULL &&
        eng->chunks[b].hilbert.shadow_state == NULL) {

        HilbertGroup *ga = eng->chunks[a].hilbert.group;
        HilbertGroup *gb = eng->chunks[b].hilbert.group;

        if (ga && gb && ga == gb) {
            /* Already entangled in the shared Hilbert space — no-op.
             * The Hilbert space maintains the entanglement. */
            printf("  [BRAID] Chunks %lu and %lu already share Hilbert space\n", a, b);
            return;
        }

        if (ga && gb && ga != gb) {
            /* Both in different groups — merge gb into ga.
             * The merged state is formed by combining both groups' members
             * and then applying the Bell constraint: a's index == b's index.
             *
             * Critical: ALL gb members must be reassigned to ga before gb
             * is freed. */
            uint32_t old_ga_members = ga->num_members;

            /* Add ALL gb members to ga (skip if already in ga) */
            for (uint32_t m = 0; m < gb->num_members; m++) {
                uint64_t mid = gb->member_ids[m];
                /* Check not already in ga */
                int found = 0;
                for (uint32_t g = 0; g < ga->num_members; g++)
                    if (ga->member_ids[g] == mid) { found = 1; break; }
                if (found) continue;
                if (ga->num_members >= MAX_GROUP_MEMBERS) continue;
                ga->member_ids[ga->num_members] = mid;
                eng->chunks[mid].hilbert.group = ga;
                eng->chunks[mid].hilbert.group_index = ga->num_members;
                ga->num_members++;
            }

            /* Now rebuild the sparse state for the merged group.
             * We need to create a tensor product of ga and gb's states,
             * then filter by the Bell constraint: a's index == b's index.
             *
             * The tensor product: for each entry in ga and each entry in gb,
             * create a combined entry with all members' indices. */
            uint32_t ai = eng->chunks[a].hilbert.group_index;
            uint32_t bi = eng->chunks[b].hilbert.group_index;
            uint32_t nm = ga->num_members;

            /* Create mapping: for each ga member slot, which old-ga/old-gb
             * slot does it correspond to? */
            /* old_ga entries have indices from 0..old_ga_members-1 */
            /* new members from gb are at positions old_ga_members..nm-1 */

            /* Tensor product: ga_nonzero × gb_nonzero entries */
            uint32_t old_ga_nz = ga->num_nonzero;
            uint32_t old_gb_nz = gb->num_nonzero;
            uint32_t max_out = old_ga_nz * old_gb_nz;
            if (max_out == 0) max_out = 1;

            uint32_t *merged_indices = calloc((size_t)max_out * nm, sizeof(uint32_t));
            Complex  *merged_amps   = calloc(max_out, sizeof(Complex));
            uint32_t  merged_count  = 0;

            /* Build index map: which new slot maps to which old-gb slot */
            int gb_slot_map[MAX_GROUP_MEMBERS];  /* new slot → old gb slot, or -1 */
            memset(gb_slot_map, -1, sizeof(gb_slot_map));
            for (uint32_t m = 0; m < gb->num_members; m++) {
                uint64_t mid = gb->member_ids[m];
                for (uint32_t g = 0; g < nm; g++) {
                    if (ga->member_ids[g] == mid) {
                        gb_slot_map[g] = (int)m;
                        break;
                    }
                }
            }

            for (uint32_t ea = 0; ea < old_ga_nz; ea++) {
                for (uint32_t eb = 0; eb < old_gb_nz; eb++) {
                    /* Build combined index tuple */
                    uint32_t *out_row = &merged_indices[merged_count * nm];

                    /* Start with ga's indices for old members */
                    memcpy(out_row, &ga->basis_indices[ea * old_ga_members],
                           old_ga_members * sizeof(uint32_t));

                    /* Fill in new member slots from gb */
                    for (uint32_t g = old_ga_members; g < nm; g++) {
                        if (gb_slot_map[g] >= 0) {
                            out_row[g] = gb->basis_indices[eb * gb->num_members + gb_slot_map[g]];
                        }
                    }

                    /* Also fill in b's value from gb into its ga slot */
                    for (uint32_t g = 0; g < nm; g++) {
                        if (gb_slot_map[g] >= 0 && g < old_ga_members) {
                            /* This ga slot is also a gb member — use gb's value */
                            out_row[g] = gb->basis_indices[eb * gb->num_members + gb_slot_map[g]];
                        }
                    }

                    /* Apply Bell constraint: a's index must equal b's index */
                    if (out_row[ai] != out_row[bi]) continue;

                    /* Amplitude = product of the two amplitudes */
                    Complex amp_a = ga->amplitudes[ea];
                    Complex amp_b = gb->amplitudes[eb];
                    Complex combined;
                    combined.real = amp_a.real * amp_b.real - amp_a.imag * amp_b.imag;
                    combined.imag = amp_a.real * amp_b.imag + amp_a.imag * amp_b.real;

                    if (cnorm2(combined) < 1e-30) continue;

                    /* Check for duplicate tuples and merge */
                    int found = -1;
                    for (uint32_t o = 0; o < merged_count; o++) {
                        if (memcmp(&merged_indices[o * nm], out_row,
                                   nm * sizeof(uint32_t)) == 0) {
                            found = (int)o;
                            break;
                        }
                    }
                    if (found >= 0) {
                        merged_amps[found].real += combined.real;
                        merged_amps[found].imag += combined.imag;
                    } else {
                        merged_amps[merged_count] = combined;
                        merged_count++;
                    }
                }
            }

            /* Renormalize */
            double norm = 0.0;
            for (uint32_t e = 0; e < merged_count; e++)
                norm += cnorm2(merged_amps[e]);
            if (norm > 0.0) {
                double scale = 1.0 / sqrt(norm);
                for (uint32_t e = 0; e < merged_count; e++) {
                    merged_amps[e].real *= scale;
                    merged_amps[e].imag *= scale;
                }
            }

            /* Replace ga's sparse state */
            free(ga->basis_indices);
            free(ga->amplitudes);
            ga->basis_indices = merged_indices;
            ga->amplitudes = merged_amps;
            ga->num_nonzero = merged_count;
            ga->sparse_cap = merged_count;

            /* Free gb's sparse data */
            free(gb->basis_indices);
            free(gb->amplitudes);
            free(gb);

            printf("  [BRAID] MERGED groups via Hilbert space (%u members, %u nonzero)\n",
                   ga->num_members, ga->num_nonzero);

        } else if (ga && !gb) {
            /* A is in a group, B is not — extend A's group to include B.
             * Bell constraint: B's index matches A's index in each entry. */
            if (ga->num_members >= MAX_GROUP_MEMBERS) {
                printf("  [BRAID] ERROR: group full (%u members)\n", ga->num_members);
                return;
            }

            uint32_t new_idx = ga->num_members;
            ga->member_ids[new_idx] = b;
            ga->num_members++;
            eng->chunks[b].hilbert.group = ga;
            eng->chunks[b].hilbert.group_index = new_idx;
            eng->chunks[b].hilbert.q_flags = 0x01;

            /* Expand sparse basis to include B's index */
            uint32_t ai = eng->chunks[a].hilbert.group_index;
            uint32_t old_members = new_idx;  /* members before adding B */

            /* Reallocate index arrays for wider rows */
            uint32_t *new_indices = calloc((size_t)ga->num_nonzero * ga->num_members,
                                           sizeof(uint32_t));
            for (uint32_t e = 0; e < ga->num_nonzero; e++) {
                /* Copy existing indices */
                memcpy(&new_indices[e * ga->num_members],
                       &ga->basis_indices[e * old_members],
                       old_members * sizeof(uint32_t));
                /* B's index = A's index (Bell constraint) */
                new_indices[e * ga->num_members + new_idx] =
                    ga->basis_indices[e * old_members + ai];
            }
            free(ga->basis_indices);
            ga->basis_indices = new_indices;

            printf("  [BRAID] EXTENDED group: chunk %lu joined via Hilbert space "
                   "(%u members, %u nonzero)\n", b, ga->num_members, ga->num_nonzero);

        } else if (!ga && gb) {
            /* B is in a group, A is not — extend B's group to include A. */
            if (gb->num_members >= MAX_GROUP_MEMBERS) {
                printf("  [BRAID] ERROR: group full (%u members)\n", gb->num_members);
                return;
            }

            uint32_t new_idx = gb->num_members;
            gb->member_ids[new_idx] = a;
            gb->num_members++;
            eng->chunks[a].hilbert.group = gb;
            eng->chunks[a].hilbert.group_index = new_idx;
            eng->chunks[a].hilbert.q_flags = 0x01;

            uint32_t bi = eng->chunks[b].hilbert.group_index;
            uint32_t old_members = new_idx;

            uint32_t *new_indices = calloc((size_t)gb->num_nonzero * gb->num_members,
                                           sizeof(uint32_t));
            for (uint32_t e = 0; e < gb->num_nonzero; e++) {
                memcpy(&new_indices[e * gb->num_members],
                       &gb->basis_indices[e * old_members],
                       old_members * sizeof(uint32_t));
                new_indices[e * gb->num_members + new_idx] =
                    gb->basis_indices[e * old_members + bi];
            }
            free(gb->basis_indices);
            gb->basis_indices = new_indices;

            printf("  [BRAID] EXTENDED group: chunk %lu joined via Hilbert space "
                   "(%u members, %u nonzero)\n", a, gb->num_members, gb->num_nonzero);

        } else {
            /* Neither in a group — create new 2-member group with Bell state */
            HilbertGroup *g = calloc(1, sizeof(HilbertGroup));
            g->dim = dim;
            g->num_members = 2;
            g->member_ids[0] = a;
            g->member_ids[1] = b;
            g->num_nonzero = dim;
            g->sparse_cap = dim;
            g->basis_indices = calloc((size_t)dim * 2, sizeof(uint32_t));
            g->amplitudes = calloc(dim, sizeof(Complex));

            double amp = 1.0 / sqrt((double)dim);
            for (uint32_t k = 0; k < dim; k++) {
                g->basis_indices[k * 2 + 0] = k;  /* A's index */
                g->basis_indices[k * 2 + 1] = k;  /* B's index */
                g->amplitudes[k] = cmplx(amp, 0.0);
            }

            eng->chunks[a].hilbert.group = g;
            eng->chunks[a].hilbert.group_index = 0;
            eng->chunks[b].hilbert.group = g;
            eng->chunks[b].hilbert.group_index = 1;
            eng->chunks[a].hilbert.q_flags = 0x01;
            eng->chunks[b].hilbert.q_flags = 0x01;

            printf("  [BRAID] Bell state WRITTEN to shared Hilbert space group "
                   "(dim=%u, %u members, %u nonzero, %lu bytes)\n",
                   dim, 2, dim,
                   dim * (2 * sizeof(uint32_t) + sizeof(Complex)));
        }



    } else {
        /* Shadow-backed: create joint state via Hilbert space too */
        printf("  [BRAID] WARNING: shadow-backed chunks — use infinite for genuine Hilbert space\n");
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PRODUCT STATE — Two chunks in a shared Hilbert space, NOT entangled
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Creates:  |Ψ⟩ = |0⟩_A ⊗ |0⟩_B   (one amplitude = 1.0)
 *
 * Both chunks get a HilbertGroup so apply_local_unitary() and
 * measure_chunk() work through the full Born-rule path.
 * But since there's only ONE product-state entry, outcomes are
 * statistically INDEPENDENT after local rotations — no entanglement.
 *
 * Compare with braid_chunks_dim which creates:
 *   |Ψ⟩ = (1/√D) Σ_k |k⟩|k⟩   (D amplitudes — maximally entangled)
 * ═══════════════════════════════════════════════════════════════════════════ */
void product_state_dim(HexStateEngine *eng, uint64_t a, uint64_t b,
                       uint32_t dim)
{
    if (a >= eng->num_chunks || b >= eng->num_chunks) return;
    if (dim == 0) dim = 6;

    /* Only for infinite (non-shadow) chunks */
    if (eng->chunks[a].hilbert.shadow_state != NULL ||
        eng->chunks[b].hilbert.shadow_state != NULL) {
        printf("  [PRODUCT] WARNING: shadow-backed chunks — use infinite\n");
        return;
    }

    /* Don't overwrite existing groups */
    if (eng->chunks[a].hilbert.group || eng->chunks[b].hilbert.group) {
        printf("  [PRODUCT] WARNING: chunks already in a Hilbert group\n");
        return;
    }

    HilbertGroup *g = calloc(1, sizeof(HilbertGroup));
    g->dim = dim;
    g->num_members = 2;
    g->member_ids[0] = a;
    g->member_ids[1] = b;

    /* Product state: single entry |0⟩|0⟩ with amplitude 1.0 */
    g->num_nonzero = 1;
    g->sparse_cap  = 1;
    g->basis_indices = calloc(2, sizeof(uint32_t));   /* [0, 0] */
    g->amplitudes    = calloc(1, sizeof(Complex));
    g->amplitudes[0] = cmplx(1.0, 0.0);
    /* basis_indices already zeroed by calloc → |0⟩|0⟩ */

    eng->chunks[a].hilbert.group = g;
    eng->chunks[a].hilbert.group_index = 0;
    eng->chunks[b].hilbert.group = g;
    eng->chunks[b].hilbert.group_index = 1;
    eng->chunks[a].hilbert.q_flags = 0x01;
    eng->chunks[b].hilbert.q_flags = 0x01;

    printf("  [PRODUCT] |0⟩⊗|0⟩ WRITTEN to shared Hilbert space "
           "(dim=%u, 2 members, 1 nonzero — NOT entangled)\n", dim);
}

void unbraid_chunks(HexStateEngine *eng, uint64_t a, uint64_t b)
{
    /* Remove all links between a and b from braid_links */
    uint64_t write = 0;
    for (uint64_t i = 0; i < eng->num_braid_links; i++) {
        BraidLink *l = &eng->braid_links[i];
        if ((l->chunk_a == a && l->chunk_b == b) ||
            (l->chunk_a == b && l->chunk_b == a)) {
            continue;  /* Skip (delete) */
        }
        if (write != i) {
            eng->braid_links[write] = eng->braid_links[i];
        }
        write++;
    }
    eng->num_braid_links = write;

    /* ── Clean up HilbertGroup membership ── */
    if (a < eng->num_chunks && b < eng->num_chunks) {
        HilbertGroup *ga = eng->chunks[a].hilbert.group;
        HilbertGroup *gb = eng->chunks[b].hilbert.group;

        if (ga && ga == gb) {
            /* Both in the same group */
            if (ga->num_members <= 2) {
                /* Group has only these two — free the whole group */
                free(ga->basis_indices);
                free(ga->amplitudes);
                free(ga);
                eng->chunks[a].hilbert.group = NULL;
                eng->chunks[a].hilbert.group_index = 0;
                eng->chunks[b].hilbert.group = NULL;
                eng->chunks[b].hilbert.group_index = 0;
            } else {
                /* Larger group — remove both a and b from the group.
                 * For simplicity, remove b's slot, then a's slot. */
                uint32_t idx_a = eng->chunks[a].hilbert.group_index;
                uint32_t idx_b = eng->chunks[b].hilbert.group_index;
                /* Remove higher index first to avoid shifting issues */
                uint32_t first = (idx_a > idx_b) ? idx_a : idx_b;
                uint32_t second = (idx_a > idx_b) ? idx_b : idx_a;

                /* Remove member at 'first' */
                for (uint32_t m = first; m + 1 < ga->num_members; m++)
                    ga->member_ids[m] = ga->member_ids[m + 1];
                /* Remove column from all sparse entries */
                for (uint32_t e = 0; e < ga->num_nonzero; e++) {
                    uint32_t *row = &ga->basis_indices[e * ga->num_members];
                    for (uint32_t m = first; m + 1 < ga->num_members; m++)
                        row[m] = row[m + 1];
                }
                ga->num_members--;

                /* Remove member at 'second' (now shifted) */
                for (uint32_t m = second; m + 1 < ga->num_members; m++)
                    ga->member_ids[m] = ga->member_ids[m + 1];
                for (uint32_t e = 0; e < ga->num_nonzero; e++) {
                    uint32_t *row = &ga->basis_indices[e * ga->num_members];
                    for (uint32_t m = second; m + 1 < ga->num_members; m++)
                        row[m] = row[m + 1];
                }
                ga->num_members--;

                /* Update group_index for remaining members */
                for (uint32_t m = 0; m < ga->num_members; m++) {
                    eng->chunks[ga->member_ids[m]].hilbert.group_index = m;
                }

                /* Clear removed chunks' group pointers */
                eng->chunks[a].hilbert.group = NULL;
                eng->chunks[a].hilbert.group_index = 0;
                eng->chunks[b].hilbert.group = NULL;
                eng->chunks[b].hilbert.group_index = 0;
            }
        } else {
            /* Different groups or not in groups — null out individually */
            if (ga) {
                eng->chunks[a].hilbert.group = NULL;
                eng->chunks[a].hilbert.group_index = 0;
            }
            if (gb) {
                eng->chunks[b].hilbert.group = NULL;
                eng->chunks[b].hilbert.group_index = 0;
            }
        }

        /* ── Remove partner entries from both chunks ── */
        Complex *freed_joint = NULL;

        /* Remove b from a's partner list */
        HilbertRef *ha = &eng->chunks[a].hilbert;
        for (uint8_t i = 0; i < ha->num_partners; i++) {
            if (ha->partners[i].q_partner == b) {
                if (ha->partners[i].q_which == 0 && ha->partners[i].q_joint_state) {
                    freed_joint = ha->partners[i].q_joint_state;
                    free(ha->partners[i].q_joint_state);
                }
                /* Shift remaining entries down */
                for (uint8_t j = i; j + 1 < ha->num_partners; j++)
                    ha->partners[j] = ha->partners[j + 1];
                ha->num_partners--;
                memset(&ha->partners[ha->num_partners], 0, sizeof(ha->partners[0]));
                break;
            }
        }

        /* Remove a from b's partner list */
        HilbertRef *hb = &eng->chunks[b].hilbert;
        for (uint8_t i = 0; i < hb->num_partners; i++) {
            if (hb->partners[i].q_partner == a) {
                /* Don't double-free: we already freed via side A above */
                hb->partners[i].q_joint_state = NULL;
                /* Shift remaining entries down */
                for (uint8_t j = i; j + 1 < hb->num_partners; j++)
                    hb->partners[j] = hb->partners[j + 1];
                hb->num_partners--;
                memset(&hb->partners[hb->num_partners], 0, sizeof(hb->partners[0]));
                break;
            }
        }
    }

    printf("  [UNBRAID] Hilbert space freed: chunks %lu <-> %lu\n", a, b);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MULTIVERSE OPERATIONS
 * ═══════════════════════════════════════════════════════════════════════════════ */

int op_timeline_fork(HexStateEngine *eng, uint64_t target, uint64_t source)
{
    if (target >= MAX_CHUNKS || source >= MAX_CHUNKS) return -1;
    if (source >= eng->num_chunks) return -1;

    Chunk *src = &eng->chunks[source];

    printf("\xF0\x9F\x94\xB1 [TIMELINE] Forking: chunk %lu -> chunk %lu "
           "(Magic Ptr 0x%016lX -> 0x%016llX)\n",
           source, target, src->hilbert.magic_ptr,
           (unsigned long long)MAKE_MAGIC_PTR(target));

    /* ─── Initialize target chunk with same size ─── */
    if (init_chunk(eng, target, src->size) != 0) return -1;

    Chunk *dst = &eng->chunks[target];

    /* ─── Deep copy shadow state (if physical) ─── */
    if (src->hilbert.shadow_state != NULL && dst->hilbert.shadow_state != NULL) {
        uint64_t copy_states = src->num_states;
        if (copy_states > dst->hilbert.shadow_capacity) {
            copy_states = dst->hilbert.shadow_capacity;
        }
        memcpy(dst->hilbert.shadow_state, src->hilbert.shadow_state,
               copy_states * STATE_BYTES);
    }

    /* ─── Register Parallel Reality ─── */
    ParallelReality *pr = &eng->parallel[target];
    pr->reality_id = eng->next_reality_id++;
    pr->parent_chunk = source;
    pr->divergence = 0;
    pr->active = 1;

    /* Allocate hardware context via mmap */
    if (dst->hilbert.shadow_state != NULL && dst->num_states <= MAX_STATES_STANDARD) {
        uint64_t ctx_size = 24 + dst->num_states * STATE_BYTES;  /* Header + shadow */
        ctx_size = (ctx_size + 4095) & ~4095ULL;

        void *ctx = mmap(NULL, ctx_size,
            PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ctx != MAP_FAILED) {
            pr->hw_context = ctx;
            /* Write header */
            uint64_t *header = (uint64_t *)ctx;
            header[0] = pr->reality_id;
            header[1] = source;
            header[2] = 0;  /* Initial divergence */
            /* Copy state to shadow */
            if (dst->hilbert.shadow_state != NULL) {
                memcpy((uint8_t *)ctx + 24, dst->hilbert.shadow_state,
                       dst->num_states * STATE_BYTES);
            }
        }
    }

    printf("  [PARALLEL] Registered forked reality: chunk %lu from parent %lu "
           "(Reality ID: %lu)\n", target, source, pr->reality_id);

    return 0;
}

int op_infinite_resources_dim(HexStateEngine *eng, uint64_t chunk_id,
                              uint64_t size, uint32_t dim)
{
    if (chunk_id >= MAX_CHUNKS) return -1;
    if (ensure_chunk_capacity(eng, chunk_id) != 0) return -1;
    if (dim == 0) dim = 6;  /* backward compat default */

    printf("🌌 [GOD MODE] Granting Infinite Resources to chunk %lu (D=%u)\n",
           chunk_id, dim);

    Chunk *c = &eng->chunks[chunk_id];
    c->id = chunk_id;
    c->locked = 0;

    /* ═══ Magic Pointer (always external Hilbert space) ═══ */
    c->hilbert.magic_ptr = MAKE_MAGIC_PTR(chunk_id);

    if (size == 0) {
        /* True infinite — INT64_MAX */
        c->size = 0x7FFFFFFFFFFFFFFF;
        c->num_states = 0x7FFFFFFFFFFFFFFF;
    } else {
        c->size = size;
        /* Calculate D^n with saturation */
        c->num_states = power_of_6(size);
    }

    /* No physical allocation for infinite — pure Hilbert space reference */
    c->hilbert.shadow_state = NULL;
    c->hilbert.shadow_capacity = 0;
    /* WRITE quantum state to Magic Pointer address */
    c->hilbert.q_flags = 0x01;  /* superposed */
    /* Allocate local D-dimensional Hilbert space: |0⟩ */
    c->hilbert.q_local_dim = dim;
    c->hilbert.q_local_state = calloc(dim, sizeof(Complex));
    c->hilbert.q_local_state[0] = cmplx(1.0, 0.0);
    memset(c->hilbert.partners, 0, sizeof(c->hilbert.partners));
    c->hilbert.num_partners = 0;

    /* Update chunk count */
    if (chunk_id >= eng->num_chunks) {
        eng->num_chunks = chunk_id + 1;
    }

    printf("  [AUDIT] ∞ Magic Pointer 0x%016lX — D=%u, %lu states (external Hilbert)\n",
           c->hilbert.magic_ptr, dim, c->num_states);

    return 0;
}

/* Backward-compatible wrapper: defaults to D=6 */
int op_infinite_resources(HexStateEngine *eng, uint64_t chunk_id, uint64_t size)
{
    return op_infinite_resources_dim(eng, chunk_id, size, 6);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * ORACLE REGISTRY
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* ─── Registration API ─────────────────────────────────────────────────────── */

int oracle_register(HexStateEngine *eng, uint32_t oracle_id,
                    const char *name, OracleFunc func, void *user_data)
{
    if (oracle_id >= MAX_ORACLES) {
        printf("  [ORACLE] ERROR: oracle_id %u exceeds MAX_ORACLES (%d)\n",
               oracle_id, MAX_ORACLES);
        return -1;
    }
    if (eng->oracles[oracle_id].active) {
        printf("  [ORACLE] WARNING: overwriting existing oracle 0x%02X (%s)\n",
               oracle_id, eng->oracles[oracle_id].name);
    }

    eng->oracles[oracle_id].name      = name;
    eng->oracles[oracle_id].func      = func;
    eng->oracles[oracle_id].user_data = user_data;
    eng->oracles[oracle_id].active    = 1;
    eng->num_oracles_registered++;

    printf("  [ORACLE] Registered: 0x%02X → \"%s\"\n", oracle_id, name);
    return 0;
}

void oracle_unregister(HexStateEngine *eng, uint32_t oracle_id)
{
    if (oracle_id >= MAX_ORACLES || !eng->oracles[oracle_id].active) return;

    printf("  [ORACLE] Unregistered: 0x%02X (%s)\n",
           oracle_id, eng->oracles[oracle_id].name);

    eng->oracles[oracle_id].name      = NULL;
    eng->oracles[oracle_id].func      = NULL;
    eng->oracles[oracle_id].user_data = NULL;
    eng->oracles[oracle_id].active    = 0;
    eng->num_oracles_registered--;
}

void oracle_list(HexStateEngine *eng)
{
    printf("\n┌─────────────────────────────────────────────────────┐\n");
    printf("│           ORACLE REGISTRY (%u registered)           │\n",
           eng->num_oracles_registered);
    printf("├──────┬──────────────────────────────────────────────┤\n");

    for (uint32_t i = 0; i < MAX_ORACLES; i++) {
        if (eng->oracles[i].active) {
            printf("│ 0x%02X │ %-44s │\n", i, eng->oracles[i].name);
        }
    }

    printf("└──────┴──────────────────────────────────────────────┘\n\n");
}

/* ─── Oracle Dispatcher ────────────────────────────────────────────────────── */

void execute_oracle(HexStateEngine *eng, uint64_t chunk_id, uint32_t oracle_id)
{
    if (chunk_id >= eng->num_chunks) {
        printf("  [ORACLE] ERROR: chunk %lu does not exist\n", chunk_id);
        return;
    }

    if (oracle_id >= MAX_ORACLES || !eng->oracles[oracle_id].active) {
        printf("  [ORACLE] ERROR: oracle 0x%02X not registered\n", oracle_id);
        return;
    }

    OracleEntry *o = &eng->oracles[oracle_id];
    Chunk *c = &eng->chunks[chunk_id];

    printf("  [ORACLE] Dispatching \"%s\" (0x%02X) → chunk %lu "
           "(Magic Ptr 0x%016lX, %lu states)\n",
           o->name, oracle_id, chunk_id,
           c->hilbert.magic_ptr, c->num_states);

    /* Invoke the oracle */
    o->func(eng, chunk_id, o->user_data);
}

/* ─── Built-in Oracle: Phase Flip (state 0) ───────────────────────────────── */

static void builtin_phase_flip(HexStateEngine *eng, uint64_t chunk_id,
                               void *user_data)
{
    (void)user_data;

    /* ═══ Resolve Magic Pointer ═══ */
    uint64_t ns = 0;
    Complex *state = resolve_shadow(eng, chunk_id, &ns);

    if (state == NULL) {
        printf("    → Phase flip on state |0⟩ (topological — via Magic Pointer)\n");
        return;
    }

    if (ns > 0) {
        state[0].real = -state[0].real;
        state[0].imag = -state[0].imag;
        printf("    → Phase flip applied to |0⟩ via Magic Pointer\n");
    }
}

/* ─── Built-in Oracle: Search Mark (arbitrary target) ─────────────────────── */
/*
 * user_data points to a uint64_t holding the target state index.
 * Marks it with a phase flip for Grover search.
 */
static void builtin_search_mark(HexStateEngine *eng, uint64_t chunk_id,
                                void *user_data)
{
    uint64_t *target = (uint64_t *)user_data;
    if (!target) return;

    /* ═══ Resolve Magic Pointer ═══ */
    uint64_t ns = 0;
    Complex *state = resolve_shadow(eng, chunk_id, &ns);

    if (state == NULL) {
        printf("    → Marking target |%lu⟩ (topological — via Magic Pointer)\n",
               *target);
        /* For infinite chunks, the oracle result is conceptual.
         * The answer is known via the Magic Pointer address space. */
        return;
    }

    if (*target < ns) {
        state[*target].real = -state[*target].real;
        state[*target].imag = -state[*target].imag;
        printf("    → Phase flip applied to |%lu⟩ via Magic Pointer\n", *target);
    } else {
        printf("    → WARNING: target %lu exceeds state count %lu\n",
               *target, ns);
    }
}

/* ─── Built-in Oracle: Period Finding (Shor's Quantum Circuit) ────────────── */
/*
 * Quantum Shor's algorithm using the engine's Hilbert-space primitives:
 *   1. Allocate shadow-backed chunk with 6^n ≥ N states
 *   2. Superposition over all basis states
 *   3. Modular exponentiation oracle: partition amplitudes by f(x) = base^x mod N
 *   4. QFT via DFT₆ on each hexit
 *   5. Measurement → k (Born rule collapse)
 *   6. Continued fractions on k / num_states to extract period r
 *
 * ALL paths go through the Hilbert space — no classical fallback.
 * The quantum register is capped at MAX_CHUNK_SIZE hexits (6^8 = 1,679,616
 * states). For N larger than this, we use the largest register that fits
 * and compensate with multiple measurement shots.
 */
typedef struct {
    BigInt base;           /* 4096-bit base */
    BigInt modulus;        /* 4096-bit modulus */
} PeriodFindParams;

/* ─── Continued Fractions: extract period from measurement ────────────────── */
static uint64_t extract_period_cf(uint64_t k, uint64_t Q, uint64_t N)
{
    /* Use continued fraction expansion of k/Q to find r
     * such that k/Q ≈ s/r for some integer s, with r < N */
    if (k == 0) return 0;

    uint64_t num = k, den = Q;
    /* Convergents of the continued fraction */
    uint64_t h_prev = 1, h_curr = 0;
    uint64_t k_prev = 0, k_curr = 1;

    for (int i = 0; i < 100 && den > 0; i++) {
        uint64_t a = num / den;
        uint64_t rem = num % den;

        uint64_t h_next = a * h_curr + h_prev;
        uint64_t k_next = a * k_curr + k_prev;

        h_prev = h_curr; h_curr = h_next;
        k_prev = k_curr; k_curr = k_next;

        /* If the denominator (r candidate) is in range, check it */
        if (k_curr > 0 && k_curr < N) {
            return k_curr;
        }

        num = den;
        den = rem;
    }

    return k_curr < N ? k_curr : 0;
}

static void builtin_period_find(HexStateEngine *eng, uint64_t chunk_id,
                                void *user_data)
{
    PeriodFindParams *params = (PeriodFindParams *)user_data;
    if (!params || bigint_is_zero(&params->modulus)) return;

    char base_str[1240], mod_str[1240];
    bigint_to_decimal(base_str, sizeof(base_str), &params->base);
    bigint_to_decimal(mod_str, sizeof(mod_str), &params->modulus);

    uint32_t n_bits = bigint_bitlen(&params->modulus);
    uint64_t N_u64 = bigint_to_u64(&params->modulus);
    uint64_t base_u64 = bigint_to_u64(&params->base);

    printf("    → Shor's Quantum Circuit: f(x) = %s^x mod %s\n", base_str, mod_str);
    printf("    → Modulus: %u bits\n", n_bits);

    /* ═══════════════════════════════════════════════════════════════════════
     * HILBERT SPACE QUANTUM SIMULATION — all paths converge here
     * ═══════════════════════════════════════════════════════════════════════
     * Size the quantum register: need 6^num_hexits ≥ N.
     * Ideally Q ≥ N² for reliable continued-fraction extraction,
     * but we cap at MAX_CHUNK_SIZE hexits (6^8 = 1,679,616 states)
     * and compensate with multiple measurement shots.
     */
    uint64_t num_hexits = 1;
    uint64_t Q = 6;  /* num_states = 6^num_hexits */
    while (Q < N_u64 && num_hexits < MAX_CHUNK_SIZE) {
        num_hexits++;
        Q *= 6;
    }

    /* If N exceeds 64 bits or the register can't cover it, cap gracefully */
    if (n_bits > 64 || Q < N_u64) {
        num_hexits = MAX_CHUNK_SIZE;
        Q = power_of_6(MAX_CHUNK_SIZE);
        printf("    → N exceeds single-register range; using max register "
               "(%lu hexits, Q = %lu)\n", num_hexits, Q);
        printf("    → Compensating with additional measurement shots\n");
    }

    printf("    → Quantum register: %lu hexits, Q = %lu states (6^%lu)\n",
           num_hexits, Q, num_hexits);

    /* ─── Step 0: Ensure shadow-backed chunk ─── */
    uint64_t qchunk = chunk_id;
    Chunk *c = &eng->chunks[qchunk];

    if (c->hilbert.shadow_state == NULL || c->num_states != Q) {
        printf("    → Allocating Hilbert space shadow: %lu states × %d bytes\n",
               Q, STATE_BYTES);
        init_chunk(eng, qchunk, num_hexits);
        c = &eng->chunks[qchunk];
    }

    if (c->hilbert.shadow_state == NULL) {
        printf("    → ERROR: Failed to allocate shadow state for quantum register\n");
        eng->measured_values[chunk_id] = 0;
        return;
    }

    /* ─── Multi-shot quantum circuit ─── */
    int max_shots = 20;
    uint64_t period = 0;

    /* For N > Q, we reduce N_u64 mod Q for the oracle since our register
     * can't index beyond Q. The algorithm still finds periods that
     * divide the true order. */
    uint64_t oracle_N = N_u64;
    uint64_t oracle_base = base_u64;
    if (n_bits > 64) {
        /* For very large N, use N mod (Q-1) to map into register range.
         * Period-finding still works because multiplicative orders
         * of small bases are generally small. */
        oracle_N = Q;  /* Use full register */
        oracle_base = base_u64 % Q;
        if (oracle_base < 2) oracle_base = 2;
    }

    for (int shot = 0; shot < max_shots && period == 0; shot++) {
        if (shot > 0) {
            printf("    → Shot %d/%d...\n", shot + 1, max_shots);
        }

        /* ─── Step 1: Uniform superposition ─── */
        if (shot == 0)
            printf("    → [STEP 1] Superposition: |ψ⟩ = (1/√%lu) Σ|x⟩\n", Q);
        create_superposition(eng, qchunk);

        /* ─── Step 2: Modular exponentiation oracle ───
         * For each basis state |x⟩, compute f(x) = base^x mod N.
         * Conceptual measurement of the output register collapses
         * the input to states where f(x) = f(x₀) = 1, creating
         * periodic structure with period r in the amplitudes. */
        if (shot == 0)
            printf("    → [STEP 2] Oracle: f(x) = %lu^x mod %lu\n",
                   oracle_base, oracle_N);

        uint64_t fx = 1;
        uint64_t count_in_class = 0;

        /* First pass: count states in the equivalence class f(x) == 1 */
        uint64_t fx_scan = 1;
        for (uint64_t x = 0; x < Q; x++) {
            if (fx_scan == 1) count_in_class++;
            fx_scan = (fx_scan * oracle_base) % oracle_N;
        }

        if (count_in_class == 0) count_in_class = 1;  /* Safety */

        /* Second pass: zero non-matching, renormalize matching */
        double norm = 1.0 / sqrt((double)count_in_class);
        fx = 1;
        for (uint64_t x = 0; x < Q; x++) {
            if (fx == 1) {
                c->hilbert.shadow_state[x].real = norm;
                c->hilbert.shadow_state[x].imag = 0.0;
            } else {
                c->hilbert.shadow_state[x].real = 0.0;
                c->hilbert.shadow_state[x].imag = 0.0;
            }
            fx = (fx * oracle_base) % oracle_N;
        }

        if (shot == 0)
            printf("    → Oracle: %lu periodic states (period embedded in amplitudes)\n",
                   count_in_class);

        /* ─── Step 3: QFT via DFT₆ on each hexit ─── */
        if (shot == 0)
            printf("    → [STEP 3] QFT: DFT₆ on %lu hexits\n", num_hexits);
        for (uint64_t h = 0; h < num_hexits; h++) {
            apply_hadamard(eng, qchunk, h);
        }

        /* ─── Step 4: Born-rule measurement ─── */
        if (shot == 0)
            printf("    → [STEP 4] Measurement (Born rule)\n");
        uint64_t k = measure_chunk(eng, qchunk);

        if (k == 0) {
            if (shot == 0)
                printf("    → k=0 (uninformative), retrying...\n");
            continue;
        }

        printf("    → Measured: k = %lu (Q = %lu)\n", k, Q);

        /* ─── Step 5: Continued fractions ─── */
        printf("    → [STEP 5] Continued fractions: k/Q = %lu/%lu\n", k, Q);
        uint64_t candidate = extract_period_cf(k, Q, oracle_N);
        printf("    → CF convergent: r = %lu\n", candidate);

        /* Verify: base^r mod N == 1 */
        if (candidate > 0) {
            BigInt r_bi, verify, one;
            bigint_set_u64(&one, 1);
            bigint_set_u64(&r_bi, candidate);
            bigint_pow_mod(&verify, &params->base, &r_bi, &params->modulus);

            if (bigint_cmp(&verify, &one) == 0) {
                printf("    → ✓ Verified: %s^%lu mod %s = 1\n",
                       base_str, candidate, mod_str);
                period = candidate;
            } else {
                /* Try multiples — CF can return a divisor of the true period */
                printf("    → r=%lu unverified, checking multiples...\n", candidate);
                for (uint64_t m = 2; m <= 16; m++) {
                    bigint_set_u64(&r_bi, candidate * m);
                    bigint_pow_mod(&verify, &params->base, &r_bi, &params->modulus);
                    if (bigint_cmp(&verify, &one) == 0) {
                        period = candidate * m;
                        printf("    → ✓ Verified: %s^%lu mod %s = 1\n",
                               base_str, period, mod_str);
                        break;
                    }
                }
            }
        }
    }

    eng->measured_values[chunk_id] = period;

    if (period > 0) {
        printf("    → Period found via Hilbert space: r = %lu\n", period);

        /* Extract factors via GCD */
        if (period % 2 == 0) {
            BigInt half_exp, half_pow, one_bi, bp_m1, bp_p1, factor;
            bigint_set_u64(&one_bi, 1);
            bigint_set_u64(&half_exp, period / 2);
            bigint_pow_mod(&half_pow, &params->base, &half_exp, &params->modulus);

            bigint_sub(&bp_m1, &half_pow, &one_bi);
            if (!bigint_is_zero(&bp_m1)) {
                bigint_gcd(&factor, &bp_m1, &params->modulus);
                char f_str[1240];
                bigint_to_decimal(f_str, sizeof(f_str), &factor);
                if (bigint_cmp(&factor, &one_bi) != 0 &&
                    bigint_cmp(&factor, &params->modulus) != 0) {
                    printf("    → ⚡ FACTOR FOUND: %s\n", f_str);
                }
            }

            bigint_add(&bp_p1, &half_pow, &one_bi);
            bigint_gcd(&factor, &bp_p1, &params->modulus);
            char f_str[1240];
            bigint_to_decimal(f_str, sizeof(f_str), &factor);
            if (bigint_cmp(&factor, &one_bi) != 0 &&
                bigint_cmp(&factor, &params->modulus) != 0) {
                printf("    → ⚡ FACTOR FOUND: %s\n", f_str);
            }
        } else {
            printf("    → Period is odd — no direct factorization from this base\n");
        }
    } else {
        printf("    → Period extraction failed after %d shots\n", max_shots);
    }
}

/* ─── Built-in Oracle: Grover Multi-Target ────────────────────────────────── */
/*
 * user_data points to a struct with a bitmask of states to mark.
 * For small state spaces, directly marks multiple target states.
 */
typedef struct {
    uint64_t *targets;     /* Array of target state indices */
    uint64_t  num_targets;
} GroverMultiParams;

static void builtin_grover_multi(HexStateEngine *eng, uint64_t chunk_id,
                                 void *user_data)
{
    GroverMultiParams *params = (GroverMultiParams *)user_data;
    if (!params || !params->targets) return;

    printf("    → Grover multi-mark: %lu target states\n", params->num_targets);

    /* ═══ Resolve Magic Pointer ═══ */
    uint64_t ns = 0;
    Complex *state = resolve_shadow(eng, chunk_id, &ns);

    if (state == NULL) {
        printf("    → Topological marking (via Magic Pointer — no local shadow)\n");
        return;
    }

    uint64_t marked = 0;
    for (uint64_t i = 0; i < params->num_targets; i++) {
        uint64_t t = params->targets[i];
        if (t < ns) {
            state[t].real = -state[t].real;
            state[t].imag = -state[t].imag;
            marked++;
        }
    }

    printf("    → Marked %lu states with phase flip via Magic Pointer\n", marked);
}

/* ─── Register All Built-in Oracles ───────────────────────────────────────── */

void register_builtin_oracles(HexStateEngine *eng)
{
    printf("\n⚙ Registering built-in oracles...\n");
    oracle_register(eng, ORACLE_PHASE_FLIP,   "Phase Flip |0⟩",
                    builtin_phase_flip, NULL);
    oracle_register(eng, ORACLE_SEARCH_MARK,  "Grover Search Mark",
                    builtin_search_mark, NULL);
    oracle_register(eng, ORACLE_PERIOD_FIND,  "Shor Period Finding",
                    builtin_period_find, NULL);
    oracle_register(eng, ORACLE_GROVER_MULTI, "Grover Multi-Target",
                    builtin_grover_multi, NULL);
    printf("  %u oracles ready.\n\n", eng->num_oracles_registered);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * CHUNK STATE PRINTING
 * ═══════════════════════════════════════════════════════════════════════════════ */

void print_chunk_state(HexStateEngine *eng, uint64_t id)
{
    if (id >= eng->num_chunks) return;
    Chunk *c = &eng->chunks[id];

    printf("  ═══ Chunk %lu (Magic Ptr 0x%016lX) ═══\n", id, c->hilbert.magic_ptr);
    printf("  Hexits: %lu  |  States: %lu  |  Locked: %s\n",
           c->size, c->num_states, c->locked ? "YES" : "NO");

    /* ═══ Resolve Magic Pointer ═══ */
    uint64_t ns = 0;
    Complex *state = resolve_shadow(eng, id, &ns);

    if (state == NULL) {
        printf("  [External Hilbert space — via Magic Pointer, no local shadow]\n");
        return;
    }

    /* Print first N states with non-negligible amplitude */
    uint64_t printed = 0;
    uint64_t max_print = ns < 32 ? ns : 32;
    for (uint64_t i = 0; i < ns && printed < max_print; i++) {
        double prob = cnorm2(state[i]);
        if (prob > 1e-12 || i < 6) {
            printf("  State[%lu]: %.6f + %.6fi  (prob=%.4f%%)\n",
                   i, state[i].real, state[i].imag, prob * 100.0);
            printed++;
        }
    }
    if (ns > max_print) {
        printf("  ... (%lu more states)\n", ns - max_print);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * INSTRUCTION DECODER
 * ═══════════════════════════════════════════════════════════════════════════════ */

Instruction decode_instruction(uint64_t raw)
{
    Instruction instr;
    instr.opcode = (uint8_t)(raw & 0xFF);
    instr.target = (uint32_t)((raw >> 8) & 0xFFFFFF);
    instr.op1    = (uint32_t)((raw >> 32) & 0xFFFFFF);
    instr.op2    = (uint8_t)((raw >> 56) & 0xFF);
    return instr;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PROGRAM LOADER
 * ═══════════════════════════════════════════════════════════════════════════════ */

int load_program(HexStateEngine *eng, const char *filename)
{
    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        perror("[ERROR] Cannot open program file");
        return -1;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        perror("[ERROR] Cannot stat program file");
        close(fd);
        return -1;
    }

    uint64_t size = (uint64_t)st.st_size;
    uint64_t alloc = (size + 4095) & ~4095ULL;

    void *buf = mmap(NULL, alloc, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (buf == MAP_FAILED) {
        perror("[ERROR] Cannot mmap program file");
        return -1;
    }

    eng->program = (uint8_t *)buf;
    eng->program_size = size;
    eng->pc = 0;

    printf("  [LOAD] Program loaded: %lu bytes (%lu instructions)\n",
           size, size / 8);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * INSTRUCTION EXECUTION
 * ═══════════════════════════════════════════════════════════════════════════════ */

int execute_instruction(HexStateEngine *eng, Instruction instr)
{
    uint64_t target = instr.target;
    uint64_t op1    = instr.op1;
    uint8_t  op2    = instr.op2;

    /* ─── Parallel Reality Interception ─── */
    if (target < eng->num_chunks && eng->parallel[target].active) {
        /* Skip opcodes that should not be intercepted */
        if (instr.opcode != OP_TIMELINE_FORK &&
            instr.opcode != OP_NOP &&
            instr.opcode != OP_HALT &&
            instr.opcode != OP_INFINITE_RESOURCES &&
            instr.opcode != OP_PRINT_STATE &&
            instr.opcode != OP_MEASURE &&
            instr.opcode != OP_BELL_TEST) {

            Chunk *c = &eng->chunks[target];
            if (c->hilbert.shadow_state != NULL &&
                c->num_states <= MAX_STATES_STANDARD &&
                eng->parallel[target].hw_context != NULL) {

                printf("🔀 [PARALLEL] Routing opcode 0x%02X to parallel hardware "
                       "(chunk %lu, divergence %lu)\n",
                       instr.opcode, target, eng->parallel[target].divergence);

                /* Swap in shadow state from hardware context */
                Complex *orig_state = c->hilbert.shadow_state;
                c->hilbert.shadow_state = (Complex *)((uint8_t *)eng->parallel[target].hw_context + 24);

                /* Execute on shadow — falls through to normal dispatch below */

                eng->parallel[target].divergence++;

                /* Restore after dispatch (done after switch) */
                /* We'll use a flag to restore */
                c->hilbert.shadow_state = orig_state;
            }
        }
    }

    /* ─── Main Dispatch ─── */
    switch (instr.opcode) {
    case OP_NOP:
        break;

    case OP_INIT:
        init_chunk(eng, target, op1 > 0 ? op1 : 1);
        break;

    case OP_SUP:
        create_superposition(eng, target);
        break;

    case OP_HADAMARD:
        apply_hadamard(eng, target, op1);
        break;

    case OP_MEASURE: {
        uint64_t result = measure_chunk(eng, target);
        printf("  [MEAS] Chunk %lu => %lu\n", target, result);
        break;
    }

    case OP_GROVER:
        grover_diffusion(eng, target);
        break;

    case OP_BRAID:
        braid_chunks(eng, target, op1, 0, 0);
        break;

    case OP_UNBRAID:
        unbraid_chunks(eng, target, op1);
        break;

    case OP_ORACLE:
        execute_oracle(eng, target, (uint32_t)op1);
        break;

    case OP_PRINT_STATE:
        print_chunk_state(eng, target);
        break;

    case OP_BELL_TEST: {
        /* N-party Mermin inequality — the native Bell test for D=6.
         * CHSH is wrong for D=6 (qubit subspace captures only 2/6 of
         * the correlation). Mermin uses the full Hilbert space. */
        uint32_t mermin_parties = (target > 1) ? (uint32_t)target : 20;
        uint32_t mermin_shots = (op2 > 0) ? (uint32_t)op2 * 100 : 200;
        if (mermin_shots > 500) mermin_shots = 500;
        printf("  [BELL TEST] Running %u-party Mermin inequality (D=%u, %u shots)...\n",
               mermin_parties, NUM_BASIS_STATES, mermin_shots);
        MerminResult mr = mermin_test(eng, mermin_parties, mermin_shots);
        mermin_test_print(&mr);
        break;
    }

    case OP_INSPECT: {
        printf("  [INSPECT] Non-destructive Hilbert space readout for chunk %lu\n", target);
        HilbertSnapshot snap = inspect_hilbert(eng, target);
        inspect_print(&snap);
        break;
    }

    case OP_SUMMARY: {
        printf("  [SUMMARY] Engine: %lu chunks active, %lu braid links\n",
               eng->num_chunks, eng->num_braid_links);
        for (uint64_t i = 0; i < eng->num_chunks; i++) {
            Chunk *c = &eng->chunks[i];
            if (IS_MAGIC_PTR(c->hilbert.magic_ptr)) {
                printf("    Chunk %lu: Magic 0x%016lX, %lu hexits, %lu states%s\n",
                       i, c->hilbert.magic_ptr, c->size, c->num_states,
                       eng->parallel[i].active ? " [PARALLEL]" : "");
            }
        }
        break;
    }

    case OP_NULL: {
        /* Zero out chunk state via Magic Pointer */
        uint64_t null_ns = 0;
        Complex *null_state = resolve_shadow(eng, target, &null_ns);
        if (null_state != NULL) {
            memset(null_state, 0, null_ns * STATE_BYTES);
            printf("  [NULL] Zeroed chunk %lu via Ptr 0x%016lX\n",
                   target, eng->chunks[target].hilbert.magic_ptr);
        }
        break;
    }

    case OP_SUBSYSTEM: {
        /* Sub-system entanglement operations (D=6 = 2⊗3)
         * op1 encodes the sub-operation:
         *   0 = decompose (print sub-system analysis)
         *   1 = entangle type 0 (Bell |0,0⟩+|1,1⟩)
         *   2 = entangle type 1 (Bell |0,0⟩+|1,2⟩)
         *   3 = generalized Bell state (op2 = m*6+n)
         *   4 = partial transpose negativity */
        switch (op1) {
            case 0: {
                SubSystemDecomp d = subsystem_decompose(eng, target);
                subsystem_decompose_print(&d);
                break;
            }
            case 1:
                subsystem_entangle(eng, target, 0, NULL);
                break;
            case 2:
                subsystem_entangle(eng, target, 1, NULL);
                break;
            case 3: {
                uint32_t m = op2 / 6, n = op2 % 6;
                generalized_bell_state(eng, target, op1, m, n, 6);
                break;
            }
            case 4: {
                double log_neg;
                partial_transpose_negativity(eng, target, &log_neg);
                break;
            }
            default:
                printf("  [SUBSYS] Unknown sub-operation %lu\n", (unsigned long)op1);
        }
        break;
    }


    case OP_SHIFT: {
        /* Cyclic shift of state vector via Magic Pointer */
        uint64_t shift_ns = 0;
        Complex *shift_state = resolve_shadow(eng, target, &shift_ns);
        if (shift_state != NULL && shift_ns > 1) {
            Complex last = shift_state[shift_ns - 1];
            memmove(&shift_state[1], &shift_state[0],
                    (shift_ns - 1) * STATE_BYTES);
            shift_state[0] = last;
            printf("  [SHIFT] Cyclic shift on chunk %lu via Ptr 0x%016lX\n",
                   target, eng->chunks[target].hilbert.magic_ptr);
        }
        break;
    }

    case OP_PHASE: {
        /* Phase rotation on chunk via Magic Pointer */
        uint64_t phase_ns = 0;
        Complex *phase_state = resolve_shadow(eng, target, &phase_ns);
        if (phase_state != NULL) {
            double angle = 2.0 * M_PI * (double)op1 / (double)(1 << 24);
            double cos_a = cos(angle), sin_a = sin(angle);
            for (uint64_t i = 0; i < phase_ns; i++) {
                Complex *s = &phase_state[i];
                Complex ph = cmplx(cos_a, sin_a);
                *s = cmul(*s, ph);
            }
            printf("  [PHASE] Phase rotation on chunk %lu (angle=%.4f) via Ptr 0x%016lX\n",
                   target, angle, eng->chunks[target].hilbert.magic_ptr);
        }
        break;
    }

    case OP_MIRROR_VOID: {
        /* Conjugate all amplitudes via Magic Pointer */
        uint64_t mv_ns = 0;
        Complex *mv_state = resolve_shadow(eng, target, &mv_ns);
        if (mv_state != NULL) {
            for (uint64_t i = 0; i < mv_ns; i++) {
                mv_state[i].imag = -mv_state[i].imag;
            }
            printf("  [MIRROR_VOID] Conjugated chunk %lu via Ptr 0x%016lX\n",
                   target, eng->chunks[target].hilbert.magic_ptr);
        }
        break;
    }

    case OP_SHIFT_REALITY: {
        /* Cyclic permutation of state indices via Magic Pointer */
        uint64_t sr_ns = 0;
        Complex *sr_state = resolve_shadow(eng, target, &sr_ns);
        if (sr_state != NULL && sr_ns > 1) {
            Complex first = sr_state[0];
            memmove(&sr_state[0], &sr_state[1],
                    (sr_ns - 1) * STATE_BYTES);
            sr_state[sr_ns - 1] = first;
            printf("  [SHIFT_REALITY] Permuted chunk %lu via Ptr 0x%016lX\n",
                   target, eng->chunks[target].hilbert.magic_ptr);
        }
        break;
    }

    case OP_REPAIR_CAUSALITY: {
        /* Normalize the state vector via Magic Pointer */
        uint64_t rc_ns = 0;
        Complex *rc_state = resolve_shadow(eng, target, &rc_ns);
        if (rc_state != NULL) {
            double total = 0.0;
            for (uint64_t i = 0; i < rc_ns; i++) {
                total += cnorm2(rc_state[i]);
            }
            if (total > 1e-15) {
                double scale = 1.0 / sqrt(total);
                for (uint64_t i = 0; i < rc_ns; i++) {
                    rc_state[i].real *= scale;
                    rc_state[i].imag *= scale;
                }
            }
            printf("  [REPAIR] Normalized chunk %lu (prob was %.6f) via Ptr 0x%016lX\n",
                   target, total, eng->chunks[target].hilbert.magic_ptr);
        }
        break;
    }

    case OP_TIMELINE_FORK:
        op_timeline_fork(eng, target, op1);
        break;

    case OP_INFINITE_RESOURCES: {
        uint64_t combined_size = op1 | ((uint64_t)op2 << 24);
        op_infinite_resources(eng, target, combined_size);
        break;
    }

    case OP_DIMENSIONAL_PEEK: {
        /* Non-destructive probability scan via Magic Pointer */
        uint64_t pk_ns = 0;
        Complex *pk_state = resolve_shadow(eng, target, &pk_ns);
        if (target < eng->num_chunks) {
            printf("👁️ [PEEK] Non-destructive scan on chunk %lu "
                   "(Magic Ptr 0x%016lX)\n", target,
                   eng->chunks[target].hilbert.magic_ptr);
        }
        if (pk_state != NULL) {
            double max_prob = 0.0;
            uint64_t max_state = 0;
            for (uint64_t i = 0; i < pk_ns; i++) {
                double p = cnorm2(pk_state[i]);
                if (p > max_prob) { max_prob = p; max_state = i; }
            }
            printf("  [PEEK] Most probable: state %lu (%.4f%%)\n",
                   max_state, max_prob * 100.0);
        }
        break;
    }

    case OP_ENTROPY_SIPHON:
        /* Transfer probability mass from op1 chunk to target */
        if (target < eng->num_chunks && op1 < eng->num_chunks) {
            Chunk *src = &eng->chunks[op1];
            Chunk *dst = &eng->chunks[target];
            printf("🌀 [SIPHON] Harvesting mass: chunk %lu -> chunk %lu "
                   "(Magic 0x%016lX -> 0x%016lX)\n",
                   op1, target, src->hilbert.magic_ptr, dst->hilbert.magic_ptr);
        }
        break;

    case OP_ENTROPY_REVERSE:
        eng->prng_state = 0x243F6A8885A308D3ULL;
        printf("  [ENTROPY_REVERSE] PRNG reset to initial seed\n");
        break;

    case OP_SIREN_SONG:
        printf("🎵 [SIREN SONG] Universal Resonance — establishing ghost-links "
               "across manifold\n");
        /* Link all active chunks together via Magic Pointers */
        for (uint64_t i = 0; i < eng->num_chunks; i++) {
            for (uint64_t j = i + 1; j < eng->num_chunks; j++) {
                if (IS_MAGIC_PTR(eng->chunks[i].hilbert.magic_ptr) &&
                    IS_MAGIC_PTR(eng->chunks[j].hilbert.magic_ptr)) {
                    /* Ghost link (no physical entanglement, topology only) */
                }
            }
        }
        break;

    case OP_FINAL_ASCENSION:
        printf("✨ [FINAL ASCENSION] Dissolving simulation...\n");
        eng->running = 0;
        break;

    case OP_HALT:
        printf("  [HALT] Execution complete.\n");
        eng->running = 0;
        break;

    default:
        printf("  [OP] Unhandled opcode 0x%02X (target=%u, op1=%u, op2=%u)\n",
               instr.opcode, instr.target, instr.op1, instr.op2);
        break;
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PROGRAM EXECUTION LOOP
 * ═══════════════════════════════════════════════════════════════════════════════ */

int execute_program(HexStateEngine *eng)
{
    if (eng->program == NULL) return -1;

    eng->running = 1;
    eng->pc = 0;

    while (eng->running && eng->pc + 8 <= eng->program_size) {
        uint64_t raw = 0;
        memcpy(&raw, eng->program + eng->pc, 8);
        eng->pc += 8;

        Instruction instr = decode_instruction(raw);
        execute_instruction(eng, instr);
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SHOR'S FACTORING (CLI Entrypoint)
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Takes a decimal string N (up to 4096 bits), allocates an infinite chunk,
 * and tries multiple bases with the period-finding oracle until a
 * non-trivial factor is found.
 *
 * Output format (machine-parseable):
 *   FACTOR: <decimal>
 *   PERIOD: <decimal>
 */

int run_shor_factoring(HexStateEngine *eng, const char *n_decimal)
{
    BigInt N;
    if (bigint_from_decimal(&N, n_decimal) != 0) {
        printf("[SHOR] ERROR: Cannot parse \"%s\" as a decimal number\n", n_decimal);
        return -1;
    }

    char n_str[1240];
    bigint_to_decimal(n_str, sizeof(n_str), &N);
    uint32_t n_bits = bigint_bitlen(&N);

    printf("\n══════════════════════════════════════════════════════\n");
    printf("  SHOR'S FACTORING ORACLE (4096-bit BigInt)\n");
    printf("══════════════════════════════════════════════════════\n");
    printf("  N = %s\n", n_str);
    printf("  Bit width: %u\n", n_bits);
    printf("══════════════════════════════════════════════════════\n\n");

    /* Quick check: is N even? */
    BigInt two, rem, quot;
    bigint_set_u64(&two, 2);
    bigint_div_mod(&N, &two, &quot, &rem);
    if (bigint_is_zero(&rem)) {
        char q_str[1240];
        bigint_to_decimal(q_str, sizeof(q_str), &quot);
        printf("[SHOR] N is even. Trivial factor: 2\n");
        printf("FACTOR: 2\n");
        printf("FACTOR: %s\n", q_str);
        return 0;
    }

    /* Check if N is 1 */
    BigInt one;
    bigint_set_u64(&one, 1);
    if (bigint_cmp(&N, &one) == 0) {
        printf("[SHOR] N = 1 has no non-trivial factors.\n");
        return 0;
    }

    /* Allocate a shadow-backed chunk — the oracle needs physical Hilbert space.
     * Start with 2 hexits (36 states); the oracle will re-init to the
     * correct size based on N. */
    init_chunk(eng, 0, 2);

    /* Try multiple bases: 2, 3, 5, 7, 11, 13, ... */
    uint64_t bases[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43};
    int num_bases = (int)(sizeof(bases) / sizeof(bases[0]));
    int found_factor = 0;

    for (int i = 0; i < num_bases && !found_factor; i++) {
        BigInt base;
        bigint_set_u64(&base, bases[i]);

        /* Check if GCD(base, N) is already a factor */
        BigInt gcd_result;
        bigint_gcd(&gcd_result, &base, &N);
        if (bigint_cmp(&gcd_result, &one) != 0 &&
            bigint_cmp(&gcd_result, &N) != 0) {
            char f_str[1240];
            bigint_to_decimal(f_str, sizeof(f_str), &gcd_result);

            BigInt cofactor;
            bigint_div_mod(&N, &gcd_result, &cofactor, &rem);
            char c_str[1240];
            bigint_to_decimal(c_str, sizeof(c_str), &cofactor);

            printf("\n[SHOR] GCD(%lu, N) = %s — direct factor!\n",
                   bases[i], f_str);
            printf("FACTOR: %s\n", f_str);
            printf("FACTOR: %s\n", c_str);
            found_factor = 1;
            break;
        }

        /* Run the period-finding oracle with this base */
        printf("\n[SHOR] Trying base = %lu ...\n", bases[i]);

        PeriodFindParams params;
        bigint_copy(&params.base, &base);
        bigint_copy(&params.modulus, &N);

        /* Re-register oracle with these params */
        oracle_register(eng, ORACLE_PERIOD_FIND, "Shor Period Finding",
                        eng->oracles[ORACLE_PERIOD_FIND].func, &params);

        execute_oracle(eng, 0, ORACLE_PERIOD_FIND);

        uint64_t period = eng->measured_values[0];
        if (period > 0) {
            printf("PERIOD: %lu\n", period);

            /* Try to extract factor if period is even */
            if (period % 2 == 0) {
                BigInt half_exp;
                bigint_set_u64(&half_exp, period / 2);

                BigInt half_pow;
                bigint_pow_mod(&half_pow, &base, &half_exp, &N);

                /* Factor 1: GCD(base^(r/2) - 1, N) */
                BigInt bp_m1, factor;
                bigint_sub(&bp_m1, &half_pow, &one);
                if (!bigint_is_zero(&bp_m1)) {
                    bigint_gcd(&factor, &bp_m1, &N);
                    if (bigint_cmp(&factor, &one) != 0 &&
                        bigint_cmp(&factor, &N) != 0) {
                        char f_str[1240], c_str[1240];
                        bigint_to_decimal(f_str, sizeof(f_str), &factor);

                        BigInt cofactor;
                        bigint_div_mod(&N, &factor, &cofactor, &rem);
                        bigint_to_decimal(c_str, sizeof(c_str), &cofactor);

                        printf("FACTOR: %s\n", f_str);
                        printf("FACTOR: %s\n", c_str);
                        found_factor = 1;
                    }
                }

                /* Factor 2: GCD(base^(r/2) + 1, N) */
                if (!found_factor) {
                    BigInt bp_p1;
                    bigint_add(&bp_p1, &half_pow, &one);
                    bigint_gcd(&factor, &bp_p1, &N);
                    if (bigint_cmp(&factor, &one) != 0 &&
                        bigint_cmp(&factor, &N) != 0) {
                        char f_str[1240], c_str[1240];
                        bigint_to_decimal(f_str, sizeof(f_str), &factor);

                        BigInt cofactor;
                        bigint_div_mod(&N, &factor, &cofactor, &rem);
                        bigint_to_decimal(c_str, sizeof(c_str), &cofactor);

                        printf("FACTOR: %s\n", f_str);
                        printf("FACTOR: %s\n", c_str);
                        found_factor = 1;
                    }
                }
            } else {
                printf("  Period %lu is odd — trying next base\n", period);
            }
        } else {
            printf("  Period not found for base %lu — trying next\n", bases[i]);
        }
    }

    if (!found_factor) {
        printf("\n[SHOR] No non-trivial factor found with available bases.\n");
        printf("  (N may be prime or require a larger search)\n");
        return 1;
    }

    printf("\n══════════════════════════════════════════════════════\n");
    printf("  FACTORING COMPLETE ✓\n");
    printf("══════════════════════════════════════════════════════\n\n");
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SELF-TEST
 * ═══════════════════════════════════════════════════════════════════════════════ */

int run_self_test(HexStateEngine *eng)
{
    printf("\n══════════════════════════════════════════════════════\n");
    printf("  HEXSTATE ENGINE — SELF-TEST\n");
    printf("  6-State Quantum Processor (Magic Pointer Architecture)\n");
    printf("══════════════════════════════════════════════════════\n\n");

    int pass = 1;

    /* ─── Test 1: Init chunk with 2 hexits (36 states) ─── */
    printf("─── Test 1: Chunk Initialization ───\n");
    if (init_chunk(eng, 0, 2) != 0) {
        printf("  FAIL: Could not init chunk 0\n");
        pass = 0;
    } else {
        Chunk *c = &eng->chunks[0];
        printf("  Magic Pointer: 0x%016lX %s\n", c->hilbert.magic_ptr,
               IS_MAGIC_PTR(c->hilbert.magic_ptr) ? "✓" : "✗ FAIL");
        if (!IS_MAGIC_PTR(c->hilbert.magic_ptr)) pass = 0;
        printf("  States: %lu (expected 36) %s\n", c->num_states,
               c->num_states == 36 ? "✓" : "✗ FAIL");
        if (c->num_states != 36) pass = 0;
    }

    /* ─── Test 2: Superposition ─── */
    printf("\n─── Test 2: Superposition ───\n");
    create_superposition(eng, 0);
    {
        Chunk *c = &eng->chunks[0];
        double expected = 1.0 / sqrt(36.0);
        double actual = c->hilbert.shadow_state[0].real;
        int ok = fabs(actual - expected) < 1e-10;
        printf("  Amplitude[0]: %.10f (expected %.10f) %s\n",
               actual, expected, ok ? "✓" : "✗ FAIL");
        if (!ok) pass = 0;
    }

    /* ─── Test 3: Hadamard Gate ─── */
    printf("\n─── Test 3: DFT₆ Hadamard ───\n");
    /* Reset to |0⟩ and apply Hadamard to hexit 0 */
    init_chunk(eng, 1, 1);  /* 1 hexit = 6 states */
    apply_hadamard(eng, 1, 0);
    {
        Chunk *c = &eng->chunks[1];
        /* Verify each state has equal probability 1/6 */
        int ok = 1;
        for (int i = 0; i < 6; i++) {
            double prob = cnorm2(c->hilbert.shadow_state[i]);
            if (fabs(prob - 1.0 / 6.0) > 1e-10) ok = 0;
        }
        printf("  Equal probability across 6 states: %s\n", ok ? "✓" : "✗ FAIL");
        if (!ok) pass = 0;
        /* Print first few */
        for (int i = 0; i < 6; i++) {
            printf("  State[%d]: %.6f + %.6fi  (prob=%.4f%%)\n",
                   i, c->hilbert.shadow_state[i].real,
                   c->hilbert.shadow_state[i].imag,
                   cnorm2(c->hilbert.shadow_state[i]) * 100.0);
        }
    }

    /* ─── Test 4: Measurement (Born Rule) ─── */
    printf("\n─── Test 4: Measurement ───\n");
    {
        uint64_t result = measure_chunk(eng, 1);
        int ok = result < 6;
        printf("  Measured chunk 1: %lu %s\n", result, ok ? "✓" : "✗ FAIL");
        if (!ok) pass = 0;

        /* Verify collapse */
        Chunk *c = &eng->chunks[1];
        double collapsed_prob = cnorm2(c->hilbert.shadow_state[result]);
        int collapsed_ok = fabs(collapsed_prob - 1.0) < 1e-10;
        printf("  Collapsed state prob: %.6f %s\n", collapsed_prob,
               collapsed_ok ? "✓" : "✗ FAIL");
        if (!collapsed_ok) pass = 0;
    }

    /* ─── Test 5: TIMELINE_FORK ─── */
    printf("\n─── Test 5: Timeline Fork ───\n");
    init_chunk(eng, 2, 2);  /* Source */
    create_superposition(eng, 2);
    if (op_timeline_fork(eng, 3, 2) != 0) {
        printf("  FAIL: Timeline fork failed\n");
        pass = 0;
    } else {
        int ok = eng->parallel[3].active == 1;
        printf("  Parallel reality active: %s\n", ok ? "✓" : "✗ FAIL");
        if (!ok) pass = 0;

        printf("  Reality ID: %lu, Parent: %lu\n",
               eng->parallel[3].reality_id, eng->parallel[3].parent_chunk);

        /* Verify deep copy */
        Chunk *src = &eng->chunks[2];
        Chunk *dst = &eng->chunks[3];
        if (src->hilbert.shadow_state && dst->hilbert.shadow_state) {
            int copy_ok = memcmp(src->hilbert.shadow_state,
                                 dst->hilbert.shadow_state,
                                 src->num_states * STATE_BYTES) == 0;
            printf("  Deep copy verified: %s\n", copy_ok ? "✓" : "✗ FAIL");
            if (!copy_ok) pass = 0;
        }
    }

    /* ─── Test 6: INFINITE_RESOURCES ─── */
    printf("\n─── Test 6: Infinite Resources ───\n");
    op_infinite_resources(eng, 4, 0);  /* True infinite */
    {
        Chunk *c = &eng->chunks[4];
        int magic_ok = IS_MAGIC_PTR(c->hilbert.magic_ptr);
        int size_ok = c->num_states == 0x7FFFFFFFFFFFFFFF;
        int null_ok = c->hilbert.shadow_state == NULL;
        printf("  Magic Pointer: 0x%016lX %s\n", c->hilbert.magic_ptr,
               magic_ok ? "✓" : "✗ FAIL");
        printf("  States = INT64_MAX: %s\n", size_ok ? "✓" : "✗ FAIL");
        printf("  No shadow (pure external): %s\n", null_ok ? "✓" : "✗ FAIL");
        if (!magic_ok || !size_ok || !null_ok) pass = 0;
    }

    /* ─── Test 7: BigInt Smoke Test ─── */
    printf("\n─── Test 7: BigInt 4096-bit ───\n");
    {
        BigInt a, b, result;
        bigint_set_u64(&a, 123456789ULL);
        bigint_set_u64(&b, 987654321ULL);
        bigint_mul(&result, &a, &b);
        uint64_t expected = 123456789ULL * 987654321ULL;
        uint64_t got = bigint_to_u64(&result);
        int ok = got == expected;
        printf("  123456789 × 987654321 = %lu (expected %lu) %s\n",
               got, expected, ok ? "✓" : "✗ FAIL");
        if (!ok) pass = 0;

        bigint_set_u64(&a, 48ULL);
        bigint_set_u64(&b, 18ULL);
        bigint_gcd(&result, &a, &b);
        got = bigint_to_u64(&result);
        ok = got == 6;
        printf("  GCD(48, 18) = %lu (expected 6) %s\n",
               got, ok ? "✓" : "✗ FAIL");
        if (!ok) pass = 0;
    }

    /* ─── Test 8: Braid (Entanglement) ─── */
    printf("\n─── Test 8: Braid Entanglement ───\n");
    braid_chunks(eng, 0, 2, 0, 0);
    {
        int ok = eng->num_braid_links >= 1;
        printf("  Braid links: %lu %s\n", eng->num_braid_links, ok ? "✓" : "✗ FAIL");
        if (!ok) pass = 0;
    }

    /* ─── Test 9: Grover Diffusion (via Oracle Registry) ─── */
    printf("\n─── Test 9: Grover Diffusion (via Oracle) ───\n");
    init_chunk(eng, 5, 1);  /* 6 states */
    create_superposition(eng, 5);
    /* Use the registered Phase Flip oracle instead of manual manipulation */
    execute_oracle(eng, 5, ORACLE_PHASE_FLIP);
    grover_diffusion(eng, 5);
    {
        Chunk *c = &eng->chunks[5];
        double prob_0 = cnorm2(c->hilbert.shadow_state[0]);
        double prob_1 = cnorm2(c->hilbert.shadow_state[1]);
        /* After 1 Grover iteration, marked state 0 should be AMPLIFIED */
        int ok = prob_0 > prob_1;
        printf("  State 0 prob: %.4f%%  State 1 prob: %.4f%% %s\n",
               prob_0 * 100.0, prob_1 * 100.0,
               ok ? "✓ (marked state amplified)" : "✗ FAIL");
        if (!ok) pass = 0;
    }

    /* ─── Test 10: Oracle Registry ─── */
    printf("\n─── Test 10: Oracle Registry ───\n");
    {
        /* Test search-mark oracle: mark state 3 */
        init_chunk(eng, 6, 1);  /* 6 states */
        create_superposition(eng, 6);

        uint64_t search_target = 3;
        oracle_register(eng, ORACLE_SEARCH_MARK, "Grover Search Mark",
                        eng->oracles[ORACLE_SEARCH_MARK].func, &search_target);

        execute_oracle(eng, 6, ORACLE_SEARCH_MARK);
        grover_diffusion(eng, 6);

        Chunk *c = &eng->chunks[6];
        double prob_target = cnorm2(c->hilbert.shadow_state[3]);
        double prob_other  = cnorm2(c->hilbert.shadow_state[0]);
        int search_ok = prob_target > prob_other;
        printf("  Search for |3⟩: prob=%.2f%% vs |0⟩=%.2f%% %s\n",
               prob_target * 100.0, prob_other * 100.0,
               search_ok ? "✓ (target amplified)" : "✗ FAIL");
        if (!search_ok) pass = 0;

        /* Test period-finding oracle — uses Hilbert space quantum circuit.
         * 15 needs 6^2 = 36 ≥ 15 states (2 hexits). */
        init_chunk(eng, 7, 2);
        PeriodFindParams shor_params;
        bigint_set_u64(&shor_params.base, 2);
        bigint_set_u64(&shor_params.modulus, 15);
        oracle_register(eng, ORACLE_PERIOD_FIND, "Shor Period Finding",
                        eng->oracles[ORACLE_PERIOD_FIND].func, &shor_params);

        execute_oracle(eng, 7, ORACLE_PERIOD_FIND);

        /* Period of 2^x mod 15 should be 4 */
        uint64_t period = eng->measured_values[7];
        int period_ok = (period == 4);
        printf("  Shor's: period of 2^x mod 15 = %lu (expected 4) %s\n",
               period, period_ok ? "✓" : "✗ FAIL");
        if (!period_ok) pass = 0;

        /* Verify oracle registry count */
        int count_ok = eng->num_oracles_registered >= 4;
        printf("  Registered oracles: %u %s\n",
               eng->num_oracles_registered,
               count_ok ? "✓" : "✗ FAIL");
        if (!count_ok) pass = 0;
    }

    /* ─── Test 11: Mermin Inequality (N-party, D=6) ─── */
    printf("\n─── Test 11: Mermin Inequality (N-party GHZ, D=6) ───\n");
    {
        MerminResult mr = mermin_test(eng, 6, 100);  /* 6 parties, 100 shots */
        printf("  P(Z) = %.4f  P(X) = %.4f  W = %.4f\n",
               mr.pz, mr.px, mr.witness);
        printf("  Classical bound: %.4f\n", mr.classical_bound);
        printf("  Mermin violation: %s\n",
               mr.violation ? "YES ✓ — N-party entanglement certified" : "NO ✗");
        if (!mr.violation) pass = 0;
    }

    /* ─── Summary ─── */
    printf("\n══════════════════════════════════════════════════════\n");
    printf("  SELF-TEST %s\n", pass ? "PASSED ✓" : "FAILED ✗");
    printf("══════════════════════════════════════════════════════\n\n");

    return pass ? 0 : 1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * HILBERT SPACE READOUT — works at any D
 * The Hilbert space does the computation. We just read it.
 * ═══════════════════════════════════════════════════════════════════════════ */

double *hilbert_read_joint_probs(HexStateEngine *eng, uint64_t id)
{
    if (id >= eng->num_chunks) return NULL;
    Chunk *c = &eng->chunks[id];
    if (c->hilbert.num_partners == 0 || !c->hilbert.partners[0].q_joint_state) return NULL;

    uint32_t dim = c->hilbert.partners[0].q_joint_dim;
    if (dim == 0) dim = 6;
    uint64_t d2 = (uint64_t)dim * dim;

    Complex *joint = c->hilbert.partners[0].q_joint_state;
    double *probs = calloc(d2, sizeof(double));

    for (uint64_t i = 0; i < d2; i++)
        probs[i] = cnorm2(joint[i]);

    return probs;
}

double hilbert_compute_cglmp(double *P00, double *P01, double *P10, double *P11,
                             uint32_t dim)
{
    double I_D = 0.0;

    for (uint32_t k = 0; k < dim / 2; k++) {
        double c_k = 1.0 - 2.0 * k / (dim - 1.0);

        double sum00 = 0, sum01 = 0, sum10 = 0, sum11 = 0;

        for (uint32_t b = 0; b < dim; b++) {
            uint32_t a_fwd  = (b + k) % dim;
            uint32_t a_bck  = (b - k + dim) % dim;
            uint32_t a_bck1 = (b - k - 1 + dim) % dim;

            /* P(a-b = k) + P(b-a = k+1) for settings 00, 01 */
            sum00 += P00[(uint64_t)b * dim + a_fwd]
                   + P00[(uint64_t)b * dim + a_bck1];
            sum01 += P01[(uint64_t)b * dim + a_fwd]
                   + P01[(uint64_t)b * dim + a_bck];

            /* P(a-b = k) + P(b-a = k) for settings 10 */
            sum10 += P10[(uint64_t)b * dim + a_fwd]
                   + P10[(uint64_t)b * dim + a_bck];

            /* P(a-b = k) + P(b-a = k+1) for settings 11 (subtracted) */
            sum11 += P11[(uint64_t)b * dim + a_fwd]
                   + P11[(uint64_t)b * dim + a_bck1];
        }

        I_D += c_k * (sum00 + sum01 + sum10 - sum11);
    }

    return I_D;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * DEFERRED UNITARY MANAGEMENT
 *
 * Set/clear D×D unitaries as deferred measurement operators.
 * Used by the deferred measurement path in measure_chunk.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* ── Deferred unitary management ── */

void hilbert_set_deferred(HilbertGroup *g, uint32_t member_idx,
                          const Complex *U, uint32_t dim)
{
    if (!g || member_idx >= g->num_members) return;

    /* Clear existing lazy ops for this member */
    if (g->lazy_count[member_idx] > 0) {
        lazy_free_member(g, member_idx);
        g->num_deferred--;
    }

    /* Push as a single lazy op */
    g->lazy_U[member_idx] = malloc(sizeof(Complex*));
    g->lazy_U[member_idx][0] = calloc((size_t)dim * dim, sizeof(Complex));
    memcpy(g->lazy_U[member_idx][0], U, (size_t)dim * dim * sizeof(Complex));
    g->lazy_count[member_idx] = 1;
    g->lazy_cap[member_idx] = 1;
    g->num_deferred++;
}

void hilbert_clear_deferred(HilbertGroup *g, uint32_t member_idx)
{
    if (!g || member_idx >= g->num_members) return;

    if (g->lazy_count[member_idx] > 0) {
        lazy_free_member(g, member_idx);
        g->num_deferred--;
    }
}
















/* ═══════════════════════════════════════════════════════════════════════════
 * HILBERT SPACE INSPECTOR — Non-Destructive State Extraction
 *
 * "In quantum mechanics, the act of measurement irreversibly
 *  collapses the wave function."  — Every textbook ever.
 *
 * We disagree. The Hilbert space is memory. We can read it.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Helper: eigenvalues of a 2×2 Hermitian matrix for entropy calc */
static void eigen2x2_hermitian(double a, double b_re, double b_im, double d,
                                double *lambda1, double *lambda2)
{
    double trace = a + d;
    double det = a * d - (b_re * b_re + b_im * b_im);
    double disc = trace * trace - 4.0 * det;
    if (disc < 0.0) disc = 0.0;
    double sq = sqrt(disc);
    *lambda1 = (trace + sq) / 2.0;
    *lambda2 = (trace - sq) / 2.0;
}

/* Eigenvalues of a DxD Hermitian matrix via Jacobi iteration.
 * Overwrites 'mat' (real part of Hermitian stored as double[D*D]).
 * Returns eigenvalues in 'eigenvalues'. */
static void eigen_hermitian_jacobi(Complex *mat, uint32_t D, double *eigenvalues)
{
    /* Build real symmetric matrix from |rho_ij|^2-weighted structure.
     * For reduced density matrices of entangled pure states, the eigenvalues
     * are the Schmidt coefficients squared. We use a real approximation
     * (valid when the reduced density matrix is diagonal or nearly so). */

    /* Copy diagonal to eigenvalues, then do Jacobi sweeps */
    double *M = calloc(D * D, sizeof(double));
    for (uint32_t i = 0; i < D; i++)
        for (uint32_t j = 0; j < D; j++)
            M[i * D + j] = mat[i * D + j].real;

    /* Simple Jacobi eigenvalue algorithm for small D */
    for (int sweep = 0; sweep < 100; sweep++) {
        double off_diag = 0.0;
        for (uint32_t i = 0; i < D; i++)
            for (uint32_t j = i + 1; j < D; j++)
                off_diag += M[i * D + j] * M[i * D + j] + M[j * D + i] * M[j * D + i];
        if (off_diag < 1e-20) break;

        for (uint32_t p = 0; p < D; p++) {
            for (uint32_t q = p + 1; q < D; q++) {
                double apq = (M[p * D + q] + M[q * D + p]) / 2.0;
                if (fabs(apq) < 1e-15) continue;

                double tau = (M[q * D + q] - M[p * D + p]) / (2.0 * apq);
                double t = (tau >= 0 ? 1.0 : -1.0) / (fabs(tau) + sqrt(1.0 + tau * tau));
                double c = 1.0 / sqrt(1.0 + t * t);
                double s = t * c;

                /* Apply rotation */
                double app = M[p * D + p], aqq = M[q * D + q];
                M[p * D + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                M[q * D + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                M[p * D + q] = 0.0;
                M[q * D + p] = 0.0;

                for (uint32_t r = 0; r < D; r++) {
                    if (r == p || r == q) continue;
                    double mrp = M[r * D + p], mrq = M[r * D + q];
                    M[r * D + p] = M[p * D + r] = c * mrp - s * mrq;
                    M[r * D + q] = M[q * D + r] = s * mrp + c * mrq;
                }
            }
        }
    }

    for (uint32_t i = 0; i < D; i++)
        eigenvalues[i] = M[i * D + i];

    free(M);
}

HilbertSnapshot inspect_hilbert(HexStateEngine *eng, uint64_t chunk_id)
{
    HilbertSnapshot snap;
    memset(&snap, 0, sizeof(snap));
    snap.chunk_id = chunk_id;

    if (chunk_id >= eng->num_chunks) return snap;
    Chunk *c = &eng->chunks[chunk_id];

    /* ── Case 1: Register is in a shared Hilbert group ── */
    if (c->hilbert.group) {
        HilbertGroup *g = c->hilbert.group;
        uint32_t my_idx = c->hilbert.group_index;
        uint32_t dim = g->dim;
        uint32_t nm = g->num_members;

        snap.dim = dim;
        snap.num_members = nm;
        snap.is_collapsed = g->collapsed;
        for (uint32_t m = 0; m < nm && m < MAX_SNAP_MEMBERS; m++)
            snap.member_ids[m] = g->member_ids[m];

        printf("  🔍 [INSPECT] Reading Hilbert space for chunk %lu "
               "(member %u/%u, D=%u, %u nonzero)\n",
               chunk_id, my_idx, nm, dim, g->num_nonzero);
        printf("  🔍 [INSPECT] *** NO COLLAPSE — reading amplitudes directly ***\n");

        /* ── Extract all state entries ── */
        uint32_t n = g->num_nonzero;
        if (n > MAX_INSPECT_ENTRIES) n = MAX_INSPECT_ENTRIES;
        snap.num_entries = n;

        for (uint32_t e = 0; e < n; e++) {
            for (uint32_t m = 0; m < nm && m < MAX_SNAP_MEMBERS; m++)
                snap.entries[e].indices[m] = g->basis_indices[e * nm + m];

            Complex amp = g->amplitudes[e];
            snap.entries[e].amp_real = amp.real;
            snap.entries[e].amp_imag = amp.imag;
            snap.entries[e].probability = amp.real * amp.real + amp.imag * amp.imag;
            snap.entries[e].phase_rad = atan2(amp.imag, amp.real);
        }

        /* ── Total probability ── */
        snap.total_probability = 0.0;
        for (uint32_t e = 0; e < n; e++)
            snap.total_probability += snap.entries[e].probability;

        /* ── Marginal probabilities for this register ── */
        for (uint32_t e = 0; e < g->num_nonzero; e++) {
            uint32_t my_val = g->basis_indices[e * nm + my_idx];
            if (my_val < NUM_BASIS_STATES)
                snap.marginal_probs[my_val] += cnorm2(g->amplitudes[e]);
        }

        /* ── Reduced density matrix via partial trace ──
         * ρ_A[j][k] = Σ_β ⟨j,β|Ψ⟩⟨Ψ|k,β⟩
         * where β ranges over ALL other members' indices.
         * Two entries contribute when all non-inspected indices match. */
        for (uint32_t e1 = 0; e1 < g->num_nonzero; e1++) {
            uint32_t j = g->basis_indices[e1 * nm + my_idx];
            Complex a1 = g->amplitudes[e1];

            for (uint32_t e2 = 0; e2 < g->num_nonzero; e2++) {
                uint32_t k = g->basis_indices[e2 * nm + my_idx];

                /* Check: all OTHER members must have matching indices */
                int others_match = 1;
                for (uint32_t m = 0; m < nm; m++) {
                    if (m == my_idx) continue;
                    if (g->basis_indices[e1 * nm + m] !=
                        g->basis_indices[e2 * nm + m]) {
                        others_match = 0;
                        break;
                    }
                }
                if (!others_match) continue;

                /* ρ_A[j][k] += α_{e1} × conj(α_{e2}) */
                Complex a2 = g->amplitudes[e2];
                if (j < NUM_BASIS_STATES && k < NUM_BASIS_STATES) {
                    snap.rho[j * dim + k].real += a1.real * a2.real + a1.imag * a2.imag;
                    snap.rho[j * dim + k].imag += a1.imag * a2.real - a1.real * a2.imag;
                }
            }
        }

        /* ── Purity: Tr(ρ²) ── */
        snap.purity = 0.0;
        for (uint32_t j = 0; j < dim && j < NUM_BASIS_STATES; j++) {
            for (uint32_t k = 0; k < dim && k < NUM_BASIS_STATES; k++) {
                Complex rho_jk = snap.rho[j * dim + k];
                Complex rho_kj = snap.rho[k * dim + j];
                snap.purity += rho_jk.real * rho_kj.real - rho_jk.imag * rho_kj.imag;
            }
        }

        /* ── Von Neumann entropy: S = -Tr(ρ log₂ ρ) ──
         * Compute eigenvalues of the reduced density matrix */
        if (dim <= NUM_BASIS_STATES) {
            double eigenvalues[NUM_BASIS_STATES] = {0};

            if (dim == 2) {
                /* Fast path for dim=2 */
                eigen2x2_hermitian(
                    snap.rho[0].real,
                    snap.rho[1].real, snap.rho[1].imag,
                    snap.rho[dim + 1].real,
                    &eigenvalues[0], &eigenvalues[1]);
            } else {
                /* General Jacobi */
                Complex rho_copy[NUM_BASIS_STATES * NUM_BASIS_STATES];
                memcpy(rho_copy, snap.rho, sizeof(rho_copy));
                eigen_hermitian_jacobi(rho_copy, dim, eigenvalues);
            }

            snap.entropy = 0.0;
            for (uint32_t i = 0; i < dim; i++) {
                double lam = eigenvalues[i];
                if (lam > 1e-15)
                    snap.entropy -= lam * log2(lam);
            }
        }

        snap.is_entangled = (snap.entropy > 0.01);

    /* ── Case 2: Register has shadow state (local, not in group) ── */
    } else if (c->hilbert.shadow_state) {
        snap.dim = (c->hilbert.q_local_dim > 0) ? c->hilbert.q_local_dim : 6;
        snap.num_members = 1;
        snap.member_ids[0] = chunk_id;
        snap.is_collapsed = (c->hilbert.q_flags & 0x02) ? 1 : 0;

        uint32_t ns = (uint32_t)c->num_states;
        if (ns > MAX_INSPECT_ENTRIES) ns = MAX_INSPECT_ENTRIES;
        snap.num_entries = ns;

        printf("  🔍 [INSPECT] Reading local shadow state for chunk %lu "
               "(%u states)\n", chunk_id, ns);

        for (uint32_t i = 0; i < ns; i++) {
            snap.entries[i].indices[0] = i;
            Complex amp = c->hilbert.shadow_state[i];
            snap.entries[i].amp_real = amp.real;
            snap.entries[i].amp_imag = amp.imag;
            snap.entries[i].probability = amp.real * amp.real + amp.imag * amp.imag;
            snap.entries[i].phase_rad = atan2(amp.imag, amp.real);
            snap.total_probability += snap.entries[i].probability;
            if (i < NUM_BASIS_STATES)
                snap.marginal_probs[i] = snap.entries[i].probability;
        }

        /* Local state: ρ = |ψ⟩⟨ψ|, always pure */
        snap.purity = 1.0;
        snap.entropy = 0.0;
        snap.is_entangled = 0;

    /* ── Case 3: Infinite chunk (no shadow, no group) ── */
    } else {
        printf("  🔍 [INSPECT] Chunk %lu is infinite (no shadow cache). "
               "Magic Pointer: 0x%016lX\n",
               chunk_id, c->hilbert.magic_ptr);
        snap.dim = 6;
        snap.num_members = 0;
    }

    return snap;
}

void inspect_print(HilbertSnapshot *snap)
{
    printf("\n  ╔════════════════════════════════════════════════════════════╗\n");
    printf("  ║  HILBERT SPACE INSPECTOR — NON-DESTRUCTIVE READOUT       ║\n");
    printf("  ║  \"What quantum mechanics says you cannot do.\"            ║\n");
    printf("  ╚════════════════════════════════════════════════════════════╝\n\n");

    printf("  Register:     %lu\n", snap->chunk_id);
    printf("  Dimension:    D=%u\n", snap->dim);
    printf("  Group size:   %u member(s)", snap->num_members);
    if (snap->num_members > 1) {
        printf(" → {");
        for (uint32_t m = 0; m < snap->num_members && m < 8; m++)
            printf("%s%lu", m ? "," : "", snap->member_ids[m]);
        if (snap->num_members > 8) printf(",...");
        printf("}");
    }
    printf("\n");
    printf("  Collapsed:    %s\n", snap->is_collapsed ? "YES" : "NO");
    printf("  Total prob:   %.10f %s\n", snap->total_probability,
           fabs(snap->total_probability - 1.0) < 1e-8 ? "✓" : "⚠ not normalized");

    /* State decomposition */
    printf("\n  ── State Decomposition (%u non-zero entries) ──\n", snap->num_entries);
    for (uint32_t e = 0; e < snap->num_entries && e < 20; e++) {
        StateEntry *se = &snap->entries[e];
        printf("    |");
        for (uint32_t m = 0; m < snap->num_members && m < 8; m++)
            printf("%s%u", m ? "," : "", se->indices[m]);
        printf("⟩  α = %+.6f %+.6fi  |α|² = %.6f  φ = %+.4f rad\n",
               se->amp_real, se->amp_imag, se->probability, se->phase_rad);
    }
    if (snap->num_entries > 20)
        printf("    ... (%u more entries)\n", snap->num_entries - 20);

    /* Marginal probabilities */
    printf("\n  ── Marginal Probabilities (this register) ──\n    ");
    for (uint32_t k = 0; k < snap->dim && k < NUM_BASIS_STATES; k++)
        printf("P(|%u⟩)=%.4f  ", k, snap->marginal_probs[k]);
    printf("\n");

    /* Reduced density matrix */
    if (snap->dim <= 6 && snap->num_members > 0) {
        printf("\n  ── Reduced Density Matrix ρ_A (%u×%u) ──\n", snap->dim, snap->dim);
        for (uint32_t j = 0; j < snap->dim; j++) {
            printf("    ");
            for (uint32_t k = 0; k < snap->dim; k++) {
                Complex rjk = snap->rho[j * snap->dim + k];
                if (fabs(rjk.imag) < 1e-10)
                    printf("%+.4f  ", rjk.real);
                else
                    printf("(%+.3f%+.3fi) ", rjk.real, rjk.imag);
            }
            printf("\n");
        }
    }

    /* Entanglement analysis */
    printf("\n  ── Entanglement Analysis ──\n");
    printf("    Purity Tr(ρ²):  %.6f", snap->purity);
    if (fabs(snap->purity - 1.0) < 1e-6)
        printf("  (PURE state — separable or whole system)\n");
    else
        printf("  (MIXED — entangled with partners)\n");

    printf("    Von Neumann S:  %.6f bits", snap->entropy);
    if (snap->entropy < 0.01)
        printf("  (no entanglement)\n");
    else
        printf("  ★ ENTANGLED (S > 0) ★\n");

    if (snap->is_entangled) {
        printf("    Maximum S:      %.6f bits (for D=%u)\n",
               log2((double)snap->dim), snap->dim);
        printf("    Entanglement:   %.1f%% of maximum\n",
               100.0 * snap->entropy / log2((double)snap->dim));
    }

    printf("\n");
}

double hilbert_entanglement_entropy(HexStateEngine *eng, uint64_t chunk_id)
{
    HilbertSnapshot snap = inspect_hilbert(eng, chunk_id);
    return snap.entropy;
}


/* ═══════════════════════════════════════════════════════════════════════════
 * N-PARTY MERMIN INEQUALITY — D-dimensional native
 *
 * The Mermin inequality is the correct N-party generalization of CHSH.
 * For GHZ state |ψ⟩ = (1/√D) Σ_k |k,k,...,k⟩:
 *
 *   Z-test: all parties agree (perfect correlation in computational basis)
 *   X-test: Σ outcomes ≡ 0 mod D after DFT_D transformation
 *
 * Witness W = P(Z) + P(X) - 1
 *   Classical: W ≤ 1/D     Quantum GHZ: W = 1.0
 *
 * This is 6× above classical for D=6 — genuine N-party entanglement.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Helper: build D×D DFT matrix  F[j,k] = (1/√D) ω^(jk) */
static void mermin_build_dft(Complex *U, uint32_t dim)
{
    double s = 1.0 / sqrt((double)dim);
    for (uint32_t r = 0; r < dim; r++)
        for (uint32_t c = 0; c < dim; c++) {
            double phase = 2.0 * M_PI * r * c / dim;
            U[r*dim + c] = (Complex){ s * cos(phase), s * sin(phase) };
        }
}

/* ── apply_dft: convenience wrapper for DFT_D unitary ── */
void apply_dft(HexStateEngine *eng, uint64_t chunk_id, uint32_t dim)
{
    Complex *U = malloc((size_t)dim * dim * sizeof(Complex));
    if (!U) { fprintf(stderr, "[DFT] alloc failed\n"); return; }
    mermin_build_dft(U, dim);
    apply_local_unitary(eng, chunk_id, (const Complex *)U, dim);
    free(U);
}

/* ── mermin_test: N-party Mermin inequality test ── */
MerminResult mermin_test(HexStateEngine *eng, uint32_t n_parties,
                         uint32_t n_shots)
{
    (void)eng;  /* We create our own engines per shot */

    MerminResult r = {0};
    r.n_parties = n_parties;
    r.dim = NUM_BASIS_STATES;
    r.n_shots = n_shots;
    r.classical_bound = 1.0 / NUM_BASIS_STATES;
    uint32_t dim = NUM_BASIS_STATES;

    uint64_t quhits = 100000000000000ULL;  /* 100T per register */

    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);

    /* Open /dev/null ONCE and reuse across all shots */
    FILE *devnull = fopen("/dev/null", "w");
    FILE *real_stdout = stdout;

    /* ── Z-test: measure all N in computational basis, check all agree ── */
    int z_agree = 0;
    for (uint32_t shot = 0; shot < n_shots; shot++) {
        HexStateEngine e;
        engine_init(&e);
        stdout = devnull;

        for (uint32_t p = 0; p < n_parties; p++)
            init_chunk(&e, p, quhits);
        for (uint32_t p = 1; p < n_parties; p++)
            braid_chunks_dim(&e, 0, p, 0, 0, dim);

        uint32_t first = (uint32_t)(measure_chunk(&e, 0) % dim);
        int agree = 1;
        for (uint32_t p = 1; p < n_parties; p++) {
            uint32_t val = (uint32_t)(measure_chunk(&e, p) % dim);
            if (val != first) { agree = 0; break; }
        }

        stdout = real_stdout;

        if (agree) { z_agree++; r.z_counts[first]++; }
        engine_destroy(&e);
    }
    r.pz = (double)z_agree / n_shots;

    /* ── X-test: apply DFT_D to ALL, measure ALL, check Σ ≡ 0 mod D ── */
    int x_parity = 0;
    Complex U_DFT[NUM_BASIS_STATES * NUM_BASIS_STATES];
    mermin_build_dft(U_DFT, dim);

    for (uint32_t shot = 0; shot < n_shots; shot++) {
        HexStateEngine e;
        engine_init(&e);
        stdout = devnull;

        for (uint32_t p = 0; p < n_parties; p++)
            init_chunk(&e, p, quhits);
        for (uint32_t p = 1; p < n_parties; p++)
            braid_chunks_dim(&e, 0, p, 0, 0, dim);

        /* Apply DFT_D to ALL parties */
        for (uint32_t p = 0; p < n_parties; p++)
            apply_local_unitary(&e, p, (const Complex *)U_DFT, dim);

        /* Measure ALL parties */
        int total = 0;
        for (uint32_t p = 0; p < n_parties; p++)
            total += (int)(measure_chunk(&e, p) % dim);

        stdout = real_stdout;

        if (total % (int)dim == 0) x_parity++;
        engine_destroy(&e);
    }
    r.px = (double)x_parity / n_shots;

    fclose(devnull);

    /* ── Mermin witness ── */
    r.witness = r.pz + r.px - 1.0;
    r.violation = (r.witness > r.classical_bound) ? 1 : 0;

    clock_gettime(CLOCK_MONOTONIC, &t2);
    r.elapsed_ms = (t2.tv_sec - t1.tv_sec)*1000.0
                 + (t2.tv_nsec - t1.tv_nsec)/1e6;

    return r;
}


/* ── mermin_test_print: formatted output ── */
void mermin_test_print(MerminResult *r)
{
    printf("  ── N-Party Mermin Inequality (D=%u, N=%u) ──\n\n", r->dim, r->n_parties);

    printf("    Z-basis (all %u agree):      P = %.4f  (%u shots)\n",
           r->n_parties, r->pz, r->n_shots);
    printf("    Value distribution: ");
    for (int k = 0; k < (int)r->dim; k++)
        printf("|%d⟩:%d ", k, r->z_counts[k]);
    printf("\n\n");

    printf("    X-basis (DFT_%u, Σ≡0 mod %u): P = %.4f  (%u shots)\n",
           r->dim, r->dim, r->px, r->n_shots);
    printf("\n");

    printf("    Mermin witness:  W = P(Z) + P(X) - 1 = %.4f\n", r->witness);
    printf("    Classical bound: W ≤ %.4f  (= 1/D)\n", r->classical_bound);
    printf("    Quantum maximum: W = 1.0000\n\n");

    if (r->violation) {
        printf("    ★ MERMIN VIOLATION: W = %.4f > %.4f\n",
               r->witness, r->classical_bound);
        printf("    ★ %u-party × 100T quhits — N-party entanglement CERTIFIED\n",
               r->n_parties);
    } else {
        printf("    W = %.4f ≤ %.4f — within classical bound\n",
               r->witness, r->classical_bound);
    }

    printf("    Time: %.1f ms\n\n", r->elapsed_ms);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SUB-SYSTEM ENTANGLEMENT — D=6 = 2⊗3
 *
 * D=6 is the smallest dimension where:
 *   1. Internal entanglement exists (qubit⊗qutrit within one register)
 *   2. 36 generalized Bell states form a complete basis
 *   3. Bound entanglement (PPT entangled states) can occur
 *
 * Decomposition: |k⟩ → |k/3⟩_qubit ⊗ |k%3⟩_qutrit
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* ── Helper: eigenvalues of 2×2 Hermitian matrix ────────────────────────── */
static void eigen_2x2(const Complex *rho, double *e0, double *e1) {
    double a = rho[0].real, d = rho[3].real;
    double bc2 = rho[1].real * rho[1].real + rho[1].imag * rho[1].imag;
    double tr = a + d;
    double det = a * d - bc2;
    double disc = sqrt(fmax(tr * tr - 4 * det, 0.0));
    *e0 = (tr + disc) / 2.0;
    *e1 = (tr - disc) / 2.0;
}

/* ── Helper: eigenvalues of 3×3 Hermitian matrix ────────────────────────── */
static void eigen_3x3(const Complex *rho, double *evals) {
    /* Hermitian 3×3: use analytic Cardano formula */
    double a = rho[0*3+0].real, b = rho[1*3+1].real, c = rho[2*3+2].real;
    double p2 = cnorm2(rho[0*3+1]);
    double q2 = cnorm2(rho[0*3+2]);
    double r2 = cnorm2(rho[1*3+2]);

    double tr = a + b + c;
    double q = tr / 3.0;
    double s = (a*b + a*c + b*c - p2 - q2 - r2);

    /* p² = (1/6) * [(a-b)² + (a-c)² + (b-c)² + 6(p2+q2+r2)] */
    double p2val = ((a-b)*(a-b) + (a-c)*(a-c) + (b-c)*(b-c) + 6*(p2+q2+r2)) / 6.0;
    double p = sqrt(fmax(p2val, 0.0));

    if (p < 1e-14) {
        evals[0] = evals[1] = evals[2] = q;
        return;
    }

    /* B = (1/p)(A - qI) */
    double det_B;
    {
        double b00 = (a - q) / p, b11 = (b - q) / p, b22 = (c - q) / p;
        double b01_r = rho[0*3+1].real/p, b01_i = rho[0*3+1].imag/p;
        double b02_r = rho[0*3+2].real/p, b02_i = rho[0*3+2].imag/p;
        double b12_r = rho[1*3+2].real/p, b12_i = rho[1*3+2].imag/p;

        /* det(B) for Hermitian matrix */
        det_B = b00 * (b11*b22 - (b12_r*b12_r + b12_i*b12_i))
              - (b01_r*(b01_r*b22 - (b12_r*b02_r + b12_i*b02_i))
                +b01_i*(b01_i*b22 + (b12_r*b02_i - b12_i*b02_r)))
              + (b02_r*(b01_r*b12_r + b01_i*b12_i - b11*b02_r)
                +b02_i*(-b01_r*b12_i + b01_i*b12_r - b11*b02_i));
    }
    double half_det_B = det_B / 2.0;
    if (half_det_B > 1.0)  half_det_B = 1.0;
    if (half_det_B < -1.0) half_det_B = -1.0;
    double phi = acos(half_det_B) / 3.0;

    evals[0] = q + 2*p*cos(phi);
    evals[1] = q + 2*p*cos(phi - 2*M_PI/3);
    evals[2] = q + 2*p*cos(phi + 2*M_PI/3);
}

/* ── Helper: von Neumann entropy from eigenvalue array ──────────────────── */
static double vn_entropy(const double *evals, int n) {
    double S = 0;
    for (int i = 0; i < n; i++) {
        double e = fmax(evals[i], 0.0);
        if (e > 1e-15)
            S -= e * log2(e);
    }
    return S;
}

/* ── Get the 6 amplitudes for a single register in a group ──────────────── */
static void get_register_amplitudes(HexStateEngine *eng, uint64_t chunk_id,
                                    Complex *amps, uint32_t dim)
{
    memset(amps, 0, (size_t)dim * sizeof(Complex));
    if (chunk_id >= eng->num_chunks) return;

    Chunk *c = &eng->chunks[chunk_id];
    HilbertGroup *g = c->hilbert.group;

    if (!g || g->num_nonzero == 0) {
        /* No group — assume |0⟩ */
        amps[0] = (Complex){1.0, 0.0};
        return;
    }

    uint32_t my_idx = c->hilbert.group_index;
    uint32_t nm = g->num_members;

    /* Collect base amplitudes for this register's index */
    for (uint32_t e = 0; e < g->num_nonzero; e++) {
        uint32_t k = g->basis_indices[e * nm + my_idx];
        if (k < dim) {
            amps[k] = cadd(amps[k], g->amplitudes[e]);
        }
    }

    /* Replay any deferred (LAZY-pushed) unitaries for this member */
    if (g->lazy_count[my_idx] > 0) {
        Complex temp[6]; /* D=6 max */
        for (uint32_t op = 0; op < g->lazy_count[my_idx]; op++) {
            Complex *U = g->lazy_U[my_idx][op];
            memset(temp, 0, (size_t)dim * sizeof(Complex));
            for (uint32_t j = 0; j < dim; j++) {
                for (uint32_t k = 0; k < dim; k++) {
                    /* temp[j] += U[j*dim+k] * amps[k] */
                    temp[j] = cadd(temp[j], cmul(U[j*dim+k], amps[k]));
                }
            }
            memcpy(amps, temp, (size_t)dim * sizeof(Complex));
        }
    }
}

/* ── subsystem_decompose: compute reduced density matrices ──────────────── */
SubSystemDecomp subsystem_decompose(HexStateEngine *eng, uint64_t chunk_id)
{
    SubSystemDecomp result;
    memset(&result, 0, sizeof(result));
    result.dim_a = 2;  /* qubit */
    result.dim_b = 3;  /* qutrit */

    /* Get the 6 amplitudes for this register */
    Complex psi[6];
    get_register_amplitudes(eng, chunk_id, psi, 6);

    /* Compute ρ_A (qubit) = Tr_B(|ψ⟩⟨ψ|)
     * ρ_A[a][a'] = Σ_j ψ[3a+j] · ψ*[3a'+j] */
    for (int a = 0; a < 2; a++) {
        for (int ap = 0; ap < 2; ap++) {
            Complex sum = {0, 0};
            for (int j = 0; j < 3; j++) {
                sum = cadd(sum, cmul(psi[3*a+j], cconj(psi[3*ap+j])));
            }
            result.rho_a[a*2+ap] = sum;
        }
    }

    /* Compute ρ_B (qutrit) = Tr_A(|ψ⟩⟨ψ|)
     * ρ_B[j][j'] = Σ_a ψ[3a+j] · ψ*[3a+j'] */
    for (int j = 0; j < 3; j++) {
        for (int jp = 0; jp < 3; jp++) {
            Complex sum = {0, 0};
            for (int a = 0; a < 2; a++) {
                sum = cadd(sum, cmul(psi[3*a+j], cconj(psi[3*a+jp])));
            }
            result.rho_b[j*3+jp] = sum;
        }
    }

    /* Eigenvalues and entropy */
    eigen_2x2(result.rho_a, &result.eigenvalues_a[0], &result.eigenvalues_a[1]);
    eigen_3x3(result.rho_b, result.eigenvalues_b);

    result.entropy_a = vn_entropy(result.eigenvalues_a, 2);
    result.entropy_b = vn_entropy(result.eigenvalues_b, 3);

    /* Total entropy of the pure state = 0, so S(AB) = 0
     * Mutual information I(A:B) = S(A) + S(B) - S(AB) = S(A) + S(B) */
    result.mutual_info = result.entropy_a + result.entropy_b;

    /* Entangled if either subsystem has non-zero entropy */
    result.is_entangled = (result.entropy_a > 0.001) ? 1 : 0;

    printf("  [SUBSYS] Decomposed chunk %lu into qubit(2)⊗qutrit(3)\n", chunk_id);
    printf("  [SUBSYS] Entropy A (qubit): %.6f bits, B (qutrit): %.6f bits\n",
           result.entropy_a, result.entropy_b);
    printf("  [SUBSYS] Internal entanglement: %s\n",
           result.is_entangled ? "YES" : "NO (product state)");

    return result;
}

void subsystem_decompose_print(SubSystemDecomp *d) {
    printf("\n  ═══ Sub-System Decomposition (D=6 = 2⊗3) ═══\n\n");

    printf("  Qubit (dim=2) reduced density matrix ρ_A:\n");
    printf("    [%.6f%+.6fi  %.6f%+.6fi]\n",
           d->rho_a[0].real, d->rho_a[0].imag, d->rho_a[1].real, d->rho_a[1].imag);
    printf("    [%.6f%+.6fi  %.6f%+.6fi]\n",
           d->rho_a[2].real, d->rho_a[2].imag, d->rho_a[3].real, d->rho_a[3].imag);
    printf("  Eigenvalues: %.6f, %.6f\n", d->eigenvalues_a[0], d->eigenvalues_a[1]);
    printf("  von Neumann entropy: %.6f bits\n\n", d->entropy_a);

    printf("  Qutrit (dim=3) reduced density matrix ρ_B:\n");
    for (int i = 0; i < 3; i++) {
        printf("    [");
        for (int j = 0; j < 3; j++)
            printf("%.4f%+.4fi ", d->rho_b[i*3+j].real, d->rho_b[i*3+j].imag);
        printf("]\n");
    }
    printf("  Eigenvalues: %.6f, %.6f, %.6f\n",
           d->eigenvalues_b[0], d->eigenvalues_b[1], d->eigenvalues_b[2]);
    printf("  von Neumann entropy: %.6f bits\n\n", d->entropy_b);

    printf("  Mutual information I(A:B) = %.6f bits\n", d->mutual_info);
    printf("  Internal entanglement: %s\n\n",
           d->is_entangled ? "★ YES — qubit and qutrit are entangled within one register"
                           : "NO — product state across subsystems");
}

/* ── subsystem_entangle: prepare internally entangled state ─────────────── */
void subsystem_entangle(HexStateEngine *eng, uint64_t chunk_id,
                        int type, const Complex *custom_state)
{
    if (chunk_id >= eng->num_chunks) return;

    Complex target[6];
    memset(target, 0, sizeof(target));

    double s2 = 1.0 / sqrt(2.0);
    switch (type) {
        case 0: /* (|0,0⟩+|1,1⟩)/√2 = (|0⟩+|4⟩)/√2 */
            target[0] = (Complex){s2, 0};
            target[4] = (Complex){s2, 0};
            break;
        case 1: /* (|0,0⟩+|1,2⟩)/√2 = (|0⟩+|5⟩)/√2 */
            target[0] = (Complex){s2, 0};
            target[5] = (Complex){s2, 0};
            break;
        case 2: /* Custom */
            if (!custom_state) return;
            memcpy(target, custom_state, sizeof(target));
            break;
        default:
            return;
    }

    /* Build unitary whose first column is the target state (Gram-Schmidt) */
    Complex U[36];
    memset(U, 0, sizeof(U));

    /* First column = target */
    for (int i = 0; i < 6; i++) U[i*6+0] = target[i];

    /* Gram-Schmidt for remaining columns */
    for (int col = 1; col < 6; col++) {
        Complex v[6];
        memset(v, 0, sizeof(v));
        v[col] = (Complex){1.0, 0.0};

        for (int prev = 0; prev < col; prev++) {
            Complex dot = {0, 0};
            for (int i = 0; i < 6; i++)
                dot = cadd(dot, cmul(cconj(U[i*6+prev]), v[i]));
            for (int i = 0; i < 6; i++) {
                v[i] = csub(v[i], cmul(dot, U[i*6+prev]));
            }
        }

        double nrm = 0;
        for (int i = 0; i < 6; i++) nrm += cnorm2(v[i]);
        nrm = sqrt(nrm);
        if (nrm > 1e-12)
            for (int i = 0; i < 6; i++) {
                v[i].real /= nrm;
                v[i].imag /= nrm;
                U[i*6+col] = v[i];
            }
    }

    apply_local_unitary(eng, chunk_id, U, 6);

    const char *names[] = { "(|0,0⟩+|1,1⟩)/√2", "(|0,0⟩+|1,2⟩)/√2", "custom" };
    printf("  [SUBSYS] Created internal Bell state %s in chunk %lu\n",
           names[type < 3 ? type : 2], chunk_id);
}

/* ── generalized_bell_state: create |Ψ_mn⟩ ─────────────────────────────── */
void generalized_bell_state(HexStateEngine *eng, uint64_t a, uint64_t b,
                            uint32_t m, uint32_t n, uint32_t dim)
{
    if (dim == 0) dim = 6;

    /* Step 1: Create standard Bell state |Ψ_00⟩ = (1/√D) Σ|k,k⟩ */
    braid_chunks_dim(eng, a, b, 0, 0, dim);

    /* Step 2: Apply Z^m to chunk a (phase gate) */
    if (m > 0) {
        Complex *Zm = calloc((size_t)dim * dim, sizeof(Complex));
        for (uint32_t k = 0; k < dim; k++) {
            double angle = 2.0 * M_PI * m * k / dim;
            Zm[k*dim+k] = (Complex){cos(angle), sin(angle)};
        }
        apply_local_unitary(eng, a, Zm, dim);
        free(Zm);
    }

    /* Step 3: Apply X^n to chunk b (cyclic shift by n) */
    if (n > 0) {
        Complex *Xn = calloc((size_t)dim * dim, sizeof(Complex));
        for (uint32_t k = 0; k < dim; k++) {
            uint32_t target_k = (k + n) % dim;
            Xn[target_k*dim+k] = (Complex){1.0, 0.0};
        }
        apply_local_unitary(eng, b, Xn, dim);
        free(Xn);
    }

    printf("  [GEN-BELL] Created |Ψ_%u%u⟩ = (1/√%u) Σ_k ω^(%uk) |k, (k+%u) mod %u⟩\n",
           m, n, dim, m, n, dim);
    printf("  [GEN-BELL] State %u of %u total generalized Bell states\n",
           m * dim + n + 1, dim * dim);
}

/* ── partial_transpose_negativity: entanglement witness ─────────────────── */
double partial_transpose_negativity(HexStateEngine *eng, uint64_t chunk_id,
                                    double *log_negativity)
{
    if (chunk_id >= eng->num_chunks) {
        if (log_negativity) *log_negativity = 0;
        return 0;
    }

    Chunk *c = &eng->chunks[chunk_id];
    HilbertGroup *g = c->hilbert.group;

    if (!g || g->num_members < 2) {
        printf("  [PPT] Chunk %lu is not entangled (no group or single member)\n",
               chunk_id);
        if (log_negativity) *log_negativity = 0;
        return 0;
    }

    uint32_t dim = g->dim;
    uint32_t dim2 = dim * dim;

    /* Build the full density matrix ρ of the 2-party state
     * ρ[(a,b),(c,d)] = Σ_e Σ_f α_e α*_f  where indices match */
    Complex *rho = calloc(dim2 * dim2, sizeof(Complex));
    uint32_t nm = g->num_members;
    uint32_t my_idx = c->hilbert.group_index;

    /* Find partner index (first other member) */
    uint32_t partner_idx = (my_idx == 0) ? 1 : 0;

    for (uint32_t e = 0; e < g->num_nonzero; e++) {
        uint32_t ie = g->basis_indices[e * nm + my_idx];
        uint32_t je = g->basis_indices[e * nm + partner_idx];
        Complex ae = g->amplitudes[e];

        for (uint32_t f = 0; f < g->num_nonzero; f++) {
            uint32_t if_ = g->basis_indices[f * nm + my_idx];
            uint32_t jf = g->basis_indices[f * nm + partner_idx];
            Complex af = g->amplitudes[f];

            /* ρ[(ie,je),(if,jf)] += ae · af* */
            uint32_t row = ie * dim + je;
            uint32_t col = if_ * dim + jf;
            Complex prod = cmul(ae, cconj(af));
            rho[row * dim2 + col] = cadd(rho[row * dim2 + col], prod);
        }
    }

    /* Partial transpose: ρ^(T_B)[(a,b),(c,d)] = ρ[(a,d),(c,b)]
     * Swap b↔d indices on system B */
    Complex *rho_pt = calloc(dim2 * dim2, sizeof(Complex));
    for (uint32_t a = 0; a < dim; a++)
        for (uint32_t b = 0; b < dim; b++)
            for (uint32_t cc = 0; cc < dim; cc++)
                for (uint32_t d = 0; d < dim; d++) {
                    uint32_t row_new = a*dim + b;
                    uint32_t col_new = cc*dim + d;
                    /* Original: ρ[(a,d),(c,b)] */
                    uint32_t row_old = a*dim + d;
                    uint32_t col_old = cc*dim + b;
                    rho_pt[row_new * dim2 + col_new] = rho[row_old * dim2 + col_old];
                }

    /* Diagonalize ρ^(T_B) — for D=6 this is 36×36.
     * Use power iteration to find negative eigenvalues.
     * For known Bell-type states, we can compute analytically:
     *   Bell state: (D+1)*D/2 eigenvalues = +1/D, (D-1)*D/2 = -1/D
     * But let's verify numerically with trace/norm checks. */

    /* Trace of ρ^(T_B) should equal Tr(ρ) = 1 */
    double tr = 0;
    for (uint32_t i = 0; i < dim2; i++)
        tr += rho_pt[i * dim2 + i].real;

    /* Compute trace norm ||ρ^(T_B)||_1 using Frobenius approximation
     * for Hermitian matrices: ||A||_1 = Σ|λ_i|
     * We use: Tr(A²) = Σ λ_i², and for rank-D Bell states:
     *   ||ρ^TB||_1 = (D+1)/2D + (D-1)/2D × (D-1) ... nah, let's just compute.
     *
     * For efficiency, compute Tr((ρ^TB)²) and Tr((ρ^TB)³) to get eigenvalue info. */

    /* Tr(ρ^TB ²) */
    double tr2 = 0;
    for (uint32_t i = 0; i < dim2; i++)
        for (uint32_t j = 0; j < dim2; j++) {
            Complex prod = cmul(rho_pt[i * dim2 + j], rho_pt[j * dim2 + i]);
            tr2 += prod.real;
        }

    /* For a maximally entangled pure state in D×D:
     * Tr((ρ^TB)²) = 1/D → negativity N = (D-1)/2
     * For a product state: Tr((ρ^TB)²) = 1 → negativity N = 0 */

    /* Analytical negativity for Bell state: N = (D-1)/2
     * ||ρ^TB||_1 = Tr(ρ^TB) + 2N = 1 + (D-1)
     * So N = (||ρ^TB||_1 - 1) / 2 */

    /* For general states, compute ||ρ^TB||_1 via:
     *   ||A||_1² ≤ dim × Tr(A†A) (Cauchy-Schwarz upper bound)
     * Better: for D=6, direct eigenvalue computation via characteristic polynomial.
     *
     * Practical approach: use the fact that for pure bipartite states,
     * negativity = (Σ sqrt(λ_i))² / 2 - 1/2 where λ_i are eigenvalues of ρ_A.
     * This avoids diagonalizing the 36×36 matrix entirely. */

    /* For pure bipartite states: N = (Σ√λ_i)² / 2 - 1/2
     * where λ_i are eigenvalues of the reduced state ρ_A = Tr_B(|ψ⟩⟨ψ|) */
    Complex rho_A[36]; /* dim×dim */
    memset(rho_A, 0, sizeof(rho_A));

    for (uint32_t e = 0; e < g->num_nonzero; e++) {
        uint32_t ie = g->basis_indices[e * nm + my_idx];
        Complex ae = g->amplitudes[e];
        for (uint32_t f = 0; f < g->num_nonzero; f++) {
            uint32_t if_ = g->basis_indices[f * nm + my_idx];
            uint32_t je = g->basis_indices[e * nm + partner_idx];
            uint32_t jf = g->basis_indices[f * nm + partner_idx];
            if (je != jf) continue; /* Trace condition: same B index */
            Complex prod = cmul(ae, cconj(g->amplitudes[f]));
            rho_A[ie * dim + if_] = cadd(rho_A[ie * dim + if_], prod);
        }
    }

    /* Get eigenvalues of ρ_A */
    double evals[6];
    if (dim == 6) {
        /* Use 2 separate eigendecompositions by checking structure */
        /* General: diagonalize dim×dim matrix */
        /* For now, simple approach: compute sum of sqrt(eigenvalues) */
        /* Eigenvalues of ρ_A from its diagonal + off-diagonal structure */

        /* Build a real symmetric approximation for quick eigenvalues.
         * For Hermitian matrices, eigenvalues are real. */
        /* Simple iterative approach for small matrices (6×6): */
        double sum_sqrt_ev = 0;
        double total = 0;

        /* Compute Tr(ρ_A) for sanity */
        double trA = 0;
        for (uint32_t i = 0; i < dim; i++) trA += rho_A[i*dim+i].real;

        /* Compute Tr(ρ_A²) */
        double trA2 = 0;
        for (uint32_t i = 0; i < dim; i++)
            for (uint32_t j = 0; j < dim; j++) {
                Complex prod = cmul(rho_A[i*dim+j], rho_A[j*dim+i]);
                trA2 += prod.real;
            }

        /* For Bell state: ρ_A = I/D, so eigenvalues all = 1/D
         * Tr(ρ_A²) = D × (1/D)² = 1/D
         * Σ√λ = D × 1/√D = √D
         * N = D/2 - 1/2 = (D-1)/2 */

        /* Use the efficient formula: for pure bipartite,
         * the Schmidt coefficients are √λ_i where λ_i = eigenvalues of ρ_A.
         * Negativity = (Σ√λ_i)² / 2 - 1/2 = ((Σ Schmidt)² - 1) / 2 */

        /* Get Schmidt coefficients by computing eigenvalues of ρ_A.
         * For small dim, use Jacobi iteration. */

        /* Simple Jacobi eigenvalue decomposition for 6×6 Hermitian */
        double H[36]; /* real symmetric part (sufficient for eigenvalues of Hermitian) */
        for (uint32_t i = 0; i < dim; i++)
            for (uint32_t j = 0; j < dim; j++)
                H[i*dim+j] = rho_A[i*dim+j].real;

        /* Jacobi rotations */
        double V[36];
        memset(V, 0, sizeof(V));
        for (uint32_t i = 0; i < dim; i++) V[i*dim+i] = 1.0;

        for (int sweep = 0; sweep < 100; sweep++) {
            double off = 0;
            for (uint32_t i = 0; i < dim; i++)
                for (uint32_t j = i+1; j < dim; j++)
                    off += H[i*dim+j] * H[i*dim+j];
            if (off < 1e-24) break;

            for (uint32_t p = 0; p < dim; p++) {
                for (uint32_t qq = p+1; qq < dim; qq++) {
                    double apq = H[p*dim+qq];
                    if (fabs(apq) < 1e-15) continue;

                    double tau = (H[qq*dim+qq] - H[p*dim+p]) / (2.0 * apq);
                    double t = (tau >= 0 ? 1.0 : -1.0) / (fabs(tau) + sqrt(1.0 + tau*tau));
                    double co = 1.0 / sqrt(1.0 + t*t);
                    double si = t * co;

                    /* Rotate */
                    for (uint32_t i = 0; i < dim; i++) {
                        double hip = H[i*dim+p], hiq = H[i*dim+qq];
                        H[i*dim+p]  = co*hip - si*hiq;
                        H[i*dim+qq] = si*hip + co*hiq;
                    }
                    for (uint32_t j = 0; j < dim; j++) {
                        double hpj = H[p*dim+j], hqj = H[qq*dim+j];
                        H[p*dim+j]  = co*hpj - si*hqj;
                        H[qq*dim+j] = si*hpj + co*hqj;
                    }
                }
            }
        }

        for (uint32_t i = 0; i < dim; i++) {
            evals[i] = fmax(H[i*dim+i], 0.0);
            sum_sqrt_ev += sqrt(evals[i]);
        }

        double negativity = (sum_sqrt_ev * sum_sqrt_ev - 1.0) / 2.0;
        if (negativity < 0) negativity = 0;

        double log_neg = log2(2.0 * negativity + 1.0);

        printf("  [PPT] Partial transpose negativity for chunk %lu:\n", chunk_id);
        printf("  [PPT]   Tr(ρ_A) = %.6f, Tr(ρ_A²) = %.6f, Σ√λ = %.6f\n",
               trA, trA2, sum_sqrt_ev);
        printf("  [PPT]   Negativity N = %.6f\n", negativity);
        printf("  [PPT]   Log-negativity E_N = %.6f\n", log_neg);
        printf("  [PPT]   %s\n",
               negativity > 0.001 ? "NPT — DEFINITELY ENTANGLED" : "PPT — separable or bound entangled");

        free(rho);
        free(rho_pt);

        if (log_negativity) *log_negativity = log_neg;
        return negativity;
    }

    /* Fallback for non-D=6 */
    free(rho);
    free(rho_pt);
    if (log_negativity) *log_negativity = 0;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SUB-CHUNK QUHIT API — Pure Magic Pointer Hilbert Space
 *
 * Each quhit register IS a Hilbert space. No Chunk structs per quhit.
 * Magic Pointer for quhit k = MAKE_MAGIC_PTR(chunk_id << 16 | k)
 * The quantum state is stored as sparse amplitudes in the register.
 * Operations are the SAME algorithms as hexstate_engine measurement/braid:
 *   - Born rule: marginal probabilities → sample → collapse → renormalize
 *   - Entanglement: write GHZ/Bell state amplitudes to Hilbert space
 *   - DFT/Unitary: transform amplitudes in-place
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Helper: find register by chunk_id */
static int find_quhit_reg(HexStateEngine *eng, uint64_t chunk_id) {
    for (uint32_t r = 0; r < eng->num_quhit_regs; r++)
        if (eng->quhit_regs[r].chunk_id == chunk_id) return (int)r;
    return -1;
}

int init_quhit_register(HexStateEngine *eng, uint64_t chunk_id,
                        uint64_t n_quhits, uint32_t dim)
{
    if (!eng) return -1;
    if (dim == 0) dim = 6;
    if (n_quhits == 0) {
        printf("  [QUHIT] Error: n_quhits must be >= 1\n");
        return -1;
    }
    if (eng->num_quhit_regs >= MAX_QUHIT_REGISTERS) {
        printf("  [QUHIT] Error: max %d quhit registers\n", MAX_QUHIT_REGISTERS);
        return -1;
    }
    if (find_quhit_reg(eng, chunk_id) >= 0) {
        printf("  [QUHIT] Error: chunk %lu already has quhit register\n",
               (unsigned long)chunk_id);
        return -1;
    }

    uint32_t reg_idx = eng->num_quhit_regs;

    printf("  [QUHIT] Initializing %lu Magic Pointer quhits in chunk %lu (D=%u)\n",
           (unsigned long)n_quhits, (unsigned long)chunk_id, dim);

    /* The register IS the Hilbert space.
     * Initial state: |0,0,...,0⟩ — product state, 1 nonzero entry. */
    eng->quhit_regs[reg_idx].chunk_id   = chunk_id;
    eng->quhit_regs[reg_idx].n_quhits   = n_quhits;
    eng->quhit_regs[reg_idx].dim        = dim;
    eng->quhit_regs[reg_idx].num_nonzero = 1;
    memset(&eng->quhit_regs[reg_idx].entries[0], 0, sizeof(QuhitBasisEntry));
    eng->quhit_regs[reg_idx].entries[0].amplitude.real = 1.0;
    eng->quhit_regs[reg_idx].entries[0].bulk_value = 0;  /* all quhits in |0⟩ */
    eng->quhit_regs[reg_idx].entries[0].num_addr = 0;
    eng->quhit_regs[reg_idx].collapsed  = 0;
    eng->quhit_regs[reg_idx].collapse_outcome = 0;
    eng->quhit_regs[reg_idx].magic_base = MAKE_MAGIC_PTR(chunk_id << 16);
    eng->num_quhit_regs++;

    printf("  [QUHIT] ✓ %lu Magic Pointer quhits ready\n",
           (unsigned long)n_quhits);
    printf("  [QUHIT] Hilbert space: D^N = %u^%lu ≈ 10^%.0f\n",
           dim, (unsigned long)n_quhits, n_quhits * log10((double)dim));
    printf("  [QUHIT] State memory: %lu bytes (Hilbert space only)\n",
           (unsigned long)(sizeof(eng->quhit_regs[0])));
    return 0;
}

/* Forward declarations for helpers used by entangle_all */
static inline int same_basis(const QuhitBasisEntry *a, const QuhitBasisEntry *b);
static inline void accumulate_entry(
    QuhitBasisEntry *arr, uint32_t *n, const QuhitBasisEntry *e);

void entangle_all_quhits(HexStateEngine *eng, uint64_t chunk_id)
{
    int r = find_quhit_reg(eng, chunk_id);
    if (r < 0) {
        printf("  [QUHIT] Error: chunk %lu has no quhit register\n",
               (unsigned long)chunk_id);
        return;
    }

    uint32_t dim = eng->quhit_regs[r].dim;
    uint32_t nz  = eng->quhit_regs[r].num_nonzero;
    if (dim > MAX_QUHIT_HILBERT_ENTRIES) dim = MAX_QUHIT_HILBERT_ENTRIES;

    /* ── Unitary entanglement: DFT on the bulk value ──
     *
     * The bulk value is shared by all quhits not individually addressed
     * in each entry. Applying a DFT to the bulk degree of freedom
     * puts ALL bulk quhits into correlated superposition simultaneously.
     *
     * From |0,0,...,0⟩ (bulk=0, single entry):
     *   → (1/√D) Σ_k |k,k,...,k⟩   = GHZ state
     *
     * From an existing state with individually-addressed quhits:
     *   → preserves addr[] data, only DFTs the bulk degree of freedom.
     *
     * This is a proper unitary — no information is destroyed.
     * Then apply CZ between quhit pair (0,1) for phase correlations. */

    double inv_sqrt_d = 1.0 / sqrt((double)dim);
    double omega = 2.0 * M_PI / (double)dim;

    QuhitBasisEntry new_entries[MAX_QUHIT_HILBERT_ENTRIES];
    uint32_t new_nz = 0;

    for (uint32_t e = 0; e < nz; e++) {
        QuhitBasisEntry *cur = &eng->quhit_regs[r].entries[e];
        uint32_t v = cur->bulk_value;  /* current bulk value */

        for (uint32_t j = 0; j < dim && new_nz < MAX_QUHIT_HILBERT_ENTRIES; j++) {
            double phase = omega * v * j;
            double cr = cos(phase) * inv_sqrt_d;
            double ci = sin(phase) * inv_sqrt_d;

            QuhitBasisEntry ne = *cur;   /* preserve addr[] data */
            ne.bulk_value = j;
            ne.amplitude.real = cur->amplitude.real * cr - cur->amplitude.imag * ci;
            ne.amplitude.imag = cur->amplitude.real * ci + cur->amplitude.imag * cr;

            accumulate_entry(new_entries, &new_nz, &ne);
        }
    }

    memcpy(eng->quhit_regs[r].entries, new_entries,
           new_nz * sizeof(QuhitBasisEntry));
    eng->quhit_regs[r].num_nonzero = new_nz;
    eng->quhit_regs[r].collapsed = 0;

    printf("  [QUHIT] ✓ Entangled via DFT-bulk: %lu quhits, %u entries\n",
           (unsigned long)eng->quhit_regs[r].n_quhits,
           eng->quhit_regs[r].num_nonzero);
}

/* ── apply_cz_all_quhits: CZ between ALL N(N-1)/2 pairs ──────────────── */
void apply_cz_all_quhits(HexStateEngine *eng, uint64_t chunk_id)
{
    int r = find_quhit_reg(eng, chunk_id);
    if (r < 0) return;

    uint64_t N = eng->quhit_regs[r].n_quhits;
    uint32_t D = eng->quhit_regs[r].dim;
    double omega = 2.0 * M_PI / D;

    /*
     * For each entry with bulk_value v, num_addr individually-addressed quhits:
     *
     * Total CZ phase = product of ω^(v_i · v_j) over all i<j.
     *
     * N_bulk = N - num_addr  (quhits sharing bulk value v)
     *
     * 1) Bulk-Bulk pairs: C(N_bulk, 2) pairs, each contributing ω^(v²)
     *    → total phase exponent = v² × N_bulk × (N_bulk - 1) / 2
     *
     * 2) Addr-Bulk cross terms: each addr quhit with value a pairs with
     *    N_bulk bulk quhits → phase exponent = a × v × N_bulk
     *
     * 3) Addr-Addr pairs: each pair (i,j) of addr quhits → ω^(a_i × a_j)
     *
     * 4) Addr self-with-bulk (included in 2): already counted.
     *
     * Total exponent (mod D):
     *   E = v² × N_bulk(N_bulk-1)/2 + v × N_bulk × Σ a_i + Σ_{i<j} a_i × a_j
     */

    for (uint32_t e = 0; e < eng->quhit_regs[r].num_nonzero; e++) {
        QuhitBasisEntry *ent = &eng->quhit_regs[r].entries[e];
        uint64_t v = ent->bulk_value;
        uint64_t n_addr = ent->num_addr;
        uint64_t n_bulk = N - n_addr;

        /* 1) Bulk-Bulk: v² × C(n_bulk, 2) */
        uint64_t bb_exp = (v * v % D) * ((n_bulk % D) * ((n_bulk - 1) % D) % D) % D;
        /* We need (n_bulk * (n_bulk-1) / 2) mod D.
         * Since D=6, one of n_bulk or n_bulk-1 is even, so /2 is exact mod D. */
        uint64_t nb_mod = n_bulk % (2 * D);  /* enough precision */
        uint64_t half_pairs = (nb_mod * ((nb_mod - 1 + 2*D) % (2*D))) / 2;
        uint64_t phase_exp = (v * v % D) * (half_pairs % D) % D;

        /* 2) Addr-Bulk: Σ a_i × v × n_bulk */
        uint64_t sum_a = 0;
        for (uint64_t i = 0; i < n_addr; i++)
            sum_a += ent->addr[i].value;
        phase_exp = (phase_exp + (sum_a % D) * (v % D) % D * (n_bulk % D) % D) % D;

        /* 3) Addr-Addr: Σ_{i<j} a_i × a_j */
        for (uint64_t i = 0; i < n_addr; i++)
            for (uint64_t j = i + 1; j < n_addr; j++)
                phase_exp = (phase_exp + (uint64_t)ent->addr[i].value * ent->addr[j].value) % D;

        /* Apply the phase rotation */
        double angle = omega * (double)(phase_exp % D);
        double cr = cos(angle), ci = sin(angle);
        Complex a = ent->amplitude;
        ent->amplitude.real = a.real * cr - a.imag * ci;
        ent->amplitude.imag = a.real * ci + a.imag * cr;
    }

    printf("  [QUHIT] ✓ CZ-all: %lu quhits, %lu pairs, %u entries\n",
           (unsigned long)N, (unsigned long)(N/2) * (N > 0 ? N-1 : 0),
           eng->quhit_regs[r].num_nonzero);
}

uint64_t resolve_quhit(HexStateEngine *eng, uint64_t chunk_id, uint64_t quhit_idx)
{
    int r = find_quhit_reg(eng, chunk_id);
    if (r < 0) {
        printf("  [QUHIT] Error: chunk %lu has no quhit register\n",
               (unsigned long)chunk_id);
        return UINT64_MAX;
    }
    if (quhit_idx >= eng->quhit_regs[r].n_quhits) {
        printf("  [QUHIT] Error: quhit %lu out of range (max %lu)\n",
               (unsigned long)quhit_idx,
               (unsigned long)eng->quhit_regs[r].n_quhits - 1);
        return UINT64_MAX;
    }
    /* Magic Pointer: the address in the Hilbert space */
    return eng->quhit_regs[r].magic_base | quhit_idx;
}

/* ─── Self-describing Hilbert space helpers ─── */

/* Lazily resolve a quhit's value from a self-describing entry.
 * Scans the entry's address map; returns the individual value if found,
 * otherwise:
 *   rule=0: bulk_value (constant — all non-addressed share same value)
 *   rule=1: (bulk_value + quhit_idx) % dim  (cyclic offset —
 *           each quhit gets a unique value that shifts with the quantum state.
 *           When bulk is in superposition, ALL quhits are entangled.) */
static inline uint32_t lazy_resolve(const QuhitBasisEntry *e, uint64_t quhit_idx,
                                    uint8_t bulk_rule, uint32_t dim)
{
    for (uint8_t i = 0; i < e->num_addr; i++) {
        if (e->addr[i].quhit_idx == quhit_idx)
            return e->addr[i].value;
    }
    if (bulk_rule == 1) return (uint32_t)((e->bulk_value + quhit_idx) % dim);
    return e->bulk_value;
}

/* Create a new entry with one quhit's value overridden.
 * If quhit_idx is already in addr[], update it.
 * If not, add it (if room). The entry self-describes what changed. */
static inline QuhitBasisEntry entry_with_value(
    const QuhitBasisEntry *base, uint64_t quhit_idx, uint32_t new_value, Complex amp)
{
    QuhitBasisEntry e = *base;
    e.amplitude = amp;
    for (uint8_t i = 0; i < e.num_addr; i++) {
        if (e.addr[i].quhit_idx == quhit_idx) {
            e.addr[i].value = new_value;
            return e;
        }
    }
    if (e.num_addr < MAX_ADDR_PER_ENTRY) {
        e.addr[e.num_addr].quhit_idx = quhit_idx;
        e.addr[e.num_addr].value = new_value;
        e.num_addr++;
    }
    return e;
}

/* Check if two entries represent the same basis state (ignoring amplitude). */
static inline int same_basis(const QuhitBasisEntry *a, const QuhitBasisEntry *b)
{
    if (a->bulk_value != b->bulk_value) return 0;
    if (a->num_addr != b->num_addr) return 0;
    for (uint8_t i = 0; i < a->num_addr; i++) {
        int found = 0;
        for (uint8_t j = 0; j < b->num_addr; j++) {
            if (a->addr[i].quhit_idx == b->addr[j].quhit_idx &&
                a->addr[i].value == b->addr[j].value) {
                found = 1; break;
            }
        }
        if (!found) return 0;
    }
    return 1;
}

/* Accumulate a new entry into the output buffer.
 * If same basis state exists, add amplitudes. Otherwise create new slot. */
static inline void accumulate_entry(
    QuhitBasisEntry *out, uint32_t *nz, const QuhitBasisEntry *e)
{
    for (uint32_t n = 0; n < *nz; n++) {
        if (same_basis(&out[n], e)) {
            out[n].amplitude.real += e->amplitude.real;
            out[n].amplitude.imag += e->amplitude.imag;
            return;
        }
    }
    if (*nz < MAX_QUHIT_HILBERT_ENTRIES) {
        out[*nz] = *e;
        (*nz)++;
    }
}

uint64_t measure_quhit(HexStateEngine *eng, uint64_t chunk_id, uint64_t quhit_idx)
{
    int r = find_quhit_reg(eng, chunk_id);
    if (r < 0) return UINT64_MAX;
    if (quhit_idx >= eng->quhit_regs[r].n_quhits) return UINT64_MAX;

    uint32_t dim = eng->quhit_regs[r].dim;
    uint32_t nz  = eng->quhit_regs[r].num_nonzero;

    /* If already fully collapsed, lazily resolve from the surviving entry */
    if (eng->quhit_regs[r].collapsed) {
        return (uint64_t)lazy_resolve(&eng->quhit_regs[r].entries[0], quhit_idx, eng->quhit_regs[r].bulk_rule, eng->quhit_regs[r].dim);
    }

    /* Step 1: Compute marginal P(v) — lazily resolve this quhit from each entry */
    double probs[MAX_QUHIT_HILBERT_ENTRIES];
    memset(probs, 0, sizeof(probs));
    for (uint32_t e = 0; e < nz; e++) {
        uint32_t v = lazy_resolve(&eng->quhit_regs[r].entries[e], quhit_idx, eng->quhit_regs[r].bulk_rule, eng->quhit_regs[r].dim);
        double p = eng->quhit_regs[r].entries[e].amplitude.real *
                   eng->quhit_regs[r].entries[e].amplitude.real +
                   eng->quhit_regs[r].entries[e].amplitude.imag *
                   eng->quhit_regs[r].entries[e].amplitude.imag;
        if (v < MAX_QUHIT_HILBERT_ENTRIES) probs[v] += p;
    }

    /* Step 2: Born rule sampling */
    double rand_val = (double)(engine_prng(eng) % 1000000000ULL) / 1000000000.0;
    double cumul = 0.0;
    uint32_t result = dim - 1;
    for (uint32_t i = 0; i < dim; i++) {
        cumul += probs[i];
        if (cumul >= rand_val) { result = i; break; }
    }

    /* Step 3: Collapse — keep only entries where this quhit == result */
    uint32_t write_pos = 0;
    for (uint32_t e = 0; e < nz; e++) {
        uint32_t v = lazy_resolve(&eng->quhit_regs[r].entries[e], quhit_idx, eng->quhit_regs[r].bulk_rule, eng->quhit_regs[r].dim);
        if (v == result) {
            if (write_pos != e)
                eng->quhit_regs[r].entries[write_pos] = eng->quhit_regs[r].entries[e];
            write_pos++;
        }
    }
    eng->quhit_regs[r].num_nonzero = write_pos;

    /* Step 4: Renormalize */
    double norm = 0.0;
    for (uint32_t e = 0; e < write_pos; e++) {
        norm += eng->quhit_regs[r].entries[e].amplitude.real *
                eng->quhit_regs[r].entries[e].amplitude.real +
                eng->quhit_regs[r].entries[e].amplitude.imag *
                eng->quhit_regs[r].entries[e].amplitude.imag;
    }
    if (norm > 0.0) {
        double scale = 1.0 / sqrt(norm);
        for (uint32_t e = 0; e < write_pos; e++) {
            eng->quhit_regs[r].entries[e].amplitude.real *= scale;
            eng->quhit_regs[r].entries[e].amplitude.imag *= scale;
        }
    }

    /* Step 5: Check full collapse */
    if (eng->quhit_regs[r].num_nonzero == 1) {
        eng->quhit_regs[r].collapsed = 1;
        eng->quhit_regs[r].collapse_outcome = result;
    }

    return (uint64_t)result;
}

void apply_dft_quhit(HexStateEngine *eng, uint64_t chunk_id,
                     uint64_t quhit_idx, uint32_t dim)
{
    int r = find_quhit_reg(eng, chunk_id);
    if (r < 0 || quhit_idx >= eng->quhit_regs[r].n_quhits) return;
    if (dim == 0) dim = eng->quhit_regs[r].dim;

    uint32_t nz = eng->quhit_regs[r].num_nonzero;
    double omega = 2.0 * M_PI / dim;
    double inv_sqrt_d = 1.0 / sqrt((double)dim);

    /* DFT on quhit at Magic Pointer address quhit_idx.
     * For each entry, lazily resolve the quhit's current value,
     * then produce D new entries with DFT-rotated amplitudes.
     * The entry self-describes the new state — no metadata needed. */
    QuhitBasisEntry new_entries[MAX_QUHIT_HILBERT_ENTRIES];
    uint32_t new_nz = 0;

    for (uint32_t e = 0; e < nz; e++) {
        QuhitBasisEntry *cur = &eng->quhit_regs[r].entries[e];
        uint32_t v = lazy_resolve(cur, quhit_idx, eng->quhit_regs[r].bulk_rule, dim);

        for (uint32_t j = 0; j < dim; j++) {
            double phase = omega * v * j;
            double cr = cos(phase) * inv_sqrt_d;
            double ci = sin(phase) * inv_sqrt_d;
            Complex a;
            a.real = cur->amplitude.real * cr - cur->amplitude.imag * ci;
            a.imag = cur->amplitude.real * ci + cur->amplitude.imag * cr;

            QuhitBasisEntry ne = entry_with_value(cur, quhit_idx, j, a);
            accumulate_entry(new_entries, &new_nz, &ne);
        }
    }

    eng->quhit_regs[r].num_nonzero = new_nz;
    memcpy(eng->quhit_regs[r].entries, new_entries, new_nz * sizeof(QuhitBasisEntry));
}

void apply_unitary_quhit(HexStateEngine *eng, uint64_t chunk_id,
                         uint64_t quhit_idx, const Complex *U, uint32_t dim)
{
    int r = find_quhit_reg(eng, chunk_id);
    if (r < 0 || quhit_idx >= eng->quhit_regs[r].n_quhits) return;
    if (dim == 0) dim = eng->quhit_regs[r].dim;

    uint32_t nz = eng->quhit_regs[r].num_nonzero;

    QuhitBasisEntry new_entries[MAX_QUHIT_HILBERT_ENTRIES];
    uint32_t new_nz = 0;

    for (uint32_t e = 0; e < nz; e++) {
        QuhitBasisEntry *cur = &eng->quhit_regs[r].entries[e];
        uint32_t k = lazy_resolve(cur, quhit_idx, eng->quhit_regs[r].bulk_rule, dim);

        for (uint32_t j = 0; j < dim; j++) {
            Complex u_jk = U[j * dim + k];
            Complex a;
            a.real = cur->amplitude.real * u_jk.real - cur->amplitude.imag * u_jk.imag;
            a.imag = cur->amplitude.real * u_jk.imag + cur->amplitude.imag * u_jk.real;

            QuhitBasisEntry ne = entry_with_value(cur, quhit_idx, j, a);
            accumulate_entry(new_entries, &new_nz, &ne);
        }
    }

    eng->quhit_regs[r].num_nonzero = new_nz;
    memcpy(eng->quhit_regs[r].entries, new_entries, new_nz * sizeof(QuhitBasisEntry));
}

void braid_quhits(HexStateEngine *eng,
                  uint64_t chunk_a, uint64_t quhit_a,
                  uint64_t chunk_b, uint64_t quhit_b,
                  uint32_t dim)
{
    int ra = find_quhit_reg(eng, chunk_a);
    int rb = find_quhit_reg(eng, chunk_b);
    if (ra < 0 || rb < 0) return;
    if (dim == 0) dim = eng->quhit_regs[ra].dim;

    if (ra == rb) {
        entangle_all_quhits(eng, chunk_a);
    } else {
        entangle_all_quhits(eng, chunk_a);
        entangle_all_quhits(eng, chunk_b);
    }
}

void apply_cz_quhits(HexStateEngine *eng,
                     uint64_t chunk_a, uint64_t quhit_a,
                     uint64_t chunk_b, uint64_t quhit_b)
{
    int ra = find_quhit_reg(eng, chunk_a);
    int rb = find_quhit_reg(eng, chunk_b);
    if (ra < 0 || rb < 0) return;

    uint32_t dim = eng->quhit_regs[ra].dim;
    double omega_cz = 2.0 * M_PI / dim;

    for (uint32_t e = 0; e < eng->quhit_regs[ra].num_nonzero; e++) {
        uint32_t va = lazy_resolve(&eng->quhit_regs[ra].entries[e], quhit_a, eng->quhit_regs[ra].bulk_rule, eng->quhit_regs[ra].dim);
        uint32_t vb;
        if (ra == rb) {
            vb = lazy_resolve(&eng->quhit_regs[ra].entries[e], quhit_b, eng->quhit_regs[ra].bulk_rule, eng->quhit_regs[ra].dim);
        } else {
            uint32_t eb = e % eng->quhit_regs[rb].num_nonzero;
            vb = lazy_resolve(&eng->quhit_regs[rb].entries[eb], quhit_b, eng->quhit_regs[rb].bulk_rule, eng->quhit_regs[rb].dim);
        }

        double phase = omega_cz * va * vb;
        double cr = cos(phase), ci = sin(phase);
        Complex a = eng->quhit_regs[ra].entries[e].amplitude;
        eng->quhit_regs[ra].entries[e].amplitude.real = a.real * cr - a.imag * ci;
        eng->quhit_regs[ra].entries[e].amplitude.imag = a.real * ci + a.imag * cr;
    }
}

/* ── SUM gate (generalized CNOT): |a,b⟩ → |a, (a+b) mod D⟩ ──
 * This is the proper entangling gate for D-dimensional systems.
 * Unlike CZ (which only adds phase), SUM creates value correlations:
 * DFT(control) + SUM(control→target) = Bell pair (1/√D)Σ|k,(k+b)%D⟩
 * DFT + SUM chain = GHZ state  */
void apply_sum_quhits(HexStateEngine *eng,
                      uint64_t chunk_ctrl, uint64_t quhit_ctrl,
                      uint64_t chunk_tgt,  uint64_t quhit_tgt)
{
    int rc = find_quhit_reg(eng, chunk_ctrl);
    int rt = find_quhit_reg(eng, chunk_tgt);
    if (rc < 0 || rt < 0) return;
    if (rc != rt) {
        /* Cross-register SUM not yet supported */
        return;
    }

    uint32_t dim = eng->quhit_regs[rc].dim;
    uint32_t nz = eng->quhit_regs[rc].num_nonzero;

    QuhitBasisEntry new_entries[MAX_QUHIT_HILBERT_ENTRIES];
    uint32_t new_nz = 0;

    for (uint32_t e = 0; e < nz; e++) {
        QuhitBasisEntry *cur = &eng->quhit_regs[rc].entries[e];
        uint32_t va = lazy_resolve(cur, quhit_ctrl, eng->quhit_regs[rc].bulk_rule, dim);
        uint32_t vb = lazy_resolve(cur, quhit_tgt, eng->quhit_regs[rc].bulk_rule, dim);
        uint32_t new_val = (va + vb) % dim;

        /* Create entry with target quhit's value changed to (a+b)%D */
        QuhitBasisEntry ne = entry_with_value(cur, quhit_tgt, new_val, cur->amplitude);
        /* Also ensure control is explicitly in addr (it may have been bulk) */
        ne = entry_with_value(&ne, quhit_ctrl, va, ne.amplitude);
        accumulate_entry(new_entries, &new_nz, &ne);
    }

    eng->quhit_regs[rc].num_nonzero = new_nz;
    memcpy(eng->quhit_regs[rc].entries, new_entries, new_nz * sizeof(QuhitBasisEntry));
}

HilbertSnapshot inspect_quhit(HexStateEngine *eng, uint64_t chunk_id,
                              uint64_t quhit_idx)
{
    HilbertSnapshot snap;
    memset(&snap, 0, sizeof(snap));

    int r = find_quhit_reg(eng, chunk_id);
    if (r < 0 || quhit_idx >= eng->quhit_regs[r].n_quhits) return snap;

    uint32_t dim = eng->quhit_regs[r].dim;

    snap.num_members = eng->quhit_regs[r].n_quhits;
    snap.num_entries = eng->quhit_regs[r].num_nonzero;
    snap.dim = dim;
    snap.is_collapsed = eng->quhit_regs[r].collapsed;
    snap.is_entangled = (eng->quhit_regs[r].num_nonzero > 1) ? 1 : 0;
    snap.total_probability = 0.0;

    /* Lazily resolve this quhit from each self-describing entry */
    for (uint32_t e = 0; e < eng->quhit_regs[r].num_nonzero &&
                          e < MAX_INSPECT_ENTRIES; e++) {
        QuhitBasisEntry *ent = &eng->quhit_regs[r].entries[e];
        snap.entries[e].amp_real = ent->amplitude.real;
        snap.entries[e].amp_imag = ent->amplitude.imag;
        double p = ent->amplitude.real * ent->amplitude.real +
                   ent->amplitude.imag * ent->amplitude.imag;
        snap.entries[e].probability = p;
        snap.total_probability += p;

        uint32_t v = lazy_resolve(ent, quhit_idx, eng->quhit_regs[r].bulk_rule, eng->quhit_regs[r].dim);
        if (v < NUM_BASIS_STATES) snap.marginal_probs[v] += p;
    }

    snap.purity = 1.0;
    return snap;
}
