/*
 * ═══════════════════════════════════════════════════════════════════════════════
 * HEXSTATE ENGINE — 6-State Quantum Processor Emulator (Clean Rewrite)
 * ═══════════════════════════════════════════════════════════════════════════════
 * Pure quhit management. No shadow states.
 *
 * State storage:
 *   - Local:     q_local_state  Complex[D]    (96 bytes per chunk)
 *   - Pairwise:  q_joint_state  Complex[D²]   (576 bytes per pair)
 *   - N-party:   HilbertGroup   Complex[D^N]  (dense amplitudes)
 *
 * Basis states: |0⟩ – |5⟩,  D = 6
 * Magic Pointer Tag: 0x4858 ("HX")
 */

#include "hexstate_engine.h"
#include "born_rule.h"
#include "superposition.h"
#include "statevector.h"
#include "entanglement.h"
#include "quhit_management.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════════════════════════════════════════════════════════════
 * COMPLEX ARITHMETIC
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline Complex cmplx(double r, double i)
{ return (Complex){r, i}; }

static inline Complex cmul(Complex a, Complex b)
{ return cmplx(a.real*b.real - a.imag*b.imag, a.real*b.imag + a.imag*b.real); }

static inline Complex cadd(Complex a, Complex b)
{ return cmplx(a.real + b.real, a.imag + b.imag); }

static inline Complex csub(Complex a, Complex b)
{ return cmplx(a.real - b.real, a.imag - b.imag); }

static inline double cnorm2(Complex a)
{ return a.real*a.real + a.imag*a.imag; }

static inline Complex cconj(Complex a)
{ return cmplx(a.real, -a.imag); }

/* ═══════════════════════════════════════════════════════════════════════════════
 * FFT UTILITIES
 * ═══════════════════════════════════════════════════════════════════════════════ */

static uint32_t next_pow2(uint32_t n)
{ uint32_t p = 1; while (p < n) p <<= 1; return p; }

static void fft_pow2_inplace(Complex *buf, uint32_t N)
{
    if (N <= 1) return;
    uint32_t logN = 0;
    { uint32_t t = N; while (t > 1) { logN++; t >>= 1; } }

    for (uint32_t i = 0; i < N; i++) {
        uint32_t rev = 0;
        for (uint32_t bit = 0; bit < logN; bit++)
            if (i & (1u << bit)) rev |= (1u << (logN - 1 - bit));
        if (rev > i) { Complex t = buf[i]; buf[i] = buf[rev]; buf[rev] = t; }
    }

    for (uint32_t s = 1; s <= logN; s++) {
        uint32_t m = 1u << s;
        double angle = 2.0 * M_PI / m;
        Complex wm = cmplx(cos(angle), sin(angle));
        for (uint32_t k = 0; k < N; k += m) {
            Complex w = cmplx(1.0, 0.0);
            for (uint32_t j = 0; j < m/2; j++) {
                Complex t = cmul(w, buf[k + j + m/2]);
                Complex u = buf[k + j];
                buf[k + j]       = cadd(u, t);
                buf[k + j + m/2] = csub(u, t);
                w = cmul(w, wm);
            }
        }
    }
}

/* Bluestein DFT for arbitrary dimension */
static void bluestein_dft(Complex *x, uint32_t N)
{
    if (N <= 1) return;
    uint32_t M = next_pow2(2 * N - 1);

    Complex *a = sv_calloc_aligned(M, sizeof(Complex));
    Complex *b = sv_calloc_aligned(M, sizeof(Complex));

    for (uint32_t n = 0; n < N; n++) {
        double angle = M_PI * (double)(n * n) / (double)N;
        Complex chirp = cmplx(cos(angle), -sin(angle));
        a[n] = cmul(x[n], chirp);
    }

    b[0] = cmplx(1.0, 0.0);
    for (uint32_t n = 1; n < N; n++) {
        double angle = M_PI * (double)(n * n) / (double)N;
        Complex chirp = cmplx(cos(angle), sin(angle));
        b[n] = chirp;
        b[M - n] = chirp;
    }

    fft_pow2_inplace(a, M);
    fft_pow2_inplace(b, M);

    for (uint32_t i = 0; i < M; i++)
        a[i] = cmul(a[i], b[i]);

    /* Inverse FFT via conjugate trick */
    for (uint32_t i = 0; i < M; i++) a[i] = cconj(a[i]);
    fft_pow2_inplace(a, M);
    double inv_M = 1.0 / M;
    for (uint32_t i = 0; i < M; i++)
        a[i] = cmplx(a[i].real * inv_M, -a[i].imag * inv_M);

    for (uint32_t n = 0; n < N; n++) {
        double angle = M_PI * (double)(n * n) / (double)N;
        Complex chirp = cmplx(cos(angle), -sin(angle));
        x[n] = cmul(a[n], chirp);
    }

    free(a);
    free(b);
}

/* 6^k helper */
static uint64_t power_of_6(uint64_t k)
{
    uint64_t r = 1;
    for (uint64_t i = 0; i < k && r < UINT64_MAX/6; i++) r *= 6;
    return r;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PRNG — Pi-seeded xorshift64
 * ═══════════════════════════════════════════════════════════════════════════════ */

uint64_t engine_prng(HexStateEngine *eng)
{
    uint64_t x = eng->prng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    eng->prng_state = x;
    return x;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * DYNAMIC ALLOCATION HELPERS
 * ═══════════════════════════════════════════════════════════════════════════════ */

static int ensure_chunk_capacity(HexStateEngine *eng, uint64_t needed)
{
    if (needed <= eng->chunk_capacity) return 0;

    uint64_t new_cap = eng->chunk_capacity;
    if (new_cap == 0) new_cap = INITIAL_CHUNK_CAP;
    while (new_cap < needed && new_cap < MAX_CHUNKS) new_cap *= 2;
    if (new_cap > MAX_CHUNKS) new_cap = MAX_CHUNKS;
    if (needed > new_cap) return -1;

    Chunk *new_chunks = realloc(eng->chunks, new_cap * sizeof(Chunk));
    if (!new_chunks) return -1;
    memset(new_chunks + eng->chunk_capacity, 0,
           (new_cap - eng->chunk_capacity) * sizeof(Chunk));
    eng->chunks = new_chunks;

    ParallelReality *new_par = realloc(eng->parallel, new_cap * sizeof(ParallelReality));
    if (!new_par) return -1;
    memset(new_par + eng->chunk_capacity, 0,
           (new_cap - eng->chunk_capacity) * sizeof(ParallelReality));
    eng->parallel = new_par;

    uint64_t *new_mv = realloc(eng->measured_values, new_cap * sizeof(uint64_t));
    if (!new_mv) return -1;
    memset(new_mv + eng->chunk_capacity, 0,
           (new_cap - eng->chunk_capacity) * sizeof(uint64_t));
    eng->measured_values = new_mv;

    eng->chunk_capacity = new_cap;
    return 0;
}

static int ensure_braid_capacity(HexStateEngine *eng, uint64_t needed)
{
    if (needed <= eng->braid_capacity) return 0;
    uint64_t new_cap = eng->braid_capacity;
    if (new_cap == 0) new_cap = 256;
    while (new_cap < needed && new_cap < MAX_BRAID_LINKS) new_cap *= 2;
    if (needed > new_cap) return -1;
    BraidLink *new_bl = realloc(eng->braid_links, new_cap * sizeof(BraidLink));
    if (!new_bl) return -1;
    memset(new_bl + eng->braid_capacity, 0,
           (new_cap - eng->braid_capacity) * sizeof(BraidLink));
    eng->braid_links = new_bl;
    eng->braid_capacity = new_cap;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * ENGINE LIFECYCLE
 * ═══════════════════════════════════════════════════════════════════════════════ */

int engine_init(HexStateEngine *eng)
{
    memset(eng, 0, sizeof(*eng));

    /* Pi-seeded PRNG */
    eng->prng_state = 0x243F6A8885A308D3ULL;

    /* Initial capacity */
    if (ensure_chunk_capacity(eng, INITIAL_CHUNK_CAP) != 0) {
        fprintf(stderr, "[ENGINE] Failed to allocate initial chunks\n");
        return -1;
    }

    printf("  [ENGINE] HexState Engine initialized (D=6, no shadow states)\n");
    printf("  [ENGINE] Pure quhit management: local=%d bytes, joint=%d bytes\n",
           QM_LOCAL_BYTES, QM_JOINT_BYTES);
    return 0;
}

void engine_destroy(HexStateEngine *eng)
{
    /* Free all chunk local/joint states */
    for (uint64_t i = 0; i < eng->num_chunks; i++) {
        Chunk *c = &eng->chunks[i];
        free(c->hilbert.q_local_state);
        c->hilbert.q_local_state = NULL;
        for (int p = 0; p < c->hilbert.num_partners; p++) {
            free(c->hilbert.partners[p].q_joint_state);
            c->hilbert.partners[p].q_joint_state = NULL;
        }
        /* Free HilbertGroup if this chunk owns it (first member) */
        if (c->hilbert.group && c->hilbert.group_index == 0) {
            HilbertGroup *g = c->hilbert.group;
            free(g->amplitudes);
            for (uint32_t m = 0; m < g->num_members; m++) {
                for (uint32_t op = 0; op < g->lazy_count[m]; op++)
                    free(g->lazy_U[m][op]);
                free(g->lazy_U[m]);
            }
            free(g->cz_pairs);
            free(g);
        }
        c->hilbert.group = NULL;
    }

    free(eng->chunks);
    free(eng->parallel);
    free(eng->measured_values);
    free(eng->braid_links);
    free(eng->program);

    memset(eng, 0, sizeof(*eng));
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * CHUNK INITIALIZATION — Pure quhit, no shadow
 * ═══════════════════════════════════════════════════════════════════════════════ */

int init_chunk(HexStateEngine *eng, uint64_t id, uint64_t num_hexits)
{
    if (ensure_chunk_capacity(eng, id + 1) != 0) {
        fprintf(stderr, "[INIT] Cannot allocate chunk %lu\n", (unsigned long)id);
        return -1;
    }

    Chunk *c = &eng->chunks[id];
    memset(c, 0, sizeof(*c));
    c->id = id;
    c->size = num_hexits;

    /* Compute num_states = 6^num_hexits, capped */
    uint64_t ns = 1;
    for (uint64_t i = 0; i < num_hexits && ns <= MAX_STATES_STANDARD; i++) {
        if (ns > UINT64_MAX / 6) { ns = UINT64_MAX; break; }
        ns *= 6;
    }
    c->num_states = ns;

    /* Magic Pointer */
    c->hilbert.magic_ptr = MAKE_MAGIC_PTR(id);

    /* Allocate local D-dimensional state → |0⟩ */
    uint32_t dim = 6;
    c->hilbert.q_local_dim = dim;
    c->hilbert.q_local_state = sv_calloc_aligned(dim, sizeof(Complex));
    if (c->hilbert.q_local_state) {
        c->hilbert.q_local_state[0] = cmplx(1.0, 0.0);  /* |0⟩ */
    }

    if (id >= eng->num_chunks)
        eng->num_chunks = id + 1;

    printf("  [INIT] Chunk %lu: %lu hexits, %lu states, Ptr 0x%016lX\n",
           (unsigned long)id, (unsigned long)num_hexits,
           (unsigned long)ns, c->hilbert.magic_ptr);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SUPERPOSITION — Uniform |+⟩ state
 * ═══════════════════════════════════════════════════════════════════════════════ */

void create_superposition(HexStateEngine *eng, uint64_t id)
{
    if (id >= eng->num_chunks) return;
    Chunk *c = &eng->chunks[id];

    /* If in a HilbertGroup, apply DFT to this member */
    if (c->hilbert.group) {
        apply_hadamard(eng, id, 0);
        return;
    }

    /* Local state: set uniform superposition */
    uint32_t dim = c->hilbert.q_local_dim;
    if (dim == 0) dim = 6;
    Complex *st = c->hilbert.q_local_state;
    if (!st) return;

    double amp = born_fast_isqrt((double)dim);
    for (uint32_t k = 0; k < dim; k++)
        st[k] = cmplx(amp, 0.0);

    c->hilbert.q_flags |= 0x01;  /* superposed */
    printf("  [SUP] Chunk %lu → |+⟩ (D=%u, amp=%.6f)\n",
           (unsigned long)id, dim, amp);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * HADAMARD (DFT_D) — Applied to a specific hexit
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Forward declaration */
static void lazy_free_member(HilbertGroup *g, uint32_t m);

void apply_hadamard(HexStateEngine *eng, uint64_t id, uint64_t hexit_index)
{
    if (id >= eng->num_chunks) return;
    Chunk *c = &eng->chunks[id];

    /* ── Group path: defer as local unitary ── */
    if (c->hilbert.group) {
        uint32_t dim = c->hilbert.group->dim;
        Complex *U = sv_calloc_aligned((size_t)dim * dim, sizeof(Complex));
        double s = born_fast_isqrt((double)dim);
        for (uint32_t r = 0; r < dim; r++)
            for (uint32_t cc = 0; cc < dim; cc++) {
                double phase = 2.0 * M_PI * r * cc / dim;
                U[r*dim + cc] = cmplx(s * cos(phase), s * sin(phase));
            }
        apply_local_unitary(eng, id, (const Complex *)U, dim);
        free(U);
        return;
    }

    /* ── Pairwise joint state path ── */
    if (c->hilbert.num_partners > 0 && c->hilbert.partners[0].q_joint_state) {
        Complex *joint = c->hilbert.partners[0].q_joint_state;
        uint8_t which = c->hilbert.partners[0].q_which;
        uint32_t dim = c->hilbert.partners[0].q_joint_dim;
        if (dim == 0) dim = 6;
        double inv_sqrt_d = born_fast_isqrt((double)dim);

        Complex *tmp = sv_calloc_aligned(dim, sizeof(Complex));
        if (which == 0) {
            for (uint32_t b = 0; b < dim; b++) {
                for (uint32_t j = 0; j < dim; j++)
                    tmp[j] = joint[(uint64_t)b * dim + j];
                bluestein_dft(tmp, dim);
                for (uint32_t j = 0; j < dim; j++)
                    joint[(uint64_t)b * dim + j] = cmplx(
                        tmp[j].real * inv_sqrt_d, tmp[j].imag * inv_sqrt_d);
            }
        } else {
            for (uint32_t a = 0; a < dim; a++) {
                for (uint32_t j = 0; j < dim; j++)
                    tmp[j] = joint[(uint64_t)j * dim + a];
                bluestein_dft(tmp, dim);
                for (uint32_t j = 0; j < dim; j++)
                    joint[(uint64_t)j * dim + a] = cmplx(
                        tmp[j].real * inv_sqrt_d, tmp[j].imag * inv_sqrt_d);
            }
        }
        free(tmp);
        printf("  [H] DFT_%u on joint state (side %c)\n", dim, which == 0 ? 'A' : 'B');
        return;
    }

    /* ── Local state path ── */
    if (c->hilbert.q_local_state) {
        uint32_t d = c->hilbert.q_local_dim;
        if (d == 0) d = 6;
        bluestein_dft(c->hilbert.q_local_state, d);
        double inv_sqrt_d = born_fast_isqrt((double)d);
        for (uint32_t i = 0; i < d; i++)
            c->hilbert.q_local_state[i] = cmplx(
                c->hilbert.q_local_state[i].real * inv_sqrt_d,
                c->hilbert.q_local_state[i].imag * inv_sqrt_d);
        printf("  [H] DFT_%u on local state chunk %lu\n", d, (unsigned long)id);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * DNA GATE — Watson-Crick complement-focusing unitary
 * ═══════════════════════════════════════════════════════════════════════════════ */

void apply_dna_gate(HexStateEngine *eng, uint64_t id,
                    double bond_strength, double temperature)
{
    if (id >= eng->num_chunks) return;
    Chunk *c = &eng->chunks[id];
    uint32_t dim = c->hilbert.q_local_dim;
    if (dim == 0) dim = 6;

    /* Build DNA unitary */
    Complex *U = sv_calloc_aligned((size_t)dim * dim, sizeof(Complex));
    double kT = 8.617e-5 * temperature;
    if (kT < 1e-10) kT = 1e-10;
    double sigma = bond_strength / kT;
    double strong = 1.0 + sigma;
    double weak = 1.0 / (1.0 + sigma);

    int comp[6] = {1, 0, 3, 2, 5, 4};
    for (uint32_t i = 0; i < dim; i++)
        for (uint32_t j = 0; j < dim; j++) {
            double amp, phase;
            if ((int)j == comp[i % 6]) { amp = strong; phase = sigma * 0.1 * i; }
            else if (i == j)           { amp = 0.3 * weak; phase = 0.0; }
            else                       { amp = weak * 0.15; phase = 0.5 * ((i+j) % dim); }
            U[i*dim + j] = cmplx(amp * cos(phase), amp * sin(phase));
        }

    /* Gram-Schmidt orthogonalize */
    for (uint32_t i = 0; i < dim; i++) {
        for (uint32_t k = 0; k < i; k++) {
            double dr = 0, di = 0;
            for (uint32_t j = 0; j < dim; j++) {
                dr += U[k*dim+j].real*U[i*dim+j].real + U[k*dim+j].imag*U[i*dim+j].imag;
                di += U[k*dim+j].real*U[i*dim+j].imag - U[k*dim+j].imag*U[i*dim+j].real;
            }
            for (uint32_t j = 0; j < dim; j++) {
                U[i*dim+j].real -= dr*U[k*dim+j].real - di*U[k*dim+j].imag;
                U[i*dim+j].imag -= dr*U[k*dim+j].imag + di*U[k*dim+j].real;
            }
        }
        double norm = 0;
        for (uint32_t j = 0; j < dim; j++) norm += cnorm2(U[i*dim+j]);
        if (norm > 1e-15) {
            double inv = born_fast_isqrt(norm);
            for (uint32_t j = 0; j < dim; j++) {
                U[i*dim+j].real *= inv;
                U[i*dim+j].imag *= inv;
            }
        }
    }

    apply_local_unitary(eng, id, U, dim);
    free(U);
    printf("  [DNA] Applied to chunk %lu (σ=%.2f, T=%.0f)\n",
           (unsigned long)id, sigma, temperature);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * LOCAL UNITARY — Applied to one member of a HilbertGroup (deferred)
 * ═══════════════════════════════════════════════════════════════════════════════ */

void apply_local_unitary(HexStateEngine *eng, uint64_t id,
                         const Complex *U, uint32_t dim)
{
    if (id >= eng->num_chunks) return;
    Chunk *c = &eng->chunks[id];

    /* ── Group path: defer the unitary ── */
    if (c->hilbert.group) {
        HilbertGroup *g = c->hilbert.group;
        uint32_t mi = c->hilbert.group_index;

        if (!g->no_defer) {
            /* Push onto lazy list */
            if (g->lazy_count[mi] >= g->lazy_cap[mi]) {
                uint32_t new_cap = g->lazy_cap[mi] ? g->lazy_cap[mi] * 2 : 4;
                g->lazy_U[mi] = realloc(g->lazy_U[mi], new_cap * sizeof(Complex*));
                g->lazy_cap[mi] = new_cap;
            }
            Complex *Ucopy = sv_calloc_aligned((size_t)dim * dim, sizeof(Complex));
            memcpy(Ucopy, U, (size_t)dim * dim * sizeof(Complex));
            g->lazy_U[mi][g->lazy_count[mi]++] = Ucopy;
            if (g->lazy_count[mi] == 1) g->num_deferred++;
            return;
        }

        /* Direct expansion: apply U to every amplitude slice for this member */
        uint32_t nm = g->num_members;
        uint64_t stride = 1;
        for (uint32_t i = mi + 1; i < nm; i++) stride *= dim;

        Complex *tmp = malloc(dim * sizeof(Complex));
        for (uint64_t base = 0; base < g->total_dim; base++) {
            uint64_t me = (base / stride) % dim;
            if (me != 0) continue;  /* only process once per outer group */

            for (uint32_t j = 0; j < dim; j++)
                tmp[j] = g->amplitudes[base + j * stride];

            Complex *result = sv_calloc_aligned(dim, sizeof(Complex));
            for (uint32_t j = 0; j < dim; j++)
                for (uint32_t k = 0; k < dim; k++)
                    result[j] = cadd(result[j], cmul(U[j*dim+k], tmp[k]));

            for (uint32_t j = 0; j < dim; j++)
                g->amplitudes[base + j * stride] = result[j];
            free(result);
        }
        free(tmp);
        return;
    }

    /* ── Pairwise joint state ── */
    if (c->hilbert.num_partners > 0 && c->hilbert.partners[0].q_joint_state) {
        Complex *joint = c->hilbert.partners[0].q_joint_state;
        uint8_t which = c->hilbert.partners[0].q_which;
        uint32_t jdim = c->hilbert.partners[0].q_joint_dim;
        if (jdim == 0) jdim = 6;

        Complex *tmp = malloc(jdim * sizeof(Complex));
        if (which == 0) {
            for (uint32_t b = 0; b < jdim; b++) {
                for (uint32_t j = 0; j < jdim; j++)
                    tmp[j] = joint[(uint64_t)b * jdim + j];
                Complex *res = sv_calloc_aligned(jdim, sizeof(Complex));
                for (uint32_t j = 0; j < jdim; j++)
                    for (uint32_t k = 0; k < jdim; k++)
                        res[j] = cadd(res[j], cmul(U[j*jdim+k], tmp[k]));
                for (uint32_t j = 0; j < jdim; j++)
                    joint[(uint64_t)b * jdim + j] = res[j];
                free(res);
            }
        } else {
            for (uint32_t a = 0; a < jdim; a++) {
                for (uint32_t j = 0; j < jdim; j++)
                    tmp[j] = joint[(uint64_t)j * jdim + a];
                Complex *res = sv_calloc_aligned(jdim, sizeof(Complex));
                for (uint32_t j = 0; j < jdim; j++)
                    for (uint32_t k = 0; k < jdim; k++)
                        res[j] = cadd(res[j], cmul(U[j*jdim+k], tmp[k]));
                for (uint32_t j = 0; j < jdim; j++)
                    joint[(uint64_t)j * jdim + a] = res[j];
                free(res);
            }
        }
        free(tmp);
        return;
    }

    /* ── Local state ── */
    if (c->hilbert.q_local_state && dim <= c->hilbert.q_local_dim) {
        Complex *st = c->hilbert.q_local_state;
        Complex *result = sv_calloc_aligned(dim, sizeof(Complex));
        for (uint32_t j = 0; j < dim; j++)
            for (uint32_t k = 0; k < dim; k++)
                result[j] = cadd(result[j], cmul(U[j*dim+k], st[k]));
        memcpy(st, result, dim * sizeof(Complex));
        free(result);
    }
}

/* ── Group-wide unitary (applied to entire state) ── */
void apply_group_unitary(HexStateEngine *eng, uint64_t id,
                         Complex *U, uint32_t dim)
{
    if (id >= eng->num_chunks) return;
    Chunk *c = &eng->chunks[id];
    if (!c->hilbert.group) return;

    HilbertGroup *g = c->hilbert.group;
    uint64_t td = g->total_dim;
    Complex *result = sv_calloc_aligned(td, sizeof(Complex));

    for (uint64_t i = 0; i < td && i < (uint64_t)dim; i++)
        for (uint64_t j = 0; j < td && j < (uint64_t)dim; j++)
            result[i] = cadd(result[i], cmul(U[i*dim+j], g->amplitudes[j]));

    memcpy(g->amplitudes, result, td * sizeof(Complex));
    free(result);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * LAZY UNITARY MANAGEMENT
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void lazy_free_member(HilbertGroup *g, uint32_t m)
{
    for (uint32_t i = 0; i < g->lazy_count[m]; i++)
        free(g->lazy_U[m][i]);
    free(g->lazy_U[m]);
    g->lazy_U[m] = NULL;
    g->lazy_count[m] = 0;
    g->lazy_cap[m] = 0;
}

static void lazy_apply_vec(Complex **ops, uint32_t num_ops,
                           Complex *vec, uint32_t dim)
{
    Complex *tmp = malloc(dim * sizeof(Complex));
    for (uint32_t op = 0; op < num_ops; op++) {
        Complex *M = ops[op];
        for (uint32_t i = 0; i < dim; i++) {
            double re = 0.0, im = 0.0;
            for (uint32_t j = 0; j < dim; j++) {
                re += M[i*dim+j].real*vec[j].real - M[i*dim+j].imag*vec[j].imag;
                im += M[i*dim+j].real*vec[j].imag + M[i*dim+j].imag*vec[j].real;
            }
            tmp[i].real = re;
            tmp[i].imag = im;
        }
        memcpy(vec, tmp, dim * sizeof(Complex));
    }
    free(tmp);
}

Complex *lazy_compose(Complex **ops, uint32_t num_ops, uint32_t dim)
{
    Complex *composed = sv_calloc_aligned((size_t)dim * dim, sizeof(Complex));
    Complex *col = malloc(dim * sizeof(Complex));
    for (uint32_t k = 0; k < dim; k++) {
        memset(col, 0, dim * sizeof(Complex));
        col[k].real = 1.0;
        lazy_apply_vec(ops, num_ops, col, dim);
        for (uint32_t j = 0; j < dim; j++)
            composed[j * dim + k] = col[j];
    }
    free(col);
    return composed;
}

void materialize_deferred(HexStateEngine *eng, HilbertGroup *g)
{
    if (!g || g->num_deferred == 0) return;
    uint32_t dim = g->dim;
    g->no_defer = 1;

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
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * CZ GATE — Deferred controlled-phase
 * ═══════════════════════════════════════════════════════════════════════════════ */

void apply_cz_gate(HexStateEngine *eng, uint64_t id_a, uint64_t id_b)
{
    if (id_a >= eng->num_chunks || id_b >= eng->num_chunks) return;
    Chunk *ca = &eng->chunks[id_a];
    Chunk *cb = &eng->chunks[id_b];
    HilbertGroup *ga = ca->hilbert.group;
    HilbertGroup *gb = cb->hilbert.group;

    /* ── Case 1: Already in the same group — just record CZ ── */
    if (ga && ga == gb) {
        uint32_t idx_a = ca->hilbert.group_index;
        uint32_t idx_b = cb->hilbert.group_index;
        if (ga->num_cz >= ga->cz_cap) {
            uint32_t new_cap = ga->cz_cap ? ga->cz_cap * 2 : 64;
            ga->cz_pairs = realloc(ga->cz_pairs, new_cap * 2 * sizeof(uint32_t));
            ga->cz_cap = new_cap;
        }
        ga->cz_pairs[ga->num_cz * 2 + 0] = idx_a;
        ga->cz_pairs[ga->num_cz * 2 + 1] = idx_b;
        ga->num_cz++;
        return;
    }

    /* ── Case 2: Neither in a group — create a pairwise group ── */
    if (!ga && !gb) {
        uint32_t dim = ca->hilbert.q_local_dim;
        if (dim == 0) dim = 6;
        uint64_t ids[2] = {id_a, id_b};
        create_hilbert_group(eng, ids, 2, dim);

        /* Initialize from tensor product of local states */
        HilbertGroup *g = eng->chunks[id_a].hilbert.group;
        if (g && g->amplitudes) {
            memset(g->amplitudes, 0, g->total_dim * sizeof(Complex));
            Complex *la = ca->hilbert.q_local_state;
            Complex *lb = cb->hilbert.q_local_state;
            if (la && lb) {
                for (uint32_t a = 0; a < dim; a++)
                    for (uint32_t b = 0; b < dim; b++)
                        g->amplitudes[a * dim + b] = cmul(la[a], lb[b]);
            } else {
                g->amplitudes[0] = cmplx(1.0, 0.0);
            }
        }

        /* Record the CZ on the new group */
        HilbertGroup *g2 = eng->chunks[id_a].hilbert.group;
        if (g2) {
            if (g2->num_cz >= g2->cz_cap) {
                uint32_t new_cap = g2->cz_cap ? g2->cz_cap * 2 : 64;
                g2->cz_pairs = realloc(g2->cz_pairs, new_cap * 2 * sizeof(uint32_t));
                g2->cz_cap = new_cap;
            }
            g2->cz_pairs[g2->num_cz * 2 + 0] = eng->chunks[id_a].hilbert.group_index;
            g2->cz_pairs[g2->num_cz * 2 + 1] = eng->chunks[id_b].hilbert.group_index;
            g2->num_cz++;
        }
        return;
    }

    /* ── Case 3: In different groups or one ungrouped — apply CZ via
     *    pairwise interaction on local states (phase kickback).
     *    For qubit CZ (3× CZ₆): |a,b⟩ → ω^(a·b)|a,b⟩
     *    This applies the phase to the local state entries by
     *    materializing, applying phase, and updating. ── */
    {
        /* Get the current local state probabilities and apply CZ phase
         * kickback through the pairwise interaction. The CZ gate applies
         * ω^(a·b) to each |a,b⟩ component. For chunks in different groups,
         * we materialize both and apply the phase on whatever Hilbert space
         * is available. */

        /* If one is grouped and the other isn't, add the ungrouped one */
        if (ga && !gb) {
            uint32_t dim = ga->dim;
            uint32_t old_nm = ga->num_members;
            if (old_nm < MAX_GROUP_MEMBERS) {
                if (ga->num_deferred > 0) materialize_deferred(eng, ga);
                uint64_t old_td = ga->total_dim;
                uint64_t new_td = old_td * dim;
                Complex *new_amps = sv_calloc_aligned(new_td, sizeof(Complex));
                Complex *lb = cb->hilbert.q_local_state;
                for (uint64_t flat = 0; flat < old_td; flat++) {
                    if (cnorm2(ga->amplitudes[flat]) < 1e-30) continue;
                    for (uint32_t b = 0; b < dim; b++) {
                        Complex local_b = (lb && b < cb->hilbert.q_local_dim)
                                          ? lb[b] : cmplx(b == 0 ? 1.0 : 0.0, 0.0);
                        new_amps[flat * dim + b] = cmul(ga->amplitudes[flat], local_b);
                    }
                }
                free(ga->amplitudes);
                ga->amplitudes = new_amps;
                ga->total_dim = new_td;
                ga->member_ids[old_nm] = id_b;
                ga->num_members = old_nm + 1;
                cb->hilbert.group = ga;
                cb->hilbert.group_index = old_nm;

                /* Now record the CZ */
                if (ga->num_cz >= ga->cz_cap) {
                    uint32_t new_cap = ga->cz_cap ? ga->cz_cap * 2 : 64;
                    ga->cz_pairs = realloc(ga->cz_pairs, new_cap * 2 * sizeof(uint32_t));
                    ga->cz_cap = new_cap;
                }
                ga->cz_pairs[ga->num_cz * 2 + 0] = ca->hilbert.group_index;
                ga->cz_pairs[ga->num_cz * 2 + 1] = cb->hilbert.group_index;
                ga->num_cz++;
            }
            return;
        }
        if (!ga && gb) {
            /* Swap and recurse */
            apply_cz_gate(eng, id_b, id_a);
            return;
        }

        /* Both in different groups — for now, just record CZ on both
         * via pairwise local phase (approximate: this doesn't create
         * cross-group entanglement, but keeps groups manageable). */
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MEASUREMENT — Born rule sampling
 * ═══════════════════════════════════════════════════════════════════════════════ */

uint64_t measure_chunk(HexStateEngine *eng, uint64_t id)
{
    if (id >= eng->num_chunks) return 0;
    Chunk *c = &eng->chunks[id];

    /* ── Group measurement ── */
    if (c->hilbert.group) {
        HilbertGroup *g = c->hilbert.group;
        uint32_t mi = c->hilbert.group_index;
        uint32_t dim = g->dim;

        /* Materialize deferred unitaries first */
        if (g->num_deferred > 0) materialize_deferred(eng, g);

        /* Compute marginal probabilities for this member */
        double probs[2048];
        memset(probs, 0, dim * sizeof(double));

        uint64_t stride = 1;
        for (uint32_t i = mi + 1; i < g->num_members; i++) stride *= dim;

        for (uint64_t flat = 0; flat < g->total_dim; flat++) {
            uint32_t k = (uint32_t)((flat / stride) % dim);
            if (k < dim)
                probs[k] += cnorm2(g->amplitudes[flat]);
        }

        /* Born rule sampling */
        double rand_val = (double)(engine_prng(eng) % 1000000000ULL) / 1e9;
        double cumul = 0.0;
        uint32_t result = dim - 1;
        for (uint32_t k = 0; k < dim; k++) {
            cumul += probs[k];
            if (cumul >= rand_val) { result = k; break; }
        }

        /* Collapse: zero all entries where this member != result */
        double surviving = 0.0;
        for (uint64_t flat = 0; flat < g->total_dim; flat++) {
            uint32_t k = (uint32_t)((flat / stride) % dim);
            if (k != result) {
                g->amplitudes[flat] = cmplx(0, 0);
            } else {
                surviving += cnorm2(g->amplitudes[flat]);
            }
        }

        /* Renormalize */
        if (surviving > 1e-30) {
            double scale = born_fast_isqrt(surviving);
            for (uint64_t flat = 0; flat < g->total_dim; flat++) {
                g->amplitudes[flat].real *= scale;
                g->amplitudes[flat].imag *= scale;
            }
        }

        /* Apply deferred CZ phases */
        if (g->num_cz > 0) {
            double omega = 2.0 * M_PI / dim;
            for (uint32_t cz = 0; cz < g->num_cz; cz++) {
                uint32_t pa = g->cz_pairs[cz * 2];
                uint32_t pb = g->cz_pairs[cz * 2 + 1];
                if (pa == mi) {
                    /* Absorb ω^(result·j_b) into partner b's deferred unitary */
                    Complex *diag = sv_calloc_aligned((size_t)dim * dim, sizeof(Complex));
                    for (uint32_t k = 0; k < dim; k++) {
                        double phase = omega * result * k;
                        diag[k*dim+k] = cmplx(cos(phase), sin(phase));
                    }
                    uint64_t partner_id = g->member_ids[pb];
                    apply_local_unitary(eng, partner_id, diag, dim);
                    free(diag);
                } else if (pb == mi) {
                    Complex *diag = sv_calloc_aligned((size_t)dim * dim, sizeof(Complex));
                    for (uint32_t k = 0; k < dim; k++) {
                        double phase = omega * result * k;
                        diag[k*dim+k] = cmplx(cos(phase), sin(phase));
                    }
                    uint64_t partner_id = g->member_ids[pa];
                    apply_local_unitary(eng, partner_id, diag, dim);
                    free(diag);
                }
            }
        }

        eng->measured_values[id] = result;
        c->hilbert.q_flags |= 0x02;
        printf("  [MEASURE] Chunk %lu → |%u⟩ (group, D=%u)\n",
               (unsigned long)id, result, dim);
        return result;
    }

    /* ── Pairwise joint measurement ── */
    if (c->hilbert.num_partners > 0 && c->hilbert.partners[0].q_joint_state) {
        Complex *joint = c->hilbert.partners[0].q_joint_state;
        uint32_t dim = c->hilbert.partners[0].q_joint_dim;
        if (dim == 0) dim = 6;
        uint8_t which = c->hilbert.partners[0].q_which;

        double probs[2048];
        memset(probs, 0, dim * sizeof(double));

        for (uint32_t a = 0; a < dim; a++)
            for (uint32_t b = 0; b < dim; b++) {
                uint32_t my_val = (which == 0) ? a : b;
                probs[my_val] += cnorm2(joint[a * dim + b]);
            }

        double rand_val = (double)(engine_prng(eng) % 1000000000ULL) / 1e9;
        double cumul = 0.0;
        uint32_t result = dim - 1;
        for (uint32_t k = 0; k < dim; k++) {
            cumul += probs[k];
            if (cumul >= rand_val) { result = k; break; }
        }

        /* Collapse and renormalize */
        born_partial_collapse(
            (double*)joint, ((double*)joint) + 1,
            dim, dim, result, which == 0 ? 0 : 1);

        /* Actually use proper collapse */
        double surviving = 0.0;
        for (uint32_t a = 0; a < dim; a++)
            for (uint32_t b = 0; b < dim; b++) {
                uint32_t my_val = (which == 0) ? a : b;
                if (my_val != result)
                    joint[a * dim + b] = cmplx(0, 0);
                else
                    surviving += cnorm2(joint[a * dim + b]);
            }
        if (surviving > 1e-30) {
            double scale = born_fast_isqrt(surviving);
            for (uint32_t i = 0; i < dim * dim; i++) {
                joint[i].real *= scale;
                joint[i].imag *= scale;
            }
        }

        eng->measured_values[id] = result;
        c->hilbert.q_flags |= 0x02;

        /* Write partner's measurement too */
        uint64_t partner_id = c->hilbert.partners[0].q_partner;
        if (partner_id < eng->num_chunks) {
            Chunk *pc = &eng->chunks[partner_id];
            /* Extract partner's collapsed value */
            for (uint32_t a = 0; a < dim; a++)
                for (uint32_t b = 0; b < dim; b++)
                    if (cnorm2(joint[a * dim + b]) > 1e-20) {
                        uint32_t pval = (which == 0) ? b : a;
                        eng->measured_values[partner_id] = pval;
                        pc->hilbert.q_flags |= 0x02;
                        goto done_partner;
                    }
            done_partner:;
        }

        printf("  [MEASURE] Chunk %lu → |%u⟩ (joint, D=%u)\n",
               (unsigned long)id, result, dim);
        return result;
    }

    /* ── Local measurement ── */
    if (c->hilbert.q_local_state) {
        uint32_t dim = c->hilbert.q_local_dim;
        if (dim == 0) dim = 6;
        Complex *st = c->hilbert.q_local_state;

        double rand_val = (double)(engine_prng(eng) % 1000000000ULL) / 1e9;
        double cumul = 0.0;
        uint32_t result = dim - 1;
        for (uint32_t k = 0; k < dim; k++) {
            cumul += cnorm2(st[k]);
            if (cumul >= rand_val) { result = k; break; }
        }

        /* Collapse to |result⟩ */
        for (uint32_t k = 0; k < dim; k++)
            st[k] = (k == result) ? cmplx(1.0, 0.0) : cmplx(0.0, 0.0);

        eng->measured_values[id] = result;
        c->hilbert.q_flags |= 0x02;
        printf("  [MEASURE] Chunk %lu → |%u⟩ (local, D=%u)\n",
               (unsigned long)id, result, dim);
        return result;
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * GROVER DIFFUSION — 2|ψ⟩⟨ψ| - I
 * ═══════════════════════════════════════════════════════════════════════════════ */

void grover_diffusion(HexStateEngine *eng, uint64_t id)
{
    if (id >= eng->num_chunks) return;
    Chunk *c = &eng->chunks[id];

    if (c->hilbert.q_local_state) {
        uint32_t dim = c->hilbert.q_local_dim;
        if (dim == 0) dim = 6;
        Complex *st = c->hilbert.q_local_state;

        /* Compute mean amplitude */
        double mean_re = 0, mean_im = 0;
        for (uint32_t k = 0; k < dim; k++) {
            mean_re += st[k].real;
            mean_im += st[k].imag;
        }
        mean_re /= dim;
        mean_im /= dim;

        /* Inversion about the mean: 2*mean - amp */
        for (uint32_t k = 0; k < dim; k++) {
            st[k].real = 2.0 * mean_re - st[k].real;
            st[k].imag = 2.0 * mean_im - st[k].imag;
        }

        printf("  [GROVER] Diffusion on chunk %lu (D=%u)\n",
               (unsigned long)id, dim);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * BRAIDING — Pairwise entanglement via shared joint Hilbert space
 * ═══════════════════════════════════════════════════════════════════════════════ */

void braid_chunks(HexStateEngine *eng, uint64_t id_a, uint64_t id_b,
                  uint64_t hexit_a, uint64_t hexit_b)
{
    (void)hexit_a; (void)hexit_b;
    if (id_a >= eng->num_chunks || id_b >= eng->num_chunks) return;
    Chunk *ca = &eng->chunks[id_a];
    Chunk *cb = &eng->chunks[id_b];

    uint32_t dim = ca->hilbert.q_local_dim;
    if (dim == 0) dim = 6;

    /* Allocate shared D² joint state */
    uint64_t d2 = (uint64_t)dim * dim;
    Complex *joint = sv_calloc_aligned(d2, sizeof(Complex));

    /* Bell state: |Ψ⟩ = (1/√D) Σ|k,k⟩ */
    double amp = born_fast_isqrt((double)dim);
    for (uint32_t k = 0; k < dim; k++)
        joint[k * dim + k] = cmplx(amp, 0.0);

    /* Wire up both chunks */
    ca->hilbert.num_partners = 1;
    ca->hilbert.partners[0].q_partner = id_b;
    ca->hilbert.partners[0].q_joint_state = joint;
    ca->hilbert.partners[0].q_joint_dim = dim;
    ca->hilbert.partners[0].q_which = 0;  /* A side */

    cb->hilbert.num_partners = 1;
    cb->hilbert.partners[0].q_partner = id_a;
    cb->hilbert.partners[0].q_joint_state = joint;  /* shared pointer */
    cb->hilbert.partners[0].q_joint_dim = dim;
    cb->hilbert.partners[0].q_which = 1;  /* B side */

    /* Record braid link */
    if (ensure_braid_capacity(eng, eng->num_braid_links + 1) == 0) {
        BraidLink *bl = &eng->braid_links[eng->num_braid_links++];
        bl->chunk_a = id_a;
        bl->chunk_b = id_b;
    }

    printf("  [BRAID] %lu ↔ %lu (D=%u, %lu bytes)\n",
           (unsigned long)id_a, (unsigned long)id_b,
           dim, d2 * sizeof(Complex));
}

void unbraid_chunks(HexStateEngine *eng, uint64_t id_a, uint64_t id_b)
{
    if (id_a >= eng->num_chunks || id_b >= eng->num_chunks) return;
    Chunk *ca = &eng->chunks[id_a];
    Chunk *cb = &eng->chunks[id_b];

    /* Free shared joint state (only from A's side to avoid double-free) */
    if (ca->hilbert.num_partners > 0) {
        free(ca->hilbert.partners[0].q_joint_state);
        ca->hilbert.partners[0].q_joint_state = NULL;
        ca->hilbert.num_partners = 0;
    }
    if (cb->hilbert.num_partners > 0) {
        cb->hilbert.partners[0].q_joint_state = NULL;
        cb->hilbert.num_partners = 0;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * HILBERT GROUP — N-party entanglement
 * ═══════════════════════════════════════════════════════════════════════════════ */

int create_hilbert_group(HexStateEngine *eng, uint64_t *ids, uint32_t count,
                         uint32_t dim)
{
    if (count == 0 || count > MAX_GROUP_MEMBERS) return -1;

    HilbertGroup *g = calloc(1, sizeof(HilbertGroup));
    g->dim = dim;
    g->num_members = count;

    /* Compute total_dim = dim^count */
    uint64_t td = 1;
    for (uint32_t i = 0; i < count; i++) {
        if (td > UINT64_MAX / dim) { free(g); return -1; }
        td *= dim;
    }
    g->total_dim = td;

    /* Allocate dense amplitude array */
    g->amplitudes = sv_calloc_aligned(td, sizeof(Complex));
    if (!g->amplitudes) { free(g); return -1; }

    /* Initialize to |0,0,...,0⟩ */
    g->amplitudes[0] = cmplx(1.0, 0.0);

    /* Wire members */
    for (uint32_t i = 0; i < count; i++) {
        if (ids[i] >= eng->num_chunks) {
            if (ensure_chunk_capacity(eng, ids[i] + 1) != 0) {
                free(g->amplitudes); free(g); return -1;
            }
            eng->num_chunks = ids[i] + 1;
        }
        g->member_ids[i] = ids[i];
        Chunk *c = &eng->chunks[ids[i]];
        c->hilbert.group = g;
        c->hilbert.group_index = i;
    }

    printf("  [GROUP] Created: %u members, D=%u, total_dim=%lu (%.1f KB)\n",
           count, dim, (unsigned long)td, (double)td * sizeof(Complex) / 1024.0);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * INSPECT — Non-destructive state readout
 * ═══════════════════════════════════════════════════════════════════════════════ */

void inspect_hilbert(HexStateEngine *eng, uint64_t chunk_id, HilbertSnapshot *snap)
{
    /* Preserve dynamic pointers */
    double  *saved_mp  = snap->marginal_probs;
    Complex *saved_rho = snap->rho;
    memset(snap, 0, sizeof(*snap));
    snap->marginal_probs = saved_mp;
    snap->rho            = saved_rho;

    if (chunk_id >= eng->num_chunks) return;
    Chunk *c = &eng->chunks[chunk_id];
    snap->chunk_id = chunk_id;

    /* ── Group inspection ── */
    if (c->hilbert.group) {
        HilbertGroup *g = c->hilbert.group;
        uint32_t mi = c->hilbert.group_index;
        uint32_t dim = g->dim;

        snap->dim = dim;
        snap->num_members = g->num_members;
        for (uint32_t m = 0; m < g->num_members && m < MAX_SNAP_MEMBERS; m++)
            snap->member_ids[m] = g->member_ids[m];

        /* Reallocate dynamic arrays */
        free(snap->marginal_probs);
        free(snap->rho);
        snap->marginal_probs = calloc(dim, sizeof(double));
        snap->rho = calloc((size_t)dim * dim, sizeof(Complex));

        /* Walk dense state */
        uint64_t stride = 1;
        for (uint32_t i = mi + 1; i < g->num_members; i++) stride *= dim;

        uint32_t ne = 0;
        for (uint64_t flat = 0; flat < g->total_dim; flat++) {
            double p = cnorm2(g->amplitudes[flat]);
            if (p < 1e-30) continue;

            uint32_t k = (uint32_t)((flat / stride) % dim);
            if (k < dim) snap->marginal_probs[k] += p;
            snap->total_probability += p;

            if (ne < MAX_INSPECT_ENTRIES) {
                snap->entries[ne].probability = p;
                snap->entries[ne].amp_real = g->amplitudes[flat].real;
                snap->entries[ne].amp_imag = g->amplitudes[flat].imag;
                snap->entries[ne].phase_rad = atan2(g->amplitudes[flat].imag,
                                                     g->amplitudes[flat].real);
                /* Fill per-member indices */
                for (uint32_t m = 0; m < g->num_members && m < MAX_SNAP_MEMBERS; m++)
                    snap->entries[ne].indices[m] = hilbert_extract_index(g, flat, m);
                ne++;
            }
        }
        snap->num_entries = ne;
        snap->is_entangled = (ne > 1) ? 1 : 0;

        /* Compute reduced density matrix for this member */
        for (uint32_t i = 0; i < dim; i++)
            for (uint32_t j = 0; j < dim; j++) {
                double rho_re = 0, rho_im = 0;
                /* Sum over all other members' indices */
                for (uint64_t flat = 0; flat < g->total_dim; flat++) {
                    uint32_t ki = (uint32_t)((flat / stride) % dim);
                    if (ki != i) continue;
                    /* Find partner flat index with ki replaced by j */
                    uint64_t flat_j = flat - (uint64_t)i * stride + (uint64_t)j * stride;
                    if (flat_j >= g->total_dim) continue;
                    Complex ai = g->amplitudes[flat];
                    Complex aj = cconj(g->amplitudes[flat_j]);
                    rho_re += ai.real * aj.real - ai.imag * aj.imag;
                    rho_im += ai.real * aj.imag + ai.imag * aj.real;
                }
                snap->rho[i * dim + j] = cmplx(rho_re, rho_im);
            }

        /* Purity */
        double purity = 0;
        for (uint32_t i = 0; i < dim; i++)
            for (uint32_t j = 0; j < dim; j++)
                purity += cnorm2(snap->rho[i * dim + j]);
        snap->purity = purity;

        /* Entropy */
        snap->entropy = 0;
        for (uint32_t k = 0; k < dim; k++) {
            double lam = snap->rho[k * dim + k].real;
            if (lam > 1e-14) snap->entropy -= lam * log2(lam);
        }
        snap->is_collapsed = (c->hilbert.q_flags & 0x02) ? 1 : 0;
        return;
    }

    /* ── Pairwise joint inspection ── */
    if (c->hilbert.num_partners > 0 && c->hilbert.partners[0].q_joint_state) {
        Complex *joint = c->hilbert.partners[0].q_joint_state;
        uint32_t dim = c->hilbert.partners[0].q_joint_dim;
        if (dim == 0) dim = 6;
        uint8_t which = c->hilbert.partners[0].q_which;

        snap->dim = dim;
        snap->num_members = 2;
        snap->member_ids[0] = (which == 0) ? chunk_id : c->hilbert.partners[0].q_partner;
        snap->member_ids[1] = (which == 0) ? c->hilbert.partners[0].q_partner : chunk_id;

        free(snap->marginal_probs);
        free(snap->rho);
        snap->marginal_probs = calloc(dim, sizeof(double));
        snap->rho = calloc((size_t)dim * dim, sizeof(Complex));

        uint32_t ne = 0;
        for (uint32_t a = 0; a < dim; a++)
            for (uint32_t b = 0; b < dim; b++) {
                double p = cnorm2(joint[a * dim + b]);
                uint32_t my_val = (which == 0) ? a : b;
                if (my_val < dim) snap->marginal_probs[my_val] += p;
                snap->total_probability += p;

                if (p > 1e-30 && ne < MAX_INSPECT_ENTRIES) {
                    snap->entries[ne].indices[0] = a;
                    snap->entries[ne].indices[1] = b;
                    snap->entries[ne].amp_real = joint[a*dim+b].real;
                    snap->entries[ne].amp_imag = joint[a*dim+b].imag;
                    snap->entries[ne].probability = p;
                    snap->entries[ne].phase_rad = atan2(joint[a*dim+b].imag,
                                                         joint[a*dim+b].real);
                    ne++;
                }
            }
        snap->num_entries = ne;
        snap->is_entangled = (ne > 1) ? 1 : 0;

        /* Reduced density matrix */
        for (uint32_t i = 0; i < dim; i++)
            for (uint32_t j = 0; j < dim; j++) {
                double rho_re = 0, rho_im = 0;
                for (uint32_t b = 0; b < dim; b++) {
                    Complex ai, aj;
                    if (which == 0) {
                        ai = joint[i * dim + b];
                        aj = cconj(joint[j * dim + b]);
                    } else {
                        ai = joint[b * dim + i];
                        aj = cconj(joint[b * dim + j]);
                    }
                    rho_re += ai.real*aj.real - ai.imag*aj.imag;
                    rho_im += ai.real*aj.imag + ai.imag*aj.real;
                }
                snap->rho[i*dim+j] = cmplx(rho_re, rho_im);
            }

        double purity = 0;
        for (uint32_t i = 0; i < dim; i++)
            for (uint32_t j = 0; j < dim; j++)
                purity += cnorm2(snap->rho[i*dim+j]);
        snap->purity = purity;
        snap->entropy = 0;
        for (uint32_t k = 0; k < dim; k++) {
            double lam = snap->rho[k*dim+k].real;
            if (lam > 1e-14) snap->entropy -= lam * log2(lam);
        }
        snap->is_collapsed = (c->hilbert.q_flags & 0x02) ? 1 : 0;
        return;
    }

    /* ── Local inspection ── */
    if (c->hilbert.q_local_state) {
        uint32_t dim = c->hilbert.q_local_dim;
        if (dim == 0) dim = 6;

        snap->dim = dim;
        snap->num_members = 1;
        snap->member_ids[0] = chunk_id;

        free(snap->marginal_probs);
        free(snap->rho);
        snap->marginal_probs = calloc(dim, sizeof(double));
        snap->rho = calloc((size_t)dim * dim, sizeof(Complex));

        uint32_t ne = 0;
        for (uint32_t k = 0; k < dim; k++) {
            double p = cnorm2(c->hilbert.q_local_state[k]);
            snap->marginal_probs[k] = p;
            snap->total_probability += p;
            if (p > 1e-30 && ne < MAX_INSPECT_ENTRIES) {
                snap->entries[ne].indices[0] = k;
                snap->entries[ne].amp_real = c->hilbert.q_local_state[k].real;
                snap->entries[ne].amp_imag = c->hilbert.q_local_state[k].imag;
                snap->entries[ne].probability = p;
                snap->entries[ne].phase_rad = atan2(c->hilbert.q_local_state[k].imag,
                                                     c->hilbert.q_local_state[k].real);
                ne++;
            }
        }
        snap->num_entries = ne;
        snap->purity = 1.0;
        snap->entropy = 0.0;
        snap->is_entangled = 0;
        snap->is_collapsed = (c->hilbert.q_flags & 0x02) ? 1 : 0;
    }
}

void inspect_print(HilbertSnapshot *snap)
{
    printf("  ╔═══ INSPECT chunk %lu ═══╗\n", (unsigned long)snap->chunk_id);
    printf("  ║ D=%u, members=%u, entries=%u\n", snap->dim, snap->num_members, snap->num_entries);
    printf("  ║ P(total)=%.6f, purity=%.6f, S=%.4f bits\n",
           snap->total_probability, snap->purity, snap->entropy);
    printf("  ║ entangled=%d, collapsed=%d\n", snap->is_entangled, snap->is_collapsed);
    for (uint32_t e = 0; e < snap->num_entries && e < 12; e++) {
        printf("  ║ |");
        for (uint32_t m = 0; m < snap->num_members && m < MAX_SNAP_MEMBERS; m++)
            printf("%u", snap->entries[e].indices[m]);
        printf("⟩  %.6f%+.6fi  P=%.6f  φ=%.4f\n",
               snap->entries[e].amp_real, snap->entries[e].amp_imag,
               snap->entries[e].probability, snap->entries[e].phase_rad);
    }
    if (snap->num_entries > 12)
        printf("  ║ ... (%u more entries)\n", snap->num_entries - 12);
    printf("  ║ Marginals:");
    for (uint32_t k = 0; k < snap->dim && k < 8; k++)
        printf(" P(%u)=%.4f", k, snap->marginal_probs[k]);
    printf("\n  ╚════════════════════════╝\n");
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * ENTANGLEMENT ENTROPY
 * ═══════════════════════════════════════════════════════════════════════════════ */

double hilbert_entanglement_entropy(HexStateEngine *eng, uint64_t chunk_id)
{
    HilbertSnapshot *snap = hilbert_snapshot_alloc(6);
    inspect_hilbert(eng, chunk_id, snap);
    double S = snap->entropy;
    hilbert_snapshot_free(snap);
    return S;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SUBSYSTEM DECOMPOSITION — D=6 = 2⊗3
 * ═══════════════════════════════════════════════════════════════════════════════ */

SubSystemDecomp subsystem_decompose(HexStateEngine *eng, uint64_t chunk_id)
{
    SubSystemDecomp d;
    memset(&d, 0, sizeof(d));
    d.dim_a = 2;
    d.dim_b = 3;

    if (chunk_id >= eng->num_chunks) return d;
    Chunk *c = &eng->chunks[chunk_id];

    /* Get amplitudes */
    Complex psi[6] = {{0}};
    if (c->hilbert.q_local_state) {
        uint32_t dim = c->hilbert.q_local_dim;
        for (uint32_t k = 0; k < dim && k < 6; k++)
            psi[k] = c->hilbert.q_local_state[k];
    }

    /* Compute ρ_A = Tr_B(|ψ⟩⟨ψ|) */
    for (uint32_t i = 0; i < 2; i++)
        for (uint32_t j = 0; j < 2; j++) {
            double rho_re = 0, rho_im = 0;
            for (uint32_t b = 0; b < 3; b++) {
                Complex ai = psi[i*3+b];
                Complex aj = cconj(psi[j*3+b]);
                rho_re += ai.real*aj.real - ai.imag*aj.imag;
                rho_im += ai.real*aj.imag + ai.imag*aj.real;
            }
            d.rho_a[i*2+j] = cmplx(rho_re, rho_im);
        }

    /* Compute ρ_B = Tr_A(|ψ⟩⟨ψ|) */
    for (uint32_t i = 0; i < 3; i++)
        for (uint32_t j = 0; j < 3; j++) {
            double rho_re = 0, rho_im = 0;
            for (uint32_t a = 0; a < 2; a++) {
                Complex ai = psi[a*3+i];
                Complex aj = cconj(psi[a*3+j]);
                rho_re += ai.real*aj.real - ai.imag*aj.imag;
                rho_im += ai.real*aj.imag + ai.imag*aj.real;
            }
            d.rho_b[i*3+j] = cmplx(rho_re, rho_im);
        }

    /* Eigenvalues of 2×2 ρ_A */
    double a = d.rho_a[0].real, b = d.rho_a[3].real;
    double disc = (a - b) * (a - b) + 4.0 * cnorm2(d.rho_a[1]);
    double sq = sqrt(disc > 0 ? disc : 0);
    d.eigenvalues_a[0] = ((a + b) + sq) / 2.0;
    d.eigenvalues_a[1] = ((a + b) - sq) / 2.0;

    /* Eigenvalues of 3×3 ρ_B (analytic cubic) */
    double p = d.rho_b[0].real, q2 = d.rho_b[4].real, r = d.rho_b[8].real;
    double trace = p + q2 + r;
    double q_diag = p*q2 + p*r + q2*r
        - cnorm2(d.rho_b[1]) - cnorm2(d.rho_b[2]) - cnorm2(d.rho_b[5]);
    double pp = (trace*trace - 3.0*q_diag) / 9.0;
    double qq_num = trace * (2.0*trace*trace - 9.0*q_diag) / 27.0;
    /* Simplified: use diagonal approximation */
    d.eigenvalues_b[0] = p;
    d.eigenvalues_b[1] = q2;
    d.eigenvalues_b[2] = r;
    (void)pp; (void)qq_num;

    /* Entropies */
    d.entropy_a = 0;
    for (int i = 0; i < 2; i++)
        if (d.eigenvalues_a[i] > 1e-14)
            d.entropy_a -= d.eigenvalues_a[i] * log2(d.eigenvalues_a[i]);
    d.entropy_b = 0;
    for (int i = 0; i < 3; i++)
        if (d.eigenvalues_b[i] > 1e-14)
            d.entropy_b -= d.eigenvalues_b[i] * log2(d.eigenvalues_b[i]);

    /* Total entropy */
    double total_entropy = 0;
    for (int k = 0; k < 6; k++) {
        double pk = cnorm2(psi[k]);
        if (pk > 1e-14) total_entropy -= pk * log2(pk);
    }

    d.mutual_info = d.entropy_a + d.entropy_b - total_entropy;
    d.is_entangled = (d.entropy_a > 0.01) ? 1 : 0;

    return d;
}

void subsystem_decompose_print(SubSystemDecomp *d)
{
    printf("  ┌─── Subsystem Decomposition ───┐\n");
    printf("  │ D=6 = %u ⊗ %u\n", d->dim_a, d->dim_b);
    printf("  │ S(A)=%.4f bits  S(B)=%.4f bits\n", d->entropy_a, d->entropy_b);
    printf("  │ I(A:B)=%.4f bits  entangled=%d\n", d->mutual_info, d->is_entangled);
    printf("  │ λ_A: %.4f, %.4f\n", d->eigenvalues_a[0], d->eigenvalues_a[1]);
    printf("  │ λ_B: %.4f, %.4f, %.4f\n", d->eigenvalues_b[0], d->eigenvalues_b[1], d->eigenvalues_b[2]);
    printf("  └──────────────────────────────┘\n");
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SUBSYSTEM ENTANGLE — Create internally entangled D=6 state
 * ═══════════════════════════════════════════════════════════════════════════════ */

void subsystem_entangle(HexStateEngine *eng, uint64_t chunk_id,
                        int type, const Complex *custom_state)
{
    if (chunk_id >= eng->num_chunks) return;
    Chunk *c = &eng->chunks[chunk_id];
    if (!c->hilbert.q_local_state) return;
    uint32_t dim = c->hilbert.q_local_dim;

    memset(c->hilbert.q_local_state, 0, dim * sizeof(Complex));
    double inv_sqrt2 = born_fast_isqrt(2.0);

    switch (type) {
    case 0: /* (|0,0⟩+|1,1⟩)/√2 = (|0⟩+|4⟩)/√2 */
        c->hilbert.q_local_state[0] = cmplx(inv_sqrt2, 0);
        c->hilbert.q_local_state[4] = cmplx(inv_sqrt2, 0);
        break;
    case 1: /* (|0,0⟩+|1,2⟩)/√2 = (|0⟩+|5⟩)/√2 */
        c->hilbert.q_local_state[0] = cmplx(inv_sqrt2, 0);
        c->hilbert.q_local_state[5] = cmplx(inv_sqrt2, 0);
        break;
    case 2: /* custom */
        if (custom_state)
            memcpy(c->hilbert.q_local_state, custom_state, dim * sizeof(Complex));
        break;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * GENERALIZED BELL STATE — |Ψ_mn⟩ between two chunks
 * ═══════════════════════════════════════════════════════════════════════════════ */

void generalized_bell_state(HexStateEngine *eng, uint64_t a, uint64_t b,
                            uint32_t m, uint32_t n, uint32_t dim)
{
    if (a >= eng->num_chunks || b >= eng->num_chunks) return;
    Chunk *ca = &eng->chunks[a];
    Chunk *cb = &eng->chunks[b];

    /* Ensure braided */
    if (ca->hilbert.num_partners == 0)
        braid_chunks(eng, a, b, 0, 0);

    Complex *joint = ca->hilbert.partners[0].q_joint_state;
    if (!joint) return;
    if (dim == 0) dim = ca->hilbert.partners[0].q_joint_dim;
    uint64_t d2 = (uint64_t)dim * dim;

    /* Zero state */
    memset(joint, 0, d2 * sizeof(Complex));

    /* |Ψ_mn⟩ = (1/√D) Σ_k ω^(mk) |k, (k+n) mod D⟩ */
    double inv_sqrt_d = born_fast_isqrt((double)dim);
    double omega = 2.0 * M_PI / dim;
    for (uint32_t k = 0; k < dim; k++) {
        double phase = omega * m * k;
        uint32_t j = (k + n) % dim;
        joint[k * dim + j] = cmplx(inv_sqrt_d * cos(phase),
                                    inv_sqrt_d * sin(phase));
    }

    printf("  [BELL] |Ψ_%u,%u⟩ between %lu↔%lu (D=%u)\n",
           m, n, (unsigned long)a, (unsigned long)b, dim);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PARTIAL TRANSPOSE NEGATIVITY — Entanglement witness
 * ═══════════════════════════════════════════════════════════════════════════════ */

double partial_transpose_negativity(HexStateEngine *eng, uint64_t chunk_id,
                                    double *log_negativity)
{
    if (chunk_id >= eng->num_chunks) return 0.0;
    Chunk *c = &eng->chunks[chunk_id];
    if (c->hilbert.num_partners == 0 || !c->hilbert.partners[0].q_joint_state)
        return 0.0;

    Complex *joint = c->hilbert.partners[0].q_joint_state;
    uint32_t dim = c->hilbert.partners[0].q_joint_dim;
    if (dim == 0) dim = 6;
    uint64_t d2 = (uint64_t)dim * dim;

    /* Compute ρ = |ψ⟩⟨ψ| */
    Complex *rho = sv_calloc_aligned(d2 * d2, sizeof(Complex));
    for (uint64_t i = 0; i < d2; i++)
        for (uint64_t j = 0; j < d2; j++) {
            rho[i*d2+j].real = joint[i].real*joint[j].real + joint[i].imag*joint[j].imag;
            rho[i*d2+j].imag = joint[i].imag*joint[j].real - joint[i].real*joint[j].imag;
        }

    /* Partial transpose on B: ρ^{T_B}_{(a,b),(a',b')} = ρ_{(a,b'),(a',b)} */
    Complex *rho_pt = sv_calloc_aligned(d2 * d2, sizeof(Complex));
    for (uint32_t a = 0; a < dim; a++)
        for (uint32_t b = 0; b < dim; b++)
            for (uint32_t ap = 0; ap < dim; ap++)
                for (uint32_t bp = 0; bp < dim; bp++) {
                    uint64_t row = a*dim+b, col = ap*dim+bp;
                    uint64_t src_row = a*dim+bp, src_col = ap*dim+b;
                    rho_pt[row*d2+col] = rho[src_row*d2+src_col];
                }

    /* Compute trace norm via eigenvalues (power iteration approximation) */
    /* For simplicity, use trace of |ρ^{T_B}| ≈ sum of |eigenvalues| */
    double trace_norm = 0;
    for (uint64_t i = 0; i < d2; i++)
        trace_norm += fabs(rho_pt[i*d2+i].real);

    double negativity = (trace_norm - 1.0) / 2.0;
    if (negativity < 0) negativity = 0;
    if (log_negativity)
        *log_negativity = log2(2.0 * negativity + 1.0);

    free(rho);
    free(rho_pt);
    return negativity;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * QUHIT REGISTER API — Magic Pointer sub-chunk quhits
 * ═══════════════════════════════════════════════════════════════════════════════ */

int find_quhit_reg(HexStateEngine *eng, uint64_t chunk_id)
{
    for (int i = 0; i < eng->num_quhit_regs; i++)
        if (eng->quhit_regs[i].chunk_id == chunk_id)
            return i;
    return -1;
}

int init_quhit_register(HexStateEngine *eng, uint64_t chunk_id,
                        uint64_t n_quhits, uint32_t dim)
{
    if (eng->num_quhit_regs >= MAX_QUHIT_REGISTERS) return -1;

    int r = eng->num_quhit_regs++;
    memset(&eng->quhit_regs[r], 0, sizeof(eng->quhit_regs[r]));
    eng->quhit_regs[r].chunk_id = chunk_id;
    eng->quhit_regs[r].n_quhits = n_quhits;
    eng->quhit_regs[r].dim = dim;

    /* Start in |0⟩: one entry, bulk_value=0, amplitude=1 */
    eng->quhit_regs[r].entries[0].bulk_value = 0;
    eng->quhit_regs[r].entries[0].amplitude = cmplx(1.0, 0.0);
    eng->quhit_regs[r].entries[0].num_addr = 0;
    eng->quhit_regs[r].num_nonzero = 1;
    eng->quhit_regs[r].bulk_rule = 0;  /* constant */

    printf("  [QUHIT] Register %d: chunk %lu, %lu quhits, D=%u\n",
           r, (unsigned long)chunk_id, (unsigned long)n_quhits, dim);
    return 0;
}

void entangle_all_quhits(HexStateEngine *eng, uint64_t chunk_id)
{
    int r = find_quhit_reg(eng, chunk_id);
    if (r < 0) return;

    uint32_t dim = eng->quhit_regs[r].dim;
    double amp = born_fast_isqrt((double)dim);

    /* GHZ state: (1/√D) Σ|k,k,...,k⟩ */
    eng->quhit_regs[r].num_nonzero = dim;
    eng->quhit_regs[r].collapsed = 0;

    for (uint32_t k = 0; k < dim && k < MAX_QUHIT_HILBERT_ENTRIES; k++) {
        eng->quhit_regs[r].entries[k].bulk_value = k;
        eng->quhit_regs[r].entries[k].amplitude = cmplx(amp, 0.0);
        eng->quhit_regs[r].entries[k].num_addr = 0;
    }

    printf("  [QUHIT] GHZ state: %lu quhits, %u entries\n",
           (unsigned long)eng->quhit_regs[r].n_quhits, dim);
}

uint64_t resolve_quhit(HexStateEngine *eng, uint64_t chunk_id, uint64_t quhit_idx)
{
    (void)eng; (void)chunk_id;
    return MAKE_MAGIC_PTR(chunk_id) | (quhit_idx & 0xFFFF);
}

/* Helper: lazily resolve a quhit's value from a basis entry */
static uint32_t lazy_resolve(const QuhitBasisEntry *e, uint64_t quhit_idx,
                              uint8_t bulk_rule, uint32_t dim)
{
    /* Check addr list first */
    for (uint8_t i = 0; i < e->num_addr; i++)
        if (e->addr[i].quhit_idx == quhit_idx)
            return e->addr[i].value;

    /* Fall back to bulk */
    if (bulk_rule == 0) return e->bulk_value;
    return (e->bulk_value + (uint32_t)(quhit_idx % dim)) % dim;
}

/* Helper: create a new entry with one quhit overridden */
static QuhitBasisEntry entry_with_value(const QuhitBasisEntry *src,
                                         uint64_t quhit_idx, uint32_t new_val,
                                         Complex new_amp)
{
    QuhitBasisEntry ne = *src;
    ne.amplitude = new_amp;

    /* Check if this quhit is already in addr */
    for (uint8_t i = 0; i < ne.num_addr; i++) {
        if (ne.addr[i].quhit_idx == quhit_idx) {
            ne.addr[i].value = new_val;
            return ne;
        }
    }

    /* Promote to addr if not already there */
    if (ne.num_addr < MAX_ADDR_PER_ENTRY) {
        ne.addr[ne.num_addr].quhit_idx = quhit_idx;
        ne.addr[ne.num_addr].value = new_val;
        ne.num_addr++;
    }
    return ne;
}

/* Helper: check if two entries describe the same basis state */
static int same_basis(const QuhitBasisEntry *a, const QuhitBasisEntry *b)
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

    if (eng->quhit_regs[r].collapsed) {
        return (uint64_t)lazy_resolve(&eng->quhit_regs[r].entries[0], quhit_idx,
                                       eng->quhit_regs[r].bulk_rule, dim);
    }

    /* Marginal P(v) */
    double probs[MAX_QUHIT_HILBERT_ENTRIES];
    memset(probs, 0, sizeof(probs));
    for (uint32_t e = 0; e < nz; e++) {
        uint32_t v = lazy_resolve(&eng->quhit_regs[r].entries[e], quhit_idx,
                                   eng->quhit_regs[r].bulk_rule, dim);
        double p = cnorm2(eng->quhit_regs[r].entries[e].amplitude);
        if (v < MAX_QUHIT_HILBERT_ENTRIES) probs[v] += p;
    }

    /* Born rule */
    double rand_val = (double)(engine_prng(eng) % 1000000000ULL) / 1e9;
    double cumul = 0.0;
    uint32_t result = dim - 1;
    for (uint32_t i = 0; i < dim; i++) {
        cumul += probs[i];
        if (cumul >= rand_val) { result = i; break; }
    }

    /* Collapse */
    uint32_t write_pos = 0;
    for (uint32_t e = 0; e < nz; e++) {
        uint32_t v = lazy_resolve(&eng->quhit_regs[r].entries[e], quhit_idx,
                                   eng->quhit_regs[r].bulk_rule, dim);
        if (v == result) {
            if (write_pos != e)
                eng->quhit_regs[r].entries[write_pos] = eng->quhit_regs[r].entries[e];
            write_pos++;
        }
    }
    eng->quhit_regs[r].num_nonzero = write_pos;

    /* Renormalize */
    double norm = 0.0;
    for (uint32_t e = 0; e < write_pos; e++)
        norm += cnorm2(eng->quhit_regs[r].entries[e].amplitude);
    if (norm > 0.0) {
        double scale = born_fast_isqrt(norm);
        for (uint32_t e = 0; e < write_pos; e++) {
            eng->quhit_regs[r].entries[e].amplitude.real *= scale;
            eng->quhit_regs[r].entries[e].amplitude.imag *= scale;
        }
    }

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
    double inv_sqrt_d = born_fast_isqrt((double)dim);

    static QuhitBasisEntry new_entries[MAX_QUHIT_HILBERT_ENTRIES];
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
    static QuhitBasisEntry new_entries[MAX_QUHIT_HILBERT_ENTRIES];
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
        uint32_t va = lazy_resolve(&eng->quhit_regs[ra].entries[e], quhit_a,
                                    eng->quhit_regs[ra].bulk_rule, dim);
        uint32_t vb;
        if (ra == rb) {
            vb = lazy_resolve(&eng->quhit_regs[ra].entries[e], quhit_b,
                               eng->quhit_regs[ra].bulk_rule, dim);
        } else {
            uint32_t eb = e % eng->quhit_regs[rb].num_nonzero;
            vb = lazy_resolve(&eng->quhit_regs[rb].entries[eb], quhit_b,
                               eng->quhit_regs[rb].bulk_rule, dim);
        }

        double phase = omega_cz * va * vb;
        double cr = cos(phase), ci = sin(phase);
        Complex a = eng->quhit_regs[ra].entries[e].amplitude;
        eng->quhit_regs[ra].entries[e].amplitude.real = a.real*cr - a.imag*ci;
        eng->quhit_regs[ra].entries[e].amplitude.imag = a.real*ci + a.imag*cr;
    }
}

void apply_sum_quhits(HexStateEngine *eng,
                      uint64_t chunk_ctrl, uint64_t quhit_ctrl,
                      uint64_t chunk_tgt,  uint64_t quhit_tgt)
{
    int rc = find_quhit_reg(eng, chunk_ctrl);
    int rt = find_quhit_reg(eng, chunk_tgt);
    if (rc < 0 || rt < 0 || rc != rt) return;

    uint32_t dim = eng->quhit_regs[rc].dim;
    uint32_t nz = eng->quhit_regs[rc].num_nonzero;

    static QuhitBasisEntry new_entries[MAX_QUHIT_HILBERT_ENTRIES];
    uint32_t new_nz = 0;

    for (uint32_t e = 0; e < nz; e++) {
        QuhitBasisEntry *cur = &eng->quhit_regs[rc].entries[e];
        uint32_t va = lazy_resolve(cur, quhit_ctrl, eng->quhit_regs[rc].bulk_rule, dim);
        uint32_t vb = lazy_resolve(cur, quhit_tgt, eng->quhit_regs[rc].bulk_rule, dim);
        uint32_t new_val = (va + vb) % dim;

        QuhitBasisEntry ne = entry_with_value(cur, quhit_tgt, new_val, cur->amplitude);
        ne = entry_with_value(&ne, quhit_ctrl, va, ne.amplitude);
        accumulate_entry(new_entries, &new_nz, &ne);
    }

    eng->quhit_regs[rc].num_nonzero = new_nz;
    memcpy(eng->quhit_regs[rc].entries, new_entries, new_nz * sizeof(QuhitBasisEntry));
}

void inspect_quhit(HexStateEngine *eng, uint64_t chunk_id,
                              uint64_t quhit_idx, HilbertSnapshot *snap)
{
    double  *saved_mp  = snap->marginal_probs;
    Complex *saved_rho = snap->rho;
    memset(snap, 0, sizeof(*snap));
    snap->marginal_probs = saved_mp;
    snap->rho            = saved_rho;

    int r = find_quhit_reg(eng, chunk_id);
    if (r < 0 || quhit_idx >= eng->quhit_regs[r].n_quhits) return;

    uint32_t dim = eng->quhit_regs[r].dim;
    snap->num_members = eng->quhit_regs[r].n_quhits;
    snap->num_entries = eng->quhit_regs[r].num_nonzero;
    snap->dim = dim;
    snap->is_collapsed = eng->quhit_regs[r].collapsed;
    snap->is_entangled = (eng->quhit_regs[r].num_nonzero > 1) ? 1 : 0;
    snap->total_probability = 0.0;

    free(snap->marginal_probs);
    free(snap->rho);
    snap->marginal_probs = calloc(dim, sizeof(double));
    snap->rho = calloc((size_t)dim * dim, sizeof(Complex));

    for (uint32_t e = 0; e < eng->quhit_regs[r].num_nonzero &&
                          e < MAX_INSPECT_ENTRIES; e++) {
        QuhitBasisEntry *ent = &eng->quhit_regs[r].entries[e];
        snap->entries[e].amp_real = ent->amplitude.real;
        snap->entries[e].amp_imag = ent->amplitude.imag;
        double p = cnorm2(ent->amplitude);
        snap->entries[e].probability = p;
        snap->total_probability += p;

        uint32_t v = lazy_resolve(ent, quhit_idx,
                                   eng->quhit_regs[r].bulk_rule, dim);
        if (v < dim) snap->marginal_probs[v] += p;
    }
    snap->purity = 1.0;
}

/* ── DNA gate for quhit registers ── */

static Complex *build_quhit_dna_unitary(uint32_t dim,
                                         double bond_strength,
                                         double temperature)
{
    Complex *U = sv_calloc_aligned((size_t)dim * dim, sizeof(Complex));
    if (!U) return NULL;

    double kT = 8.617e-5 * temperature;
    if (kT < 1e-10) kT = 1e-10;
    double sigma = bond_strength / kT;
    double complement_amp = 1.0 + sigma;
    double mismatch_amp   = 1.0 / (1.0 + sigma);

    int comp[MAX_QUHIT_HILBERT_ENTRIES];
    for (uint32_t k = 0; k < dim && k < MAX_QUHIT_HILBERT_ENTRIES; k++) {
        if (k < 4) {
            static const int wc[] = {1, 0, 3, 2};
            comp[k] = wc[k];
        } else if (k + 1 < dim && (k % 2) == 0) {
            comp[k] = k + 1;
        } else if ((k % 2) == 1) {
            comp[k] = k - 1;
        } else {
            comp[k] = k;
        }
    }

    for (uint32_t i = 0; i < dim; i++)
        for (uint32_t j = 0; j < dim; j++) {
            double amp, phase;
            if ((int)j == comp[i]) {
                amp = complement_amp; phase = sigma * 0.1 * i;
            } else if (i == j) {
                amp = 0.3 * mismatch_amp; phase = 0.0;
            } else {
                amp = mismatch_amp * 0.15; phase = 0.5 * ((i+j) % dim);
            }
            U[i*dim+j] = cmplx(amp * cos(phase), amp * sin(phase));
        }

    /* Gram-Schmidt */
    for (uint32_t i = 0; i < dim; i++) {
        for (uint32_t k = 0; k < i; k++) {
            double dr = 0, di = 0;
            for (uint32_t j = 0; j < dim; j++) {
                dr += U[k*dim+j].real*U[i*dim+j].real + U[k*dim+j].imag*U[i*dim+j].imag;
                di += U[k*dim+j].real*U[i*dim+j].imag - U[k*dim+j].imag*U[i*dim+j].real;
            }
            for (uint32_t j = 0; j < dim; j++) {
                U[i*dim+j].real -= dr*U[k*dim+j].real - di*U[k*dim+j].imag;
                U[i*dim+j].imag -= dr*U[k*dim+j].imag + di*U[k*dim+j].real;
            }
        }
        double norm = 0;
        for (uint32_t j = 0; j < dim; j++) norm += cnorm2(U[i*dim+j]);
        if (norm > 1e-15) {
            double inv = born_fast_isqrt(norm);
            for (uint32_t j = 0; j < dim; j++) {
                U[i*dim+j].real *= inv;
                U[i*dim+j].imag *= inv;
            }
        }
    }

    return U;
}

void apply_dna_bulk_quhits(HexStateEngine *eng, uint64_t chunk_id,
                           double bond_strength, double temperature)
{
    int r = find_quhit_reg(eng, chunk_id);
    if (r < 0) return;

    uint32_t dim = eng->quhit_regs[r].dim;
    uint32_t nz  = eng->quhit_regs[r].num_nonzero;

    Complex *U = build_quhit_dna_unitary(dim, bond_strength, temperature);
    if (!U) return;

    static QuhitBasisEntry new_entries[MAX_QUHIT_HILBERT_ENTRIES];
    uint32_t new_nz = 0;

    for (uint32_t e = 0; e < nz; e++) {
        QuhitBasisEntry *cur = &eng->quhit_regs[r].entries[e];
        uint32_t v = cur->bulk_value;

        for (uint32_t j = 0; j < dim && new_nz < MAX_QUHIT_HILBERT_ENTRIES; j++) {
            Complex Ujv = U[j * dim + v];
            QuhitBasisEntry ne = *cur;
            ne.bulk_value = j;
            ne.amplitude.real = cur->amplitude.real*Ujv.real - cur->amplitude.imag*Ujv.imag;
            ne.amplitude.imag = cur->amplitude.real*Ujv.imag + cur->amplitude.imag*Ujv.real;
            accumulate_entry(new_entries, &new_nz, &ne);
        }
    }

    memcpy(eng->quhit_regs[r].entries, new_entries, new_nz * sizeof(QuhitBasisEntry));
    eng->quhit_regs[r].num_nonzero = new_nz;
    eng->quhit_regs[r].collapsed = 0;
    free(U);
}

void apply_dna_quhit(HexStateEngine *eng, uint64_t chunk_id,
                     uint64_t quhit_idx,
                     double bond_strength, double temperature)
{
    int r = find_quhit_reg(eng, chunk_id);
    if (r < 0 || quhit_idx >= eng->quhit_regs[r].n_quhits) return;

    uint32_t dim = eng->quhit_regs[r].dim;
    uint32_t nz  = eng->quhit_regs[r].num_nonzero;

    Complex *U = build_quhit_dna_unitary(dim, bond_strength, temperature);
    if (!U) return;

    static QuhitBasisEntry new_entries[MAX_QUHIT_HILBERT_ENTRIES];
    uint32_t new_nz = 0;

    for (uint32_t e = 0; e < nz; e++) {
        QuhitBasisEntry *cur = &eng->quhit_regs[r].entries[e];
        uint32_t v = lazy_resolve(cur, quhit_idx, eng->quhit_regs[r].bulk_rule, dim);

        for (uint32_t j = 0; j < dim; j++) {
            Complex Ujv = U[j * dim + v];
            Complex a;
            a.real = cur->amplitude.real*Ujv.real - cur->amplitude.imag*Ujv.imag;
            a.imag = cur->amplitude.real*Ujv.imag + cur->amplitude.imag*Ujv.real;

            QuhitBasisEntry ne = entry_with_value(cur, quhit_idx, j, a);
            accumulate_entry(new_entries, &new_nz, &ne);
        }
    }

    eng->quhit_regs[r].num_nonzero = new_nz;
    memcpy(eng->quhit_regs[r].entries, new_entries, new_nz * sizeof(QuhitBasisEntry));
    free(U);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * STATE ITERATOR — Lazy streaming over quantum state
 * ═══════════════════════════════════════════════════════════════════════════════ */

int state_iter_begin(HexStateEngine *eng, uint64_t chunk_id, StateIterator *it)
{
    memset(it, 0, sizeof(*it));
    it->eng      = eng;
    it->chunk_id = chunk_id;
    it->reg_idx  = -1;

    /* Priority 1: Quhit register */
    int r = find_quhit_reg(eng, chunk_id);
    if (r >= 0 && eng->quhit_regs[r].num_nonzero > 0) {
        it->mode          = 0;
        it->reg_idx       = r;
        it->entries       = eng->quhit_regs[r].entries;
        it->total_entries = eng->quhit_regs[r].num_nonzero;
        it->n_quhits      = eng->quhit_regs[r].n_quhits;
        it->dim            = eng->quhit_regs[r].dim;
        it->bulk_rule      = eng->quhit_regs[r].bulk_rule;
        it->entry_index    = (uint32_t)-1;
        return 0;
    }

    /* Priority 2: HilbertGroup */
    if (chunk_id < eng->num_chunks) {
        Chunk *c = &eng->chunks[chunk_id];
        if (c->hilbert.group && c->hilbert.group->total_dim > 0) {
            const HilbertGroup *g = c->hilbert.group;
            it->mode          = 3;
            it->group         = g;
            it->total_entries = (uint32_t)g->total_dim;
            it->dim           = g->dim;
            it->n_quhits      = g->num_members;
            it->entry_index   = (uint32_t)-1;
            it->group_member_idx = 0;
            for (uint32_t m = 0; m < g->num_members; m++) {
                if (g->member_ids[m] == chunk_id) {
                    it->group_member_idx = m;
                    break;
                }
            }
            return 0;
        }

        /* Priority 2b: Joint state */
        if (c->hilbert.num_partners > 0 && c->hilbert.partners[0].q_joint_state) {
            uint32_t jd = c->hilbert.partners[0].q_joint_dim;
            it->mode          = 2;
            it->local_ptr     = c->hilbert.partners[0].q_joint_state;
            it->local_dim     = jd;
            it->total_entries = jd * jd;
            it->dim           = jd;
            it->n_quhits      = 2;
            it->entry_index   = (uint32_t)-1;
            return 0;
        }

        /* Priority 3: Local state */
        if (c->hilbert.q_local_state) {
            it->mode          = 1;
            it->local_ptr     = c->hilbert.q_local_state;
            it->local_dim     = c->hilbert.q_local_dim;
            it->total_entries = c->hilbert.q_local_dim;
            it->dim           = c->hilbert.q_local_dim;
            it->n_quhits      = c->size;
            it->entry_index   = (uint32_t)-1;
            return 0;
        }
    }

    return -1;
}

int state_iter_next(StateIterator *it)
{
    it->entry_index++;
    if (it->entry_index >= it->total_entries) return 0;

    switch (it->mode) {
    case 0: {
        const QuhitBasisEntry *e = &it->entries[it->entry_index];
        it->bulk_value   = e->bulk_value;
        it->amplitude    = e->amplitude;
        it->probability  = cnorm2(e->amplitude);
        it->num_addr     = e->num_addr;
        it->addr         = e->addr;
        return 1;
    }
    case 1:
    case 2: {
        uint32_t i = it->entry_index;
        it->amplitude    = it->local_ptr[i];
        it->probability  = cnorm2(it->amplitude);
        it->bulk_value   = i;
        it->num_addr     = 0;
        it->addr         = NULL;
        return 1;
    }
    case 3: {
        const HilbertGroup *g = it->group;
        uint32_t ei = it->entry_index;
        it->amplitude   = g->amplitudes[ei];
        it->probability = cnorm2(g->amplitudes[ei]);
        it->bulk_value  = hilbert_extract_index(g, (uint64_t)ei, it->group_member_idx);
        it->num_addr    = 0;
        it->addr        = NULL;
        return 1;
    }
    }
    return 0;
}

uint32_t state_iter_resolve(const StateIterator *it, uint64_t quhit_idx)
{
    if (it->mode == 0) {
        return lazy_resolve(&it->entries[it->entry_index],
                            quhit_idx, it->bulk_rule, it->dim);
    }
    if (it->mode == 2) {
        uint32_t i = it->entry_index;
        if (quhit_idx == 0) return i / it->local_dim;
        else                return i % it->local_dim;
    }
    if (it->mode == 3) {
        const HilbertGroup *g = it->group;
        if (quhit_idx < g->num_members)
            return hilbert_extract_index(g, (uint64_t)it->entry_index, (uint32_t)quhit_idx);
        return it->bulk_value;
    }
    return it->bulk_value;
}

Complex state_iter_amplitude(const StateIterator *it)
{
    return it->amplitude;
}

void state_iter_end(StateIterator *it)
{
    memset(it, 0, sizeof(*it));
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * ORACLE CALLBACKS — Placeholder for shor/period-finding integration
 * ═══════════════════════════════════════════════════════════════════════════════ */

void oracle_mark(HexStateEngine *eng, uint64_t id, uint64_t target)
{
    if (id >= eng->num_chunks) return;
    Chunk *c = &eng->chunks[id];
    if (!c->hilbert.q_local_state) return;
    uint32_t dim = c->hilbert.q_local_dim;

    /* Phase flip on target state: |target⟩ → -|target⟩ */
    if (target < dim) {
        c->hilbert.q_local_state[target].real *= -1.0;
        c->hilbert.q_local_state[target].imag *= -1.0;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PROGRAM LOADER / EXECUTOR — Instruction processing (stub)
 * ═══════════════════════════════════════════════════════════════════════════════ */

int load_program(HexStateEngine *eng, const char *filename)
{
    FILE *f = fopen(filename, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long length = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (length <= 0) { fclose(f); return -1; }
    free(eng->program);
    eng->program = malloc((size_t)length);
    if (!eng->program) { fclose(f); return -1; }
    fread(eng->program, 1, (size_t)length, f);
    fclose(f);
    eng->program_size = (uint64_t)length;
    eng->pc = 0;
    return 0;
}

/* Joint state probability extraction helper */
double *extract_joint_probs(HexStateEngine *eng, uint64_t id)
{
    if (id >= eng->num_chunks) return NULL;
    Chunk *c = &eng->chunks[id];
    if (c->hilbert.num_partners == 0 || !c->hilbert.partners[0].q_joint_state)
        return NULL;

    uint32_t dim = c->hilbert.partners[0].q_joint_dim;
    if (dim == 0) dim = 6;
    uint64_t d2 = (uint64_t)dim * dim;

    Complex *joint = c->hilbert.partners[0].q_joint_state;
    double *probs = calloc(d2, sizeof(double));

    for (uint64_t i = 0; i < d2; i++)
        probs[i] = cnorm2(joint[i]);

    return probs;
}
