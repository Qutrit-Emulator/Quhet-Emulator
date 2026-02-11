/*
 * ═══════════════════════════════════════════════════════════════════════════════
 * HEXSTATE ENGINE — 6-State Quantum Processor Emulator
 * ═══════════════════════════════════════════════════════════════════════════════
 * C port of the Qutrit Engine with 6 basis states (|0⟩ – |5⟩).
 * All chunks reference an external Hilbert space via Magic Pointers.
 * Supports 4096-bit BigInt arithmetic.
 *
 * Magic Pointer Tag: 0x4858 ("HX" — HexState)
 */

#ifndef HEXSTATE_ENGINE_H
#define HEXSTATE_ENGINE_H

#include <stdint.h>
#include <stddef.h>
#include "bigint.h"

/* ─── Configuration Constants ─────────────────────────────────────────────── */
#define NUM_BASIS_STATES      6
#define MAX_CHUNK_SIZE        8           /* 6^8 = 1,679,616 states */
#define MAX_STATES_STANDARD   1679616     /* 6^8 */
#define MAX_CHUNKS            16777216
#define INITIAL_CHUNK_CAP     4096        /* Start small, grow as needed */
#define MAX_BRAID_LINKS       16777216
#define MAX_ADDONS            32
#define CAUSAL_SAFEGUARD      4096

/* Magic Pointer tag for Hilbert space references */
#define MAGIC_TAG             0x4858ULL   /* "HX" */
#define MAGIC_MASK            0xFFFF000000000000ULL
#define MAKE_MAGIC_PTR(id)    ((MAGIC_TAG << 48) | ((uint64_t)(id) & 0xFFFFFFFFFFFFULL))
#define IS_MAGIC_PTR(ptr)     (((ptr) >> 48) == MAGIC_TAG)
#define MAGIC_PTR_ID(ptr)     ((ptr) & 0xFFFFFFFFFFFFULL)

/* State encoding: 6 basis states × complex amplitude */
#define HEXIT_SIZE            96          /* 6 × 16 bytes (6 complex doubles) */
#define STATE_BYTES           16          /* Complex amplitude: 8 (real) + 8 (imag) */

/* ─── Opcode Definitions ──────────────────────────────────────────────────── */
/* Core (Epoch 1) */
#define OP_NOP                0x00
#define OP_INIT               0x01
#define OP_SUP                0x02
#define OP_HADAMARD           0x03
#define OP_PHASE              0x04
#define OP_CPHASE             0x05
#define OP_SWAP               0x06
#define OP_MEASURE            0x07
#define OP_GROVER             0x08
#define OP_BRAID              0x09
#define OP_UNBRAID            0x0A
#define OP_ORACLE             0x0B
#define OP_ADDON              0x0C
#define OP_PRINT_STATE        0x0D
#define OP_BELL_TEST          0x0E
#define OP_SHIFT              0x10
#define OP_REPAIR             0x11
#define OP_NULL               0x14
#define OP_SUMMARY            0x1F
#define OP_HALT               0xFF

/* Epoch 11: Multiverse Horizon */
#define OP_TIMELINE_FORK      0xA8
#define OP_INFINITE_RESOURCES 0xA9
#define OP_DIMENSIONAL_PEEK   0x5A
#define OP_ENTROPY_SIPHON     0x44

/* Epoch 2 */
#define OP_MIRROR_VOID        0x32
#define OP_SHIFT_REALITY      0x3C
#define OP_REPAIR_CAUSALITY   0x42

/* Phase 6: Omega */
#define OP_ENTROPY_REVERSE    0x5E
#define OP_QUANTUM_TUNNEL     0x78
#define OP_FINAL_ASCENSION    0xF2
#define OP_SIREN_SONG         0x72

/* ─── Data Structures ─────────────────────────────────────────────────────── */

typedef struct {
    double real;
    double imag;
} Complex;

/* ─── Shared multi-party Hilbert space ───
 * When registers are braided, they join the SAME group and share a
 * single state vector. The Hilbert space IS the shared memory —
 * collapse is automatic because all members read from it.
 * Sparse representation: for GHZ states, only D entries regardless
 * of group size. */
#define MAX_GROUP_MEMBERS 1024
typedef struct HilbertGroup {
    uint32_t  dim;              /* Per-register dimension (6) */
    uint32_t  num_members;      /* How many registers share this state */
    uint64_t  member_ids[MAX_GROUP_MEMBERS]; /* Chunk IDs in order */
    /* Sparse state: only store nonzero amplitudes */
    uint32_t  num_nonzero;      /* Number of nonzero amplitude entries */
    uint32_t  sparse_cap;       /* Allocated capacity */
    uint32_t *basis_indices;    /* Flattened: num_nonzero × num_members indices */
    Complex  *amplitudes;       /* num_nonzero amplitudes */
    uint8_t   collapsed;        /* 1 if a measurement has collapsed this group */
} HilbertGroup;

/* Hilbert Space Reference (Magic Pointer) */
typedef struct {
    uint64_t  magic_ptr;          /* (MAGIC_TAG << 48) | chunk_id */
    Complex  *shadow_state;       /* Local mmap'd shadow cache (may be NULL for infinite) */
    uint64_t  shadow_capacity;    /* Number of states in shadow cache */
    /* ─── Quantum state stored AT this Magic Pointer address ───
     * The Hilbert space is the computation substrate.
     * We WRITE state and transformations to it, READ results from it. */
    uint8_t   q_flags;            /* bit 0 = superposed, bit 1 = measured */
    /* ─── Local single-particle Hilbert space ─── */
    Complex  *q_local_state;      /* Local D-dimensional state vector (D amplitudes) */
    uint32_t  q_local_dim;        /* Dimension of local state (default 6) */
    /* ─── Shared multi-party Hilbert space group ─── */
    HilbertGroup *group;          /* Shared state (NULL if not in a group) */
    uint32_t  group_index;        /* This register's position within the group */
    /* ─── Joint quantum state (pairwise fallback) ─── */
#define MAX_BRAID_PARTNERS 1024
    struct {
        Complex  *q_joint_state;  /* Shared 2-particle state: dim² amplitudes */
        uint32_t  q_joint_dim;    /* Dimension of joint state (default 6) */
        uint64_t  q_partner;      /* Partner chunk ID */
        uint8_t   q_which;        /* 0 = A side, 1 = B side of joint state */
    } partners[MAX_BRAID_PARTNERS];
    uint16_t  num_partners;       /* Number of active braid partners */
} HilbertRef;

/* Chunk: logical unit of quantum state */
typedef struct {
    uint64_t   id;
    uint64_t   size;              /* Number of hexits */
    uint64_t   num_states;        /* 6^size (or INT64_MAX for infinite) */
    HilbertRef hilbert;           /* External Hilbert space reference */
    uint8_t    locked;            /* 1 = immutable */
} Chunk;

/* Braid link (entanglement) */
typedef struct {
    uint64_t chunk_a;
    uint64_t chunk_b;
    uint64_t hexit_a;
    uint64_t hexit_b;
} BraidLink;

/* Parallel Reality tracking */
typedef struct {
    uint64_t  reality_id;
    uint64_t  parent_chunk;
    uint64_t  divergence;
    void     *hw_context;         /* mmap'd hardware context */
    uint8_t   active;
} ParallelReality;

/* ─── Oracle Registry ─────────────────────────────────────────────────────── */

#define MAX_ORACLES  256

/* Built-in oracle IDs */
#define ORACLE_PHASE_FLIP      0x00  /* Phase-flip state 0 */
#define ORACLE_SEARCH_MARK     0x01  /* Mark target state(s) for Grover search */
#define ORACLE_PERIOD_FIND     0x02  /* Shor's period-finding (modular exp) */
#define ORACLE_GROVER_MULTI    0x03  /* Grover with multiple marked states */
#define ORACLE_CUSTOM_BASE     0x10  /* User-registered oracles start here */

struct HexStateEngine_s;  /* Forward declaration */

/* Oracle function signature:
 *   eng       — engine pointer (full access to all state)
 *   chunk_id  — target chunk
 *   user_data — arbitrary user context passed at registration
 */
typedef void (*OracleFunc)(struct HexStateEngine_s *eng,
                           uint64_t chunk_id, void *user_data);

typedef struct {
    const char  *name;       /* Human-readable name */
    OracleFunc   func;       /* Implementation */
    void        *user_data;  /* Arbitrary context (owned by caller) */
    uint8_t      active;     /* 1 = registered */
} OracleEntry;

/* ─── Engine State ────────────────────────────────────────────────────────── */

typedef struct HexStateEngine_s {
    /* Chunks (dynamically allocated via mmap) */
    Chunk           *chunks;
    uint64_t        num_chunks;
    uint64_t        chunk_capacity;

    /* Braid links */
    BraidLink      *braid_links;
    uint64_t        num_braid_links;
    uint64_t        braid_capacity;

    /* Parallel realities (dynamically allocated, same capacity as chunks) */
    ParallelReality *parallel;
    uint64_t        next_reality_id;

    /* Measurement results (dynamically allocated, same capacity as chunks) */
    uint64_t        *measured_values;

    /* Oracle registry */
    OracleEntry     oracles[MAX_ORACLES];
    uint32_t        num_oracles_registered;

    /* PRNG state (Pi-seeded) */
    uint64_t        prng_state;

    /* Program execution */
    uint8_t        *program;
    uint64_t        program_size;
    uint64_t        pc;           /* Program counter (byte offset) */
    int             running;

    /* BigInt workspace */
    BigInt          bigint_temp[3];
} HexStateEngine;

/* ─── Instruction Word Layout ─────────────────────────────────────────────── */
/* 64-bit instruction:
 *   [63:56]  Op2     (8 bits)
 *   [55:32]  Op1     (24 bits)
 *   [31:8]   Target  (24 bits)
 *   [7:0]    Opcode  (8 bits)
 */
typedef struct {
    uint8_t  opcode;
    uint32_t target;    /* 24-bit chunk/target index */
    uint32_t op1;       /* 24-bit operand 1 */
    uint8_t  op2;       /* 8-bit operand 2 */
} Instruction;

/* ─── API ─────────────────────────────────────────────────────────────────── */

/* Lifecycle */
int  engine_init(HexStateEngine *eng);
void engine_destroy(HexStateEngine *eng);

/* Chunk operations */
int  init_chunk(HexStateEngine *eng, uint64_t id, uint64_t num_hexits);
void create_superposition(HexStateEngine *eng, uint64_t id);
void apply_hadamard(HexStateEngine *eng, uint64_t id, uint64_t hexit_index);
uint64_t measure_chunk(HexStateEngine *eng, uint64_t id);
void grover_diffusion(HexStateEngine *eng, uint64_t id);

/* Entanglement */
void braid_chunks(HexStateEngine *eng, uint64_t a, uint64_t b,
                  uint64_t hexit_a, uint64_t hexit_b);
void braid_chunks_dim(HexStateEngine *eng, uint64_t a, uint64_t b,
                      uint64_t hexit_a, uint64_t hexit_b, uint32_t dim);
void unbraid_chunks(HexStateEngine *eng, uint64_t a, uint64_t b);

/* Multiverse operations */
int  op_timeline_fork(HexStateEngine *eng, uint64_t target, uint64_t source);
int  op_infinite_resources(HexStateEngine *eng, uint64_t chunk, uint64_t size);

/* Oracle registry */
int  oracle_register(HexStateEngine *eng, uint32_t oracle_id,
                     const char *name, OracleFunc func, void *user_data);
void oracle_unregister(HexStateEngine *eng, uint32_t oracle_id);
void oracle_list(HexStateEngine *eng);
void execute_oracle(HexStateEngine *eng, uint64_t chunk_id, uint32_t oracle_id);
void register_builtin_oracles(HexStateEngine *eng);

/* Program execution */
int  load_program(HexStateEngine *eng, const char *filename);
Instruction decode_instruction(uint64_t raw);
int  execute_program(HexStateEngine *eng);
int  execute_instruction(HexStateEngine *eng, Instruction instr);

/* Self-test */
int  run_self_test(HexStateEngine *eng);

/* Shor's factoring (CLI entrypoint) */
int  run_shor_factoring(HexStateEngine *eng, const char *n_decimal);

/* Utility */
uint64_t engine_prng(HexStateEngine *eng);
void     print_chunk_state(HexStateEngine *eng, uint64_t id);

/* ═══ Hilbert space readout — works at any D ═══ */

/* Read joint probability P(a,b) directly from |ψ|² in the Hilbert space.
 * Returns malloc'd array of dim*dim doubles. Caller frees.
 * dim is read from q_joint_dim (whatever D the state was braided at). */
double *hilbert_read_joint_probs(HexStateEngine *eng, uint64_t id);

/* ═══ Bell inequality tools ═══ */

typedef struct { double theta_A; double theta_B; } BellPhaseCtx;

/* Apply measurement-setting–dependent phase rotation to joint state */
void bell_phase_oracle(HexStateEngine *eng, uint64_t id, BellPhaseCtx *ctx);

/* Compute CGLMP I_D Bell inequality from 4 probability tables.
 * Classical bound: I_D ≤ 2.  Quantum violation: I_D > 2.
 * Works at any D. */
double hilbert_compute_cglmp(double *P00, double *P01, double *P10, double *P11,
                             uint32_t dim);

/* One-shot Bell test: braid at dim D, apply phases, QFT, read I_D.
 * Returns I_D > 0 for violation. Works at any D.
 * Needs oracle 0xBE registered for phase injection. */
double hilbert_bell_test(HexStateEngine *eng, double alpha, double beta, uint32_t dim);

#endif /* HEXSTATE_ENGINE_H */
