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
#define NUM_BASIS_STATES      2048  /* Increased to support high-D XEB benchmarking */
#define MAX_CHUNK_SIZE        8           /* 6^8 = 1,679,616 states */
#define MAX_STATES_STANDARD   1679616     /* 6^8 */
#define MAX_CHUNKS            16777216
#define INITIAL_CHUNK_CAP     4096        /* Start small, grow as needed */

/* ═══ Sub-Chunk Quhit Addressing ═══
 * Virtual chunk IDs for individually addressable quhits within a chunk.
 * Maps (chunk_id, quhit_idx) → virtual_chunk_id in a reserved range. */
#define QUHIT_VCHUNK_BASE     1024ULL    /* Virtual chunk ID start — low enough for mmap */
#define MAX_QUHIT_REGISTERS   256          /* Max chunks with quhit addressing */
#define MAX_QUHITS_PER_REG    4096         /* Max individually addressable quhits per chunk */
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
#define OP_INSPECT            0x0F    /* Non-destructive Hilbert space readout */
#define OP_SHIFT              0x10
#define OP_REPAIR             0x11
#define OP_NULL               0x14
#define OP_SUBSYSTEM          0x15    /* Sub-system entanglement (D=6=2⊗3) */
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
#define MAX_GROUP_MEMBERS 131072  /* 128K — supports up to 100K+ qudit groups */
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

    /* ═══ Lazy Local Unitaries ═══
     * Instead of composing D×D matrices eagerly (O(D³) per gate),
     * store each operation as a separate D×D matrix in a per-member list.
     * At measurement, replay as matrix-vector products: O(D² × L) total
     * where L = number of ops. This is a factor of D faster than composition.
     *
     * State is implicitly:
     *   |Ψ⟩ = [Π CZ] · Σ_k α_k · ⊗_m (U_L · ... · U_1 |index_{m,k}⟩) */
    Complex  **lazy_U[MAX_GROUP_MEMBERS]; /* Array of D×D matrix pointers per member */
    uint32_t  lazy_count[MAX_GROUP_MEMBERS]; /* Number of ops per member */
    uint32_t  lazy_cap[MAX_GROUP_MEMBERS];   /* Capacity per member */
    uint32_t  num_deferred;     /* Total members with pending ops */
    uint8_t   no_defer;         /* If set, force expansion (used during materialization) */

    /* ═══ Deferred CZ (Non-Local) Gates ═══
     * CZ|j,k⟩ = ω^(j·k)|j,k⟩ where ω = e^(2πi/D)
     * Stored as pairs — NOT materialized.  At measurement time,
     * CZ phases cancel in marginals (|ω|²=1). After sampling
     * outcome v for member a, absorb ω^(v·j_b) into partner b's
     * deferred unitary via diagonal left-multiplication.
     * Cost: O((N+E) × D²) — still polynomial. */
    uint32_t *cz_pairs;         /* Pairs: [a0,b0, a1,b1, ...] (member indices) */
    uint32_t  num_cz;           /* Number of CZ pairs stored */
    uint32_t  cz_cap;           /* Allocated capacity for pairs */
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
#define MAX_BRAID_PARTNERS 2   /* Legacy pairwise (HilbertGroup supersedes this) */
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

    /* ═══ Sub-Chunk Quhit Registers — Pure Magic Pointer Hilbert Space ═══
     *
     * Each register IS a Hilbert space. No Chunk structs per quhit.
     * Magic Pointer for quhit k = (MAGIC_TAG << 48) | (chunk_id << 16) | k
     * The quantum state is stored directly as sparse amplitudes.
     * Operations (measure, DFT, braid) work on the amplitudes. */
#define MAX_QUHIT_HILBERT_ENTRIES 36  /* D² max for 2-member non-GHZ states */
    struct {
        uint64_t  chunk_id;                   /* Parent chunk ID */
        uint64_t  n_quhits;                   /* Number of Magic Pointer quhits */
        uint32_t  dim;                        /* Dimension per quhit (D=6) */
        /* ─── The Hilbert Space ─── */
        uint32_t  num_nonzero;                /* Nonzero amplitude entries */
        uint32_t  entry_value[MAX_QUHIT_HILBERT_ENTRIES]; /* Basis value per entry */
        Complex   amplitudes[MAX_QUHIT_HILBERT_ENTRIES];  /* Complex amplitudes */
        uint8_t   collapsed;                  /* 1 = measurement has occurred */
        uint32_t  collapse_outcome;           /* Determined value (all members) */
        uint64_t  magic_base;                 /* Base Magic Pointer for this register */
    } quhit_regs[MAX_QUHIT_REGISTERS];
    uint32_t        num_quhit_regs;
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
void apply_group_unitary(HexStateEngine *eng, uint64_t id,
                         Complex *U, uint32_t dim);
void apply_local_unitary(HexStateEngine *eng, uint64_t id,
                         const Complex *U, uint32_t dim);
void materialize_deferred(HexStateEngine *eng, HilbertGroup *g);
Complex *lazy_compose(Complex **ops, uint32_t num_ops, uint32_t dim);
void apply_cz_gate(HexStateEngine *eng, uint64_t id_a, uint64_t id_b);
uint64_t measure_chunk(HexStateEngine *eng, uint64_t id);
void grover_diffusion(HexStateEngine *eng, uint64_t id);

/* Entanglement */
void braid_chunks(HexStateEngine *eng, uint64_t a, uint64_t b,
                  uint64_t hexit_a, uint64_t hexit_b);
void braid_chunks_dim(HexStateEngine *eng, uint64_t a, uint64_t b,
                      uint64_t hexit_a, uint64_t hexit_b, uint32_t dim);
void product_state_dim(HexStateEngine *eng, uint64_t a, uint64_t b,
                       uint32_t dim);  /* |0⟩⊗|0⟩ product state in shared Hilbert space */
void unbraid_chunks(HexStateEngine *eng, uint64_t a, uint64_t b);

/* Multiverse operations */
int  op_timeline_fork(HexStateEngine *eng, uint64_t target, uint64_t source);
int  op_infinite_resources(HexStateEngine *eng, uint64_t chunk, uint64_t size);
int  op_infinite_resources_dim(HexStateEngine *eng, uint64_t chunk,
                                uint64_t size, uint32_t dim);

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

/* ═══ Bell inequality test — Hilbert-space-native engine operation ═══
 *
 * The Bell test is OUTSOURCED TO THE HILBERT SPACE:
 *   • Correlators computed analytically from HilbertGroup amplitudes
 *   • Zero statistical error — exact quantum predictions
 *   • CHSH: S = 2√2 (Tsirelson bound, exact)
 *   • CGLMP: I_D from full D-dimensional Hilbert space
 *   • Optional empirical confirmation via Born-rule sampling
 * ═══════════════════════════════════════════════════════════════════ */

/* ── Deferred unitary management (direct Hilbert space manipulation) ── */

/* Set a D×D unitary as the deferred measurement operator for member_idx.
 * Does NOT modify the base state. The Hilbert space holds the math. */
void hilbert_set_deferred(HilbertGroup *g, uint32_t member_idx,
                          const Complex *U, uint32_t dim);

/* Clear (free) the deferred unitary for member_idx. */
void hilbert_clear_deferred(HilbertGroup *g, uint32_t member_idx);

/* ═══ N-Party Mermin Inequality — D-dimensional native ═══════════════════
 *
 * The Mermin inequality is the correct N-party generalization of CHSH.
 * For an N-party GHZ state |GHZ⟩ = (1/√D) Σ_k |k⟩^N :
 *
 *   Z-test: all N parties measure in computational basis → all agree
 *   X-test: all N parties measure in DFT_D basis → Σ outcomes ≡ 0 mod D
 *
 * No classical state can satisfy BOTH:
 *   W = P(Z agree) + P(X parity) - 1
 *   Classical: W ≤ 1/D     Quantum GHZ: W = 1.0
 *
 * Invocable via mermin_test() engine function.
 * ═══════════════════════════════════════════════════════════════════════ */

/* Result from an N-party Mermin inequality test */
typedef struct {
    /* ── Core results ── */
    double   pz;                /* P(all agree in Z-basis) */
    double   px;                /* P(Σ ≡ 0 mod D in X-basis) */
    double   witness;           /* W = pz + px - 1 */
    double   classical_bound;   /* 1/D */

    /* ── Metadata ── */
    uint32_t n_parties;         /* Number of registers in GHZ */
    uint32_t dim;               /* Hilbert space dimension (D) */
    uint32_t n_shots;           /* Shots per test */
    int      violation;         /* 1 if W > 1/D */
    double   elapsed_ms;        /* Wall-clock time */

    /* ── Z-basis value distribution ── */
    int      z_counts[NUM_BASIS_STATES]; /* Per-value counts from Z-test */
} MerminResult;

/* Run an N-party Mermin inequality test.
 * Creates n_parties registers (100T quhits each), star-topology GHZ,
 * then runs Z-test and X-test with n_shots each.
 * Returns MerminResult with witness W and violation flag. */
MerminResult mermin_test(HexStateEngine *eng, uint32_t n_parties,
                         uint32_t n_shots);

/* Print formatted Mermin test results */
void mermin_test_print(MerminResult *r);

/* ── DFT_D convenience function ──
 * Apply the D-dimensional Discrete Fourier Transform to a chunk.
 * This is the generalized Hadamard for D>2:
 *   F[j,k] = (1/√D) ω^(jk),  ω = e^(2πi/D)
 * Works on both shadow and Hilbert space states. */
void apply_dft(HexStateEngine *eng, uint64_t chunk_id, uint32_t dim);

/* ═══ Hilbert Space Inspector — non-destructive state extraction ═══
 *
 * CONTROVERSIAL: In real quantum mechanics, you CANNOT read the
 * quantum state without collapsing it. But in the HexState Engine,
 * the Hilbert space IS memory. Magic Pointers literally point to
 * the amplitudes. We can just... read them.
 *
 * These functions read the full quantum state on demand:
 * amplitudes, phases, probabilities, entanglement structure —
 * all WITHOUT collapsing or modifying the state.
 *
 * Invocable via OP_INSPECT (0x0F) instruction.
 * ═══════════════════════════════════════════════════════════════ */

#define MAX_INSPECT_ENTRIES 256

#define MAX_SNAP_MEMBERS 16   /* Max members to track in snapshot */

/* A single entry in the state decomposition */
typedef struct {
    uint32_t  indices[MAX_SNAP_MEMBERS]; /* Basis index per member */
    double    amp_real;                   /* Re(α) */
    double    amp_imag;                   /* Im(α) */
    double    probability;               /* |α|² */
    double    phase_rad;                 /* arg(α) in radians */
} StateEntry;

/* Complete snapshot of a Hilbert space — read without collapse */
typedef struct {
    /* Identity */
    uint64_t  chunk_id;            /* Which register was inspected */
    uint32_t  dim;                 /* D (per-register dimension) */
    uint32_t  num_members;         /* How many registers share this state */
    uint64_t  member_ids[MAX_SNAP_MEMBERS];

    /* Full state decomposition */
    uint32_t  num_entries;         /* Number of non-zero amplitudes */
    StateEntry entries[MAX_INSPECT_ENTRIES];

    /* Aggregate statistics */
    double    total_probability;   /* Should be 1.0 for normalized state */
    double    purity;              /* Tr(ρ²) — 1.0 = pure, 1/D = maximally mixed */
    double    entropy;             /* Von Neumann entropy S(ρ_A) for this register */
    int       is_entangled;        /* 1 if non-zero entropy with partners */
    int       is_collapsed;        /* 1 if measurement has occurred */

    /* Marginal probabilities for the inspected register */
    double    marginal_probs[NUM_BASIS_STATES]; /* P(k) for k=0..D-1 */

    /* Reduced density matrix for inspected register: dim × dim complex */
    Complex   rho[NUM_BASIS_STATES * NUM_BASIS_STATES];
} HilbertSnapshot;

/* Non-destructive readout: extract full state without collapse.
 * This is what quantum mechanics says you CANNOT DO.
 * We do it anyway because the Hilbert space is memory. */
HilbertSnapshot inspect_hilbert(HexStateEngine *eng, uint64_t chunk_id);

/* Print formatted inspection results */
void inspect_print(HilbertSnapshot *snap);

/* Compute von Neumann entanglement entropy between chunk_id and
 * its partners. Returns S = -Tr(ρ_A log₂ ρ_A). S > 0 = entangled. */
double hilbert_entanglement_entropy(HexStateEngine *eng, uint64_t chunk_id);

/* ═══ Sub-System Entanglement — D=6 = 2⊗3 Novel Capability ═══════════════
 *
 * D=6 is the SMALLEST dimension with THREE unique capabilities:
 *   1. Internal entanglement (qubit⊗qutrit within a single register)
 *   2. 36 generalized Bell states (vs 4 for qubits)
 *   3. Bound entanglement (PPT entangled states)
 *
 * The decomposition maps: |k⟩ → |k/3⟩_qubit ⊗ |k%3⟩_qutrit
 *   |0⟩=|0,0⟩  |1⟩=|0,1⟩  |2⟩=|0,2⟩  |3⟩=|1,0⟩  |4⟩=|1,1⟩  |5⟩=|1,2⟩
 * ═══════════════════════════════════════════════════════════════════════ */

/* Sub-system decomposition result */
typedef struct {
    uint32_t dim_a, dim_b;     /* Factor dimensions (2, 3 for D=6) */
    Complex  rho_a[4];         /* 2×2 reduced density matrix (qubit) */
    Complex  rho_b[9];         /* 3×3 reduced density matrix (qutrit) */
    double   entropy_a;        /* von Neumann entropy of qubit subsystem (bits) */
    double   entropy_b;        /* von Neumann entropy of qutrit subsystem (bits) */
    double   mutual_info;      /* Mutual information I(A:B) (bits) */
    int      is_entangled;     /* 1 if subsystems are internally entangled */
    double   eigenvalues_a[2]; /* Eigenvalues of ρ_A */
    double   eigenvalues_b[3]; /* Eigenvalues of ρ_B */
} SubSystemDecomp;

/* Decompose D=6 register into qubit(2)⊗qutrit(3) subsystems.
 * Reads amplitudes from HilbertGroup, computes partial traces,
 * eigenvalues, and von Neumann entropy for each subsystem. */
SubSystemDecomp subsystem_decompose(HexStateEngine *eng, uint64_t chunk_id);
void subsystem_decompose_print(SubSystemDecomp *d);

/* Create internally entangled state within a single D=6 register.
 *   type 0: (|0,0⟩+|1,1⟩)/√2 — maximally entangled (qubit-qutrit Bell)
 *   type 1: (|0,0⟩+|1,2⟩)/√2 — maximally entangled (shifted Bell)
 *   type 2: custom state (pass 6-component Complex vector) */
void subsystem_entangle(HexStateEngine *eng, uint64_t chunk_id,
                        int type, const Complex *custom_state);

/* Create generalized Bell state |Ψ_mn⟩ between two chunks:
 *   |Ψ_mn⟩ = (1/√D) Σ_k ω^(mk) |k, (k+n) mod D⟩
 *   For D=6: 36 unique maximally entangled states (m,n ∈ 0..5)
 *   Superdense coding: log₂(36) = 5.17 bits/pair (vs 2 for qubits) */
void generalized_bell_state(HexStateEngine *eng, uint64_t a, uint64_t b,
                            uint32_t m, uint32_t n, uint32_t dim);

/* Compute partial transpose negativity (entanglement witness).
 *   Returns negativity N ≥ 0.  N > 0 ⟹ NPT ⟹ definitely entangled.
 *   Stores log-negativity E_N = log₂(2N+1) in *log_negativity if non-NULL.
 *   For D=6 Bell state: N = 2.5, E_N = 2.58 */
double partial_transpose_negativity(HexStateEngine *eng, uint64_t chunk_id,
                                    double *log_negativity);

/* ═══ Sub-Chunk Quhit API — Pure Magic Pointer Hilbert Space ══════════════
 *
 * Each quhit register IS a Hilbert space. No Chunk structs per quhit.
 * Magic Pointer for quhit k = (MAGIC_TAG << 48) | (chunk_id << 16) | k
 * The quantum state is stored as sparse amplitudes in the register.
 * Operations (measure, DFT, braid) work directly on the amplitudes.
 *
 * Usage:
 *   init_quhit_register(eng, 0, 100000000, 6);  // 100M Magic Pointer quhits
 *   entangle_all_quhits(eng, 0);                 // GHZ state across all
 *   uint64_t val = measure_quhit(eng, 0, 42);    // Born rule on quhit 42
 *   // ALL 100M quhits now collapsed to val
 * ═══════════════════════════════════════════════════════════════════════ */

/* Initialize N individually addressable quhits as Magic Pointers.
 * Each quhit starts in |0⟩. State stored in Hilbert space (not per-Chunk). */
int init_quhit_register(HexStateEngine *eng, uint64_t chunk_id,
                        uint64_t n_quhits, uint32_t dim);

/* Entangle ALL quhits in a register into a GHZ state:
 * |GHZ⟩ = (1/√D) Σ_k |k,k,...,k⟩
 * Writes the entanglement directly to the Hilbert space. */
void entangle_all_quhits(HexStateEngine *eng, uint64_t chunk_id);

/* Resolve (chunk_id, quhit_idx) → Magic Pointer address.
 * Returns the Magic Pointer for a specific quhit. */
uint64_t resolve_quhit(HexStateEngine *eng, uint64_t chunk_id, uint64_t quhit_idx);

/* Measure a specific quhit — Born rule on the Hilbert space.
 * Collapses that quhit and all entangled partners. */
uint64_t measure_quhit(HexStateEngine *eng, uint64_t chunk_id, uint64_t quhit_idx);

/* Apply DFT to a specific quhit's contribution to the Hilbert space. */
void apply_dft_quhit(HexStateEngine *eng, uint64_t chunk_id,
                     uint64_t quhit_idx, uint32_t dim);

/* Apply a custom D×D unitary to a specific quhit. */
void apply_unitary_quhit(HexStateEngine *eng, uint64_t chunk_id,
                         uint64_t quhit_idx, const Complex *U, uint32_t dim);

/* Entangle two specific quhits (within same chunk or across chunks).
 * Creates Bell state |Ψ⟩ = Σ|k,k⟩/√D in their joint Hilbert space. */
void braid_quhits(HexStateEngine *eng,
                  uint64_t chunk_a, uint64_t quhit_a,
                  uint64_t chunk_b, uint64_t quhit_b,
                  uint32_t dim);

/* Apply CZ gate between two specific quhits. */
void apply_cz_quhits(HexStateEngine *eng,
                     uint64_t chunk_a, uint64_t quhit_a,
                     uint64_t chunk_b, uint64_t quhit_b);

/* Non-destructive inspection of a specific quhit's Hilbert space. */
HilbertSnapshot inspect_quhit(HexStateEngine *eng, uint64_t chunk_id,
                              uint64_t quhit_idx);

#endif /* HEXSTATE_ENGINE_H */
