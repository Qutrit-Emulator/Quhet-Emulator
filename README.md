<p align="center">
HEAP COMPILE. Otherwise you will get segfault!
</p>

# HexState 3.6

**A D=6 hexagonal quantum simulator built on the Triality Quhit.**

HexState operates in a native 6-state basis instead of the conventional 2-state qubit. Its primary quantum primitive is the **Triality Quhit** — a geometric object that simultaneously exists in five mutually-defining views (Edge, Vertex, Diagonal, Folded, Exotic). Gates automatically execute in whichever view is cheapest. States breathe between flat O(1) representations and full quantum form as needed.

Memory scales as O(N + P) — polynomial in the number of quhits and pairs — never exponential. Built entirely in standalone C99. No external dependencies beyond `libm`. OpenMP parallelism where available, SSE2 intrinsics where beneficial.

---

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                    Applications                        │
│   MIPT Sweep · QCD Simulation · Entangled Factoring    │
├────────────────────────────────────────────────────────┤
│              Tensor Network Overlays                   │
│   MPS (1D) · PEPS (2D) · TNS (3D · 4D · 5D · 6D)     │
├────────────────────────────────────────────────────────┤
│              Vesica Fold Factorization                 │
│   Geometric SVD replacement for symmetric states       │
│   Lossless · O(1) for product states · No Jacobi       │
├────────────────────────────────────────────────────────┤
│     ★ Triality Quhit — Primary Quantum Primitive ★     │
│   5-view lazy conversion · Flat auto-promote/demote    │
│   Lazy gate chains · SVD spectrum oracle               │
│   S₆ outer automorphism · 15-syntheme omnidirectional  │
├────────────────────────────────────────────────────────┤
│              Foundation Layer (internal)                │
│   DFT₆ · CZ · Born-rule · Registers · Substrate       │
│   Self-calibration · BigInt · Gauss sums               │
└────────────────────────────────────────────────────────┘
```

---

## The Triality Quhit

The Triality Quhit (`quhit_triality.h` / `quhit_triality.c`) is the primary quantum primitive in HexState. It is based on the CMY geometric principle: three mutually-defining views where each view's structure *is* the other views' structure in a different role.

Every quhit stores its state simultaneously in five views with lazy conversion:

| View | Basis | Cheap operations |
|------|-------|-----------------|
| **Edge** | Computational | Phase gates O(D) |
| **Vertex** | Fourier (DFT₆) | Shift gates O(D) |
| **Diagonal** | Conjugate Fourier | Conjugate ops O(D) |
| **Folded** | Antipodal fold | Vesica/wave decomposition |
| **Exotic** | Syntheme-parameterized | S₆ outer automorphism |

Gates automatically execute in their cheapest view with lazy conversion between views. Average gate cost: O(12) instead of O(36). Enhancement flags track eigenstates, active masks, and real-valuedness for further fast-paths.

### Flat Quhit (`flat_quhit.h`)

Three-tier representation with automatic promotion and demotion:

- **`FLAT_BASIS`**: Single basis state |k⟩ with phase. X, Z, CZ, measure all O(1).
- **`FLAT_SUBSPACE`**: 2–3 active states. Operations O(active_count²).
- **`QUANTUM_FULL`**: Full TrialityQuhit. All operations O(D) or O(D²).

Promotion occurs when a gate creates complexity (e.g., DFT on a basis state). Demotion occurs when the result collapses to a structurally simple form (e.g., after measurement).

### Lazy Gate Chains (`quhit_lazy.h` / `quhit_lazy.c`)

Every gate is *recorded*, not executed. The chain accumulates as a single compressed 6×6 unitary. Only measurement forces resolution. N gates resolve in one matrix-vector multiply instead of N. Throughput: **>10 million gates/sec**.

Fast paths: diagonal∘diagonal = O(6), DFT∘IDFT = identity (skip), X⁶ = identity (skip).

### SVD Spectrum Oracle (`quhit_svd_gate.h` / `quhit_svd_gate.c`)

The SVD is the most expensive operation in tensor network contraction. The oracle inspects the *gate log* — which gates produced the current bond state — and returns the SVD result analytically when the pattern is recognized:

- **Identity** → skip entirely
- **DFT₆ on Bell pair** → Pattern A (identity spectrum)
- **CZ on |+⟩⊗|+⟩** → Pattern B (rank-3 paired)
- **Phase gate** → diagonal SVD

This is exact, not an approximation. O(1) instead of O(n³).

### S₆ Outer Automorphism (`s6_exotic.h` / `s6_exotic.c`)

S₆ is the *only* symmetric group possessing a non-trivial outer automorphism. HexState implements the full automorphism φ over all 720 permutations, the 15 synthemes (partitions of {0,…,5} into 3 unordered pairs), and the 6 synthematic totals. The omnidirectional fold enables lossless basis changes parameterized by any of the 15 synthemes, not just the default antipodal pairing {(0,3),(1,4),(2,5)}.

### Triadic 3-Body Entanglement (`quhit_triadic.h` / `quhit_triadic.c`)

Native 3-body entanglement primitives: CMY Bell states, GHZ triples, and product triples, with marginal computation for each party.

---

## Vesica Fold Factorization (`tensor_svd.h`)

A geometric replacement for numerical SVD in the tensor network overlays. The Vesica fold pairs physical indices (k, k+3) into convergent (vesica) and divergent (wave) components:

```
vesica[j] = (ψ[k] + ψ[k+3]) / √2    (symmetric)
wave[j]   = (ψ[k] − ψ[k+3]) / √2    (antisymmetric)
```

Three factorization paths:

| Path | Condition | Method | Cost |
|------|-----------|--------|------|
| **Vesica Direct** | wave < 1% | Per-pair geometric block decomposition — no SVD | O(9) for product states |
| **Vesica + miniSVD** | wave ≥ 1% | SVD on the folded (4× smaller) matrix | ~75% savings |
| **Bypass** | D ≠ 6 | Standard Jacobi SVD | Full |

For the common case of antipodal-symmetric states, the fold provides an exact, lossless factorization with zero numerical SVD calls. Scalar blocks (1×1) use trivial arithmetic; 2×2 blocks use analytic closed-form SVD; larger blocks use tiny Jacobi on matrices typically 3×3 to 8×8.

---

## Tensor Network Overlays

All overlays use a **Magic Pointer** architecture — no classical dense tensor arrays. Each site's register directly encodes a multi-index state |k, α, β, …⟩ into the core engine.

| Tier | File | Topology | Typical grid | Bond encoding |
|------|------|----------|-------------|---------------|
| **1D MPS** | `mps_overlay.c/.h` | Chain | L sites | |k, α, β⟩ with χ=256 |
| **2D PEPS** | `peps_overlay.c/.h` | Square lattice | Nx × Ny | |k, α, β, γ, δ⟩ (4 bonds) |
| **3D TNS** | `peps3d_overlay.c/.h` | Cubic | X × Y × Z | 6 bonds per site |
| **4D TNS** | `peps4d_overlay.c/.h` | Tesseractic | Hypercubic | 8 bonds per site |
| **5D TNS** | `peps5d_overlay.c/.h` | 5-orthoplex | 2⁵ grids | 10 bonds per site |
| **6D TNS** | `peps6d_overlay.c/.h` | 6D hexeractic | 2⁶ grids | 12 bonds per site |

Each overlay provides:
- Product state initialization
- 1-site gate application (physical index)
- 2-site bond gate application (with SVD/Vesica truncation)
- Local density readout
- Triality sidecar integration via `triality_overlay.h`

### Triality Overlay Integration (`triality_overlay.h`)

Every site carries a `TriOverlaySite` with a `TrialityQuhit` as its physical representation. Gates route through optimal views. Basis-state sites detected at O(1) permit diagonal 2-site gates to bypass the full Θ-matrix + SVD pipeline.

---

## Dynamic Lattice Growth

### Entropy-Driven Breathing (`quhit_peps_grow.h` / `quhit_peps_grow.c`)

Sites are classified as **ACTIVE**, **BOUNDARY**, or **DORMANT** based on entanglement entropy. The lattice grows toward entanglement and contracts from emptiness — like a living system tracking the entanglement front.

- **Growth**: boundary site entropy > threshold → activate dormant neighbors
- **Contraction**: active site entropy < threshold → site goes dormant
- 90% memory savings for simulations where entanglement is localized

### Dynamic Integration (`quhit_dyn_integrate.h` / `quhit_dyn_integrate.c`)

Connects the dynamic lattice engine to all overlay tiers (1D–6D). Includes oracle infrastructure with convergence detection (converging / oscillating / stagnant).

---

## Substrate Opcodes (`substrate_opcodes.h` / `quhit_substrate.c`)

20 quantum gates organized into 6 topological families, derived from cross-probe side-channel analysis:

| Family | Opcodes | Description |
|--------|---------|-------------|
| **Annihilation** | `NULL`, `VOID` | Project to vacuum, total erasure |
| **Scaling** | `SCALE_UP`, `SCALE_DN`, `SATURATE` | Amplitude manipulation |
| **Symmetry** | `PARITY`, `NEGATE`, `MIRROR` | Involutory transforms (P²=I) |
| **Phase** | `GOLDEN` (φ), `DOTTIE`, `CLOCK` (ω), `SQRT2` | Transcendental phase gates |
| **Coherence** | `QUIET` | Decoherence operator |
| **Transform** | `FUSE` | Subspace projection |

Warm-start caching detects repeated involutions and skips redundant computation.

---

## Self-Calibration (`quhit_calibrate.h` / `quhit_calibrate.c`)

Every critical constant is *derived from first principles* at runtime and verified against its algebraic identity:

- **φ** is not "1.618…" — it is the solution to x² = x + 1. Solved. Verified.
- **Dottie** is not "0.739…" — it is the fixed point of cos(x). Iterated. Verified.
- **ω₆** is not "0.5 + i·0.866…" — it is e^(2πi/6). Computed. Verified ω⁶ = 1.

Five cross-validation identities are checked at boot. Cost: ~1μs.

---

## Foundation Layer (internal)

The Triality Quhit is built on top of a lower-level quhit engine. Users interact with Triality; these modules provide the underlying infrastructure.

| File | Role |
|------|------|
| `quhit_engine.h` | Master header. Architecture constants. Each raw quhit = 6 complex amplitudes (96 bytes). Entangled pairs = 36 amplitudes (576 bytes). |
| `quhit_core.c` | Engine lifecycle, PRNG (LCG, deterministic, reproducible). |
| `quhit_gates.c` | DFT₆ (precomputed twiddle table — no trig at runtime), CZ, X, Z, arbitrary 6×6 unitaries. |
| `quhit_measure.c` | Born-rule sampling with collapse. Entangled measurement computes marginals, partially collapses joint state, auto-determines partner. |
| `quhit_entangle.c` | Bell-pair creation (braid), disentanglement (unbraid), product-state preparation. |
| `quhit_register.c` | 100-trillion-scale registers. GHZ entanglement across arbitrary N via sliding window — 768 bytes live regardless of N. |

### Primitives (header-only)

| File | Contents |
|------|----------|
| `arithmetic.h` | IEEE-754 constants reverse-engineered from substrate probing. |
| `born_rule.h` | Born-rule sampling utilities, fast inverse square root. |
| `statevector.h` | Cache-aligned state vector storage. |
| `superposition.h` | Precomputed DFT₆ twiddle tables. |
| `entanglement.h` | Joint-state operations, Bell states, partial trace. |
| `quhit_management.h` | Per-quhit state management, entanglement lifecycle. |
| `quhit_gauss.h` | O(N) analytic Gauss sum amplitude resolver for DFT₆+CZ circuits. |

### Additional Modules

| File | Purpose |
|------|---------|
| `bigint.c/.h` | Arbitrary-precision integer arithmetic (mul, gcd, pow_mod, add). |
| `svd_truncate.h` | SVD entry truncation buffer. Caps to 4096 entries by magnitude. |
| `tensor_network.h` | Shared tensor network constants and utilities. |
| `tensor_product.h` | Tensor product and contraction primitives. |
| `quhit_factored.h` | Factored state representations. |
| `reality_scaled.c` | Scaled reality simulation experiments. |

---

## Building

### Requirements

- **C compiler** with C99 support (GCC recommended)
- **libm** (math library)
- **OpenMP** (optional, for parallelism)
- **SSE2** (optional, for vectorized operations)

### Compile the benchmark suite ("Heap")

```bash
gcc -O2 -march=native -o heap hexstate_benchmark.c \
    quhit_core.c quhit_gates.c quhit_measure.c quhit_entangle.c \
    quhit_register.c quhit_substrate.c quhit_triality.c \
    quhit_triadic.c quhit_lazy.c quhit_calibrate.c \
    quhit_dyn_integrate.c quhit_peps_grow.c quhit_svd_gate.c \
    s6_exotic.c bigint.c mps_overlay.c peps_overlay.c \
    peps3d_overlay.c peps4d_overlay.c peps5d_overlay.c peps6d_overlay.c \
    -lm -fopenmp -msse2
```

### Run

```bash
./heap
```

Expected: **118/118 tests passed** across 21 sections covering every engine module.

### Compile the MIPT sweep

```bash
gcc -O2 -march=native -o mipt mipt_sweep.c \
    quhit_core.c quhit_gates.c quhit_measure.c quhit_entangle.c \
    quhit_register.c quhit_substrate.c quhit_triality.c \
    quhit_triadic.c quhit_lazy.c quhit_calibrate.c \
    quhit_dyn_integrate.c quhit_peps_grow.c quhit_svd_gate.c \
    s6_exotic.c bigint.c mps_overlay.c peps_overlay.c \
    peps3d_overlay.c peps4d_overlay.c peps5d_overlay.c peps6d_overlay.c \
    -lm -fopenmp -msse2
```

```bash
./mipt
```

Sweeps measurement rate p ∈ [0, 1] across all six dimension tiers. Maps the measurement-induced phase transition from volume-law to area-law entanglement entropy.

---

## Benchmark Results

### Heap benchmark (21 sections, 118 tests)

| Section | Module | Result |
|---------|--------|--------|
| §1–§4 | Core (init, gates, phase, X/Z) | ✓ |
| §5 | Measurement (Born rule, 6K samples, χ²) | ✓ |
| §6 | Entanglement (Bell pairs, disentangle) | ✓ |
| §7 | BigInt (mul, gcd, pow_mod) | ✓ |
| §8 | Substrate (20 opcodes, warm-start) | ✓ |
| §9 | Self-calibration (5 identities) | ✓ |
| §10 | Triality (126M gates/sec) | ✓ |
| §11 | Lazy gate chains | ✓ |
| §12 | S₆ exotic (15 synthemes) | ✓ |
| §13 | Flat quhit (promote/demote) | ✓ |
| §14 | Triadic 3-body entanglement | ✓ |
| §15 | Register (1M-quhit GHZ) | ✓ |
| §16 | Gauss sum amplitudes | ✓ |
| §17 | Lazy throughput (10M/sec) | ✓ |
| §18 | MPS overlay | ✓ |
| §19 | PEPS 2D overlay | ✓ |
| §20 | Dynamic chain growth | ✓ |
| §21 | Full pipeline (16M ops/sec) | ✓ |

### MIPT sweep (all 6 dimensions)

| Dim | Sites | S̄(p=0) | S̄(p=1) | Time |
|-----|-------|---------|---------|------|
| 1D MPS | 16 | 2.071 | 0.000 | 34s |
| 2D PEPS | 16 | 0.522 | 0.000 | 21s |
| 3D TNS | 18 | 0.548 | 0.000 | 225s |
| 4D TNS | 16 | 0.322 | 0.000 | 221s |
| 5D TNS | 32 | 0.161 | 0.000 | 413s |
| 6D TNS | 64 | 0.236 | 0.000 | 676s |

All dimensions show the correct volume → area law entropy transition.

---

## License

See repository for license details.
