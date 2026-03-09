<p align="center">
HEAP COMPILE. Otherwise you will get segfault!
</p>

<p align="center">
If there is no test folder, don't clone this!
</p>

<p align="center">
Note: 3.7 is somewhat heavier than prior released, because now we're keeping 4096 amplitudes instead of 36.. if running the same bench as Willow, it will complete in roughly 15 minutes.
</p>


# HexState Engine v3.7

**D=6 hexagonal quantum simulator with native S₆ outer automorphism, triality views, and tensor network overlays spanning 1D to 6D.**

Pure C. No dependencies beyond libc and libm. Runs on consumer hardware.

---

## Architecture

HexState's fundamental unit is the **quhit** — a D=6 qudit native to the hexagonal Hilbert space. Six basis states `|0⟩...|5⟩` encode the 2×3 symmetry of the hexagon. All operations preserve this structure natively, without decomposition into qubits.

### Core Engine (`quhit_engine.h`)

| Capacity | Limit |
|:---|:---|
| Quhits | 256K |
| Entangled pairs | 256K |
| Registers | 16K |
| Basis type | `unsigned __int128` (128-bit) |
| Arithmetic | 4096-bit fixed-point (`bigint.h`) |

The engine uses **Magic Pointers** — sparse register encoding where each site's tensor is a quhit register with implicit zero compression. No dense arrays. RAM cost scales with non-zero entries, not Hilbert space dimension.

**Component files:**
- `quhit_core.c` — Engine lifecycle, PRNG, chunk init
- `quhit_gates.c` — CZ, DFT₆, X, Z, phase, local unitaries
- `quhit_measure.c` — Born-rule measurement, inspect, snapshot
- `quhit_entangle.c` — Bell pairs, product states, braid/unbraid
- `quhit_register.c` — 100T-scale register API, sparse state vectors
- `quhit_substrate.c` — Hardware substrate and ISA simulation

---

### Triality Layer (`quhit_triality.h`)

Every quantum state is simultaneously held in **three mutually-defining views**:

| View | Basis | Role |
|:---|:---|:---|
| **Edge** | Computational `\|k⟩` | Gate application, measurement |
| **Vertex** | Fourier `DFT₆\|k⟩` | Interference, phase analysis |
| **Diagonal** | Conjugate `DFT₆²\|k⟩` | Entanglement structure |

Plus two derived views:
- **Folded** — Antipodal vesica fold (stage 1 of factored DFT₆)
- **Exotic** — Syntheme-parameterized via S₆ outer automorphism

View switching is achieved via DFT₆, which has the key property **DFT₆⁴ = I** (order 4). Forward and inverse transforms cycle through exactly 4 positions.

**Performance optimizations:**
- **Eigenstate Phase-Lock** — Zero-cost view conversion when state is a DFT₆ eigenstate (O(1) pointer relabel)
- **Subspace Confinement** — `active_mask` (6-bit) skips zero basis states (up to 56% savings on CZ)
- **Real-Valued Fast Path** — 2× arithmetic savings when imaginary parts are zero
- **Folded View** — O(18) intermediate DFT₆ instead of O(36)

Tracked in real-time by the `TrialityStats` counter API.

---

### Three-Body Entanglement (`quhit_triadic.h`)

Entanglement is natively **three-body**, not pairwise. The `TriadicJoint` holds 6³ = 216 complex amplitudes for three entangled quhits — the full joint state `|a,b,c⟩`.

The basis partitions into **CMY channels**:

| Channel | States | Color |
|:---|:---|:---|
| C (Cyan) | `\|0⟩, \|1⟩` | Qubit subspace |
| M (Magenta) | `\|2⟩, \|3⟩` | Qubit subspace |
| Y (Yellow) | `\|4⟩, \|5⟩` | Qubit subspace |

Each channel is a qubit subspace. Three channels × two states = six. Memory: 216 × 16 = 3,456 bytes per triple (fits in L1 cache).

---

### S₆ Outer Automorphism (`s6_exotic.h`)

S₆ is the **only** symmetric group with a non-trivial outer automorphism. HexState implements:

- **φ (the automorphism)** — lookup table over all 720 elements of S₆
- **15 synthemes** — all partitions of {0,...,5} into 3 unordered pairs
- **6 synthematic totals** — maximal sets of 5 disjoint synthemes
- **Exotic invariant Δ(ψ)** — measures hexagonal polarization:
  - Δ = 0: automorphism-transparent (generic, could be simulated with qubits)
  - Δ > 0: hexagonally polarized (structure unique to D=6)
- **Dual measurement** — probabilities in both standard and exotic bases simultaneously
- **Exotic fingerprint** — per-conjugacy-class breakdown (11 values for S₆'s 11 classes)
- **Syntheme-parameterized fold** — vesica/wave decomposition using any of the 15 synthemes

---

### Tensor Network Overlays

All overlays use the same Magic Pointer register substrate. Sites are `TriOverlaySite` with triality sidecar for gate routing.

| Overlay | Dimension | Bond χ | Tensor indices | Header |
|:---|:---:|:---:|:---|:---|
| **MPS** | 1D | 256 | `\|k,α,β⟩` (3) | `mps_overlay.h` |
| **PEPS** | 2D | 512 | `\|k,u,d,l,r⟩` (5) | `peps_overlay.h` |
| **PEPS 3D** | 3D | 256 | 7 indices | `peps3d_overlay.h` |
| **PEPS 4D** | 4D | 128 | 9 indices | `peps4d_overlay.h` |
| **PEPS 5D** | 5D | 128 | 11 indices | `peps5d_overlay.h` |
| **TNS 6D** | 6D | 128 | `\|k,b₀..b₁₁⟩` (13) | `peps6d_overlay.h` |

**Common API pattern** (example for 6D):
```c
Tns6dGrid *g = tns6d_init(Lx, Ly, Lz, Lw, Lv, Lu);
tns6d_gate_1site(g, x,y,z,w,v,u, U_re, U_im);   // 1-site gate
tns6d_gate_x(g, x,y,z,w,v,u, G_re, G_im);        // 2-site bond gate (X axis)
// ... gate_y, gate_z, gate_w, gate_v, gate_u
tns6d_local_density(g, x,y,z,w,v,u, probs);       // Marginal probabilities
int outcome = tns6d_measure_site(g, x,y,z,w,v,u); // Born-rule + collapse
tns6d_free(g);
```

**6D specifics**: 2⁶ = 64 sites → 6⁶⁴ ≈ 10⁴⁹ Hilbert space. 12 bonds per site (2 per spatial axis). Each bond carries χ = 128 singular values managed by the vesica fold SVD.

---

### SVD Engine (`tensor_svd.h`)

Truncated SVD via Jacobi eigendecomposition of M†M, shared across all overlays.

- **Vesica fold SVD** — S₆ syntheme-aware truncation using the optimal pairing for the active subspace
- **Sparse power iteration** — for high-dimensional tensors where dense SVD is impractical
- **7-layer numerical stability**:
  - Kahan compensated sums for Jacobi convergence
  - FMA (Fused Multiply-Add) in Modified Gram-Schmidt
  - ε-derived thresholds (no magic numbers)
  - Information-theoretic rank truncation: σⱼ/σ₀ < ε
  - Newton iteration refinement
  - Correct Jacobi diagonal sign and eigenvector phase orientation
- **SVD Spectrum Oracle** — caches (σ, U, V†) per bond, dual-key (gate hash + register fingerprint), skips redundant O(n³) SVDs
- **Reconstruction accuracy**: 10⁻¹⁶

---

### Entropy-Adaptive Dynamics (`quhit_dyn_integrate.h`)

**DynChain** — a breathing lattice that grows or contracts based on real-time entanglement entropy.

**Five oracles** drive the dynamics:
1. **Entropy Gradient** — linear prediction of site entropy trends
2. **Mutual Information** — inter-site correlation detection (optimal coupling pairs)
3. **Convergence Horizon** — oscillation/stagnation detection via fidelity trace
4. **Boltzmann Gate Selector** — self-tuned β for gate recommendation (0–5)
5. **Ouroboros** — self-referential feedback loop combining all oracle scores

Active region `[start, end]` expands when entropy rises at boundaries, contracts when interior sites equilibrate. The oracle history ring buffer (depth 8) prevents thrashing.

---

### Triality Overlay Integration (`triality_overlay.h`)

Every tensor network site gets a triality sidecar. Gates are classified at application time:

| Class | Action | Cost |
|:---|:---|:---|
| `GATE_IDENTITY` | Skip | O(0) |
| `GATE_DIAGONAL` | Phase in Edge view | O(D) |
| `GATE_DFT` | View switch | O(1) |
| `GATE_GENERAL` | Full unitary multiply | O(D²) |

Two-site gates use the **diagonal 2-site fastpath** when both sites are basis states, applying CZ as a single phase kick — O(1) instead of O(D⁴).

---

## Build

All targets compile with GCC. No external dependencies.

```bash
# Core compilation (all source files)
gcc -O2 -march=native -o <target> <target>.c \
    quhit_core.c quhit_gates.c quhit_measure.c quhit_entangle.c \
    quhit_register.c quhit_substrate.c quhit_triality.c quhit_triadic.c \
    quhit_lazy.c quhit_calibrate.c quhit_dyn_integrate.c quhit_peps_grow.c \
    quhit_svd_gate.c s6_exotic.c bigint.c \
    mps_overlay.c peps_overlay.c peps3d_overlay.c peps4d_overlay.c \
    peps5d_overlay.c peps6d_overlay.c \
    -lm -fopenmp -msse2
```

**Requirements**: GCC with `__int128` support, SSE2, OpenMP. Linux x86_64.

---

## File Reference

### Core Engine
| File | Description |
|:---|:---|
| `quhit_engine.h` | Engine types, constants, API declarations |
| `quhit_core.c` | Lifecycle, PRNG, chunk initialization |
| `quhit_gates.c` | CZ, DFT₆, X, Z, phase, arbitrary unitaries |
| `quhit_measure.c` | Born-rule measurement, inspection, snapshots |
| `quhit_entangle.c` | Bell pairs, product states, braid/unbraid |
| `quhit_register.c` | 100T-scale sparse register API |
| `quhit_substrate.c` | Hardware substrate layer, ISA opcodes |

### Triality & S₆
| File | Description |
|:---|:---|
| `quhit_triality.h/c` | Triality quhit: Edge/Vertex/Diagonal/Folded/Exotic views |
| `quhit_triadic.h/c` | Three-body joints (216 amplitudes), CMY channels |
| `s6_exotic.h/c` | S₆ outer automorphism, 15 synthemes, exotic invariant |
| `triality_overlay.h` | Gate classification, masked register ops, CZ routing |

### Tensor Networks
| File | Description |
|:---|:---|
| `mps_overlay.h/c` | 1D Matrix Product State (D=6, χ=256) |
| `peps_overlay.h/c` | 2D PEPS (D=6, χ=512) |
| `peps3d_overlay.h/c` | 3D tensor network (χ=256) |
| `peps4d_overlay.h/c` | 4D tensor network (χ=128) |
| `peps5d_overlay.h/c` | 5D tensor network (χ=128) |
| `peps6d_overlay.h/c` | 6D tensor network (χ=128, 13-index tensors) |
| `tensor_svd.h` | Jacobi SVD with vesica fold, sparse power iteration |

### Dynamics & Optimization
| File | Description |
|:---|:---|
| `quhit_dyn_integrate.h/c` | DynChain entropy-adaptive lattice with 5 oracles |
| `quhit_lazy.h/c` | Heisenberg-picture lazy evaluation (segment chains) |
| `quhit_calibrate.h/c` | Calibration and threshold tuning |
| `quhit_peps_grow.h/c` | Dynamic PEPS lattice growth |
| `quhit_svd_gate.h/c` | SVD-based gate decomposition |

### Numerical Infrastructure
| File | Description |
|:---|:---|
| `bigint.h/c` | 4096-bit arbitrary-precision integer arithmetic |
| `arithmetic.h` | Fixed-point side-channel primitives |
| `born_rule.h` | Born-rule fast paths (isqrt, recip) |
| `statevector.h` | State vector streaming interface |
| `superposition.h` | Superposition management |
| `flat_quhit.h` | Flat (non-triality) quhit fallback |
| `svd_truncate.h` | SVD truncation policies |

---

## License

MIT
