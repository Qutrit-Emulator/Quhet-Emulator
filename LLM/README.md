# HexState LLM Quantizer — Gemma 4 Q2_K

**HPC-Optimized 2-bit quantization for Gemma 4 (E2B/E4B) models using the HexState Ouroboros engine.**

This tool quantizes Gemma 4 models from BF16 GGUF to Q2_K (2.625 bits per weight) using
sensitivity-aware precision allocation derived from the HexState quantum-inspired
belief propagation framework.

## How It Works

Standard Q2_K quantization treats every weight block identically — same grid search
parameters, same error tolerance. HexState does something fundamentally different:

```
┌─────────────────────────────────────────────────────────┐
│  For each weight tensor:                                │
│                                                         │
│  1. Build HPCGraph over 256-weight superblocks          │
│  2. Encode 6 scale candidates as Boltzmann amplitudes   │
│     into Z₆ quhit local states (triality DFT)          │
│  3. Wire CZ edges (nearest-neighbor) + hexagram edges   │
│  4. Run Möbius amplitude belief propagation             │
│  5. Extract marginal entropy per block:                 │
│       high entropy → sensitive → needs precision        │
│       low entropy  → confident → can compress harder    │
│  6. Sensitivity-weighted MSE grid search:               │
│       sensitive blocks: grid=200, patience=10           │
│       insensitive blocks: grid=50, patience=3           │
│  7. Pack into Q2_K blocks (scales|qs|d|dmin)            │
└─────────────────────────────────────────────────────────┘
```

The same mathematical machinery that powers HexState's integer factoring engine
(HPCGraph, triality, Möbius sheets, CZ entanglement) is repurposed here for
intelligent bit allocation across weight blocks.

## Quick Start

### Prerequisites

```bash
# HexState engine dependencies
sudo apt install libgmp-dev libmpfr-dev

# For inference testing
# Build llama.cpp with Vulkan support (see llama.cpp docs)
```

### Build

```bash
cd LLM

# Build the shared library (for Python integration)
gcc -O2 -std=gnu99 -shared -fPIC -I.. \
    -DHEXSTATE_LIBRARY -o libhexstate_q2k.so \
    hexstate_quantize.c ../quhit_triality.c ../quhit_hexagram.c ../s6_exotic.c \
    -lm -lgmp -lmpfr

# Build the standalone binary (for direct safetensors → GGUF)
make -f Makefile.quantize
```

### Quantize Gemma 4

The recommended workflow uses the Python GGUF re-quantizer with the HPC C backend.
This preserves all Gemma 4 metadata (per-layer FFN sizes, sliding window patterns,
shared KV heads, tokenizer, chat template) while using HexState for the actual
weight quantization:

```bash
# Step 1: Convert HF model to BF16 GGUF (using llama.cpp)
python3 llama.cpp/convert_hf_to_gguf.py gemma-4-E2B-it/ \
    --outfile Gemma-4-E2B-it-BF16.gguf --outtype bf16

# Step 2: HexState Q2_K quantization (auto-detects libhexstate_q2k.so)
python3 LLM/hexstate_requantize.py \
    Gemma-4-E2B-it-BF16.gguf \
    Gemma-4-E2B-it-Q2_K.gguf \
    --keep-metadata
```

When `libhexstate_q2k.so` is present in the `LLM/` directory, the quantizer
automatically uses the HPC engine. You'll see this in the banner:

```
  ╔════════════════════════════════════════════════════════════════╗
  ║  HexState GGUF Re-Quantizer                                  ║
  ║  GGUF → Q2_K GGUF with metadata passthrough                  ║
  ║  Engine: HPC (BP + MSE Grid + Sensitivity Propagation)       ║
  ╚════════════════════════════════════════════════════════════════╝
```

Without the `.so`, it falls back to a pure-Python numpy implementation (still
correct, just without HPC optimization).

### Run Inference

```bash
llama-server \
    -m Gemma-4-E2B-it-Q2_K.gguf \
    --jinja --port 8899 \
    -ngl 12 -c 4096 \
    --reasoning-budget 0
```

> **Note:** Gemma 4 with Q2_K currently requires Vulkan for GPU offloading.
> CUDA/Metal support depends on your llama.cpp build.

## Architecture

```
hexstate_requantize.py          Python GGUF-to-GGUF pipeline
    │                           Reads source GGUF, copies all metadata,
    │                           re-quantizes weight tensors to Q2_K
    │
    ├── libhexstate_q2k.so      HPC C engine (loaded via ctypes)
    │   ├── HPCGraph             Sensitivity graph over weight blocks
    │   ├── triality DFT         Z₆ amplitude encoding
    │   ├── Möbius BP            Belief propagation convergence
    │   └── MSE grid search      Sensitivity-weighted scale optimization
    │
    ├── hexstate_quantize.c      Source for the C engine + standalone binary
    ├── gguf_format.h            GGUF v3 binary format (BlockQ2K, FP16, etc.)
    ├── safetensors_reader.h     Direct safetensors loading (standalone mode)
    ├── tokenizer_reader.h       HF tokenizer.json parser
    ├── imatrix_reader.h         Importance matrix loader
    └── Makefile.quantize        Build configuration
```

## Gemma 4 Specifics

Gemma 4 has several architectural features that require careful handling:

| Feature | How We Handle It |
|---------|-----------------|
| **Per-layer FFN sizes** (6144/12288) | Metadata passthrough from source GGUF |
| **Sliding window attention** (512 tokens) | Preserved in `gemma4.attention.sliding_window` KV |
| **Shared KV heads** (GQA, 8→1) | Preserved in per-layer `n_embd_k_gqa` arrays |
| **262,144 token vocabulary** | Kept as BF16 (not quantized) |
| **`<unused*>` control tokens** | Token types patched to CONTROL (type=3) |
| **Weight-tied embeddings** | Handled by llama.cpp internally |
| **Gated Delta Net (linear attention)** | Transparent — handled at inference |

## Q2_K Block Layout

The Q2_K block format (84 bytes per 256 weights) must match ggml's `block_q2_K` exactly:

```
Offset  Size  Field
──────  ────  ─────────────────────────────────
  0      16   scales[16]    4-bit scale | 4-bit min per sub-block
 16      64   qs[64]        packed 2-bit quants (4 per byte)
 80       2   d             fp16 super-block scale
 82       2   dmin          fp16 super-block min scale
──────  ────
 84 bytes total = 2.625 bits per weight
```

> **Critical:** The field order is `scales → qs → d → dmin`. Many implementations
> incorrectly place `d, dmin` first — this causes silent data corruption where
> the model loads but generates garbage.

## Optimizer Modes

The standalone C binary supports three optimization strategies:

```bash
./hexstate_quantize model_dir/ output.gguf --optimizer hybrid   # (default)
./hexstate_quantize model_dir/ output.gguf --optimizer hpc      # BP only
./hexstate_quantize model_dir/ output.gguf --optimizer mse      # grid search only
```

| Mode | Description | Speed | Quality |
|------|------------|-------|---------|
| `hpc` | BP sensitivity only, reference quantization | Fast | Good |
| `mse` | MSE grid search (from llm-compressor) | Medium | Better |
| `hybrid` | BP sensitivity → weighted MSE grid | Recommended | Best |

Optional importance matrix for further quality improvement:
```bash
./hexstate_quantize model_dir/ output.gguf --imatrix importance.dat
```

## License

Part of the HexState project.
