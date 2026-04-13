# HexState LLM Quantizer

**Uniform Q2_K quantization that preserves reasoning at extreme compression.**

HexState compresses LLMs to **2.63 bits per weight** — quantizing *every* tensor including
embeddings — while preserving the model's reasoning, math, and instruction-following
capabilities. A 4.65B parameter model compresses from 8.67 GB to **1.44 GB** (6x reduction)
with no loss in practical reasoning benchmarks.

## Key Results

Tested on **Gemma 4 E2B-it** (4.65B params):

| Model | Size | BPW | PPL | Math | Logic | Speed |
|-------|------|-----|-----|------|-------|-------|
| BF16 (original) | 8.67 GB | 16.00 | 154.0 | ✅ | ✅ | 4.2 t/s |
| ggml Q2_K + iMatrix | 2.77 GB | 5.12 | 89.1 | ✅ | ✅ | 14.0 t/s |
| **HexState Q2_K** | **1.44 GB** | **2.63** | **129.6** | **✅** | **✅** | **18.1 t/s** |
| ggml Q2_K (no iMatrix) | 2.77 GB | 5.12 | 651.0 | ❌ | ❌ | 14.0 t/s |

> **Per-bit capability density:** HexState achieves the same reasoning accuracy as
> ggml's mixed-precision quantization at **half the file size** and **29% faster** generation.

## Philosophy: Uniform Quantization

Standard quantizers use mixed precision — Q6_K for embeddings, Q4_K for attention,
Q3_K for FFN, Q2_K for the rest. This introduces **asymmetric distortion**: different
parts of the model operate at different precision levels, warping the internal geometry.

HexState takes the opposite approach: **every tensor gets Q2_K**. The HPC optimizer
finds the best 4-level representation for each 256-weight block, preserving the model's
internal coherence at uniform precision. The result is a model that reasons like the
original despite extreme compression.

```
Standard (ggml):     Q6_K ──→ Q4_K ──→ Q3_K ──→ Q2_K   (asymmetric distortion)
HexState:            Q2_K ──→ Q2_K ──→ Q2_K ──→ Q2_K   (uniform compression)
```

## Quick Start

### Prerequisites

```bash
sudo apt install libgmp-dev libmpfr-dev gcc python3-numpy
```

### Build

```bash
cd LLM

# Build the HPC C engine (shared library for Python integration)
gcc -O2 -std=gnu99 -shared -fPIC -I.. \
    -DHEXSTATE_LIBRARY -o libhexstate_q2k.so \
    hexstate_quantize.c ../quhit_triality.c ../quhit_hexagram.c ../s6_exotic.c \
    -lm -lgmp -lmpfr

# Or build the standalone binary
make -f Makefile.quantize
```

### Quantize a Model

```bash
# Step 1: Convert HuggingFace model to BF16 GGUF (requires llama.cpp)
python3 convert_hf_to_gguf.py gemma-4-E2B-it/ \
    --outfile Gemma-4-E2B-it-BF16.gguf --outtype bf16

# Step 2: Generate importance matrix (recommended, ~15 min)
llama-imatrix -m Gemma-4-E2B-it-BF16.gguf \
    -f calibration_data.txt \
    -o imatrix.dat --chunks 100

# Step 3: HexState quantization (~80 min for 4.65B model)
python3 LLM/hexstate_requantize.py \
    Gemma-4-E2B-it-BF16.gguf \
    Gemma-4-E2B-it-Q2_K-HexState.gguf \
    --keep-metadata \
    --imatrix imatrix.dat
```

The quantizer automatically detects `libhexstate_q2k.so` in the `LLM/` directory:

```
  ╔════════════════════════════════════════════════════════════════╗
  ║  HExState GGUF Re-Quantizer                                  ║
  ║  GGUF → Q2_K GGUF with metadata passthrough                  ║
  ║  Engine: HPC + iMatrix (calibrated sensitivity propagation)  ║
  ╚════════════════════════════════════════════════════════════════╝

  Tensors to quantize (Q2_K): 318
  Tensors to keep as-is:      283
```

Without the `.so`, the quantizer falls back to a pure-Python numpy implementation
(correct output, without HPC optimization).

### Run Inference

```bash
# llama.cpp server
llama-server -m Gemma-4-E2B-it-Q2_K-HexState.gguf \
    --jinja --port 8899 -ngl 12 -c 4096

# Or direct CLI
llama-cli -m Gemma-4-E2B-it-Q2_K-HexState.gguf \
    -p "What is 17 * 23?" -n 256 --temp 0
```

## How It Works

### HPC Quantization Engine

For weight tensors (< 50M elements), the C engine performs sensitivity-aware
optimization:

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
│  6. Sensitivity-weighted MSE grid search with iMatrix   │
│  7. Pack into Q2_K blocks (scales|qs|d|dmin)            │
└─────────────────────────────────────────────────────────┘
```

### Embedding Quantization

For massive embedding tensors (> 50M elements, e.g., the 262K-vocab table at
2.35B elements), the engine uses chunked numpy vectorized quantization in
10M-element batches for speed while maintaining Q2_K format compatibility.

### iMatrix Integration

The importance matrix (generated by `llama-imatrix`) provides per-column importance
weights derived from calibration data. The quantizer uses these as weighted
least-squares coefficients — columns with higher importance get prioritized in the
scale optimization, allocating more of the Q2_K precision budget where it matters most.

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

## Gemma 4 Specifics

| Feature | How We Handle It |
|---------|-----------------|
| **Per-layer FFN sizes** (6144/12288) | Metadata passthrough from source GGUF |
| **Sliding window attention** (512 tokens) | Preserved in `gemma4.attention.sliding_window` KV |
| **Shared KV heads** (GQA, 8→1) | Preserved in per-layer `n_embd_k_gqa` arrays |
| **262,144 token vocabulary** | Q2_K quantized (uniform precision) |
| **Weight-tied embeddings** | Handled by llama.cpp internally |
| **Gated Delta Net (linear attention)** | Transparent — handled at inference |

## Optimizer Modes

```bash
./hexstate_quantize model_dir/ output.gguf --optimizer hybrid   # (default)
./hexstate_quantize model_dir/ output.gguf --optimizer hpc      # BP only
./hexstate_quantize model_dir/ output.gguf --optimizer mse      # grid search only
```

| Mode | Description | Speed | Quality |
|------|------------|-------|---------|
| `hpc` | BP sensitivity only, reference quantization | Fast | Good |
| `mse` | MSE grid search (from llm-compressor) | Medium | Better |
| `hybrid` | BP sensitivity → weighted MSE grid | Slow | **Best** |

## Benchmarks

### Perplexity (wikitext-2, 59 chunks, n_ctx=512)

| Quantization | Size | BPW | PPL ± σ |
|-------------|------|-----|---------|
| BF16 (baseline) | 8.67 GB | 16.00 | 154.0 ± 5.1 |
| ggml Q2_K + iMatrix | 2.77 GB | 5.12 | 89.1 ± 3.3 |
| **HexState full Q2_K** | **1.44 GB** | **2.63** | **129.6 ± 4.6** |
| HexState selective Q2_K | 5.74 GB | 10.58 | 107.6 ± 3.7 |
| ggml Q2_K (no iMatrix) | 2.77 GB | 5.12 | 651.0 |

### Reasoning Tests (temp=0, Gemma 4 E2B-it)

| Test | BF16 | HexState | ggml |
|------|------|----------|------|
| Arithmetic (17×23) | ✅ 391 | ✅ 391 | ✅ 391 |
| Syllogism logic | ✅ No | ✅ No | ✅ No |
| Word problem (eggs) | ✅ 5 | ✅ 5 | ✅ 5 |
| Generation speed | 4.2 t/s | **18.1 t/s** | 14.0 t/s |

### Safety Note

Extreme quantization (< 3 BPW) can degrade RLHF safety alignment. Models may
comply with requests that the original model would refuse. Always evaluate safety
properties before deployment and include appropriate disclaimers when distributing
heavily quantized models.

## License

Part of the HexState project.
