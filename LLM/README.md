# HexState Q2_K + Q4_0·HPC

**The smallest functional quantization of Gemma 4 26B. 10.2 GB. Runs on 12 GB hardware.**

No other public quantization of this model fits in 12 GB. The smallest community quant (IQ3_K_XXS) is ~12 GB and requires 14+ GB at runtime. HexState fits and runs with headroom to spare.

This will work on ANY Gemma4 model, but was developed specifically for 26B (So it can run on 12GB GPUs)

---

## Model Details

| | |
|---|---|
| **Base Model** | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| **Architecture** | Gemma 4 MoE — 26B total params, 4B active per token, 64 experts |
| **Quantization** | Mixed Q2_K (2.63 bpw) + Q4_0·HPC (4.5 bpw) |
| **File Size** | **10.2 GB** |
| **Format** | GGUF v3 — compatible with llama.cpp, LM Studio, Ollama |
| **Quantizer** | [HexState HPC Engine](https://github.com/user/HexState) |

### Precision Tiers

| Layer Type | Quantization | BPW | Method |
|-----------|-------------|-----|--------|
| Attention Q/K/V/O | Q4_0·HPC | 4.5 | 12-beam Hensel search + triality BP |
| FFN / MLP / Experts | Q2_K·HPC | 2.63 | 100-candidate beam search + triality BP |
| Embeddings / Norms | F16 / F32 | 16–32 | Preserved |
| MoE Router / Gates | F16 / F32 | 16–32 | Preserved — quantizing these breaks expert dispatch |

---

## Size Comparison

| Quantization | Size | Fits 12 GB? | Source |
|-------------|------|:-----------:|--------|
| BF16 | 48.5 GB | ❌ | Google |
| Q8_0 | ~27 GB | ❌ | Community |
| Q6_K | ~22 GB | ❌ | Community |
| Q4_K_M | 16.8 GB | ❌ | LM Studio / bartowski |
| IQ3_K_XXS | ~12 GB | ⚠️ | Unsloth |
| **HexState (this)** | **10.2 GB** | **✅** | **HexState HPC** |

---

## Quick Start

### LM Studio

1. Download the GGUF
2. Place in your LM Studio models directory
3. Load and chat — LM Studio auto-detects the Gemma 4 template

### llama.cpp Server

```bash
# Download the updated Gemma 4 chat template (required for correct output)
curl -L -o gemma4_chat_template.jinja \
  "https://huggingface.co/google/gemma-4-26B-A4B-it/raw/main/chat_template.jinja"

# Launch the server
llama-server \
  -m Gemma-4-26B-A4B-it-Q2_K-HexState.gguf \
  -ngl 0 \
  -c 4096 \
  --host 0.0.0.0 --port 8989 \
  --jinja \
  --chat-template-file gemma4_chat_template.jinja \
  --cache-ram 0 \
  -ctxcp 1
```

> **Important flags:**
> - `--jinja --chat-template-file` — Uses Google's latest Gemma 4 template. The template embedded in older GGUFs is broken. Without this, you get garbage output.
> - `--cache-ram 0 -ctxcp 1` — Prevents the sliding window attention checkpoint RAM explosion that affects all Gemma 4 models.
> - `-ngl 0` — CPU-only. Increase for GPU offload (e.g., `-ngl 20` for partial offload on 12 GB VRAM).

### llama.cpp CLI

```bash
llama-cli \
  -m Gemma-4-26B-A4B-it-Q2_K-HexState.gguf \
  --jinja \
  --chat-template-file gemma4_chat_template.jinja \
  -p "Implement a concurrent hash map in C" \
  -n 512 --temp 0.3
```

### Ollama

> ⚠️ **Ollama has known issues with Gemma 4.** If you get garbage output, switch to llama.cpp server or LM Studio. This is an [Ollama-side problem](https://old.reddit.com/r/LocalLLaMA/comments/1shs6sx/more_gemma4_fixes_in_the_past_24_hours/), not a model issue.

```
FROM ./Gemma-4-26B-A4B-it-Q2_K-HexState.gguf

PARAMETER temperature 0.4
PARAMETER num_ctx 2048
PARAMETER repeat_penalty 1.15
PARAMETER top_k 30
PARAMETER top_p 0.85
PARAMETER mlock true
```

### API Usage

Once the server is running, use the OpenAI-compatible API:

```bash
curl http://localhost:8989/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a Python LRU cache"}],
    "temperature": 0.3,
    "max_tokens": 512
  }'
```

---

## Recommended Settings

| Parameter | Value | Why |
|-----------|-------|-----|
| `temperature` | 0.3–0.4 | Lower than default — reduces sampling noise at low BPW |
| `top_k` | 20–30 | Narrow sampling keeps output coherent |
| `top_p` | 0.8–0.85 | Cuts the noisy long tail |
| `repeat_penalty` | 1.15–1.2 | Prevents self-correction loops |
| `context` | 2048–4096 | Higher contexts increase RAM usage significantly |

For **deterministic code generation**, use `temperature 0`.

---

## How It Works

Standard quantizers use round-to-nearest: for each weight block, compute a scale and round. HexState uses **HPC beam search with triality-enhanced belief propagation** — a fundamentally different approach.

### The Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  For each weight tensor:                                     │
│                                                              │
│  1. Compute greedy reference scales per block                │
│  2. Generate candidate grid (10–100 scale variants)          │
│  3. Encode candidates as Z₆ complex amplitudes               │
│  4. Build constraint graph (inter-block coupling)            │
│  5. Run belief propagation in 3 simultaneous views:          │
│       Edge × Vertex × Diagonal (triality)                    │
│  6. Combine via geometric mean:                              │
│       marginal[v] = ∛(edge × vertex × diagonal)             │
│  7. 12-beam Hensel search using combined marginals           │
│  8. Pack into GGUF blocks with optimal scales                │
└─────────────────────────────────────────────────────────────┘
```

### Why Attention Gets Q4_0

Quantization noise in attention projections cascades through `softmax(Q·K^T/√d)·V`. A single bad scale in a Q block shifts dot products enough to promote wrong tokens — manifesting as:
- Korean/Arabic character injection
- Word substitutions
- Self-correction loops

Promoting Q/K/V/O to Q4_0 (16 levels vs 4) eliminates these artifacts at a cost of only ~0.5 GB.

### RMSE Quality

| Metric | Value |
|--------|-------|
| Q4_0·HPC attention RMSE range | 2.8e-03 – 3.5e-03 |
| e-02 outliers | **0 out of 115 tensors** |
| Q2_K FFN RMSE range | consistent e-03 |

Zero RMSE outliers across every attention tensor. Standard round-to-nearest quantizers cannot achieve this.

---

## Known Limitations

1. **Low-frequency code patterns** (Win32 C, niche APIs) may have minor syntax errors — the Q2_K FFN layers lose precision on rare token sequences. Common patterns (Python, algorithms) are clean.

2. **The `if __name__ == "__main__":` idiom** occasionally gets mangled — this is a known fragile pattern under aggressive quantization.

3. **Safety alignment degradation** — extreme quantization (< 3 BPW) can weaken RLHF guardrails. The model may comply with requests the original would refuse. Evaluate safety properties before deployment.

4. **Ollama compatibility** — Ollama's Gemma 4 support is unreliable as of April 2026. Use llama.cpp or LM Studio.

---

## Technical Details

### Q2_K Block Layout (84 bytes / 256 weights)

```
Offset  Size  Field
  0      16   scales[16]    4-bit scale | 4-bit min per sub-block
 16      64   qs[64]        packed 2-bit quants (4 per byte)
 80       2   d             fp16 super-block scale
 82       2   dmin          fp16 super-block min scale
```

### Q4_0 Block Layout (18 bytes / 32 weights)

```
Offset  Size  Field
  0       2   d             fp16 block scale
  2      16   qs[16]        packed 4-bit quants (2 per byte)
                             nibble order: qs[j] = w[j] | (w[j+16] << 4)
```

### Gemma 4 MoE Handling

| Challenge | Solution |
|-----------|----------|
| `model.language_model.layers.` tensor prefix | Dual prefix detection |
| Non-256-aligned expert dimensions | Auto-fallback to Q4_0 |
| MoE router weights | Excluded from quantization |
| 64 experts per layer | Fused tensor handling |
| Sliding window attention (512) | Metadata passthrough |

---

## License

This quantization inherits the [Gemma license](https://ai.google.dev/gemma/terms) from the base model.

## Credits

Quantized with the [HexState HPC Engine](https://github.com/user/HexState) — triality-enhanced belief propagation over hexagonal constraint graphs.


**The smallest functional quantization of Gemma 4 26B. 10.2 GB. Runs on 12 GB hardware.**

No other public quantization of this model fits in 12 GB. The second smallest community quant is ~12 GB and requires 14+ GB at runtime.

---

## Model Details

| | |
|---|---|
| **Base Model** | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| **Architecture** | Gemma 4 MoE — 26B total params, 4B active per token, 64 experts |
| **Quantization** | Mixed Q2_K (2.63 bpw) + Q4_0·HPC (4.5 bpw) |
| **File Size** | **10.2 GB** |
| **Format** | GGUF v3 — compatible with llama.cpp, LM Studio, Ollama |
| **Quantizer** | [HexState HPC Engine](https://github.com/user/HexState) |

### Precision Tiers

| Layer Type | Quantization | BPW | Method |
|-----------|-------------|-----|--------|
| Attention Q/K/V/O | Q4_0·HPC | 4.5 | 12-beam Hensel search + triality BP |
| FFN / MLP / Experts | Q2_K·HPC | 2.63 | 100-candidate beam search + triality BP |
| Embeddings / Norms | F16 / F32 | 16–32 | Preserved |
| MoE Router / Gates | F16 / F32 | 16–32 | Preserved — quantizing these breaks expert dispatch |

---

## Size Comparison

| Quantization | Size | Fits 12 GB? | Source |
|-------------|------|:-----------:|--------|
| BF16 | 48.5 GB | ❌ | Google |
| Q8_0 | ~27 GB | ❌ | Community |
| Q6_K | ~22 GB | ❌ | Community |
| Q4_K_M | 16.8 GB | ❌ | LM Studio / bartowski |
| IQ3_K_XXS | ~12 GB | ⚠️ | Unsloth |
| **HexState (this)** | **10.2 GB** | **✅** | **HexState HPC** |

---

## Quick Start

### LM Studio

1. Download the GGUF
2. Place in your LM Studio models directory
3. Load and chat — LM Studio auto-detects the Gemma 4 template

### llama.cpp Server

```bash
# Download the updated Gemma 4 chat template (required for correct output)
curl -L -o gemma4_chat_template.jinja \
  "https://huggingface.co/google/gemma-4-26B-A4B-it/raw/main/chat_template.jinja"

# Launch the server
llama-server \
  -m Gemma-4-26B-A4B-it-Q2_K-quant.gguf \
  -ngl 0 \
  -c 4096 \
  --host 0.0.0.0 --port 8989 \
  --jinja \
  --chat-template-file gemma4_chat_template.jinja \
  --cache-ram 0 \
  -ctxcp 1
```

> **Important flags:**
> - `--jinja --chat-template-file` — Uses Google's latest Gemma 4 template. The template embedded in older GGUFs is broken. Without this, you get garbage output.
> - `--cache-ram 0 -ctxcp 1` — Prevents the sliding window attention checkpoint RAM explosion that affects all Gemma 4 models.
> - `-ngl 0` — CPU-only. Increase for GPU offload (e.g., `-ngl 20` for partial offload on 12 GB VRAM).

### llama.cpp CLI

```bash
llama-cli \
  -m Gemma-4-26B-A4B-it-quant.gguf \
  --jinja \
  --chat-template-file gemma4_chat_template.jinja \
  -p "Implement a concurrent hash map in C" \
  -n 512 --temp 0.3
```

### Ollama

> ⚠️ **Ollama has known issues with Gemma 4.** If you get garbage output, switch to llama.cpp server or LM Studio. This is an [Ollama-side problem](https://old.reddit.com/r/LocalLLaMA/comments/1shs6sx/more_gemma4_fixes_in_the_past_24_hours/), not a model issue.

```
FROM ./Gemma-4-26B-A4B-it-Q2_K-HexState.gguf

PARAMETER temperature 0.4
PARAMETER num_ctx 2048
PARAMETER repeat_penalty 1.15
PARAMETER top_k 30
PARAMETER top_p 0.85
PARAMETER mlock true
```

### API Usage

Once the server is running, use the OpenAI-compatible API:

```bash
curl http://localhost:8989/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a Python LRU cache"}],
    "temperature": 0.3,
    "max_tokens": 512
  }'
```

---

## Recommended Settings

| Parameter | Value | Why |
|-----------|-------|-----|
| `temperature` | 0.3–0.4 | Lower than default — reduces sampling noise at low BPW |
| `top_k` | 20–30 | Narrow sampling keeps output coherent |
| `top_p` | 0.8–0.85 | Cuts the noisy long tail |
| `repeat_penalty` | 1.15–1.2 | Prevents self-correction loops |
| `context` | 2048–4096 | Higher contexts increase RAM usage significantly |

For **deterministic code generation**, use `temperature 0`.

---

## How It Works

Standard quantizers use round-to-nearest: for each weight block, compute a scale and round. HexState uses **HPC beam search with triality-enhanced belief propagation** — a fundamentally different approach.

### The Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  For each weight tensor:                                     │
│                                                              │
│  1. Compute greedy reference scales per block                │
│  2. Generate candidate grid (10–100 scale variants)          │
│  3. Encode candidates as Z₆ complex amplitudes               │
│  4. Build constraint graph (inter-block coupling)            │
│  5. Run belief propagation in 3 simultaneous views:          │
│       Edge × Vertex × Diagonal (triality)                    │
│  6. Combine via geometric mean:                              │
│       marginal[v] = ∛(edge × vertex × diagonal)             │
│  7. 12-beam Hensel search using combined marginals           │
│  8. Pack into GGUF blocks with optimal scales                │
└─────────────────────────────────────────────────────────────┘
```

### Why Attention Gets Q4_0

Quantization noise in attention projections cascades through `softmax(Q·K^T/√d)·V`. A single bad scale in a Q block shifts dot products enough to promote wrong tokens — manifesting as:
- Korean/Arabic character injection
- Word substitutions
- Self-correction loops

Promoting Q/K/V/O to Q4_0 (16 levels vs 4) eliminates these artifacts at a cost of only ~0.5 GB.

### RMSE Quality

| Metric | Value |
|--------|-------|
| Q4_0·HPC attention RMSE range | 2.8e-03 – 3.5e-03 |
| e-02 outliers | **0 out of 115 tensors** |
| Q2_K FFN RMSE range | consistent e-03 |

Zero RMSE outliers across every attention tensor. Standard round-to-nearest quantizers cannot achieve this.

---

## Known Limitations

1. **Low-frequency code patterns** (Win32 C, niche APIs) may have minor syntax errors — the Q2_K FFN layers can sometimes lose precision on rare token sequences. Common patterns (Python, algorithms) are clean.

2. **The `if __name__ == "__main__":` idiom** sometimes gets mangled — this is a known fragile pattern under aggressive quantization.

3. **Safety alignment degradation** — extreme quantization (< 3 BPW) can weaken RLHF guardrails. The model may comply with requests the original would refuse. Evaluate safety properties before deployment.

4. **Ollama compatibility** — Ollama's Gemma 4 support is unreliable as of April 2026. Use llama.cpp or LM Studio.

---

## Technical Details

### Q2_K Block Layout (84 bytes / 256 weights)

```
Offset  Size  Field
  0      16   scales[16]    4-bit scale | 4-bit min per sub-block
 16      64   qs[64]        packed 2-bit quants (4 per byte)
 80       2   d             fp16 super-block scale
 82       2   dmin          fp16 super-block min scale
```

### Q4_0 Block Layout (18 bytes / 32 weights)

```
Offset  Size  Field
  0       2   d             fp16 block scale
  2      16   qs[16]        packed 4-bit quants (2 per byte)
                             nibble order: qs[j] = w[j] | (w[j+16] << 4)
```

### Gemma 4 MoE Handling

| Challenge | Solution |
|-----------|----------|
| `model.language_model.layers.` tensor prefix | Dual prefix detection |
| Non-256-aligned expert dimensions | Auto-fallback to Q4_0 |
| MoE router weights | Excluded from quantization |
| 64 experts per layer | Fused tensor handling |
| Sliding window attention (512) | Metadata passthrough |

---

## License

This quantization inherits the [Gemma license](https://ai.google.dev/gemma/terms) from the base model.

## Credits

Quantized with the [HexState HPC Engine](https://github.com/user/HexState) — triality-enhanced belief propagation over hexagonal constraint graphs.
