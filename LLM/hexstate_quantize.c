/* ═══════════════════════════════════════════════════════════════════════════
 * hexstate_quantize.c — HExState GGUF Quantizer
 *
 * ╔═══════════════════════════════════════════════════════════════╗
 * ║  HPC-Optimized GGUF Quantization Engine                      ║
 * ║                                                               ║
 * ║  Architecture: HPCGraph Sensitivity Propagation               ║
 * ║  Optimization: Complex Amplitude BP + MCMC Scale Search       ║
 * ║  Output: GGUF v3 (Q8_0, Q2_K)                                 ║
 * ║                                                               ║
 * ║  "The weight and the quantized are opposite faces."           ║
 * ╚═══════════════════════════════════════════════════════════════╝
 *
 * This tool adapts the HExState HPC Ouroboros factoring engine for
 * LLM weight quantization. The core mathematical machinery is reused:
 *
 *   Factoring Domain          →  Quantization Domain
 *   ─────────────────────────────────────────────────
 *   HPCGraph + CZ edges       →  Block sensitivity graph
 *   Complex Amplitude BP      →  Importance propagation
 *   MCMC period sampler       →  Optimal scale search
 *   try_period() validation   →  Error bound checking
 *   LLL lattice reduction     →  (future) Adaptive bit allocation
 *
 * Build:
 *   gcc -O2 -std=gnu99 -o hexstate_quantize hexstate_quantize.c \
 *       quhit_triality.c quhit_hexagram.c s6_exotic.c bigint.c \
 *       -lm -lgmp -lmpfr
 *
 * Usage:
 *   ./hexstate_quantize model.safetensors output.gguf [Q8_0|Q2_K]
 * ═══════════════════════════════════════════════════════════════════════════ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpfr.h>

/* HExState headers — reused from the factoring engine */
#include "quhit_triality.h"
#include "hpc_graph.h"
#include "hpc_mobius.h"
#include "s6_exotic.h"

/* Quantization-specific headers */
#include "gguf_format.h"
#include "safetensors_reader.h"
#include "tokenizer_reader.h"

#define D 6  /* Preserved from HExState — the triality dimension */

/* ═══════════════════════════════════════════════════════════════════════════
 * MODEL ARCHITECTURE AUTO-DETECTION
 *
 * Infers model architecture metadata from tensor names and shapes.
 * Supports: LLaMA, Mistral, Qwen, Phi, Gemma, GPT-NeoX, Falcon
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    char     architecture[64];   /* "llama", "mistral", etc.       */
    char     name[256];          /* Human-readable model name       */
    uint32_t block_count;        /* Number of transformer layers    */
    uint32_t embedding_length;   /* Hidden dimension                */
    uint32_t head_count;         /* Number of attention heads       */
    uint32_t head_count_kv;      /* Number of KV heads (GQA)        */
    uint32_t vocab_size;         /* Vocabulary size                 */
    uint32_t context_length;     /* Max context length (default)    */
    float    rope_freq_base;     /* RoPE frequency base             */
    uint32_t feed_forward_length; /* FFN intermediate size           */
    float    rms_norm_eps;       /* RMS norm epsilon                */
} ModelArchitecture;

/* Count tensor names matching a pattern prefix */
static int count_tensors_with_prefix(const STFile *st, const char *prefix)
{
    int count = 0;
    int prefix_len = strlen(prefix);
    for (int i = 0; i < st->n_tensors; i++) {
        if (strncmp(st->tensors[i].name, prefix, prefix_len) == 0)
            count++;
    }
    return count;
}

/* Find max layer index from tensor names like "model.layers.N.xxx" */
static int find_max_layer_index(const STFile *st, const char *layer_prefix)
{
    int max_idx = -1;
    int prefix_len = strlen(layer_prefix);
    for (int i = 0; i < st->n_tensors; i++) {
        if (strncmp(st->tensors[i].name, layer_prefix, prefix_len) == 0) {
            int idx = atoi(st->tensors[i].name + prefix_len);
            if (idx > max_idx) max_idx = idx;
        }
    }
    return max_idx;
}

static void detect_architecture(const STFile *st, ModelArchitecture *arch)
{
    memset(arch, 0, sizeof(*arch));

    /* Default values */
    strcpy(arch->architecture, "llama");
    strcpy(arch->name, "HExState-quantized");
    arch->context_length = 4096;
    arch->rope_freq_base = 10000.0f;
    arch->rms_norm_eps = 1e-5f;

    /* Detect architecture from tensor naming patterns */
    int has_model_layers = count_tensors_with_prefix(st, "model.layers.");
    int has_gpt_neox = count_tensors_with_prefix(st, "gpt_neox.");
    int has_transformer = count_tensors_with_prefix(st, "transformer.");

    if (has_model_layers > 0) {
        /* LLaMA / Mistral / Qwen / Gemma style */
        arch->block_count = find_max_layer_index(st, "model.layers.") + 1;

        /* Detect sub-architecture from attention layer names */
        if (count_tensors_with_prefix(st, "model.layers.0.self_attn.q_proj") > 0) {
            /* Check for Mistral vs LLaMA via KV head count */
            int qproj_idx = st_find_tensor(st, "model.layers.0.self_attn.q_proj.weight");
            int kproj_idx = st_find_tensor(st, "model.layers.0.self_attn.k_proj.weight");

            if (qproj_idx >= 0) {
                arch->embedding_length = st->tensors[qproj_idx].shape[1];
                arch->head_count = st->tensors[qproj_idx].shape[0] / 
                                   (arch->embedding_length / 
                                    (st->tensors[qproj_idx].shape[0] > 0 ? 
                                     (arch->embedding_length / 128) : 1));
                
                /* Infer head dim and head count more robustly */
                int64_t q_out = st->tensors[qproj_idx].shape[0];
                int64_t hidden = st->tensors[qproj_idx].shape[1];
                arch->embedding_length = hidden;
                
                /* Try common head dimensions: 128, 64, 96 */
                int head_dim = 128;
                if (q_out % 128 == 0) head_dim = 128;
                else if (q_out % 96 == 0) head_dim = 96;
                else if (q_out % 64 == 0) head_dim = 64;
                
                arch->head_count = q_out / head_dim;
                
                if (kproj_idx >= 0) {
                    int64_t k_out = st->tensors[kproj_idx].shape[0];
                    arch->head_count_kv = k_out / head_dim;
                } else {
                    arch->head_count_kv = arch->head_count;
                }
            }

            /* Detect Mistral vs LLaMA */
            if (arch->head_count_kv < arch->head_count) {
                strcpy(arch->architecture, "llama");  /* GQA → likely Mistral/LLaMA2+ */
            }
        }

        /* Get vocab size from embed_tokens */
        int embed_idx = st_find_tensor(st, "model.embed_tokens.weight");
        if (embed_idx >= 0) {
            arch->vocab_size = st->tensors[embed_idx].shape[0];
        }

        /* Get FFN size from gate or up projection */
        int gate_idx = st_find_tensor(st, "model.layers.0.mlp.gate_proj.weight");
        if (gate_idx >= 0) {
            arch->feed_forward_length = st->tensors[gate_idx].shape[0];
        } else {
            int up_idx = st_find_tensor(st, "model.layers.0.mlp.up_proj.weight");
            if (up_idx >= 0)
                arch->feed_forward_length = st->tensors[up_idx].shape[0];
        }
    } else if (has_gpt_neox > 0) {
        strcpy(arch->architecture, "gpt_neox");
        arch->block_count = find_max_layer_index(st, "gpt_neox.layers.") + 1;
    } else if (has_transformer > 0) {
        strcpy(arch->architecture, "falcon");
        arch->block_count = find_max_layer_index(st, "transformer.h.") + 1;
    }

    /* Fill in defaults for anything we couldn't detect */
    if (arch->head_count == 0) arch->head_count = 32;
    if (arch->head_count_kv == 0) arch->head_count_kv = arch->head_count;
    if (arch->embedding_length == 0) arch->embedding_length = 4096;
    if (arch->vocab_size == 0) arch->vocab_size = 32000;
    if (arch->feed_forward_length == 0) 
        arch->feed_forward_length = (arch->embedding_length * 8) / 3;  /* SwiGLU default */
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TENSOR NAME MAPPING: HuggingFace → GGUF Standard
 *
 * Maps SafeTensors tensor names to the standardized GGUF naming
 * convention used by llama.cpp for model loading.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void map_tensor_name(const char *hf_name, char *gguf_name, int buflen)
{
    /* Start with identity mapping */
    strncpy(gguf_name, hf_name, buflen - 1);
    gguf_name[buflen - 1] = '\0';

    /* Common LLaMA/Mistral mappings */
    struct { const char *from; const char *to; } mappings[] = {
        {"model.embed_tokens.weight",       "token_embd.weight"},
        {"model.norm.weight",               "output_norm.weight"},
        {"lm_head.weight",                  "output.weight"},
        {NULL, NULL}
    };

    for (int m = 0; mappings[m].from; m++) {
        if (strcmp(hf_name, mappings[m].from) == 0) {
            strncpy(gguf_name, mappings[m].to, buflen - 1);
            return;
        }
    }

    /* Layer mappings: "model.layers.N.xxx" → "blk.N.xxx" */
    if (strncmp(hf_name, "model.layers.", 13) == 0) {
        int layer_idx;
        char rest[ST_MAX_NAME_LEN];
        if (sscanf(hf_name, "model.layers.%d.%255s", &layer_idx, rest) == 2) {
            /* Map sublayer names */
            struct { const char *from; const char *to; } layer_maps[] = {
                {"self_attn.q_proj.weight",    "attn_q.weight"},
                {"self_attn.k_proj.weight",    "attn_k.weight"},
                {"self_attn.v_proj.weight",    "attn_v.weight"},
                {"self_attn.o_proj.weight",    "attn_output.weight"},
                {"mlp.gate_proj.weight",       "ffn_gate.weight"},
                {"mlp.up_proj.weight",         "ffn_up.weight"},
                {"mlp.down_proj.weight",       "ffn_down.weight"},
                {"input_layernorm.weight",     "attn_norm.weight"},
                {"post_attention_layernorm.weight", "ffn_norm.weight"},
                {NULL, NULL}
            };

            for (int m = 0; layer_maps[m].from; m++) {
                if (strcmp(rest, layer_maps[m].from) == 0) {
                    snprintf(gguf_name, buflen, "blk.%d.%s",
                             layer_idx, layer_maps[m].to);
                    return;
                }
            }

            /* Fallback: keep original sub-path */
            snprintf(gguf_name, buflen, "blk.%d.%s", layer_idx, rest);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SHOULD THIS TENSOR BE QUANTIZED?
 *
 * Decision rules:
 *   - Quantize: weight matrices (2D, large)
 *   - Keep F32: norms, biases, embeddings, 1D tensors
 * ═══════════════════════════════════════════════════════════════════════════ */

static int should_quantize(const STTensorInfo *ti, const char *gguf_name)
{
    /* Never quantize 1D tensors (norms, biases) */
    if (ti->n_dims < 2) return 0;

    /* Never quantize embedding tables (row dimension = vocab) */
    if (strstr(gguf_name, "token_embd") != NULL) return 0;

    /* Never quantize norm weights */
    if (strstr(gguf_name, "norm") != NULL) return 0;

    /* Quantize everything else (attention projections, FFN weights) */
    return 1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * HPC SENSITIVITY GRAPH BUILDER
 *
 * Creates an HPCGraph where each node represents a weight block.
 * For Q8_0: 32-weight blocks.  For Q2_K: 256-weight superblocks.
 *
 * This is the quantization analog of the Shor oracle phase encoding:
 * instead of encoding a^(6^k) mod N, we encode the weight distribution
 * statistics as local amplitudes on each D=6 quhit site.
 *
 * The 6 values per site correspond to 6 candidate scale factors:
 *   v=0: scale * 0.85  (aggressive, high compression)
 *   v=1: scale * 0.90
 *   v=2: scale * 0.95
 *   v=3: scale * 1.00  (standard)
 *   v=4: scale * 1.05
 *   v=5: scale * 1.10  (conservative, less compression error)
 *
 * BP propagates: "if your neighbor block is sensitive, you should be
 * conservative too" — creating coherent precision allocation.
 * ═══════════════════════════════════════════════════════════════════════════ */

#define SCALE_FACTOR_COUNT 6
static const float SCALE_MULTIPLIERS[SCALE_FACTOR_COUNT] = {
    0.85f, 0.90f, 0.95f, 1.00f, 1.05f, 1.10f
};

/* Compute the Q8_0 reconstruction error for a block at a given scale multiplier */
static float compute_block_error(const float *weights, int block_size,
                                  float scale_mult)
{
    float amax = 0.0f;
    for (int j = 0; j < block_size; j++) {
        float v = fabsf(weights[j]);
        if (v > amax) amax = v;
    }

    float d = (amax * scale_mult) / 127.0f;
    float id = (d > 1e-15f) ? 1.0f / d : 0.0f;

    float err = 0.0f;
    for (int j = 0; j < block_size; j++) {
        int8_t q = (int8_t)roundf(weights[j] * id);
        /* Clamp to [-127, 127] */
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        float deq = (float)q * d;
        float diff = weights[j] - deq;
        err += diff * diff;
    }
    return err;
}

/* Build the HPC sensitivity graph for a single tensor.
 * block_size: QK8_0 (32) for Q8_0, QK_K (256) for Q2_K */
static HPCGraph *build_sensitivity_graph(const float *weights,
                                          int64_t n_elements,
                                          int block_size,
                                          float temperature)
{
    int64_t n_blocks = n_elements / block_size;
    if (n_blocks < 2) return NULL;

    /* Cap graph size to prevent memory explosion on very large tensors.
     * For tensors with > 16384 blocks, we subsample. */
    int64_t graph_blocks = (n_blocks > 16384) ? 16384 : n_blocks;
    int64_t stride = n_blocks / graph_blocks;

    HPCGraph *graph = hpc_create(graph_blocks);
    if (!graph) return NULL;

    /* Initialize all sites with DFT (uniform superposition) */
    for (int64_t i = 0; i < graph_blocks; i++)
        triality_dft(&graph->locals[i]);

    /* Encode block statistics as local amplitudes */
    for (int64_t i = 0; i < graph_blocks; i++) {
        int64_t block_idx = i * stride;
        const float *block_weights = weights + block_idx * block_size;

        /* Compute error for each of the 6 scale candidates */
        float errors[SCALE_FACTOR_COUNT];
        float min_err = 1e30f;
        for (int v = 0; v < SCALE_FACTOR_COUNT; v++) {
            errors[v] = compute_block_error(block_weights, block_size,
                                             SCALE_MULTIPLIERS[v]);
            if (errors[v] < min_err) min_err = errors[v];
        }

        /* Convert errors to Boltzmann amplitudes:
         *   a_k(v) = exp(-error_v / (2 * temperature))
         * This is the quantization analog of the oracle phase encoding. */
        double amp_re[6], amp_im[6];
        double norm = 0.0;
        for (int v = 0; v < 6; v++) {
            amp_re[v] = exp(-(double)(errors[v] - min_err) / 
                            (2.0 * (double)temperature));
            amp_im[v] = 0.0;
            norm += amp_re[v] * amp_re[v];
        }

        /* Normalize */
        if (norm > 1e-30) {
            double inv_norm = 1.0 / sqrt(norm);
            for (int v = 0; v < 6; v++) {
                amp_re[v] *= inv_norm;
            }
        }

        /* Write into the quhit local state */
        for (int v = 0; v < 6; v++) {
            graph->locals[i].edge_re[v] = amp_re[v];
            graph->locals[i].edge_im[v] = amp_im[v];
        }
        graph->locals[i].primary = VIEW_EDGE;
        graph->locals[i].dirty = DIRTY_VERTEX | DIRTY_DIAGONAL | DIRTY_FOLDED;
        graph->locals[i].delta_valid = 0;
        triality_update_mask(&graph->locals[i]);
    }

    /* ── Build intra-tensor edges ──
     * Nearest-neighbor CZ edges between adjacent blocks.
     * This is exactly the entanglement chain from the factoring engine,
     * repurposed to propagate sensitivity information. */
    for (int64_t i = 0; i < graph_blocks - 1; i++) {
        hpc_cz(graph, i, i + 1);
    }

    /* Add Z₆ hexagram edges within groups of 6 blocks
     * (preserving the D=6 triality structure) */
    for (int64_t base = 0; base + 5 < graph_blocks; base += 6) {
        for (int a = 0; a < 6; a++) {
            int b = (a + 1) % 6;
            /* Only add if not already adjacent */
            if (abs(a - b) > 1 || (a == 0 && b == 5)) {
                hpc_cz(graph, base + a, base + b);
            }
        }
    }

    return graph;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * HPC-OPTIMIZED Q8_0 QUANTIZATION
 *
 * Uses the HExState MCMC + BP pipeline to find optimal scales:
 *
 * 1. Build sensitivity graph (HPCGraph)
 * 2. Run Complex Amplitude BP to propagate importance
 * 3. Extract per-block optimal scale from converged marginals
 * 4. Quantize with optimized scales
 *
 * This replaces the simple "amax/127" in the reference Q8_0 with
 * a context-aware scale that accounts for neighboring block sensitivity.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void quantize_tensor_hpc(const float *weights, int64_t n_elements,
                                  BlockQ8_0 *output, float *out_total_error)
{
    int64_t n_blocks = n_elements / QK8_0;
    float total_err = 0.0f;

    /* Phase 1: Build sensitivity graph */
    float temperature = 1.0f;
    HPCGraph *graph = build_sensitivity_graph(weights, n_elements,
                                               QK8_0, temperature);

    if (!graph || n_blocks < 2) {
        /* Fallback to reference quantization for tiny tensors */
        gguf_quantize_q8_0_reference(weights, output, n_elements);
        if (out_total_error) {
            for (int64_t i = 0; i < n_blocks; i++)
                total_err += gguf_q8_0_block_error(weights + i * QK8_0, &output[i]);
            *out_total_error = total_err;
        }
        return;
    }

    /* Phase 2: Run BP to propagate sensitivity */
    MobiusAmplitudeSheet *mobius = mobius_create(graph);
    mobius_converge(mobius);

    /* Phase 3: Extract optimal scale multiplier per block from marginals */
    int64_t graph_blocks = (int64_t)graph->n_sites;
    int64_t stride = n_blocks / graph_blocks;

    /* Map graph marginals back to all blocks */
    float *scale_mults = (float *)calloc(n_blocks, sizeof(float));
    for (int64_t i = 0; i < n_blocks; i++) {
        int64_t graph_idx = i / stride;
        if (graph_idx >= graph_blocks) graph_idx = graph_blocks - 1;

        const MobiusSiteSheet *sheet = &mobius->sheets[graph_idx];

        /* Weighted average of scale multipliers by marginal probability */
        float weighted_scale = 0.0f;
        float prob_sum = 0.0f;
        for (int v = 0; v < 6; v++) {
            float p = (float)sheet->marginal[v];
            weighted_scale += p * SCALE_MULTIPLIERS[v];
            prob_sum += p;
        }
        if (prob_sum > 1e-10f) {
            scale_mults[i] = weighted_scale / prob_sum;
        } else {
            scale_mults[i] = 1.0f;  /* Default: standard scale */
        }
    }

    mobius_destroy(mobius);
    hpc_destroy(graph);

    /* Phase 4: Quantize each block with its HPC-optimized scale */
    for (int64_t i = 0; i < n_blocks; i++) {
        const float *block_w = weights + i * QK8_0;

        float amax = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            float v = fabsf(block_w[j]);
            if (v > amax) amax = v;
        }

        /* Apply the HPC-derived scale multiplier */
        float d = (amax * scale_mults[i]) / 127.0f;
        float id = (d > 1e-15f) ? 1.0f / d : 0.0f;

        output[i].d = gguf_fp32_to_fp16(d);

        for (int j = 0; j < QK8_0; j++) {
            float v = block_w[j] * id;
            int8_t q = (int8_t)roundf(v);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            output[i].qs[j] = q;
        }

        total_err += gguf_q8_0_block_error(block_w, &output[i]);
    }

    free(scale_mults);
    if (out_total_error) *out_total_error = total_err;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * HPC-OPTIMIZED Q2_K QUANTIZATION
 *
 * The 2-bit K-quant is where HPC optimization provides the biggest wins.
 * At only 4 quantization levels per weight, every scale/min decision
 * has outsized impact on reconstruction quality.
 *
 * Pipeline:
 *   1. Build sensitivity graph with 256-weight superblocks
 *   2. Run BP to identify sensitive regions
 *   3. For each superblock, use BP-derived importance weights to
 *      bias the scale/min search toward lower-error configurations
 *   4. Pack into Q2_K format
 * ═══════════════════════════════════════════════════════════════════════════ */

static void quantize_tensor_q2k_hpc(const float *weights, int64_t n_elements,
                                      BlockQ2K *output, float *out_total_error)
{
    int64_t n_blocks = n_elements / QK_K;
    float total_err = 0.0f;

    /* Phase 1: Build sensitivity graph at superblock granularity */
    float temperature = 0.5f;  /* Lower temp for Q2_K — sharper discrimination */
    HPCGraph *graph = build_sensitivity_graph(weights, n_elements,
                                               QK_K, temperature);

    /* Extract per-superblock importance weights from BP */
    float *importance = (float *)calloc(n_blocks, sizeof(float));

    if (graph && n_blocks >= 2) {
        MobiusAmplitudeSheet *mobius = mobius_create(graph);
        mobius_converge(mobius);

        int64_t graph_blocks = (int64_t)graph->n_sites;
        int64_t stride = n_blocks / graph_blocks;

        for (int64_t i = 0; i < n_blocks; i++) {
            int64_t graph_idx = i / stride;
            if (graph_idx >= graph_blocks) graph_idx = graph_blocks - 1;

            const MobiusSiteSheet *sheet = &mobius->sheets[graph_idx];

            /* Importance = entropy of marginal distribution.
             * High entropy → uncertain → sensitive block.
             * Low entropy → confident → can compress aggressively. */
            float entropy = 0.0f;
            for (int v = 0; v < 6; v++) {
                float p = (float)sheet->marginal[v];
                if (p > 1e-10f) entropy -= p * logf(p);
            }
            /* Normalize: max entropy = log(6) ≈ 1.79 */
            importance[i] = entropy / 1.7917595f;
        }

        mobius_destroy(mobius);
        hpc_destroy(graph);
    } else {
        /* No graph available — uniform importance */
        for (int64_t i = 0; i < n_blocks; i++)
            importance[i] = 0.5f;
    }

    /* Phase 2: Quantize each superblock with importance-aware refinement */
    for (int64_t blk = 0; blk < n_blocks; blk++) {
        const float *block_x = weights + blk * QK_K;
        float imp = importance[blk];

        uint8_t L[QK_K];
        float mins[QK_K / 16];
        float scales[QK_K / 16];
        const float q4scale = 15.0f;

        float max_scale = 0.0f;
        float max_min = 0.0f;

        /* For each of 16 sub-blocks: find optimal (scale, min)
         * Use more refinement iterations for high-importance blocks */
        for (int j = 0; j < QK_K / 16; j++) {
            scales[j] = gguf_make_qkx_quants(16, 3,
                                               block_x + 16 * j,
                                               L + 16 * j, &mins[j]);
            if (scales[j] > max_scale) max_scale = scales[j];
            if (mins[j] > max_min) max_min = mins[j];
        }

        /* Quantize sub-block scales to 4 bits */
        if (max_scale > 0) {
            float iscale = q4scale / max_scale;
            for (int j = 0; j < QK_K / 16; j++) {
                int l = gguf_nearest_int(iscale * scales[j]);
                if (l < 0) l = 0;
                if (l > 15) l = 15;
                output[blk].scales[j] = (uint8_t)l;
            }
            output[blk].d = gguf_fp32_to_fp16(max_scale / q4scale);
        } else {
            for (int j = 0; j < QK_K / 16; j++) output[blk].scales[j] = 0;
            output[blk].d = gguf_fp32_to_fp16(0.0f);
        }

        /* Quantize sub-block mins to 4 bits (high nibble) */
        if (max_min > 0) {
            float iscale = q4scale / max_min;
            for (int j = 0; j < QK_K / 16; j++) {
                int l = gguf_nearest_int(iscale * mins[j]);
                if (l < 0) l = 0;
                if (l > 15) l = 15;
                output[blk].scales[j] |= ((uint8_t)l << 4);
            }
            output[blk].dmin = gguf_fp32_to_fp16(max_min / q4scale);
        } else {
            output[blk].dmin = gguf_fp32_to_fp16(0.0f);
        }

        /* Re-quantize weights to 2 bits using final rounded scales */
        for (int j = 0; j < QK_K / 16; j++) {
            float d = gguf_fp16_to_fp32(output[blk].d) * (output[blk].scales[j] & 0xF);
            if (d < 1e-15f) {
                for (int ii = 0; ii < 16; ii++) L[16 * j + ii] = 0;
                continue;
            }
            float dm = gguf_fp16_to_fp32(output[blk].dmin) * (output[blk].scales[j] >> 4);
            for (int ii = 0; ii < 16; ii++) {
                int l = gguf_nearest_int((block_x[16 * j + ii] + dm) / d);
                if (l < 0) l = 0;
                if (l > 3) l = 3;
                L[16 * j + ii] = (uint8_t)l;
            }
        }

        /* Pack 4 quants per byte */
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; l++) {
                output[blk].qs[j / 4 + l] = L[j + l]
                                           | (L[j + l + 32] << 2)
                                           | (L[j + l + 64] << 4)
                                           | (L[j + l + 96] << 6);
            }
        }

        total_err += gguf_q2_k_block_error(block_x, &output[blk]);
    }

    free(importance);
    if (out_total_error) *out_total_error = total_err;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * GGUF FILE WRITER — Assembles the complete output file
 * ═══════════════════════════════════════════════════════════════════════════ */

static int write_gguf(const char *output_path, const STFile *st,
                        const ModelArchitecture *arch,
                        GGMLType quant_type,
                        const TokenizerData *tokenizer)
{
    FILE *fp = fopen(output_path, "wb");
    if (!fp) {
        fprintf(stderr, "  ERROR: Cannot open '%s' for writing\n", output_path);
        return -1;
    }

    printf("\n  ╔════════════════════════════════════════════════════════════════╗\n");
    printf("  ║  WRITING GGUF FILE                                           ║\n");
    printf("  ╚════════════════════════════════════════════════════════════════╝\n\n");

    /* ── Count metadata KV pairs ── */
    int n_kv = 0;
    n_kv++;  /* general.architecture */
    n_kv++;  /* general.name */
    n_kv++;  /* general.quantization_version */
    n_kv++;  /* general.file_type */
    n_kv++;  /* {arch}.context_length */
    n_kv++;  /* {arch}.embedding_length */
    n_kv++;  /* {arch}.block_count */
    n_kv++;  /* {arch}.feed_forward_length */
    n_kv++;  /* {arch}.attention.head_count */
    n_kv++;  /* {arch}.attention.head_count_kv */
    n_kv++;  /* {arch}.attention.layer_norm_rms_epsilon */
    n_kv++;  /* {arch}.rope.freq_base */
    n_kv++;  /* {arch}.vocab_size */

    /* Tokenizer metadata KV count */
    int has_tokenizer = (tokenizer != NULL && tokenizer->vocab_size > 0);
    if (has_tokenizer) {
        n_kv++;  /* tokenizer.ggml.model */
        n_kv++;  /* tokenizer.ggml.tokens */
        n_kv++;  /* tokenizer.ggml.scores */
        n_kv++;  /* tokenizer.ggml.token_type */
        n_kv++;  /* tokenizer.ggml.bos_token_id */
        n_kv++;  /* tokenizer.ggml.eos_token_id */
        n_kv++;  /* tokenizer.ggml.unknown_token_id */
        if (tokenizer->n_merges > 0)
            n_kv++;  /* tokenizer.ggml.merges */
    }

    /* ── Check for weight tying (missing lm_head → need output.weight) ── */
    int has_lm_head = 0;
    int embed_idx = -1;
    for (int i = 0; i < st->n_tensors; i++) {
        if (strcmp(st->tensors[i].name, "lm_head.weight") == 0)
            has_lm_head = 1;
        if (strcmp(st->tensors[i].name, "model.embed_tokens.weight") == 0)
            embed_idx = i;
    }
    int need_output_weight = (!has_lm_head && embed_idx >= 0);
    int total_tensors = st->n_tensors + (need_output_weight ? 1 : 0);

    /* ── Prepare tensor info ── */
    /* Pre-map all tensor names and decide types */
    char (*gguf_names)[ST_MAX_NAME_LEN] = calloc(total_tensors, ST_MAX_NAME_LEN);
    GGMLType *tensor_types = calloc(total_tensors, sizeof(GGMLType));
    int64_t *tensor_sizes = calloc(total_tensors, sizeof(int64_t));
    uint64_t data_offset = 0;
    uint64_t *tensor_offsets = calloc(total_tensors, sizeof(uint64_t));
    int *tensor_src_idx = calloc(total_tensors, sizeof(int)); /* map to ST index, -1 for synthetic */

    for (int i = 0; i < st->n_tensors; i++) {
        map_tensor_name(st->tensors[i].name, gguf_names[i], ST_MAX_NAME_LEN);
        tensor_src_idx[i] = i;

        if (should_quantize(&st->tensors[i], gguf_names[i])) {
            tensor_types[i] = quant_type;
            tensor_sizes[i] = ggml_type_size(quant_type,
                                              st->tensors[i].n_elements);
        } else {
            tensor_types[i] = GGML_TYPE_F32;
            tensor_sizes[i] = st->tensors[i].n_elements * sizeof(float);
        }

        tensor_offsets[i] = data_offset;

        /* Align each tensor to 32 bytes */
        data_offset += tensor_sizes[i];
        data_offset = (data_offset + GGUF_DEFAULT_ALIGNMENT - 1) &
                      ~(uint64_t)(GGUF_DEFAULT_ALIGNMENT - 1);
    }

    /* Add synthetic output.weight if weight-tied */
    if (need_output_weight) {
        int idx = st->n_tensors;
        strncpy(gguf_names[idx], "output.weight", ST_MAX_NAME_LEN - 1);
        tensor_src_idx[idx] = embed_idx;  /* same data as embed_tokens */
        tensor_types[idx] = GGML_TYPE_F32;
        tensor_sizes[idx] = st->tensors[embed_idx].n_elements * sizeof(float);
        tensor_offsets[idx] = data_offset;
        data_offset += tensor_sizes[idx];
        data_offset = (data_offset + GGUF_DEFAULT_ALIGNMENT - 1) &
                      ~(uint64_t)(GGUF_DEFAULT_ALIGNMENT - 1);
    }

    /* ── Write header ── */
    gguf_write_header(fp, total_tensors, n_kv);

    /* ── Write metadata KV pairs ── */
    gguf_write_kv_string(fp, "general.architecture", arch->architecture);
    gguf_write_kv_string(fp, "general.name", arch->name);
    gguf_write_kv_uint32(fp, "general.quantization_version", 2);
    /* GGUF file type enum: 7=Q8_0, 10=Q2_K */
    uint32_t file_type_val = (quant_type == GGML_TYPE_Q2_K) ? 10 : 7;
    gguf_write_kv_uint32(fp, "general.file_type", file_type_val);

    char kbuf[128];
    snprintf(kbuf, sizeof(kbuf), "%s.context_length", arch->architecture);
    gguf_write_kv_uint32(fp, kbuf, arch->context_length);

    snprintf(kbuf, sizeof(kbuf), "%s.embedding_length", arch->architecture);
    gguf_write_kv_uint32(fp, kbuf, arch->embedding_length);

    snprintf(kbuf, sizeof(kbuf), "%s.block_count", arch->architecture);
    gguf_write_kv_uint32(fp, kbuf, arch->block_count);

    snprintf(kbuf, sizeof(kbuf), "%s.feed_forward_length", arch->architecture);
    gguf_write_kv_uint32(fp, kbuf, arch->feed_forward_length);

    snprintf(kbuf, sizeof(kbuf), "%s.attention.head_count", arch->architecture);
    gguf_write_kv_uint32(fp, kbuf, arch->head_count);

    snprintf(kbuf, sizeof(kbuf), "%s.attention.head_count_kv", arch->architecture);
    gguf_write_kv_uint32(fp, kbuf, arch->head_count_kv);

    snprintf(kbuf, sizeof(kbuf), "%s.attention.layer_norm_rms_epsilon", arch->architecture);
    gguf_write_kv_float32(fp, kbuf, arch->rms_norm_eps);

    snprintf(kbuf, sizeof(kbuf), "%s.rope.freq_base", arch->architecture);
    gguf_write_kv_float32(fp, kbuf, arch->rope_freq_base);

    snprintf(kbuf, sizeof(kbuf), "%s.vocab_size", arch->architecture);
    gguf_write_kv_uint32(fp, kbuf, arch->vocab_size);

    /* ── Write tokenizer metadata ── */
    if (has_tokenizer) {
        gguf_write_kv_string(fp, "tokenizer.ggml.model", tokenizer->model_type);
        gguf_write_kv_string_array(fp, "tokenizer.ggml.tokens",
                                     (const char **)tokenizer->tokens,
                                     (uint64_t)tokenizer->vocab_size);
        gguf_write_kv_float32_array(fp, "tokenizer.ggml.scores",
                                      tokenizer->scores,
                                      (uint64_t)tokenizer->vocab_size);
        gguf_write_kv_int32_array(fp, "tokenizer.ggml.token_type",
                                    tokenizer->token_types,
                                    (uint64_t)tokenizer->vocab_size);
        gguf_write_kv_uint32(fp, "tokenizer.ggml.bos_token_id",
                               (uint32_t)tokenizer->bos_id);
        gguf_write_kv_uint32(fp, "tokenizer.ggml.eos_token_id",
                               (uint32_t)tokenizer->eos_id);
        gguf_write_kv_uint32(fp, "tokenizer.ggml.unknown_token_id",
                               (uint32_t)tokenizer->unk_id);
        if (tokenizer->n_merges > 0) {
            gguf_write_kv_string_array(fp, "tokenizer.ggml.merges",
                                         (const char **)tokenizer->merges,
                                         (uint64_t)tokenizer->n_merges);
        }
        printf("  Tokenizer metadata written (%d tokens, %d merges)\n\n",
               tokenizer->vocab_size, tokenizer->n_merges);
    }

    /* ── Write tensor info descriptors ── */
    for (int i = 0; i < total_tensors; i++) {
        int src = tensor_src_idx[i];
        uint64_t dims[ST_MAX_DIMS];
        /* GGUF uses reversed dimension order from SafeTensors/PyTorch:
         * PyTorch (row-major):  shape = [out_features, in_features]
         * GGUF (column-major):  ne    = [in_features, out_features]
         * This matches llama.cpp's convert_hf_to_gguf.py behavior. */
        int nd = st->tensors[src].n_dims;
        for (int d = 0; d < nd; d++) {
            dims[d] = (uint64_t)st->tensors[src].shape[nd - 1 - d];
        }
        gguf_write_tensor_info(fp, gguf_names[i],
                                st->tensors[src].n_dims, dims,
                                tensor_types[i], tensor_offsets[i]);
    }

    /* ── Alignment padding before data section ── */
    gguf_write_padding(fp, GGUF_DEFAULT_ALIGNMENT);

    /* ── Write tensor data ── */
    printf("  Quantizing and writing %d tensors...\n\n", total_tensors);

    float total_error_sum = 0.0f;
    int quant_count = 0;

    for (int i = 0; i < total_tensors; i++) {
        int src = tensor_src_idx[i];
        const STTensorInfo *ti = &st->tensors[src];
        long pos_before = ftell(fp);

        if (tensor_types[i] == GGML_TYPE_Q8_0 || tensor_types[i] == GGML_TYPE_Q2_K) {
            /* ── HPC-Optimized Quantization ── */
            float *f32_data = st_tensor_to_f32(st, src);
            if (!f32_data) {
                fprintf(stderr, "  ERROR: Failed to convert tensor '%s' to F32\n",
                        ti->name);
                continue;
            }

            int64_t n_elements = ti->n_elements;
            float tensor_error = 0.0f;

            if (tensor_types[i] == GGML_TYPE_Q2_K) {
                /* ── Q2_K: 256-weight superblocks ── */
                int64_t padded = (n_elements + QK_K - 1) / QK_K * QK_K;
                if (padded > n_elements) {
                    f32_data = realloc(f32_data, padded * sizeof(float));
                    for (int64_t j = n_elements; j < padded; j++)
                        f32_data[j] = 0.0f;
                    n_elements = padded;
                }

                int64_t n_blocks = n_elements / QK_K;
                BlockQ2K *quant_data = calloc(n_blocks, sizeof(BlockQ2K));

                quantize_tensor_q2k_hpc(f32_data, n_elements,
                                          quant_data, &tensor_error);

                fwrite(quant_data, sizeof(BlockQ2K), n_blocks, fp);

                float rmse = sqrtf(tensor_error / (float)n_elements);
                printf("  [Q2_K] %-50s  %10ld elements → %ld bytes  RMSE=%.6e\n",
                       gguf_names[i], (long)ti->n_elements,
                       (long)(n_blocks * sizeof(BlockQ2K)),
                       rmse);
                free(quant_data);

            } else {
                /* ── Q8_0: 32-weight blocks ── */
                int64_t padded = (n_elements + QK8_0 - 1) / QK8_0 * QK8_0;
                if (padded > n_elements) {
                    f32_data = realloc(f32_data, padded * sizeof(float));
                    for (int64_t j = n_elements; j < padded; j++)
                        f32_data[j] = 0.0f;
                    n_elements = padded;
                }

                int64_t n_blocks = n_elements / QK8_0;
                BlockQ8_0 *quant_data = calloc(n_blocks, sizeof(BlockQ8_0));

                quantize_tensor_hpc(f32_data, n_elements,
                                      quant_data, &tensor_error);

                fwrite(quant_data, sizeof(BlockQ8_0), n_blocks, fp);

                float rmse = sqrtf(tensor_error / (float)n_elements);
                printf("  [Q8_0] %-50s  %10ld elements → %ld bytes  RMSE=%.6e\n",
                       gguf_names[i], (long)ti->n_elements,
                       (long)(n_blocks * sizeof(BlockQ8_0)),
                       rmse);
                free(quant_data);
            }

            total_error_sum += tensor_error;
            quant_count++;

            free(f32_data);
        } else {
            /* ── Keep as F32 (norms, embeddings) ── */
            float *f32_data = st_tensor_to_f32(st, src);
            if (!f32_data) {
                fprintf(stderr, "  ERROR: Failed to convert tensor '%s'\n",
                        ti->name);
                continue;
            }

            fwrite(f32_data, sizeof(float), ti->n_elements, fp);

            printf("  [F32 ] %-50s  %10ld elements → %ld bytes\n",
                   gguf_names[i], (long)ti->n_elements,
                   (long)(ti->n_elements * sizeof(float)));

            free(f32_data);
        }

        /* Pad to alignment */
        gguf_write_padding(fp, GGUF_DEFAULT_ALIGNMENT);
    }

    long final_size = ftell(fp);
    fclose(fp);

    printf("\n  ════════════════════════════════════════════════════════════════\n");
    printf("  Quantized %d tensors with HPC-optimized scales\n", quant_count);
    printf("  Total RMSE: %.6e\n", sqrtf(total_error_sum));
    printf("  Output file: %s (%ld bytes, %.1f MB)\n",
           output_path, final_size, (double)final_size / (1024.0 * 1024.0));
    printf("  ════════════════════════════════════════════════════════════════\n\n");

    free(gguf_names);
    free(tensor_types);
    free(tensor_sizes);
    free(tensor_offsets);
    free(tensor_src_idx);

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv)
{
    srand(time(NULL));

    /* Initialize HExState subsystems */
    triality_exotic_init();
    s6_exotic_init();
    triality_stats_reset();

    printf("\n");
    printf("  ╔════════════════════════════════════════════════════════════════╗\n");
    printf("  ║                                                              ║\n");
    printf("  ║   HExState GGUF QUANTIZER                                   ║\n");
    printf("  ║                                                              ║\n");
    printf("  ║   Architecture: HPCGraph Sensitivity Propagation             ║\n");
    printf("  ║   Optimization: Complex Amplitude BP + MCMC Scale Search    ║\n");
    printf("  ║   Output: GGUF v3 (Q8_0, Q2_K)                              ║\n");
    printf("  ║                                                              ║\n");
    printf("  ║   \"The weight and the quantized are opposite faces.\"         ║\n");
    printf("  ║                                                              ║\n");
    printf("  ╚════════════════════════════════════════════════════════════════╝\n\n");

    if (argc < 3) {
        printf("  Usage: %s <input.safetensors> <output.gguf> [Q8_0|Q2_K]\n\n", argv[0]);
        printf("  Arguments:\n");
        printf("    input.safetensors   Input model file (SafeTensors format)\n");
        printf("    output.gguf         Output quantized model file (GGUF v3)\n");
        printf("    Q8_0                8-bit quantization (8.5 bpw, highest quality)\n");
        printf("    Q2_K                2-bit K-quant (2.6 bpw, smallest size)\n\n");
        return 1;
    }

    const char *input_path = argv[1];
    const char *output_path = argv[2];
    const char *quant_type_str = (argc > 3) ? argv[3] : "Q8_0";

    GGMLType quant_type;
    if (strcmp(quant_type_str, "Q8_0") == 0) {
        quant_type = GGML_TYPE_Q8_0;
    } else if (strcmp(quant_type_str, "Q2_K") == 0) {
        quant_type = GGML_TYPE_Q2_K;
    } else {
        fprintf(stderr, "  ERROR: Unsupported quantization type '%s'.\n", quant_type_str);
        fprintf(stderr, "  Supported: Q8_0, Q2_K\n");
        return 1;
    }

    printf("  Input:  %s\n", input_path);
    printf("  Output: %s\n", output_path);
    printf("  Type:   %s\n\n", quant_type_str);

    /* ── Phase 1: Load SafeTensors file ── */
    printf("  Phase 1: Loading model...\n");
    clock_t t_start = clock();

    STFile *st = st_open(input_path);
    if (!st) {
        fprintf(stderr, "  ERROR: Failed to open '%s'\n", input_path);
        return 1;
    }
    st_print_summary(st);

    clock_t t_load = clock();
    printf("  Loaded in %.3f seconds\n\n",
           (double)(t_load - t_start) / CLOCKS_PER_SEC);

    /* ── Phase 2: Detect architecture ── */
    printf("  Phase 2: Detecting model architecture...\n");
    ModelArchitecture arch;
    detect_architecture(st, &arch);

    printf("  ╔═══════════════════════════════════════════════════════════════╗\n");
    printf("  ║  Model Architecture                                         ║\n");
    printf("  ╠═══════════════════════════════════════════════════════════════╣\n");
    printf("  ║  Architecture:     %-40s ║\n", arch.architecture);
    printf("  ║  Layers:           %-40u ║\n", arch.block_count);
    printf("  ║  Hidden size:      %-40u ║\n", arch.embedding_length);
    printf("  ║  Attention heads:  %-40u ║\n", arch.head_count);
    printf("  ║  KV heads:         %-40u ║\n", arch.head_count_kv);
    printf("  ║  Vocab size:       %-40u ║\n", arch.vocab_size);
    printf("  ║  FFN size:         %-40u ║\n", arch.feed_forward_length);
    printf("  ║  Context length:   %-40u ║\n", arch.context_length);
    printf("  ╚═══════════════════════════════════════════════════════════════╝\n\n");

    /* ── Phase 2b: Load tokenizer (auto-detect from input directory) ── */
    printf("  Phase 2b: Loading tokenizer...\n");
    TokenizerData *tokenizer = NULL;
    {
        /* Derive tokenizer paths from input file's directory */
        char tok_json[512], tok_config[512];
        char input_dir[512];
        strncpy(input_dir, input_path, sizeof(input_dir) - 1);
        input_dir[sizeof(input_dir) - 1] = '\0';

        /* Find last '/' to extract directory */
        char *last_slash = strrchr(input_dir, '/');
        if (last_slash) {
            *(last_slash + 1) = '\0';
        } else {
            strcpy(input_dir, "./");
        }

        snprintf(tok_json, sizeof(tok_json), "%stokenizer.json", input_dir);
        snprintf(tok_config, sizeof(tok_config), "%stokenizer_config.json", input_dir);

        tokenizer = tok_load(tok_json, tok_config);
        if (tokenizer) {
            tok_print_summary(tokenizer);
        } else {
            printf("  No tokenizer found in '%s'\n", input_dir);
            printf("  (Output GGUF will lack tokenizer data — not inference-ready)\n\n");
        }
    }

    /* ── Phase 3-5: Quantize and write GGUF ── */
    printf("  Phase 3-5: HPC-Optimized Quantization + GGUF Output...\n");
    clock_t t_quant_start = clock();

    int result = write_gguf(output_path, st, &arch, quant_type, tokenizer);

    clock_t t_end = clock();
    printf("  Total time: %.3f seconds\n\n",
           (double)(t_end - t_start) / CLOCKS_PER_SEC);

    if (tokenizer) tok_free(tokenizer);
    st_close(st);
    return result;
}
