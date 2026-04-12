/* ═══════════════════════════════════════════════════════════════════════════
 * hexstate_quantize.c — HExState GGUF Quantizer
 *
 * ╔═══════════════════════════════════════════════════════════════╗
 * ║  HPC-Optimized GGUF Quantization Engine                      ║
 * ║                                                               ║
 * ║  Architecture: HPCGraph Sensitivity Propagation               ║
 * ║  Optimization: Complex Amplitude BP + MCMC Scale Search       ║
 * ║  Enhancements: MSE Grid Search, Importance Matrix Weighting   ║
 * ║  Output: GGUF v3 (Q2_K)                                       ║
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
 * Additional techniques ported from llm-compressor:
 *   MSE grid search           →  Optimal min/max range shrinking
 *   Importance matrix (imatrix) →  Per-channel error weighting
 *
 * Build:
 *   make -f Makefile.quantize
 *
 * Usage:
 *   ./hexstate_quantize <input> <output.gguf> [options]
 *
 * Input can be:
 *   - A single .safetensors file
 *   - A model directory containing sharded .safetensors files
 *
 * Options:
 *   --optimizer hpc|mse|hybrid   Scale optimization strategy (default: hybrid)
 *   --imatrix <file>             Importance matrix for weighted quantization
 *   --verbose                    Per-block diagnostics
 * ═══════════════════════════════════════════════════════════════════════════ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
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
#include "imatrix_reader.h"

#define D 6  /* Preserved from HExState — the triality dimension */

/* ═══════════════════════════════════════════════════════════════════════════
 * OPTIMIZER MODE
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef enum {
    OPT_HPC,     /* HExState BP only          */
    OPT_MSE,     /* MSE grid search only      */
    OPT_HYBRID   /* HPC sensitivity + MSE     */
} OptimizerMode;

/* ═══════════════════════════════════════════════════════════════════════════
 * MODEL ARCHITECTURE AUTO-DETECTION
 *
 * Infers model architecture metadata from tensor names and shapes.
 * Supports: LLaMA, Mistral, Qwen2, Phi-3, Gemma, GPT-NeoX, Falcon, DeepSeek
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    char     architecture[64];   /* "llama", "phi3", "gemma", etc.  */
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
    int      has_bias;           /* Whether attention has biases    */
    int      tie_word_embeddings; /* Whether output = embed_tokens  */
} ModelArchitecture;

/* Count tensor names matching a pattern prefix */
static int count_tensors_with_prefix(const STMultiFile *mf, const char *prefix)
{
    int count = 0;
    int prefix_len = strlen(prefix);
    for (int i = 0; i < mf->n_tensors; i++) {
        if (strncmp(mf->tensor_map[i].name, prefix, prefix_len) == 0)
            count++;
    }
    return count;
}

/* Find max layer index from tensor names like "model.layers.N.xxx" */
static int find_max_layer_index(const STMultiFile *mf, const char *layer_prefix)
{
    int max_idx = -1;
    int prefix_len = strlen(layer_prefix);
    for (int i = 0; i < mf->n_tensors; i++) {
        if (strncmp(mf->tensor_map[i].name, layer_prefix, prefix_len) == 0) {
            int idx = atoi(mf->tensor_map[i].name + prefix_len);
            if (idx > max_idx) max_idx = idx;
        }
    }
    return max_idx;
}

/* ── Config.json reader for definitive architecture parameters ── */

typedef struct {
    int      valid;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t num_attention_heads;
    uint32_t num_key_value_heads;
    uint32_t num_hidden_layers;
    uint32_t vocab_size;
    uint32_t max_position_embeddings;
    float    rope_theta;
    float    rms_norm_eps;
    char     model_type[64];
    int      tie_word_embeddings;
} ConfigJson;

static ConfigJson parse_config_json(const char *path)
{
    ConfigJson cfg;
    memset(&cfg, 0, sizeof(cfg));

    FILE *f = fopen(path, "rb");
    if (!f) return cfg;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *json = (char *)malloc(size + 1);
    if (!json) { fclose(f); return cfg; }
    fread(json, 1, size, f);
    json[size] = '\0';
    fclose(f);

    cfg.valid = 1;

    /* Simple key-value extraction */
    const char *p;

    p = tok_find_key(json, "hidden_size");
    if (p) cfg.hidden_size = (uint32_t)strtol(p, NULL, 10);

    p = tok_find_key(json, "intermediate_size");
    if (p) cfg.intermediate_size = (uint32_t)strtol(p, NULL, 10);

    p = tok_find_key(json, "num_attention_heads");
    if (p) cfg.num_attention_heads = (uint32_t)strtol(p, NULL, 10);

    p = tok_find_key(json, "num_key_value_heads");
    if (p) cfg.num_key_value_heads = (uint32_t)strtol(p, NULL, 10);

    p = tok_find_key(json, "num_hidden_layers");
    if (p) cfg.num_hidden_layers = (uint32_t)strtol(p, NULL, 10);

    p = tok_find_key(json, "vocab_size");
    if (p) cfg.vocab_size = (uint32_t)strtol(p, NULL, 10);

    p = tok_find_key(json, "max_position_embeddings");
    if (p) cfg.max_position_embeddings = (uint32_t)strtol(p, NULL, 10);

    p = tok_find_key(json, "rope_theta");
    if (p) cfg.rope_theta = (float)strtod(p, NULL);

    p = tok_find_key(json, "rms_norm_eps");
    if (p) cfg.rms_norm_eps = (float)strtod(p, NULL);

    p = tok_find_key(json, "model_type");
    if (p && *p == '"') {
        char buf[64];
        tok_extract_string(p, buf, sizeof(buf));
        strncpy(cfg.model_type, buf, sizeof(cfg.model_type) - 1);
    }

    p = tok_find_key(json, "tie_word_embeddings");
    if (p) cfg.tie_word_embeddings = (strncmp(p, "true", 4) == 0);

    free(json);
    return cfg;
}

static void detect_architecture(const STMultiFile *mf, ModelArchitecture *arch,
                                  const char *config_json_path)
{
    memset(arch, 0, sizeof(*arch));

    /* Default values */
    strcpy(arch->architecture, "llama");
    strcpy(arch->name, "HExState-quantized");
    arch->context_length = 4096;
    arch->rope_freq_base = 10000.0f;
    arch->rms_norm_eps = 1e-5f;

    /* ── Try config.json for definitive parameters ── */
    ConfigJson cfg = {0};
    if (config_json_path) {
        cfg = parse_config_json(config_json_path);
    }

    if (cfg.valid) {
        /* Map model_type to GGUF architecture name */
        if (strcmp(cfg.model_type, "llama") == 0 ||
            strcmp(cfg.model_type, "mistral") == 0) {
            strcpy(arch->architecture, "llama");
        } else if (strcmp(cfg.model_type, "qwen2") == 0) {
            strcpy(arch->architecture, "qwen2");
        } else if (strcmp(cfg.model_type, "qwen2_moe") == 0) {
            strcpy(arch->architecture, "qwen2moe");
        } else if (strcmp(cfg.model_type, "phi3") == 0 ||
                   strcmp(cfg.model_type, "phi") == 0) {
            strcpy(arch->architecture, "phi3");
        } else if (strcmp(cfg.model_type, "gemma") == 0 ||
                   strcmp(cfg.model_type, "gemma2") == 0) {
            strcpy(arch->architecture, "gemma");
        } else if (strcmp(cfg.model_type, "deepseek_v2") == 0) {
            strcpy(arch->architecture, "llama");
        } else if (strcmp(cfg.model_type, "gpt_neox") == 0) {
            strcpy(arch->architecture, "gpt_neox");
        } else if (strcmp(cfg.model_type, "falcon") == 0) {
            strcpy(arch->architecture, "falcon");
        } else if (cfg.model_type[0]) {
            /* Unknown — try llama as fallback */
            strcpy(arch->architecture, "llama");
        }

        if (cfg.hidden_size) arch->embedding_length = cfg.hidden_size;
        if (cfg.intermediate_size) arch->feed_forward_length = cfg.intermediate_size;
        if (cfg.num_attention_heads) arch->head_count = cfg.num_attention_heads;
        if (cfg.num_key_value_heads) arch->head_count_kv = cfg.num_key_value_heads;
        if (cfg.num_hidden_layers) arch->block_count = cfg.num_hidden_layers;
        if (cfg.vocab_size) arch->vocab_size = cfg.vocab_size;
        if (cfg.max_position_embeddings) arch->context_length = cfg.max_position_embeddings;
        if (cfg.rope_theta > 0) arch->rope_freq_base = cfg.rope_theta;
        if (cfg.rms_norm_eps > 0) arch->rms_norm_eps = cfg.rms_norm_eps;
        arch->tie_word_embeddings = cfg.tie_word_embeddings;

        printf("  Architecture determined from config.json: %s\n", cfg.model_type);
    }

    /* ── Fall back to tensor name pattern detection ── */
    int has_model_layers = count_tensors_with_prefix(mf, "model.layers.");
    int has_gpt_neox = count_tensors_with_prefix(mf, "gpt_neox.");
    int has_transformer = count_tensors_with_prefix(mf, "transformer.");

    /* Architecture-specific detection */
    int has_qkv_proj = count_tensors_with_prefix(mf, "model.layers.0.self_attn.qkv_proj");
    int has_kv_a_proj = count_tensors_with_prefix(mf, "model.layers.0.self_attn.kv_a_proj_with_mqa");
    int has_final_norm = (st_multi_find_tensor(mf, "model.final_norm.weight") >= 0);

    if (has_qkv_proj > 0 && !cfg.valid) {
        strcpy(arch->architecture, "phi3");
    } else if (has_kv_a_proj > 0 && !cfg.valid) {
        strcpy(arch->architecture, "llama");  /* DeepSeek uses llama arch */
    } else if (has_final_norm && !cfg.valid) {
        strcpy(arch->architecture, "gemma");
    }

    if (has_model_layers > 0 && arch->block_count == 0) {
        arch->block_count = find_max_layer_index(mf, "model.layers.") + 1;
    }

    /* Infer dimensions from tensor shapes if not from config.json */
    if (arch->embedding_length == 0 || arch->head_count == 0) {
        int qproj_idx = st_multi_find_tensor(mf, "model.layers.0.self_attn.q_proj.weight");
        int kproj_idx = st_multi_find_tensor(mf, "model.layers.0.self_attn.k_proj.weight");

        if (qproj_idx >= 0) {
            const STTensorInfo *ti = st_multi_tensor_info(mf, qproj_idx);
            int64_t q_out = ti->shape[0];
            int64_t hidden = ti->shape[1];
            if (arch->embedding_length == 0) arch->embedding_length = hidden;

            /* Try common head dimensions: 128, 64, 96 */
            int head_dim = 128;
            if (q_out % 128 == 0) head_dim = 128;
            else if (q_out % 96 == 0) head_dim = 96;
            else if (q_out % 64 == 0) head_dim = 64;

            if (arch->head_count == 0) arch->head_count = q_out / head_dim;

            if (kproj_idx >= 0 && arch->head_count_kv == 0) {
                const STTensorInfo *kt = st_multi_tensor_info(mf, kproj_idx);
                arch->head_count_kv = kt->shape[0] / head_dim;
            }
        }
    }

    if (arch->vocab_size == 0) {
        int embed_idx = st_multi_find_tensor(mf, "model.embed_tokens.weight");
        if (embed_idx >= 0) {
            const STTensorInfo *ti = st_multi_tensor_info(mf, embed_idx);
            arch->vocab_size = ti->shape[0];
        }
    }

    if (arch->feed_forward_length == 0) {
        int gate_idx = st_multi_find_tensor(mf, "model.layers.0.mlp.gate_proj.weight");
        if (gate_idx >= 0) {
            const STTensorInfo *ti = st_multi_tensor_info(mf, gate_idx);
            arch->feed_forward_length = ti->shape[0];
        } else {
            int up_idx = st_multi_find_tensor(mf, "model.layers.0.mlp.up_proj.weight");
            if (up_idx >= 0) {
                const STTensorInfo *ti = st_multi_tensor_info(mf, up_idx);
                arch->feed_forward_length = ti->shape[0];
            }
        }
    }

    /* Check for attention bias */
    arch->has_bias = (st_multi_find_tensor(mf, "model.layers.0.self_attn.q_proj.bias") >= 0);

    if (has_gpt_neox > 0 && arch->block_count == 0) {
        strcpy(arch->architecture, "gpt_neox");
        arch->block_count = find_max_layer_index(mf, "gpt_neox.layers.") + 1;
    }
    if (has_transformer > 0 && arch->block_count == 0) {
        strcpy(arch->architecture, "falcon");
        arch->block_count = find_max_layer_index(mf, "transformer.h.") + 1;
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
 *
 * Enhanced with mappings for Phi-3, Gemma, DeepSeek, MoE, and bias tensors.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Returns 1 if this tensor should be skipped (not written to GGUF) */
static int should_skip_tensor(const char *hf_name)
{
    /* Rotary embeddings are computed at runtime, not stored */
    if (strstr(hf_name, "rotary_emb.inv_freq") != NULL) return 1;
    if (strstr(hf_name, "rotary_emb.cos_cached") != NULL) return 1;
    if (strstr(hf_name, "rotary_emb.sin_cached") != NULL) return 1;
    return 0;
}

static void map_tensor_name(const char *hf_name, char *gguf_name, int buflen)
{
    /* Start with identity mapping */
    strncpy(gguf_name, hf_name, buflen - 1);
    gguf_name[buflen - 1] = '\0';

    /* Top-level mappings (common to all architectures) */
    struct { const char *from; const char *to; } mappings[] = {
        {"model.embed_tokens.weight",       "token_embd.weight"},
        {"model.norm.weight",               "output_norm.weight"},
        {"model.final_norm.weight",         "output_norm.weight"},  /* Gemma */
        {"lm_head.weight",                  "output.weight"},
        {"model.embed_tokens.bias",         "token_embd.bias"},
        {"model.norm.bias",                 "output_norm.bias"},
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
                /* Standard attention projections */
                {"self_attn.q_proj.weight",         "attn_q.weight"},
                {"self_attn.k_proj.weight",         "attn_k.weight"},
                {"self_attn.v_proj.weight",         "attn_v.weight"},
                {"self_attn.o_proj.weight",         "attn_output.weight"},
                /* Attention biases */
                {"self_attn.q_proj.bias",           "attn_q.bias"},
                {"self_attn.k_proj.bias",           "attn_k.bias"},
                {"self_attn.v_proj.bias",           "attn_v.bias"},
                {"self_attn.o_proj.bias",           "attn_output.bias"},
                /* Phi-3 fused QKV */
                {"self_attn.qkv_proj.weight",       "attn_qkv.weight"},
                {"self_attn.qkv_proj.bias",         "attn_qkv.bias"},
                /* DeepSeek MLA */
                {"self_attn.kv_a_proj_with_mqa.weight", "attn_kv_a_mqa.weight"},
                {"self_attn.kv_b_proj.weight",      "attn_kv_b.weight"},
                /* Standard FFN (SwiGLU) */
                {"mlp.gate_proj.weight",            "ffn_gate.weight"},
                {"mlp.up_proj.weight",              "ffn_up.weight"},
                {"mlp.down_proj.weight",            "ffn_down.weight"},
                /* FFN biases */
                {"mlp.gate_proj.bias",              "ffn_gate.bias"},
                {"mlp.up_proj.bias",                "ffn_up.bias"},
                {"mlp.down_proj.bias",              "ffn_down.bias"},
                /* MoE gate */
                {"mlp.gate.weight",                 "ffn_gate_inp.weight"},
                /* MoE expert weights */
                {"mlp.experts.gate_proj.weight",    "ffn_gate_exps.weight"},
                {"mlp.experts.up_proj.weight",      "ffn_up_exps.weight"},
                {"mlp.experts.down_proj.weight",    "ffn_down_exps.weight"},
                /* Norm layers */
                {"input_layernorm.weight",          "attn_norm.weight"},
                {"post_attention_layernorm.weight", "ffn_norm.weight"},
                {"input_layernorm.bias",            "attn_norm.bias"},
                {"post_attention_layernorm.bias",   "ffn_norm.bias"},
                /* Gemma pre/post feedforward norm */
                {"pre_feedforward_layernorm.weight", "ffn_norm.weight"},
                {"post_feedforward_layernorm.weight", "ffn_post_norm.weight"},
                {NULL, NULL}
            };

            for (int m = 0; layer_maps[m].from; m++) {
                if (strcmp(rest, layer_maps[m].from) == 0) {
                    snprintf(gguf_name, buflen, "blk.%d.%s",
                             layer_idx, layer_maps[m].to);
                    return;
                }
            }

            /* MoE expert layer mapping: model.layers.N.mlp.experts.E.xxx */
            int expert_idx;
            char expert_rest[ST_MAX_NAME_LEN];
            if (sscanf(rest, "mlp.experts.%d.%255s", &expert_idx, expert_rest) == 2) {
                struct { const char *from; const char *to; } expert_maps[] = {
                    {"gate_proj.weight", "ffn_gate_exp.weight"},
                    {"up_proj.weight",   "ffn_up_exp.weight"},
                    {"down_proj.weight", "ffn_down_exp.weight"},
                    {NULL, NULL}
                };
                for (int m = 0; expert_maps[m].from; m++) {
                    if (strcmp(expert_rest, expert_maps[m].from) == 0) {
                        snprintf(gguf_name, buflen, "blk.%d.%s.%d",
                                 layer_idx, expert_maps[m].to, expert_idx);
                        return;
                    }
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

    /* Never quantize LM head output — use exact match, not substring,
     * to avoid matching "attn_output.weight" */
    if (strcmp(gguf_name, "output.weight") == 0) return 0;

    /* Never quantize norm weights */
    if (strstr(gguf_name, "norm") != NULL) return 0;

    /* Never quantize bias tensors */
    if (strstr(gguf_name, ".bias") != NULL) return 0;

    /* Never quantize MoE gate routing weights */
    if (strstr(gguf_name, "ffn_gate_inp") != NULL) return 0;

    /* Quantize everything else (attention projections, FFN weights) */
    return 1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * HPC SENSITIVITY GRAPH BUILDER
 *
 * Creates an HPCGraph where each node represents a weight block.
 * For Q2_K: 256-weight superblocks.
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

/* Compute the Q2_K sub-block reconstruction error for a block at a given
 * scale multiplier, optionally weighted by importance vector */
static float compute_block_error_q2k(const float *weights, int block_size,
                                       float scale_mult,
                                       const float *importance, int imp_offset)
{
    float min_val = weights[0];
    float max_val = weights[0];
    for (int j = 1; j < block_size; j++) {
        if (weights[j] < min_val) min_val = weights[j];
        if (weights[j] > max_val) max_val = weights[j];
    }
    if (min_val > 0) min_val = 0;

    float range = (max_val - min_val) * scale_mult;
    float d = range / 3.0f;
    if (d < 1e-15f) return 0.0f;

    float err = 0.0f;
    for (int j = 0; j < block_size; j++) {
        int q = gguf_nearest_int((weights[j] - min_val) / d);
        if (q < 0) q = 0;
        if (q > 3) q = 3;
        float deq = min_val + (float)q * d;
        float diff = weights[j] - deq;
        float w = (importance && imp_offset + j < block_size * 256) ?
                  importance[imp_offset + j] : 1.0f;
        err += diff * diff * w;
    }
    return err;
}

/* Build the HPC sensitivity graph for a single tensor.
 * block_size: QK_K (256) for Q2_K */
static HPCGraph *build_sensitivity_graph(const float *weights,
                                           int64_t n_elements,
                                           int block_size,
                                           float temperature,
                                           const float *importance)
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
            errors[v] = compute_block_error_q2k(block_weights, block_size,
                                                  SCALE_MULTIPLIERS[v],
                                                  importance,
                                                  (int)(block_idx * block_size));
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
     * Nearest-neighbor CZ edges between adjacent blocks. */
    for (int64_t i = 0; i < graph_blocks - 1; i++) {
        hpc_cz(graph, i, i + 1);
    }

    /* Add Z₆ hexagram edges within groups of 6 blocks
     * (preserving the D=6 triality structure) */
    for (int64_t base = 0; base + 5 < graph_blocks; base += 6) {
        for (int a = 0; a < 6; a++) {
            int b = (a + 1) % 6;
            if (abs(a - b) > 1 || (a == 0 && b == 5)) {
                hpc_cz(graph, base + a, base + b);
            }
        }
    }

    return graph;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MSE GRID SEARCH (ported from llm-compressor observers/mse.py)
 *
 * For a Q2_K sub-block, progressively shrink the min/max range to find
 * the candidate that minimizes weighted reconstruction error.
 *
 *   for p in [1.0, 1.0 - 1/grid, 1.0 - 2/grid, ...] down to (1 - maxshrink):
 *     candidate_min = p * min
 *     candidate_max = p * max
 *     error = ||x - quantize(x, candidate_min, candidate_max)||^norm
 *     if error < best: update best
 *     else: patience--; if patience == 0: break
 *
 * This is a direct C port of llm-compressor's _grid_search_mse.
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    float maxshrink;    /* Maximum shrink factor (0.0 to 1.0)         */
    int   grid;         /* Number of grid divisions                   */
    int   patience;     /* Early stopping patience                    */
    float norm;         /* Error norm exponent (2.0 = MSE, 2.4 = ...)*/
} MSEGridConfig;

static const MSEGridConfig MSE_DEFAULT_CONFIG = {
    .maxshrink = 0.20f,
    .grid      = 100,
    .patience  = 5,
    .norm      = 2.4f
};

/* Grid search for optimal scale/min for a Q2_K sub-block of n weights
 * with nmax = 3 quantization levels.
 * Returns optimized scale; stores absolute min in *out_min.
 * importance: per-element weights (can be NULL for uniform). */
static float mse_grid_search_q2k_subblock(const float *x, int n, int nmax,
                                            uint8_t *L, float *out_min,
                                            const float *importance,
                                            const MSEGridConfig *cfg)
{
    float min_val = x[0], max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] < min_val) min_val = x[i];
        if (x[i] > max_val) max_val = x[i];
    }
    if (max_val == min_val) {
        for (int i = 0; i < n; i++) L[i] = 0;
        *out_min = -min_val;
        return 0.0f;
    }
    if (min_val > 0) min_val = 0;

    float best_scale = 0.0f;
    float best_min = -min_val;
    float best_error = 1e30f;
    int no_improve = 0;

    int shrink_steps = (int)(cfg->maxshrink * cfg->grid);
    if (shrink_steps < 1) shrink_steps = 1;

    for (int step = 0; step <= shrink_steps; step++) {
        float p = 1.0f - (float)step / (float)cfg->grid;

        float cand_min = p * min_val;
        float cand_max = p * max_val;

        if (cand_max <= cand_min) continue;

        float iscale = (float)nmax / (cand_max - cand_min);
        float scale = 1.0f / iscale;

        /* Quantize and measure error */
        float err = 0.0f;
        uint8_t tmp_L[256];
        for (int i = 0; i < n; i++) {
            int l = gguf_nearest_int(iscale * (x[i] - cand_min));
            if (l < 0) l = 0;
            if (l > nmax) l = nmax;
            tmp_L[i] = (uint8_t)l;

            float deq = cand_min + scale * (float)l;
            float diff = fabsf(x[i] - deq);
            /* Apply error norm */
            float e = diff;
            if (cfg->norm != 1.0f) {
                e = powf(diff, cfg->norm);
            }
            /* Apply importance weighting */
            if (importance) e *= importance[i];
            err += e;
        }

        if (err < best_error) {
            best_error = err;
            best_scale = scale;
            best_min = -cand_min;
            memcpy(L, tmp_L, n);
            no_improve = 0;
        } else {
            no_improve++;
            if (no_improve >= cfg->patience) break;
        }
    }

    /* Iterative refinement on the best candidate (from ggml) */
    float cur_min = -best_min;
    float cur_scale = best_scale;
    if (cur_scale > 1e-15f) {
        float iscale = 1.0f / cur_scale;
        for (int itry = 0; itry < 3; itry++) {
            float sumlx = 0;
            int suml2 = 0;
            for (int i = 0; i < n; i++) {
                int l = gguf_nearest_int(iscale * (x[i] - cur_min));
                if (l < 0) l = 0;
                if (l > nmax) l = nmax;
                L[i] = (uint8_t)l;
                sumlx += (x[i] - cur_min) * l;
                suml2 += l * l;
            }
            if (suml2 > 0) cur_scale = sumlx / suml2;
            float sum = 0;
            for (int i = 0; i < n; i++)
                sum += x[i] - cur_scale * L[i];
            cur_min = 0.7f * cur_min + 0.3f * sum / n;
            if (cur_min > 0) cur_min = 0;
            if (cur_scale > 1e-15f) iscale = 1.0f / cur_scale;
        }
    }

    *out_min = -cur_min;
    return cur_scale;
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
 *   4. Apply MSE grid search for per-sub-block scale/min optimization
 *   5. Pack into Q2_K format
 * ═══════════════════════════════════════════════════════════════════════════ */

static void quantize_tensor_q2k_hpc(const float *weights, int64_t n_elements,
                                      BlockQ2K *output, float *out_total_error,
                                      OptimizerMode opt_mode,
                                      const float *imat_importance,
                                      int verbose)
{
    int64_t n_blocks = n_elements / QK_K;
    float total_err = 0.0f;

    /* Phase 1: Build sensitivity graph at superblock granularity */
    float *block_importance = NULL;

    if (opt_mode == OPT_HPC || opt_mode == OPT_HYBRID) {
        float temperature = 0.5f;  /* Lower temp for Q2_K — sharper discrimination */
        HPCGraph *graph = build_sensitivity_graph(weights, n_elements,
                                                    QK_K, temperature,
                                                    imat_importance);

        /* Extract per-superblock importance weights from BP */
        block_importance = (float *)calloc(n_blocks, sizeof(float));

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
                block_importance[i] = entropy / 1.7917595f;
            }

            mobius_destroy(mobius);
            hpc_destroy(graph);
        } else {
            for (int64_t i = 0; i < n_blocks; i++)
                block_importance[i] = 0.5f;
        }
    }

    /* Phase 2: Quantize each superblock */
    MSEGridConfig mse_cfg = MSE_DEFAULT_CONFIG;

    for (int64_t blk = 0; blk < n_blocks; blk++) {
        const float *block_x = weights + blk * QK_K;

        /* Adjust MSE grid search parameters based on HPC importance */
        MSEGridConfig local_cfg = mse_cfg;
        if (block_importance) {
            float imp = block_importance[blk];
            /* Higher importance → more search effort, less aggressive shrink */
            if (imp > 0.7f) {
                local_cfg.maxshrink = 0.10f;  /* Less shrinking for sensitive blocks */
                local_cfg.grid = 200;          /* Finer grid */
                local_cfg.patience = 10;       /* More patience */
            } else if (imp < 0.3f) {
                local_cfg.maxshrink = 0.30f;  /* More aggressive for insensitive blocks */
                local_cfg.grid = 50;
                local_cfg.patience = 3;
            }
        }

        uint8_t L[QK_K];
        float mins[QK_K / 16];
        float scales[QK_K / 16];
        const float q4scale = 15.0f;

        float max_scale = 0.0f;
        float max_min = 0.0f;

        /* Sub-block importance: combine imatrix and block-level HPC importance */
        const float *sub_imp = NULL;
        float local_imp_buf[16];

        for (int j = 0; j < QK_K / 16; j++) {
            const float *sub_x = block_x + 16 * j;

            /* Build per-element importance for this sub-block */
            if (imat_importance) {
                int64_t base_idx = blk * QK_K + 16 * j;
                for (int k = 0; k < 16; k++) {
                    int64_t g_idx = base_idx + k;
                    local_imp_buf[k] = (g_idx < n_elements) ?
                                       imat_importance[g_idx % (n_elements / QK_K > 0 ? n_elements : 1)] :
                                       1.0f;
                }
                sub_imp = local_imp_buf;
            }

            if (opt_mode == OPT_MSE || opt_mode == OPT_HYBRID) {
                /* MSE grid search (ported from llm-compressor) */
                scales[j] = mse_grid_search_q2k_subblock(
                    sub_x, 16, 3, L + 16 * j, &mins[j],
                    sub_imp, &local_cfg);
            } else {
                /* Reference quantization (simple ggml approach) */
                scales[j] = gguf_make_qkx_quants(16, 3,
                                                    sub_x,
                                                    L + 16 * j, &mins[j]);
            }

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

    free(block_importance);
    if (out_total_error) *out_total_error = total_err;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PROGRESS REPORTING
 * ═══════════════════════════════════════════════════════════════════════════ */

static void print_progress_bar(int current, int total, const char *label,
                                 clock_t start_time)
{
    if (total <= 0) return;
    float pct = (float)current / (float)total;
    int bar_width = 40;
    int filled = (int)(pct * bar_width);

    double elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    double eta = (pct > 0.01f) ? elapsed / pct * (1.0 - pct) : 0.0;

    printf("\r  [");
    for (int i = 0; i < bar_width; i++) {
        if (i < filled) printf("█");
        else if (i == filled) printf("▓");
        else printf("░");
    }
    printf("] %3d%% (%d/%d) %.0fs ETA:%.0fs  %s",
           (int)(pct * 100), current, total, elapsed, eta, label);
    fflush(stdout);

    if (current == total) printf("\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * GGUF FILE WRITER — Assembles the complete output file
 * ═══════════════════════════════════════════════════════════════════════════ */

static int write_gguf(const char *output_path, const STMultiFile *mf,
                        const ModelArchitecture *arch,
                        const TokenizerData *tokenizer,
                        OptimizerMode opt_mode,
                        const IMatrixData *imatrix,
                        int verbose)
{
    FILE *fp = fopen(output_path, "wb");
    if (!fp) {
        fprintf(stderr, "  ERROR: Cannot open '%s' for writing\n", output_path);
        return -1;
    }

    printf("\n  ╔════════════════════════════════════════════════════════════════╗\n");
    printf("  ║  WRITING GGUF FILE                                           ║\n");
    printf("  ╚════════════════════════════════════════════════════════════════╝\n\n");

    /* ── Determine which tensors to include ── */
    int *include_list = (int *)calloc(mf->n_tensors, sizeof(int));
    int n_include = 0;
    for (int i = 0; i < mf->n_tensors; i++) {
        if (!should_skip_tensor(mf->tensor_map[i].name)) {
            include_list[n_include++] = i;
        } else {
            if (verbose) printf("  SKIP: %s (not needed in GGUF)\n", mf->tensor_map[i].name);
        }
    }

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

    /* ── Check for weight tying ──
     * If tie_word_embeddings is set and there's no separate lm_head,
     * llama.cpp handles this internally — do NOT duplicate the tensor.
     * Only add output.weight if the model has a separate lm_head.weight. */
    int has_lm_head = (st_multi_find_tensor(mf, "lm_head.weight") >= 0);
    int total_tensors = n_include;

    if (arch->tie_word_embeddings && !has_lm_head) {
        printf("  Weight-tied embeddings detected — llama.cpp handles internally\n\n");
    }

    /* ── Prepare tensor info ── */
    char (*gguf_names)[ST_MAX_NAME_LEN] = calloc(total_tensors, ST_MAX_NAME_LEN);
    GGMLType *tensor_types = calloc(total_tensors, sizeof(GGMLType));
    int64_t *tensor_sizes = calloc(total_tensors, sizeof(int64_t));
    uint64_t data_offset = 0;
    uint64_t *tensor_offsets = calloc(total_tensors, sizeof(uint64_t));
    int *tensor_src_idx = calloc(total_tensors, sizeof(int)); /* map to unified ST index */
    char (*tensor_hf_names)[ST_MAX_NAME_LEN] = calloc(total_tensors, ST_MAX_NAME_LEN);

    GGMLType quant_type = GGML_TYPE_Q2_K;

    for (int i = 0; i < n_include; i++) {
        int src = include_list[i];
        const STTensorInfo *ti = st_multi_tensor_info(mf, src);
        map_tensor_name(mf->tensor_map[src].name, gguf_names[i], ST_MAX_NAME_LEN);
        strncpy(tensor_hf_names[i], mf->tensor_map[src].name, ST_MAX_NAME_LEN - 1);
        tensor_src_idx[i] = src;

        if (should_quantize(ti, gguf_names[i])) {
            tensor_types[i] = quant_type;
            tensor_sizes[i] = ggml_type_size(quant_type, ti->n_elements);
        } else if (ti->n_dims >= 2) {
            /* 2D non-quantized tensors (embeddings, output) → F16 */
            tensor_types[i] = GGML_TYPE_F16;
            tensor_sizes[i] = ti->n_elements * sizeof(uint16_t);
        } else {
            /* 1D tensors (norms, biases) → F32 */
            tensor_types[i] = GGML_TYPE_F32;
            tensor_sizes[i] = ti->n_elements * sizeof(float);
        }

        tensor_offsets[i] = data_offset;

        /* Align each tensor to 32 bytes */
        data_offset += tensor_sizes[i];
        data_offset = (data_offset + GGUF_DEFAULT_ALIGNMENT - 1) &
                      ~(uint64_t)(GGUF_DEFAULT_ALIGNMENT - 1);
    }

    /* ── Write header ── */
    gguf_write_header(fp, total_tensors, n_kv);

    /* ── Write metadata KV pairs ── */
    gguf_write_kv_string(fp, "general.architecture", arch->architecture);
    gguf_write_kv_string(fp, "general.name", arch->name);
    gguf_write_kv_uint32(fp, "general.quantization_version", 2);
    gguf_write_kv_uint32(fp, "general.file_type", 10);  /* Q2_K = 10 */

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
        const STTensorInfo *ti = st_multi_tensor_info(mf, src);
        uint64_t dims[ST_MAX_DIMS];
        /* GGUF uses reversed dimension order from SafeTensors/PyTorch */
        int nd = ti->n_dims;
        for (int d = 0; d < nd; d++) {
            dims[d] = (uint64_t)ti->shape[nd - 1 - d];
        }
        gguf_write_tensor_info(fp, gguf_names[i],
                                ti->n_dims, dims,
                                tensor_types[i], tensor_offsets[i]);
    }

    /* ── Alignment padding before data section ── */
    gguf_write_padding(fp, GGUF_DEFAULT_ALIGNMENT);

    /* ── Write tensor data ── */
    printf("  Quantizing and writing %d tensors...\n\n", total_tensors);

    float total_error_sum = 0.0f;
    int quant_count = 0;
    int64_t total_elements_quantized = 0;
    int64_t total_bytes_quantized = 0;
    int64_t total_bytes_unquantized = 0;
    clock_t quant_start = clock();

    for (int i = 0; i < total_tensors; i++) {
        int src = tensor_src_idx[i];
        const STTensorInfo *ti = st_multi_tensor_info(mf, src);

        print_progress_bar(i, total_tensors, gguf_names[i], quant_start);

        if (tensor_types[i] == GGML_TYPE_Q2_K) {
            /* ── HPC-Optimized Q2_K Quantization ── */
            float *f32_data = st_multi_tensor_to_f32(mf, src);
            if (!f32_data) {
                fprintf(stderr, "\n  ERROR: Failed to convert tensor '%s' to F32\n",
                        ti->name);
                continue;
            }

            int64_t n_elements = ti->n_elements;
            float tensor_error = 0.0f;

            /* Pad to QK_K boundary */
            int64_t padded = (n_elements + QK_K - 1) / QK_K * QK_K;
            if (padded > n_elements) {
                f32_data = realloc(f32_data, padded * sizeof(float));
                for (int64_t j = n_elements; j < padded; j++)
                    f32_data[j] = 0.0f;
                n_elements = padded;
            }

            int64_t n_blocks = n_elements / QK_K;
            BlockQ2K *quant_data = calloc(n_blocks, sizeof(BlockQ2K));

            /* Look up imatrix importance for this tensor */
            const float *imp = NULL;
            if (imatrix) {
                const IMatrixEntry *ime = imatrix_find_any(imatrix,
                    gguf_names[i], tensor_hf_names[i]);
                if (ime && ime->n_values > 0) {
                    imp = ime->normalized;
                    if (verbose)
                        printf("\n    imatrix: using %d importance weights for %s\n",
                               ime->n_values, gguf_names[i]);
                }
            }

            quantize_tensor_q2k_hpc(f32_data, n_elements,
                                      quant_data, &tensor_error,
                                      opt_mode, imp, verbose);

            fwrite(quant_data, sizeof(BlockQ2K), n_blocks, fp);

            float rmse = sqrtf(tensor_error / (float)ti->n_elements);
            if (verbose) {
                printf("\n  [Q2_K] %-50s  %10ld elements → %ld bytes  RMSE=%.6e\n",
                       gguf_names[i], (long)ti->n_elements,
                       (long)(n_blocks * sizeof(BlockQ2K)), rmse);
            }

            total_error_sum += tensor_error;
            total_elements_quantized += ti->n_elements;
            total_bytes_quantized += n_blocks * sizeof(BlockQ2K);
            quant_count++;

            free(quant_data);
            free(f32_data);
        } else if (tensor_types[i] == GGML_TYPE_F16) {
            /* ── Store as F16 (embeddings, output, 2D non-quantized) ── */
            float *f32_data = st_multi_tensor_to_f32(mf, src);
            if (!f32_data) {
                fprintf(stderr, "\n  ERROR: Failed to convert tensor '%s'\n",
                        ti->name);
                continue;
            }

            /* Convert F32 → F16 */
            uint16_t *f16_data = (uint16_t *)malloc(ti->n_elements * sizeof(uint16_t));
            for (int64_t j = 0; j < ti->n_elements; j++)
                f16_data[j] = gguf_fp32_to_fp16(f32_data[j]);

            fwrite(f16_data, sizeof(uint16_t), ti->n_elements, fp);

            total_bytes_unquantized += ti->n_elements * sizeof(uint16_t);

            if (verbose) {
                printf("\n  [F16 ] %-50s  %10ld elements → %ld bytes\n",
                       gguf_names[i], (long)ti->n_elements,
                       (long)(ti->n_elements * sizeof(uint16_t)));
            }

            free(f16_data);
            free(f32_data);
        } else {
            /* ── Keep as F32 (1D: norms, biases) ── */
            float *f32_data = st_multi_tensor_to_f32(mf, src);
            if (!f32_data) {
                fprintf(stderr, "\n  ERROR: Failed to convert tensor '%s'\n",
                        ti->name);
                continue;
            }

            fwrite(f32_data, sizeof(float), ti->n_elements, fp);

            total_bytes_unquantized += ti->n_elements * sizeof(float);

            if (verbose) {
                printf("\n  [F32 ] %-50s  %10ld elements → %ld bytes\n",
                       gguf_names[i], (long)ti->n_elements,
                       (long)(ti->n_elements * sizeof(float)));
            }

            free(f32_data);
        }

        /* Pad to alignment */
        gguf_write_padding(fp, GGUF_DEFAULT_ALIGNMENT);
    }

    print_progress_bar(total_tensors, total_tensors, "done", quant_start);

    long final_size = ftell(fp);
    fclose(fp);

    /* ── Final summary ── */
    printf("\n  ╔════════════════════════════════════════════════════════════════╗\n");
    printf("  ║  QUANTIZATION SUMMARY                                        ║\n");
    printf("  ╠════════════════════════════════════════════════════════════════╣\n");
    printf("  ║  Tensors quantized (Q2_K): %-33d ║\n", quant_count);
    printf("  ║  Elements quantized:     %15ld                   ║\n",
           (long)total_elements_quantized);
    printf("  ║  Q2_K data:              %12ld bytes (%6.1f MB)    ║\n",
           (long)total_bytes_quantized,
           (double)total_bytes_quantized / (1024.0 * 1024.0));
    printf("  ║  F32 data (unquantized): %12ld bytes (%6.1f MB)    ║\n",
           (long)total_bytes_unquantized,
           (double)total_bytes_unquantized / (1024.0 * 1024.0));

    /* Compute original model size (all as F32) */
    int64_t original_f32_size = 0;
    for (int i = 0; i < total_tensors; i++) {
        const STTensorInfo *ti = st_multi_tensor_info(mf, tensor_src_idx[i]);
        original_f32_size += ti->n_elements * sizeof(float);
    }
    float compression_ratio = (original_f32_size > 0) ?
                               (float)original_f32_size / (float)final_size : 0.0f;
    float effective_bpw = (total_elements_quantized > 0) ?
                           8.0f * (float)total_bytes_quantized / (float)total_elements_quantized :
                           0.0f;

    printf("  ║  Effective bits/weight:  %15.2f                       ║\n",
           effective_bpw);
    printf("  ║  Compression ratio:      %15.1fx                      ║\n",
           compression_ratio);
    printf("  ║  Total RMSE:             %15.6e                  ║\n",
           sqrtf(total_error_sum));
    printf("  ║  Output file:            %ld bytes (%.1f MB)%*s║\n",
           final_size, (double)final_size / (1024.0 * 1024.0),
           (int)(20 - snprintf(NULL, 0, "%ld bytes (%.1f MB)",
                               final_size, (double)final_size / (1024.0 * 1024.0))), "");
    printf("  ╚════════════════════════════════════════════════════════════════╝\n\n");

    free(include_list);
    free(gguf_names);
    free(tensor_types);
    free(tensor_sizes);
    free(tensor_offsets);
    free(tensor_src_idx);
    free(tensor_hf_names);

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
    printf("  ║   HExState GGUF QUANTIZER v2.0                              ║\n");
    printf("  ║                                                              ║\n");
    printf("  ║   Architecture: HPCGraph Sensitivity Propagation             ║\n");
    printf("  ║   Optimization: HPC BP + MSE Grid Search + iMatrix          ║\n");
    printf("  ║   Output: GGUF v3 (Q2_K, 2.625 bpw)                        ║\n");
    printf("  ║                                                              ║\n");
    printf("  ║   \"The weight and the quantized are opposite faces.\"         ║\n");
    printf("  ║                                                              ║\n");
    printf("  ╚════════════════════════════════════════════════════════════════╝\n\n");

    if (argc < 3) {
        printf("  Usage: %s <input> <output.gguf> [options]\n\n", argv[0]);
        printf("  Input:\n");
        printf("    Single .safetensors file, or\n");
        printf("    Model directory with sharded .safetensors files\n\n");
        printf("  Options:\n");
        printf("    --optimizer hpc|mse|hybrid   Scale optimization (default: hybrid)\n");
        printf("    --imatrix <file>             Importance matrix for Q2_K quality\n");
        printf("    --verbose                    Per-block diagnostics\n\n");
        return 1;
    }

    const char *input_path = argv[1];
    const char *output_path = argv[2];
    OptimizerMode opt_mode = OPT_HYBRID;
    const char *imatrix_path = NULL;
    int verbose = 0;

    /* Parse options */
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--optimizer") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "hpc") == 0) opt_mode = OPT_HPC;
            else if (strcmp(argv[i], "mse") == 0) opt_mode = OPT_MSE;
            else if (strcmp(argv[i], "hybrid") == 0) opt_mode = OPT_HYBRID;
            else {
                fprintf(stderr, "  ERROR: Unknown optimizer '%s'. Use hpc, mse, or hybrid.\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--imatrix") == 0 && i + 1 < argc) {
            imatrix_path = argv[++i];
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        } else {
            fprintf(stderr, "  ERROR: Unknown option '%s'\n", argv[i]);
            return 1;
        }
    }

    const char *opt_names[] = {"HPC (BP only)", "MSE (grid search)", "Hybrid (HPC+MSE)"};
    printf("  Input:      %s\n", input_path);
    printf("  Output:     %s\n", output_path);
    printf("  Quant type: Q2_K (2.625 bpw)\n");
    printf("  Optimizer:  %s\n", opt_names[opt_mode]);
    if (imatrix_path) printf("  iMatrix:    %s\n", imatrix_path);
    printf("\n");

    /* ── Phase 1: Load model ── */
    printf("  Phase 1: Loading model...\n");
    clock_t t_start = clock();

    /* Determine if input is a file or directory */
    struct stat st;
    if (stat(input_path, &st) != 0) {
        fprintf(stderr, "  ERROR: Cannot access '%s'\n", input_path);
        return 1;
    }

    STMultiFile *mf = NULL;
    char input_dir[512] = "";

    if (S_ISDIR(st.st_mode)) {
        /* Input is a directory — open all shards */
        mf = st_open_dir(input_path);
        strncpy(input_dir, input_path, sizeof(input_dir) - 2);
        int dlen = strlen(input_dir);
        if (dlen > 0 && input_dir[dlen - 1] != '/') {
            input_dir[dlen] = '/';
            input_dir[dlen + 1] = '\0';
        }
    } else {
        /* Input is a single file — wrap in STMultiFile */
        STFile *sf = st_open(input_path);
        if (!sf) {
            fprintf(stderr, "  ERROR: Failed to open '%s'\n", input_path);
            return 1;
        }
        mf = (STMultiFile *)calloc(1, sizeof(STMultiFile));
        mf->shards[0] = sf;
        mf->n_shards = 1;
        for (int i = 0; i < sf->n_tensors && mf->n_tensors < ST_MAX_TENSORS; i++) {
            strncpy(mf->tensor_map[mf->n_tensors].name,
                    sf->tensors[i].name, ST_MAX_NAME_LEN - 1);
            mf->tensor_map[mf->n_tensors].shard_idx = 0;
            mf->tensor_map[mf->n_tensors].tensor_idx = i;
            mf->n_tensors++;
        }

        /* Extract directory from file path */
        strncpy(input_dir, input_path, sizeof(input_dir) - 1);
        char *last_slash = strrchr(input_dir, '/');
        if (last_slash) {
            *(last_slash + 1) = '\0';
        } else {
            strcpy(input_dir, "./");
        }
    }

    if (!mf) {
        fprintf(stderr, "  ERROR: Failed to load model from '%s'\n", input_path);
        return 1;
    }

    st_multi_print_summary(mf);

    clock_t t_load = clock();
    printf("  Loaded in %.3f seconds\n\n",
           (double)(t_load - t_start) / CLOCKS_PER_SEC);

    /* ── Phase 2: Detect architecture ── */
    printf("  Phase 2: Detecting model architecture...\n");

    /* Try to read config.json from model directory */
    char config_path[1024];
    snprintf(config_path, sizeof(config_path), "%sconfig.json", input_dir);
    const char *config_ptr = NULL;
    {
        FILE *check = fopen(config_path, "rb");
        if (check) {
            fclose(check);
            config_ptr = config_path;
            printf("  Found config.json: %s\n", config_path);
        }
    }

    ModelArchitecture arch;
    detect_architecture(mf, &arch, config_ptr);

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
    printf("  ║  Has bias:         %-40s ║\n", arch.has_bias ? "yes" : "no");
    printf("  ║  Tied embeddings:  %-40s ║\n", arch.tie_word_embeddings ? "yes" : "no");
    printf("  ╚═══════════════════════════════════════════════════════════════╝\n\n");

    /* ── Phase 2b: Load tokenizer ── */
    printf("  Phase 2b: Loading tokenizer...\n");
    TokenizerData *tokenizer = NULL;
    {
        char tok_json[512], tok_config[512];
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

    /* ── Phase 2c: Load importance matrix (optional) ── */
    IMatrixData *imatrix = NULL;
    if (imatrix_path) {
        printf("  Phase 2c: Loading importance matrix...\n");
        imatrix = imatrix_load(imatrix_path);
        if (imatrix) {
            imatrix_print_summary(imatrix);
        } else {
            printf("  WARNING: Failed to load imatrix from '%s'\n", imatrix_path);
            printf("  Proceeding without importance weighting.\n\n");
        }
    }

    /* ── Phase 3-5: Quantize and write GGUF ── */
    printf("  Phase 3: HPC-Optimized Q2_K Quantization + GGUF Output...\n");
    clock_t t_quant_start = clock();

    int result = write_gguf(output_path, mf, &arch, tokenizer,
                              opt_mode, imatrix, verbose);

    clock_t t_end = clock();
    printf("  Total time: %.3f seconds\n\n",
           (double)(t_end - t_start) / CLOCKS_PER_SEC);

    if (imatrix) imatrix_free(imatrix);
    if (tokenizer) tok_free(tokenizer);
    st_multi_close(mf);
    return result;
}
