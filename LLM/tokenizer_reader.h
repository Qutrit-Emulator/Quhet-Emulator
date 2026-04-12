/*
 * tokenizer_reader.h — HuggingFace tokenizer.json Parser
 *
 * Extracts vocabulary, merge rules, and special token IDs from
 * HuggingFace tokenizer.json files for embedding into GGUF.
 *
 * Supports: LLaMA/Mistral BPE tokenizers (sentencepiece-derived)
 */

#ifndef TOKENIZER_READER_H
#define TOKENIZER_READER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define TOK_MAX_TOKENS    256000   /* Max supported vocab size      */
#define TOK_MAX_MERGES    512000   /* Max supported merge rules     */
#define TOK_MAX_TOKEN_LEN 512      /* Max length of a single token  */

/* Token types matching GGUF tokenizer.ggml.token_type */
typedef enum {
    TOK_TYPE_NORMAL    = 1,
    TOK_TYPE_UNKNOWN   = 2,
    TOK_TYPE_CONTROL   = 3,
    TOK_TYPE_USER_DEF  = 4,
    TOK_TYPE_UNUSED    = 5,
    TOK_TYPE_BYTE      = 6
} TokenType;

typedef struct {
    char   **tokens;         /* Token strings indexed by ID            */
    float   *scores;         /* Token scores/priorities                */
    int32_t *token_types;    /* Token type enum per token              */
    int32_t  vocab_size;     /* Total vocabulary size                  */

    char   **merges;         /* BPE merge rule strings                 */
    int32_t  n_merges;       /* Number of merge rules                  */

    int32_t  bos_id;         /* Beginning of sequence token ID         */
    int32_t  eos_id;         /* End of sequence token ID               */
    int32_t  unk_id;         /* Unknown token ID                       */
    int32_t  pad_id;         /* Padding token ID (-1 if none)          */

    char     model_type[32]; /* "llama", "gpt2", etc.                  */
} TokenizerData;

/* ═══════════════════════════════════════════════════════════════════
 * JSON HELPER — Minimal extraction utilities
 *
 * These are NOT a general JSON parser — they target the specific
 * structure of HuggingFace tokenizer.json files.
 * ═══════════════════════════════════════════════════════════════════ */

/* Skip whitespace */
static inline const char *tok_skip_ws(const char *p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

/* Extract a JSON string value starting at the opening quote.
 * Handles basic escape sequences. Returns pointer after closing quote.
 * Copies unescaped string into buf. */
static const char *tok_extract_string(const char *p, char *buf, int buflen)
{
    if (*p != '"') return NULL;
    p++;  /* skip opening quote */

    int i = 0;
    while (*p && *p != '"' && i < buflen - 1) {
        if (*p == '\\' && p[1]) {
            p++;
            switch (*p) {
                case '"':  buf[i++] = '"'; break;
                case '\\': buf[i++] = '\\'; break;
                case '/':  buf[i++] = '/'; break;
                case 'n':  buf[i++] = '\n'; break;
                case 'r':  buf[i++] = '\r'; break;
                case 't':  buf[i++] = '\t'; break;
                case 'u': {
                    /* Parse \uXXXX unicode escape */
                    if (p[1] && p[2] && p[3] && p[4]) {
                        unsigned int cp = 0;
                        char hex[5] = {p[1], p[2], p[3], p[4], 0};
                        cp = (unsigned int)strtoul(hex, NULL, 16);
                        p += 4;
                        /* Encode as UTF-8 */
                        if (cp < 0x80) {
                            buf[i++] = (char)cp;
                        } else if (cp < 0x800) {
                            if (i + 1 < buflen - 1) {
                                buf[i++] = (char)(0xC0 | (cp >> 6));
                                buf[i++] = (char)(0x80 | (cp & 0x3F));
                            }
                        } else {
                            if (i + 2 < buflen - 1) {
                                buf[i++] = (char)(0xE0 | (cp >> 12));
                                buf[i++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                                buf[i++] = (char)(0x80 | (cp & 0x3F));
                            }
                        }
                    }
                    break;
                }
                default: buf[i++] = *p; break;
            }
        } else {
            buf[i++] = *p;
        }
        p++;
    }
    buf[i] = '\0';

    if (*p == '"') p++;  /* skip closing quote */
    return p;
}

/* Find a key in JSON and return pointer to its value */
static const char *tok_find_key(const char *json, const char *key)
{
    char search[TOK_MAX_TOKEN_LEN + 4];
    snprintf(search, sizeof(search), "\"%s\"", key);

    const char *p = strstr(json, search);
    if (!p) return NULL;

    p += strlen(search);
    p = tok_skip_ws(p);
    if (*p == ':') p++;
    p = tok_skip_ws(p);
    return p;
}

/* ═══════════════════════════════════════════════════════════════════
 * VOCAB PARSER — Extract "model": { "vocab": { ... } }
 * ═══════════════════════════════════════════════════════════════════ */

static int tok_parse_vocab(const char *json, TokenizerData *td)
{
    /* Find "vocab" key inside "model" object */
    const char *model_p = tok_find_key(json, "model");
    if (!model_p) return -1;

    /* Extract model type */
    const char *type_p = tok_find_key(model_p, "type");
    if (type_p) {
        char type_buf[64];
        tok_extract_string(type_p, type_buf, sizeof(type_buf));
        if (strcasecmp(type_buf, "BPE") == 0) {
            strcpy(td->model_type, "llama");
        } else {
            strncpy(td->model_type, type_buf, sizeof(td->model_type) - 1);
        }
    }

    /* Find "vocab": { */
    const char *vocab_p = tok_find_key(model_p, "vocab");
    if (!vocab_p || *vocab_p != '{') return -1;
    vocab_p++;  /* skip '{' */

    /* Parse each "token_string": id pair */
    char token_buf[TOK_MAX_TOKEN_LEN];
    int max_id = -1;

    /* First pass: count entries and find max ID */
    const char *scan = vocab_p;
    int count = 0;
    while (*scan && *scan != '}') {
        scan = tok_skip_ws(scan);
        if (*scan == ',') { scan++; continue; }
        if (*scan != '"') break;

        /* Skip key */
        char dummy[TOK_MAX_TOKEN_LEN];
        scan = tok_extract_string(scan, dummy, sizeof(dummy));
        if (!scan) break;
        scan = tok_skip_ws(scan);
        if (*scan == ':') scan++;
        scan = tok_skip_ws(scan);

        /* Read value (integer) */
        int id = (int)strtol(scan, (char **)&scan, 10);
        if (id > max_id) max_id = id;
        count++;
    }

    if (count == 0 || max_id < 0) return -1;

    td->vocab_size = max_id + 1;

    /* Allocate arrays */
    td->tokens = (char **)calloc(td->vocab_size, sizeof(char *));
    td->scores = (float *)calloc(td->vocab_size, sizeof(float));
    td->token_types = (int32_t *)calloc(td->vocab_size, sizeof(int32_t));

    /* Initialize with defaults */
    for (int i = 0; i < td->vocab_size; i++) {
        td->tokens[i] = strdup("");
        td->scores[i] = 0.0f;
        td->token_types[i] = TOK_TYPE_NORMAL;
    }

    /* Second pass: fill in tokens */
    scan = vocab_p;
    while (*scan && *scan != '}') {
        scan = tok_skip_ws(scan);
        if (*scan == ',') { scan++; continue; }
        if (*scan != '"') break;

        scan = tok_extract_string(scan, token_buf, sizeof(token_buf));
        if (!scan) break;
        scan = tok_skip_ws(scan);
        if (*scan == ':') scan++;
        scan = tok_skip_ws(scan);

        int id = (int)strtol(scan, (char **)&scan, 10);

        if (id >= 0 && id < td->vocab_size) {
            free(td->tokens[id]);
            td->tokens[id] = strdup(token_buf);
            /* Score = negative index for BPE ordering (higher ID = lower priority) */
            td->scores[id] = -(float)id;
        }
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * MERGES PARSER — Extract "model": { "merges": [ ... ] }
 * ═══════════════════════════════════════════════════════════════════ */

static int tok_parse_merges(const char *json, TokenizerData *td)
{
    const char *model_p = tok_find_key(json, "model");
    if (!model_p) return -1;

    const char *merges_p = tok_find_key(model_p, "merges");
    if (!merges_p || *merges_p != '[') return -1;
    merges_p++;  /* skip '[' */

    /* Allocate with growth pattern — start with 64k slots */
    int capacity = 65536;
    td->merges = (char **)calloc(capacity, sizeof(char *));
    td->n_merges = 0;

    /* Extract merge strings */
    const char *scan = merges_p;
    char merge_buf[TOK_MAX_TOKEN_LEN * 2];
    while (*scan && *scan != ']' && td->n_merges < TOK_MAX_MERGES) {
        scan = tok_skip_ws(scan);
        if (*scan == ',') { scan++; continue; }
        if (*scan != '"') { scan++; continue; }

        scan = tok_extract_string(scan, merge_buf, sizeof(merge_buf));
        if (!scan) break;

        /* Grow if needed */
        if (td->n_merges >= capacity) {
            capacity *= 2;
            td->merges = (char **)realloc(td->merges, capacity * sizeof(char *));
        }

        td->merges[td->n_merges] = strdup(merge_buf);
        td->n_merges++;
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 * SPECIAL TOKENS — Extract from "added_tokens" array
 * ═══════════════════════════════════════════════════════════════════ */

static void tok_parse_added_tokens(const char *json, TokenizerData *td)
{
    const char *added_p = tok_find_key(json, "added_tokens");
    if (!added_p || *added_p != '[') return;
    added_p++;

    /* Scan through the array of objects */
    while (*added_p && *added_p != ']') {
        added_p = tok_skip_ws(added_p);
        if (*added_p == ',') { added_p++; continue; }
        if (*added_p != '{') { added_p++; continue; }

        /* Find end of this object */
        const char *obj_start = added_p;
        int depth = 1;
        added_p++;
        while (*added_p && depth > 0) {
            if (*added_p == '{') depth++;
            if (*added_p == '}') depth--;
            added_p++;
        }

        /* Extract content and id from this object */
        char content[TOK_MAX_TOKEN_LEN] = "";
        int id = -1;
        int is_special = 0;

        const char *id_p = tok_find_key(obj_start, "id");
        if (id_p) id = (int)strtol(id_p, NULL, 10);

        const char *content_p = tok_find_key(obj_start, "content");
        if (content_p && *content_p == '"')
            tok_extract_string(content_p, content, sizeof(content));

        const char *special_p = tok_find_key(obj_start, "special");
        if (special_p) {
            is_special = (strncmp(special_p, "true", 4) == 0);
        }

        /* Mark special tokens */
        if (id >= 0 && id < td->vocab_size) {
            if (is_special) {
                td->token_types[id] = TOK_TYPE_CONTROL;
            }
            /* Update token string if needed */
            if (content[0] && (!td->tokens[id] || !td->tokens[id][0])) {
                free(td->tokens[id]);
                td->tokens[id] = strdup(content);
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * SPECIAL TOKEN IDs — Extract from tokenizer_config.json
 * ═══════════════════════════════════════════════════════════════════ */

static void tok_parse_config(const char *config_json, TokenizerData *td)
{
    /* Look for bos_token, eos_token, unk_token content strings */
    /* Then find their IDs in the vocab */

    /* Search for token content in the config */
    struct { const char *key; int32_t *id_ptr; const char *default_content; } specials[] = {
        {"bos_token", &td->bos_id, "<s>"},
        {"eos_token", &td->eos_id, "</s>"},
        {"unk_token", &td->unk_id, "<unk>"},
        {NULL, NULL, NULL}
    };

    for (int s = 0; specials[s].key; s++) {
        const char *p = tok_find_key(config_json, specials[s].key);
        if (!p) {
            /* Try to find in vocab by default content */
            for (int i = 0; i < td->vocab_size; i++) {
                if (td->tokens[i] && strcmp(td->tokens[i], specials[s].default_content) == 0) {
                    *specials[s].id_ptr = i;
                    break;
                }
            }
            continue;
        }

        /* The value might be a string directly or an object with "content" */
        if (*p == '"') {
            char content[TOK_MAX_TOKEN_LEN];
            tok_extract_string(p, content, sizeof(content));
            /* Find this content in vocab */
            for (int i = 0; i < td->vocab_size; i++) {
                if (td->tokens[i] && strcmp(td->tokens[i], content) == 0) {
                    *specials[s].id_ptr = i;
                    break;
                }
            }
        } else if (*p == '{') {
            /* Object with "content" field */
            const char *cp = tok_find_key(p, "content");
            if (cp && *cp == '"') {
                char content[TOK_MAX_TOKEN_LEN];
                tok_extract_string(cp, content, sizeof(content));
                for (int i = 0; i < td->vocab_size; i++) {
                    if (td->tokens[i] && strcmp(td->tokens[i], content) == 0) {
                        *specials[s].id_ptr = i;
                        break;
                    }
                }
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * MAIN API — Load tokenizer from directory
 * ═══════════════════════════════════════════════════════════════════ */

static char *tok_read_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *buf = (char *)malloc(size + 1);
    if (!buf) { fclose(f); return NULL; }

    fread(buf, 1, size, f);
    buf[size] = '\0';
    fclose(f);
    return buf;
}

static TokenizerData *tok_load(const char *tokenizer_json_path,
                                const char *config_json_path)
{
    TokenizerData *td = (TokenizerData *)calloc(1, sizeof(TokenizerData));
    if (!td) return NULL;

    td->bos_id = 1;
    td->eos_id = 2;
    td->unk_id = 0;
    td->pad_id = -1;
    strcpy(td->model_type, "llama");

    /* Read tokenizer.json */
    char *json = tok_read_file(tokenizer_json_path);
    if (!json) {
        fprintf(stderr, "  WARNING: Could not read '%s'\n", tokenizer_json_path);
        free(td);
        return NULL;
    }

    /* Parse vocab */
    if (tok_parse_vocab(json, td) != 0) {
        fprintf(stderr, "  WARNING: Failed to parse vocab from tokenizer.json\n");
        free(json);
        free(td);
        return NULL;
    }

    /* Parse merges */
    tok_parse_merges(json, td);

    /* Parse added tokens (special tokens) */
    tok_parse_added_tokens(json, td);

    /* Detect byte tokens: <0x00> through <0xFF> */
    for (int i = 0; i < td->vocab_size; i++) {
        if (td->tokens[i] && td->tokens[i][0] == '<' &&
            td->tokens[i][1] == '0' && td->tokens[i][2] == 'x' &&
            strlen(td->tokens[i]) == 6 && td->tokens[i][5] == '>') {
            td->token_types[i] = TOK_TYPE_BYTE;
        }
    }

    free(json);

    /* Read config if available */
    if (config_json_path) {
        char *config = tok_read_file(config_json_path);
        if (config) {
            tok_parse_config(config, td);
            free(config);
        }
    }

    return td;
}

static void tok_free(TokenizerData *td)
{
    if (!td) return;
    if (td->tokens) {
        for (int i = 0; i < td->vocab_size; i++)
            free(td->tokens[i]);
        free(td->tokens);
    }
    if (td->merges) {
        for (int i = 0; i < td->n_merges; i++)
            free(td->merges[i]);
        free(td->merges);
    }
    free(td->scores);
    free(td->token_types);
    free(td);
}

/* Print summary */
static void tok_print_summary(const TokenizerData *td)
{
    printf("  ╔═══════════════════════════════════════════════════════════════╗\n");
    printf("  ║  Tokenizer                                                  ║\n");
    printf("  ╠═══════════════════════════════════════════════════════════════╣\n");
    printf("  ║  Model:            %-40s ║\n", td->model_type);
    printf("  ║  Vocab size:       %-40d ║\n", td->vocab_size);
    printf("  ║  Merges:           %-40d ║\n", td->n_merges);
    printf("  ║  BOS token:        %-3d  %-36s ║\n", td->bos_id,
           (td->bos_id >= 0 && td->bos_id < td->vocab_size) ? td->tokens[td->bos_id] : "");
    printf("  ║  EOS token:        %-3d  %-36s ║\n", td->eos_id,
           (td->eos_id >= 0 && td->eos_id < td->vocab_size) ? td->tokens[td->eos_id] : "");
    printf("  ║  UNK token:        %-3d  %-36s ║\n", td->unk_id,
           (td->unk_id >= 0 && td->unk_id < td->vocab_size) ? td->tokens[td->unk_id] : "");
    printf("  ╚═══════════════════════════════════════════════════════════════╝\n\n");
}

#endif /* TOKENIZER_READER_H */
