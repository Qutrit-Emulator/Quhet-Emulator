/*
 * safetensors_reader.h — SafeTensors Binary Format Reader
 *
 * ╔═══════════════════════════════════════════════════════════════╗
 * ║  HExState SafeTensors Input Module                           ║
 * ║  Parses HuggingFace SafeTensors files in pure C              ║
 * ║  Supports mmap for zero-copy tensor access                   ║
 * ╚═══════════════════════════════════════════════════════════════╝
 *
 * SafeTensors file layout:
 *   [8 bytes: header_size (uint64_t LE)]
 *   [header_size bytes: JSON metadata]
 *   [rest of file: raw tensor data]
 *
 * JSON header maps tensor names → {dtype, shape, data_offsets}
 * Offsets are relative to the start of the data section.
 */

#ifndef SAFETENSORS_READER_H
#define SAFETENSORS_READER_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* ═══════════════════════════════════════════════════════════════════════
 * CONSTANTS
 * ═══════════════════════════════════════════════════════════════════════ */

#define ST_MAX_TENSORS      4096
#define ST_MAX_NAME_LEN     256
#define ST_MAX_DIMS         8
#define ST_MAX_HEADER_SIZE  (100 * 1024 * 1024)  /* 100 MB safety limit */

/* ═══════════════════════════════════════════════════════════════════════
 * TENSOR DTYPE
 * ═══════════════════════════════════════════════════════════════════════ */

typedef enum {
    ST_DTYPE_F32,
    ST_DTYPE_F16,
    ST_DTYPE_BF16,
    ST_DTYPE_F64,
    ST_DTYPE_I8,
    ST_DTYPE_I16,
    ST_DTYPE_I32,
    ST_DTYPE_I64,
    ST_DTYPE_U8,
    ST_DTYPE_BOOL,
    ST_DTYPE_UNKNOWN
} STDtype;

static inline int st_dtype_size(STDtype dtype)
{
    switch (dtype) {
        case ST_DTYPE_F32:  return 4;
        case ST_DTYPE_F16:  return 2;
        case ST_DTYPE_BF16: return 2;
        case ST_DTYPE_F64:  return 8;
        case ST_DTYPE_I8:   return 1;
        case ST_DTYPE_I16:  return 2;
        case ST_DTYPE_I32:  return 4;
        case ST_DTYPE_I64:  return 8;
        case ST_DTYPE_U8:   return 1;
        case ST_DTYPE_BOOL: return 1;
        default: return 0;
    }
}

static inline STDtype st_parse_dtype(const char *s, int len)
{
    if (len == 3 && strncmp(s, "F32", 3) == 0)  return ST_DTYPE_F32;
    if (len == 3 && strncmp(s, "F16", 3) == 0)  return ST_DTYPE_F16;
    if (len == 4 && strncmp(s, "BF16", 4) == 0) return ST_DTYPE_BF16;
    if (len == 3 && strncmp(s, "F64", 3) == 0)  return ST_DTYPE_F64;
    if (len == 2 && strncmp(s, "I8", 2) == 0)   return ST_DTYPE_I8;
    if (len == 3 && strncmp(s, "I16", 3) == 0)  return ST_DTYPE_I16;
    if (len == 3 && strncmp(s, "I32", 3) == 0)  return ST_DTYPE_I32;
    if (len == 3 && strncmp(s, "I64", 3) == 0)  return ST_DTYPE_I64;
    if (len == 2 && strncmp(s, "U8", 2) == 0)   return ST_DTYPE_U8;
    if (len == 4 && strncmp(s, "BOOL", 4) == 0) return ST_DTYPE_BOOL;
    return ST_DTYPE_UNKNOWN;
}

/* ═══════════════════════════════════════════════════════════════════════
 * TENSOR DESCRIPTOR
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    char     name[ST_MAX_NAME_LEN];
    STDtype  dtype;
    int      n_dims;
    int64_t  shape[ST_MAX_DIMS];
    int64_t  n_elements;        /* Product of shape dims          */
    uint64_t data_offset_begin; /* Offset from data section start */
    uint64_t data_offset_end;
    uint64_t data_size;         /* end - begin                    */
} STTensorInfo;

/* ═══════════════════════════════════════════════════════════════════════
 * SAFETENSORS FILE HANDLE
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    /* File mapping */
    int          fd;
    uint8_t     *mmap_base;
    size_t       file_size;

    /* Header */
    uint64_t     header_size;
    char        *header_json;    /* Not null-terminated in file,
                                    we add a null for parsing */

    /* Data section */
    uint8_t     *data_base;      /* Points into mmap at header+8 */

    /* Tensor catalog */
    STTensorInfo tensors[ST_MAX_TENSORS];
    int          n_tensors;
} STFile;

/* ═══════════════════════════════════════════════════════════════════════
 * MINIMAL JSON PARSER
 *
 * This is a hand-rolled, zero-allocation JSON parser designed
 * specifically for the SafeTensors header format. It does NOT handle
 * arbitrary JSON — only the specific structure used by SafeTensors.
 *
 * Expected format:
 * {
 *   "__metadata__": { ... },
 *   "tensor_name": {
 *     "dtype": "F16",
 *     "shape": [1024, 4096],
 *     "data_offsets": [0, 8388608]
 *   },
 *   ...
 * }
 * ═══════════════════════════════════════════════════════════════════════ */

/* Skip whitespace */
static inline const char *st_skip_ws(const char *p)
{
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

/* Parse a JSON string (returns pointer after closing quote).
 * Copies string content to buf (up to buflen-1 chars). */
static inline const char *st_parse_json_string(const char *p, char *buf, int buflen)
{
    if (*p != '"') return NULL;
    p++;
    int i = 0;
    while (*p && *p != '"') {
        if (*p == '\\') {
            p++;  /* skip escape */
            if (!*p) return NULL;
        }
        if (i < buflen - 1) buf[i++] = *p;
        p++;
    }
    buf[i] = '\0';
    if (*p == '"') p++;
    return p;
}

/* Parse a JSON integer */
static inline const char *st_parse_json_int(const char *p, int64_t *out)
{
    char numbuf[32];
    int i = 0;
    if (*p == '-') { numbuf[i++] = *p; p++; }
    while (*p >= '0' && *p <= '9' && i < 30) {
        numbuf[i++] = *p;
        p++;
    }
    numbuf[i] = '\0';
    *out = strtoll(numbuf, NULL, 10);
    return p;
}

/* Skip a JSON value (string, number, object, array, bool, null) */
static inline const char *st_skip_json_value(const char *p)
{
    p = st_skip_ws(p);
    if (*p == '"') {
        /* String */
        p++;
        while (*p && *p != '"') {
            if (*p == '\\') p++;
            if (*p) p++;
        }
        if (*p == '"') p++;
        return p;
    }
    if (*p == '{') {
        /* Object */
        int depth = 1;
        p++;
        while (*p && depth > 0) {
            if (*p == '{') depth++;
            else if (*p == '}') depth--;
            else if (*p == '"') {
                p++;
                while (*p && *p != '"') {
                    if (*p == '\\') p++;
                    if (*p) p++;
                }
            }
            if (*p) p++;
        }
        return p;
    }
    if (*p == '[') {
        /* Array */
        int depth = 1;
        p++;
        while (*p && depth > 0) {
            if (*p == '[') depth++;
            else if (*p == ']') depth--;
            else if (*p == '"') {
                p++;
                while (*p && *p != '"') {
                    if (*p == '\\') p++;
                    if (*p) p++;
                }
            }
            if (*p) p++;
        }
        return p;
    }
    /* Number, bool, null — skip until delimiter */
    while (*p && *p != ',' && *p != '}' && *p != ']' &&
           *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') {
        p++;
    }
    return p;
}

/* Parse the SafeTensors JSON header and populate the tensor catalog */
static inline int st_parse_header(STFile *st)
{
    const char *p = st->header_json;
    p = st_skip_ws(p);
    if (*p != '{') return -1;
    p++;

    st->n_tensors = 0;

    while (*p) {
        p = st_skip_ws(p);
        if (*p == '}') break;
        if (*p == ',') { p++; continue; }

        /* Parse key */
        char key[ST_MAX_NAME_LEN];
        p = st_parse_json_string(p, key, sizeof(key));
        if (!p) return -1;

        p = st_skip_ws(p);
        if (*p != ':') return -1;
        p++;
        p = st_skip_ws(p);

        /* Skip __metadata__ */
        if (strcmp(key, "__metadata__") == 0) {
            p = st_skip_json_value(p);
            continue;
        }

        /* Parse tensor object */
        if (*p != '{') {
            p = st_skip_json_value(p);
            continue;
        }
        p++;

        STTensorInfo *ti = &st->tensors[st->n_tensors];
        memset(ti, 0, sizeof(*ti));
        strncpy(ti->name, key, ST_MAX_NAME_LEN - 1);

        while (*p) {
            p = st_skip_ws(p);
            if (*p == '}') { p++; break; }
            if (*p == ',') { p++; continue; }

            char field[64];
            p = st_parse_json_string(p, field, sizeof(field));
            if (!p) return -1;

            p = st_skip_ws(p);
            if (*p != ':') return -1;
            p++;
            p = st_skip_ws(p);

            if (strcmp(field, "dtype") == 0) {
                char dtype_str[16];
                p = st_parse_json_string(p, dtype_str, sizeof(dtype_str));
                if (!p) return -1;
                ti->dtype = st_parse_dtype(dtype_str, strlen(dtype_str));
            } else if (strcmp(field, "shape") == 0) {
                /* Parse array of ints */
                if (*p != '[') return -1;
                p++;
                ti->n_dims = 0;
                ti->n_elements = 1;
                while (*p) {
                    p = st_skip_ws(p);
                    if (*p == ']') { p++; break; }
                    if (*p == ',') { p++; continue; }
                    int64_t dim_val;
                    p = st_parse_json_int(p, &dim_val);
                    if (ti->n_dims < ST_MAX_DIMS) {
                        ti->shape[ti->n_dims++] = dim_val;
                        ti->n_elements *= dim_val;
                    }
                }
            } else if (strcmp(field, "data_offsets") == 0) {
                /* Parse [begin, end] */
                if (*p != '[') return -1;
                p++;
                p = st_skip_ws(p);
                int64_t begin_val, end_val;
                p = st_parse_json_int(p, &begin_val);
                p = st_skip_ws(p);
                if (*p == ',') p++;
                p = st_skip_ws(p);
                p = st_parse_json_int(p, &end_val);
                p = st_skip_ws(p);
                if (*p == ']') p++;
                ti->data_offset_begin = (uint64_t)begin_val;
                ti->data_offset_end = (uint64_t)end_val;
                ti->data_size = ti->data_offset_end - ti->data_offset_begin;
            } else {
                p = st_skip_json_value(p);
            }
        }

        if (st->n_tensors < ST_MAX_TENSORS)
            st->n_tensors++;
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * OPEN / CLOSE A SAFETENSORS FILE
 * ═══════════════════════════════════════════════════════════════════════ */

static inline STFile *st_open(const char *path)
{
    STFile *st = (STFile *)calloc(1, sizeof(STFile));
    if (!st) return NULL;

    /* Open file */
    st->fd = open(path, O_RDONLY);
    if (st->fd < 0) {
        fprintf(stderr, "st_open: cannot open '%s'\n", path);
        free(st);
        return NULL;
    }

    /* Get file size */
    struct stat sb;
    if (fstat(st->fd, &sb) < 0) {
        close(st->fd);
        free(st);
        return NULL;
    }
    st->file_size = sb.st_size;

    /* Memory-map the entire file */
    st->mmap_base = (uint8_t *)mmap(NULL, st->file_size, PROT_READ,
                                      MAP_PRIVATE, st->fd, 0);
    if (st->mmap_base == MAP_FAILED) {
        fprintf(stderr, "st_open: mmap failed for '%s'\n", path);
        close(st->fd);
        free(st);
        return NULL;
    }

    /* Read header size (first 8 bytes, little-endian uint64) */
    memcpy(&st->header_size, st->mmap_base, sizeof(uint64_t));

    if (st->header_size > ST_MAX_HEADER_SIZE ||
        st->header_size + 8 > st->file_size) {
        fprintf(stderr, "st_open: invalid header size %lu\n",
                (unsigned long)st->header_size);
        munmap(st->mmap_base, st->file_size);
        close(st->fd);
        free(st);
        return NULL;
    }

    /* Copy header JSON and null-terminate for our parser */
    st->header_json = (char *)malloc(st->header_size + 1);
    memcpy(st->header_json, st->mmap_base + 8, st->header_size);
    st->header_json[st->header_size] = '\0';

    /* Data section starts right after header */
    st->data_base = st->mmap_base + 8 + st->header_size;

    /* Parse the header */
    if (st_parse_header(st) != 0) {
        fprintf(stderr, "st_open: failed to parse header of '%s'\n", path);
        free(st->header_json);
        munmap(st->mmap_base, st->file_size);
        close(st->fd);
        free(st);
        return NULL;
    }

    return st;
}

static inline void st_close(STFile *st)
{
    if (!st) return;
    free(st->header_json);
    if (st->mmap_base && st->mmap_base != MAP_FAILED)
        munmap(st->mmap_base, st->file_size);
    if (st->fd >= 0)
        close(st->fd);
    free(st);
}

/* ═══════════════════════════════════════════════════════════════════════
 * TENSOR DATA ACCESS
 *
 * Returns a raw pointer into the mmap'd region.
 * Caller must interpret the bytes according to the tensor's dtype.
 * ═══════════════════════════════════════════════════════════════════════ */

static inline const void *st_tensor_data(const STFile *st, int tensor_idx)
{
    if (tensor_idx < 0 || tensor_idx >= st->n_tensors) return NULL;
    return st->data_base + st->tensors[tensor_idx].data_offset_begin;
}

/* ═══════════════════════════════════════════════════════════════════════
 * TENSOR → FLOAT32 CONVERSION
 *
 * Converts tensor data to float32, handling FP16 and BF16 input.
 * Caller must free the returned buffer.
 * ═══════════════════════════════════════════════════════════════════════ */

/* Forward declaration of fp16/bf16 converters from gguf_format.h */
/* (Already included when both headers are used together) */

static inline float *st_tensor_to_f32(const STFile *st, int tensor_idx)
{
    const STTensorInfo *ti = &st->tensors[tensor_idx];
    const uint8_t *raw = (const uint8_t *)st_tensor_data(st, tensor_idx);
    if (!raw) return NULL;

    float *out = (float *)malloc(ti->n_elements * sizeof(float));
    if (!out) return NULL;

    switch (ti->dtype) {
        case ST_DTYPE_F32:
            memcpy(out, raw, ti->n_elements * sizeof(float));
            break;

        case ST_DTYPE_F16: {
            const uint16_t *fp16 = (const uint16_t *)raw;
            for (int64_t i = 0; i < ti->n_elements; i++) {
                out[i] = gguf_fp16_to_fp32(fp16[i]);
            }
            break;
        }

        case ST_DTYPE_BF16: {
            const uint16_t *bf16 = (const uint16_t *)raw;
            for (int64_t i = 0; i < ti->n_elements; i++) {
                out[i] = gguf_bf16_to_fp32(bf16[i]);
            }
            break;
        }

        case ST_DTYPE_F64: {
            const double *f64 = (const double *)raw;
            for (int64_t i = 0; i < ti->n_elements; i++) {
                out[i] = (float)f64[i];
            }
            break;
        }

        default:
            /* For integer types, just cast */
            for (int64_t i = 0; i < ti->n_elements; i++) {
                switch (ti->dtype) {
                    case ST_DTYPE_I8:  out[i] = (float)((int8_t *)raw)[i]; break;
                    case ST_DTYPE_I16: out[i] = (float)((int16_t *)raw)[i]; break;
                    case ST_DTYPE_I32: out[i] = (float)((int32_t *)raw)[i]; break;
                    case ST_DTYPE_U8:  out[i] = (float)raw[i]; break;
                    default: out[i] = 0.0f; break;
                }
            }
            break;
    }

    return out;
}

/* ═══════════════════════════════════════════════════════════════════════
 * FIND TENSOR BY NAME
 * ═══════════════════════════════════════════════════════════════════════ */

static inline int st_find_tensor(const STFile *st, const char *name)
{
    for (int i = 0; i < st->n_tensors; i++) {
        if (strcmp(st->tensors[i].name, name) == 0)
            return i;
    }
    return -1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * DIAGNOSTICS
 * ═══════════════════════════════════════════════════════════════════════ */

static inline void st_print_summary(const STFile *st)
{
    printf("  ╔═══════════════════════════════════════════════════════════════╗\n");
    printf("  ║  SafeTensors File Summary                                   ║\n");
    printf("  ╠═══════════════════════════════════════════════════════════════╣\n");
    printf("  ║  File size:    %12lu bytes                             ║\n",
           (unsigned long)st->file_size);
    printf("  ║  Header size:  %12lu bytes                             ║\n",
           (unsigned long)st->header_size);
    printf("  ║  Tensors:      %12d                                   ║\n",
           st->n_tensors);
    printf("  ╚═══════════════════════════════════════════════════════════════╝\n\n");

    const char *dtype_names[] = {
        "F32", "F16", "BF16", "F64", "I8", "I16", "I32", "I64",
        "U8", "BOOL", "???"
    };

    for (int i = 0; i < st->n_tensors; i++) {
        const STTensorInfo *ti = &st->tensors[i];
        printf("  [%3d] %-50s %4s [", i, ti->name,
               dtype_names[ti->dtype < ST_DTYPE_UNKNOWN ? ti->dtype : ST_DTYPE_UNKNOWN]);
        for (int d = 0; d < ti->n_dims; d++) {
            printf("%ld%s", (long)ti->shape[d], d < ti->n_dims - 1 ? "×" : "");
        }
        printf("]  %lu bytes\n", (unsigned long)ti->data_size);
    }
    printf("\n");
}

#endif /* SAFETENSORS_READER_H */
