/*
 * imatrix_reader.h — Importance Matrix File Reader
 *
 * ╔═══════════════════════════════════════════════════════════════╗
 * ║  HExState Importance Matrix Input Module                     ║
 * ║  Reads llama.cpp-compatible .imatrix binary files            ║
 * ║  Provides per-channel importance weights for quantization    ║
 * ╚═══════════════════════════════════════════════════════════════╝
 *
 * Importance matrices capture E[x²] per input channel from calibration
 * data. This information biases quantization toward preserving
 * high-importance channels, significantly improving perplexity at
 * low bit widths (Q2_K).
 *
 * File format (llama.cpp imatrix):
 *   [4 bytes: n_entries (int32)]
 *   For each entry:
 *     [4 bytes: name_len (int32)]
 *     [name_len bytes: tensor name (utf-8, no null terminator)]
 *     [4 bytes: n_values (int32)]
 *     [4 bytes: n_samples (int32)]  -- (count of calibration tokens)
 *     [n_values * 4 bytes: float32 importance values]
 */

#ifndef IMATRIX_READER_H
#define IMATRIX_READER_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IMAT_MAX_ENTRIES  8192
#define IMAT_MAX_NAME_LEN 512

/* ═══════════════════════════════════════════════════════════════════════
 * IMPORTANCE MATRIX ENTRY
 * ═══════════════════════════════════════════════════════════════════════ */

typedef struct {
    char     name[IMAT_MAX_NAME_LEN];
    int32_t  n_values;
    int32_t  n_samples;
    float   *values;       /* Raw importance values (E[x²] per channel) */
    float   *normalized;   /* Normalized: values / mean(values)         */
} IMatrixEntry;

typedef struct {
    IMatrixEntry *entries;
    int32_t       n_entries;
} IMatrixData;

/* ═══════════════════════════════════════════════════════════════════════
 * LOAD IMATRIX FILE
 * ═══════════════════════════════════════════════════════════════════════ */

static IMatrixData *imatrix_load(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "  imatrix_load: cannot open '%s'\n", path);
        return NULL;
    }

    IMatrixData *imat = (IMatrixData *)calloc(1, sizeof(IMatrixData));
    if (!imat) { fclose(f); return NULL; }

    /* Read entry count */
    int32_t n_entries;
    if (fread(&n_entries, sizeof(int32_t), 1, f) != 1 ||
        n_entries <= 0 || n_entries > IMAT_MAX_ENTRIES) {
        fprintf(stderr, "  imatrix_load: invalid entry count %d\n", n_entries);
        free(imat);
        fclose(f);
        return NULL;
    }

    imat->n_entries = n_entries;
    imat->entries = (IMatrixEntry *)calloc(n_entries, sizeof(IMatrixEntry));

    for (int i = 0; i < n_entries; i++) {
        IMatrixEntry *e = &imat->entries[i];

        /* Read tensor name */
        int32_t name_len;
        if (fread(&name_len, sizeof(int32_t), 1, f) != 1) goto fail;
        if (name_len <= 0 || name_len >= IMAT_MAX_NAME_LEN) goto fail;

        if (fread(e->name, 1, name_len, f) != (size_t)name_len) goto fail;
        e->name[name_len] = '\0';

        /* Read value count and sample count */
        if (fread(&e->n_values, sizeof(int32_t), 1, f) != 1) goto fail;
        if (fread(&e->n_samples, sizeof(int32_t), 1, f) != 1) goto fail;

        if (e->n_values <= 0 || e->n_values > 1024 * 1024) goto fail;

        /* Read importance values */
        e->values = (float *)malloc(e->n_values * sizeof(float));
        if (!e->values) goto fail;
        if (fread(e->values, sizeof(float), e->n_values, f) !=
            (size_t)e->n_values) goto fail;

        /* Normalize: divide by mean so that mean(normalized) = 1.0 */
        e->normalized = (float *)malloc(e->n_values * sizeof(float));
        if (!e->normalized) goto fail;

        double sum = 0.0;
        for (int j = 0; j < e->n_values; j++)
            sum += (double)e->values[j];

        double mean = sum / (double)e->n_values;
        if (mean > 1e-30) {
            float inv_mean = (float)(1.0 / mean);
            for (int j = 0; j < e->n_values; j++)
                e->normalized[j] = e->values[j] * inv_mean;
        } else {
            /* Degenerate: all zeros → uniform */
            for (int j = 0; j < e->n_values; j++)
                e->normalized[j] = 1.0f;
        }
    }

    fclose(f);
    return imat;

fail:
    fprintf(stderr, "  imatrix_load: parse error in '%s'\n", path);
    /* Clean up partially loaded data */
    for (int i = 0; i < imat->n_entries; i++) {
        free(imat->entries[i].values);
        free(imat->entries[i].normalized);
    }
    free(imat->entries);
    free(imat);
    fclose(f);
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════
 * FIND IMPORTANCE DATA FOR A TENSOR
 *
 * Looks up by GGUF tensor name. Returns NULL if not found.
 * ═══════════════════════════════════════════════════════════════════════ */

static const IMatrixEntry *imatrix_find(const IMatrixData *imat,
                                         const char *tensor_name)
{
    if (!imat) return NULL;
    for (int i = 0; i < imat->n_entries; i++) {
        if (strcmp(imat->entries[i].name, tensor_name) == 0)
            return &imat->entries[i];
    }
    return NULL;
}

/* Also try the HuggingFace-style tensor name */
static const IMatrixEntry *imatrix_find_any(const IMatrixData *imat,
                                              const char *gguf_name,
                                              const char *hf_name)
{
    const IMatrixEntry *e = imatrix_find(imat, gguf_name);
    if (e) return e;
    return imatrix_find(imat, hf_name);
}

/* ═══════════════════════════════════════════════════════════════════════
 * CLEANUP
 * ═══════════════════════════════════════════════════════════════════════ */

static void imatrix_free(IMatrixData *imat)
{
    if (!imat) return;
    for (int i = 0; i < imat->n_entries; i++) {
        free(imat->entries[i].values);
        free(imat->entries[i].normalized);
    }
    free(imat->entries);
    free(imat);
}

/* ═══════════════════════════════════════════════════════════════════════
 * SUMMARY
 * ═══════════════════════════════════════════════════════════════════════ */

static void imatrix_print_summary(const IMatrixData *imat)
{
    printf("  ╔═══════════════════════════════════════════════════════════════╗\n");
    printf("  ║  Importance Matrix                                          ║\n");
    printf("  ╠═══════════════════════════════════════════════════════════════╣\n");
    printf("  ║  Entries:          %-40d ║\n", imat->n_entries);

    /* Show first few entries as samples */
    int show = imat->n_entries < 5 ? imat->n_entries : 5;
    for (int i = 0; i < show; i++) {
        const IMatrixEntry *e = &imat->entries[i];
        printf("  ║  [%3d] %-30s %6d ch, %4d samples ║\n",
               i, e->name, e->n_values, e->n_samples);
    }
    if (imat->n_entries > 5)
        printf("  ║  ... and %d more entries                                    ║\n",
               imat->n_entries - 5);

    printf("  ╚═══════════════════════════════════════════════════════════════╝\n\n");
}

#endif /* IMATRIX_READER_H */
