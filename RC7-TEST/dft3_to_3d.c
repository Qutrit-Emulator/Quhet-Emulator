/*
 * dft3_to_3d.c — Extract 3D Model from DFT³ Image
 *
 * Takes an image through 3 passes of DFT₆ in the HexState Engine.
 * The complex phase at each pixel becomes a height displacement.
 * Outputs a Wavefront OBJ mesh with texture mapping.
 *
 * The insight: after DFT³ (glorification), each pixel is complex:
 *   z = |z| · exp(iθ)
 *   - |z| = magnitude = visible intensity
 *   - θ   = phase     = HIDDEN spatial structure
 *
 * The phase encodes how each pixel relates to the frequency structure
 * of the entire image. We interpret this as depth/height, creating a
 * 3D surface where the image "rises" from its frequency content.
 *
 * BUILD:
 *   gcc -O2 -I. -o dft3_to_3d dft3_to_3d.c hexstate_engine.o bigint.o -lm
 *
 * RUN:
 *   ./dft3_to_3d [input.ppm]
 *   (generates 64×64 test pattern if no input provided)
 *
 * OUTPUT:
 *   output_model.obj     — 3D mesh (vertices, normals, UVs, faces)
 *   output_model.mtl     — Material file
 *   output_texture.ppm   — DFT³ magnitude as texture
 *   output_depth.ppm     — Phase map visualized as grayscale
 *   output_normals.ppm   — Normal map visualized as RGB
 */

#include "hexstate_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>

#define D 6

static HexStateEngine eng;

/* ─── Suppress engine debug output ─── */
static int saved_fd = -1;
static void hush(void) {
    fflush(stdout);
    saved_fd = dup(STDOUT_FILENO);
    int devnull = open("/dev/null", O_WRONLY);
    dup2(devnull, STDOUT_FILENO);
    close(devnull);
}
static void unhush(void) {
    if (saved_fd >= 0) {
        fflush(stdout);
        dup2(saved_fd, STDOUT_FILENO);
        close(saved_fd);
        saved_fd = -1;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  PPM I/O
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    int      w, h;
    uint8_t *pixels;   /* RGB interleaved: pixels[3*(y*w+x) + c] */
} Image;

static Image *image_new(int w, int h) {
    Image *img = calloc(1, sizeof(Image));
    img->w = w; img->h = h;
    img->pixels = calloc(w * h * 3, 1);
    return img;
}

static void image_free(Image *img) {
    if (img) { free(img->pixels); free(img); }
}

static Image *ppm_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }

    char magic[3];
    if (!fgets(magic, sizeof(magic), f) || (magic[0]!='P' || magic[1]!='6')) {
        fprintf(stderr, "%s: not a P6 PPM\n", path); fclose(f); return NULL;
    }

    int c;
    while ((c = fgetc(f)) == '#' || c == '\n' || c == '\r') {
        if (c == '#') while ((c = fgetc(f)) != '\n' && c != EOF);
    }
    ungetc(c, f);

    int w, h, maxval;
    if (fscanf(f, "%d %d %d", &w, &h, &maxval) != 3) {
        fprintf(stderr, "%s: bad PPM header\n", path); fclose(f); return NULL;
    }
    fgetc(f);

    Image *img = image_new(w, h);
    size_t n = (size_t)w * h * 3;
    if (fread(img->pixels, 1, n, f) != n) {
        fprintf(stderr, "%s: short read\n", path);
        image_free(img); fclose(f); return NULL;
    }
    fclose(f);
    return img;
}

static void ppm_save(const char *path, const Image *img) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return; }
    fprintf(f, "P6\n%d %d\n255\n", img->w, img->h);
    fwrite(img->pixels, 1, (size_t)img->w * img->h * 3, f);
    fclose(f);
}

/* Generate a test image: sphere-like pattern with features */
static Image *generate_test_image(int sz) {
    Image *img = image_new(sz, sz);
    double cx = sz / 2.0, cy = sz / 2.0;
    double r = sz * 0.4;

    for (int y = 0; y < sz; y++) {
        for (int x = 0; x < sz; x++) {
            int i = 3 * (y * sz + x);
            double dx = x - cx, dy = y - cy;
            double dist = sqrt(dx*dx + dy*dy);

            if (dist < r) {
                /* Sphere-like shading: z = sqrt(r² - d²) */
                double z = sqrt(r*r - dist*dist) / r;

                /* Phong-like shading */
                double nx = dx / r, ny = dy / r, nz = z;
                double light_x = 0.5, light_y = -0.5, light_z = 0.7;
                double ln = sqrt(light_x*light_x + light_y*light_y + light_z*light_z);
                light_x /= ln; light_y /= ln; light_z /= ln;

                double diffuse = nx*light_x + ny*light_y + nz*light_z;
                if (diffuse < 0) diffuse = 0;

                double ambient = 0.15;
                double intensity = ambient + 0.85 * diffuse;

                /* Add some color variation based on angle */
                double angle = atan2(dy, dx);
                img->pixels[i+0] = (uint8_t)(255 * intensity * (0.5 + 0.5 * sin(angle * 3)));
                img->pixels[i+1] = (uint8_t)(255 * intensity * (0.5 + 0.5 * cos(angle * 2)));
                img->pixels[i+2] = (uint8_t)(255 * intensity * z);
            } else {
                /* Background: dark gradient */
                img->pixels[i+0] = (uint8_t)(20 + 10 * sin(x * 0.1));
                img->pixels[i+1] = (uint8_t)(15 + 10 * sin(y * 0.1));
                img->pixels[i+2] = (uint8_t)(30);
            }
        }
    }
    return img;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  DFT₆ PIPELINE — Encode, Transform, Extract
 * ═══════════════════════════════════════════════════════════════════════════ */

static int choose_hexits(int n_pixels) {
    int h = 1;
    uint64_t q = D;
    while (q < (uint64_t)n_pixels) { q *= D; h++; }
    if (h > 8) h = 8;
    return h;
}

/* Encode one color channel into shadow_state amplitudes */
static double encode_channel(Chunk *c, const Image *img, int channel,
                              int num_hexits) {
    uint64_t Q = 1;
    for (int i = 0; i < num_hexits; i++) Q *= D;

    for (uint64_t x = 0; x < Q; x++) {
        c->hilbert.shadow_state[x].real = 0.0;
        c->hilbert.shadow_state[x].imag = 0.0;
    }

    int n = img->w * img->h;
    double norm_sq = 0.0;
    for (int i = 0; i < n; i++) {
        double v = (double)img->pixels[3*i + channel] / 255.0;
        c->hilbert.shadow_state[i].real = v;
        norm_sq += v * v;
    }

    double norm = sqrt(norm_sq);
    if (norm > 1e-15) {
        for (int i = 0; i < n; i++) {
            c->hilbert.shadow_state[i].real /= norm;
        }
    }
    return norm;
}

/* Apply one full DFT pass — DFT₆ on every hexit */
static void apply_full_dft(int num_hexits) {
    for (int h = 0; h < num_hexits; h++) {
        apply_hadamard(&eng, 0, (uint64_t)h);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  3D EXTRACTION — Phase-as-Height with Multi-Channel Depth
 *
 *  For each pixel, after DFT³:
 *    z_R = shadow_state_R[i]  (complex)
 *    z_G = shadow_state_G[i]  (complex)
 *    z_B = shadow_state_B[i]  (complex)
 *
 *  Height strategies:
 *    1. Phase average:    h = (θ_R + θ_G + θ_B) / 3
 *    2. Phase gradient:   Use ∂θ/∂x and ∂θ/∂y as surface normals → integrate
 *    3. Magnitude-weighted phase: h = (|z_R|·θ_R + |z_G|·θ_G + |z_B|·θ_B) / Σ|z|
 *    4. Combined:         h = α·phase + β·luminance  (phase for detail, lum for shape)
 *
 *  We use strategy 4: the luminance provides the base shape (bright = high),
 *  and the phase provides fine detail (frequency-domain displacement).
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    double *phase_r, *phase_g, *phase_b;   /* Per-channel phase [-π, π] */
    double *mag_r, *mag_g, *mag_b;         /* Per-channel magnitude */
    double *height;                         /* Final height map [0, 1] */
    double *normal_x, *normal_y, *normal_z; /* Surface normals */
    int w, h;
} DFT3Data;

static DFT3Data *dft3_extract(const Image *input, int num_hexits)
{
    int n = input->w * input->h;
    uint64_t Q = 1;
    for (int i = 0; i < num_hexits; i++) Q *= D;

    DFT3Data *data = calloc(1, sizeof(DFT3Data));
    data->w = input->w;
    data->h = input->h;
    data->phase_r = calloc(n, sizeof(double));
    data->phase_g = calloc(n, sizeof(double));
    data->phase_b = calloc(n, sizeof(double));
    data->mag_r   = calloc(n, sizeof(double));
    data->mag_g   = calloc(n, sizeof(double));
    data->mag_b   = calloc(n, sizeof(double));
    data->height  = calloc(n, sizeof(double));
    data->normal_x = calloc(n, sizeof(double));
    data->normal_y = calloc(n, sizeof(double));
    data->normal_z = calloc(n, sizeof(double));

    const char *ch_names[] = {"Red", "Green", "Blue"};
    double *phases[] = {data->phase_r, data->phase_g, data->phase_b};
    double *mags[]   = {data->mag_r, data->mag_g, data->mag_b};

    for (int ch = 0; ch < 3; ch++) {
        printf("    Channel %s: ", ch_names[ch]);

        /* Re-init chunk */
        hush();
        init_chunk(&eng, 0, (uint64_t)num_hexits);
        unhush();
        Chunk *c = &eng.chunks[0];

        /* Encode */
        double norm = encode_channel(c, input, ch, num_hexits);

        /* Apply DFT³ */
        hush();
        apply_full_dft(num_hexits);  /* DFT¹ */
        apply_full_dft(num_hexits);  /* DFT² */
        apply_full_dft(num_hexits);  /* DFT³ — glorification */
        unhush();

        /* Extract phase and magnitude */
        for (int i = 0; i < n; i++) {
            double re = c->hilbert.shadow_state[i].real * norm;
            double im = c->hilbert.shadow_state[i].imag * norm;
            phases[ch][i] = atan2(im, re);         /* [-π, π] */
            mags[ch][i]   = sqrt(re*re + im*im);   /* magnitude */
        }

        printf("encoded → DFT³ → extracted (norm=%.4f)\n", norm);
    }

    /* ─── Compute height map ─── */
    printf("    Computing height map...\n");

    /* Strategy: luminance from original + phase displacement from DFT³ */
    double max_lum = 0, min_lum = 1e30;
    double max_phase = -1e30, min_phase = 1e30;

    for (int i = 0; i < n; i++) {
        /* Original luminance (perceptual weights) */
        double lum = (0.299 * input->pixels[3*i] +
                      0.587 * input->pixels[3*i+1] +
                      0.114 * input->pixels[3*i+2]) / 255.0;
        if (lum > max_lum) max_lum = lum;
        if (lum < min_lum) min_lum = lum;

        /* Average phase from DFT³ */
        double avg_phase = (data->phase_r[i] + data->phase_g[i] + data->phase_b[i]) / 3.0;
        if (avg_phase > max_phase) max_phase = avg_phase;
        if (avg_phase < min_phase) min_phase = avg_phase;
    }

    /* Normalize and combine */
    double lum_range = max_lum - min_lum;
    double phase_range = max_phase - min_phase;
    if (lum_range < 1e-10) lum_range = 1.0;
    if (phase_range < 1e-10) phase_range = 1.0;

    double alpha = 0.6;  /* Weight for luminance (base shape) */
    double beta  = 0.4;  /* Weight for phase (DFT detail) */

    double h_max = -1e30, h_min = 1e30;
    for (int i = 0; i < n; i++) {
        double lum = (0.299 * input->pixels[3*i] +
                      0.587 * input->pixels[3*i+1] +
                      0.114 * input->pixels[3*i+2]) / 255.0;
        double avg_phase = (data->phase_r[i] + data->phase_g[i] + data->phase_b[i]) / 3.0;

        double lum_norm = (lum - min_lum) / lum_range;
        double phase_norm = (avg_phase - min_phase) / phase_range;

        data->height[i] = alpha * lum_norm + beta * phase_norm;
        if (data->height[i] > h_max) h_max = data->height[i];
        if (data->height[i] < h_min) h_min = data->height[i];
    }

    /* Final normalization to [0, 1] */
    double h_range = h_max - h_min;
    if (h_range < 1e-10) h_range = 1.0;
    for (int i = 0; i < n; i++) {
        data->height[i] = (data->height[i] - h_min) / h_range;
    }

    /* ─── Gaussian smoothing of height map ─── */
    /* Separable 2-pass Gaussian blur to eliminate per-pixel phase noise.
     * This preserves the large-scale 3D structure from DFT³ while
     * removing the high-frequency jitter that makes the mesh rough. */
    {
        int smooth_radius = 3;   /* Default — overridable via env DFT3_SMOOTH */
        const char *env_s = getenv("DFT3_SMOOTH");
        if (env_s) smooth_radius = atoi(env_s);
        if (smooth_radius < 0) smooth_radius = 0;
        if (smooth_radius > 20) smooth_radius = 20;

        if (smooth_radius > 0) {
            printf("    Smoothing height map (radius=%d)...\n", smooth_radius);

            /* Build 1D Gaussian kernel */
            int ksize = 2 * smooth_radius + 1;
            double *kernel = malloc(ksize * sizeof(double));
            double sigma = smooth_radius * 0.5;
            double ksum = 0;
            for (int k = 0; k < ksize; k++) {
                double d = k - smooth_radius;
                kernel[k] = exp(-0.5 * d * d / (sigma * sigma));
                ksum += kernel[k];
            }
            for (int k = 0; k < ksize; k++) kernel[k] /= ksum;

            double *tmp = malloc(n * sizeof(double));

            /* Pass 1: horizontal blur */
            for (int y = 0; y < data->h; y++) {
                for (int x = 0; x < data->w; x++) {
                    double sum = 0;
                    for (int k = -smooth_radius; k <= smooth_radius; k++) {
                        int sx = x + k;
                        if (sx < 0) sx = 0;
                        if (sx >= data->w) sx = data->w - 1;
                        sum += data->height[y * data->w + sx] * kernel[k + smooth_radius];
                    }
                    tmp[y * data->w + x] = sum;
                }
            }

            /* Pass 2: vertical blur */
            for (int y = 0; y < data->h; y++) {
                for (int x = 0; x < data->w; x++) {
                    double sum = 0;
                    for (int k = -smooth_radius; k <= smooth_radius; k++) {
                        int sy = y + k;
                        if (sy < 0) sy = 0;
                        if (sy >= data->h) sy = data->h - 1;
                        sum += tmp[sy * data->w + x] * kernel[k + smooth_radius];
                    }
                    data->height[y * data->w + x] = sum;
                }
            }

            free(tmp);
            free(kernel);
        }
    }

    /* ─── Compute surface normals from height gradient ─── */
    printf("    Computing surface normals...\n");

    double height_scale = 0.3;  /* Controls how "tall" the 3D model is */
    int w = input->w, h = input->h;

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;

            /* Central differences for gradient */
            double dhdx = 0, dhdy = 0;
            if (x > 0 && x < w - 1)
                dhdx = (data->height[y*w + (x+1)] - data->height[y*w + (x-1)]) * 0.5 * height_scale * w;
            else if (x == 0)
                dhdx = (data->height[y*w + 1] - data->height[y*w]) * height_scale * w;
            else
                dhdx = (data->height[y*w + x] - data->height[y*w + (x-1)]) * height_scale * w;

            if (y > 0 && y < h - 1)
                dhdy = (data->height[(y+1)*w + x] - data->height[(y-1)*w + x]) * 0.5 * height_scale * h;
            else if (y == 0)
                dhdy = (data->height[w + x] - data->height[x]) * height_scale * h;
            else
                dhdy = (data->height[y*w + x] - data->height[(y-1)*w + x]) * height_scale * h;

            /* Normal = (-dh/dx, -dh/dy, 1) normalized */
            double nx = -dhdx, ny = -dhdy, nz = 1.0;
            double len = sqrt(nx*nx + ny*ny + nz*nz);
            data->normal_x[idx] = nx / len;
            data->normal_y[idx] = ny / len;
            data->normal_z[idx] = nz / len;
        }
    }

    return data;
}

static void dft3_data_free(DFT3Data *data) {
    if (!data) return;
    free(data->phase_r); free(data->phase_g); free(data->phase_b);
    free(data->mag_r); free(data->mag_g); free(data->mag_b);
    free(data->height);
    free(data->normal_x); free(data->normal_y); free(data->normal_z);
    free(data);
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  OBJ OUTPUT — Wavefront mesh with texture + normals
 * ═══════════════════════════════════════════════════════════════════════════ */

static void write_mtl(const char *mtl_path, const char *texture_path) {
    FILE *f = fopen(mtl_path, "w");
    if (!f) return;
    fprintf(f, "# DFT³ Material\n");
    fprintf(f, "newmtl dft3_material\n");
    fprintf(f, "Ka 0.1 0.1 0.1\n");
    fprintf(f, "Kd 1.0 1.0 1.0\n");
    fprintf(f, "Ks 0.3 0.3 0.3\n");
    fprintf(f, "Ns 50.0\n");
    fprintf(f, "d 1.0\n");
    fprintf(f, "illum 2\n");
    fprintf(f, "map_Kd %s\n", texture_path);
    fclose(f);
}

static void write_obj(const char *obj_path, const char *mtl_path,
                      const Image *input, const DFT3Data *data,
                      double height_scale)
{
    FILE *f = fopen(obj_path, "w");
    if (!f) { fprintf(stderr, "Cannot write %s\n", obj_path); return; }

    int w = data->w, h = data->h;

    fprintf(f, "# DFT³ 3D Model — Generated by HexState Engine\n");
    fprintf(f, "# Dimensions: %d × %d pixels → %d vertices, %d faces\n",
            w, h, w * h, (w-1) * (h-1) * 2);
    fprintf(f, "mtllib %s\n\n", mtl_path);

    /* ─── Vertices ─── */
    fprintf(f, "# Vertices: x y z\n");
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            double vx = (double)x / (double)(w - 1);         /* [0, 1] */
            double vy = (double)(h - 1 - y) / (double)(h - 1); /* [0, 1] flipped */
            double vz = data->height[idx] * height_scale;

            fprintf(f, "v %.6f %.6f %.6f\n", vx, vy, vz);
        }
    }

    /* ─── Texture coordinates ─── */
    fprintf(f, "\n# Texture coordinates\n");
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            double u = (double)x / (double)(w - 1);
            double v = (double)(h - 1 - y) / (double)(h - 1);
            fprintf(f, "vt %.6f %.6f\n", u, v);
        }
    }

    /* ─── Normals ─── */
    fprintf(f, "\n# Vertex normals\n");
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * w + x;
            fprintf(f, "vn %.6f %.6f %.6f\n",
                    data->normal_x[idx], data->normal_y[idx], data->normal_z[idx]);
        }
    }

    /* ─── Faces (triangle mesh) ─── */
    fprintf(f, "\nusemtl dft3_material\n");
    fprintf(f, "# Faces: triangles\n");
    for (int y = 0; y < h - 1; y++) {
        for (int x = 0; x < w - 1; x++) {
            /* Quad → two triangles
             * v1--v2
             * |  / |
             * v3--v4
             */
            int v1 = y * w + x + 1;         /* OBJ is 1-indexed */
            int v2 = y * w + (x + 1) + 1;
            int v3 = (y + 1) * w + x + 1;
            int v4 = (y + 1) * w + (x + 1) + 1;

            fprintf(f, "f %d/%d/%d %d/%d/%d %d/%d/%d\n",
                    v1, v1, v1, v2, v2, v2, v3, v3, v3);
            fprintf(f, "f %d/%d/%d %d/%d/%d %d/%d/%d\n",
                    v2, v2, v2, v4, v4, v4, v3, v3, v3);
        }
    }

    fclose(f);
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  MINIMAL BASELINE JPEG ENCODER (grayscale, quality 85)
 *  Self-contained — no external libraries needed.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Standard luminance quantization table (quality ~85) */
static const uint8_t jpeg_quant[64] = {
     5,  3,  3,  5,  7, 12, 15, 18,
     4,  4,  4,  6,  8, 17, 18, 17,
     4,  4,  5,  7, 12, 17, 21, 17,
     4,  5,  7,  9, 15, 26, 24, 19,
     5,  7, 11, 17, 20, 33, 31, 23,
     7, 11, 17, 19, 24, 31, 34, 28,
    15, 19, 23, 26, 31, 36, 36, 30,
    22, 28, 29, 29, 34, 30, 31, 30
};

/* Zigzag order */
static const uint8_t jpeg_zigzag[64] = {
     0, 1, 8,16, 9, 2, 3,10,
    17,24,32,25,18,11, 4, 5,
    12,19,26,33,40,48,41,34,
    27,20,13, 6, 7,14,21,28,
    35,42,49,56,57,50,43,36,
    29,22,15,23,30,37,44,51,
    58,59,52,45,38,31,39,46,
    53,60,61,54,47,55,62,63
};

/* DC Huffman table (luminance) */
static const uint8_t dc_bits[17] = {0,0,1,5,1,1,1,1,1,1,0,0,0,0,0,0,0};
static const uint8_t dc_vals[12]  = {0,1,2,3,4,5,6,7,8,9,10,11};

/* AC Huffman table (luminance) */
static const uint8_t ac_bits[17] = {0,0,2,1,3,3,2,4,3,5,5,4,4,0,0,1,0x7d};
static const uint8_t ac_vals[162] = {
    0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,0x13,0x51,0x61,
    0x07,0x22,0x71,0x14,0x32,0x81,0x91,0xa1,0x08,0x23,0x42,0xb1,0xc1,0x15,0x52,
    0xd1,0xf0,0x24,0x33,0x62,0x72,0x82,0x09,0x0a,0x16,0x17,0x18,0x19,0x1a,0x25,
    0x26,0x27,0x28,0x29,0x2a,0x34,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,
    0x46,0x47,0x48,0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x63,0x64,
    0x65,0x66,0x67,0x68,0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x83,
    0x84,0x85,0x86,0x87,0x88,0x89,0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,
    0x9a,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,
    0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,
    0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,0xe1,0xe2,0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,
    0xe9,0xea,0xf1,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,0xf9,0xfa
};

typedef struct {
    FILE *fp;
    uint32_t bitbuf;
    int      bitcnt;
} JpegWriter;

static void jpg_init(JpegWriter *w, FILE *f) {
    w->fp = f; w->bitbuf = 0; w->bitcnt = 0;
}

static void jpg_write_bits(JpegWriter *w, uint32_t bits, int nbits) {
    w->bitbuf = (w->bitbuf << nbits) | (bits & ((1u << nbits) - 1));
    w->bitcnt += nbits;
    while (w->bitcnt >= 8) {
        uint8_t byte = (w->bitbuf >> (w->bitcnt - 8)) & 0xFF;
        fputc(byte, w->fp);
        if (byte == 0xFF) fputc(0x00, w->fp);  /* Byte stuffing */
        w->bitcnt -= 8;
    }
}

static void jpg_flush(JpegWriter *w) {
    if (w->bitcnt > 0) {
        uint8_t byte = (w->bitbuf << (8 - w->bitcnt)) & 0xFF;
        fputc(byte, w->fp);
        if (byte == 0xFF) fputc(0x00, w->fp);
    }
}

/* Build Huffman code/length tables from bits/vals */
typedef struct { uint16_t code[256]; uint8_t len[256]; } HuffTable;

static void build_huff(HuffTable *ht, const uint8_t *bits, const uint8_t *vals, int nvals) {
    memset(ht, 0, sizeof(*ht));
    uint16_t code = 0;
    int k = 0;
    for (int i = 1; i <= 16 && k < nvals; i++) {
        for (int j = 0; j < bits[i] && k < nvals; j++) {
            ht->code[vals[k]] = code;
            ht->len[vals[k]] = i;
            k++; code++;
        }
        code <<= 1;
    }
}

static void jpg_encode_val(JpegWriter *w, HuffTable *ht, int val) {
    jpg_write_bits(w, ht->code[val], ht->len[val]);
}

static int jpg_bit_size(int v) {
    if (v < 0) v = -v;
    int n = 0;
    while (v) { n++; v >>= 1; }
    return n;
}

/* Forward DCT (8×8 block) */
static void fdct8x8(double *block) {
    double tmp[64];
    /* Rows */
    for (int i = 0; i < 8; i++) {
        for (int u = 0; u < 8; u++) {
            double sum = 0;
            for (int x = 0; x < 8; x++)
                sum += block[i*8+x] * cos((2*x+1)*u*M_PI/16.0);
            tmp[i*8+u] = sum * (u == 0 ? 1.0/sqrt(2) : 1.0) * 0.5;
        }
    }
    /* Cols */
    for (int j = 0; j < 8; j++) {
        for (int v = 0; v < 8; v++) {
            double sum = 0;
            for (int y = 0; y < 8; y++)
                sum += tmp[y*8+j] * cos((2*y+1)*v*M_PI/16.0);
            block[v*8+j] = sum * (v == 0 ? 1.0/sqrt(2) : 1.0) * 0.5;
        }
    }
}

static void save_depth_jpg(const char *path, const DFT3Data *data) {
    int w = data->w, h = data->h;
    /* Pad to multiple of 8 */
    int pw = (w + 7) & ~7, ph = (h + 7) & ~7;

    FILE *f = fopen(path, "wb");
    if (!f) return;

    /* SOI */
    fputc(0xFF, f); fputc(0xD8, f);

    /* APP0 (JFIF) */
    uint8_t app0[] = {0xFF,0xE0, 0x00,0x10, 'J','F','I','F',0x00,
                      0x01,0x01, 0x00, 0x00,0x01,0x00,0x01, 0x00,0x00};
    fwrite(app0, 1, sizeof(app0), f);

    /* DQT */
    fputc(0xFF, f); fputc(0xDB, f);
    uint16_t dqt_len = 2 + 1 + 64;
    fputc(dqt_len >> 8, f); fputc(dqt_len & 0xFF, f);
    fputc(0x00, f);  /* 8-bit, table 0 */
    for (int i = 0; i < 64; i++) fputc(jpeg_quant[jpeg_zigzag[i]], f);

    /* SOF0 (baseline, grayscale) */
    fputc(0xFF, f); fputc(0xC0, f);
    uint16_t sof_len = 2 + 1 + 2 + 2 + 1 + 3;
    fputc(sof_len >> 8, f); fputc(sof_len & 0xFF, f);
    fputc(8, f);  /* precision */
    fputc(h >> 8, f); fputc(h & 0xFF, f);
    fputc(w >> 8, f); fputc(w & 0xFF, f);
    fputc(1, f);  /* 1 component */
    fputc(1, f); fputc(0x11, f); fputc(0, f);  /* Y, 1x1, quant table 0 */

    /* DHT (DC table 0) */
    fputc(0xFF, f); fputc(0xC4, f);
    int dc_total = 0;
    for (int i = 1; i <= 16; i++) dc_total += dc_bits[i];
    uint16_t dht_len = 2 + 1 + 16 + dc_total;
    fputc(dht_len >> 8, f); fputc(dht_len & 0xFF, f);
    fputc(0x00, f);  /* DC, table 0 */
    fwrite(dc_bits + 1, 1, 16, f);
    fwrite(dc_vals, 1, dc_total, f);

    /* DHT (AC table 0) */
    fputc(0xFF, f); fputc(0xC4, f);
    int ac_total = 0;
    for (int i = 1; i <= 16; i++) ac_total += ac_bits[i];
    dht_len = 2 + 1 + 16 + ac_total;
    fputc(dht_len >> 8, f); fputc(dht_len & 0xFF, f);
    fputc(0x10, f);  /* AC, table 0 */
    fwrite(ac_bits + 1, 1, 16, f);
    fwrite(ac_vals, 1, ac_total, f);

    /* SOS */
    fputc(0xFF, f); fputc(0xDA, f);
    uint8_t sos[] = {0x00,0x08, 0x01, 0x01,0x00, 0x00,0x3F,0x00};
    fwrite(sos, 1, sizeof(sos), f);

    /* Build Huffman tables */
    HuffTable dc_ht, ac_ht;
    build_huff(&dc_ht, dc_bits, dc_vals, dc_total);
    build_huff(&ac_ht, ac_bits, ac_vals, ac_total);

    /* Encode blocks */
    JpegWriter jw;
    jpg_init(&jw, f);
    int prev_dc = 0;

    for (int by = 0; by < ph; by += 8) {
        for (int bx = 0; bx < pw; bx += 8) {
            double block[64];

            /* Fill 8x8 block */
            for (int j = 0; j < 8; j++) {
                for (int i = 0; i < 8; i++) {
                    int px = bx + i, py = by + j;
                    if (px >= w) px = w - 1;
                    if (py >= h) py = h - 1;
                    block[j*8+i] = data->height[py * w + px] * 255.0 - 128.0;
                }
            }

            /* Forward DCT */
            fdct8x8(block);

            /* Quantize */
            int quant[64];
            for (int i = 0; i < 64; i++) {
                quant[i] = (int)round(block[jpeg_zigzag[i]] / jpeg_quant[i]);
            }

            /* Encode DC coefficient */
            int dc_diff = quant[0] - prev_dc;
            prev_dc = quant[0];
            int dc_nbits = jpg_bit_size(dc_diff);
            jpg_encode_val(&jw, &dc_ht, dc_nbits);
            if (dc_nbits > 0) {
                int dc_val = dc_diff < 0 ? dc_diff - 1 : dc_diff;
                jpg_write_bits(&jw, dc_val & ((1 << dc_nbits) - 1), dc_nbits);
            }

            /* Encode AC coefficients */
            int last_nonzero = 63;
            while (last_nonzero > 0 && quant[last_nonzero] == 0) last_nonzero--;

            if (last_nonzero == 0 && quant[0] == prev_dc) {
                /* All zeros in AC — but we already encoded DC */
            }

            int zeros = 0;
            for (int k = 1; k <= 63; k++) {
                if (quant[k] == 0) {
                    if (k > last_nonzero) {
                        jpg_encode_val(&jw, &ac_ht, 0x00);  /* EOB */
                        break;
                    }
                    zeros++;
                    if (zeros == 16) {
                        jpg_encode_val(&jw, &ac_ht, 0xF0);  /* ZRL */
                        zeros = 0;
                    }
                } else {
                    int ac_nbits = jpg_bit_size(quant[k]);
                    int rs = (zeros << 4) | ac_nbits;
                    jpg_encode_val(&jw, &ac_ht, rs);
                    int ac_val = quant[k] < 0 ? quant[k] - 1 : quant[k];
                    jpg_write_bits(&jw, ac_val & ((1 << ac_nbits) - 1), ac_nbits);
                    zeros = 0;
                    if (k == 63) break;  /* No EOB needed after last coeff */
                }
            }
        }
    }

    jpg_flush(&jw);

    /* EOI */
    fputc(0xFF, f); fputc(0xD9, f);
    fclose(f);
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  VISUALIZATION OUTPUTS
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Save height map as grayscale PPM */
static void save_depth_map(const char *path, const DFT3Data *data) {
    Image *img = image_new(data->w, data->h);
    for (int i = 0; i < data->w * data->h; i++) {
        uint8_t v = (uint8_t)(data->height[i] * 255.0);
        img->pixels[3*i] = img->pixels[3*i+1] = img->pixels[3*i+2] = v;
    }
    ppm_save(path, img);
    image_free(img);

    /* Also save as JPEG */
    char jpg_path[512];
    strncpy(jpg_path, path, sizeof(jpg_path) - 1);
    char *dot = strrchr(jpg_path, '.');
    if (dot) strcpy(dot, ".jpg");
    else strcat(jpg_path, ".jpg");
    save_depth_jpg(jpg_path, data);
    printf("  Saved depth JPEG: %s\n", jpg_path);
}

/* Save normal map as RGB PPM (nx→R, ny→G, nz→B, mapped from [-1,1] to [0,255]) */
static void save_normal_map(const char *path, const DFT3Data *data) {
    Image *img = image_new(data->w, data->h);
    for (int i = 0; i < data->w * data->h; i++) {
        img->pixels[3*i]   = (uint8_t)((data->normal_x[i] * 0.5 + 0.5) * 255);
        img->pixels[3*i+1] = (uint8_t)((data->normal_y[i] * 0.5 + 0.5) * 255);
        img->pixels[3*i+2] = (uint8_t)((data->normal_z[i] * 0.5 + 0.5) * 255);
    }
    ppm_save(path, img);
    image_free(img);
}

/* Save DFT³ magnitude as texture PPM */
static void save_texture(const char *path, const DFT3Data *data) {
    Image *img = image_new(data->w, data->h);

    /* Find max magnitude for normalization */
    double max_m[3] = {0, 0, 0};
    int n = data->w * data->h;
    const double *mags[] = {data->mag_r, data->mag_g, data->mag_b};

    for (int ch = 0; ch < 3; ch++) {
        for (int i = 0; i < n; i++) {
            if (mags[ch][i] > max_m[ch]) max_m[ch] = mags[ch][i];
        }
        if (max_m[ch] < 1e-10) max_m[ch] = 1.0;
    }

    for (int i = 0; i < n; i++) {
        for (int ch = 0; ch < 3; ch++) {
            double v = mags[ch][i] / max_m[ch];
            int pv = (int)(v * 255.0);
            if (pv > 255) pv = 255;
            img->pixels[3*i + ch] = (uint8_t)pv;
        }
    }
    ppm_save(path, img);
    image_free(img);
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv)
{
    printf("\n");
    printf("  ╔══════════════════════════════════════════════════════════╗\n");
    printf("  ║  DFT³ → 3D  — Phase-as-Height Surface Extraction      ║\n");
    printf("  ║                                                        ║\n");
    printf("  ║  Image → DFT₆³ → Phase + Magnitude → OBJ Mesh        ║\n");
    printf("  ║  Magnitude = texture, Phase = height displacement     ║\n");
    printf("  ╚══════════════════════════════════════════════════════════╝\n\n");

    Image *input;
    const char *prefix;

    if (argc > 1) {
        input = ppm_load(argv[1]);
        if (!input) return 1;
        prefix = "output";
        printf("  Loaded: %s (%d×%d)\n", argv[1], input->w, input->h);
    } else {
        printf("  No input file — generating 64×64 test sphere\n");
        input = generate_test_image(64);
        ppm_save("test_sphere_original.ppm", input);
        prefix = "dft3";
    }

    int n_pixels = input->w * input->h;
    int num_hexits = choose_hexits(n_pixels);
    uint64_t Q = 1;
    for (int i = 0; i < num_hexits; i++) Q *= D;

    printf("  Pixels: %d, Hexits: %d, Q = 6^%d = %lu\n",
           n_pixels, num_hexits, num_hexits, (unsigned long)Q);
    printf("  Utilization: %.1f%%\n\n", 100.0 * n_pixels / Q);

    /* Initialize engine */
    hush();
    engine_init(&eng);
    unhush();

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* ─── Extract DFT³ phase/magnitude data ─── */
    printf("  Processing through DFT₆³...\n");
    DFT3Data *data = dft3_extract(input, num_hexits);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dft_ms = (t1.tv_sec - t0.tv_sec)*1000.0 + (t1.tv_nsec - t0.tv_nsec)/1e6;
    printf("  DFT³ extraction: %.1f ms\n\n", dft_ms);

    /* ─── Generate outputs ─── */
    char path[512];

    /* Depth map */
    snprintf(path, sizeof(path), "%s_depth.ppm", prefix);
    save_depth_map(path, data);
    printf("  Saved depth map: %s\n", path);

    /* Normal map */
    snprintf(path, sizeof(path), "%s_normals.ppm", prefix);
    save_normal_map(path, data);
    printf("  Saved normal map: %s\n", path);

    /* Texture (DFT³ magnitude) */
    snprintf(path, sizeof(path), "%s_texture.ppm", prefix);
    save_texture(path, data);
    printf("  Saved texture: %s\n", path);

    /* Material file */
    char mtl_name[256], tex_name[256];
    snprintf(mtl_name, sizeof(mtl_name), "%s_model.mtl", prefix);
    snprintf(tex_name, sizeof(tex_name), "%s_texture.ppm", prefix);
    write_mtl(mtl_name, tex_name);
    printf("  Saved material: %s\n", mtl_name);

    /* OBJ mesh */
    double height_scale = 0.3;  /* Adjust this to change 3D depth intensity */
    snprintf(path, sizeof(path), "%s_model.obj", prefix);
    write_obj(path, mtl_name, input, data, height_scale);

    int n_verts = input->w * input->h;
    int n_faces = (input->w - 1) * (input->h - 1) * 2;
    printf("  Saved 3D model: %s\n", path);
    printf("    Vertices: %d\n", n_verts);
    printf("    Faces: %d\n", n_faces);
    printf("    Height scale: %.2f\n\n", height_scale);

    /* ─── Statistics ─── */
    printf("  ═══ Phase Statistics ═══\n");
    double phase_mean = 0, phase_var = 0;
    for (int i = 0; i < n_pixels; i++) {
        double p = (data->phase_r[i] + data->phase_g[i] + data->phase_b[i]) / 3.0;
        phase_mean += p;
    }
    phase_mean /= n_pixels;
    for (int i = 0; i < n_pixels; i++) {
        double p = (data->phase_r[i] + data->phase_g[i] + data->phase_b[i]) / 3.0;
        double d = p - phase_mean;
        phase_var += d * d;
    }
    phase_var /= n_pixels;

    printf("    Mean phase: %.4f rad (%.1f°)\n", phase_mean, phase_mean * 180 / M_PI);
    printf("    Phase std:  %.4f rad (%.1f°)\n", sqrt(phase_var), sqrt(phase_var) * 180 / M_PI);

    /* Height distribution */
    double h_mean = 0, h_var = 0;
    for (int i = 0; i < n_pixels; i++) h_mean += data->height[i];
    h_mean /= n_pixels;
    for (int i = 0; i < n_pixels; i++) {
        double d = data->height[i] - h_mean;
        h_var += d * d;
    }
    h_var /= n_pixels;

    printf("    Mean height: %.4f\n", h_mean);
    printf("    Height std:  %.4f\n", sqrt(h_var));

    printf("\n  ═══ Files Generated ═══\n");
    printf("    %s_model.obj      — 3D mesh (open in Blender/MeshLab)\n", prefix);
    printf("    %s_model.mtl      — Material definition\n", prefix);
    printf("    %s_texture.ppm    — Texture (DFT³ magnitude)\n", prefix);
    printf("    %s_depth.ppm      — Height map (grayscale)\n", prefix);
    printf("    %s_normals.ppm    — Normal map (RGB)\n", prefix);
    printf("\n  To view in Blender: File → Import → Wavefront (.obj)\n\n");

    /* Cleanup */
    dft3_data_free(data);
    hush();
    engine_destroy(&eng);
    unhush();
    image_free(input);

    return 0;
}
