#ifndef SPARSE_BP_H
#define SPARSE_BP_H

#include <string.h>
#include <math.h>

static inline void norm_p_q(double *p) {
    double s = 0.0;
    for (int i = 0; i < 3329; i++) s += p[i];
    if (s < 1e-300) return;
    
    double threshold = 1e-8;
    s = 0.0;
    for (int i = 0; i < 3329; i++) {
        if (p[i] < threshold) p[i] = 0.0;
        else s += p[i];
    }
    
    if (s > 1e-300) {
        double inv = 1.0 / s;
        for (int i = 0; i < 3329; i++) p[i] *= inv;
    }
}

/* Butterfly: x = a + Zb, y = a - Zb mod Q */

static inline void bf_out_a(const double *in_b, const double *in_x, const double *in_y, double *out_a, int Z) {
    memset(out_a, 0, 3329 * sizeof(double));
    int nsp_b=0; int idx_b[3329];
    int nsp_x=0; int idx_x[3329];
    for(int i=0; i<3329; i++) {
        if(in_b[i]>1e-12) idx_b[nsp_b++] = i;
        if(in_x[i]>1e-12) idx_x[nsp_x++] = i;
    }
    int has_y = 0;
    for(int i=0; i<3329; i++) if(in_y[i] > 1e-12) { has_y = 1; break; }
    
    for(int i=0; i<nsp_b; i++) {
        int b = idx_b[i]; double pb = in_b[b];
        int zb = (Z * b) % 3329;
        for(int j=0; j<nsp_x; j++) {
            int x = idx_x[j]; double px = in_x[x];
            int a = mod_q(x - zb);
            int y = mod_q(a - zb);
            if (has_y) {
                double py = in_y[y];
                if (py < 1e-12) py = 1e-8;
                out_a[a] += pb * px * py;
            } else {
                out_a[a] += pb * px;
            }
        }
    }
    norm_p_q(out_a);
}

static inline void bf_out_b(const double *in_a, const double *in_x, const double *in_y, double *out_b, int Z) {
    memset(out_b, 0, 3329 * sizeof(double));
    int nsp_a=0; int idx_a[3329];
    int nsp_x=0; int idx_x[3329];
    for(int i=0; i<3329; i++) {
        if(in_a[i]>1e-12) idx_a[nsp_a++] = i;
        if(in_x[i]>1e-12) idx_x[nsp_x++] = i;
    }
    int has_y = 0;
    for(int i=0; i<3329; i++) if(in_y[i] > 1e-12) { has_y = 1; break; }
    
    /* Fermat's Little Theorem: Z_inv = Z^(q-2) mod q */
    int Z_inv = 1;
    int p = 3327;
    int base = Z;
    while(p > 0) {
        if(p & 1) Z_inv = (Z_inv * base) % 3329;
        base = (base * base) % 3329;
        p >>= 1;
    }
    
    for(int i=0; i<nsp_a; i++) {
        int a = idx_a[i]; double pa = in_a[a];
        for(int j=0; j<nsp_x; j++) {
            int x = idx_x[j]; double px = in_x[x];
            int zb = mod_q(x - a);
            int b = mod_q(zb * Z_inv);
            int y = mod_q(a - zb);
            if (has_y) {
                double py = in_y[y];
                if (py < 1e-12) py = 1e-8;
                out_b[b] += pa * px * py;
            } else {
                out_b[b] += pa * px;
            }
        }
    }
    norm_p_q(out_b);
}

static inline void bf_out_x(const double *in_b, const double *in_a, const double *in_y, double *out_x, int Z) {
    memset(out_x, 0, 3329 * sizeof(double));
    int nsp_b=0; int idx_b[3329];
    int nsp_a=0; int idx_a[3329];
    for(int i=0; i<3329; i++) {
        if(in_b[i]>1e-12) idx_b[nsp_b++] = i;
        if(in_a[i]>1e-12) idx_a[nsp_a++] = i;
    }
    int has_y = 0;
    for(int i=0; i<3329; i++) if(in_y[i] > 1e-12) { has_y = 1; break; }
    
    for(int i=0; i<nsp_b; i++) {
        int b = idx_b[i]; double pb = in_b[b];
        int zb = (Z * b) % 3329;
        int z2b = (2 * zb) % 3329;
        for(int j=0; j<nsp_a; j++) {
            int a = idx_a[j]; double pa = in_a[a];
            int x = mod_q(a + zb);
            int y = mod_q(x - z2b);
            if (has_y) {
                double py = in_y[y];
                if (py < 1e-12) py = 1e-8;
                out_x[x] += pb * pa * py;
            } else {
                out_x[x] += pb * pa;
            }
        }
    }
    norm_p_q(out_x);
}

static inline void bf_out_y(const double *in_b, const double *in_a, const double *in_x, double *out_y, int Z) {
    memset(out_y, 0, 3329 * sizeof(double));
    int nsp_b=0; int idx_b[3329];
    int nsp_a=0; int idx_a[3329];
    for(int i=0; i<3329; i++) {
        if(in_b[i]>1e-12) idx_b[nsp_b++] = i;
        if(in_a[i]>1e-12) idx_a[nsp_a++] = i;
    }
    int has_x = 0;
    for(int i=0; i<3329; i++) if(in_x[i] > 1e-12) { has_x = 1; break; }
    
    for(int i=0; i<nsp_b; i++) {
        int b = idx_b[i]; double pb = in_b[b];
        int zb = (Z * b) % 3329;
        int z2b = (2 * zb) % 3329;
        for(int j=0; j<nsp_a; j++) {
            int a = idx_a[j]; double pa = in_a[a];
            int y = mod_q(a - zb);
            int x = mod_q(y + z2b);
            if(has_x) {
                double px = in_x[x];
                if (px < 1e-12) px = 1e-8;
                out_y[y] += pb * pa * px;
            } else {
                out_y[y] += pb * pa;
            }
        }
    }
    norm_p_q(out_y);
}

/* LEAF CHECK NODES */
static inline void leaf_out_s0(const double *in_s1, const double *in_y0, const double *in_y1, 
                        double *out_s0, int a0, int a1, int Z) {
    memset(out_s0, 0, 3329 * sizeof(double));
    int nsp_s1=0; int idx_s1[3329];
    for(int i=0; i<3329; i++) if(in_s1[i]>1e-12) idx_s1[nsp_s1++] = i;
    for (int i=0; i<nsp_s1; i++) {
        int s1 = idx_s1[i]; double ps1 = in_s1[s1];
        int z_a1_s1 = mod_q(Z * mod_q(a1 * s1));
        int a0_s1 = mod_q(a0 * s1);
        for (int v = 0; v < 3329; v++) {
            int y0 = mod_q(a0 * v + z_a1_s1);
            if (in_y0[y0] < 1e-30) continue;
            int y1 = mod_q(a1 * v + a0_s1);
            double p = in_y0[y0] * in_y1[y1];
            if (p > 1e-30) out_s0[v] += ps1 * p;
        }
    }
    norm_p_q(out_s0);
}

static inline void leaf_out_s1(const double *in_s0, const double *in_y0, const double *in_y1, 
                        double *out_s1, int a0, int a1, int Z) {
    memset(out_s1, 0, 3329 * sizeof(double));
    int nsp_s0=0; int idx_s0[3329];
    for(int i=0; i<3329; i++) if(in_s0[i]>1e-12) idx_s0[nsp_s0++] = i;
    for (int i=0; i<nsp_s0; i++) {
        int s0 = idx_s0[i]; double ps0 = in_s0[s0];
        int a0_s0 = mod_q(a0 * s0);
        int a1_s0 = mod_q(a1 * s0);
        for (int v = 0; v < 3329; v++) {
            int y0 = mod_q(a0_s0 + mod_q(Z * mod_q(a1 * v)));
            if (in_y0[y0] < 1e-30) continue;
            int y1 = mod_q(a1_s0 + a0 * v);
            double p = in_y0[y0] * in_y1[y1];
            if (p > 1e-30) out_s1[v] += ps0 * p;
        }
    }
    norm_p_q(out_s1);
}

static inline void leaf_out_y0(const double *in_s0, const double *in_s1, const double *in_y1, 
                        double *out_y0, int a0, int a1, int Z) {
    memset(out_y0, 0, 3329 * sizeof(double));
    int nsp_s0=0; int idx_s0[3329];
    for(int i=0; i<3329; i++) if(in_s0[i]>1e-12) idx_s0[nsp_s0++] = i;
    for (int i=0; i<nsp_s0; i++) {
        int s0 = idx_s0[i]; double ps0 = in_s0[s0];
        int a0_s0 = mod_q(a0 * s0);   int a1_s0 = mod_q(a1 * s0);
        for (int s1 = 0; s1 < 3329; s1++) {
            int y1 = mod_q(a1_s0 + a0 * s1);
            if (in_y1[y1] < 1e-30) continue;
            if (in_s1[s1] < 1e-30) continue;
            int y0 = mod_q(a0_s0 + mod_q(Z * mod_q(a1 * s1)));
            double p = ps0 * in_s1[s1] * in_y1[y1];
            out_y0[y0] += p;
        }
    }
    norm_p_q(out_y0);
}

static inline void leaf_out_y1(const double *in_s0, const double *in_s1, const double *in_y0, 
                        double *out_y1, int a0, int a1, int Z) {
    memset(out_y1, 0, 3329 * sizeof(double));
    int nsp_s0=0; int idx_s0[3329];
    for(int i=0; i<3329; i++) if(in_s0[i]>1e-12) idx_s0[nsp_s0++] = i;
    for (int i=0; i<nsp_s0; i++) {
        int s0 = idx_s0[i]; double ps0 = in_s0[s0];
        int a0_s0 = mod_q(a0 * s0);   int a1_s0 = mod_q(a1 * s0);
        for (int s1 = 0; s1 < 3329; s1++) {
            int y0 = mod_q(a0_s0 + mod_q(Z * mod_q(a1 * s1)));
            if (in_y0[y0] < 1e-30) continue;
            if (in_s1[s1] < 1e-30) continue;
            int y1 = mod_q(a1_s0 + a0 * s1);
            double p = ps0 * in_s1[s1] * in_y0[y0];
            out_y1[y1] += p;
        }
    }
    norm_p_q(out_y1);
}

#endif
