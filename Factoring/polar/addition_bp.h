#ifndef ADDITION_BP_H
#define ADDITION_BP_H
#include "sparse_bp.h"

/* C = A + B mod q */
static inline void add_out_c(const double *in_a, const double *in_b, double *out_c) {
    memset(out_c, 0, 3329 * sizeof(double));
    int nsp_a=0, nsp_b=0;
    int idx_a[3329], idx_b[3329];
    for(int i=0; i<3329; i++) {
        if(in_a[i]>1e-12) idx_a[nsp_a++] = i;
        if(in_b[i]>1e-12) idx_b[nsp_b++] = i;
    }
    for(int i=0; i<nsp_a; i++) {
        int a = idx_a[i]; double pa = in_a[a];
        for(int j=0; j<nsp_b; j++) {
            int b = idx_b[j]; double pb = in_b[b];
            int c = mod_q(a + b);
            out_c[c] += pa * pb;
        }
    }
    norm_p_q(out_c);
}

/* A = C - B mod q */
static inline void add_out_a(const double *in_b, const double *in_c, double *out_a) {
    memset(out_a, 0, 3329 * sizeof(double));
    int nsp_b=0, nsp_c=0;
    int idx_b[3329], idx_c[3329];
    for(int i=0; i<3329; i++) {
        if(in_b[i]>1e-12) idx_b[nsp_b++] = i;
        if(in_c[i]>1e-12) idx_c[nsp_c++] = i;
    }
    for(int i=0; i<nsp_c; i++) {
        int c = idx_c[i]; double pc = in_c[c];
        for(int j=0; j<nsp_b; j++) {
            int b = idx_b[j]; double pb = in_b[b];
            int a = mod_q(c - b);
            out_a[a] += pc * pb;
        }
    }
    norm_p_q(out_a);
}

/* B = C - A mod q */
static inline void add_out_b(const double *in_a, const double *in_c, double *out_b) {
    add_out_a(in_a, in_c, out_b);
}

/* Scalar multiplication map: T = a * S mod q */
static inline void scale_out_t(const double *in_s, double *out_t, int a) {
    memset(out_t, 0, 3329 * sizeof(double));
    for(int s=0; s<3329; s++) {
        if(in_s[s] > 1e-12) {
            int t = mod_q(a * s);
            out_t[t] += in_s[s];
        }
    }
    norm_p_q(out_t);
}

/* S = a^-1 * T mod q */
static inline void scale_out_s(const double *in_t, double *out_s, int a_inv) {
    scale_out_t(in_t, out_s, a_inv);
}

#endif
