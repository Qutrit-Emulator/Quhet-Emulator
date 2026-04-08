#ifndef MATRIX_LEAF_H
#define MATRIX_LEAF_H
#include "sparse_bp.h"
#include "addition_bp.h"

// Invert 4x4 matrix mod 3329. Returns 1 if invertible, 0 if singular.
static inline int invert_4x4_mod_q(int M[4][4], int inv[4][4]) {
    int A[4][8];
    for(int i=0;i<4;i++) {
        for(int j=0;j<4;j++) { A[i][j] = mod_q(M[i][j]); A[i][j+4] = (i==j)?1:0; }
    }
    for(int i=0;i<4;i++) {
        int pivot_row = i;
        while(pivot_row < 4 && A[pivot_row][i] == 0) pivot_row++;
        if(pivot_row == 4) return 0; // Singular
        
        if(pivot_row != i) {
            for(int j=0;j<8;j++) { int tmp=A[i][j]; A[i][j]=A[pivot_row][j]; A[pivot_row][j]=tmp; }
        }
        
        int p_inv = 1;
        int p_val = A[i][i];
        // Fermat's Little Theorem: a^(q-2) mod q
        int p = 3327; // 3329 - 2
        int base = p_val;
        while(p > 0) {
            if(p & 1) p_inv = (p_inv * base) % 3329;
            base = (base * base) % 3329;
            p >>= 1;
        }
        
        for(int j=0;j<8;j++) A[i][j] = (A[i][j] * p_inv) % 3329;
        
        for(int k=0;k<4;k++) {
            if(k != i) {
                int factor = A[k][i];
                for(int j=0;j<8;j++) {
                    A[k][j] = mod_q(A[k][j] - factor * A[i][j]);
                }
            }
        }
    }
    for(int i=0;i<4;i++) for(int j=0;j<4;j++) inv[i][j] = A[i][j+4];
    return 1;
}

// Compute output marginal out = M * in
static inline void matrix_vector_bp(int M[4][4], const double in[4][3329], double out[4][3329]) {
    for(int r=0; r<4; r++) {
        double T[4][3329];
        for(int c=0; c<4; c++) {
            scale_out_t(in[c], T[c], M[r][c]);
        }
        double sum1[3329], sum2[3329];
        add_out_c(T[0], T[1], sum1);
        add_out_c(T[2], T[3], sum2);
        add_out_c(sum1, sum2, out[r]);
    }
}

#endif
