#include "sparse_bp.h"

typedef struct { double p[2][3329]; } VarNode;

static void polar_bp_k2(const KyberInstance *K, int *s_out, double marg[512][7],
                        const int *fixed, const int *fval, double *s_out_f, unsigned seed,
                        int quiet, int max_iters)
{
    VarNode (*S)[8][256] = calloc(2, 8 * 256 * sizeof(VarNode));
    VarNode (*Y)[8][256] = calloc(2, 8 * 256 * sizeof(VarNode));
    VarNode (*new_S)[8][256] = calloc(2, 8 * 256 * sizeof(VarNode));
    VarNode (*new_Y)[8][256] = calloc(2, 8 * 256 * sizeof(VarNode));
    
    double B_eta[3329] = {0};
    double pr[7]; compute_prior(pr, KYBER_ETA, 7);
    for(int v=0; v<7; v++) B_eta[mod_q(v - KYBER_ETA)] = pr[v];
    
    for(int c=0; c<2; c++) {
        for(int l=0; l<8; l++) {
            for(int i=0; i<256; i++) {
                for(int v=0; v<3329; v++) {
                    S[c][l][i].p[0][v] = S[c][l][i].p[1][v] = 0.0;
                    Y[c][l][i].p[0][v] = Y[c][l][i].p[1][v] = 0.0;
                }
            }
        }
    }
    
    int spatial_b[2][256];
    for(int c=0; c<2; c++) {
        int b_ntt[256];
        for(int i=0; i<256; i++) b_ntt[i] = K->b[c][i];
        inv_ntt_c(b_ntt); // In-place inverse NTT
        for(int i=0; i<256; i++) spatial_b[c][i] = b_ntt[i];
    }
    
    for(int c=0; c<2; c++) {
        for(int i=0; i<256; i++) {
            int g_idx = c * 256 + i;
            if (fixed[g_idx]) {
                int fv = mod_q(fval[g_idx]);
                memset(S[c][0][i].p[0], 0, 3329*sizeof(double));
                S[c][0][i].p[0][fv] = 1.0;
            } else {
                for(int v=0; v<3329; v++) {
                    double p = B_eta[v];
                    if (s_out_f) {
                        int val = v > 1664 ? v - 3329 : v;
                        double dist = s_out_f[g_idx] - val;
                    /* Sharp Gaussian: sigma=0.5, creates real contrast between adjacent integers */
                    p *= exp(-4.0 * dist * dist);
                    }
                    S[c][0][i].p[0][v] = p;
                }
                norm_p_q(S[c][0][i].p[0]);
            }
            
            for(int v=0; v<3329; v++) {
                Y[c][0][i].p[0][v] = B_eta[mod_q(spatial_b[c][i] - v)];
            }
        }
    }
    
    if(!quiet) printf("  [Polar BP] Commencing Resonance Wave (k=2)...\n");
    
    for(int it=0; it<max_iters; it++) {
        double max_delta = 0.0;
        
        /* FORWARD PASS (l=0 to 6) */
        for(int l=0; l<7; l++) {
            int len = 128 >> l;
            int k_idx = (1 << l);
            
            for(int c=0; c<2; c++) {
                memcpy(new_S[c][l], S[c][l], 256*sizeof(VarNode));
                memcpy(new_Y[c][l], Y[c][l], 256*sizeof(VarNode));
                memcpy(new_S[c][l+1], S[c][l+1], 256*sizeof(VarNode));
                memcpy(new_Y[c][l+1], Y[c][l+1], 256*sizeof(VarNode));
            }
            
            #pragma omp parallel for reduction(max:max_delta)
            for(int obj=0; obj<128; obj++) {
                int group = obj / len;
                int start = group * 2 * len;
                int left = start + obj % len;
                int right = left + len;
                int zeta = bm_zetas[k_idx + group];
                
                for(int c=0; c<2; c++) {
                    bf_out_x(S[c][l][right].p[0], S[c][l][left].p[0], S[c][l+1][right].p[1], new_S[c][l+1][left].p[0], zeta);
                    bf_out_y(S[c][l][right].p[0], S[c][l][left].p[0], S[c][l+1][left].p[1], new_S[c][l+1][right].p[0], zeta);
                    
                    bf_out_x(Y[c][l][right].p[0], Y[c][l][left].p[0], Y[c][l+1][right].p[1], new_Y[c][l+1][left].p[0], zeta);
                    bf_out_y(Y[c][l][right].p[0], Y[c][l][left].p[0], Y[c][l+1][left].p[1], new_Y[c][l+1][right].p[0], zeta);
                }
            }
            
            for(int c=0; c<2; c++) {
                for(int i=0; i<256; i++) {
                    for(int v=0; v<3329; v++) {
                        double uS = 0.85*new_S[c][l+1][i].p[0][v] + 0.15*S[c][l+1][i].p[0][v];
                        double uY = 0.85*new_Y[c][l+1][i].p[0][v] + 0.15*Y[c][l+1][i].p[0][v];
                        double dS = uS - S[c][l+1][i].p[0][v];
                        double dY = uY - Y[c][l+1][i].p[0][v];
                        if(dS*dS > max_delta) max_delta = dS*dS;
                        if(dY*dY > max_delta) max_delta = dY*dY;
                        S[c][l+1][i].p[0][v] = uS;
                        Y[c][l+1][i].p[0][v] = uY;
                    }
                    norm_p_q(S[c][l+1][i].p[0]);
                    norm_p_q(Y[c][l+1][i].p[0]);
                }
            }
        }
        
        /* LEAF NODES AT L=7 */
        for(int c=0; c<2; c++) {
            memcpy(new_S[c][7], S[c][7], 256*sizeof(VarNode));
            memcpy(new_Y[c][7], Y[c][7], 256*sizeof(VarNode));
        }
        
        #pragma omp parallel for reduction(max:max_delta)
        for(int i=0; i<128; i++) {
            int v0 = 2*i, v1 = 2*i+1;
            int Z = bm_zetas[64 + i];
            
            int a00_0 = A_ntt_data[0][0][v0], a00_1 = A_ntt_data[0][0][v1];
            int a01_0 = A_ntt_data[0][1][v0], a01_1 = A_ntt_data[0][1][v1];
            int a10_0 = A_ntt_data[1][0][v0], a10_1 = A_ntt_data[1][0][v1];
            int a11_0 = A_ntt_data[1][1][v0], a11_1 = A_ntt_data[1][1][v1];
            
            double tmp_s0e[3329], tmp_s0o[3329], tmp_s1e[3329], tmp_s1o[3329];
            double tmp_y0e[3329], tmp_y0o[3329], tmp_y1e[3329], tmp_y1o[3329];
            
            // To update S0e, we receive messages from Y0 and Y1.
            // But wait, the formulas in leaf_k2 were for ONE Equation Y = A00 S0 + A01 S1!
            // Kyber-512 has TWO equations Y0 and Y1.
            // S0 receives messages from Y0 constraints AND Y1 constraints!
            // Because Y0 and Y1 are conditionally independent given S,
            // S0's incoming message is the pointwise product of message from Y0 constraint and Y1 constraint.
            
            int M[4][4] = {
                {a00_0, mod_q(Z * a00_1), a01_0, mod_q(Z * a01_1)},
                {a00_1, a00_0,            a01_1, a01_0},
                {a10_0, mod_q(Z * a10_1), a11_0, mod_q(Z * a11_1)},
                {a10_1, a10_0,            a11_1, a11_0}
            };

            double in_S[4][3329];
            memcpy(in_S[0], S[0][7][v0].p[0], 3329*sizeof(double));
            memcpy(in_S[1], S[0][7][v1].p[0], 3329*sizeof(double));
            memcpy(in_S[2], S[1][7][v0].p[0], 3329*sizeof(double));
            memcpy(in_S[3], S[1][7][v1].p[0], 3329*sizeof(double));

            double out_Y[4][3329];
            matrix_vector_bp(M, in_S, out_Y);
            memcpy(new_Y[0][7][v0].p[1], out_Y[0], 3329*sizeof(double));
            memcpy(new_Y[0][7][v1].p[1], out_Y[1], 3329*sizeof(double));
            memcpy(new_Y[1][7][v0].p[1], out_Y[2], 3329*sizeof(double));
            memcpy(new_Y[1][7][v1].p[1], out_Y[3], 3329*sizeof(double));
            
            int M_inv[4][4];
            if (invert_4x4_mod_q(M, M_inv)) {
                double in_Y[4][3329];
                memcpy(in_Y[0], Y[0][7][v0].p[0], 3329*sizeof(double));
                memcpy(in_Y[1], Y[0][7][v1].p[0], 3329*sizeof(double));
                memcpy(in_Y[2], Y[1][7][v0].p[0], 3329*sizeof(double));
                memcpy(in_Y[3], Y[1][7][v1].p[0], 3329*sizeof(double));
                
                double out_S[4][3329];
                matrix_vector_bp(M_inv, in_Y, out_S);
                memcpy(new_S[0][7][v0].p[1], out_S[0], 3329*sizeof(double));
                memcpy(new_S[0][7][v1].p[1], out_S[1], 3329*sizeof(double));
                memcpy(new_S[1][7][v0].p[1], out_S[2], 3329*sizeof(double));
                memcpy(new_S[1][7][v1].p[1], out_S[3], 3329*sizeof(double));
            } else {
                for(int v=0; v<3329; v++) {
                    new_S[0][7][v0].p[1][v] = 1.0/3329;
                    new_S[0][7][v1].p[1][v] = 1.0/3329;
                    new_S[1][7][v0].p[1][v] = 1.0/3329;
                    new_S[1][7][v1].p[1][v] = 1.0/3329;
                }
            }
        }
        
        for(int c=0; c<2; c++) {
            for(int i=0; i<256; i++) {
                for(int v=0; v<3329; v++) {
                    double uS = 0.85*new_S[c][7][i].p[1][v] + 0.15*S[c][7][i].p[1][v];
                    double uY = 0.85*new_Y[c][7][i].p[1][v] + 0.15*Y[c][7][i].p[1][v];
                    double dS = uS - S[c][7][i].p[1][v];
                    double dY = uY - Y[c][7][i].p[1][v];
                    if(dS*dS > max_delta) max_delta = dS*dS;
                    if(dY*dY > max_delta) max_delta = dY*dY;
                    S[c][7][i].p[1][v] = uS;
                    Y[c][7][i].p[1][v] = uY;
                }
                norm_p_q(S[c][7][i].p[1]);
                norm_p_q(Y[c][7][i].p[1]);
            }
        }
        
        /* BACKWARD PASS (UP THE FFT TREE L=6 to L=0) */
        for(int l=6; l>=0; l--) {
            int len = 128 >> l;
            int k_idx = (1 << l);
            
            for(int c=0; c<2; c++) {
                memcpy(new_S[c][l], S[c][l], 256*sizeof(VarNode));
                memcpy(new_Y[c][l], Y[c][l], 256*sizeof(VarNode));
            }
            
            #pragma omp parallel for reduction(max:max_delta)
            for(int obj=0; obj<128; obj++) {
                int group = obj / len;
                int start = group * 2 * len;
                int left = start + obj % len;
                int right = left + len;
                int zeta = bm_zetas[k_idx + group];
                
                for(int c=0; c<2; c++) {
                    bf_out_a(S[c][l][right].p[0], S[c][l+1][left].p[1], S[c][l+1][right].p[1], new_S[c][l][left].p[1], zeta);
                    bf_out_b(S[c][l][left].p[0], S[c][l+1][left].p[1], S[c][l+1][right].p[1], new_S[c][l][right].p[1], zeta);
                    
                    bf_out_a(Y[c][l][right].p[0], Y[c][l+1][left].p[1], Y[c][l+1][right].p[1], new_Y[c][l][left].p[1], zeta);
                    bf_out_b(Y[c][l][left].p[0], Y[c][l+1][left].p[1], Y[c][l+1][right].p[1], new_Y[c][l][right].p[1], zeta);
                }
            }
            
            for(int c=0; c<2; c++) {
                for(int i=0; i<256; i++) {
                    for(int v=0; v<3329; v++) {
                        double uS = 0.85*new_S[c][l][i].p[1][v] + 0.15*S[c][l][i].p[1][v];
                        double uY = 0.85*new_Y[c][l][i].p[1][v] + 0.15*Y[c][l][i].p[1][v];
                        double dS = uS - S[c][l][i].p[1][v];
                        double dY = uY - Y[c][l][i].p[1][v];
                        if(dS*dS > max_delta) max_delta = dS*dS;
                        if(dY*dY > max_delta) max_delta = dY*dY;
                        S[c][l][i].p[1][v] = uS;
                        Y[c][l][i].p[1][v] = uY;
                    }
                    norm_p_q(S[c][l][i].p[1]);
                    norm_p_q(Y[c][l][i].p[1]);
                }
            }
        }
        
        if(!quiet) printf("    [Polar BP] Iter %3d: max_delta = %.3e\n", it + 1, max_delta);
        if(max_delta < 1e-8) {
            if(!quiet) printf("    [Polar BP] CONVERGED (Sparse Tensor State achieved local equilibrium).\n");
            break;
        }
    }
    
    /* Decrypt & extract marginal probabilities */
    for(int c=0; c<2; c++) {
        for(int i=0; i<256; i++) {
            int g_idx = c * 256 + i;
            double sm = 0;
            double belief[3329];
            for(int v=0; v<3329; v++) {
                belief[v] = S[c][0][i].p[0][v] * S[c][0][i].p[1][v];
                sm += belief[v];
            }
            if (sm < 1e-300) sm = 1.0;
            
            int best_v = 0; double best_p = 0;
            for(int v=0; v<3329; v++) {
                double p = belief[v] / sm;
                if(p > best_p) { best_p = p; best_v = v; }
                
                // Pack into the dense contiguous 7-state marginal mapping used by decimation
                int d_idx = mod_q(v + 3); // Shift by +3 since eta=3
                if (d_idx < 7 && d_idx >= 0) {
                    marg[g_idx][d_idx] = p;
                }
            }
            int signed_v = best_v > 1664 ? best_v - 3329 : best_v;
            s_out[g_idx] = signed_v;
        }
    }
    free(S); free(Y); free(new_S); free(new_Y);
}
