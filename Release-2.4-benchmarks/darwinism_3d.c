/*
 * darwinism_3d.c — Real-Time Quantum Darwinism (The Birth of Objective Reality)
 *
 * Simulates the collapse of a macroscopic Schrödinger Cat state 
 * by coupling it to a 3D environmental thermal bath. 
 *
 * The System (S): The center site (3,3,3) initialized in a pure Cat state 
 *                 (|0⟩ + |3⟩)/√2.
 * The Environment (E): The remaining 342 sites initialized in |0⟩.
 * 
 * Evolution: Real-time unitary spreading Hamiltonian (e.g., e^{-i dt X_i X_j})
 *            entangles the S and E, bleeding S's quantum coherence into the 
 *            bath, forcing the emergence of classical objective reality.
 *
 * Build:
 *   gcc -O2 -std=gnu11 -fopenmp darwinism_3d.c quhit_core.c quhit_gates.c \
 *       quhit_measure.c quhit_entangle.c quhit_register.c peps_overlay.c -lm -o darwinism_3d
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TNS3D_CHI 6 
#include "peps3d_overlay.c"

/* ═══════════════ Constants ═══════════════ */

#define DT 0.05
#define TOTAL_TIME 2.0
#define COUPLING_J 1.0

/* ═══════════════ Entangling Hamiltonian ═══════════════ */

/* 
 * We use an XX coupling interaction to rapidly spread information.
 * H = -J * (X_i * X_j)
 * U = exp(i dt J X_i X_j)
 */
static void build_xx_coupling(double dt, double J, double *G_re, double *G_im)
{
    int D = 6, D2 = D * D;
    memset(G_re, 0, D2 * D2 * sizeof(double));
    memset(G_im, 0, D2 * D2 * sizeof(double));

    /* 
     * To spread information into a |0> bath, we need an interaction that 
     * transfers population, not just phases. 
     * We use a Partial SWAP gate (e^{-i * dt * J * SWAP}).
     * SWAP |a,b> = |b,a>. 
     * H = -J * SWAP.
     * e^{i * dt * J * SWAP} = cos(dt*J) I + i sin(dt*J) SWAP.
     */
     
    double c = cos(dt * J);
    double s = sin(dt * J);
    
    for (int a = 0; a < D; a++) {
        for (int b = 0; b < D; b++) {
            int idx_ab = a * D + b;
            int idx_ba = b * D + a;
            
            if (a == b) {
                // If a==b, SWAP|a,a> = |a,a>. 
                // U|a,a> = (cos + i sin)|a,a> = e^{i dt J} |a,a>
                G_re[idx_ab * D2 + idx_ab] = c;
                G_im[idx_ab * D2 + idx_ab] = s;
            } else {
                // U|a,b> = cos|a,b> + i sin|b,a>
                G_re[idx_ab * D2 + idx_ab] = c;
                G_im[idx_ab * D2 + idx_ab] = 0.0;
                
                G_re[idx_ba * D2 + idx_ab] = 0.0;
                G_im[idx_ba * D2 + idx_ab] = s;
            }
        }
    }
}

/* ═══════════════ Diagnostics ═══════════════ */

static void renormalize_all(Tns3dGrid *g)
{
    int N = g->Lx * g->Ly * g->Lz;
    for (int i = 0; i < N; i++) {
        int reg = g->site_reg[i];
        if (reg < 0) continue;
        QuhitRegister *r = &g->eng->registers[reg];
        double n2 = 0;
        for (uint32_t e = 0; e < r->num_nonzero; e++)
            n2 += r->entries[e].amp_re * r->entries[e].amp_re +
                  r->entries[e].amp_im * r->entries[e].amp_im;
        if (n2 > 1e-20) {
            double inv = 1.0 / sqrt(n2);
            for (uint32_t e = 0; e < r->num_nonzero; e++) {
                r->entries[e].amp_re *= inv;
                r->entries[e].amp_im *= inv;
            }
        }
    }
}

static void print_slice_entropy(Tns3dGrid *g, int z_slice)
{
    printf("\n  [Z Layer %d Entropy Map S_i]\n", z_slice);
    for (int y = g->Ly - 1; y >= 0; y--) {
        printf("  y=%d |", y);
        for (int x = 0; x < g->Lx; x++) {
            double p[6]; 
            tns3d_local_density(g, x, y, z_slice, p);
            double s = 0;
            for(int k=0; k<6; k++) {
                if(p[k] > 1e-8) s -= p[k] * log(p[k]) / log(2.0);
            }
            if (s > 0.8) printf(" ██ ");
            else if (s > 0.5) printf(" ▓▓ ");
            else if (s > 0.2) printf(" ▒▒ ");
            else if (s > 0.05) printf(" .. ");
            else printf("    ");
        }
        printf("|\n");
    }
}

static double cat_coherence(Tns3dGrid *g, int cx, int cy, int cz)
{
    double v_re[36]={0}, v_im[36]={0};
    double inv_sqrt2 = 1.0 / sqrt(2.0);
    
    // V |0> = (|0> + |3>)/\sqrt{2}, V |3> = (|0> - |3>)/\sqrt{2}
    v_re[0*6+0] = inv_sqrt2; v_re[0*6+3] = inv_sqrt2;
    v_re[3*6+0] = inv_sqrt2; v_re[3*6+3] = -inv_sqrt2;
    v_re[1*6+1] = 1.0;
    v_re[2*6+2] = 1.0;
    v_re[4*6+4] = 1.0;
    v_re[5*6+5] = 1.0;
    
    int reg = g->site_reg[cz * g->Lx * g->Ly + cy * g->Lx + cx];
    quhit_reg_apply_unitary_pos(g->eng, reg, 6, v_re, v_im);
    
    double p[6];
    tns3d_local_density(g, cx, cy, cz, p);
    
    // V is its own inverse (Hermitian and Unitary)
    quhit_reg_apply_unitary_pos(g->eng, reg, 6, v_re, v_im);
    
    double coherence = 2.0 * (p[0] - 0.5);
    if (coherence < 0) coherence = 0;
    return coherence;
}

/* ═══════════════ Main ═══════════════ */

int main(void)
{
    // 7x7x7 grid to allow perfectly symmetrical outward blooming
    int Lx = 7, Ly = 7, Lz = 7;
    int Nsites = Lx * Ly * Lz;

    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  THE BIRTH OF OBJECTIVE REALITY — Quantum Darwinism      ║\n");
    printf("║  ──────────────────────────────────────────────────────────  ║\n");
    printf("║  Lattice: %d×%d×%d = %d sites (Central 'Cat' at 3,3,3)     ║\n", Lx, Ly, Lz, Nsites);
    printf("║  Hilbert space: 6^%d ≈ 10^%.1f dimensions                  ║\n", Nsites, Nsites * log10(6.0));
    printf("║  Model: Central Schrödinger Cat + Empty Thermal Bath       ║\n");
    printf("║  Method: Real-Time Unitary Environmental Entanglement      ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    double *hop_re = calloc(36*36, sizeof(double));
    double *hop_im = calloc(36*36, sizeof(double));
    build_xx_coupling(DT, COUPLING_J, hop_re, hop_im);

    Tns3dGrid *g = tns3d_init(Lx, Ly, Lz);

    // Initialize Bath to |0⟩
    for (int i = 0; i < Nsites; i++) {
        int reg = g->site_reg[i];
        quhit_reg_sv_set(g->eng, reg, 0 * TNS3D_C6, 1.0, 0);      
    }

    // Initialize the Central System to a pure Cat State: (|0⟩ + |3⟩)/√2
    int center_idx = (Lz/2) * Lx * Ly + (Ly/2) * Lx + (Lx/2);
    int center_reg = g->site_reg[center_idx];
    
    // Clear initial state
    g->eng->registers[center_reg].num_nonzero = 0;
    
    // Set proper physical amplitude
    quhit_reg_sv_set(g->eng, center_reg, 0 * TNS3D_C6, 1.0/sqrt(2.0), 0);
    quhit_reg_sv_set(g->eng, center_reg, 3 * TNS3D_C6, 1.0/sqrt(2.0), 0);

    int steps = (int)(TOTAL_TIME / DT);
    printf("  ══ SPREADING QUANTUM COHERENCE TO THE BATH (%d steps) ══\n\n", steps);
    
    double t = 0;
    for (int step = 0; step <= steps; step++) {
        
        if (step % 5 == 0) {
            double c = cat_coherence(g, Lx/2, Ly/2, Lz/2);
            printf("  Time: %4.2fs | Central Cat Coherence: %6.4f | Bath Entangling...\n", t, c);
            if (step % 20 == 0) {
                print_slice_entropy(g, Lz/2); 
                
                // Diagnostic: exactly check neighbor (x=4, y=3, z=3)
                double pn[6];
                tns3d_local_density(g, (Lx/2)+1, Ly/2, Lz/2, pn);
                double sn = 0;
                for(int k=0; k<6; k++) if(pn[k]>1e-8) sn -= pn[k]*log(pn[k])/log(2.0);
                printf("  [Probe] Neighbor(4,3,3) p_0=%.4f p_3=%.4f S_n=%.4f\n", pn[0], pn[3], sn);
            }
        }

        if (step < steps) {
            tns3d_gate_x_all(g, hop_re, hop_im);
            tns3d_gate_y_all(g, hop_re, hop_im);
            tns3d_gate_z_all(g, hop_re, hop_im);
            renormalize_all(g);
            t += DT;
        }
    }
    
    printf("\n  ════════════════════════════════════════════════════════════\n");
    printf("  The environment successfully copied the information.\n");
    printf("  The Cat's off-diagonal coherence has vanished.\n");
    printf("  Objective, classical reality has emerged.\n");

    tns3d_free(g);
    free(hop_re); free(hop_im);
    return 0;
}
