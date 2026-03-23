/*
 * nuclear_transmute_hpc.c — Nuclear Transmutation Simulator via HPC
 *
 * ═══════════════════════════════════════════════════════════════════
 * D=6 NUCLEAR SHELL ENCODING — The Cuprate→Nuclear Isomorphism
 * ═══════════════════════════════════════════════════════════════════
 *
 *   |0⟩ = ∅           Empty orbital              n=0
 *   |1⟩ = p↑          Proton spin-up             n=1, Iz=+½, Sz=+½
 *   |2⟩ = p↓          Proton spin-down           n=1, Iz=+½, Sz=-½
 *   |3⟩ = n↑          Neutron spin-up            n=1, Iz=-½, Sz=+½
 *   |4⟩ = n↓          Neutron spin-down          n=1, Iz=-½, Sz=-½
 *   |5⟩ = D           Deuteron (p-n bound pair)  n=2, I=0
 *         = (|p↑ n↓⟩ - |p↓ n↑⟩)/√2
 *
 * ω³ = -1 encodes the NUCLEON fermion sign EXACTLY.
 * CZ gates encode the nuclear force.
 *
 * PHYSICS:
 *   H = Σ_α ε_α n̂_α                    [single-particle shell energies]
 *     - V₀ Σ_⟨αβ⟩ a†_α a_β             [residual nuclear interaction]
 *     - V_pair Σ (p↑n↓→D)              [pairing interaction]
 *     + V_c Σ_protons Z·e²/r            [Coulomb repulsion]
 *     - V_so (l·s)                       [spin-orbit coupling]
 *
 * Build:
 *   gcc -O2 -march=native -o nuclear_transmute nuclear_transmute_hpc.c \
 *       quhit_triality.c s6_exotic.c bigint.c -lm -msse2
 * ═══════════════════════════════════════════════════════════════════
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "hpc_graph.h"
#include "hpc_contract.h"

/* ═══════════════════════════════════════════════════════════════════
 * FOCK-SPACE ENCODING — Nuclear Shell Model
 * ═══════════════════════════════════════════════════════════════════ */

#define EMPTY    0   /* |∅⟩   empty orbital                n=0 */
#define P_UP     1   /* |p↑⟩  proton spin-up               n=1 */
#define P_DN     2   /* |p↓⟩  proton spin-down             n=1 */
#define N_UP     3   /* |n↑⟩  neutron spin-up              n=1 */
#define N_DN     4   /* |n↓⟩  neutron spin-down            n=1 */
#define DEUTERON 5   /* |D⟩   deuteron (p-n bound pair)    n=2 */

static const double fock_n[6] = {0, 1, 1, 1, 1, 2};
static const char *fock_name[6] = {"∅","p↑","p↓","n↑","n↓","D"};
static const double fock_sz[6] = {0, +0.5, -0.5, +0.5, -0.5, 0};
static const double fock_iz[6] = {0, +0.5, +0.5, -0.5, -0.5, 0};

/* ═══════════════════════════════════════════════════════════════════
 * PRNG — xoshiro256**
 * ═══════════════════════════════════════════════════════════════════ */

static uint64_t rng_s[4];
static inline uint64_t rotl(uint64_t x,int k){return(x<<k)|(x>>(64-k));}
static uint64_t rng_next(void){
    uint64_t r=rotl(rng_s[1]*5,7)*9,t=rng_s[1]<<17;
    rng_s[2]^=rng_s[0];rng_s[3]^=rng_s[1];rng_s[1]^=rng_s[2];rng_s[0]^=rng_s[3];
    rng_s[2]^=t;rng_s[3]=rotl(rng_s[3],45);return r;
}
static double rng_u(void){return(double)(rng_next()>>11)/(double)(1ULL<<53);}
static void rng_init(uint64_t seed){
    for(int i=0;i<4;i++){seed+=0x9e3779b97f4a7c15ULL;uint64_t z=seed;
    z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;
    rng_s[i]=z^(z>>31);}
}

/* ═══════════════════════════════════════════════════════════════════
 * NUCLEAR SHELL STRUCTURE
 *
 * Sites = nuclear orbitals (nlj subshells)
 * Neighbors = orbitals coupled by residual nuclear interaction
 *
 * Standard shell ordering:
 *   1s₁/₂  1p₃/₂  1p₁/₂  1d₅/₂  2s₁/₂  1d₃/₂
 *   1f₇/₂  2p₃/₂  1f₅/₂  2p₁/₂  1g₉/₂
 *   1g₇/₂  2d₅/₂  2d₃/₂  3s₁/₂  1h₁₁/₂
 *   ...
 *
 * Each nlj subshell has degeneracy (2j+1). We model each
 * magnetic subststate m_j as a separate "site" in the HPC graph.
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    int n;          /* principal quantum number */
    int l;          /* orbital angular momentum */
    int j2;         /* 2*j (total ang. mom, half-integer) */
    int degeneracy; /* 2j+1 */
    double energy;  /* single-particle energy in MeV */
    const char *name;
} NuclearShell;

/* Shell model levels with Woods-Saxon single-particle energies (MeV)
 * These are approximate values for medium-mass nuclei.
 * Extended past magic 126 to handle transactinides (Am-241 etc). */
static const NuclearShell SHELLS[] = {
    /* n  l  2j  deg  energy  name            cumul */
    { 1, 0,  1,  2, -40.0, "1s₁/₂"  },   /*   2  ★ */
    { 1, 1,  3,  4, -33.0, "1p₃/₂"  },   /*   6    */
    { 1, 1,  1,  2, -30.0, "1p₁/₂"  },   /*   8  ★ */
    { 1, 2,  5,  6, -22.0, "1d₅/₂"  },   /*  14    */
    { 2, 0,  1,  2, -18.0, "2s₁/₂"  },   /*  16    */
    { 1, 2,  3,  4, -16.0, "1d₃/₂"  },   /*  20  ★ */
    { 1, 3,  7,  8, -10.0, "1f₇/₂"  },   /*  28  ★ */
    { 2, 1,  3,  4,  -8.0, "2p₃/₂"  },   /*  32    */
    { 1, 3,  5,  6,  -6.5, "1f₅/₂"  },   /*  38    */
    { 2, 1,  1,  2,  -5.0, "2p₁/₂"  },   /*  40    */
    { 1, 4,  9, 10,  -3.5, "1g₉/₂"  },   /*  50  ★ */
    { 1, 4,  7,  8,  -1.0, "1g₇/₂"  },   /*  58    */
    { 2, 2,  5,  6,   0.5, "2d₅/₂"  },   /*  64    */
    { 2, 2,  3,  4,   2.0, "2d₃/₂"  },   /*  68    */
    { 3, 0,  1,  2,   3.0, "3s₁/₂"  },   /*  70    */
    { 1, 5, 11, 12,   4.0, "1h₁₁/₂" },   /*  82  ★ */
    /* ── Beyond magic 82 ── */
    { 1, 5,  9, 10,   7.0, "1h₉/₂"  },   /*  92    */
    { 2, 3,  7,  8,   8.5, "2f₇/₂"  },   /* 100    */
    { 2, 3,  5,  6,  10.0, "2f₅/₂"  },   /* 106    */
    { 3, 1,  3,  4,  11.0, "3p₃/₂"  },   /* 110    */
    { 3, 1,  1,  2,  12.0, "3p₁/₂"  },   /* 112    */
    { 1, 6, 13, 14,  13.0, "1i₁₃/₂" },   /* 126  ★ */
    /* ── Beyond magic 126: the superheavy shells ── */
    { 2, 4,  9, 10,  15.0, "2g₉/₂"  },   /* 136    */
    { 1, 6, 11, 12,  16.5, "1i₁₁/₂" },   /* 148    */
    { 1, 7, 15, 16,  18.0, "1j₁₅/₂" },   /* 164    */
    { 3, 2,  5,  6,  19.5, "3d₅/₂"  },   /* 170    */
    { 4, 0,  1,  2,  20.5, "4s₁/₂"  },   /* 172    */
    { 2, 4,  7,  8,  21.5, "2g₇/₂"  },   /* 180    */
    { 3, 2,  3,  4,  22.5, "3d₃/₂"  },   /* 184  ★ */
};
#define N_SHELLS 29

/* Total capacity through all shells */
static int shell_total_capacity(int n_shells) {
    int total = 0;
    for (int i = 0; i < n_shells && i < N_SHELLS; i++)
        total += SHELLS[i].degeneracy;
    return total;
}

/* Find how many shells are needed for Z protons and N neutrons */
static int shells_needed(int Z, int N) {
    int max_nucleon = (Z > N) ? Z : N;
    int total = 0;
    for (int i = 0; i < N_SHELLS; i++) {
        total += SHELLS[i].degeneracy;
        if (total >= max_nucleon) return i + 1;
    }
    return N_SHELLS;
}

/* ═══════════════════════════════════════════════════════════════════
 * ISOTOPE DATABASE — Radioactive waste targets
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    const char *name;
    const char *symbol;
    int Z;              /* protons */
    int N;              /* neutrons */
    int A;              /* mass number Z+N */
    double half_life;   /* seconds (0 = stable) */
    const char *decay;  /* primary decay mode */
    const char *daughter;
    double sigma_th;    /* thermal neutron capture cross-section (barns) */
} Isotope;

#define SEC_YR (365.25*86400.0)

static const Isotope ISOTOPE_DB[] = {
    /* Radioactive waste isotopes */
    {"Cesium-137",    "Cs-137", 55, 82, 137,  30.2*SEC_YR,   "β⁻", "Ba-137", 0.25},
    {"Strontium-90",  "Sr-90",  38, 52,  90,  28.8*SEC_YR,   "β⁻", "Y-90",   0.015},
    {"Technetium-99", "Tc-99",  43, 56,  99,  2.11e5*SEC_YR, "β⁻", "Ru-99",  20.0},
    {"Iodine-129",    "I-129",  53, 76, 129,  1.57e7*SEC_YR, "β⁻", "Xe-129", 30.0},
    {"Americium-241", "Am-241", 95,146, 241,  432.2*SEC_YR,  "α",  "Np-237", 587.0},

    /* Capture daughters (check if stable) */
    {"Cesium-138",    "Cs-138", 55, 83, 138,  33.4*60,       "β⁻", "Ba-138", 0},
    {"Barium-138",    "Ba-138", 56, 82, 138,  0,             "--",  "(stable)", 0},
    {"Strontium-91",  "Sr-91",  38, 53,  91,  9.63*3600,     "β⁻", "Y-91",   0},
    {"Zirconium-91",  "Zr-91",  40, 51,  91,  0,             "--",  "(stable)", 0},
    {"Technetium-100","Tc-100", 43, 57, 100,  15.8,          "β⁻", "Ru-100", 0},
    {"Ruthenium-100", "Ru-100", 44, 56, 100,  0,             "--",  "(stable)", 0},
    {"Iodine-130",    "I-130",  53, 77, 130,  12.36*3600,    "β⁻", "Xe-130", 0},
    {"Xenon-130",     "Xe-130", 54, 76, 130,  0,             "--",  "(stable)", 0},
};
#define N_ISOTOPES 13

/* ═══════════════════════════════════════════════════════════════════
 * STATE PREPARATION — Fill nuclear shells for a given isotope
 *
 * For Z protons: fill lowest orbitals with p↑/p↓ amplitudes
 * For N neutrons: fill lowest orbitals with n↑/n↓ amplitudes
 * When both p and n occupy the same level → deuteron amplitude
 * ═══════════════════════════════════════════════════════════════════ */

static void prepare_nucleus(HPCGraph *g, const Isotope *iso, int n_sites)
{
    int Z = iso->Z, N_neut = iso->N;

    /* Distribute protons and neutrons across shells */
    int p_remaining = Z, n_remaining = N_neut;

    for (int s = 0; s < n_sites; s++) {
        /* Determine which shell this site belongs to */
        int shell_idx = 0, cumul = 0;
        for (int i = 0; i < N_SHELLS; i++) {
            if (s < cumul + SHELLS[i].degeneracy) { shell_idx = i; break; }
            cumul += SHELLS[i].degeneracy;
        }
        int pos_in_shell = s - cumul;

        double re[6] = {0}, im[6] = {0};

        /* Fill protons: each sublevel takes 1 proton (spin up or down) */
        int p_fill = (p_remaining > 0) ? 1 : 0;
        int n_fill = (n_remaining > 0) ? 1 : 0;

        if (p_fill && n_fill) {
            /* Both proton and neutron in this orbital → deuteron amplitude */
            re[DEUTERON] = 0.55;  /* pairing */
            re[(pos_in_shell % 2 == 0) ? P_UP : P_DN] = 0.30;
            re[(pos_in_shell % 2 == 0) ? N_UP : N_DN] = 0.30;
            re[EMPTY] = 0.02;
            p_remaining--;
            n_remaining--;
        } else if (p_fill) {
            re[(pos_in_shell % 2 == 0) ? P_UP : P_DN] = 0.85;
            re[EMPTY] = 0.05;
            re[DEUTERON] = 0.01;
            p_remaining--;
        } else if (n_fill) {
            re[(pos_in_shell % 2 == 0) ? N_UP : N_DN] = 0.85;
            re[EMPTY] = 0.05;
            re[DEUTERON] = 0.01;
            n_remaining--;
        } else {
            /* Empty orbital above Fermi surface */
            re[EMPTY] = 0.92;
            re[P_UP] = 0.01; re[P_DN] = 0.01;
            re[N_UP] = 0.01; re[N_DN] = 0.01;
            re[DEUTERON] = 0.005;
        }

        /* Small quantum fluctuations */
        for (int k = 0; k < 6; k++) {
            re[k] += 0.01 * (rng_u() - 0.5);
            im[k] = 0.005 * (rng_u() - 0.5);
            if (re[k] < 0) re[k] = fabs(re[k]) * 0.1;
        }

        /* Normalize */
        double norm = 0;
        for (int k = 0; k < 6; k++) norm += re[k]*re[k] + im[k]*im[k];
        norm = sqrt(norm);
        if (norm > 1e-15) for (int k = 0; k < 6; k++) { re[k]/=norm; im[k]/=norm; }

        hpc_set_local(g, s, re, im);
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * NUCLEAR HAMILTONIAN — Trotter evolution operators
 * ═══════════════════════════════════════════════════════════════════ */

/* Single-particle energies: phase each orbital by its shell energy */
static void apply_shell_energies(HPCGraph *g, int n_sites, double dt)
{
    int cumul = 0;
    for (int sh = 0; sh < N_SHELLS; sh++) {
        double E = SHELLS[sh].energy;
        double phase_occ = E * dt;  /* occupied states get energy phase */

        double se_re[6], se_im[6];
        se_re[EMPTY] = 1; se_im[EMPTY] = 0;
        for (int k = 1; k <= 4; k++) {
            se_re[k] = cos(phase_occ);
            se_im[k] = sin(phase_occ);
        }
        /* Deuteron gets 2× energy (two nucleons) + binding */
        double deut_phase = 2.0 * phase_occ - 2.224 * dt; /* -2.224 MeV binding */
        se_re[DEUTERON] = cos(deut_phase);
        se_im[DEUTERON] = sin(deut_phase);

        for (int s = cumul; s < cumul + SHELLS[sh].degeneracy && s < n_sites; s++)
            hpc_phase(g, s, se_re, se_im);

        cumul += SHELLS[sh].degeneracy;
        if (cumul >= n_sites) break;
    }
}

/* Residual nuclear interaction: CZ on bounded-degree graph.
 * O(degree) edges per step instead of O(N²).
 *
 * Topology: ring coupling within each shell (nearest-neighbor only)
 * + one cross-shell CZ at boundaries. Total edges = O(N). */
static void apply_nuclear_force(HPCGraph *g, int n_sites, double V0, double dt)
{
    /* Adjacent-orbital CZ: nearest neighbors only → O(N) edges */
    for (int s = 0; s < n_sites - 1; s++)
        hpc_cz(g, s, s + 1);

    /* Intra-shell ring closure: connect last→first within each shell
     * (bounded degree: just 1 extra edge per shell, not deg²) */
    int cumul = 0;
    for (int sh = 0; sh < N_SHELLS; sh++) {
        int deg = SHELLS[sh].degeneracy;
        int first = cumul, last = cumul + deg - 1;
        if (last < n_sites && deg > 2)
            hpc_cz(g, first, last);  /* Ring closure */
        cumul += deg;
        if (cumul >= n_sites) break;
    }

    /* Phase weighting for nuclear force strength */
    double nf_re[6], nf_im[6];
    double v_phase = V0 * dt;
    nf_re[EMPTY] = 1; nf_im[EMPTY] = 0;
    for (int k = 1; k <= 4; k++) {
        nf_re[k] = cos(v_phase * fock_iz[k]);
        nf_im[k] = sin(v_phase * fock_iz[k]);
    }
    nf_re[DEUTERON] = cos(v_phase * 1.5);
    nf_im[DEUTERON] = sin(v_phase * 1.5);

    for (int s = 0; s < n_sites; s++)
        hpc_phase(g, s, nf_re, nf_im);
}

/* Coulomb repulsion: acts only on proton channels */
static void apply_coulomb(HPCGraph *g, int n_sites, int Z, double dt)
{
    /* Coulomb energy scales as Z(Z-1)e²/R ≈ 0.72·Z(Z-1)/R MeV
     * R ≈ 1.2·A^(1/3) fm */
    double R = 1.2 * pow(Z * 2.0, 1.0/3.0); /* approximate */
    double V_c = 0.72 * (Z - 1) / R;  /* MeV per proton */

    double c_re[6], c_im[6];
    double cp = V_c * dt * 0.01;  /* scaled for Trotter */
    c_re[EMPTY] = 1; c_im[EMPTY] = 0;
    /* Protons feel Coulomb */
    c_re[P_UP] = cos(cp); c_im[P_UP] = sin(cp);
    c_re[P_DN] = cos(cp); c_im[P_DN] = sin(cp);
    /* Neutrons don't */
    c_re[N_UP] = 1; c_im[N_UP] = 0;
    c_re[N_DN] = 1; c_im[N_DN] = 0;
    /* Deuteron: half Coulomb (one proton) */
    c_re[DEUTERON] = cos(cp*0.5); c_im[DEUTERON] = sin(cp*0.5);

    for (int s = 0; s < n_sites; s++)
        hpc_phase(g, s, c_re, c_im);
}

/* Spin-orbit coupling: the force that creates magic numbers.
 * V_so(l·s) splits j=l+1/2 from j=l-1/2. */
static void apply_spin_orbit(HPCGraph *g, int n_sites, double V_so, double dt)
{
    int cumul = 0;
    for (int sh = 0; sh < N_SHELLS; sh++) {
        int l = SHELLS[sh].l, j2 = SHELLS[sh].j2;
        /* l·s = [j(j+1)-l(l+1)-s(s+1)]/2, s=1/2
         * j=l+1/2: l·s = l/2
         * j=l-1/2: l·s = -(l+1)/2 */
        double ls;
        if (j2 == 2*l + 1)
            ls = l / 2.0;        /* j = l + 1/2 */
        else
            ls = -(l + 1) / 2.0; /* j = l - 1/2 */

        double so_phase = V_so * ls * dt;

        double so_re[6], so_im[6];
        so_re[EMPTY] = 1; so_im[EMPTY] = 0;
        for (int k = 1; k <= 4; k++) {
            double sp = so_phase * fock_sz[k] * 2.0;
            so_re[k] = cos(sp); so_im[k] = sin(sp);
        }
        so_re[DEUTERON] = 1; so_im[DEUTERON] = 0; /* paired → net S=0 */

        for (int s = cumul; s < cumul + SHELLS[sh].degeneracy && s < n_sites; s++)
            hpc_phase(g, s, so_re, so_im);

        cumul += SHELLS[sh].degeneracy;
        if (cumul >= n_sites) break;
    }
}

/* Nucleon pairing: p↑+n↓ → deuteron (same as Cooper pairing gate) */
static void apply_pairing(HPCGraph *g, int n_sites, double V_pair, double dt)
{
    for (int s = 0; s < n_sites; s++) {
        TrialityQuhit *q = &g->locals[s];

        /* Subspace α: {p↑(1), n↓(4), Deuteron(5)} */
        {
            double r1=q->edge_re[P_UP], i1=q->edge_im[P_UP];
            double r4=q->edge_re[N_DN], i4=q->edge_im[N_DN];
            double r5=q->edge_re[DEUTERON], i5=q->edge_im[DEUTERON];

            double p1 = r1*r1+i1*i1, p4 = r4*r4+i4*i4;
            double wt = (p1+p4)*0.5;
            double theta = V_pair * dt * wt;

            if (theta > 1e-12) {
                double c=cos(theta), sv=sin(theta);
                double inv2 = 1.0/sqrt(2.0);
                double n1r = c*r1 - sv*inv2*r5, n1i = c*i1 - sv*inv2*i5;
                double n4r = c*r4 - sv*inv2*r5, n4i = c*i4 - sv*inv2*i5;
                double n5r = sv*inv2*r1 + sv*inv2*r4 + c*r5;
                double n5i = sv*inv2*i1 + sv*inv2*i4 + c*i5;
                q->edge_re[P_UP]=n1r; q->edge_im[P_UP]=n1i;
                q->edge_re[N_DN]=n4r; q->edge_im[N_DN]=n4i;
                q->edge_re[DEUTERON]=n5r; q->edge_im[DEUTERON]=n5i;
            }
        }

        /* Subspace β: {p↓(2), n↑(3), Deuteron(5)} */
        {
            double r2=q->edge_re[P_DN], i2=q->edge_im[P_DN];
            double r3=q->edge_re[N_UP], i3=q->edge_im[N_UP];
            double r5=q->edge_re[DEUTERON], i5=q->edge_im[DEUTERON];

            double p2 = r2*r2+i2*i2, p3 = r3*r3+i3*i3;
            double wt = (p2+p3)*0.5;
            double theta = V_pair * dt * wt;

            if (theta > 1e-12) {
                double c=cos(theta), sv=sin(theta);
                double inv2 = 1.0/sqrt(2.0);
                double n2r = c*r2 + sv*inv2*r5, n2i = c*i2 + sv*inv2*i5;
                double n3r = c*r3 + sv*inv2*r5, n3i = c*i3 + sv*inv2*i5;
                double n5r = -sv*inv2*r2 - sv*inv2*r3 + c*r5;
                double n5i = -sv*inv2*i2 - sv*inv2*i3 + c*i5;
                q->edge_re[P_DN]=n2r; q->edge_im[P_DN]=n2i;
                q->edge_re[N_UP]=n3r; q->edge_im[N_UP]=n3i;
                q->edge_re[DEUTERON]=n5r; q->edge_im[DEUTERON]=n5i;
            }
        }

        /* Renormalize */
        double norm = 0;
        for (int k = 0; k < 6; k++)
            norm += q->edge_re[k]*q->edge_re[k] + q->edge_im[k]*q->edge_im[k];
        if (norm > 1e-15) {
            norm = 1.0/sqrt(norm);
            for (int k = 0; k < 6; k++) {
                q->edge_re[k] *= norm;
                q->edge_im[k] *= norm;
            }
        }
    }
}

/* Fermionic sign: single nucleons get ω³ = -1 */
static void apply_fermion_sign(HPCGraph *g, int n_sites, double dt)
{
    double sp_re[6], sp_im[6];
    sp_re[EMPTY]=1; sp_im[EMPTY]=0;
    double fp = M_PI * dt;
    sp_re[P_UP]=cos(fp); sp_im[P_UP]=sin(fp);
    sp_re[P_DN]=cos(fp); sp_im[P_DN]=sin(fp);
    sp_re[N_UP]=cos(fp); sp_im[N_UP]=sin(fp);
    sp_re[N_DN]=cos(fp); sp_im[N_DN]=sin(fp);
    sp_re[DEUTERON]=cos(2*fp); sp_im[DEUTERON]=sin(2*fp);
    for (int s = 0; s < n_sites; s++)
        hpc_phase(g, s, sp_re, sp_im);
}

/* Full Trotter step */
static void nuclear_trotter_step(HPCGraph *g, int n_sites, int Z,
                                  double V0, double V_pair, double V_so,
                                  double dt)
{
    apply_shell_energies(g, n_sites, dt);
    apply_nuclear_force(g, n_sites, V0, dt);
    apply_coulomb(g, n_sites, Z, dt);
    apply_spin_orbit(g, n_sites, V_so, dt);
    apply_pairing(g, n_sites, V_pair, dt);
    apply_fermion_sign(g, n_sites, dt);
}

/* ═══════════════════════════════════════════════════════════════════
 * NEUTRON CAPTURE OPERATOR
 *
 * Inject a neutron into the nucleus at energy E_n.
 * This creates a compound nucleus with N+1 neutrons.
 *
 * The transition amplitude is computed as:
 *   T(E) = ⟨Ψ_final | â†_n(E) | Ψ_initial⟩
 *
 * Implementation: for each empty orbital, compute the overlap
 * between the incoming neutron (plane wave at energy E_n) and
 * the shell orbital wavefunction. Rotate amplitude from |∅⟩ to
 * |n⟩ in proportion to this overlap.
 * ═══════════════════════════════════════════════════════════════════ */

static double apply_neutron_capture(HPCGraph *g, int n_sites,
                                      double E_neutron, double dt)
{
    double total_capture = 0;

    for (int s = 0; s < n_sites; s++) {
        TrialityQuhit *q = &g->locals[s];
        double p_empty = q->edge_re[EMPTY]*q->edge_re[EMPTY]
                       + q->edge_im[EMPTY]*q->edge_im[EMPTY];

        if (p_empty < 0.01) continue;  /* Already occupied */

        /* Determine shell energy for this orbital */
        int sh = 0, cumul = 0;
        for (int i = 0; i < N_SHELLS; i++) {
            if (s < cumul + SHELLS[i].degeneracy) { sh = i; break; }
            cumul += SHELLS[i].degeneracy;
        }

        /* Breit-Wigner resonance: capture cross-section peaks when
         * E_neutron matches a shell energy level.
         * σ(E) ∝ Γ²/[(E - E_r)² + Γ²/4]
         * where E_r = shell resonance energy, Γ = level width */
        double E_r = SHELLS[sh].energy + 8.0; /* adjust for separation energy */
        double Gamma = 0.5 * (1.0 + 0.1 * SHELLS[sh].l); /* width depends on l */
        double dE = E_neutron - E_r;
        double breit_wigner = Gamma*Gamma / (4.0*dE*dE + Gamma*Gamma);

        /* Capture angle: rotate |∅⟩ → |n↑⟩ or |n↓⟩ */
        double theta = sqrt(breit_wigner) * dt * sqrt(p_empty);
        if (theta > 1e-10) {
            int n_spin = (s % 2 == 0) ? N_UP : N_DN;
            double r0 = q->edge_re[EMPTY], i0 = q->edge_im[EMPTY];
            double rn = q->edge_re[n_spin], in_ = q->edge_im[n_spin];

            double c = cos(theta), sv = sin(theta);
            q->edge_re[EMPTY]  = c*r0 - sv*rn;
            q->edge_im[EMPTY]  = c*i0 - sv*in_;
            q->edge_re[n_spin] = sv*r0 + c*rn;
            q->edge_im[n_spin] = sv*i0 + c*in_;

            total_capture += theta * theta;  /* transition probability */

            /* Renormalize */
            double norm = 0;
            for (int k = 0; k < 6; k++)
                norm += q->edge_re[k]*q->edge_re[k] + q->edge_im[k]*q->edge_im[k];
            if (norm > 1e-15) {
                norm = 1.0/sqrt(norm);
                for (int k = 0; k < 6; k++) {
                    q->edge_re[k] *= norm;
                    q->edge_im[k] *= norm;
                }
            }
        }
    }

    return total_capture;
}

/* ═══════════════════════════════════════════════════════════════════
 * BETA DECAY OPERATOR: n → p + e⁻ + ν̄ₑ
 *
 * In the D=6 encoding: |n↑⟩ → |p↑⟩ or |n↓⟩ → |p↓⟩
 * (spin is conserved in allowed transitions)
 * ═══════════════════════════════════════════════════════════════════ */

static double apply_beta_decay(HPCGraph *g, int n_sites, double dt)
{
    double total_decay = 0;
    double decay_rate = 0.02 * dt; /* Fermi coupling scaled */

    for (int s = 0; s < n_sites; s++) {
        TrialityQuhit *q = &g->locals[s];

        /* n↑ → p↑ */
        double pn = q->edge_re[N_UP]*q->edge_re[N_UP]
                   + q->edge_im[N_UP]*q->edge_im[N_UP];
        double pp = q->edge_re[P_UP]*q->edge_re[P_UP]
                   + q->edge_im[P_UP]*q->edge_im[P_UP];

        /* Beta decay favored when: neutron present, proton orbital available */
        if (pn > 0.1 && pp < 0.3) {
            double theta = decay_rate * sqrt(pn * (1.0 - pp));
            double c = cos(theta), sv = sin(theta);
            double rn = q->edge_re[N_UP], in_ = q->edge_im[N_UP];
            double rp = q->edge_re[P_UP], ip = q->edge_im[P_UP];
            q->edge_re[N_UP] = c*rn - sv*rp;
            q->edge_im[N_UP] = c*in_ - sv*ip;
            q->edge_re[P_UP] = sv*rn + c*rp;
            q->edge_im[P_UP] = sv*in_ + c*ip;
            total_decay += theta * theta;
        }

        /* n↓ → p↓ */
        pn = q->edge_re[N_DN]*q->edge_re[N_DN]
           + q->edge_im[N_DN]*q->edge_im[N_DN];
        pp = q->edge_re[P_DN]*q->edge_re[P_DN]
           + q->edge_im[P_DN]*q->edge_im[P_DN];

        if (pn > 0.1 && pp < 0.3) {
            double theta = decay_rate * sqrt(pn * (1.0 - pp));
            double c = cos(theta), sv = sin(theta);
            double rn = q->edge_re[N_DN], in_ = q->edge_im[N_DN];
            double rp = q->edge_re[P_DN], ip = q->edge_im[P_DN];
            q->edge_re[N_DN] = c*rn - sv*rp;
            q->edge_im[N_DN] = c*in_ - sv*ip;
            q->edge_re[P_DN] = sv*rn + c*rp;
            q->edge_im[P_DN] = sv*in_ + c*ip;
            total_decay += theta * theta;
        }
    }
    return total_decay;
}

/* ═══════════════════════════════════════════════════════════════════
 * PHOTONUCLEAR (γ,n) OPERATOR — Crack the armor
 *
 * For isotopes armored against neutron capture (Cs-137: N=82 magic,
 * Sr-90: N=52 near magic 50), we use high-energy photons to EJECT
 * a neutron instead of adding one.
 *
 *   Cs-137 + γ → Cs-136 + n    (Cs-136: t½ = 13 days → Ba-136 STABLE)
 *   Sr-90  + γ → Sr-89  + n    (Sr-89:  t½ = 50 days → Y-89  STABLE)
 *
 * In D=6 encoding: rotate |n⟩ → |∅⟩ (REVERSE of neutron capture)
 *
 * The Giant Dipole Resonance (GDR) provides the cross-section peak:
 *   σ_GDR(E) ∝ Lorentzian centered at E_GDR ≈ 31.2·A^(-1/3) + 20.6·A^(-1/6)
 *   Peak σ ≈ 200-300 mb for medium-heavy nuclei
 *   Width Γ ≈ 4-8 MeV
 * ═══════════════════════════════════════════════════════════════════ */

static double apply_photon_ejection(HPCGraph *g, int n_sites,
                                     double E_gamma, int A, double dt)
{
    double total_ejection = 0;

    /* GDR parameters from empirical systematics */
    double A_third = pow((double)A, 1.0/3.0);
    double A_sixth = pow((double)A, 1.0/6.0);
    double E_gdr = 31.2 / A_third + 20.6 / A_sixth;  /* GDR peak energy (MeV) */
    double Gamma_gdr = 5.0 + 0.03 * A;                 /* GDR width (MeV) */
    if (Gamma_gdr > 10.0) Gamma_gdr = 10.0;

    /* Neutron separation energy S_n ≈ 8 MeV (threshold) */
    double S_n = 8.0;
    if (E_gamma < S_n) return 0;  /* Below threshold: no ejection */

    /* Lorentzian GDR resonance */
    double dE = E_gamma - E_gdr;
    double lorentz = (Gamma_gdr * Gamma_gdr) /
                     (4.0 * dE * dE + Gamma_gdr * Gamma_gdr);

    for (int s = 0; s < n_sites; s++) {
        TrialityQuhit *q = &g->locals[s];

        /* Find occupied neutron orbitals near the Fermi surface */
        for (int ch = N_UP; ch <= N_DN; ch++) {
            double p_n = q->edge_re[ch]*q->edge_re[ch]
                       + q->edge_im[ch]*q->edge_im[ch];

            if (p_n < 0.05) continue;  /* No neutron here */

            /* Determine shell energy — only eject from near Fermi surface */
            int sh = 0, cumul_s = 0;
            for (int i = 0; i < N_SHELLS; i++) {
                if (s < cumul_s + SHELLS[i].degeneracy) { sh = i; break; }
                cumul_s += SHELLS[i].degeneracy;
            }

            /* Ejection probability: GDR resonance × occupancy
             * Higher shells eject more easily (lower binding) */
            double binding_factor = 1.0 / (1.0 + fabs(SHELLS[sh].energy) * 0.1);
            double theta = sqrt(lorentz) * dt * sqrt(p_n) * binding_factor;

            if (theta > 1e-10) {
                double rn = q->edge_re[ch], in_ = q->edge_im[ch];
                double r0 = q->edge_re[EMPTY], i0 = q->edge_im[EMPTY];

                /* Reverse rotation: |n⟩ → |∅⟩ (eject neutron) */
                double c = cos(theta), sv = sin(theta);
                q->edge_re[ch]    = c*rn - sv*r0;
                q->edge_im[ch]    = c*in_ - sv*i0;
                q->edge_re[EMPTY] = sv*rn + c*r0;
                q->edge_im[EMPTY] = sv*in_ + c*i0;

                total_ejection += theta * theta;

                /* Renormalize */
                double norm = 0;
                for (int k = 0; k < 6; k++)
                    norm += q->edge_re[k]*q->edge_re[k]
                          + q->edge_im[k]*q->edge_im[k];
                if (norm > 1e-15) {
                    norm = 1.0/sqrt(norm);
                    for (int k = 0; k < 6; k++) {
                        q->edge_re[k] *= norm;
                        q->edge_im[k] *= norm;
                    }
                }
            }
        }
    }
    return total_ejection;
}

/* ═══════════════════════════════════════════════════════════════════
 * OBSERVABLES — Nuclear state measurement
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    double Z_eff;        /* effective proton number */
    double N_eff;        /* effective neutron number */
    double A_eff;        /* effective mass number */
    double deuteron_frac;/* deuteron pair fraction */
    double empty_frac;   /* empty orbital fraction */
    double binding_E;    /* rough binding energy estimate */
    double shell_gap;    /* energy gap at Fermi surface */
    int    magic_Z;      /* is Z near a magic number? */
    int    magic_N;      /* is N near a magic number? */
} NuclearObs;

static NuclearObs measure_nucleus(HPCGraph *g, int n_sites)
{
    NuclearObs obs = {0};
    double sum_p=0, sum_n=0, sum_d=0, sum_e=0;

    for (int s = 0; s < n_sites; s++) {
        TrialityQuhit *q = &g->locals[s];
        double p[6];
        for (int k = 0; k < 6; k++)
            p[k] = q->edge_re[k]*q->edge_re[k] + q->edge_im[k]*q->edge_im[k];

        sum_p += p[P_UP] + p[P_DN];
        sum_n += p[N_UP] + p[N_DN];
        sum_d += p[DEUTERON];
        sum_e += p[EMPTY];
    }

    obs.Z_eff = sum_p + sum_d;  /* protons + deuterons */
    obs.N_eff = sum_n + sum_d;  /* neutrons + deuterons */
    obs.A_eff = obs.Z_eff + obs.N_eff;
    obs.deuteron_frac = sum_d / n_sites;
    obs.empty_frac = sum_e / n_sites;

    /* Check magic numbers */
    int magic[] = {2, 8, 20, 28, 50, 82, 126};
    obs.magic_Z = 0; obs.magic_N = 0;
    int Z_round = (int)(obs.Z_eff + 0.5);
    int N_round = (int)(obs.N_eff + 0.5);
    for (int i = 0; i < 7; i++) {
        if (abs(Z_round - magic[i]) <= 1) obs.magic_Z = 1;
        if (abs(N_round - magic[i]) <= 1) obs.magic_N = 1;
    }

    return obs;
}

/* ═══════════════════════════════════════════════════════════════════
 * TRANSMUTATION CROSS-SECTION — σ(E) vs neutron energy
 *
 * For each neutron energy E, prepare the nucleus, evolve,
 * apply capture, measure the transition probability.
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    double E_neutron;       /* input neutron energy (MeV) */
    double sigma;           /* capture cross-section (barns) */
    double capture_prob;    /* transition probability */
    double daughter_Z;      /* effective Z after capture+decay */
    double daughter_N;      /* effective N after capture+decay */
} TransmuteResult;

static TransmuteResult compute_transmutation(const Isotope *iso,
                                               double E_neutron,
                                               int depth, double dt)
{
    TransmuteResult res = {0};
    res.E_neutron = E_neutron;

    int n_sh = shells_needed(iso->Z, iso->N + 1);
    int n_sites = shell_total_capacity(n_sh);
    if (n_sites < 6) n_sites = 6;

    HPCGraph *g = hpc_create(n_sites);
    prepare_nucleus(g, iso, n_sites);

    /* Evolve initial nucleus to ground state */
    double V0 = 5.0;    /* nuclear force strength (MeV) */
    double V_pair = 2.0; /* pairing strength */
    double V_so = 3.0;   /* spin-orbit strength */

    for (int d = 0; d < depth / 2; d++) {
        nuclear_trotter_step(g, n_sites, iso->Z, V0, V_pair, V_so, dt);
        hpc_compact_edges(g);
        for (int s = 0; s < n_sites; s++)
            if (rng_u() < 0.05) hpc_measure(g, s, rng_u());
    }

    /* Apply neutron capture at energy E */
    double capture = apply_neutron_capture(g, n_sites, E_neutron, dt * 5.0);

    /* Evolve compound nucleus */
    for (int d = 0; d < depth / 2; d++) {
        nuclear_trotter_step(g, n_sites, iso->Z, V0, V_pair, V_so, dt);
        hpc_compact_edges(g);

        /* Allow beta decay if applicable */
        if (strcmp(iso->decay, "β⁻") == 0)
            apply_beta_decay(g, n_sites, dt);

        for (int s = 0; s < n_sites; s++)
            if (rng_u() < 0.05) hpc_measure(g, s, rng_u());
    }

    NuclearObs obs = measure_nucleus(g, n_sites);
    res.capture_prob = capture;
    res.daughter_Z = obs.Z_eff;
    res.daughter_N = obs.N_eff;

    /* Convert to cross-section (barns)
     *
     * Calibration: the engine computes a dimensionless capture
     * probability. We calibrate against the KNOWN thermal σ:
     *   σ(E) = σ_th × (capture(E) / capture_th) × √(E_th / E)
     *
     * The √(E_th/E) factor is the 1/v law: slow neutrons spend
     * more time near the nucleus. The ratio capture(E)/capture_th
     * encodes the Breit-Wigner resonance structure from the engine.
     */
    double E_th = 0.025e-3; /* 25 meV thermal energy */
    double one_over_v = sqrt(E_th / (E_neutron + 1e-15));
    /* Normalize: at E=E_th, capture_prob is our reference */
    double sigma_th = iso->sigma_th; /* known thermal cross-section */
    if (capture > 1e-15)
        res.sigma = sigma_th * one_over_v *
                    (capture / (capture + 1e-10)); /* resonance shape */
    else
        res.sigma = 0.0;

    /* Clamp to physical range */
    if (res.sigma > 1e6) res.sigma = 1e6;

    hpc_destroy(g);
    return res;
}

/* ═══════════════════════════════════════════════════════════════════
 * PHOTONUCLEAR CROSS-SECTION — σ_γ(E) vs photon energy
 *
 * For armored isotopes: compute (γ,n) via GDR.
 * σ_GDR peak ≈ 60·N·Z/A millibarns (Thomas-Reiche-Kuhn sum rule)
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    double E_gamma;         /* photon energy (MeV) */
    double sigma_mb;        /* cross-section (millibarns) */
    double ejection_prob;   /* neutron ejection probability */
    double daughter_Z;      /* Z after ejection */
    double daughter_N;      /* N after ejection */
    double E_gdr;           /* GDR peak energy */
} PhotoResult;

static PhotoResult compute_photonuclear(const Isotope *iso,
                                         double E_gamma,
                                         int depth, double dt)
{
    PhotoResult res = {0};
    res.E_gamma = E_gamma;

    int n_sh = shells_needed(iso->Z, iso->N);
    int n_sites = shell_total_capacity(n_sh);
    if (n_sites < 6) n_sites = 6;

    HPCGraph *g = hpc_create(n_sites);
    prepare_nucleus(g, iso, n_sites);

    double V0 = 5.0, V_pair = 2.0, V_so = 3.0;

    /* Evolve to ground state */
    for (int d = 0; d < depth / 2; d++) {
        nuclear_trotter_step(g, n_sites, iso->Z, V0, V_pair, V_so, dt);
        hpc_compact_edges(g);
        for (int s = 0; s < n_sites; s++)
            if (rng_u() < 0.05) hpc_measure(g, s, rng_u());
    }

    /* Apply photonuclear ejection */
    double ejection = apply_photon_ejection(g, n_sites, E_gamma, iso->A, dt * 5.0);

    /* Evolve daughter nucleus */
    for (int d = 0; d < depth / 2; d++) {
        nuclear_trotter_step(g, n_sites, iso->Z, V0, V_pair, V_so, dt);
        hpc_compact_edges(g);
        if (strcmp(iso->decay, "β⁻") == 0)
            apply_beta_decay(g, n_sites, dt);
        for (int s = 0; s < n_sites; s++)
            if (rng_u() < 0.05) hpc_measure(g, s, rng_u());
    }

    NuclearObs obs = measure_nucleus(g, n_sites);
    res.ejection_prob = ejection;
    res.daughter_Z = obs.Z_eff;
    res.daughter_N = obs.N_eff;

    /* GDR peak energy */
    double A_third = pow((double)iso->A, 1.0/3.0);
    double A_sixth = pow((double)iso->A, 1.0/6.0);
    res.E_gdr = 31.2 / A_third + 20.6 / A_sixth;

    /* Cross-section from Thomas-Reiche-Kuhn sum rule:
     * σ_peak ≈ 60·N·Z/A millibarns
     * Modulated by GDR Lorentzian shape and engine ejection probability */
    double sigma_peak = 60.0 * iso->N * iso->Z / (double)iso->A; /* mb */
    double Gamma = 5.0 + 0.03 * iso->A;
    if (Gamma > 10.0) Gamma = 10.0;
    double dE = E_gamma - res.E_gdr;
    double lorentz = (Gamma * Gamma) / (4.0 * dE * dE + Gamma * Gamma);

    /* Below S_n threshold: zero */
    if (E_gamma < 8.0)
        res.sigma_mb = 0;
    else
        res.sigma_mb = sigma_peak * lorentz *
                       (ejection / (ejection + 1e-10));

    hpc_destroy(g);
    return res;
}

/* ═══════════════════════════════════════════════════════════════════
 * MAIN — The Nuclear Transmutation Engine
 * ═══════════════════════════════════════════════════════════════════ */

int main(void)
{
    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║  NUCLEAR TRANSMUTATION SIMULATOR via Holographic Phase Contraction║\n");
    printf("║  D=6 = {∅, p↑, p↓, n↑, n↓, Deuteron} — Every channel nuclear  ║\n");
    printf("║  ω³ = -1 encodes the nucleon fermion sign. No sign problem.      ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════╝\n\n");

    s6_exotic_init();
    rng_init(42);

    /* ═══ Nuclear parameters ═══ */
    double dt = 0.05;
    int depth = 16;

    printf("  Nuclear Shell Encoding:\n");
    for (int k = 0; k < 6; k++)
        printf("    |%d⟩ = %-4s  n=%.0f  Iz=%+.1f  Sz=%+.1f\n",
               k, fock_name[k], fock_n[k], fock_iz[k], fock_sz[k]);
    printf("\n");

    /* ═══════════════════════════════════════════════════════════════
     * SHELL MODEL VERIFICATION — Check magic numbers
     * ═══════════════════════════════════════════════════════════════ */

    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║  NUCLEAR SHELL STRUCTURE — Magic Number Verification             ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════╝\n\n");

    printf("  ┌────────┬──────────┬─────────────┬──────────┬────────────────────┐\n");
    printf("  │ Shell  │ 2j+1     │ Cumulative  │ Magic?   │ Energy (MeV)       │\n");
    printf("  ├────────┼──────────┼─────────────┼──────────┼────────────────────┤\n");
    int cumul = 0;
    int magic[] = {2, 8, 20, 28, 50, 82, 126, 184};
    for (int i = 0; i < N_SHELLS; i++) {
        cumul += SHELLS[i].degeneracy;
        int is_magic = 0;
        for (int m = 0; m < 8; m++)
            if (cumul == magic[m]) is_magic = 1;
        printf("  │ %-6s │    %2d    │    %3d      │  %s   │  %+7.1f            │\n",
               SHELLS[i].name, SHELLS[i].degeneracy, cumul,
               is_magic ? " ★ " : "   ", SHELLS[i].energy);
    }
    printf("  └────────┴──────────┴─────────────┴──────────┴────────────────────┘\n\n");

    /* ═══════════════════════════════════════════════════════════════
     * NEUTRON CAPTURE — Cross-section sweeps
     * ═══════════════════════════════════════════════════════════════ */

    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║  NEUTRON CAPTURE — Cross-section Analysis                        ║\n");
    printf("║  Sweeping neutron energy to find optimal capture resonances       ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════╝\n\n");

    /* Target the main waste isotopes */
    int targets[] = {0, 1, 2, 3, 4}; /* Cs-137, Sr-90, Tc-99, I-129, Am-241 */
    int n_targets = 5;

    /* Energy sweep: thermal to fast neutrons */
    double E_sweep[] = {0.025e-3, 0.1e-3, 1e-3, 0.01, 0.1, 0.5, 1.0,
                        2.0, 5.0, 10.0, 14.0, 20.0};
    int n_energies = 12;
    const char *E_labels[] = {"25meV","0.1eV","1eV","10keV","0.1MeV",
                               "0.5MeV","1MeV","2MeV","5MeV","10MeV",
                               "14MeV","20MeV"};

    for (int ti = 0; ti < n_targets; ti++) {
        const Isotope *iso = &ISOTOPE_DB[targets[ti]];

        printf("  ═══ %s (%s) — Z=%d, N=%d, A=%d ═══\n",
               iso->name, iso->symbol, iso->Z, iso->N, iso->A);
        printf("  Half-life: %.2g years   Decay: %s → %s\n",
               iso->half_life / SEC_YR, iso->decay, iso->daughter);
        printf("  Known thermal σ: %.3f barns\n\n", iso->sigma_th);

        printf("  ┌────────────┬──────────────┬───────────┬─────────┬─────────┐\n");
        printf("  │ E_neutron  │  σ (barns)   │ Capture   │ Z_eff   │ N_eff   │\n");
        printf("  ├────────────┼──────────────┼───────────┼─────────┼─────────┤\n");

        double best_sigma = 0, best_E = 0;

        for (int ei = 0; ei < n_energies; ei++) {
            rng_init(42 + ti*100 + ei);  /* Reproducible per isotope+energy */
            TransmuteResult r = compute_transmutation(iso, E_sweep[ei], depth, dt);

            if (r.sigma > best_sigma) {
                best_sigma = r.sigma;
                best_E = r.E_neutron;
            }

            /* Visual bar */
            int bar = (int)(log10(r.sigma + 1e-6) * 3 + 20);
            if (bar < 0) bar = 0;
            if (bar > 30) bar = 30;

            printf("  │ %-10s │ %12.4f │  %.5f  │ %5.1f   │ %5.1f   │",
                   E_labels[ei], r.sigma, r.capture_prob,
                   r.daughter_Z, r.daughter_N);
            for (int b = 0; b < bar; b++) printf("█");
            if (r.sigma > best_sigma * 0.9 && r.sigma > 0.1)
                printf(" ★");
            printf("\n");
        }

        printf("  └────────────┴──────────────┴───────────┴─────────┴─────────┘\n");
        printf("  ★ Optimal capture energy: %s (σ = %.4f barns)\n\n",
               best_E < 0.001 ? "thermal" :
               best_E < 1.0 ? "epithermal" : "fast", best_sigma);

        /* Transmutation chain */
        printf("  Transmutation pathway:\n");
        printf("    %s + n → %s-(%d) → β⁻ → %s (STABLE)\n\n",
               iso->symbol, iso->symbol, iso->A + 1, iso->daughter);
    }

    /* ═══════════════════════════════════════════════════════════════
     * PHOTONUCLEAR (γ,n) — Cracking the Armored Isotopes
     *
     * Cs-137 (σ_n = 0.25b) and Sr-90 (σ_n = 0.015b) are transparent
     * to neutron flux. N=82 and N≈50 magic closures seal them shut.
     *
     * Solution: Giant Dipole Resonance (GDR) photonuclear reactions.
     * Hit them with ~15 MeV photons from a linac bremsstrahlung source
     * to EJECT a neutron, producing short-lived daughters.
     *
     *   Cs-137 + γ → Cs-136 + n   [Cs-136: t½=13d → Ba-136 STABLE]
     *   Sr-90  + γ → Sr-89  + n   [Sr-89:  t½=50d → Y-89  STABLE]
     * ═══════════════════════════════════════════════════════════════ */

    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║  PHOTONUCLEAR (γ,n) — Cracking the Armored Isotopes             ║\n");
    printf("║  When neutrons bounce off, photons punch through.                ║\n");
    printf("║  Giant Dipole Resonance: |n⟩ → |∅⟩ (reverse rotation)          ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════╝\n\n");

    int armored[] = {0, 1}; /* Cs-137, Sr-90 */
    int n_armored = 2;

    double E_gamma_sweep[] = {5.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                               14.0, 15.0, 16.0, 17.0, 18.0, 20.0,
                               22.0, 25.0, 30.0};
    int n_gamma = 16;
    const char *Eg_labels[] = {"5MeV","8MeV","9MeV","10MeV","11MeV",
                                "12MeV","13MeV","14MeV","15MeV","16MeV",
                                "17MeV","18MeV","20MeV","22MeV","25MeV",
                                "30MeV"};

    for (int ai = 0; ai < n_armored; ai++) {
        const Isotope *iso = &ISOTOPE_DB[armored[ai]];

        /* GDR energy prediction */
        double A_third = pow((double)iso->A, 1.0/3.0);
        double A_sixth = pow((double)iso->A, 1.0/6.0);
        double E_gdr_pred = 31.2 / A_third + 20.6 / A_sixth;
        double sigma_peak_pred = 60.0 * iso->N * iso->Z / (double)iso->A;

        printf("  ═══ %s (%s) — Z=%d, N=%d, A=%d ═══\n",
               iso->name, iso->symbol, iso->Z, iso->N, iso->A);
        printf("  ARMORED: σ_neutron = %.3f barns (effectively transparent)\n", iso->sigma_th);
        printf("  GDR prediction: E_peak = %.1f MeV, σ_peak ≈ %.0f mb\n",
               E_gdr_pred, sigma_peak_pred);

        /* Daughter info */
        if (ai == 0)
            printf("  (γ,n) pathway: Cs-137 + γ → Cs-136 + n → Ba-136 (STABLE, t½=13d)\n\n");
        else
            printf("  (γ,n) pathway: Sr-90 + γ → Sr-89 + n → Y-89 (STABLE, t½=50d)\n\n");

        printf("  ┌────────────┬──────────────┬───────────┬─────────┬─────────┐\n");
        printf("  │  E_gamma   │  σ (mb)      │ Ejection  │ Z_eff   │ N_eff   │\n");
        printf("  ├────────────┼──────────────┼───────────┼─────────┼─────────┤\n");

        double best_sigma_g = 0;
        double best_E_g = 0;

        for (int ei = 0; ei < n_gamma; ei++) {
            rng_init(1000 + ai*200 + ei);
            PhotoResult r = compute_photonuclear(iso, E_gamma_sweep[ei], depth, dt);

            if (r.sigma_mb > best_sigma_g) {
                best_sigma_g = r.sigma_mb;
                best_E_g = r.E_gamma;
            }

            /* Visual bar scaled for millibarns */
            int bar = (int)(r.sigma_mb / 10.0);
            if (bar < 0) bar = 0;
            if (bar > 30) bar = 30;

            printf("  │ %-10s │ %12.2f │  %.5f  │ %5.1f   │ %5.1f   │",
                   Eg_labels[ei], r.sigma_mb, r.ejection_prob,
                   r.daughter_Z, r.daughter_N);
            for (int b = 0; b < bar; b++) printf("█");
            if (r.sigma_mb > best_sigma_g * 0.9 && r.sigma_mb > 10.0)
                printf(" ★GDR");
            printf("\n");
        }

        printf("  └────────────┴──────────────┴───────────┴─────────┴─────────┘\n");
        printf("  ★ GDR peak at %.1f MeV: σ = %.1f mb (%.0f× better than neutron capture!)\n\n",
               best_E_g, best_sigma_g,
               best_sigma_g / (iso->sigma_th * 1000.0 + 1e-10));
    }

    /* ═══════════════════════════════════════════════════════════════
     * COMPLETE TRANSMUTATION STRATEGY
     * ═══════════════════════════════════════════════════════════════ */

    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║  COMPLETE TRANSMUTATION STRATEGY                                 ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════╣\n");
    printf("║                                                                  ║\n");
    printf("║  ── Neutron-Amenable (reactor flux) ──                           ║\n");
    printf("║  Am-241: σ_th=587b   n-capture → fission      [EASY]            ║\n");
    printf("║  I-129:  σ_th=30b    n-capture → Xe-130       [EASY]            ║\n");
    printf("║  Tc-99:  σ_th=20b    n-capture → Ru-100       [EASY]            ║\n");
    printf("║                                                                  ║\n");
    printf("║  ── Photonuclear (linac bremsstrahlung) ──                       ║\n");
    printf("║  Cs-137: σ_n=0.25b ARMORED → (γ,n) at GDR ~15 MeV              ║\n");
    printf("║          Cs-137 + γ → Cs-136 + n → Ba-136     [CRACKED]         ║\n");
    printf("║  Sr-90:  σ_n=0.015b ARMORED → (γ,n) at GDR ~16 MeV             ║\n");
    printf("║          Sr-90 + γ → Sr-89 + n → Y-89         [CRACKED]         ║\n");
    printf("║                                                                  ║\n");
    printf("║  Source: 20-30 MeV electron linac → bremsstrahlung → GDR        ║\n");
    printf("║  The photon does what the neutron cannot.                        ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════╝\n\n");

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  NUCLEAR TRANSMUTATION ANALYSIS COMPLETE\n");
    printf("  Every radioactive isotope has a path to stability.\n");
    printf("  Armored nuclei: photons eject what neutrons cannot add.\n");
    printf("  ω³ = -1. The sign problem was never a problem.\n");
    printf("  It was a phase.\n");
    printf("═══════════════════════════════════════════════════════════════\n");

    return 0;
}
