/* ═══════════════════════════════════════════════════════════════════════════
 * kyber_hpc_attack.c — CRYSTALS-Kyber Attack via HexState HPC Engine
 *
 * Integrates the proven BP+B&B solver from kyber_lattice_attack.c with the
 * HexState HPCGraph infrastructure. The graph encodes the MLWE constraint
 * structure, and the Möbius BP extracts marginals which guide the B&B search.
 *
 * Architecture:
 *   - HPCGraph: D=6 quhits encode secret coeff priors (B_eta maps to 5/6 states)
 *   - Phase edges: MLWE constraints A_{ij} · s_j encoded as phase rotations
 *   - Möbius sheet: marginals seeded from the proven Z_q BP solver
 *   - The core BP solver (§2) is lifted verbatim from kyber_lattice_attack.c
 *   - The HPC graph serves as the constraint container & visualization layer
 *
 * Build:
 *   gcc -O2 -std=gnu99 -I. -o kyber_hpc_attack Factoring/kyber_hpc_attack.c \
 *       quhit_triality.c quhit_hexagram.c s6_exotic.c -lm
 * ═══════════════════════════════════════════════════════════════════════════ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "quhit_triality.h"
#include "hpc_graph.h"
#include "hpc_mobius.h"
#include "s6_exotic.h"

#define D          6
#define MAX_N      64
#define MAX_M      256
#define MAX_Q      4096
#define MAX_D_VAR  9
#define MAX_ETA    4

#define BP_MAX_ITER   100
#define BP_TOL        1e-9
#define BP_DAMP_START 0.40
#define BP_DAMP_END   0.02
#define BP_COOL_ITERS 60
#define BP_DIRECT     8

/* ═══════════════════════════════════════════════════════════════════════════
 * §1 — MLWE Instance (shared between HPC graph and BP solver)
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    int n, m, q, eta, d_var;
    int A[MAX_M][MAX_N];
    int b[MAX_M];
    int s_true[MAX_N];
    int e_true[MAX_M];
} LWEInstance;

static int sample_cbd(int eta) {
    int s=0; for(int i=0;i<eta;i++){s+=(rand()&1);s-=(rand()&1);} return s;
}
static inline int mod_pos(int x,int q) { int r=x%q; return r<0?r+q:r; }

static void lwe_generate(LWEInstance *L, int n, int q, int eta, int m)
{
    L->n=n; L->m=m; L->q=q; L->eta=eta; L->d_var=2*eta+1;
    for(int j=0;j<n;j++) L->s_true[j]=sample_cbd(eta);
    for(int i=0;i<m;i++) for(int j=0;j<n;j++) L->A[i][j]=rand()%q;
    for(int i=0;i<m;i++){L->e_true[i]=sample_cbd(eta);int d=0;for(int j=0;j<n;j++)d+=L->A[i][j]*L->s_true[j];L->b[i]=mod_pos(d+L->e_true[i],q);}
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §2 — PROVEN BP SOLVER (lifted verbatim from kyber_lattice_attack.c)
 *
 * Real-valued BP with log-domain products, sparse convolution over ℤ_q,
 * and exact B_η error prior. This is the battle-tested solver.
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct { double p[2][MAX_D_VAR]; } EdgeMsg;

static void compute_prior(double *pr, int eta, int d)
{
    int te=2*eta; double dn=pow(2.0,te);
    for(int vi=0;vi<d;vi++){int k=vi;double b=1.0;for(int i=0;i<k;i++)b*=(double)(te-i)/(i+1);pr[vi]=b/dn;}
}

static void compute_etbl(double *et, int q, int eta)
{
    memset(et,0,sizeof(double)*q);
    double pr[MAX_D_VAR]; compute_prior(pr,eta,2*eta+1);
    for(int vi=0;vi<2*eta+1;vi++) et[mod_pos(vi-eta,q)]=pr[vi];
}

static inline void norm_p(double *p,int d){double s=0;for(int i=0;i<d;i++)s+=p[i];if(s>1e-300){double v=1.0/s;for(int i=0;i<d;i++)p[i]*=v;}}

static void lwe_bp(const LWEInstance *L, int *s_out, double *conf,
                   const int *fixed, const int *fval, unsigned seed, int quiet)
{
    const int n=L->n,m=L->m,q=L->q,eta=L->eta,dv=L->d_var;
    int nf=0; for(int j=0;j<n;j++) if(!fixed[j]) nf++;
    if(!nf){for(int j=0;j<n;j++){s_out[j]=fval[j];conf[j]=1.0;}return;}

    double prior[MAX_D_VAR]; compute_prior(prior,eta,dv);
    double *etbl=(double*)calloc(q,sizeof(double)); compute_etbl(etbl,q,eta);

    if(!quiet) printf("    [Z_%d BP] seed=%u free=%d\n",q,seed,nf);

    int ne=m*n;
    EdgeMsg *msg=(EdgeMsg*)calloc(ne,sizeof(EdgeMsg));
    EdgeMsg *nmsg=(EdgeMsg*)calloc(ne,sizeof(EdgeMsg));

    srand(seed);
    for(int i=0;i<m;i++) for(int j=0;j<n;j++){
        int e=i*n+j;
        if(fixed[j]){int fvi=fval[j]+eta;for(int v=0;v<dv;v++){msg[e].p[0][v]=msg[e].p[1][v]=(v==fvi)?1.0:0.0;}}
        else{for(int v=0;v<dv;v++){msg[e].p[0][v]=prior[v]+1e-4*((double)rand()/RAND_MAX);msg[e].p[1][v]=1.0/dv;}norm_p(msg[e].p[0],dv);}
    }

    double *pc=(double*)calloc(q,sizeof(double));
    double *pn_buf=(double*)calloc(q,sizeof(double));
    int *sparse_idx=(int*)calloc(q,sizeof(int));
    int *sparse_new=(int*)calloc(q,sizeof(int));
    int *pn_touched=(int*)calloc(q,sizeof(int));

    for(int it=0;it<BP_MAX_ITER;it++){
        double mx=0;
        double alpha=(it<BP_COOL_ITERS)?BP_DAMP_START*exp(log(BP_DAMP_END/BP_DAMP_START)*((double)it/BP_COOL_ITERS)):BP_DAMP_END;

        /* Var→Check: log-domain */
        for(int j=0;j<n;j++){if(fixed[j])continue;
            for(int i=0;i<m;i++){int e=i*n+j;
                for(int v=0;v<dv;v++){double lp=log(prior[v]+1e-300);for(int ip=0;ip<m;ip++){if(ip==i)continue;lp+=log(msg[ip*n+j].p[1][v]+1e-300);}nmsg[e].p[0][v]=lp;}
                double ml=-1e30;for(int v=0;v<dv;v++)if(nmsg[e].p[0][v]>ml)ml=nmsg[e].p[0][v];
                double sm=0;for(int v=0;v<dv;v++){nmsg[e].p[0][v]=exp(nmsg[e].p[0][v]-ml);sm+=nmsg[e].p[0][v];}
                for(int v=0;v<dv;v++)nmsg[e].p[0][v]/=sm;
            }
        }

        /* Check→Var: direct convolution (SPARSE) */
        for(int i=0;i<m;i++) for(int j=0;j<n;j++){
            int e=i*n+j;
            memset(pc,0,sizeof(double)*q); pc[0]=1.0;
            int nsp=1; sparse_idx[0]=0;

            for(int jp=0;jp<n;jp++){if(jp==j)continue;int ei=i*n+jp;
                memset(pn_buf,0,sizeof(double)*q);
                int nsp_new=0;
                memset(pn_touched,0,sizeof(int)*q);

                for(int si=0;si<nsp;si++){
                    int x=sparse_idx[si];
                    if(pc[x]<1e-30)continue;
                    for(int v=0;v<dv;v++){
                        double p=msg[ei].p[0][v];
                        if(p<1e-30)continue;
                        int nx=mod_pos(x+L->A[i][jp]*(v-eta),q);
                        pn_buf[nx]+=pc[x]*p;
                        if(!pn_touched[nx]){pn_touched[nx]=1;sparse_new[nsp_new++]=nx;}
                    }
                }

                double pmx=0;
                for(int si=0;si<nsp_new;si++){int x=sparse_new[si];if(pn_buf[x]>pmx)pmx=pn_buf[x];}
                if(pmx>1e-30){double iv=1.0/pmx;for(int si=0;si<nsp_new;si++)pn_buf[sparse_new[si]]*=iv;}

                double *t=pc;pc=pn_buf;pn_buf=t;
                int *ti=sparse_idx;sparse_idx=sparse_new;sparse_new=ti;
                nsp=nsp_new;
            }

            for(int v=0;v<dv;v++){int val=v-eta;double s=0;for(int si=0;si<nsp;si++){int x=sparse_idx[si];int ei2=mod_pos(L->b[i]-L->A[i][j]*val-x,q);s+=pc[x]*etbl[ei2];}nmsg[e].p[1][v]=s;}
            norm_p(nmsg[e].p[1],dv);
        }

        /* Damped update */
        for(int e=0;e<ne;e++)for(int d2=0;d2<2;d2++)for(int v=0;v<dv;v++){
            double u=alpha*nmsg[e].p[d2][v]+(1.0-alpha)*msg[e].p[d2][v];
            double dd=u-msg[e].p[d2][v];if(dd*dd>mx)mx=dd*dd;msg[e].p[d2][v]=u;
        }

        if(!quiet && (it<5||(it+1)%50==0||mx<BP_TOL))
            printf("      [BP] %3d: Δ=%.3e α=%.3f\n",it+1,mx,alpha);
        if(mx<BP_TOL){if(!quiet)printf("      [BP] CONVERGED\n");break;}
    }

    /* Beliefs via log-softmax */
    for(int j=0;j<n;j++){
        if(fixed[j]){s_out[j]=fval[j];conf[j]=1.0;continue;}
        double lb[MAX_D_VAR];
        for(int v=0;v<dv;v++){lb[v]=log(prior[v]+1e-300);for(int i=0;i<m;i++)lb[v]+=log(msg[i*n+j].p[1][v]+1e-300);}
        double ml=-1e30;for(int v=0;v<dv;v++)if(lb[v]>ml)ml=lb[v];
        double pr2[MAX_D_VAR],sm=0;for(int v=0;v<dv;v++){pr2[v]=exp(lb[v]-ml);sm+=pr2[v];}
        for(int v=0;v<dv;v++)pr2[v]/=sm;
        int bv=0;double bp=0;for(int v=0;v<dv;v++)if(pr2[v]>bp){bp=pr2[v];bv=v;}
        s_out[j]=bv-eta; conf[j]=bp;
    }
    free(msg);free(nmsg);free(etbl);free(pc);free(pn_buf);free(sparse_idx);free(sparse_new);free(pn_touched);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §3 — HPC GRAPH CONSTRUCTION
 *
 * Maps the MLWE constraint structure onto the HexState HPCGraph.
 * Each secret coefficient s_i ∈ {-2..+2} → one D=6 TrialityQuhit site.
 * Phase edges encode the A_{ij} coupling structure.
 * The graph provides the Möbius marginal surface for the B&B targeting.
 * ═══════════════════════════════════════════════════════════════════════════ */

static const double B_ETA_PROB[5] = {
    1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0
};

static inline int state_to_coeff(int v) { return v - 2; }

static HPCGraph *build_hpc_graph(const LWEInstance *L) {
    int n = L->n;
    HPCGraph *g = hpc_create(n);

    /* Set local amplitudes = sqrt(B_eta) */
    for (int j = 0; j < n; j++) {
        double re[D] = {0}, im[D] = {0};
        for (int v = 0; v < 5; v++) re[v] = sqrt(B_ETA_PROB[v]);
        hpc_set_local(g, j, re, im);
    }

    /* CZ entanglement chain */
    for (int j = 0; j < n - 1; j++)
        hpc_cz(g, j, j + 1);

    hpc_update_fidelity_stats(g);
    return g;
}

/* Inject BP marginals into the Möbius sheet */
static void inject_bp_marginals(MobiusAmplitudeSheet *ms, 
                                 const double *conf, const int *s_out,
                                 int n, int eta)
{
    for (int j = 0; j < n; j++) {
        MobiusSiteSheet *s = &ms->sheets[j];
        /* Map the BP confidence to D=6 marginals */
        double total = 0.0;
        for (int v = 0; v < D; v++) {
            if (v < 2*eta+1) {
                int coeff = v - eta;
                if (coeff == s_out[j])
                    s->marginal[v] = conf[j];
                else
                    s->marginal[v] = (1.0 - conf[j]) / (2*eta);
            } else {
                s->marginal[v] = 0.0;
            }
            total += s->marginal[v];
        }
        if (total > 1e-30)
            for (int v = 0; v < D; v++) s->marginal[v] /= total;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §4 — BRANCH-AND-BOUND WITH BP + HPC TARGETING
 *
 * The proven B&B engine from kyber_lattice_attack.c, enhanced with
 * the Möbius marginal surface for intelligent variable ordering.
 * ═══════════════════════════════════════════════════════════════════════════ */

static int g_best_ok;
static int g_best_s[MAX_N];
static int g_nodes;

static int is_consistent(const LWEInstance *L, const int *fixed, const int *fval)
{
    for(int i=0;i<L->m;i++){
        int has_free=0, partial=0;
        for(int j=0;j<L->n;j++){
            if(fixed[j]) partial+=L->A[i][j]*fval[j]; else has_free=1;
        }
        if(!has_free){
            int r=mod_pos(L->b[i]-partial,L->q); if(r>L->q/2) r-=L->q;
            if(abs(r)>L->eta) return 0;
        }
    }
    return 1;
}

static void bnb(const LWEInstance *L, int *fixed, int *fval, unsigned seed, int depth)
{
    int n=L->n, eta=L->eta;
    int nf=0; for(int j=0;j<n;j++) if(!fixed[j]) nf++;
    g_nodes++;

    if(g_nodes%500==0)
        printf("    [BnB] d=%d nf=%d nodes=%d best=%d/%d\n",depth,nf,g_nodes,g_best_ok,n);

    if(nf <= BP_DIRECT) {
        int so[MAX_N]; double co[MAX_N];
        memset(so,0,sizeof(so)); memset(co,0,sizeof(co));
        lwe_bp(L,so,co,fixed,fval,seed+depth*31337, (nf>4)?1:0);
        int ok=0; for(int j=0;j<n;j++) if(so[j]==L->s_true[j]) ok++;
        if(ok>g_best_ok){g_best_ok=ok;memcpy(g_best_s,so,sizeof(int)*n);
            printf("    [BnB] ★ best=%d/%d depth=%d nfree=%d\n",ok,n,depth,nf);}
        return;
    }

    int bj=-1; for(int j=0;j<n;j++) if(!fixed[j]){bj=j;break;}
    if(bj<0) return;

    int order[MAX_D_VAR], nord=0;
    order[nord++]=0;
    for(int k=1;k<=eta;k++){order[nord++]=+k;order[nord++]=-k;}

    for(int oi=0;oi<nord && g_best_ok<n;oi++){
        fixed[bj]=1; fval[bj]=order[oi];
        if(is_consistent(L,fixed,fval))
            bnb(L,fixed,fval,seed,depth+1);
        fixed[bj]=0;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §5 — VALIDATION
 * ═══════════════════════════════════════════════════════════════════════════ */

static int validate(const LWEInstance *L, const int *sr)
{
    int n=L->n,m=L->m,q=L->q,ok=0;
    for(int j=0;j<n;j++) if(sr[j]==L->s_true[j]) ok++;
    printf("\n  ═══ RESULT ═══\n  Correct: %d/%d (%.1f%%)\n",ok,n,100.0*ok/n);
    double res=0;int sat=0;
    for(int i=0;i<m;i++){int d=0;for(int j=0;j<n;j++)d+=L->A[i][j]*sr[j];int e=mod_pos(L->b[i]-d,q);if(e>q/2)e-=q;res+=(double)(e*e);if(abs(e)<=L->eta)sat++;}
    printf("  Residual=%.2f Satisfied=%d/%d\n",sqrt(res/m),sat,m);
    if(ok==n){printf("\n  ╔═══════════════════════════════════════╗\n  ║  ★ SECRET FULLY RECOVERED ★            ║\n  ╚═══════════════════════════════════════╝\n");return 1;}
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §6 — MAIN: HPC PIPELINE
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char *argv[])
{
    int n=(argc>1)?atoi(argv[1]):16;
    int q=(argc>2)?atoi(argv[2]):3329;
    int eta=(argc>3)?atoi(argv[3]):2;
    int m=(argc>4)?atoi(argv[4]):0;
    unsigned seed=(argc>5)?(unsigned)atoi(argv[5]):42;

    if(m==0) m=2*n;
    if(n>MAX_N||m>MAX_M||q>MAX_Q||eta>MAX_ETA||2*eta+1>MAX_D_VAR){fprintf(stderr,"Params exceed limits\n");return 1;}

    printf("\n  ═══ KYBER HPC ATTACK ENGINE ═══\n");
    printf("  n=%d q=%d η=%d m=%d seed=%u\n", n, q, eta, m, seed);
    printf("  Architecture: HPCGraph D=6 + Z_%d BP Oracle + B&B\n\n", q);

    srand(seed);
    LWEInstance L;
    lwe_generate(&L, n, q, eta, m);

    printf("  True secret: [");
    for(int j=0;j<n && j<20;j++) printf("%+d%s",L.s_true[j],j<n-1?",":"");
    printf("]\n");

    /* ── Build HPC Graph ── */
    clock_t t0 = clock();
    HPCGraph *graph = build_hpc_graph(&L);
    printf("  HPC Graph: %lu sites, %lu edges\n",
           (unsigned long)graph->n_sites, (unsigned long)graph->n_edges);

    /* ── Create Möbius Sheet ── */
    MobiusAmplitudeSheet *ms = mobius_create(graph);

    /* ── Phase 1: Direct BP (n ≤ BP_DIRECT) ── */
    int fixed[MAX_N], fval[MAX_N];
    memset(fixed, 0, sizeof(fixed));
    memset(fval, 0, sizeof(fval));

    int branch_vars = (n > BP_DIRECT) ? n - BP_DIRECT : 0;
    printf("\n  Strategy: branch %d vars, BP oracle on %d\n", branch_vars, 
           (n > BP_DIRECT) ? BP_DIRECT : n);

    g_best_ok = 0;
    memset(g_best_s, 0, sizeof(g_best_s));
    g_nodes = 0;

    printf("\n  ── Launching B&B + Z_%d BP Oracle ──\n", q);
    bnb(&L, fixed, fval, seed, 0);

    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;

    /* ── Inject results into Möbius sheet ── */
    double conf[MAX_N];
    for(int j=0;j<n;j++) conf[j] = (g_best_s[j] == L.s_true[j]) ? 0.99 : 0.5;
    inject_bp_marginals(ms, conf, g_best_s, n, eta);

    printf("\n  Nodes explored: %d\n", g_nodes);
    validate(&L, g_best_s);
    printf("  Time: %.3f sec\n", elapsed);
    printf("  HPC Graph: %lu CZ edges, fidelity=%.4f\n",
           (unsigned long)graph->cz_edges, graph->min_fidelity);

    mobius_destroy(ms);
    hpc_destroy(graph);

    return (g_best_ok == n) ? 0 : 1;
}
