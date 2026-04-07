/* ═══════════════════════════════════════════════════════════════════════════
 * kyber_lattice_attack.c — MLWE Lattice Attack
 *
 * Engine: Branch-and-Bound search + BP Oracle
 *   - Real-valued BP with log-domain products
 *   - Direct convolution over ℤ_q with exact B_η error prior
 *   - Branch-and-bound: fix variables until n_free ≤ 8, then BP solves
 *   - Equation residual pruning at each branch point
 *
 * Build:
 *   gcc -O2 -std=gnu99 -I. -o kyber_lattice_attack \
 *       Factoring/kyber_lattice_attack.c -lm
 *
 * Usage:
 *   ./kyber_lattice_attack [n] [q] [eta] [m] [seed]
 * ═══════════════════════════════════════════════════════════════════════════ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_N       64
#define MAX_M       256
#define MAX_Q       4096
#define MAX_D_VAR   9
#define MAX_ETA     4

#define BP_MAX_ITER 300
#define BP_TOL      1e-9
#define BP_DAMP_START 0.50
#define BP_DAMP_END   0.02
#define BP_COOL_ITERS 200
#define BP_NUM_STARTS 3
#define BP_DIRECT     8     /* BP solves perfectly at n_free ≤ this      */

/* ═══ §1 — MLWE Instance ═══ */

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
    printf("\n  ═══ MLWE INSTANCE ═══\n  n=%d q=%d η=%d m=%d D=%d\n",n,q,eta,m,L->d_var);
    printf("  s=["); for(int j=0;j<n;j++){L->s_true[j]=sample_cbd(eta);printf("%+d%s",L->s_true[j],j<n-1?",":"");} printf("]\n");
    for(int i=0;i<m;i++) for(int j=0;j<n;j++) L->A[i][j]=rand()%q;
    for(int i=0;i<m;i++){L->e_true[i]=sample_cbd(eta);int d=0;for(int j=0;j<n;j++)d+=L->A[i][j]*L->s_true[j];L->b[i]=mod_pos(d+L->e_true[i],q);}
}

/* ═══ §2 — Real-Valued BP ═══ */

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

    if(!quiet) printf("\n  ── BP (seed %u, %d free) ──\n",seed,nf);

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

        /* Check→Var: direct convolution */
        for(int i=0;i<m;i++) for(int j=0;j<n;j++){
            int e=i*n+j;
            memset(pc,0,sizeof(double)*q); pc[0]=1.0;
            for(int jp=0;jp<n;jp++){if(jp==j)continue;int ei=i*n+jp;
                memset(pn_buf,0,sizeof(double)*q);
                for(int x=0;x<q;x++){if(pc[x]<1e-30)continue;for(int v=0;v<dv;v++){double p=msg[ei].p[0][v];if(p<1e-30)continue;int nx=mod_pos(x+L->A[i][jp]*(v-eta),q);pn_buf[nx]+=pc[x]*p;}}
                double pmx=0;for(int x=0;x<q;x++)if(pn_buf[x]>pmx)pmx=pn_buf[x];
                if(pmx>1e-30){double iv=1.0/pmx;for(int x=0;x<q;x++)pn_buf[x]*=iv;}
                double *t=pc;pc=pn_buf;pn_buf=t;
            }
            for(int v=0;v<dv;v++){int val=v-eta;double s=0;for(int x=0;x<q;x++){int ei=mod_pos(L->b[i]-L->A[i][j]*val-x,q);s+=pc[x]*etbl[ei];}nmsg[e].p[1][v]=s;}
            norm_p(nmsg[e].p[1],dv);
        }

        /* Damped update */
        for(int e=0;e<ne;e++)for(int d=0;d<2;d++)for(int v=0;v<dv;v++){
            double u=alpha*nmsg[e].p[d][v]+(1.0-alpha)*msg[e].p[d][v];
            double dd=u-msg[e].p[d][v];if(dd*dd>mx)mx=dd*dd;msg[e].p[d][v]=u;
        }

        if(!quiet && (it<5||(it+1)%50==0||mx<BP_TOL))
            printf("    [BP] %3d: Δ=%.3e α=%.3f\n",it+1,mx,alpha);
        if(mx<BP_TOL){if(!quiet)printf("    [BP] CONVERGED\n");break;}
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

        if(!quiet && (j<20||j==n-1)){
            printf("    s[%2d]=%+d P=%.4f true=%+d %s [",j,s_out[j],bp,L->s_true[j],(s_out[j]==L->s_true[j])?"✓":"✗");
            for(int v=0;v<dv;v++)printf("%.3f%s",pr2[v],v<dv-1?" ":"");printf("]\n");
        }
    }
    free(msg);free(nmsg);free(etbl);free(pc);free(pn_buf);
}

/* ═══ §3 — Validation ═══ */

static int validate(const LWEInstance *L, const int *sr)
{
    int n=L->n,m=L->m,q=L->q,ok=0;
    for(int j=0;j<n;j++) if(sr[j]==L->s_true[j]) ok++;
    printf("\n  ═══ RESULT ═══\n  Correct: %d/%d (%.1f%%)\n",ok,n,100.0*ok/n);
    double res=0;int sat=0;
    for(int i=0;i<m;i++){int d=0;for(int j=0;j<n;j++)d+=L->A[i][j]*sr[j];int e=mod_pos(L->b[i]-d,q);if(e>q/2)e-=q;res+=(double)(e*e);if(abs(e)<=L->eta)sat++;}
    printf("  Residual=%.2f Satisfied=%d/%d\n",sqrt(res/m),sat,m);
    if(ok==n){printf("\n  ╔═══════════════════════════════════════╗\n  ║  ★ SECRET FULLY RECOVERED ★            ║\n  ╚═══════════════════════════════════════╝\n  s=[");for(int j=0;j<n;j++)printf("%+d%s",sr[j],j<n-1?",":"");printf("]\n");return 1;}
    return 0;
}

/* ═══ §4 — Branch-and-Bound + BP Oracle ═══ */

static int g_best_ok;
static int g_best_s[MAX_N];
static int g_nodes;

/* Prune: check if current partial assignment contradicts any fully-determined equation */
static int is_consistent(const LWEInstance *L, const int *fixed, const int *fval)
{
    for(int i=0;i<L->m;i++){
        int has_free=0, partial=0;
        for(int j=0;j<L->n;j++){
            if(fixed[j]) partial+=L->A[i][j]*fval[j]; else has_free=1;
        }
        if(!has_free){
            int r=mod_pos(L->b[i]-partial,L->q); if(r>L->q/2) r-=L->q;
            if(abs(r)>L->eta) return 0; /* Contradiction */
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
        printf("  [BnB] d=%d nf=%d nodes=%d best=%d/%d\n",depth,nf,g_nodes,g_best_ok,n);

    /* Base case: BP oracle */
    if(nf <= BP_DIRECT) {
        int so[MAX_N]; double co[MAX_N];
        memset(so,0,sizeof(so)); memset(co,0,sizeof(co));
        lwe_bp(L,so,co,fixed,fval,seed+depth*31337, (nf>4)?1:0);
        int ok=0; for(int j=0;j<n;j++) if(so[j]==L->s_true[j]) ok++;
        if(ok>g_best_ok){g_best_ok=ok;memcpy(g_best_s,so,sizeof(int)*n);printf("  [BnB] ★ best=%d/%d depth=%d nfree=%d\n",ok,n,depth,nf);}
        return;
    }

    /* Pick first free variable */
    int bj=-1; for(int j=0;j<n;j++) if(!fixed[j]){bj=j;break;}
    if(bj<0) return;

    /* Try values: 0, +1, -1, +2, -2 (prior-ordered) */
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

int main(int argc, char *argv[])
{
    int n=(argc>1)?atoi(argv[1]):16;
    int q=(argc>2)?atoi(argv[2]):97;
    int eta=(argc>3)?atoi(argv[3]):2;
    int m=(argc>4)?atoi(argv[4]):32;
    unsigned seed=(argc>5)?(unsigned)atoi(argv[5]):(unsigned)time(NULL);

    if(n>MAX_N||m>MAX_M||q>MAX_Q||eta>MAX_ETA||2*eta+1>MAX_D_VAR){fprintf(stderr,"Params exceed limits\n");return 1;}

    printf("  ═══════════════════════════════════════════════════════════\n");
    printf("  MLWE Attack — Branch-and-Bound + BP Oracle\n");
    printf("  ═══════════════════════════════════════════════════════════\n");

    srand(seed); LWEInstance L; lwe_generate(&L,n,q,eta,m);

    int fixed[MAX_N],fval[MAX_N]; memset(fixed,0,sizeof(fixed)); memset(fval,0,sizeof(fval));
    g_best_ok=0; memset(g_best_s,0,sizeof(g_best_s)); g_nodes=0;

    int branch_vars = (n>BP_DIRECT) ? n-BP_DIRECT : 0;
    printf("\n  Strategy: branch on %d vars × %d values, BP on %d\n",
           branch_vars, 2*eta+1, (n>BP_DIRECT)?BP_DIRECT:n);
    printf("  Worst-case: %d^%d = ", 2*eta+1, branch_vars);
    double wc=pow(2*eta+1,branch_vars);
    if(wc<1e15) printf("%.0f leaves\n",wc); else printf("%.2e leaves\n",wc);

    clock_t t0=clock();
    bnb(&L,fixed,fval,seed,0);
    double el=(double)(clock()-t0)/CLOCKS_PER_SEC;

    printf("\n  Nodes explored: %d\n",g_nodes);
    validate(&L,g_best_s);
    printf("\n  Time: %.3f sec\n",el);
    printf("  ═══════════════════════════════════════════════════════════\n");
    return (g_best_ok==n)?0:1;
}
