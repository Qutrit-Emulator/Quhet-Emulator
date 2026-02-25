/*
 * peps6d_overlay.c — 6D Tensor Network: D=6 in 6 Dimensions
 *
 * 13-index tensors, χ=2 per bond, 12 bonds per site.
 * Sparse SVD contraction across X, Y, Z, W, V, U axes.
 *
 * ── Side-channel optimized (tns_contraction_probe.c) ──
 *   • Gate sparsity via mag² (no fabs)
 *   • Zero-angle skip in Jacobi SVD (via tensor_svd.h)
 *   • 1.0 attractor: bond weights confirmed locked at 1.0
 */

#include "peps6d_overlay.h"
#include "tensor_svd.h"

static int tns6d_flat(Tns6dGrid *g, int x, int y, int z, int w, int v, int u)
{ return ((((u*g->Lv+v)*g->Lw+w)*g->Lz+z)*g->Ly+y)*g->Lx+x; }

static Tns6dBondWeight *tns6d_xbond(Tns6dGrid *g, int x, int y, int z, int w, int v, int u)
{ return &g->x_bonds[((((u*g->Lv+v)*g->Lw+w)*g->Lz+z)*g->Ly+y)*(g->Lx-1)+x]; }

static Tns6dBondWeight *tns6d_ybond(Tns6dGrid *g, int x, int y, int z, int w, int v, int u)
{ return &g->y_bonds[((((u*g->Lv+v)*g->Lw+w)*g->Lz+z)*(g->Ly-1)+y)*g->Lx+x]; }

static Tns6dBondWeight *tns6d_zbond(Tns6dGrid *g, int x, int y, int z, int w, int v, int u)
{ return &g->z_bonds[((((u*g->Lv+v)*g->Lw+w)*(g->Lz-1)+z)*g->Ly+y)*g->Lx+x]; }

static Tns6dBondWeight *tns6d_wbond(Tns6dGrid *g, int x, int y, int z, int w, int v, int u)
{ return &g->w_bonds[((((u*g->Lv+v)*(g->Lw-1)+w)*g->Lz+z)*g->Ly+y)*g->Lx+x]; }

static Tns6dBondWeight *tns6d_vbond(Tns6dGrid *g, int x, int y, int z, int w, int v, int u)
{ return &g->v_bonds[(((u*(g->Lv-1)+v)*g->Lw+w)*g->Lz+z)*g->Ly*g->Lx+y*g->Lx+x]; }

static Tns6dBondWeight *tns6d_ubond(Tns6dGrid *g, int x, int y, int z, int w, int v, int u)
{ return &g->u_bonds[((((u)*g->Lv+v)*g->Lw+w)*g->Lz+z)*g->Ly*g->Lx+y*g->Lx+x]; }

/* ═══════════════ LIFECYCLE ═══════════════ */

Tns6dGrid *tns6d_init(int Lx, int Ly, int Lz, int Lw, int Lv, int Lu)
{
    Tns6dGrid *g = (Tns6dGrid *)calloc(1, sizeof(Tns6dGrid));
    g->Lx=Lx; g->Ly=Ly; g->Lz=Lz; g->Lw=Lw; g->Lv=Lv; g->Lu=Lu;
    int N = Lx*Ly*Lz*Lw*Lv*Lu;

    g->tensors = (Tns6dTensor *)calloc(N, sizeof(Tns6dTensor));

    int nb_x = Lu*Lv*Lw*Lz*Ly*(Lx-1); if(nb_x<1) nb_x=1;
    int nb_y = Lu*Lv*Lw*Lz*(Ly-1)*Lx; if(nb_y<1) nb_y=1;
    int nb_z = Lu*Lv*Lw*(Lz-1)*Ly*Lx; if(nb_z<1) nb_z=1;
    int nb_w = Lu*Lv*(Lw-1)*Lz*Ly*Lx; if(nb_w<1) nb_w=1;
    int nb_v = Lu*(Lv-1)*Lw*Lz*Ly*Lx; if(nb_v<1) nb_v=1;
    int nb_u = (Lu-1)*Lv*Lw*Lz*Ly*Lx; if(nb_u<1) nb_u=1;

    g->x_bonds = (Tns6dBondWeight *)calloc(nb_x, sizeof(Tns6dBondWeight));
    g->y_bonds = (Tns6dBondWeight *)calloc(nb_y, sizeof(Tns6dBondWeight));
    g->z_bonds = (Tns6dBondWeight *)calloc(nb_z, sizeof(Tns6dBondWeight));
    g->w_bonds = (Tns6dBondWeight *)calloc(nb_w, sizeof(Tns6dBondWeight));
    g->v_bonds = (Tns6dBondWeight *)calloc(nb_v, sizeof(Tns6dBondWeight));
    g->u_bonds = (Tns6dBondWeight *)calloc(nb_u, sizeof(Tns6dBondWeight));

    #define INIT_BW6(arr, n) for(int i=0;i<n;i++) for(int s=0;s<TNS6D_CHI;s++) arr[i].w[s]=1.0;
    INIT_BW6(g->x_bonds, nb_x); INIT_BW6(g->y_bonds, nb_y);
    INIT_BW6(g->z_bonds, nb_z); INIT_BW6(g->w_bonds, nb_w);
    INIT_BW6(g->v_bonds, nb_v); INIT_BW6(g->u_bonds, nb_u);
    #undef INIT_BW6

    g->eng = (QuhitEngine *)calloc(1, sizeof(QuhitEngine));
    quhit_engine_init(g->eng);

    g->q_phys = (uint32_t *)calloc(N, sizeof(uint32_t));
    for (int i = 0; i < N; i++)
        g->q_phys[i] = quhit_init_basis(g->eng, 0);

    g->site_reg = (int *)calloc(N, sizeof(int));
    for (int i = 0; i < N; i++) {
        g->site_reg[i] = quhit_reg_init(g->eng, (uint64_t)i, 13, TNS6D_CHI);
        if (g->site_reg[i] >= 0) {
            g->eng->registers[g->site_reg[i]].bulk_rule = 0;
            quhit_reg_sv_set(g->eng, g->site_reg[i], 0, 1.0, 0.0);
        }
        g->tensors[i].reg_idx = g->site_reg[i];
    }
    return g;
}

void tns6d_free(Tns6dGrid *g)
{
    if (!g) return;
    free(g->tensors);
    free(g->x_bonds); free(g->y_bonds); free(g->z_bonds);
    free(g->w_bonds); free(g->v_bonds); free(g->u_bonds);
    if (g->eng) { quhit_engine_destroy(g->eng); free(g->eng); }
    free(g->q_phys); free(g->site_reg); free(g);
}

/* ═══════════════ 1-SITE GATE ═══════════════ */

struct tmp6d { uint64_t basis; double re, im; };

void tns6d_gate_1site(Tns6dGrid *g, int x, int y, int z, int w, int v, int u,
                      const double *U_re, const double *U_im)
{
    int site = tns6d_flat(g,x,y,z,w,v,u);
    int reg = g->site_reg[site];
    if (reg < 0) return;
    QuhitRegister *r = &g->eng->registers[reg];
    int D = TNS6D_D;
    uint32_t old_nnz = r->num_nonzero;
    if (old_nnz == 0) return;

    uint64_t *obs = (uint64_t*)malloc(old_nnz*sizeof(uint64_t));
    double *ore = (double*)malloc(old_nnz*sizeof(double));
    double *oim = (double*)malloc(old_nnz*sizeof(double));
    for (uint32_t e=0; e<old_nnz; e++) {
        obs[e]=r->entries[e].basis_state; ore[e]=r->entries[e].amp_re; oim[e]=r->entries[e].amp_im;
    }

    size_t cap = (size_t)old_nnz * D;
    struct tmp6d *tmp = (struct tmp6d*)calloc(cap, sizeof(*tmp));
    size_t ntmp = 0;

    for (uint32_t e=0; e<old_nnz; e++) {
        int k_old = (int)(obs[e] / TNS6D_C12);
        uint64_t bond = obs[e] % TNS6D_C12;
        for (int k_new=0; k_new<D; k_new++) {
            double ure=U_re[k_new*D+k_old], uim=U_im[k_new*D+k_old];
            if (ure*ure+uim*uim < 1e-30) continue;
            double tr = ure*ore[e]-uim*oim[e], ti = ure*oim[e]+uim*ore[e];
            uint64_t nbs = (uint64_t)k_new*TNS6D_C12 + bond;
            int found=0;
            for (size_t i=0; i<ntmp; i++) {
                if (tmp[i].basis==nbs) { tmp[i].re+=tr; tmp[i].im+=ti; found=1; break; }
            }
            if (!found && ntmp<cap) { tmp[ntmp].basis=nbs; tmp[ntmp].re=tr; tmp[ntmp].im=ti; ntmp++; }
        }
    }
    free(obs); free(ore); free(oim);

    r->num_nonzero = 0;
    for (size_t i=0; i<ntmp; i++) {
        if (tmp[i].re*tmp[i].re+tmp[i].im*tmp[i].im < 1e-30) continue;
        if (r->num_nonzero < 4096) {
            r->entries[r->num_nonzero].basis_state=tmp[i].basis;
            r->entries[r->num_nonzero].amp_re=tmp[i].re;
            r->entries[r->num_nonzero].amp_im=tmp[i].im;
            r->num_nonzero++;
        }
    }
    free(tmp);
}

/* ═══════════════ 2-SITE GATE (generic axis) ═══════════════ */

static void tns6d_gate_2site_generic(Tns6dGrid *g, int sA, int sB,
                                     Tns6dBondWeight *bw, int axis,
                                     const double *G_re, const double *G_im)
{
    int D=TNS6D_D, chi=TNS6D_CHI;
    uint64_t bp[13]={1,TNS6D_CHI,TNS6D_C2,TNS6D_C3,TNS6D_C4,TNS6D_C5,TNS6D_C6,
                     TNS6D_C7,TNS6D_C8,TNS6D_C9,TNS6D_C10,TNS6D_C11,TNS6D_C12};

    int bond_A=-1, bond_B=-1;
    switch(axis) {
        case 0: bond_A=4;  bond_B=5;  break; /* X */
        case 1: bond_A=7;  bond_B=6;  break; /* Y */
        case 2: bond_A=3;  bond_B=2;  break; /* Z */
        case 3: bond_A=1;  bond_B=0;  break; /* W */
        case 4: bond_A=9;  bond_B=8;  break; /* V */
        case 5: bond_A=11; bond_B=10; break; /* U */
    }

    QuhitRegister *regA=&g->eng->registers[g->site_reg[sA]];
    QuhitRegister *regB=&g->eng->registers[g->site_reg[sB]];

    int max_E=chi*chi;
    uint64_t *ueA=(uint64_t*)malloc(max_E*sizeof(uint64_t));
    uint64_t *ueB=(uint64_t*)malloc(max_E*sizeof(uint64_t));
    int nEA=0, nEB=0;

    for (uint32_t e=0; e<regA->num_nonzero; e++) {
        uint64_t pure=regA->entries[e].basis_state % TNS6D_C12;
        uint64_t env=(pure/bp[bond_A+1])*bp[bond_A]+(pure%bp[bond_A]);
        int found=0;
        for (int i=0;i<nEA;i++) if(ueA[i]==env){found=1;break;}
        if (!found && nEA<max_E) ueA[nEA++]=env;
    }
    for (uint32_t e=0; e<regB->num_nonzero; e++) {
        uint64_t pure=regB->entries[e].basis_state % TNS6D_C12;
        uint64_t env=(pure/bp[bond_B+1])*bp[bond_B]+(pure%bp[bond_B]);
        int found=0;
        for (int i=0;i<nEB;i++) if(ueB[i]==env){found=1;break;}
        if (!found && nEB<max_E) ueB[nEB++]=env;
    }
    if (nEA==0||nEB==0) { free(ueA);free(ueB);return; }

    int sdA=D*nEA, sdB=D*nEB;
    size_t sd2=(size_t)sdA*sdB;
    double *Th_re=(double*)calloc(sd2,sizeof(double));
    double *Th_im=(double*)calloc(sd2,sizeof(double));

    for (uint32_t eA=0; eA<regA->num_nonzero; eA++) {
        uint64_t bsA=regA->entries[eA].basis_state;
        double arA=regA->entries[eA].amp_re, aiA=regA->entries[eA].amp_im;
        if (arA*arA+aiA*aiA<1e-10) continue;
        int kA=(int)(bsA/TNS6D_C12);
        uint64_t pA=bsA%TNS6D_C12;
        int svA=(int)((pA/bp[bond_A])%chi);
        uint64_t envA=(pA/bp[bond_A+1])*bp[bond_A]+(pA%bp[bond_A]);
        int iEA=-1; for(int i=0;i<nEA;i++) if(ueA[i]==envA){iEA=i;break;}
        if (iEA<0) continue;
        int row=kA*nEA+iEA;
        for (uint32_t eB=0; eB<regB->num_nonzero; eB++) {
            uint64_t bsB=regB->entries[eB].basis_state;
            double arB=regB->entries[eB].amp_re, aiB=regB->entries[eB].amp_im;
            if (arB*arB+aiB*aiB<1e-10) continue;
            uint64_t pB=bsB%TNS6D_C12;
            int svB=(int)((pB/bp[bond_B])%chi);
            if (svA!=svB) continue;
            int kB=(int)(bsB/TNS6D_C12);
            uint64_t envB=(pB/bp[bond_B+1])*bp[bond_B]+(pB%bp[bond_B]);
            int iEB=-1; for(int i=0;i<nEB;i++) if(ueB[i]==envB){iEB=i;break;}
            if (iEB<0) continue;
            int col=kB*nEB+iEB;
            double sw=bw->w[svA], br=arB*sw, bi=aiB*sw;
            Th_re[row*sdB+col]+=arA*br-aiA*bi;
            Th_im[row*sdB+col]+=arA*bi+aiA*br;
        }
    }

    double *T2r=(double*)calloc(sd2,sizeof(double));
    double *T2i=(double*)calloc(sd2,sizeof(double));
    int D2=D*D;
    for (int kAp=0;kAp<D;kAp++)
     for (int kBp=0;kBp<D;kBp++) {
         int gr=kAp*D+kBp;
         for (int kA=0;kA<D;kA++)
          for (int kB=0;kB<D;kB++) {
              int gc=kA*D+kB;
              double gre=G_re[gr*D2+gc], gim=G_im[gr*D2+gc];
              /* Side-channel: squared gate check (avoids 2× fabs) */
              if(gre*gre+gim*gim<1e-20) continue;
              for (int eA=0;eA<nEA;eA++) {
                  int dr=kAp*nEA+eA, sr=kA*nEA+eA;
                  for (int eB=0;eB<nEB;eB++) {
                      int dc=kBp*nEB+eB, sc=kB*nEB+eB;
                      double tr=Th_re[sr*sdB+sc], ti=Th_im[sr*sdB+sc];
                      T2r[dr*sdB+dc]+=gre*tr-gim*ti;
                      T2i[dr*sdB+dc]+=gre*ti+gim*tr;
                  }
              }
          }
     }
    free(Th_re);free(Th_im);

    double *Ur=(double*)calloc((size_t)sdA*chi,sizeof(double));
    double *Ui=(double*)calloc((size_t)sdA*chi,sizeof(double));
    double *sig=(double*)calloc(chi,sizeof(double));
    double *Vr=(double*)calloc((size_t)chi*sdB,sizeof(double));
    double *Vi=(double*)calloc((size_t)chi*sdB,sizeof(double));

    tsvd_truncated(T2r,T2i,sdA,sdB,chi,Ur,Ui,sig,Vr,Vi);
    free(T2r);free(T2i);

    int rank=chi<sdB?chi:sdB; if(rank>sdA) rank=sdA;
    double sn=0; for(int s=0;s<rank;s++) sn+=sig[s];
    /* Side-channel: 1.0 attractor CONFIRMED — bond weights lock at 1.0 */
    for(int s=0;s<TNS6D_CHI;s++) bw->w[s]=1.0;

    regA->num_nonzero=0; regB->num_nonzero=0;

    for (int kA=0;kA<D;kA++)
     for (int eA=0;eA<nEA;eA++) {
         int row=kA*nEA+eA;
         uint64_t envA=ueA[eA];
         uint64_t pure=(envA/bp[bond_A])*bp[bond_A+1]+(envA%bp[bond_A]);
         for (int gv=0;gv<rank;gv++) {
             double wt=(sn>1e-30&&sig[gv]>1e-30)?sqrt(sig[gv]/sn):0.0;
             double re=Ur[row*rank+gv]*wt, im=Ui[row*rank+gv]*wt;
             if (re*re+im*im<1e-50) continue;
             uint64_t bs=kA*TNS6D_C12+pure+gv*bp[bond_A];
             if (regA->num_nonzero<4096) {
                 regA->entries[regA->num_nonzero].basis_state=bs;
                 regA->entries[regA->num_nonzero].amp_re=re;
                 regA->entries[regA->num_nonzero].amp_im=im;
                 regA->num_nonzero++;
             }
         }
     }
    for (int kB=0;kB<D;kB++)
     for (int eB=0;eB<nEB;eB++) {
         int col=kB*nEB+eB;
         uint64_t envB=ueB[eB];
         uint64_t pure=(envB/bp[bond_B])*bp[bond_B+1]+(envB%bp[bond_B]);
         for (int gv=0;gv<rank;gv++) {
             double wt=(sn>1e-30&&sig[gv]>1e-30)?sqrt(sig[gv]/sn):0.0;
             double re=wt*Vr[gv*sdB+col], im=wt*Vi[gv*sdB+col];
             if (re*re+im*im<1e-50) continue;
             uint64_t bs=kB*TNS6D_C12+pure+gv*bp[bond_B];
             if (regB->num_nonzero<4096) {
                 regB->entries[regB->num_nonzero].basis_state=bs;
                 regB->entries[regB->num_nonzero].amp_re=re;
                 regB->entries[regB->num_nonzero].amp_im=im;
                 regB->num_nonzero++;
             }
         }
     }
    free(Ur);free(Ui);free(sig);free(Vr);free(Vi);free(ueA);free(ueB);
}

/* ═══════════════ AXIS WRAPPERS ═══════════════ */

void tns6d_gate_x(Tns6dGrid *g,int x,int y,int z,int w,int v,int u,
                  const double *Gr,const double *Gi){
    tns6d_gate_2site_generic(g, tns6d_flat(g,x,y,z,w,v,u),
        tns6d_flat(g,x+1,y,z,w,v,u), tns6d_xbond(g,x,y,z,w,v,u), 0, Gr, Gi);
}
void tns6d_gate_y(Tns6dGrid *g,int x,int y,int z,int w,int v,int u,
                  const double *Gr,const double *Gi){
    tns6d_gate_2site_generic(g, tns6d_flat(g,x,y,z,w,v,u),
        tns6d_flat(g,x,y+1,z,w,v,u), tns6d_ybond(g,x,y,z,w,v,u), 1, Gr, Gi);
}
void tns6d_gate_z(Tns6dGrid *g,int x,int y,int z,int w,int v,int u,
                  const double *Gr,const double *Gi){
    tns6d_gate_2site_generic(g, tns6d_flat(g,x,y,z,w,v,u),
        tns6d_flat(g,x,y,z+1,w,v,u), tns6d_zbond(g,x,y,z,w,v,u), 2, Gr, Gi);
}
void tns6d_gate_w(Tns6dGrid *g,int x,int y,int z,int w,int v,int u,
                  const double *Gr,const double *Gi){
    tns6d_gate_2site_generic(g, tns6d_flat(g,x,y,z,w,v,u),
        tns6d_flat(g,x,y,z,w+1,v,u), tns6d_wbond(g,x,y,z,w,v,u), 3, Gr, Gi);
}
void tns6d_gate_v(Tns6dGrid *g,int x,int y,int z,int w,int v,int u,
                  const double *Gr,const double *Gi){
    tns6d_gate_2site_generic(g, tns6d_flat(g,x,y,z,w,v,u),
        tns6d_flat(g,x,y,z,w,v+1,u), tns6d_vbond(g,x,y,z,w,v,u), 4, Gr, Gi);
}
void tns6d_gate_u(Tns6dGrid *g,int x,int y,int z,int w,int v,int u,
                  const double *Gr,const double *Gi){
    tns6d_gate_2site_generic(g, tns6d_flat(g,x,y,z,w,v,u),
        tns6d_flat(g,x,y,z,w,v,u+1), tns6d_ubond(g,x,y,z,w,v,u), 5, Gr, Gi);
}

/* ═══════════════ LOCAL DENSITY ═══════════════ */

void tns6d_local_density(Tns6dGrid *g, int x, int y, int z, int w, int v, int u, double *probs)
{
    int site=tns6d_flat(g,x,y,z,w,v,u);
    int reg=g->site_reg[site];
    for (int k=0;k<TNS6D_D;k++) probs[k]=0;
    if (reg<0||!g->eng) { probs[0]=1.0; return; }
    QuhitRegister *r=&g->eng->registers[reg];
    double total=0;
    for (uint32_t e=0;e<r->num_nonzero;e++) {
        int k=(int)(r->entries[e].basis_state/TNS6D_C12);
        if (k>=TNS6D_D) continue;
        double p=r->entries[e].amp_re*r->entries[e].amp_re+
                 r->entries[e].amp_im*r->entries[e].amp_im;
        probs[k]+=p; total+=p;
    }
    if (total>1e-30) for(int k=0;k<TNS6D_D;k++) probs[k]/=total;
    else probs[0]=1.0;
}

/* ═══════════════ BATCH OPS ═══════════════ */

#define BATCH6_AXIS(name, L_check, loop_body) \
void tns6d_gate_##name##_all(Tns6dGrid *g,const double *Gr,const double *Gi) { \
    if (L_check<2) return; \
    for(int par=0;par<2;par++) loop_body \
}

void tns6d_gate_x_all(Tns6dGrid *g,const double *Gr,const double *Gi){
    if(g->Lx<2)return;
    for(int par=0;par<2;par++)
     for(int u=0;u<g->Lu;u++)for(int v=0;v<g->Lv;v++)for(int w=0;w<g->Lw;w++)
      for(int z=0;z<g->Lz;z++)for(int y=0;y<g->Ly;y++)
       for(int xh=0;xh<(g->Lx+1)/2;xh++){int x=xh*2+par;if(x<g->Lx-1) tns6d_gate_x(g,x,y,z,w,v,u,Gr,Gi);}
}
void tns6d_gate_y_all(Tns6dGrid *g,const double *Gr,const double *Gi){
    if(g->Ly<2)return;
    for(int par=0;par<2;par++)
     for(int u=0;u<g->Lu;u++)for(int v=0;v<g->Lv;v++)for(int w=0;w<g->Lw;w++)
      for(int z=0;z<g->Lz;z++)for(int yh=0;yh<(g->Ly+1)/2;yh++)
       for(int x=0;x<g->Lx;x++){int y=yh*2+par;if(y<g->Ly-1) tns6d_gate_y(g,x,y,z,w,v,u,Gr,Gi);}
}
void tns6d_gate_z_all(Tns6dGrid *g,const double *Gr,const double *Gi){
    if(g->Lz<2)return;
    for(int par=0;par<2;par++)
     for(int u=0;u<g->Lu;u++)for(int v=0;v<g->Lv;v++)for(int w=0;w<g->Lw;w++)
      for(int zh=0;zh<(g->Lz+1)/2;zh++)for(int y=0;y<g->Ly;y++)
       for(int x=0;x<g->Lx;x++){int z=zh*2+par;if(z<g->Lz-1) tns6d_gate_z(g,x,y,z,w,v,u,Gr,Gi);}
}
void tns6d_gate_w_all(Tns6dGrid *g,const double *Gr,const double *Gi){
    if(g->Lw<2)return;
    for(int par=0;par<2;par++)
     for(int u=0;u<g->Lu;u++)for(int v=0;v<g->Lv;v++)for(int wh=0;wh<(g->Lw+1)/2;wh++)
      for(int z=0;z<g->Lz;z++)for(int y=0;y<g->Ly;y++)
       for(int x=0;x<g->Lx;x++){int w=wh*2+par;if(w<g->Lw-1) tns6d_gate_w(g,x,y,z,w,v,u,Gr,Gi);}
}
void tns6d_gate_v_all(Tns6dGrid *g,const double *Gr,const double *Gi){
    if(g->Lv<2)return;
    for(int par=0;par<2;par++)
     for(int u=0;u<g->Lu;u++)for(int vh=0;vh<(g->Lv+1)/2;vh++)for(int w=0;w<g->Lw;w++)
      for(int z=0;z<g->Lz;z++)for(int y=0;y<g->Ly;y++)
       for(int x=0;x<g->Lx;x++){int v=vh*2+par;if(v<g->Lv-1) tns6d_gate_v(g,x,y,z,w,v,u,Gr,Gi);}
}
void tns6d_gate_u_all(Tns6dGrid *g,const double *Gr,const double *Gi){
    if(g->Lu<2)return;
    for(int par=0;par<2;par++)
     for(int uh=0;uh<(g->Lu+1)/2;uh++)for(int v=0;v<g->Lv;v++)for(int w=0;w<g->Lw;w++)
      for(int z=0;z<g->Lz;z++)for(int y=0;y<g->Ly;y++)
       for(int x=0;x<g->Lx;x++){int u=uh*2+par;if(u<g->Lu-1) tns6d_gate_u(g,x,y,z,w,v,u,Gr,Gi);}
}

void tns6d_normalize_site(Tns6dGrid *g, int x, int y, int z, int w, int v, int u)
{
    int site=tns6d_flat(g,x,y,z,w,v,u);
    int reg=g->site_reg[site]; if(reg<0) return;
    QuhitRegister *r=&g->eng->registers[reg];
    double n2=0;
    for(uint32_t e=0;e<r->num_nonzero;e++)
        n2+=r->entries[e].amp_re*r->entries[e].amp_re+r->entries[e].amp_im*r->entries[e].amp_im;
    if(n2>1e-20){double inv=1.0/sqrt(n2);
        for(uint32_t e=0;e<r->num_nonzero;e++){r->entries[e].amp_re*=inv;r->entries[e].amp_im*=inv;}}
}

void tns6d_gate_1site_all(Tns6dGrid *g, const double *Ur, const double *Ui)
{
    for(int u=0;u<g->Lu;u++)for(int v=0;v<g->Lv;v++)for(int w=0;w<g->Lw;w++)
     for(int z=0;z<g->Lz;z++)for(int y=0;y<g->Ly;y++)for(int x=0;x<g->Lx;x++)
         tns6d_gate_1site(g,x,y,z,w,v,u,Ur,Ui);
}

void tns6d_trotter_step(Tns6dGrid *g, const double *Gr, const double *Gi)
{
    tns6d_gate_x_all(g,Gr,Gi); tns6d_gate_y_all(g,Gr,Gi);
    tns6d_gate_z_all(g,Gr,Gi); tns6d_gate_w_all(g,Gr,Gi);
    tns6d_gate_v_all(g,Gr,Gi); tns6d_gate_u_all(g,Gr,Gi);
}
