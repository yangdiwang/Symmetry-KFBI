#!/usr/bin/env python3
import math, time, json, sys, importlib.util, os
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, bmat, csc_matrix
from scipy.sparse.linalg import splu, LinearOperator
from scipy.linalg import cho_factor, cho_solve, qr

BASE_PATH='/mnt/data/kfbi_shape_work/kfbi_sphere_cap_exterior_trace.py'
CAP_PATH='/mnt/data/KFBI3D-CapTest/sphere_cap_c1_mortar_test.py'
spec=importlib.util.spec_from_file_location('base',BASE_PATH)
base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
spec2=importlib.util.spec_from_file_location('scap',CAP_PATH)
scap=importlib.util.module_from_spec(spec2); spec2.loader.exec_module(scap)

# Keep the same reference analysis atlas as the sphere experiment.
A0=scap.A_SQUARE; RHO=scap.RHO_C; THETA_C=scap.THETA_C; SIDES=scap.SIDES
PAT=base.PAT; RHO_FIT=base.RHO_FIT
BOX_MIN=-1.5; BOX_SIDE=3.0
base.BOX_MIN=BOX_MIN; base.BOX_SIDE=BOX_SIDE

class Shape:
    name='shape'
    def map_ref(self,S): raise NotImplementedError
    def ref_direction(self,P): raise NotImplementedError
    def F(self,P): raise NotImplementedError
    def gradF(self,P): raise NotImplementedError
    def normal(self,P):
        g=self.gradF(P); return g/np.linalg.norm(g,axis=-1,keepdims=True)
    def inside(self,P): return self.F(P)<0
    def graph_hessian(self,Q,t1,t2):
        # Hessian of local height r=w(s,t), outward normal direction.
        Q=np.asarray(Q,float); t1=np.asarray(t1,float); t2=np.asarray(t2,float)
        eps=2e-5
        g0=self.gradF(Q); gn=np.linalg.norm(g0,axis=-1)
        gp1=self.gradF(Q+eps*t1); gm1=self.gradF(Q-eps*t1)
        gp2=self.gradF(Q+eps*t2); gm2=self.gradF(Q-eps*t2)
        Ht1=(gp1-gm1)/(2*eps); Ht2=(gp2-gm2)/(2*eps)
        h11=-np.sum(t1*Ht1,axis=-1)/gn
        h22=-np.sum(t2*Ht2,axis=-1)/gn
        h12a=-np.sum(t1*Ht2,axis=-1)/gn
        h12b=-np.sum(t2*Ht1,axis=-1)/gn
        h12=.5*(h12a+h12b)
        return h11,h12,h22
    def project_tangent_graph(self,Q,t1,t2,n,s,t,H=None):
        # Vectorized solve F(Q+s t1+t t2+r n)=0 along the center normal.
        Q=np.asarray(Q,float); s=np.asarray(s,float); t=np.asarray(t,float)
        if H is None: H=self.graph_hessian(Q,t1,t2)
        h11,h12,h22=H
        r=.5*(h11[:,None]*s[None,:]**2+2*h12[:,None]*s[None,:]*t[None,:]+h22[:,None]*t[None,:]**2)
        P=Q[:,None,:]+t1[:,None,:]*s[None,:,None]+t2[:,None,:]*t[None,:,None]+n[:,None,:]*r[:,:,None]
        for _ in range(5):
            f=self.F(P); g=self.gradF(P); den=np.sum(g*n[:,None,:],axis=-1)
            dr=np.divide(f,den,out=np.zeros_like(f),where=np.abs(den)>1e-14)
            r-=dr
            P=Q[:,None,:]+t1[:,None,:]*s[None,:,None]+t2[:,None,:]*t[None,:,None]+n[:,None,:]*r[:,:,None]
        return P

class Ellipsoid(Shape):
    name='ellipsoid'
    def __init__(self,a=1.20,b=0.90,c=0.72): self.axes=np.array([a,b,c],float)
    def map_ref(self,S): return np.asarray(S)*self.axes
    def ref_direction(self,P):
        Y=np.asarray(P)/self.axes; return Y/np.linalg.norm(Y,axis=-1,keepdims=True)
    def F(self,P):
        P=np.asarray(P); return np.sum((P/self.axes)**2,axis=-1)-1.0
    def gradF(self,P): return 2*np.asarray(P)/(self.axes**2)

class Flower(Shape):
    name='flower'
    def __init__(self,eps=0.16,eta=0.035): self.eps=eps; self.eta=eta
    def R(self,D):
        D=np.asarray(D); x,y,z=D[...,0],D[...,1],D[...,2]
        h4=x**4-6*x*x*y*y+y**4
        return 1.0+self.eps*h4+self.eta*(3*z*z-1.0)
    def gradR_D(self,D):
        D=np.asarray(D); x,y,z=D[...,0],D[...,1],D[...,2]
        gx=self.eps*(4*x**3-12*x*y*y)
        gy=self.eps*(-12*x*x*y+4*y**3)
        gz=self.eta*6*z
        return np.stack([gx,gy,gz],axis=-1)
    def map_ref(self,S):
        S=np.asarray(S); return self.R(S)[...,None]*S
    def ref_direction(self,P):
        P=np.asarray(P); return P/np.linalg.norm(P,axis=-1,keepdims=True)
    def F(self,P):
        P=np.asarray(P); r=np.linalg.norm(P,axis=-1); safe=np.where(r>1e-14,r,1.0); D=P/safe[...,None]
        val=r-self.R(D)
        return np.where(r>1e-14,val,-1.0)
    def gradF(self,P):
        P=np.asarray(P); r=np.linalg.norm(P,axis=-1); safe=np.where(r>1e-14,r,1.0); D=P/safe[...,None]
        gD=self.gradR_D(D); proj=gD-np.sum(gD*D,axis=-1)[...,None]*D
        G=D-proj/safe[...,None]
        if np.ndim(r)==0:
            return G if r>1e-14 else np.array([1.0,0.0,0.0])
        return np.where((r>1e-14)[...,None],G,np.array([1.0,0.0,0.0]))

@dataclass
class GPatch:
    name:str; kind:str; shape:Shape; sign:int=1; side:str=''
    def refX(self,u,v):
        if self.kind=='central':
            q=np.array([A0*(2*u-1),A0*(2*v-1)]); r2=float(q@q)
            return np.array([q[0],q[1],self.sign*math.sqrt(max(0.,1-r2))])
        if self.kind=='ring':
            q0=scap.qi(self.side,v); q1=RHO*q0/np.linalg.norm(q0); q=(1-u)*q0+u*q1
            return np.array([q[0],q[1],self.sign*math.sqrt(max(0.,1-float(q@q)))])
        if self.kind=='belt':
            q0=scap.qi(self.side,v); phi=math.atan2(q0[1],q0[0]); theta=THETA_C+u*(math.pi-2*THETA_C)
            st=math.sin(theta); return np.array([st*math.cos(phi),st*math.sin(phi),math.cos(theta)])
        raise KeyError(self.kind)
    def X(self,u,v): return self.shape.map_ref(self.refX(u,v))
    def jac(self,u,v):
        h=2e-6
        def deriv(which):
            x=u if which==0 else v
            if x<h:
                return (self.X(u+h,v)-self.X(u,v))/h if which==0 else (self.X(u,v+h)-self.X(u,v))/h
            if x>1-h:
                return (self.X(u,v)-self.X(u-h,v))/h if which==0 else (self.X(u,v)-self.X(u,v-h))/h
            return (self.X(u+h,v)-self.X(u-h,v))/(2*h) if which==0 else (self.X(u,v+h)-self.X(u,v-h))/(2*h)
        return deriv(0),deriv(1)

CURRENT_SHAPE=None

def make_patches(shape):
    ps=[GPatch('NC','central',shape,+1)]
    ps += [GPatch('N'+s,'ring',shape,+1,s) for s in SIDES]
    ps += [GPatch('B'+s,'belt',shape,+1,s) for s in SIDES]
    ps += [GPatch('SC','central',shape,-1)]
    ps += [GPatch('S'+s,'ring',shape,-1,s) for s in SIDES]
    return ps

def physical_normal_derivative_row(patch,spl,edge,t):
    u,v=scap.edge_uv(edge,t); X=patch.X(u,v); Xu,Xv=patch.jac(u,v)
    tau=Xv if edge[0]=='u' else Xu; tau=tau/np.linalg.norm(tau)
    n=patch.shape.normal(X[None,:])[0]
    nu=np.cross(n,tau); nu=nu/np.linalg.norm(nu)
    J=np.column_stack([Xu,Xv]); xi=np.linalg.solve(J.T@J,J.T@nu)
    return xi[0]*spl.tensor_du(u,v)+xi[1]*spl.tensor_dv(u,v),X,nu

def build_problem_generic(n):
    shape=CURRENT_SHAPE; patches=make_patches(shape); nameidx={p.name:i for i,p in enumerate(patches)}
    seams=scap.make_seams(nameidx); spl=scap.TensorSpline(n); nloc=len(patches)*n*n; uf=scap.UF(nloc)
    for ia,ea,ib,eb,_ in seams:
        for la,lb in zip(scap.edge_ids(n,ea),scap.edge_ids(n,eb)): uf.union(ia*n*n+la,ib*n*n+lb)
    roots={}; l2m=np.empty(nloc,dtype=int)
    for k in range(nloc):
        r=uf.find(k)
        if r not in roots: roots[r]=len(roots)
        l2m[k]=roots[r]
    nm=len(roots); e=spl.elements; xgm,wgm=np.polynomial.legendre.leggauss(3)
    cr=[];cc=[];cv=[];crow=0
    for ia,ea,ib,eb,label in seams:
        pa,pb=patches[ia],patches[ib]; acc=[{} for _ in range(n)]
        for it in range(e):
            ta,tb=it/e,(it+1)/e
            for gt,wt0 in zip(xgm,wgm):
                t=.5*((tb-ta)*gt+tb+ta); wt=.5*(tb-ta)*wt0
                ra,Xa,nu=physical_normal_derivative_row(pa,spl,ea,t)
                ub,vb=scap.edge_uv(eb,t); Xb=pb.X(ub,vb); XuB,XvB=pb.jac(ub,vb)
                if np.linalg.norm(Xa-Xb)>2e-7: raise RuntimeError(f'seam mismatch {label} {np.linalg.norm(Xa-Xb)}')
                JB=np.column_stack([XuB,XvB]); xiB=np.linalg.solve(JB.T@JB,JB.T@nu)
                rb=xiB[0]*spl.tensor_du(ub,vb)+xiB[1]*spl.tensor_dv(ub,vb)
                ua,va=scap.edge_uv(ea,t); XuA,XvA=pa.jac(ua,va); tangent=XvA if ea[0]=='u' else XuA; ds=np.linalg.norm(tangent)
                psi=spl.b(t); fac=wt*ds
                for kt,psiv in enumerate(psi):
                    if abs(psiv)<1e-14: continue
                    d=acc[kt]
                    for q,val in enumerate(ra):
                        if abs(val)>1e-13:
                            col=l2m[ia*n*n+q]; d[col]=d.get(col,0.0)+fac*psiv*val
                    for q,val in enumerate(rb):
                        if abs(val)>1e-13:
                            col=l2m[ib*n*n+q]; d[col]=d.get(col,0.0)-fac*psiv*val
        for kt in range(n):
            for col,val in acc[kt].items():
                if abs(val)>1e-13: cr.append(crow);cc.append(col);cv.append(val)
            crow+=1
    C=coo_matrix((cv,(cr,cc)),shape=(crow,nm)).tocsr(); Cd=C.toarray(); Q,R,piv=qr(Cd.T,mode='economic',pivoting=True)
    diag=np.abs(np.diag(R)); tol=(diag[0] if diag.size else 1.0)*1e-10; rank=int(np.sum(diag>tol)); keep=np.sort(piv[:rank]); Cind=C[keep,:].tocsr()
    return patches,seams,spl,l2m,nm,None,None,C,Cind,rank

def eval_patch(ip,patch,spl,l2m,c,u,v):
    ids=l2m[ip*spl.n*spl.n:(ip+1)*spl.n*spl.n]; return float(spl.tensor(u,v)@c[ids])

def eval_dnu(ip,patch,edge,t,spl,l2m,c,common_nu=None):
    u,v=scap.edge_uv(edge,t); Xu,Xv=patch.jac(u,v); X=patch.X(u,v)
    if common_nu is None:
        tau=(Xv if edge[0]=='u' else Xu).copy(); tau=tau/np.linalg.norm(tau); n=patch.shape.normal(X[None,:])[0]
        common_nu=np.cross(n,tau); common_nu/=np.linalg.norm(common_nu)
    J=np.column_stack([Xu,Xv]); xi=np.linalg.solve(J.T@J,J.T@common_nu)
    row=xi[0]*spl.tensor_du(u,v)+xi[1]*spl.tensor_dv(u,v); ids=l2m[ip*spl.n*spl.n:(ip+1)*spl.n*spl.n]
    return float(row@c[ids]),common_nu

# cap-like namespace consumed by base.Atlas/build_atlas.
gcap=SimpleNamespace(open_uniform_knots=scap.open_uniform_knots, build_problem=build_problem_generic,
                     make_seams=scap.make_seams, edge_uv=scap.edge_uv, eval_patch=eval_patch, eval_dnu=eval_dnu)
base.cap=gcap

def locate_points(P,nameidx):
    D=CURRENT_SHAPE.ref_direction(np.asarray(P,float)); x,y,z=D[:,0],D[:,1],D[:,2]
    theta=np.arccos(np.clip(z,-1,1)); rho=np.hypot(x,y); maxabs=np.maximum(np.abs(x),np.abs(y))
    side=np.empty(len(D),dtype='U1'); vertical=np.abs(y)>=np.abs(x)
    side[vertical&(y>=0)]='N'; side[vertical&(y<0)]='S'; side[~vertical&(x>=0)]='E'; side[~vertical&(x<0)]='W'
    patch=np.empty(len(D),dtype=int); u=np.empty(len(D)); v=np.empty(len(D))
    north=theta<THETA_C-1e-11; south=theta>math.pi-THETA_C+1e-11; belt=~(north|south)
    for region,prefix in ((north,'N'),(south,'S')):
        central=region&(maxabs<=A0+1e-11); patch[central]=nameidx[prefix+'C']; u[central]=(x[central]/A0+1)/2; v[central]=(y[central]/A0+1)/2
        ring=region&~central
        for sd in SIDES:
            m=ring&(side==sd)
            if not np.any(m): continue
            if sd=='N': ss=x[m]/y[m]
            elif sd=='S': ss=x[m]/(-y[m])
            elif sd=='E': ss=y[m]/x[m]
            else: ss=y[m]/(-x[m])
            vv=(ss+1)/2
            if sd in ('N','S'): qx=A0*ss; qy=np.full_like(ss,A0 if sd=='N' else -A0)
            else: qx=np.full_like(ss,A0 if sd=='E' else -A0); qy=A0*ss
            r0=np.hypot(qx,qy); uu=(rho[m]-r0)/(RHO-r0)
            patch[m]=nameidx[prefix+sd];u[m]=uu;v[m]=vv
    for sd in SIDES:
        m=belt&(side==sd)
        if not np.any(m): continue
        if sd=='N': ss=x[m]/y[m]
        elif sd=='S': ss=x[m]/(-y[m])
        elif sd=='E': ss=y[m]/x[m]
        else: ss=y[m]/(-x[m])
        patch[m]=nameidx['B'+sd];u[m]=(theta[m]-THETA_C)/(math.pi-2*THETA_C);v[m]=(ss+1)/2
    return patch,np.clip(u,0,1),np.clip(v,0,1)
base.locate_points=locate_points

def local_sample_points(Q,delta):
    Q=np.asarray(Q,float); n=CURRENT_SHAPE.normal(Q); t1,t2=base.local_frames(n); H=CURRENT_SHAPE.graph_hessian(Q,t1,t2)
    s=delta*PAT[:,0]; t=delta*PAT[:,1]; P=CURRENT_SHAPE.project_tangent_graph(Q,t1,t2,n,s,t,H)
    return P,t1,t2,n,H

def pweights_general(d,t1,t2,n,H):
    s=np.sum(d*t1[:,None,:],axis=-1) if d.ndim==3 else np.sum(d*t1,axis=-1)
    t=np.sum(d*t2[:,None,:],axis=-1) if d.ndim==3 else np.sum(d*t2,axis=-1)
    r=np.sum(d*n[:,None,:],axis=-1) if d.ndim==3 else np.sum(d*n,axis=-1)
    h11,h12,h22=H
    if d.ndim==3:
        h11=h11[:,None];h12=h12[:,None];h22=h22[:,None]
    w0=np.stack([np.ones_like(s),
                 s+h11*s*r+h12*t*r,
                 t+h12*s*r+h22*t*r,
                 .5*(s*s-r*r),s*t,.5*(t*t-r*r)],axis=-1)
    w1=np.stack([r-.5*h11*s*s-h12*s*t-.5*h22*t*t+.5*(h11+h22)*r*r,
                 s*r,t*r],axis=-1)
    return w0,w1

def exact_gn_pts(P): return np.sum(base.exact_grad_pts(P)*CURRENT_SHAPE.normal(np.asarray(P,float)),axis=-1)
base.exact_gn_pts=exact_gn_pts

def grid_crossings(N):
    h=BOX_SIDE/N; n=N-1; coords=BOX_MIN+h*np.arange(1,N)
    X=coords[:,None,None];Y=coords[None,:,None];Z=coords[None,None,:]
    pts=np.stack(np.broadcast_arrays(X,Y,Z),axis=-1); inside=CURRENT_SHAPE.inside(pts)
    targets=[];neighs=[];targetflat=[];signs=[]
    for axis in range(3):
      for step in (-1,1):
        d=[0,0,0];d[axis]=step;di,dj,dk=d
        si=slice(max(0,-di),min(n,n-di));sj=slice(max(0,-dj),min(n,n-dj));sk=slice(max(0,-dk),min(n,n-dk))
        ni=slice(max(0,di),min(n,n+di));nj=slice(max(0,dj),min(n,n+dj));nk=slice(max(0,dk),min(n,n+dk))
        mask=inside[si,sj,sk]!=inside[ni,nj,nk];loc=np.argwhere(mask);baseidx=np.array([max(0,-di),max(0,-dj),max(0,-dk)])
        idx=loc+baseidx;nb=idx+np.array(d);P=np.column_stack([coords[idx[:,0]],coords[idx[:,1]],coords[idx[:,2]]]);Q=np.column_stack([coords[nb[:,0]],coords[nb[:,1]],coords[nb[:,2]]])
        targets.append(P);neighs.append(Q);targetflat.append((idx[:,0]*n+idx[:,1])*n+idx[:,2]);signs.append(np.where(inside[idx[:,0],idx[:,1],idx[:,2]],1.0,-1.0))
    P=np.vstack(targets);Q=np.vstack(neighs);tf=np.concatenate(targetflat);sg=np.concatenate(signs);d=Q-P
    flo=CURRENT_SHAPE.F(P); lo=np.zeros(len(P));hi=np.ones(len(P))
    for _ in range(38):
        mid=.5*(lo+hi); M=P+mid[:,None]*d; fm=CURRENT_SHAPE.F(M); same=(fm*flo)>0;lo=np.where(same,mid,lo);hi=np.where(same,hi,mid)
    tt=.5*(lo+hi);C=P+tt[:,None]*d
    return coords,inside,tf,sg,Q,C

def build_spread(N,atlas,kind):
    h=BOX_SIDE/N;delta=.35*h;D3,D2=base.jet_maps(delta);coords,inside,tf,sg,Q,X=grid_crossings(N);E=len(tf)
    samples,t1,t2,nrm,H=local_sample_points(X,delta);Bsam=atlas.basis_matrix(samples.reshape(-1,3));w0,w1=pweights_general(Q[:,None,:]-X[:,None,:],t1,t2,nrm,H)
    w0=w0[:,0,:];w1=w1[:,0,:]
    if kind==0: alpha=w0@D3;A=base.aggregation_matrix(alpha);Prows=A@Bsam
    else: alpha=w1@D2;A=base.aggregation_matrix(alpha,first_count=9);Prows=A@Bsam
    ng=N-1;M=ng**3;G=coo_matrix((sg/(h*h),(tf,np.arange(E))),shape=(M,E)).tocsr();S=(G@Prows).tocsr();S.eliminate_zeros()
    uS=base.exact_u_pts(samples);j0=uS@D3.T;gnS=exact_gn_pts(samples[:,:9,:]);j1=gnS@D2.T
    p0=np.sum(w0*j0,axis=1);p1=np.sum(w1*j1,axis=1);f0=np.asarray(G@p0).ravel();f1=np.asarray(G@p1).ravel()
    return dict(S=S,known_j0=f0,known_j1=f1,coords=coords,inside=inside,crossings=E)

def build_trace_both(N,atlas):
    h=BOX_SIDE/N;delta=.35*h;D3,D2=base.jet_maps(delta);q=atlas.test_points;nt=len(q)
    samples,t1,t2,nrm,H=local_sample_points(q,delta);Bsam=atlas.basis_matrix(samples.reshape(-1,3))
    M0v=np.zeros((nt,6));M1v=np.zeros((nt,3));M0n=np.zeros((nt,6));M1n=np.zeros((nt,3));rrs=[];ccs=[];vvs=[];nvs=[]
    ox,oy,oz=np.meshgrid(np.arange(4),np.arange(4),np.arange(4),indexing='ij');ox=ox.ravel();oy=oy.ravel();oz=oz.ravel();chunk=256
    for a in range(0,nt,chunk):
        b=min(nt,a+chunk);qq=q[a:b];nc=nrm[a:b];t1c=t1[a:b];t2c=t2[a:b];Hc=tuple(x[a:b] for x in H);m=b-a
        queries=qq[:,None,:]+(RHO_FIT[None,:,None]*h)*nc[:,None,:];coord=(queries-BOX_MIN)/h;lower=np.floor(coord).astype(int);frac=coord-lower
        wx=base.cubic_weights_vec(frac[:,:,0]);wy=base.cubic_weights_vec(frac[:,:,1]);wz=base.cubic_weights_vec(frac[:,:,2]);W=wx[:,:,ox]*wy[:,:,oy]*wz[:,:,oz]
        ig=lower[:,:,0,None]-1+ox;jg=lower[:,:,1,None]-1+oy;kg=lower[:,:,2,None]-1+oz
        ing=(ig>=1)&(ig<=N-1)&(jg>=1)&(jg<=N-1)&(kg>=1)&(kg<=N-1);flat=((ig-1)*(N-1)+(jg-1))*(N-1)+(kg-1)
        row=np.broadcast_to(np.arange(a,b)[:,None,None],W.shape);v0=W*base.TRACE_W0[None,:,None];v1=W*(base.TRACE_W1[None,:,None]/h);mask=ing.ravel()
        rrs.append(row.ravel()[mask]);ccs.append(flat.ravel()[mask]);vvs.append(v0.ravel()[mask]);nvs.append(v1.ravel()[mask])
        Xg=np.stack([BOX_MIN+h*ig,BOX_MIN+h*jg,BOX_MIN+h*kg],axis=-1); ins=CURRENT_SHAPE.inside(Xg)
        d=Xg-qq[:,None,None,:]
        # Flatten support dimension into a point dimension for pweights helper.
        ss=np.sum(d*t1c[:,None,None,:],axis=-1);tt=np.sum(d*t2c[:,None,None,:],axis=-1);rr=np.sum(d*nc[:,None,None,:],axis=-1)
        h11,h12,h22=[x[:,None,None] for x in Hc]
        W0=np.stack([np.ones_like(ss),ss+h11*ss*rr+h12*tt*rr,tt+h12*ss*rr+h22*tt*rr,.5*(ss*ss-rr*rr),ss*tt,.5*(tt*tt-rr*rr)],axis=-1)
        W1=np.stack([rr-.5*h11*ss*ss-h12*ss*tt-.5*h22*tt*tt+.5*(h11+h22)*rr*rr,ss*rr,tt*rr],axis=-1)
        neg=-W*ins
        M0v[a:b]=np.sum(neg[:,:,:,None]*base.TRACE_W0[None,:,None,None]*W0,axis=(1,2));M1v[a:b]=np.sum(neg[:,:,:,None]*base.TRACE_W0[None,:,None,None]*W1,axis=(1,2))
        M0n[a:b]=np.sum(neg[:,:,:,None]*(base.TRACE_W1[None,:,None,None]/h)*W0,axis=(1,2));M1n[a:b]=np.sum(neg[:,:,:,None]*(base.TRACE_W1[None,:,None,None]/h)*W1,axis=(1,2))
    rows=np.concatenate(rrs);cols=np.concatenate(ccs);vals0=np.concatenate(vvs);vals1=np.concatenate(nvs);M=(N-1)**3
    Rv=coo_matrix((vals0,(rows,cols)),shape=(nt,M)).tocsr();Rv.sum_duplicates();Rn=coo_matrix((vals1,(rows,cols)),shape=(nt,M)).tocsr();Rn.sum_duplicates()
    T0v=(base.aggregation_matrix(M0v@D3)@Bsam).tocsr();T0n=(base.aggregation_matrix(M0n@D3)@Bsam).tocsr();T1v=(base.aggregation_matrix(M1v@D2,first_count=9)@Bsam).tocsr();T1n=(base.aggregation_matrix(M1n@D2,first_count=9)@Bsam).tocsr()
    uS=base.exact_u_pts(samples);j0=uS@D3.T;gnS=exact_gn_pts(samples[:,:9,:]);j1=gnS@D2.T
    return dict(Rv=Rv,Rn=Rn,T0v=T0v,T0n=T0n,T1v=T1v,T1n=T1n,known_j0_v=np.sum(M0v*j0,axis=1),known_j1_v=np.sum(M1v*j1,axis=1),known_j0_n=np.sum(M0n*j0,axis=1),known_j1_n=np.sum(M1n*j1,axis=1))

def density_diagnostics(atlas,c,target_kind):
    vals=np.asarray(atlas.Btest@c).ravel(); q=atlas.test_points
    if target_kind=='value_mean0':
        tar=base.exact_u_pts(q);tar-=np.sum(atlas.test_weights*tar)/atlas.area
    else: tar=exact_gn_pts(q)
    err=np.abs(vals-tar);D=CURRENT_SHAPE.ref_direction(q);capmask=np.abs(D[:,2])>math.cos(THETA_C);caperr=float(np.max(err[capmask]))
    poles=CURRENT_SHAPE.map_ref(np.array([[0.,0.,1.],[0.,0.,-1.]]));pv=np.asarray(atlas.basis_matrix(poles)@c).ravel()
    if target_kind=='value_mean0':
        mean=np.sum(atlas.test_weights*base.exact_u_pts(q))/atlas.area;pt=base.exact_u_pts(poles)-mean
    else:pt=exact_gn_pts(poles)
    poleerr=float(np.max(np.abs(pv-pt)));seams=scap.make_seams(atlas.nameidx);c0=0.;c1=0.;sq=[]
    for ia,ea,ib,eb,label in seams:
        pa,pb=atlas.patches[ia],atlas.patches[ib]
        for tt in np.linspace(.01,.99,31):
            ua,va=scap.edge_uv(ea,tt);ub,vb=scap.edge_uv(eb,tt);vaa=eval_patch(ia,pa,atlas.spl,atlas.l2m,c,ua,va);vbb=eval_patch(ib,pb,atlas.spl,atlas.l2m,c,ub,vb);c0=max(c0,abs(vaa-vbb))
            da,nu=eval_dnu(ia,pa,ea,tt,atlas.spl,atlas.l2m,c);db,_=eval_dnu(ib,pb,eb,tt,atlas.spl,atlas.l2m,c,nu);dj=abs(da-db);c1=max(c1,dj);sq.append(dj*dj)
    cres=float(np.max(np.abs(atlas.C@c))) if atlas.C.shape[0] else 0.
    return dict(cap_density_linf=caperr,pole_density_linf=poleerr,c0_jump_linf=float(c0),c1_jump_linf=float(c1),c1_jump_rms=float(math.sqrt(np.mean(sq))),c1_constraint_linf=cres)
base.density_diagnostics=density_diagnostics

# Wrap build_test_projector to record geometry conditioning.
def build_test_projector(atlas):
    base.build_test_projector(atlas)
    cond=[]
    for p in atlas.patches:
        for u in np.linspace(.05,.95,7):
            for v in np.linspace(.05,.95,7):
                Xu,Xv=p.jac(u,v); s=np.linalg.svd(np.column_stack([Xu,Xv]),compute_uv=False); cond.append(float(s[0]/s[-1]))
    atlas.max_jac_condition=max(cond)

def evaluate_metrics(N,atlas,poisson,sp0,sp1,tr):
    nm=atlas.nm;inside=sp0['inside'];coords=sp0['coords'];X=coords[:,None,None];Y=coords[None,:,None];Z=coords[None,None,:]
    exact=np.exp(base.AA*X)*np.cos(base.BB*Y)*np.cos(base.CC*Z);exact_flat=exact.ravel();ins=inside.ravel();res={}
    known_pot=poisson.solve(sp0['known_j1']);known_trace=tr['Rv']@known_pot+tr['known_j1_v'];rhs_c=-atlas.project_trace(np.asarray(known_trace).ravel());const_c=atlas.project_trace(np.ones(len(atlas.test_points)))
    def mvN(x):
        c=atlas.project_c1(x[:nm]);lam=x[nm];pot=poisson.solve(sp0['S']@c);tv=tr['Rv']@pot+tr['T0v']@c;rc=atlas.project_trace(np.asarray(tv).ravel())+lam*const_c;return np.r_[rc,atlas.mean_vec@c]
    op=LinearOperator((nm+1,nm+1),matvec=mvN,dtype=float);sol,info,hist=base.gmres_solve(op,np.r_[rhs_c,0.]);cN=atlas.project_c1(sol[:nm]);potN=poisson.solve(sp0['S']@cN+sp0['known_j1']);traceN=np.asarray(tr['Rv']@potN+tr['T0v']@cN+tr['known_j1_v']).ravel();shift=float(np.mean(exact_flat[ins]-potN[ins]));errN=float(np.max(np.abs(potN[ins]+shift-exact_flat[ins])));dens=np.asarray(atlas.Btest@cN).ravel();tar=base.exact_u_pts(atlas.test_points);tar-=np.sum(atlas.test_weights*tar)/atlas.area
    res['neumann']=dict(iterations=len(hist),info=int(info),gmres_final=hist[-1] if hist else None,interior_linf=errN,exterior_trace_linf=float(np.max(np.abs(traceN))),density_linf=float(np.max(np.abs(dens-tar))),constant_shift=shift);res['neumann'].update(density_diagnostics(atlas,cN,'value_mean0'))
    known_pot=poisson.solve(sp1['known_j0']);known_trace=tr['Rn']@known_pot+tr['known_j0_n'];rhs=-atlas.project_trace(np.asarray(known_trace).ravel())
    def mvD(c):
        cp=atlas.project_c1(c);pot=poisson.solve(sp1['S']@cp);tn=tr['Rn']@pot+tr['T1n']@cp;return atlas.project_trace(np.asarray(tn).ravel())
    opD=LinearOperator((nm,nm),matvec=mvD,dtype=float);solD,infoD,histD=base.gmres_solve(opD,rhs);cD=atlas.project_c1(solD);potD=poisson.solve(sp1['S']@cD+sp1['known_j0']);traceD=np.asarray(tr['Rn']@potD+tr['T1n']@cD+tr['known_j0_n']).ravel();errD=float(np.max(np.abs(potD[ins]-exact_flat[ins])));densD=np.asarray(atlas.Btest@cD).ravel();gn=exact_gn_pts(atlas.test_points)
    res['dirichlet']=dict(iterations=len(histD),info=int(infoD),gmres_final=histD[-1] if histD else None,interior_linf=errD,exterior_normal_trace_linf=float(np.max(np.abs(traceD))),density_linf=float(np.max(np.abs(densD-gn))));res['dirichlet'].update(density_diagnostics(atlas,cD,'normal'))
    return res

def run_level(shape,N,ncoef):
    global CURRENT_SHAPE;CURRENT_SHAPE=shape;base.cap=gcap;base.BOX_MIN=BOX_MIN;base.BOX_SIDE=BOX_SIDE
    t=time.time();atlas,rank=base.build_atlas(ncoef);ta=time.time()-t
    t=time.time();build_test_projector(atlas);tp=time.time()-t
    t=time.time();sp0=build_spread(N,atlas,0);ts0=time.time()-t
    t=time.time();sp1=build_spread(N,atlas,1);ts1=time.time()-t
    t=time.time();tr=build_trace_both(N,atlas);tt=time.time()-t
    poisson=base.PoissonBox(N);t=time.time();metrics=evaluate_metrics(N,atlas,poisson,sp0,sp1,tr);tv=time.time()-t
    coords=sp0['coords'];ins=sp0['inside'].ravel();X=coords[:,None,None];Y=coords[None,:,None];Z=coords[None,None,:];exact=(np.exp(base.AA*X)*np.cos(base.BB*Y)*np.cos(base.CC*Z)).ravel();oracle=poisson.solve(sp0['known_j0']+sp0['known_j1']);oracle_err=float(np.max(np.abs(oracle[ins]-exact[ins])))
    return dict(shape=shape.name,N=N,h=BOX_SIDE/N,ncoef=ncoef,elements=ncoef-3,local_dofs=len(atlas.patches)*ncoef*ncoef,c0_merged=atlas.nm,c1_rank=rank,effective_dofs=atlas.nm-rank,test_points=len(atlas.test_points),area=atlas.area,max_jac_condition=atlas.max_jac_condition,crossings=sp0['crossings'],oracle_exact_jump_interior_linf=oracle_err,timings=dict(atlas=ta,test_projector=tp,spread_J0=ts0,spread_J1=ts1,trace=tt,solves=tv),**metrics)

def main():
    shape_arg=sys.argv[1] if len(sys.argv)>1 else 'ellipsoid';levels=[int(x) for x in sys.argv[2:]] or [32,64,128];shape=Ellipsoid() if shape_arg=='ellipsoid' else Flower();mapping={32:6,64:9,128:15};out=[]
    outdir='/mnt/data/KFBI3D-CapTest';os.makedirs(outdir,exist_ok=True)
    for N in levels:
        ncoef=mapping.get(N,max(6,round(3+3*N/32)));print(f'RUN {shape.name} N={N} ncoef={ncoef}',flush=True);t=time.time();r=run_level(shape,N,ncoef);r['total_seconds']=time.time()-t;out.append(r);print(json.dumps(r,indent=2),flush=True)
        with open(f'{outdir}/kfbi_{shape.name}_cap_exterior_trace_results.json','w') as f:json.dump(out,f,indent=2)
    for key in ('neumann','dirichlet'):
        for i in range(1,len(out)):
            out[i][key]['order']=math.log(out[i-1][key]['interior_linf']/out[i][key]['interior_linf'],2)/math.log(out[i]['N']/out[i-1]['N'],2)
    for i in range(1,len(out)):
        out[i]['oracle_order']=math.log(out[i-1]['oracle_exact_jump_interior_linf']/out[i]['oracle_exact_jump_interior_linf'],2)/math.log(out[i]['N']/out[i-1]['N'],2)
    with open(f'{outdir}/kfbi_{shape.name}_cap_exterior_trace_results.json','w') as f:json.dump(out,f,indent=2)
    import csv
    with open(f'{outdir}/kfbi_{shape.name}_cap_exterior_trace_refinement.csv','w',newline='') as f:
        fields=['shape','N','ncoef','effective_dofs','test_points','max_jac_condition','oracle_exact_jump_interior_linf','oracle_order','neu_iter','neu_err','neu_order','neu_trace','neu_density','dir_iter','dir_err','dir_order','dir_trace','dir_density','neu_c1jump','dir_c1jump']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in out:
            w.writerow(dict(shape=r['shape'],N=r['N'],ncoef=r['ncoef'],effective_dofs=r['effective_dofs'],test_points=r['test_points'],max_jac_condition=r['max_jac_condition'],oracle_exact_jump_interior_linf=r['oracle_exact_jump_interior_linf'],oracle_order=r.get('oracle_order',''),neu_iter=r['neumann']['iterations'],neu_err=r['neumann']['interior_linf'],neu_order=r['neumann'].get('order',''),neu_trace=r['neumann']['exterior_trace_linf'],neu_density=r['neumann']['density_linf'],dir_iter=r['dirichlet']['iterations'],dir_err=r['dirichlet']['interior_linf'],dir_order=r['dirichlet'].get('order',''),dir_trace=r['dirichlet']['exterior_normal_trace_linf'],dir_density=r['dirichlet']['density_linf'],neu_c1jump=r['neumann']['c1_jump_linf'],dir_c1jump=r['dirichlet']['c1_jump_linf']))

if __name__=='__main__':main()
