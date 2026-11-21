"""Fitted models and equal-budget pose design. Truth is never used by selection.

PIGEs are separately measured; finite A travel remains cubic. C runout uses
gauge-fixed Fourier functions; nonperiodic C angle scale has its own parameter.
The research fit uses six virtual channels: three Cartesian displacements and
three separately available angular measurements. A conventional R-test alone
does not provide the angular channels.
"""
import numpy as np
from fiveaxis import ErrorModel,Machine,rx,rz,Q_SCALE

LEVER=100. # mm, rotation-vector observations converted to equivalent length
NOISE=np.array([.0002]*3+[LEVER*.5e-6]*3)

def rotation_vector_small(R):
    # antisymmetric part = sin(theta)*axis; exact angle recovery at small angles
    v=np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])/2
    n=np.linalg.norm(v)
    return v if n<1e-12 else v*np.arcsin(np.clip(n,0,1))/n

def fourier(c,order=2):
    return np.array([v for k in range(1,order+1) for v in (np.sin(k*c),np.cos(k*c)-1)])

def basis(a,c,kind='periodic',order=2):
    u=a/(np.pi/2);v=c/np.pi
    return np.array([u,u*u,u**3]),fourier(c,order) if kind=='periodic' else np.array([v,v*v,v**3])

class RotaryModel(ErrorModel):
    def __init__(self,kind='periodic',order=2):
        super().__init__(convention='relative')
        self.kind=kind;self.order=order
        self.rotary=np.zeros((6,2*order if kind=='periodic' else 3));self.scale=0.
    def evaluate(self,q):
        e=super().evaluate(q)
        e[4]=self.rotary@basis(q[3],q[4],self.kind,self.order)[1]
        if self.kind=='periodic':e[4,5]+=self.scale*q[4]
        return e

def scales(kind='periodic',order=2):
    channel=np.array([.001]*3+[1e-5]*3)
    return np.r_[np.repeat(channel,3),np.repeat(channel,2*order if kind=='periodic' else 3),[1e-5] if kind=='periodic' else []]

def sensitivity(a,c,L,H,kind='periodic',order=2,orientation=True):
    RA=rx(-a)[:3,:3];RC=rz(-c)[:3,:3];bc=RC@np.array([L,0.,H]);ba=RA@bc
    e=np.eye(3);BA=np.column_stack([e,np.column_stack([np.cross(v,ba) for v in e])])
    BC=np.column_stack([RA,np.column_stack([RA@np.cross(v,bc) for v in e])])
    if orientation:
        BA=np.r_[BA,np.c_[np.zeros((3,3)),LEVER*e]]
        BC=np.r_[BC,np.c_[np.zeros((3,3)),LEVER*RA]]
    bA,bC=basis(a,c,kind,order)
    M=np.concatenate([np.concatenate([BA[:,j,None]*bA for j in range(6)],axis=1),
                      np.concatenate([BC[:,j,None]*bC for j in range(6)],axis=1)],axis=1)
    if kind=='periodic':M=np.c_[M,BC[:,5]*c]
    return M*scales(kind,order)

def candidates():
    # Endpoints excluded for C to avoid duplicate 0/360 poses in the pool.
    return np.array([(a,c,L,H) for a in np.deg2rad([-60,-30,0,30,60])
        for c in np.linspace(0,2*np.pi,16,endpoint=False)
        for L,H in [(70,35),(70,110),(140,35),(140,110)]])

def select_poses(pool,budget,method='d_optimal',seed=0):
    if not 1<=budget<=len(pool):raise ValueError('Pose budget outside candidate pool')
    if method=='random':return np.random.default_rng(seed).choice(len(pool),budget,replace=False)
    if method=='uniform':
        # Space-filling, deterministic farthest point traversal in normalized
        # A/C/L/H feature space; circular C distance avoids a false seam.
        v=np.c_[pool[:,0]/(np.pi/3),np.sin(pool[:,1]),np.cos(pool[:,1]),pool[:,2]/140,pool[:,3]/110]
        chosen=[0];dist=np.sum((v-v[0])**2,axis=1)
        for _ in range(budget-1):
            dist[chosen]=-1;j=int(np.argmax(dist));chosen.append(j)
            dist=np.minimum(dist,np.sum((v-v[j])**2,axis=1))
        return np.array(chosen)
    if method!='d_optimal':raise ValueError(method)
    B=np.array([sensitivity(*r)/NOISE[:,None]*.0002 for r in pool]);P=B.shape[-1]
    # Ridge is only a design initialization; final fits/covariances unregularized.
    F=np.eye(P)*1e-10;chosen=[];available=np.ones(len(pool),bool)
    for _ in range(budget):
        inv=np.linalg.solve(F,np.eye(P))
        U=np.einsum('nij,jk->nik',B,inv)
        G=np.eye(B.shape[1])[None]+np.einsum('nij,nkj->nik',U,B)
        gain=np.linalg.slogdet(G)[1];gain[~available]=-np.inf
        j=int(np.argmax(gain));chosen.append(j);available[j]=False
        F+=B[j].T@B[j]
    return np.array(chosen)

def unpack(z,model):
    physical=z*scales(model.kind,model.order);model.coeff[3,:,1:]=physical[:18].reshape(6,3)
    n=model.rotary.size;model.rotary=physical[18:18+n].reshape(model.rotary.shape)
    if model.kind=='periodic':model.scale=float(physical[-1])
    return model

def fit(pool,observations,pige,linear_coeff,kind='periodic',order=2):
    m=RotaryModel(kind,order);m.pige=np.array(pige);m.coeff[:3]=linear_coeff
    # Remove separately measured PIGE contribution using exact fitted chain.
    nominal=np.array([(rx(-a)@rz(-c)@np.array([L,0,H,1]))[:3] for a,c,L,H in pool])
    def model_table(row):
        a,c,L,H=row;PY,PZ,PA,PC=m.pige_blocks()
        G=PA@rx(-a)@PC@rz(-c);R0=(rx(-a)@rz(-c))[:3,:3]
        return np.r_[(G@np.array([L,0,H,1]))[:3]-(R0@np.array([L,0,H])),LEVER*rotation_vector_small(G[:3,:3]@R0.T)]
    offset=np.array([model_table(r) for r in pool])
    Araw=np.array([sensitivity(*r,kind,order) for r in pool])
    A=(Araw/NOISE[None,:,None]).reshape(-1,Araw.shape[-1]);y=((observations-offset)/NOISE).ravel()
    U,s,V=np.linalg.svd(A,full_matrices=False);rank=int(np.sum(s>s[0]*1e-10))
    z=np.linalg.lstsq(A,y,rcond=1e-10)[0];unpack(z,m)
    return m,dict(rank=rank,parameters=A.shape[1],condition=float(s[0]/s[-1]) if rank==A.shape[1] else None,
        residual_um=float(1000*np.sqrt(np.mean(((A@z-y).reshape(-1,6)*NOISE)[:,:3]**2))),singular=s,z=z,A=A)

def rtest_signal(plant,pool):
    signals=[]
    for a,c,L,H in pool:
        p,R=plant.table(a,c,L,H,return_rotation=True);R0=(rx(-a)@rz(-c))[:3,:3]
        signals.append(np.r_[p-R0@np.array([L,0,H]),LEVER*rotation_vector_small(R@R0.T)])
    return np.array(signals)

def auxiliary_calibration(plant,seed=91,noise_um=.2):
    # Independent direct 6DoF observations; no truth coefficients passed to fit.
    rng=np.random.default_rng(seed);pige=plant.pige+rng.normal(size=11)*np.array([.5e-6]*3+[noise_um/1000]*3+[.5e-6]*3+[noise_um/1000,.5e-6])
    coeff=np.zeros((3,6,4));u=np.linspace(-1,1,51);B=np.c_[u,u*u,u**3]
    for i in range(3):
        y=np.array([plant.errors(np.eye(5)[i]*v*200)[i] for v in u])
        y+=rng.normal(size=y.shape)*np.array([noise_um/1000]*3+[.5e-6]*3)
        coeff[i,:,1:]=np.linalg.lstsq(B,y,rcond=None)[0].T
    return pige,coeff
