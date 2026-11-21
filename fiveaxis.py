"""XYZ-AC geometric twin. Length mm, all internal angles rad; twist order [v,w].
Paper: Xiang & Altintas (2016), DOI 10.1016/j.ijmachtools.2015.11.006.
Synthetic coefficients are explicitly separate from the authors' experiments.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation

AXES = ('X','Y','Z','A','C')
PDGE_NAMES = tuple(f'{kind}{d}{a.lower()}' for a in AXES for kind in ('delta','eps') for d in 'xyz')
ABS_PIGE_NAMES = ('Sxy','Syz','Sxz','delta_yoa','delta_zoa','Sboa','Scoa','delta_xoc','delta_yoc','Saoc','Sboc')
REL_PIGE_NAMES = ('Sxy','Syz','Sxz','dxAX','dyAX','dzAX','alphaAX','betaAX','gammaAX','dyCA','betaCA')
Q_SCALE = np.array([200.,200.,200.,np.pi/2,np.pi])
EYE = np.eye(4)

def skew(v):
    x,y,z=np.asarray(v); return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])

def trans(v):
    g=np.eye(4); g[:3,3]=v; return g

def rot(axis, theta):
    g=np.eye(4); g[:3,:3]=Rotation.from_rotvec(np.asarray(axis)*theta).as_matrix(); return g

def rx(t): return rot([1.,0.,0.],t)
def ry(t): return rot([0.,1.,0.],t)
def rz(t): return rot([0.,0.,1.],t)

def inverse(g):
    h=np.eye(4); h[:3,:3]=g[:3,:3].T; h[:3,3]=-h[:3,:3]@g[:3,3]; return h

def exp_twist(v, w, theta=1.):
    """Exact SE(3) exponential, stable for small angles and arbitrary w norm."""
    phi=np.asarray(w,dtype=float)*theta; rho=np.asarray(v,dtype=float)*theta
    a=np.linalg.norm(phi); K=skew(phi)
    if a<1e-6:
        V=np.eye(3)+(0.5-a*a/24)*K+(1/6-a*a/120)*(K@K)
    else:
        V=np.eye(3)+(1-np.cos(a))/a**2*K+(a-np.sin(a))/a**3*(K@K)
    g=np.eye(4); g[:3,:3]=Rotation.from_rotvec(phi).as_matrix(); g[:3,3]=V@rho
    return g

def pdge_matrix(e):
    # Ordered individual error exponentials, paper Eq. (12), NOT exp(sum).
    return trans(e[:3])@rx(e[3])@ry(e[4])@rz(e[5])

@dataclass
class ErrorModel:
    coeff: np.ndarray = field(default_factory=lambda:np.zeros((5,6,4)))
    pige: np.ndarray = field(default_factory=lambda:np.zeros(11))
    convention: str = 'absolute'
    origin: np.ndarray = field(default_factory=lambda:np.zeros(5))

    def evaluate(self,q):
        u=(np.asarray(q)+self.origin)/Q_SCALE
        u0=self.origin/Q_SCALE
        b=np.stack([u**k-u0**k for k in range(4)],axis=-1)
        return np.einsum('ajk,ak->aj',self.coeff,b)

    def pige_blocks(self):
        p=self.pige
        PY=rz(p[0]); PZ=rx(p[1])@ry(-p[2])
        if self.convention=='absolute':
            PA=trans([0,p[3],p[4]])@ry(p[5])@rz(-p[6])
            PC=trans([p[7],p[8],0])@rx(p[9])@ry(-p[10])
        elif self.convention=='relative':
            PA=trans(p[3:6])@rx(p[6])@ry(p[7])@rz(p[8])
            PC=trans([0,p[9],0])@ry(p[10])
        else: raise ValueError('convention must be absolute or relative')
        return PY,PZ,PA,PC

    def subset(self,scheme):
        if scheme=='full': return self
        e=ErrorModel(convention=self.convention,origin=self.origin.copy())
        if scheme=='none': return e
        if scheme!='positioning': raise ValueError(scheme)
        for a in range(3): e.coeff[a,a]=self.coeff[a,a]
        e.coeff[3,3]=self.coeff[3,3]; e.coeff[4,5]=self.coeff[4,5]
        return e

def synthetic_model(scale=1.,seed=20261007):
    rng=np.random.default_rng(seed)
    e=ErrorModel()
    for a in range(5):
        for j in range(6):
            amp=(0.020 if j<3 else 25e-6)*scale
            e.coeff[a,j,1:]=amp*rng.normal(size=3)*[0.7,0.2,0.12]
    e.pige=np.array([35e-6,-20e-6,30e-6, .025,-.020,40e-6,-25e-6,.035,-.015,30e-6,45e-6])*scale
    return e

@dataclass
class Machine:
    model: ErrorModel = field(default_factory=ErrorModel)
    tool_home: np.ndarray = field(default_factory=lambda:np.array([0.,0.,150.]))
    part_home: np.ndarray = field(default_factory=lambda:np.array([0.,0.,0.]))

    def chains(self,q):
        x,y,z,a,c=np.asarray(q); e=self.model.evaluate(q)
        PY,PZ,PA,PC=self.model.pige_blocks()
        gw=trans([-x,0,0])@pdge_matrix(e[0])@PA@pdge_matrix(e[3])@rx(-a)@PC@pdge_matrix(e[4])@rz(-c)@trans(self.part_home)
        gt=PY@trans([0,y,0])@pdge_matrix(e[1])@PZ@trans([0,0,z])@pdge_matrix(e[2])@trans(self.tool_home)
        return gw,gt

    def matrix(self,q):
        gw,gt=self.chains(q); return inverse(gw)@gt

    def fk(self,q):
        g=self.matrix(q); return g[:3,3].copy(),g[:3,2].copy()

    def ideal(self): return Machine(ErrorModel(),self.tool_home.copy(),self.part_home.copy())

    def ik(self,p,o,previous=None,branch=1):
        """Closed form ideal IK, continuous branch selected near previous command."""
        o=np.asarray(o,dtype=float); n=np.linalg.norm(o)
        if n<1e-12: raise ValueError('Tool direction cannot be zero')
        o=o/n; r=np.hypot(o[0],o[1]); candidates=[]
        for sg in (1,-1):
            a=sg*np.arctan2(r,o[2])
            c=np.arctan2(sg*o[0],-sg*o[1]) if r>1e-10 else (previous[4] if previous is not None else 0.)
            if previous is not None: c+=2*np.pi*np.round((previous[4]-c)/(2*np.pi))
            xyz=(rx(-a)@rz(-c))[:3,:3]@(np.asarray(p)+self.part_home)-self.tool_home
            candidates.append(np.r_[xyz,a,c])
        if previous is None: return candidates[0 if branch>=0 else 1]
        return min(candidates,key=lambda q:np.linalg.norm((q-previous)/Q_SCALE))

    def jacobian(self,q,lever=100.):
        q=np.asarray(q,dtype=float); J=np.zeros((6,5))
        for j in range(5):
            h=1e-3 if j<3 else 1e-6; d=np.zeros(5); d[j]=h
            pp,op=self.fk(q+d); pm,om=self.fk(q-d)
            J[:,j]=np.r_[pp-pm,lever*(op-om)]/(2*h)
        return J

def angle_error(o,ref):
    return np.arctan2(np.linalg.norm(np.cross(o,ref),axis=-1),np.sum(o*ref,axis=-1))

def compensate(machine,p,o,q0=None,method='paper',iterations=5,tol_mm=1e-7,tol_rad=1e-9):
    ideal=machine.ideal(); q=ideal.ik(p,o) if q0 is None else np.array(q0,dtype=float)
    pc,oc=ideal.fk(q); hist=[]
    for k in range(iterations+1):
        pa,oa=machine.fk(q); ep=np.asarray(p)-pa; eo=np.asarray(o)-oa
        hist.append([np.linalg.norm(ep),float(angle_error(oa,np.asarray(o)))])
        if hist[-1][0]<tol_mm and hist[-1][1]<tol_rad: break
        if k==iterations: break
        if method=='paper':
            pc=pc+ep; oc=oc+eo; oc/=np.linalg.norm(oc)
            q=ideal.ik(pc,oc,previous=q)
        elif method=='damped':
            J=machine.jacobian(q)@np.diag(Q_SCALE)
            u,s,vt=np.linalg.svd(J,full_matrices=False)
            lam=max(1e-8,1e-4*s[0])
            dq=Q_SCALE*(vt.T@((s/(s*s+lam*lam))*(u.T@np.r_[ep,100*eo])))
            # Trust region and residual-decreasing backtracking.
            dq*=min(1.,0.1/max(np.linalg.norm(dq/Q_SCALE),1e-15))
            before=np.linalg.norm(np.r_[ep,100*eo])
            for fac in (1.,.5,.25,.125,.0625):
                pn,on=machine.fk(q+fac*dq)
                if np.linalg.norm(np.r_[p-pn,100*(o-on)])<before:
                    q+=fac*dq; break
            else: break
        else: raise ValueError(method)
    return q,np.asarray(hist)

def helix(n=121,turns=1.5):
    t=np.linspace(0,1,n); ph=2*np.pi*turns*t
    p=np.c_[45*np.cos(ph),45*np.sin(ph),20+60*t]
    a=np.deg2rad(32+12*np.sin(2*np.pi*t)); c=ph
    o=np.c_[np.sin(c)*np.sin(a),-np.cos(c)*np.sin(a),np.cos(a)]
    return t,p,o

def trajectory(machine,p,o,scheme='full',method='paper',iterations=5):
    predicted=Machine(machine.model.subset(scheme),machine.tool_home,machine.part_home)
    qs=[]; pp=[]; oo=[]; histories=[]; prev=None
    for pi,oi in zip(p,o):
        qi=machine.ideal().ik(pi,oi,previous=prev)
        if scheme!='none': qi,h=compensate(predicted,pi,oi,qi,method,iterations)
        else: h=np.array([[0.,0.]])
        pa,oa=machine.fk(qi); qs.append(qi); pp.append(pa); oo.append(oa); histories.append(h); prev=qi
    pp=np.asarray(pp); oo=np.asarray(oo)
    # Local normal-plane contour error, distinct from pointwise following error.
    tangent=np.gradient(p,axis=0); tangent/=np.linalg.norm(tangent,axis=1,keepdims=True)
    err=pp-p; normal=err-np.sum(err*tangent,axis=1,keepdims=True)*tangent
    return dict(q=np.asarray(qs),p=pp,o=oo,pos_um=1000*np.linalg.norm(err,axis=1),ori_urad=1e6*angle_error(oo,o),contour_um=1000*np.linalg.norm(normal,axis=1),history=histories)

def pcs_map(coeff,offset,x):
    """Eq.34 scalar datum subtraction; full pose frame mapping is inverse(Gbp)Gbt."""
    return np.polynomial.polynomial.polyval(offset+np.asarray(x),coeff)-np.polynomial.polynomial.polyval(offset,coeff)

def paden_kahan_orientation(u,v,wA,wC):
    """General intersecting two-axis direction problem, paper Eqs.37-41.
    Returns both feasible (a,c) branches; no position or joint-limit selection.
    """
    u,v,wA,wC=[np.asarray(x,dtype=float) for x in [u,v,wA,wC]]
    wA=wA/np.linalg.norm(wA);wC=wC/np.linalg.norm(wC)
    if abs(np.linalg.norm(u)-np.linalg.norm(v))>1e-8: raise ValueError('Direction norms must agree')
    d=wC@wA;den=1-d*d
    if den<1e-12: raise ValueError('Parallel rotary axes: singular subproblem')
    alpha=(d*(wA@u)-(wC@v))/(d*d-1)
    beta=(d*(wC@v)-(wA@u))/(d*d-1)
    rad=(u@u-alpha*alpha-beta*beta-2*alpha*beta*d)/den
    if rad<-1e-12:return []
    solutions=[]
    for gamma in [np.sqrt(max(rad,0)),-np.sqrt(max(rad,0))]:
        mid=alpha*wC+beta*wA+gamma*np.cross(wC,wA)
        up=u-wA*(wA@u);cp=mid-wA*(wA@mid)
        vp=v-wC*(wC@v);cq=mid-wC*(wC@mid)
        a=np.arctan2(wA@np.cross(up,cp),up@cp)
        c=-np.arctan2(wC@np.cross(vp,cq),vp@cq)
        solutions.append((float(a),float(c)))
    return solutions

def error_catalogue():
    return list(PDGE_NAMES)+list(ABS_PIGE_NAMES)
