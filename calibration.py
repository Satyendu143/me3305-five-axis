"""Ballbar/R-test synthetic calibration; paper Eqs.17-29,44-45.
All PDGE cubic functions are gauge-fixed to zero at angle=0.
The relative eight-PIGE convention is used ONLY in this calibration module.
"""
import numpy as np
from functools import lru_cache
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from fiveaxis import rx,ry,rz,trans,pdge_matrix

P8=('dxAX','dyAX','dzAX','alphaAX','betaAX','gammaAX','dyCA','betaCA')
S8=np.array([.001,.001,.001,1e-5,1e-5,1e-5,.001,1e-5])
S44=np.r_[S8,np.tile(np.repeat([.001,.001,.001,1e-5,1e-5,1e-5],3),2)]

def table_exact(a,c,L,H,theta=None):
    b=np.array([L,0.,H,1.])
    if theta is None: return (rx(-a)@rz(-c)@b)[:3]
    p=theta[:8]; co=theta[8:].reshape(2,6,3)
    da=co[0]@np.array([a/(np.pi/2),(a/(np.pi/2))**2,(a/(np.pi/2))**3])
    dc=co[1]@np.array([c/np.pi,(c/np.pi)**2,(c/np.pi)**3])
    PA=trans(p[:3])@rx(p[3])@ry(p[4])@rz(p[5])
    PC=trans([0,p[6],0])@ry(p[7])
    return (PA@pdge_matrix(da)@rx(-a)@PC@pdge_matrix(dc)@rz(-c)@b)[:3]

def table_sensitivity(a,c,L,H):
    RA=rx(-a)[:3,:3]; RC=rz(-c)[:3,:3]; b=np.array([L,0.,H])
    bc=RC@b; ba=RA@bc; e=np.eye(3)
    M=np.zeros((3,44)); M[:,:3]=e
    for j in range(3): M[:,3+j]=np.cross(e[j],ba)
    M[:,6]=RA@e[1]; M[:,7]=RA@np.cross(e[1],bc)
    for axis,(u,B,pos) in enumerate([(a/(np.pi/2),ba,np.eye(3)),(c/np.pi,bc,RA)]):
        local=np.column_stack([pos[:,j] for j in range(3)]+[pos@np.cross(e[j],B) for j in range(3)])
        for j in range(6):
            M[:,8+axis*18+j*3:8+axis*18+j*3+3]=local[:,j,None]*np.array([u,u*u,u*u*u])
    return M

def directions(axis,a,c):
    if axis=='A':
        s=np.sin(-a); co=np.cos(-a)
        return np.array([[1.,0,0],[0,-s,co],[0,co,s]])
    s=np.sin(-c); co=np.cos(-c)
    return np.array([[0,0,1.],[co,s,0],[-s,co,0]])

def design(sensor='ballbar',H=50.,H1=100.,L=100.,n=25):
    """Each record is (axis,a,c,L,H,direction). Includes mixed A=-45 C sweeps."""
    rows=[]; records=[]
    configs=[('A',lh,0.) for lh in [(0.,H),(0.,H1),(L,H)]]
    configs += [('C',lh,a) for a in [0.,-np.pi/4] for lh in [(L,H),(1.5*L,H),(L,H1)]]
    for axis,(li,hi),fixed in configs:
        angles=np.linspace(-np.pi/3,np.pi/6,n) if axis=='A' else np.linspace(0,2*np.pi,n)
        for angle in angles:
            a,c=(angle,0.) if axis=='A' else (fixed,angle)
            ns=directions(axis,a,c) if sensor=='ballbar' else np.eye(3)
            M=table_sensitivity(a,c,li,hi)
            for k,ni in enumerate(ns):
                rows.append(ni@M*S44); records.append((axis,a,c,li,hi,ni,k))
    return np.asarray(rows),records

def truth(seed=42):
    rng=np.random.default_rng(seed); z=rng.normal(size=44)
    z[:8]=[46,21,-12,3.,-4.,-3.,20.5,5.]
    z[8:]=z[8:]*np.tile(np.repeat([8.,8.,8.,2.,2.,2.],3),2)*np.tile([1.,.3,.1],12)
    return z

def readings(records,z,sensor='ballbar',noise_um=.5,seed=4,exact=True):
    rng=np.random.default_rng(seed)
    y=exact_predictions(records,z,sensor if exact else 'rtest')
    return y+rng.normal(0,noise_um/1000,len(y))

def exact_predictions(records,z,sensor='ballbar'):
    ar=np.array([[r[1],r[2],r[3],r[4]] for r in records]); a,c,L,H=ar.T
    ns=np.array([r[5] for r in records]); b=np.c_[L,0*L,H]
    p=z[:8]*S8; coeff=(z[8:]*S44[8:]).reshape(2,6,3)
    u=a/(np.pi/2); v=c/np.pi
    da=np.c_[u,u*u,u**3]@coeff[0].T; dc=np.c_[v,v*v,v**3]@coeff[1].T
    RA=Rotation.from_euler('x',-a[:,None]).as_matrix(); RC=Rotation.from_euler('z',-c[:,None]).as_matrix()
    RPA=Rotation.from_euler('XYZ',p[3:6]).as_matrix(); RPC=ry(p[7])[:3,:3]
    RDA=Rotation.from_euler('XYZ',da[:,3:]).as_matrix(); RDC=Rotation.from_euler('XYZ',dc[:,3:]).as_matrix()
    mul=lambda R,b:np.einsum('...ij,...j->...i',R,b)
    bc=mul(RC,b); nominal=mul(RA,bc)
    actual=p[:3]+mul(RPA,da[:,:3]+mul(RDA,mul(RA,np.array([0,p[6],0])+mul(RPC,dc[:,:3]+mul(RDC,bc)))))
    d=actual-nominal
    return np.linalg.norm(100*ns+d,axis=1)-100 if sensor=='ballbar' else np.sum(ns*d,axis=1)

def refine(records,y,z0,sensor='ballbar',noise_um=.5):
    result=least_squares(lambda z:exact_predictions(records,z,sensor)-y,z0,
                         xtol=1e-11,ftol=1e-11,gtol=1e-12,max_nfev=50)
    fit=solve(result.jac,y,noise_um)
    fit.update(z=result.x,residual_um=1000*result.fun,success=bool(result.success),nfev=result.nfev)
    return fit

def solve(A,y,noise_um=.5,method='joint',iterations=100):
    U,s,Vt=np.linalg.svd(A,full_matrices=False); cutoff=s[0]*1e-10
    rank=int(np.sum(s>cutoff))
    if method=='joint': z=np.linalg.lstsq(A,y,rcond=1e-10)[0]; convergence=[]
    else:
        AI,AD=A[:,:8],A[:,8:]; pi=np.linalg.pinv(AI); pd=np.linalg.pinv(AD)
        zI=pi@y; zD=pd@(y-AI@zI); convergence=[]
        if method=='alternating':
            for _ in range(iterations):
                zI=pi@(y-AD@zD); zD=pd@(y-AI@zI)
                convergence.append(float(np.linalg.norm(y-AI@zI-AD@zD)))
        elif method!='separate': raise ValueError(method)
        z=np.r_[zI,zD]
    inv=np.where(s>cutoff,1/np.maximum(s,cutoff)**2,0.)
    covariance=(noise_um/1000)**2*(Vt.T*inv)@Vt
    return dict(z=z,rank=rank,singular=s,condition=float(s[0]/s[-1]) if rank==A.shape[1] else float('inf'),covariance=covariance,residual_um=1000*(y-A@z),convergence=convergence)

def a_design(a,H,L):
    s=np.sin(-a); c=np.cos(-a)
    return np.array([[1,0,0,0,H*c,H*s],[0,-s,c,0,-L*c,-L*s],[0,c,s,-H,-L*s,L*c]],float)

def a_readings(a,H,H1,L,e):
    return np.array([a_design(a,hi,li)@e for hi,li in [(H,0),(H1,0),(H,L)]])

def a_corrected(read,a,H,H1,L):
    """Algebraic inversion of Eq.28; valid also at sin(a)=0, cos(a)=0."""
    if abs(H1-H)<1e-12 or abs(L)<1e-12: raise ValueError('Need H1!=H and L!=0')
    ra,rr,rt=read.T; s=np.sin(-a); c=np.cos(-a)
    ex=-(rt[1]-rt[0])/(H1-H)
    dy=-s*rr[0]+c*(rt[0]+H*ex)
    dz=c*rr[0]+s*(rt[0]+H*ex)
    dR=(rr[2]-rr[0])/L; dT=(rt[2]-rt[0])/L
    ey=-c*dR-s*dT; ez=-s*dR+c*dT
    dx=ra[0]-H*(c*ey+s*ez)
    return np.array([dx,dy,dz,ex,ey,ez])

def a_printed_eq29(read,a,H,H1,L):
    """Literal diagnostic implementation of PRINTED Eq.29, not used for fitting."""
    ra,rr,rt=read.T; s=np.sin(-a); c=np.cos(-a)
    with np.errstate(divide='ignore',invalid='ignore'):
        dz=(H*(rr[1]*c+rt[1]*s)+(H1-2*H)*(rr[0]*c+rt[0]*s))/(H1-H)
        ex=((rr[1]-rr[0])*c+(rt[1]-rt[0])*s)/((H1-H)*s)
        dy=(dz*c-rr[0])/s
        ez=(c*(rt[2]+H*ex)-s*rr[2]-dy)/L
        ey=-(rr[2]-dz*c+dy*s+L*ez*s)/(L*c)
        dx=ra[0]-H*ey*c-H*ez*s
    return np.array([dx,dy,dz,ex,ey,ez])

def eccentricity(angle,reading):
    """Fit offset+cos+sin, paper Step1 eccentricities Eq.20-23."""
    B=np.c_[np.ones_like(angle),np.cos(angle),np.sin(angle)]
    return np.linalg.lstsq(B,reading,rcond=None)[0]

def piges_from_eccentricities(eyRA,ezRA,eyAA,ezAA,exRC,eyRC,exAC,eyAC,H,L):
    if H==0 or L==0: raise ValueError('Nonzero H,L required')
    dy,dz=-eyRA,-ezRA; gamma=eyAA/H; beta=-ezAA/H
    alpha=-eyAC/L; betaCA=exAC/L-beta
    dx=-exRC-H*(betaCA+beta); dyCA=-eyRC-dy+H*alpha
    return np.array([dx,dy,dz,alpha,beta,gamma,dyCA,betaCA])
