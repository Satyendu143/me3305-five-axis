"""Periodic cubic-spline comparators, shared-nuisance TCP covariance.

These are implementations of established interpolation and uncertainty ideas,
not new algorithms. Covariance is first-order about nominal zero error maps.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from fiveaxis import ErrorModel,Machine,rx,rz
from periodic_model import RotaryModel,LEVER,NOISE,scales,unpack
from publication_studies import matrix,offsets

class PeriodicBasis:
    def __init__(self,knots=5):
        self.knots=knots;x=np.linspace(0,2*np.pi,knots+1)
        values=np.r_[np.eye(knots),np.eye(knots)[:1]]
        self.curve=CubicSpline(x,values,bc_type='periodic',axis=0)
    def __call__(self,c):
        # Drop one dependent coordinate and impose datum-zero runout.
        return (self.curve(np.mod(c,2*np.pi))-self.curve(0))[...,1:]

class SplineModel(RotaryModel):
    def __init__(self,knots=5):
        super().__init__();self.kind='spline';self.knots=knots
        self.spline=PeriodicBasis(knots);self.rotary=np.zeros((6,knots-1))
    def evaluate(self,q):
        e=ErrorModel.evaluate(self,q);e[4]=self.rotary@self.spline(q[4]);e[4,5]+=self.scale*q[4]
        return e

def spline_matrix(pool,knots=5):
    b=PeriodicBasis(knots);out=[];channel=np.array([.001]*3+[1e-5]*3)
    for a,c,L,H in pool:
        RA=rx(-a)[:3,:3];RC=rz(-c)[:3,:3];point=RC@np.array([L,0.,H])
        BC=np.c_[RA,np.column_stack([RA@np.cross(e,point) for e in np.eye(3)])]
        BC=np.r_[BC,np.c_[np.zeros((3,3)),LEVER*RA]]
        old=matrix(np.array([[a,c,L,H]]))[0];bas=b(c)
        block=np.concatenate([BC[:,j,None]*bas*channel[j] for j in range(6)],axis=1)
        out.append(np.c_[old[:,:18],block,old[:,-1:]])
    return np.array(out)

def spline_from_z(z,pige,linear,knots=5):
    m=SplineModel(knots);channel=np.array([.001]*3+[1e-5]*3)
    scale=np.r_[np.repeat(channel,3),np.repeat(channel,knots-1),1e-5]
    physical=z*scale;m.coeff[3,:,1:]=physical[:18].reshape(6,3)
    m.rotary=physical[18:-1].reshape(6,knots-1);m.scale=float(physical[-1])
    m.pige=np.array(pige);m.coeff[:3]=linear;return m

PIGE_SCALE=np.array([1e-5]*3+[.001]*3+[1e-5]*3+[.001,1e-5])
PIGE_SD=np.array([.5e-6]*3+[.0002]*3+[.5e-6]*3+[.0002,.5e-6])
LINEAR_SCALE=np.tile(np.repeat([.001]*3+[1e-5]*3,3),3)
ETA_SCALE=np.r_[PIGE_SCALE,LINEAR_SCALE]

def auxiliary_covariance():
    C=np.zeros((65,65));C[:11,:11]=np.diag((PIGE_SD/PIGE_SCALE)**2)
    u=np.linspace(-1,1,51);B=np.c_[u,u*u,u**3];inv=np.linalg.inv(B.T@B)
    for axis in range(3):
        for ch in range(6):
            j=11+(axis*6+ch)*3;sd=.0002 if ch<3 else .5e-6
            scale=.001 if ch<3 else 1e-5;C[j:j+3,j:j+3]=(sd/scale)**2*inv
    return C

def perturb_model(z,eta):
    m=unpack(z,RotaryModel());phys=np.asarray(eta)*ETA_SCALE
    m.pige=phys[:11];m.coeff[:3,:,1:]=phys[11:].reshape(3,6,3);return m

def tcp_jacobians(q,step=.001,tool_length=150.):
    # Central differences in dimensionless coefficient units, at zero maps.
    q=np.asarray(q);H=np.zeros((len(q),3,43));G=np.zeros((len(q),3,65))
    for j in range(108):
        z=np.zeros(43);eta=np.zeros(65)
        (z if j<43 else eta)[j if j<43 else j-43]=step
        plus=Machine(perturb_model(z,eta),tool_home=np.array([0.,0.,tool_length]))
        minus=Machine(perturb_model(-z,-eta),tool_home=np.array([0.,0.,tool_length]))
        diff=np.array([plus.fk(v)[0]-minus.fk(v)[0] for v in q])/(2*step)
        (H if j<43 else G)[:,:,j if j<43 else j-43]=diff
    return H,G

def offset_jacobian(pool,step=.001):
    D=np.zeros((len(pool),6,65))
    for j in range(11):
        eta=np.zeros(11);eta[j]=PIGE_SCALE[j]*step
        D[:,:,j]=(offsets(pool,eta)-offsets(pool,-eta))/(2*step)
    return D

def tcp_covariance(q,training_pool,fit,tool_length=150.):
    if fit.get('cov') is None: raise ValueError('TCP covariance requires an identifiable full-rank rotary fit')
    H,G=tcp_jacobians(q,tool_length=tool_length)
    D=offset_jacobian(training_pool);Dw=(D/NOISE[None,:,None]).reshape(-1,65)
    # The same noisy PIGE is subtracted during fitting and added at prediction.
    # Ignoring this covariance or treating these effects as independent is wrong.
    K=G-np.einsum('nij,jk->nik',H,fit['pinv']@Dw)
    rot=np.einsum('nij,jk,nlk->nil',H,fit['cov'],H)
    full=rot+np.einsum('nij,jk,nlk->nil',K,auxiliary_covariance(),K)
    return dict(rotary=rot,full=full,H=H,G=G,K=K,Dw=Dw)

def conformal_threshold(scores,alpha=.05):
    scores=np.asarray(scores,dtype=float)
    if scores.ndim!=1 or len(scores)==0 or not np.isfinite(scores).all():
        raise ValueError('Conformal scores must be a nonempty finite one-dimensional array')
    if not np.isfinite(alpha) or not 0<alpha<1: raise ValueError('alpha must lie strictly between zero and one')
    k=int(np.ceil((len(scores)+1)*(1-alpha)))
    return float(np.partition(scores,k-1)[k-1]) if k<=len(scores) else float('inf')

def squared_distance(error,covariance):
    return np.einsum('ni,ni->n',error,np.linalg.solve(covariance,error[...,None])[...,0])
