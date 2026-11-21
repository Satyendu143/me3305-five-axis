"""Vectorized exact geometry and covariance ablations for the publication audit.

The vectorized evaluator is tested against the scalar machine implementation;
it accelerates propagation, and is not an additional independent truth plant.
"""
import numpy as np
from periodic_model import NOISE, LEVER, scales
from journal_models import ETA_SCALE, auxiliary_covariance, conformal_threshold

def _rotation(axis,angle,shape):
    a=np.broadcast_to(angle,shape);c=np.cos(a);s=np.sin(a)
    R=np.broadcast_to(np.eye(3),shape+(3,3)).copy()
    i,j=[(1,2),(2,0),(0,1)][axis]
    R[...,i,i]=c;R[...,j,j]=c;R[...,i,j]=-s;R[...,j,i]=s
    return R

def _start(shape):return np.broadcast_to(np.eye(3),shape+(3,3)).copy(),np.zeros(shape+(3,))
def _rotate(f,axis,angle):
    U,t=f;return U@_rotation(axis,angle,U.shape[:-2]),t

def _translate(f,t):
    U,v=f;return U,v+np.einsum('...ij,...j->...i',U,np.broadcast_to(t,v.shape))
def _err(f,e):
    f=_translate(f,e[...,:3])
    for j in range(3):f=_rotate(f,j,e[...,j+3])
    return f

def physical_errors(z,eta,q):
    z=np.atleast_2d(z);eta=np.atleast_2d(eta);q=np.asarray(q)
    if z.shape[1]!=43 or eta.shape!=(len(z),65) or q.ndim!=2 or q.shape[1]!=5:raise ValueError('Expected z[S,43], eta[S,65], q[N,5]')
    a=z*scales();b=eta*ETA_SCALE;E=np.zeros((len(z),len(q),5,6))
    u=q[:,:3]/200;B=np.stack([u,u*u,u*u*u],axis=-1)
    E[:,:,:3]=np.einsum('saij,naj->snai',b[:,11:].reshape(-1,3,6,3),B)
    u=q[:,3]/(np.pi/2);B=np.c_[u,u*u,u*u*u]
    E[:,:,3]=np.einsum('sij,nj->sni',a[:,:18].reshape(-1,6,3),B)
    c=q[:,4];B=np.column_stack([v for k in [1,2] for v in [np.sin(k*c),np.cos(k*c)-1]])
    E[:,:,4]=np.einsum('sij,nj->sni',a[:,18:-1].reshape(-1,6,4),B)
    E[:,:,4,5]+=a[:,-1,None]*c
    return E,b[:,:11]

def _table(E,p,a,c,initial=None):
    shape=E.shape[:2];f=_start(shape) if initial is None else initial
    f=_translate(f,p[:,None,3:6])
    for j in range(3):f=_rotate(f,j,p[:,None,6+j])
    f=_err(f,E[:,:,3]);f=_rotate(f,0,-a)
    t=np.zeros(shape+(3,));t[...,1]=p[:,None,9];f=_translate(f,t)
    f=_rotate(f,1,p[:,None,10]);f=_err(f,E[:,:,4]);return _rotate(f,2,-c)

def batch_fk(z,eta,q,tool_length=150.):
    q=np.asarray(q);E,p=physical_errors(z,eta,q);shape=E.shape[:2]
    w=_start(shape);t=np.zeros(shape+(3,));t[...,0]=-q[:,0];w=_translate(w,t);w=_err(w,E[:,:,0]);w=_table(E,p,q[:,3],q[:,4],w)
    f=_rotate(_start(shape),2,p[:,None,0]);t*=0;t[...,1]=q[:,1];f=_translate(f,t);f=_err(f,E[:,:,1])
    f=_rotate(f,0,p[:,None,1]);f=_rotate(f,1,-p[:,None,2]);t*=0;t[...,2]=q[:,2];f=_translate(f,t);f=_err(f,E[:,:,2]);f=_translate(f,np.array([0.,0.,tool_length]))
    return np.einsum('snji,snj->sni',w[0],f[1]-w[1])

def batch_signal(z,eta,pool):
    pool=np.asarray(pool);q=np.zeros((len(pool),5));q[:,3:]=pool[:,:2]
    E,p=physical_errors(z,eta,q);U,v=_table(E,p,q[:,3],q[:,4]);b=np.c_[pool[:,2],np.zeros(len(pool)),pool[:,3]]
    R0=_rotation(0,-pool[:,0],(len(pool),))@_rotation(2,-pool[:,1],(len(pool),))
    pos=np.einsum('snij,nj->sni',U,b)+v-np.einsum('nij,nj->ni',R0,b)
    R=U@R0.transpose(0,2,1);w=np.stack([R[...,2,1]-R[...,1,2],R[...,0,2]-R[...,2,0],R[...,1,0]-R[...,0,1]],axis=-1)/2
    n=np.linalg.norm(w,axis=-1);factor=np.ones_like(n);np.divide(np.arcsin(np.clip(n,0,1)),n,out=factor,where=n>1e-12)
    return np.concatenate([pos,LEVER*w*factor[...,None]],axis=-1)

def batch_offset(eta,pool):return batch_signal(np.zeros((len(eta),43)),eta,pool)

def covariance_variants(base,pinv,channel_correlation):
    """Actual OLS covariance under block-correlated raw rotary readings.

    The deliberately naive comparator preserves Cov(z) and Cov(eta) but sets
    Cov(z,eta) to zero; it isolates the missing cross-covariance exactly.
    """
    P=pinv.reshape(pinv.shape[0],-1,6)
    Cz=np.einsum('ink,kl,jnl->ij',P,channel_correlation,P)
    H,G,K=base['H'],base['G'],base['K'];C=auxiliary_covariance();B=G-K
    rot=np.einsum('nij,jk,nlk->nil',H,Cz,H)
    joint=rot+np.einsum('nij,jk,nlk->nil',K,C,K)
    naive=rot+np.einsum('nij,jk,nlk->nil',G,C,G)+np.einsum('nij,jk,nlk->nil',B,C,B)
    return dict(rotary=rot,joint=joint,naive=naive)

def many_scores(error,cov):
    inv=np.linalg.inv(cov);return np.sqrt(np.maximum(np.einsum('sni,nij,snj->sn',error,inv,error),0))

def many_threshold(scores,alpha=.05):
    scores=np.asarray(scores)
    if scores.ndim!=2:raise ValueError('Expected [draw,calibration sample] scores')
    # Delegate validation and order convention to the public implementation.
    return np.array([conformal_threshold(s,alpha) for s in scores])
