"""Independent validation plant: NumPy only, direct vector/frame propagation.

No import from fiveaxis/calibration/periodic_model is permitted. Units mm, rad.
The truth uses sinusoidal linear errors, non-cubic A errors, four C harmonics,
and C scale drift. It is NOT the fitted model used to generate its own answers.
"""
import numpy as np

def rotation(j, a):
    c,s=np.cos(a),np.sin(a)
    if j==0:return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    if j==1:return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def compose(frame, R=None, t=None):
    U,v=frame
    return U@(np.eye(3) if R is None else R),v+U@(np.zeros(3) if t is None else t)

def error_step(frame,e):
    frame=compose(frame,t=e[:3])
    for j in range(3):frame=compose(frame,rotation(j,e[j+3]))
    return frame

class IndependentPlant:
    def __init__(self,seed=831,mismatch=1.,tool_length=150.):
        rng=np.random.default_rng(seed)
        self.linear=rng.normal(size=(3,6,3))*np.array([.010]*3+[15e-6]*3)[None,:,None]
        self.a_coeff=rng.normal(size=(6,3))*np.array([.012]*3+[18e-6]*3)[:,None]*[1,.25,.08]
        self.c_coeff=rng.normal(size=(6,8))*np.array([.010]*3+[15e-6]*3)[:,None]*np.tile([1,1,.4,.4,.06,.06,.025,.025],(6,1))
        self.c_scale=2.5e-6
        self.mismatch=float(mismatch)
        self.c_coeff[:,4:]*=self.mismatch
        self.pige=np.array([25e-6,-18e-6,21e-6,.032,.018,-.014,23e-6,-31e-6,19e-6,.024,28e-6])
        self.tool_home=np.array([0.,0.,float(tool_length)]);self.part_home=np.zeros(3)

    def errors(self,q):
        q=np.asarray(q);e=np.zeros((5,6))
        for i in range(3):
            u=q[i]/200
            e[i]=self.linear[i]@np.array([u,u*u,np.sin(2.1*u)])
        u=q[3]/(np.pi/2)
        e[3]=self.a_coeff@np.array([u,u*u,u**3])
        e[3]+=self.mismatch*np.array([.0007,-.0005,.0006,1e-6,-.8e-6,.6e-6])*(np.sin(2.7*u)-2.7*u)
        b=np.array([v for k in range(1,5) for v in (np.sin(k*q[4]),np.cos(k*q[4])-1)])
        e[4]=self.c_coeff@b;e[4,5]+=self.c_scale*q[4]
        return e

    def table(self,a,c,L,H,errors=None,pige=None,return_rotation=False):
        e=self.errors([0,0,0,a,c]) if errors is None else errors
        p=self.pige if pige is None else pige
        f=(np.eye(3),np.zeros(3))
        f=compose(f,t=p[3:6])
        for j in range(3):f=compose(f,rotation(j,p[6+j]))
        f=error_step(f,e[3]);f=compose(f,rotation(0,-a))
        f=compose(f,t=[0,p[9],0]);f=compose(f,rotation(1,p[10]))
        f=error_step(f,e[4]);f=compose(f,rotation(2,-c))
        point=f[0]@np.array([L,0,H])+f[1]
        return (point,f[0]) if return_rotation else point

    def fk(self,q,thermal=None,errors=None,pige=None):
        x,y,z,a,c=q;e=self.errors(q) if errors is None else errors
        p=self.pige if pige is None else pige
        w=(np.eye(3),np.array([-x,0.,0.]));w=error_step(w,e[0])
        w=compose(w,t=p[3:6])
        for j in range(3):w=compose(w,rotation(j,p[6+j]))
        w=error_step(w,e[3]);w=compose(w,rotation(0,-a))
        w=compose(w,t=[0,p[9],0]);w=compose(w,rotation(1,p[10]))
        w=error_step(w,e[4]);w=compose(w,rotation(2,-c));w=compose(w,t=self.part_home)
        t=(rotation(2,p[0]),np.zeros(3));t=compose(t,t=[0,y,0]);t=error_step(t,e[1])
        t=compose(t,rotation(0,p[1]));t=compose(t,rotation(1,-p[2]))
        t=compose(t,t=[0,0,z]);t=error_step(t,e[2]);t=compose(t,t=self.tool_home)
        if thermal is not None:t=(t[0],t[1]+np.asarray(thermal))
        return w[0].T@(t[1]-w[1]),w[0].T@t[0][:,2]

    def configuration(self):
        return {k:getattr(self,k).tolist() if isinstance(getattr(self,k),np.ndarray) else getattr(self,k)
                for k in ('linear','a_coeff','c_coeff','c_scale','pige','tool_home','part_home','mismatch')}
