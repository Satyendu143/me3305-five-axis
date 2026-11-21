"""Coupled time-domain experiment. Sensor-based controller, independent plant.

All schemes share a path, environment and plant. Calibration and evaluation
records are disjoint. No truth coefficient is read by the remedy estimator.
Feedforward has no closed-loop claim, and commands are simulation-only.
"""
import numpy as np
from fiveaxis import Machine,compensate,angle_error,rx,rz

TAU1=np.array([.012,.016,.014,.019,.025]);TAU2=np.array([.003,.004,.003,.005,.004])
WIDTH=np.array([.024,.018,.020,18e-6,24e-6])
THERMAL_B=np.array([[.005,-.002,.001],[.001,.003,-.0008],[.002,.001,.006]])

def thermal_record(minutes,phase):
    t=np.asarray(minutes)
    return np.c_[4*(1-np.exp(-t/18))+1.1*np.sin(t/8+phase),
                 2*(1-np.exp(-t/12))+.8*np.sin(t/13+.4+phase),
                 3*(1-np.exp(-t/24))+.7*np.sin(t/6+.9+phase)]

def thermal_truth(T,dt_min):
    lag=np.empty_like(T);lag[0]=T[0];alpha=1-np.exp(-dt_min/2.4)
    for k in range(1,len(T)):lag[k]=lag[k-1]+alpha*(T[k]-lag[k-1])
    # Static sensor map is misspecified: heat lag and nonlinear cross term.
    d=lag@THERMAL_B.T
    d[:,2]+=.0004*T[:,0]*T[:,2]
    return d

def servo_truth(u,dt,variable=False,integrator='sequential'):
    x1=np.array(u[0]);x2=x1.copy();out=np.empty_like(u);out[0]=x2
    for k in range(1,len(u)):
        factor=1+.12*np.sin(2*np.pi*k/len(u)) if variable else 1.
        tau1=TAU1*factor
        a=1-np.exp(-dt/tau1);b=1-np.exp(-dt/TAU2)
        if integrator=='exact_zoh':
            old1=x1.copy();old2=x2.copy();e1=1-a;e2=1-b
            cross=tau1/(tau1-TAU2)*(e1-e2)
            x1=u[k]+(old1-u[k])*e1
            x2=u[k]+(old2-u[k])*e2+(old1-u[k])*cross
        elif integrator=='sequential':
            x1+=a*(u[k]-x1);x2+=b*(x1-x2)
        else:raise ValueError(integrator)
        out[k]=x2
    return out

def backlash_truth(u,variable=False):
    y=np.empty_like(u);y[0]=u[0]
    for k in range(1,len(u)):
        w=WIDTH*(1+.12*np.sin(np.r_[u[k,:3]/40,u[k,3:]])) if variable else WIDTH
        y[k]=np.clip(y[k-1],u[k]-w/2,u[k]+w/2)
    return y

def calibrate_dynamic(dt=.01,seed=992,servo_integrator='sequential'):
    rng=np.random.default_rng(seed)
    # Thermal training: disjoint 60-minute warm-up, RTDs + displacement probe.
    tm=np.linspace(0,60,601);T=thermal_record(tm,.3)
    d=thermal_truth(T,tm[1]-tm[0]);sensorT=T+rng.normal(0,.06,T.shape)
    sensorD=d+rng.normal(0,.0005,d.shape)
    X=np.c_[np.ones(len(tm)),sensorT];beta=np.linalg.lstsq(X,sensorD,rcond=None)[0]
    # Servo identification with motor-side encoder, before mechanical play.
    ts=np.arange(0,8,dt);u=np.sin(2*np.pi*.8*ts)[:,None]*np.array([4,3,5,.08,.13])
    u+=.25*np.sin(2*np.pi*2.1*ts)[:,None]*np.array([4,3,5,.08,.13])
    motor=servo_truth(u,dt,integrator=servo_integrator);enc=motor+rng.normal(size=motor.shape)*np.array([.0002]*3+[.3e-6]*2)
    r=u[1:]-enc[:-1];v=enc[1:]-enc[:-1]
    alpha=np.clip(np.sum(r*v,axis=0)/np.sum(r*r,axis=0),.02,.99)
    # Slow reversible scan with independent output scale/rotary encoder.
    z=np.r_[np.linspace(0,1,301),np.linspace(1,-1,601),np.linspace(-1,0,301)]
    sweep=z[:,None]*np.array([10,9,11,.3,.4]);load=backlash_truth(sweep)
    obs=load+rng.normal(size=load.shape)*np.array([.0003]*3+[.4e-6]*2)
    direction=np.sign(np.gradient(sweep,axis=0));offset=sweep-obs
    width=np.array([np.median(offset[direction[:,j]>0,j])-np.median(offset[direction[:,j]<0,j]) for j in range(5)])
    return dict(beta=beta,alpha=alpha,width=width,dt=dt,
        thermal_time=tm,thermal_T=sensorT,thermal_displacement=sensorD,
        servo_time=ts,servo_command=u,servo_encoder=enc,
        backlash_command=sweep,backlash_encoder=obs)

def path(n=1501,duration=30.,turns=2.5):
    t=np.linspace(0,duration,n);s=t/duration;ph=2*np.pi*turns*s
    p=np.c_[35*np.cos(ph)+5*np.sin(3*ph),30*np.sin(ph),20+35*s+4*np.sin(2*ph)]
    a=np.deg2rad(35+14*np.sin(2*np.pi*s))
    o=np.c_[np.sin(ph)*np.sin(a),-np.cos(ph)*np.sin(a),np.cos(a)]
    return t,p,o

def geometric_commands(model,p,o,tool_home=None,part_home=None):
    ideal=Machine();estimated=Machine(model)
    if tool_home is not None:ideal.tool_home=np.array(tool_home);estimated.tool_home=np.array(tool_home)
    if part_home is not None:ideal.part_home=np.array(part_home);estimated.part_home=np.array(part_home)
    q=[];prev=None
    for pi,oi in zip(p,o):
        nominal=ideal.ik(pi,oi,previous=prev)
        qi,_=compensate(estimated,pi,oi,nominal,iterations=3)
        q.append(qi);prev=qi
    return np.array(q)

def run_integrated(plant,model,duration=30.,n=1501,seed=801,turns=2.5,calibration=None,
                   thermal_mode='accelerated',full_factorial=False,rtd_noise=None,
                   servo_integrator='sequential',prepared_commands=None):
    t,p,o=path(n,duration,turns);dt=t[1]-t[0]
    cal=calibrate_dynamic(dt,servo_integrator=servo_integrator) if calibration is None else calibration
    if abs(cal['dt']-dt)>1e-12:raise ValueError('Servo calibration sampling period mismatch')
    rng=np.random.default_rng(seed)
    # Thermal process evolves through a prewarmed 15--45 minute scenario while
    # the mechanical test takes duration seconds: explicitly accelerated clock.
    if thermal_mode not in ('accelerated','physical'):raise ValueError(thermal_mode)
    # Physical-time mode evolves exactly duration/60 minutes, not 30 minutes.
    minutes=np.linspace(15,45 if thermal_mode=='accelerated' else 15+duration/60,n);T=thermal_record(minutes,1.9)
    d=thermal_truth(T,minutes[1]-minutes[0])
    noise=rng.normal(0,.06,T.shape) if rtd_noise is None else np.asarray(rtd_noise)
    if noise.shape!=T.shape:raise ValueError('RTD noise record shape mismatch')
    RTD=T+noise
    predicted=np.c_[np.ones(n),RTD]@cal['beta']
    # Low-pass noisy RTD prediction before differentiating in servo inverse.
    filt=1-np.exp(-dt/(.02/-np.log(.92)))
    for k in range(1,n):predicted[k]=filt*predicted[k]+(1-filt)*predicted[k-1]
    if prepared_commands is None:
        qc=geometric_commands(model,p,o,plant.tool_home,plant.part_home)
        ideal=Machine();qn=[];prev=None
        ideal.tool_home=plant.tool_home.copy();ideal.part_home=plant.part_home.copy()
        for pi,oi in zip(p,o):prev=ideal.ik(pi,oi,previous=prev);qn.append(prev)
    else:
        qc,qn=prepared_commands
        if np.shape(qc)!=(n,5) or np.shape(qn)!=(n,5):raise ValueError('Prepared commands shape')
    schemes=['uncompensated','geometry','geometry_thermal','geometry_thermal_backlash','all_remedies']
    flags={name:(i>0,i>=2,i>=3,i>=4) for i,name in enumerate(schemes)}
    if full_factorial:
        flags={'uncompensated':(False,False,False,False)}
        for th in (False,True):
            for play in (False,True):
                for ff in (False,True):
                    name='geometry'+('_thermal' if th else '')+('_backlash' if play else '')+('_ff' if ff else '')
                    flags[name]=(True,th,play,ff)
        flags['all_remedies']=flags.pop('geometry_thermal_backlash_ff')
    results={}
    for name,(geo,th,play,ff) in flags.items():
        desired=np.array(qn) if not geo else qc.copy()
        if th:
            # Table-frame target correction is first-order in estimated MCS drift;
            # ideal translation chain maps MCS drift to [-dx,dy,dz] drives.
            desired[:,:3]-=predicted*np.array([-1,1,1])
        if play:
            # Inverse constant-width play; hold previous sign at zero increments.
            sign=np.sign(np.diff(desired,axis=0,prepend=desired[:1]))
            for k in range(1,n):sign[k]=np.where(sign[k]==0,sign[k-1],sign[k])
            desired+=sign*cal['width']/2
        u=desired.copy()
        if ff:u[1:]=desired[:-1]+np.diff(desired,axis=0)/cal['alpha']
        motor=servo_truth(u,dt,variable=True,integrator=servo_integrator);drive=backlash_truth(motor,variable=True)
        poses=[plant.fk(q,thermal=di) for q,di in zip(drive,d)]
        pa=np.array([x[0] for x in poses]);oa=np.array([x[1] for x in poses]);error=pa-p
        tangent=np.gradient(p,axis=0);tangent/=np.linalg.norm(tangent,axis=1,keepdims=True)
        normal=error-np.sum(error*tangent,axis=1,keepdims=True)*tangent
        results[name]=dict(q_command=u,q_motor=motor,q_load=drive,p=pa,o=oa,
            error_vector_um=1000*error,pos_um=1000*np.linalg.norm(error,axis=1),
            contour_um=1000*np.linalg.norm(normal,axis=1),ori_urad=1e6*angle_error(oa,o))
    # Assessment probe is not connected to controller; encoder traces represent
    # motor/output separation, unlike an unrealistically omniscient feedback loop.
    encoder_noise=np.array([.0003]*3+[.4e-6]*2)
    for v in results.values():v['output_encoder']=v['q_load']+rng.normal(size=v['q_load'].shape)*encoder_noise
    probe=results['all_remedies']['error_vector_um']+rng.normal(0,.5,(n,3))
    return dict(t=t,p_target=p,o_target=o,minutes=minutes,RTD=RTD,thermal_true=d,
                thermal_estimate=predicted,probe_error_um=probe,schemes=results,calibration=cal)
