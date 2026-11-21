"""Disclosed teaching models of errors OUTSIDE the paper's 41 static channels.
Sensors are virtual; noise values are assumed, not manufacturer specifications.
Remedy estimators train on independent synthetic calibration data.
"""
import numpy as np

def rms(x): return float(np.sqrt(np.nanmean(np.asarray(x)**2)))

def thermal(noise_um=.5,temp_noise_K=.08,seed=71):
    rng=np.random.default_rng(seed)
    def temperatures(t,phase=0):
        return np.c_[4*(1-np.exp(-t/600))+np.sin(t/350+phase),
                     2*(1-np.exp(-t/900))+.5*np.cos(t/250+phase),
                     1.5*np.sin(t/650+phase)]
    B=np.array([[5.,-2.,1.],[1.,3.,-.8],[2.,1.,6.]]) # um/K, synthetic
    tr=np.linspace(0,3600,240); Ttr=temperatures(tr,.7)
    X=np.c_[np.ones(len(tr)),Ttr+rng.normal(0,temp_noise_K,Ttr.shape)]
    y=Ttr@B.T+rng.normal(0,noise_um,(len(tr),3))
    beta=np.linalg.lstsq(X,y,rcond=None)[0]
    t=np.linspace(0,3600,361); T=temperatures(t,1.5)
    Ts=T+rng.normal(0,temp_noise_K,T.shape); drift=T@B.T
    prediction=np.c_[np.ones(len(t)),Ts]@beta
    residual=drift-prediction
    return dict(t=t,T=T,sensors=Ts,before=drift,after=residual,prediction=prediction,beta=beta,train_condition=float(np.linalg.cond(X)),rms_before=rms(drift),rms_after=rms(residual))

def play(command,width):
    command=np.asarray(command); out=np.zeros_like(command); out[0]=command[0]
    for k in range(1,len(command)):
        out[k]=np.clip(out[k-1],command[k]-width/2,command[k]+width/2)
    return out

def backlash(width_um=24.,noise_um=.3,seed=72):
    rng=np.random.default_rng(seed); width=width_um/1000
    # Calibration trianglar ramp; no truth is handed to the estimator.
    cmd=np.r_[np.linspace(0,20,201),np.linspace(20,-20,401),np.linspace(-20,0,201)]
    pos=play(cmd,width); sensor=pos+rng.normal(0,noise_um/1000,len(pos))
    direction=np.sign(np.gradient(cmd)); d=cmd-sensor
    estimate=max(0.,np.median(d[direction>0])-np.median(d[direction<0]))
    t=np.linspace(0,10,1001); desired=25*np.sin(2*np.pi*t/5)
    actual=play(desired,width)
    vc=np.sign(np.gradient(desired)); compensated_command=desired+estimate/2*vc
    corrected=play(compensated_command,width)
    scale_before=actual+rng.normal(0,noise_um/1000,len(t))
    scale_after=corrected+rng.normal(0,noise_um/1000,len(t))
    # Initial condition is a real transient; preserve it in the exported data.
    return dict(t=t,desired=desired,before=1000*(actual-desired),after=1000*(corrected-desired),sensor_before=1000*(scale_before-desired),sensor_after=1000*(scale_after-desired),estimated_width_um=estimate*1000,rms_before=rms(1000*(actual-desired)),rms_after=rms(1000*(corrected-desired)))

def lag(command,tau,dt,initial=None):
    command=np.asarray(command,dtype=float)
    alpha=1-np.exp(-dt/tau); y=np.empty_like(command); y[0]=command[0] if initial is None else initial
    for k in range(1,len(command)): y[k]=(1-alpha)*y[k-1]+alpha*command[k]
    return y

def servo(tau_ms=20.,noise_um=.2,seed=73):
    rng=np.random.default_rng(seed); dt=.001; tau=tau_ms/1000
    tr=np.arange(0,2,dt); command=10*(tr>.5); plant=lag(command,tau,dt)
    sensed=plant+rng.normal(0,noise_um/1000,len(tr))
    r=command[1:]-sensed[:-1]; v=sensed[1:]-sensed[:-1]
    # Fit alpha only on the independent transient; avoid accumulated noise in dwell.
    ix=(tr[1:]>.5)&(tr[1:]<.75)
    alpha=np.clip(np.dot(r[ix],v[ix])/np.dot(r[ix],r[ix]),1e-5,.99)
    estimate=-dt/np.log(1-alpha)
    t=np.arange(0,5,dt); desired=25*np.sin(2*np.pi*t/2)
    before=lag(desired,tau,dt)
    ff=desired.copy(); ff[1:]=(desired[1:]-(1-alpha)*desired[:-1])/alpha
    after=lag(ff,tau,dt)
    return dict(t=t,desired=desired,before=1000*(before-desired),after=1000*(after-desired),sensor_before=1000*(before-desired)+rng.normal(0,noise_um,len(t)),sensor_after=1000*(after-desired)+rng.normal(0,noise_um,len(t)),estimated_tau_ms=estimate*1000,rms_before=rms(1000*(before-desired)),rms_after=rms(1000*(after-desired)))

def abbe(offset_mm=120.,angular_noise_urad=.3,seed=74):
    rng=np.random.default_rng(seed); x=np.linspace(-200,200,401)
    target=8*np.sin(x/80); theta=30*np.sin(x/100)+10*np.cos(x/55) # um, urad
    encoder=target+offset_mm*theta/1000+rng.normal(0,.2,len(x))
    angle_sensor=theta+rng.normal(0,angular_noise_urad,len(x))
    corrected=encoder-offset_mm*angle_sensor/1000
    collocated=target+rng.normal(0,.2,len(x))
    return dict(x=x,theta=theta,angle_sensor=angle_sensor,target=target,before=encoder-target,after=corrected-target,collocated=collocated-target,rms_before=rms(encoder-target),rms_after=rms(corrected-target))

def sensor_faults(noise_um=.5,seed=75):
    rng=np.random.default_rng(seed); t=np.linspace(0,60,601); truth=6*np.sin(t/8)+.1*t
    sensors=truth[:,None]+rng.normal(0,noise_um,(len(t),3))
    sensors[180:260,0]+=15; sensors[350:390,0]=np.nan
    raw=sensors[:,0]; median=np.nanmedian(sensors,axis=1)
    # A sensor plausibility alarm, not a guaranteed physical diagnosis.
    alarm=np.abs(raw-median)>max(3*noise_um,2.)
    return dict(t=t,truth=truth,sensors=sensors,before=raw-truth,after=median-truth,alarm=alarm,rms_before=rms(raw-truth),rms_after=rms(median-truth))

REMEDY_TEXT={
    'Geometry':'Laser displacement + straightness/angular optics; rotary multi-pose ballbar or 3D R-test; fit an observable map and compensate all axes.',
    'Thermal':'Measure temperatures at independent heat sources and TCP drift during calibration; fit a thermal model, validate on a different heating cycle, then feed forward. Correlated temperatures cannot identify separate coefficients.',
    'Backlash':'Use bidirectional sweeps and an external linear scale; estimate play width and apply direction-aware inverse, or preload/repair the transmission. A motor encoder alone misses downstream lost motion.',
    'Servo lag':'Use drive encoders plus synchronized TCP metrology; identify delay/dynamics then use feedforward or reduce feed. Respect actuator limits; the simple first-order model is educational.',
    'Abbe offset':'Co-locate the measurement line with the TCP, or add angular metrology and subtract lever-arm times measured angle. Angular noise is amplified by the offset.',
    'Sensor fault':'Use independent redundant channels, median/plausibility tests and recalibration. A common bias shared by every sensor remains invisible.',
    'Singularity':'Check the task Jacobian, preserve branch continuity, reorient the workpiece/tool path away from A=0, and use damped updates. Adding sensors does not create an actuator degree of freedom.',
    'Poor calibration':'Increase height separation H1-H, use nonzero radial lever L, and cover varied angles. Report rank and covariance; a tiny fit residual alone does not imply identifiable parameters.'}
