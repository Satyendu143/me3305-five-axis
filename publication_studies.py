"""Frozen numerical preprint protocol, independent held-out evaluation.

No hardware measurements or new-algorithm claim. NumPy/SciPy-free fits and
greedy task I-optimality; design uses a declared task distribution, never truth.
Run: OPENBLAS_NUM_THREADS=1 python publication_studies.py
"""
from pathlib import Path
import argparse,csv,json,time,platform,sys,itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fiveaxis import ErrorModel,Machine,compensate,angle_error,rx,rz,helix
from independent_plant import IndependentPlant
from periodic_model import (RotaryModel,NOISE,LEVER,basis,sensitivity,scales,
    candidates,select_poses,rtest_signal,auxiliary_calibration,rotation_vector_small,unpack)
from dynamic_sim import run_integrated,geometric_commands,calibrate_dynamic

ROOT=Path(__file__).resolve().parent;RES=ROOT/'results';FIG=ROOT/'report/figures'
METHODS=['space_filling','random','d_optimal','task_i_optimal']
CONFIGS=[dict(name='compact',fixture_scale=.7,tool_length=100.),
         dict(name='reference',fixture_scale=1.,tool_length=150.),
         dict(name='extended',fixture_scale=1.3,tool_length=220.)]
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'savefig.dpi':180})

def savefig(name,fig):
    fig.tight_layout();fig.savefig(FIG/(name+'.png'),bbox_inches='tight');plt.close(fig)

def matrix(pool,kind='periodic',order=2,orientation=True):
    if kind!='wrapped_cubic':return np.array([sensitivity(*r,kind=kind,order=order,orientation=orientation) for r in pool])
    result=[]
    for a,c,L,H in pool:
        # Continuous, datum-zero wrapped cubic runout: two free coefficients
        # per channel. Endpoint constraints separate runout from angle scale.
        wrapped=float(np.mod(c,2*np.pi))
        v=wrapped/(2*np.pi);bc=np.array([v*(1-v),v*(1-v)*(2*v-1)])
        RA=rx(-a)[:3,:3];RC=rz(-c)[:3,:3];point=RC@np.array([L,0.,H])
        BC=np.c_[RA,np.column_stack([RA@np.cross(e,point) for e in np.eye(3)])]
        if orientation:BC=np.r_[BC,np.c_[np.zeros((3,3)),LEVER*RA]]
        channels=np.array([.001]*3+[1e-5]*3)
        cubic=sensitivity(a,c,L,H,kind='cubic',orientation=orientation)
        cblock=np.concatenate([BC[:,j,None]*bc*channels[j] for j in range(6)],axis=1)
        scale=sensitivity(a,c,L,H,orientation=orientation)[:,-1:]
        result.append(np.c_[cubic[:,:18],cblock,scale])
    return np.array(result)

class WrappedCubic(RotaryModel):
    """Gauge-fixed one-revolution polynomial lookup plus independent scale.

    Runout matches at the seam, but slopes generally do not. Two independent
    cubic basis functions per channel leave angle scale identifiable.
    """
    def __init__(self):
        super().__init__('cubic');self.kind='wrapped_cubic';self.rotary=np.zeros((6,2))
    def evaluate(self,q):
        e=ErrorModel.evaluate(self,q);c=float(np.mod(q[4],2*np.pi));v=c/(2*np.pi)
        e[4]=self.rotary@np.array([v*(1-v),v*(1-v)*(2*v-1)]);e[4,5]+=self.scale*q[4]
        return e

def model_from_z(z,pige,linear,kind='periodic',order=2):
    if kind=='wrapped_cubic':
        m=WrappedCubic();physical=z*np.r_[scales('cubic')[:18],np.repeat([.001]*3+[1e-5]*3,2),1e-5]
        m.coeff[3,:,1:]=physical[:18].reshape(6,3);m.rotary=physical[18:30].reshape(6,2);m.scale=float(physical[-1])
    else:m=unpack(z,RotaryModel(kind,order))
    m.pige=np.array(pige);m.coeff[:3]=linear
    return m

def offsets(pool,pige):
    m=RotaryModel();m.pige=pige;_,_,PA,PC=m.pige_blocks();out=[]
    for a,c,L,H in pool:
        R0=(rx(-a)@rz(-c))[:3,:3];G=PA@rx(-a)@PC@rz(-c)
        out.append(np.r_[(G@np.array([L,0,H,1]))[:3]-R0@np.array([L,0,H]),LEVER*rotation_vector_small(G[:3,:3]@R0.T)])
    return np.array(out)

def solver(M,ids,sigma=NOISE):
    A=(M[ids]/sigma[None,:,None]).reshape(-1,M.shape[-1]);U,s,V=np.linalg.svd(A,full_matrices=False)
    rank=int(np.sum(s>s[0]*1e-10));pinv=np.linalg.pinv(A,rcond=1e-10)
    covariance=pinv@pinv.T if rank==M.shape[-1] else None
    return dict(A=A,pinv=pinv,rank=rank,cov=covariance,condition=float(s[0]/s[-1]) if rank==M.shape[-1] else None)

def task_select(pool,task,budget,orientation=True):
    # Rescale blocks for conditioning only; final inference remains unregularized.
    sigma=NOISE if orientation else NOISE[:3]
    B=matrix(pool,orientation=orientation)/sigma[None,:,None]*.0002
    H=matrix(task)[:,:3];C=np.einsum('nij,nik->jk',H,H)/len(task)
    F=np.eye(B.shape[-1])*1e-10;available=np.ones(len(pool),bool);chosen=[]
    for _ in range(budget):
        inv=np.linalg.solve(F,np.eye(F.shape[0]));U=np.einsum('ij,nkj->nik',inv,B)
        G=np.eye(B.shape[1])[None]+np.einsum('nij,njk->nik',B,U)
        # Woodbury variance reduction tr(C U (I+B F^-1 B^T)^-1 U^T).
        V=np.einsum('nji,jk,nkl->nil',U,C,U)
        gain=np.trace(np.linalg.solve(G,V),axis1=1,axis2=2)
        gain[~available]=-np.inf;j=int(np.argmax(gain));chosen.append(j);available[j]=False
        F+=B[j].T@B[j]
    return np.array(chosen)

def cluster_interval(values,seed=710):
    v=np.asarray(values);rng=np.random.default_rng(seed)
    boot=v[rng.integers(0,len(v),(2000,len(v)))].mean(axis=1)
    return [float(x) for x in np.percentile(boot,[2.5,97.5])]

def write_csv(name,rows):
    with (RES/name).open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def design_panel(seeds=12,repeats=8):
    rows=[];biasrows=[];budgets=[16,28,48];rng=np.random.default_rng(6191)
    task0=np.c_[rng.uniform(-np.pi/3,np.pi/3,192),rng.uniform(0,2*np.pi,192),rng.uniform(70,140,192),rng.uniform(35,110,192)]
    hold0=np.c_[rng.uniform(-np.pi/3,np.pi/3,128),rng.uniform(0,2*np.pi,128),rng.uniform(70,140,128),rng.uniform(35,110,128)]
    # Task and held-out geometry distributions match, but coordinates are disjoint.
    for ci,config in enumerate(CONFIGS):
        pool=candidates();pool[:,2:]*=config['fixture_scale'];task=task0.copy();task[:,2:]*=config['fixture_scale']
        held=hold0.copy();held[:,2:]*=config['fixture_scale'];M=matrix(pool);H=matrix(held)
        orders={'space_filling':select_poses(pool,48,'uniform'),'d_optimal':select_poses(pool,48),
                'task_i_optimal':task_select(pool,task,48)}
        designs={}
        for budget,method,trial in itertools.product(budgets,METHODS,range(repeats)):
            ids=orders[method][:budget] if method!='random' else select_poses(pool,budget,'random',trial+430)
            designs[budget,method,trial]=(ids,solver(M,ids))
        np.savez_compressed(RES/f'publication_design_{config["name"]}.npz',pool=pool,task=task,heldout=held,**orders)
        for strength in [0.,1.,3.]:
            for si in range(seeds):
                plant=IndependentPlant(831+si,mismatch=strength,tool_length=config['tool_length'])
                # Same PIGE realization within a seed, different between seeds.
                plant.pige*=np.random.default_rng(231+si).uniform(.8,1.2,11)
                signal=rtest_signal(plant,pool);truth=rtest_signal(plant,held)
                trueoff=offsets(pool,plant.pige);trueheldoff=offsets(held,plant.pige)
                for trial in range(repeats):
                    # Resample shared auxiliary metrology as well as rotary noise.
                    pige,linear=auxiliary_calibration(plant,seed=900000+1000*ci+20*si+trial)
                    off=offsets(pool,pige);holdoff=offsets(held,pige)
                    eps=np.random.default_rng(55000+20*si+trial).normal(size=signal.shape)*NOISE
                    for budget,method in itertools.product(budgets,METHODS):
                        ids,f=designs[budget,method,trial]
                        z=f['pinv']@((signal[ids]+eps[ids]-off[ids])/NOISE).ravel()
                        pred=np.einsum('nij,j->ni',H,z)+holdoff
                        rms=float(1000*np.sqrt(np.mean(np.sum((pred[:,:3]-truth[:,:3])**2,axis=1))))
                        zp=f['pinv']@((signal[ids]-trueoff[ids])/NOISE).ravel()
                        bias=np.einsum('nij,j->ni',H,zp)+trueheldoff-truth
                        brms=float(1000*np.sqrt(np.mean(np.sum(bias[:,:3]**2,axis=1))))
                        covariance=f['cov'];std=None if covariance is None else float(1000*np.sqrt(np.trace(np.einsum('nij,jk,nlk->il',H[:,:3],covariance,H[:,:3]))/len(held)))
                        rows.append(dict(config=config['name'],mismatch=strength,truth_seed=831+si,noise_trial=trial,
                            poses=budget,scalar_readings=6*budget,method=method,rms_um=rms,noise_only_rms_um=std,
                            noiseless_rotary_bias_um=brms,rank=f['rank'],condition=f['condition']))
                print('design',config['name'],'mismatch',strength,'seed',831+si,flush=True)
    write_csv('publication_design_trials.csv',rows)
    aggregate=[]
    for strength,budget,method in itertools.product([0.,1.,3.],budgets,METHODS):
        subset=[r for r in rows if r['mismatch']==strength and r['poses']==budget and r['method']==method]
        # Cluster on truth seed; all three configurations and repeated noise remain nested.
        cluster=[np.mean([r['rms_um'] for r in subset if r['truth_seed']==831+si]) for si in range(seeds)]
        aggregate.append(dict(mismatch=strength,poses=budget,method=method,mean_rms_um=float(np.mean(cluster)),
            seed_cluster_ci95=cluster_interval(cluster),noise_only_rms_um=float(np.mean([r['noise_only_rms_um'] for r in subset])),
            bias_um=float(np.mean([r['noiseless_rotary_bias_um'] for r in subset])),min_rank=min(r['rank'] for r in subset)))
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,strength in zip(axes,[0.,1.,3.]):
        for method in METHODS:
            rs=[r for r in aggregate if r['mismatch']==strength and r['method']==method]
            ax.errorbar([6*r['poses'] for r in rs],[r['mean_rms_um'] for r in rs],
                yerr=np.array([[r['mean_rms_um']-r['seed_cluster_ci95'][0],r['seed_cluster_ci95'][1]-r['mean_rms_um']] for r in rs]).T,
                fmt='o-',capsize=2,label=method.replace('_',' '),markersize=4)
        ax.set_title(f'Omitted-error multiplier {strength:g}');ax.set_xlabel('Scalar readings');ax.set_ylabel('Held-out displacement RMS (µm)')
    axes[0].legend(fontsize=7);savefig('publication_design_panel',fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for method in METHODS:
        rs=[r for r in aggregate if r['poses']==48 and r['method']==method]
        axes[0].plot([r['mismatch'] for r in rs],[r['bias_um'] for r in rs],'o-',label=method.replace('_',' '))
        axes[1].plot([r['mismatch'] for r in rs],[r['noise_only_rms_um'] for r in rs],'o-',label=method.replace('_',' '))
    axes[0].set_ylabel('Noiseless rotary bias RMS (µm)');axes[1].set_ylabel('Conditional rotary-noise RMS (µm)')
    for ax in axes:ax.set_xlabel('Omitted-error multiplier');ax.legend(fontsize=7)
    savefig('publication_bias_variance',fig)
    paired=[]
    for strength,budget in itertools.product([0.,1.,3.],budgets):
        for baseline in ['space_filling','d_optimal','random']:
            deltas=[]
            for si in range(seeds):
                a=[r['rms_um'] for r in rows if r['mismatch']==strength and r['poses']==budget and r['method']=='task_i_optimal' and r['truth_seed']==831+si]
                b=[r['rms_um'] for r in rows if r['mismatch']==strength and r['poses']==budget and r['method']==baseline and r['truth_seed']==831+si]
                deltas.append(float(np.mean(a)-np.mean(b)))
            paired.append(dict(mismatch=strength,poses=budget,comparison='task_i_optimal minus '+baseline,
                delta_um=float(np.mean(deltas)),seed_cluster_ci95=cluster_interval(deltas)))
    return dict(aggregate=aggregate,paired=paired,records=len(rows),seeds=seeds,repeats=repeats,
                protocol='3 configurations x 3 mismatch levels x seeds x 3 budgets x 4 methods x noise repeats; auxiliary and rotary noise resampled; seed-cluster percentile bootstrap, 2000 resamples')

def representation_panel(seeds=12):
    rows=[];examples={}
    for config in CONFIGS:
        pool=candidates();pool[:,2:]*=config['fixture_scale'];ids=select_poses(pool,60)
        fitcache={}
        for kind,order in [('cubic',2),('wrapped_cubic',2),('periodic',2),('periodic',4)]:
            fitcache[kind,order]=(matrix(pool,kind,order),solver(matrix(pool,kind,order),ids))
            assert fitcache[kind,order][1]['rank']==fitcache[kind,order][0].shape[-1], 'Representation fit is not identifiable'
        for si in range(seeds):
            plant=IndependentPlant(831+si,tool_length=config['tool_length']);pige,linear=auxiliary_calibration(plant,seed=720+si)
            sig=rtest_signal(plant,pool);obs=sig+np.random.default_rng(170+si).normal(size=sig.shape)*NOISE;off=offsets(pool,pige)
            models={}
            for (kind,order),(M,f) in fitcache.items():
                z=f['pinv']@((obs[ids]-off[ids])/NOISE).ravel()
                models[(kind+str(order)) if kind=='periodic' else kind]=model_from_z(z,pige,linear,kind,order)
            for turns in [1,3]:
                _,p,o=helix(121,turns);p*=config['fixture_scale']
                q0=[];ideal=Machine(tool_home=plant.tool_home);prev=None
                for pi,oi in zip(p,o):prev=ideal.ik(pi,oi,previous=prev);q0.append(prev)
                for name,model in [('uncompensated',None)]+list(models.items()):
                    q=np.array(q0) if model is None else geometric_commands(model,p,o,plant.tool_home,plant.part_home)
                    values=[plant.fk(v) for v in q];pa=np.array([v[0] for v in values]);oa=np.array([v[1] for v in values])
                    pos=1000*np.linalg.norm(pa-p,axis=1);orient=1e6*angle_error(oa,o)
                    rows.append(dict(config=config['name'],truth_seed=831+si,turns=turns,model=name,
                        parameters=0 if model is None else (36 if name=='cubic' else 31 if name=='wrapped_cubic' else 43 if name=='periodic2' else 67),
                        tcp_rms_um=float(np.sqrt(np.mean(pos**2))),tcp_max_um=float(pos.max()),axis_rms_urad=float(np.sqrt(np.mean(orient**2)))))
                    if config['name']=='reference' and si==0 and turns==3:examples[name]=(np.rad2deg(q[:,4]),pos)
            print('representation',config['name'],831+si,flush=True)
    write_csv('publication_representation_trials.csv',rows)
    aggregate=[]
    for turns,name in itertools.product([1,3],['uncompensated','cubic','wrapped_cubic','periodic2','periodic4']):
        rs=[r for r in rows if r['turns']==turns and r['model']==name]
        cluster=[np.mean([r['tcp_rms_um'] for r in rs if r['truth_seed']==831+si]) for si in range(seeds)]
        aggregate.append(dict(turns=turns,model=name,mean_rms_um=float(np.mean(cluster)),seed_cluster_ci95=cluster_interval(cluster),
            mean_max_um=float(np.mean([r['tcp_max_um'] for r in rs])),parameters=rs[0]['parameters']))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,(c,p) in examples.items():
        axes[0].semilogy(c,np.maximum(p,1e-5),label=name)
        if name!='cubic':axes[1].plot(c,p,label=name)
    axes[0].set_ylabel('TCP error (µm, log scale)');axes[1].set_ylabel('TCP error (µm)')
    axes[0].set_title('All representations');axes[1].set_title('Bounded baselines and harmonic ablation')
    for ax in axes:ax.set_xlabel('Unwrapped C (deg)');ax.legend(fontsize=7)
    savefig('publication_wrapped_baseline',fig)
    return dict(aggregate=aggregate,records=len(rows),seeds=seeds,protocol='3 configurations x seeds x 2 held-out revolutions x 5 schemes; 60 common calibration poses; paired auxiliary/noise seeds')

def uncertainty_and_sensors(repeats=400):
    pool=candidates();ids=select_poses(pool,48);M=matrix(pool);f=solver(M,ids);rng=np.random.default_rng(905)
    held=np.c_[rng.uniform(-np.pi/3,np.pi/3,128),rng.uniform(0,2*np.pi,128),rng.uniform(70,140,128),rng.uniform(35,110,128)]
    H=matrix(held)[:,:3];C=H.reshape(-1,43);cov=C@f['cov']@C.T
    eps=rng.normal(size=(6*len(ids),repeats));Z=f['pinv']@eps;prediction=C@Z
    sd=np.std(prediction,axis=1,ddof=1);pred=np.sqrt(np.maximum(np.diag(cov),0))
    # Joint xyz ellipsoid uses whitened quadratic distance (95% chi-square df=3).
    coverage=[]
    for i in range(len(held)):
        Ci=H[i]@f['cov']@H[i].T;E=H[i]@Z
        coverage.extend(np.einsum('it,ij,jt->t',E,np.linalg.inv(Ci),E)<7.814727903)
    result=dict(matched_linear_noise_trials=repeats,relative_sd_error=float(np.linalg.norm(sd-pred)/np.linalg.norm(pred)),
        joint_95_coverage=float(np.mean(coverage)),scope='matched first-order model and rotary noise only, not full physical/model-discrepancy uncertainty')
    nonlinear_coverage=[]
    for strength in [0.,1.,3.]:
        plant=IndependentPlant(mismatch=strength);signals=rtest_signal(plant,pool);truth=rtest_signal(plant,held)
        cov3=np.array([h@f['cov']@h.T for h in H]);inv3=np.linalg.inv(cov3)
        for auxiliary in [False,True]:
            indicators=[]
            for trial in range(repeats):
                pige=plant.pige if not auxiliary else auxiliary_calibration(plant,seed=83000+trial)[0]
                off=offsets(pool[ids],pige);ho=offsets(held,pige)
                eps=np.random.default_rng(17000+trial).normal(size=(len(ids),6))*NOISE
                z=f['pinv']@((signals[ids]+eps-off)/NOISE).ravel()
                error=np.einsum('nij,j->ni',H,z)+ho[:,:3]-truth[:,:3]
                d2=np.einsum('ni,nij,nj->n',error,inv3,error)
                indicators.extend(d2<7.814727903)
            nonlinear_coverage.append(dict(mismatch=strength,auxiliary_resampled=auxiliary,
                joint_95_coverage=float(np.mean(indicators))))
    result['nonlinear_conditional_coverage']=nonlinear_coverage
    fig,ax=plt.subplots(figsize=(7,4))
    for aux in [False,True]:
        rs=[r for r in nonlinear_coverage if r['auxiliary_resampled']==aux]
        ax.plot([r['mismatch'] for r in rs],[100*r['joint_95_coverage'] for r in rs],'o-',
            label='Auxiliary resampled' if aux else 'Exact auxiliary maps')
    ax.axhline(95,color='black',ls='--',label='Nominal 95%');ax.set_ylim(0,101)
    ax.set_xlabel('Omitted-error multiplier');ax.set_ylabel('Conditional ellipsoid coverage (%)');ax.legend(fontsize=8)
    savefig('publication_mismatch_coverage',fig)
    fig,ax=plt.subplots(figsize=(6,5));ax.scatter(pred*1000,sd*1000,s=10,alpha=.5);lim=max(pred.max(),sd.max())*1000
    ax.plot([0,lim],[0,lim],'k--');ax.set_xlabel('Predicted marginal SD (µm)');ax.set_ylabel('Monte Carlo SD (µm)')
    ax.set_title(f'{repeats} matched-linear trials; 95% joint coverage {100*result["joint_95_coverage"]:.1f}%')
    savefig('publication_uncertainty_control',fig)
    sensor=[]
    for readings in [96,168,288]:
        for orientation in [False,True]:
            pp=readings//(6 if orientation else 3);chosen=task_select(pool,held,pp,orientation)
            Mm=matrix(pool,orientation=orientation);ff=solver(Mm,chosen,NOISE if orientation else NOISE[:3])
            std=float(1000*np.sqrt(np.trace(C@ff['cov']@C.T)/len(held))) if ff['cov'] is not None else None
            sensor.append(dict(scalar_readings=readings,sensor='position_plus_angles' if orientation else 'position_only',
                poses=pp,rank=ff['rank'],noise_only_rms_um=std))
    result['sensor_budget']=sensor
    fig,ax=plt.subplots(figsize=(7,4))
    for label in ['position_only','position_plus_angles']:
        rs=[r for r in sensor if r['sensor']==label];ax.plot([r['scalar_readings'] for r in rs],[r['noise_only_rms_um'] for r in rs],'o-',label=label.replace('_',' '))
    ax.set_xlabel('Scalar readings');ax.set_ylabel('Predicted displacement-vector noise RMS (µm)');ax.legend(fontsize=8)
    savefig('publication_sensor_budget',fig)
    np.savez_compressed(RES/'publication_uncertainty_control.npz',predicted_sd=pred,monte_carlo_sd=sd,parameter_noise=Z,heldout=held)
    return result

def dynamic_panel():
    from research_experiments import load_models
    model=load_models()['periodic'];plant=IndependentPlant();rows=[];convergence=[]
    for mode in ['physical','accelerated']:
        for duration,turns in [(30.,2.5),(60.,1.25)]:
            r=run_integrated(plant,model,duration=duration,n=int(duration/.01)+1,turns=turns,thermal_mode=mode,full_factorial=True,servo_integrator='exact_zoh')
            for name,v in r['schemes'].items():
                rows.append(dict(thermal_mode=mode,duration_s=duration,turns=turns,scheme=name,
                    tcp_rms_um=float(np.sqrt(np.mean(v['pos_um']**2))),contour_rms_um=float(np.sqrt(np.mean(v['contour_um']**2))),
                    axis_rms_urad=float(np.sqrt(np.mean(v['ori_urad']**2))),max_command_jump_linear_mm=float(abs(np.diff(v['q_command'][:,:3],axis=0)).max())))
            key=f'{mode}_{int(duration)}'
            np.savez_compressed(RES/f'publication_dynamic_{key}.npz',t=r['t'],minutes=r['minutes'],RTD=r['RTD'],
                thermal_true=r['thermal_true'],thermal_estimate=r['thermal_estimate'],probe_error_um=r['probe_error_um'],
                **{name+'__'+field:v[field] for name,v in r['schemes'].items() for field in ['pos_um','contour_um','q_command','q_motor','output_encoder']})
            fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
            for name in ['uncompensated','geometry','geometry_thermal','geometry_backlash','geometry_ff','all_remedies']:
                v=r['schemes'][name];axes[0].plot(r['t'],v['pos_um'],label=name.replace('_',' '),lw=.9)
            for j in range(3):
                axes[1].plot(r['t'],r['RTD'][:,j],label=f'RTD {j+1}')
                axes[2].plot(r['t'],r['probe_error_um'][:,j],label=f'Assessment probe {"xyz"[j]}',lw=.7)
            axes[0].set_ylabel('TCP error (µm)');axes[1].set_ylabel('Δ temperature (K)');axes[2].set_ylabel('Probe residual (µm)')
            axes[0].set_title(f'{mode} thermal clock; {duration:g} s / {turns:g} turns / Δt=0.01 s')
            for ax in axes:ax.legend(fontsize=7,ncol=3)
            axes[-1].set_xlabel('Mechanical time (s)');savefig('publication_dynamic_'+key,fig)
            print('dynamic',key,flush=True)
    # Fixed parameter estimates re-discretized at each evaluation dt; do not
    # re-identify sensors when checking the integration grid.
    base=calibrate_dynamic(dt=.005,servo_integrator='exact_zoh');tau=-.005/np.log1p(-base['alpha'])
    common_noise=np.random.default_rng(880).normal(0,.06,(24001,3))
    from dynamic_sim import path
    _,pt,ot=path(24001);qfine=geometric_commands(model,pt,ot);ideal=Machine();qnom=[];prev=None
    for pi,oi in zip(pt,ot):prev=ideal.ik(pi,oi,previous=prev);qnom.append(prev)
    qnom=np.array(qnom)
    for mode in ['physical','accelerated']:
        for dt in [.04,.02,.01,.005,.0025,.00125]:
            cal=dict(base);cal['dt']=dt;cal['alpha']=1-np.exp(-dt/tau)
            step=int(round(dt/.00125))
            r=run_integrated(plant,model,n=int(30/dt)+1,thermal_mode=mode,calibration=cal,
                rtd_noise=common_noise[::step],servo_integrator='exact_zoh',prepared_commands=(qfine[::step],qnom[::step]))
            for name in ['uncompensated','all_remedies']:
                v=r['schemes'][name];convergence.append(dict(thermal_mode=mode,dt_s=dt,scheme=name,
                    tcp_rms_um=float(np.sqrt(np.mean(v['pos_um']**2))),contour_rms_um=float(np.sqrt(np.mean(v['contour_um']**2)))))
            print('grid',mode,dt,flush=True)
    write_csv('publication_dynamic_ablations.csv',rows);write_csv('publication_grid.csv',convergence)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,mode in zip(axes,['physical','accelerated']):
        for name in ['uncompensated','all_remedies']:
            rs=[r for r in convergence if r['thermal_mode']==mode and r['scheme']==name]
            ax.plot([r['dt_s'] for r in rs],[r['tcp_rms_um'] for r in rs],'o-',label=name)
        ax.set_xlabel('Sampling interval (s)');ax.set_ylabel('TCP RMS (µm)');ax.set_title(mode+' thermal clock');ax.legend(fontsize=8)
    savefig('publication_sampling_convergence',fig)
    return dict(ablations=rows,grid=convergence,scope='Exact ZOH two-pole plant; two clocks, two paths, 8 geometry-present thermal/play/feedforward combinations plus uncompensated; no saturation or collision model')

def run(seeds=12,repeats=8,skip_dynamic=False):
    start=time.time();RES.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    out=dict(scope='Synthetic numerical benchmark; no empirical validation; fixed XYZ-AC topology',
        configurations=CONFIGS,seeds=seeds,noise_repeats=repeats,design=design_panel(seeds,repeats),
        representation=representation_panel(seeds),uncertainty=uncertainty_and_sensors())
    if not skip_dynamic:out['dynamic']=dynamic_panel()
    out['wall_seconds']=time.time()-start
    (RES/'publication_metrics.json').write_text(json.dumps(out,indent=2,allow_nan=False))
    (RES/'publication_environment.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,platform=platform.platform()),indent=2))
    print('Protocol complete:',out['wall_seconds'],'seconds',flush=True);return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=12);p.add_argument('--repeats',type=int,default=8);p.add_argument('--skip-dynamic',action='store_true');p.add_argument('--uncertainty-only',action='store_true');p.add_argument('--dynamic-only',action='store_true');p.add_argument('--representation-only',action='store_true');a=p.parse_args()
    if a.uncertainty_only:
        out=json.loads((RES/'publication_metrics.json').read_text());out['uncertainty']=uncertainty_and_sensors()
        (RES/'publication_metrics.json').write_text(json.dumps(out,indent=2,allow_nan=False))
    elif a.dynamic_only:
        out=json.loads((RES/'publication_metrics.json').read_text());out['dynamic']=dynamic_panel()
        (RES/'publication_metrics.json').write_text(json.dumps(out,indent=2,allow_nan=False))
    elif a.representation_only:
        out=json.loads((RES/'publication_metrics.json').read_text());out['representation']=representation_panel(out['seeds'])
        (RES/'publication_metrics.json').write_text(json.dumps(out,indent=2,allow_nan=False))
    else:run(a.seeds,a.repeats,a.skip_dynamic)
