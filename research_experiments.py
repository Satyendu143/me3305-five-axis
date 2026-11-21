"""Four research upgrades: run, export numerical evidence, generate paper figures.

Usage: python research_experiments.py --trials 24 --points 1501
Run the legacy experiments.py separately to regenerate the reproduction supplement.
"""
from pathlib import Path
import json,argparse,csv,copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fiveaxis import Machine,helix,angle_error
from independent_plant import IndependentPlant
from periodic_model import (candidates,select_poses,fit,rtest_signal,auxiliary_calibration,
                            sensitivity,NOISE,RotaryModel,scales)
from dynamic_sim import run_integrated,geometric_commands

ROOT=Path(__file__).resolve().parent
FIG=ROOT/'report/figures';RES=ROOT/'results'
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.22,'savefig.dpi':180})

def savefig(name,fig):
    fig.tight_layout();fig.savefig(FIG/f'{name}.png',bbox_inches='tight');plt.close(fig)

def summary(v):
    v=np.asarray(v);return dict(rms=float(np.sqrt(np.mean(v*v))),max=float(np.max(v)),p95=float(np.percentile(v,95)))

def static_test(plant,model,turns,n=241,compensated=True):
    t,p,o=helix(n,turns)
    if compensated:q=geometric_commands(model,p,o)
    else:
        ideal=Machine();q=[];prev=None
        for pi,oi in zip(p,o):prev=ideal.ik(pi,oi,previous=prev);q.append(prev)
        q=np.array(q)
    actual=[plant.fk(v) for v in q];pa=np.array([v[0] for v in actual]);oa=np.array([v[1] for v in actual])
    return dict(c_deg=np.rad2deg(q[:,4]),p=pa,q=q,pos_um=1000*np.linalg.norm(pa-p,axis=1),ori_urad=1e6*angle_error(oa,o))

def prediction_matrix(pool,kind='periodic'):
    return np.array([sensitivity(*r,kind=kind) for r in pool])

def build_studies(trials=24,points=1501):
    FIG.mkdir(parents=True,exist_ok=True);RES.mkdir(parents=True,exist_ok=True)
    plant=IndependentPlant();pool=candidates();picked=select_poses(pool,60)
    pige,linear=auxiliary_calibration(plant)
    signal=rtest_signal(plant,pool);noise=np.random.default_rng(77).normal(size=signal.shape)*NOISE
    models={};fits={}
    for kind in ['cubic','periodic']:
        models[kind],f=fit(pool[picked],(signal+noise)[picked],pige,linear,kind)
        fits[kind]={k:f[k] for k in ('rank','parameters','condition','residual_um')}
    # Persist model values so dashboard/reports never silently use fresh fits.
    modeldata={k:dict(kind=m.kind,order=m.order,coeff=m.coeff.tolist(),pige=m.pige.tolist(),
                      rotary=m.rotary.tolist(),scale=m.scale) for k,m in models.items()}
    (RES/'research_models.json').write_text(json.dumps(modeldata,indent=2))
    (RES/'independent_truth.json').write_text(json.dumps(plant.configuration(),indent=2))
    with (RES/'research_calibration.csv').open('w',newline='') as h:
        w=csv.writer(h);w.writerow(['a_rad','c_rad','L_mm','H_mm','dx_mm','dy_mm','dz_mm','rx_equiv_mm','ry_equiv_mm','rz_equiv_mm'])
        w.writerows(np.c_[pool[picked],(signal+noise)[picked]])
    static={};metrics={'scope':'Synthetic independent-plant numerical study; no hardware validation','trials':trials,'fits':fits}
    for turns,label in [(1,'one_revolution'),(3,'three_revolutions')]:
        static[label]={}
        for kind in ['uncompensated','cubic','periodic']:
            v=static_test(plant,models.get(kind,models['periodic']),turns,compensated=kind!='uncompensated')
            static[label][kind]=v;np.savez_compressed(RES/f'research_static_{label}_{kind}.npz',**v)
        metrics[label]={k:{field:summary(v[field]) for field in ('pos_um','ori_urad')} for k,v in static[label].items()}
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for j,label in enumerate(static):
        for k,v in static[label].items():
            axes[0,j].plot(v['c_deg'],v['pos_um'],label=k);axes[1,j].plot(v['c_deg'],v['ori_urad'],label=k)
        axes[0,j].set_title(label.replace('_',' '));axes[0,j].set_ylabel('TCP error (µm)')
        axes[1,j].set_ylabel('Tool-axis error (µrad)');axes[1,j].set_xlabel('Unwrapped C (deg)')
        axes[0,j].legend(fontsize=8)
    savefig('research_generalization',fig)
    c=np.linspace(0,6*np.pi,601);truth=np.array([plant.errors([0,0,0,0,x])[4] for x in c])
    fig,axes=plt.subplots(2,3,figsize=(12,6))
    for j,ax in enumerate(axes.flat):
        fac=1000 if j<3 else 1e6
        ax.plot(np.rad2deg(c),truth[:,j]*fac,color='black',label='Independent truth')
        for k,m in models.items():ax.plot(np.rad2deg(c),[m.evaluate([0,0,0,0,x])[4,j]*fac for x in c],label=k)
        ax.axvspan(0,360,color='green',alpha=.08);ax.set_title(('dx','dy','dz','rx','ry','rz')[j])
        ax.set_xlabel('C (deg)');ax.set_ylabel('µm' if j<3 else 'µrad')
    axes[0,0].legend(fontsize=8);savefig('research_periodic_maps',fig)

    # Independent implementation cross-check, not independent physical validation.
    rng=np.random.default_rng(192);qs=rng.uniform([-90,-90,-100,-1.0,-6], [90,90,0,1.0,10],(80,5))
    matched=[];unmatched=[];m=models['periodic'];mc=Machine(m)
    for q in qs:
        p1,o1=mc.fk(q);p2,o2=plant.fk(q,errors=m.evaluate(q),pige=m.pige)
        matched.append([np.linalg.norm(p1-p2)*1000,np.linalg.norm(o1-o2)*1e6])
        pt,ot=plant.fk(q);unmatched.append([np.linalg.norm(p1-pt)*1000,np.linalg.norm(o1-ot)*1e6])
    metrics['implementation_check']={'poses':80,'max_position_um':float(np.max(np.array(matched)[:,0])),
        'max_direction_difference_urad':float(np.max(np.array(matched)[:,1])),
        'truth_model_position_rms_um':float(np.sqrt(np.mean(np.array(unmatched)[:,0]**2)))}
    np.savez_compressed(RES/'research_independence.npz',q=qs,matched=matched,truth_model=unmatched)

    # Equal-budget design: same pool, same six readings/noise at each pose,
    # fixed separately measured auxiliary map, 24 repeated rotary noise trials.
    budgets=[16,28,48];methods=['uniform','random','d_optimal'];design=[]
    rng=np.random.default_rng(291)
    held=np.c_[rng.uniform(-np.pi/3,np.pi/3,96),rng.uniform(0,2*np.pi,96),rng.uniform(70,140,96),rng.uniform(35,110,96)]
    Hmat=prediction_matrix(held);trueheld=rtest_signal(plant,held)
    # Exact zero-PDGE PIGE baseline for held-out predictions, same as fit offset.
    zmodel=RotaryModel();zmodel.pige=pige
    from fiveaxis import rx,rz
    from periodic_model import LEVER,rotation_vector_small
    offsets=[]
    for a,c0,L,H in held:
        _,_,PA,PC=zmodel.pige_blocks();G=PA@rx(-a)@PC@rz(-c0);R0=(rx(-a)@rz(-c0))[:3,:3]
        offsets.append(np.r_[(G@np.array([L,0,H,1]))[:3]-R0@np.array([L,0,H]),LEVER*rotation_vector_small(G[:3,:3]@R0.T)])
    offsets=np.array(offsets)
    design_arrays=[]
    orders={k:select_poses(pool,max(budgets),k,seed=3) for k in ('uniform','d_optimal')}
    for budget in budgets:
        for method in methods:
            errors=[];ranks=[];conditions=[];predicted_stds=[];biases=[]
            for trial in range(trials):
                ids=(orders[method][:budget] if method!='random' else select_poses(pool,budget,method,seed=trial+3))
                trialnoise=np.random.default_rng(3000+trial).normal(size=signal.shape)*NOISE
                _,f=fit(pool[ids],(signal+trialnoise)[ids],pige,linear)
                pred=np.einsum('nij,j->ni',Hmat,f['z'])+offsets
                err=1000*np.linalg.norm(pred[:,:3]-trueheld[:,:3],axis=1)
                errors.append(float(np.sqrt(np.mean(err**2))));ranks.append(f['rank']);conditions.append(f['condition'])
                if f['rank']==f['parameters']:
                    cov=np.linalg.inv(f['A'].T@f['A'])
                    variance=np.einsum('nij,jk,nik->ni',Hmat[:,:3],cov,Hmat[:,:3])
                    predicted_stds.append(float(1000*np.sqrt(np.mean(np.sum(variance,axis=1)))))
                _,fz=fit(pool[ids],signal[ids],pige,linear)
                ez=np.einsum('nij,j->ni',Hmat,fz['z'])+offsets-trueheld
                biases.append(float(1000*np.sqrt(np.mean(np.sum(ez[:,:3]**2,axis=1)))))
                design_arrays.append([budget,methods.index(method),trial,errors[-1],f['rank']])
            row=dict(method=method,poses=budget,scalar_readings=6*budget,mean_rms_um=float(np.mean(errors)),
                sd_rms_um=float(np.std(errors,ddof=1)) if trials>1 else 0.,p05=float(np.percentile(errors,5)),p95=float(np.percentile(errors,95)),
                min_rank=min(ranks),full_rank_trials=int(np.sum(np.array(ranks)==43)),
                median_condition=float(np.median([x for x in conditions if x is not None])) if any(x is not None for x in conditions) else None,
                predicted_noise_rms_um=float(np.mean(predicted_stds)) if predicted_stds else None,
                noiseless_bias_rms_um=float(np.mean(biases)))
            design.append(row)
    metrics['design']=design
    np.savez_compressed(RES/'research_design.npz',trial_results=design_arrays,pool=pool,heldout=held,
        uniform=orders['uniform'],d_optimal=orders['d_optimal'])
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for method in methods:
        rows=[r for r in design if r['method']==method];x=[r['scalar_readings'] for r in rows]
        y=np.array([r['mean_rms_um'] for r in rows]);lo=[r['p05'] for r in rows];hi=[r['p95'] for r in rows]
        axes[0].plot(x,y,'o-',label=method);axes[0].fill_between(x,lo,hi,alpha=.12)
        axes[1].plot(x,[r['predicted_noise_rms_um'] for r in rows],'o-',label=method)
    axes[0].set_ylabel('Held-out displacement RMS (µm)');axes[1].set_ylabel('Predicted noise-only RMS (µm)')
    axes[0].set_title(f'{trials} noise trials; shaded 5–95% quantiles');axes[1].set_title('Full-rank linear covariance only')
    for ax in axes:ax.set_xlabel('Scalar readings (6 per pose)');ax.legend(fontsize=8)
    savefig('research_design',fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,method in zip(axes,['uniform','d_optimal']):
        ids=orders[method][:48];v=pool[ids]
        ax.scatter(np.rad2deg(v[:,1]),np.rad2deg(v[:,0]),s=v[:,2]/2,c=v[:,3],cmap='viridis',alpha=.75,edgecolors='black',linewidths=.4)
        ax.set_title(method.replace('_',' ')+'; size=radius, color=height');ax.set_xlabel('C (deg)');ax.set_ylabel('A (deg)')
    savefig('research_pose_selection',fig)
    # Coverage and sensor information are distinct: a single C sweep at A=0
    # has 18 zero A-map columns, which orientation sensors cannot recover.
    sweep=pool[pool[:,0]==0]
    ranks={};fig,ax=plt.subplots(figsize=(8,4.5))
    for label,pp,ori in [('single_tilt_position',sweep,False),('mixed_tilt_position',pool[picked],False),('mixed_tilt_augmented',pool[picked],True)]:
        A=np.concatenate([sensitivity(*r,orientation=ori) for r in pp]);sigma=NOISE if ori else NOISE[:3]
        A=A/np.tile(sigma,len(pp))[:,None];s=np.linalg.svd(A,compute_uv=False);rank=int(np.sum(s>s[0]*1e-10))
        ranks[label]=dict(rank=rank,parameters=43,condition=float(s[0]/s[-1]) if rank==43 else None)
        ax.semilogy(np.arange(1,44),np.maximum(s/s[0],1e-16),'o-',markersize=3,label=label.replace('_',' '))
    ax.set_xlabel('Singular-value index');ax.set_ylabel('Normalized singular value');ax.legend(fontsize=8)
    savefig('research_observability',fig);metrics['sensor_coverage']=ranks

    integrated=run_integrated(plant,models['periodic'],n=points)
    metrics['integrated']={k:{field:summary(v[field]) for field in ('pos_um','contour_um','ori_urad')} for k,v in integrated['schemes'].items()}
    metrics['dynamic_conditions']=dict(duration_s=30,points=points,dt_s=30/(points-1),turns=2.5,
        thermal_clock='Accelerated 15–45 min thermal history during 30 s mechanical trajectory',
        fit_alpha=integrated['calibration']['alpha'].tolist(),fit_width=integrated['calibration']['width'].tolist())
    raw={k:v for k,v in integrated.items() if isinstance(v,np.ndarray)}
    for k,v in integrated['schemes'].items():
        for field,arr in v.items():raw[f'{k}__{field}']=arr
    for k,v in integrated['calibration'].items():
        if isinstance(v,np.ndarray):raw['calibration__'+k]=v
    np.savez_compressed(RES/'research_integrated.npz',**raw)
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for k,v in integrated['schemes'].items():
        for ax,field in zip(axes,['pos_um','contour_um','ori_urad']):ax.plot(integrated['t'],v[field],label=k.replace('_',' '),linewidth=1)
    for ax,label in zip(axes,['TCP following error (µm)','Normal-plane contour error (µm)','Tool-axis error (µrad)']):ax.set_ylabel(label)
    axes[0].legend(fontsize=8,ncol=2);axes[-1].set_xlabel('Mechanical time (s)');savefig('research_integrated',fig)
    fig,axes=plt.subplots(3,1,figsize=(11,8))
    for j in range(3):axes[0].plot(integrated['minutes'],integrated['RTD'][:,j],label=f'RTD {j+1}')
    axes[0].set_ylabel('Temperature change (K)');axes[0].legend();axes[0].set_xlabel('Accelerated thermal time (min)')
    for j in range(3):
        axes[1].plot(integrated['t'],integrated['thermal_true'][:,j]*1000,label=f'truth {"xyz"[j]}')
        axes[1].plot(integrated['t'],integrated['thermal_estimate'][:,j]*1000,'--',label=f'RTD map {"xyz"[j]}')
    axes[1].set_ylabel('MCS thermal displacement (µm)');axes[1].legend(fontsize=8,ncol=3)
    for j in range(3):axes[2].plot(integrated['t'],integrated['probe_error_um'][:,j],label=f'Assessment probe {"xyz"[j]}',linewidth=.8)
    axes[2].set_ylabel('Post-remedy probe residual (µm)');axes[2].legend(fontsize=8,ncol=3);axes[2].set_xlabel('Mechanical time (s)')
    savefig('research_thermal_sensors',fig)
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
    for k in ['geometry','geometry_thermal_backlash','all_remedies']:
        v=integrated['schemes'][k]
        axes[0].plot(integrated['t'],(v['q_command'][:,0]-v['q_motor'][:,0])*1000,label=k.replace('_',' '))
        axes[1].plot(integrated['t'],(v['output_encoder'][:,0]-v['q_motor'][:,0])*1000,label=k.replace('_',' '),linewidth=.8)
    axes[0].set_ylabel('X command − motor (µm)');axes[1].set_ylabel('X output encoder − motor (µm)')
    axes[0].set_title('Feedforward changes command–motor lag; it does not remove mechanical play')
    axes[1].set_xlabel('Time (s)');axes[0].legend(fontsize=8);savefig('research_encoder_sensors',fig)

    # Second mechanical path to check duration/feed sensitivity, without refits.
    second=run_integrated(plant,models['periodic'],duration=60,n=2*(points-1)+1,turns=1.25,seed=802,
                          calibration=integrated['calibration'])
    metrics['second_path']={k:{field:summary(v[field]) for field in ('pos_um','contour_um','ori_urad')} for k,v in second['schemes'].items()}
    np.savez_compressed(RES/'research_second_path.npz',t=second['t'],**{k:v['pos_um'] for k,v in second['schemes'].items()})
    fig,ax=plt.subplots(figsize=(9,4))
    for k,v in second['schemes'].items():ax.plot(second['t'],v['pos_um'],label=k.replace('_',' '))
    ax.set_ylabel('TCP error (µm)');ax.set_xlabel('Time (s)');ax.set_title('Second path: 60 s, 1.25 C revolutions');ax.legend(fontsize=8,ncol=2)
    savefig('research_second_path',fig)
    (RES/'research_metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
    print(json.dumps({'static':metrics['three_revolutions'],'integrated':metrics['integrated'],'design':design},indent=2))
    return metrics

def load_models():
    data=json.loads((RES/'research_models.json').read_text());models={}
    for k,d in data.items():
        m=RotaryModel(d['kind'],d['order']);m.coeff=np.array(d['coeff']);m.pige=np.array(d['pige']);m.rotary=np.array(d['rotary']);m.scale=d['scale'];models[k]=m
    return models

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trials',type=int,default=24);p.add_argument('--points',type=int,default=1501);a=p.parse_args()
    build_studies(a.trials,a.points)
