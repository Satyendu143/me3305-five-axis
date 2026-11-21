"""Journal extensions: equal-capacity maps and measured-response coverage.

Frozen protocol: 12 seeds, two truth families, three configurations; a separate
12-seed / 100-noise-draw TCP coverage audit. All observations are synthetic.
"""
from pathlib import Path
import argparse,json,time,itertools,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fiveaxis import Machine,helix,angle_error
from independent_plant import IndependentPlant
from validation_plant import LocalizedPlant,MatchedPlant
from periodic_model import candidates,select_poses,rtest_signal,auxiliary_calibration,NOISE
from publication_studies import (CONFIGS,matrix,solver,offsets,model_from_z,
    cluster_interval,write_csv,savefig)
from dynamic_sim import geometric_commands
from journal_models import (spline_matrix,spline_from_z,tcp_covariance,
    conformal_threshold,squared_distance)
ROOT=Path(__file__).resolve().parent;RES=ROOT/'results'
MODELS=['wrapped_cubic','fourier2','spline5','fourier4','spline9']

def representation(seeds=12):
    rows=[];selection=[];examples={};rng=np.random.default_rng(60032)
    val0=np.c_[rng.uniform(-np.pi/3,np.pi/3,32),rng.uniform(0,2*np.pi,32),rng.uniform(70,140,32),rng.uniform(35,110,32)]
    for ci,config in enumerate(CONFIGS):
        pool=candidates();pool[:,2:]*=config['fixture_scale'];ids=select_poses(pool,60)
        validation=val0.copy();validation[:,2:]*=config['fixture_scale']
        matrices={name:(spline_matrix(pool,int(name[6:])) if name.startswith('spline') else matrix(pool,'wrapped_cubic' if name=='wrapped_cubic' else 'periodic',2 if name=='wrapped_cubic' else int(name[-1]))) for name in MODELS}
        vmatrices={name:(spline_matrix(validation,int(name[6:])) if name.startswith('spline') else matrix(validation,'wrapped_cubic' if name=='wrapped_cubic' else 'periodic',2 if name=='wrapped_cubic' else int(name[-1]))) for name in MODELS}
        fits={name:solver(M,ids) for name,M in matrices.items()}
        assert all(fits[name]['rank']==matrices[name].shape[-1] for name in MODELS)
        for family,cls in [('harmonic',IndependentPlant),('localized',LocalizedPlant)]:
            for si in range(seeds):
                plant=cls(831+si,tool_length=config['tool_length']);pige,linear=auxiliary_calibration(plant,seed=720+si)
                obs=rtest_signal(plant,pool)+np.random.default_rng(170+si).normal(size=(len(pool),6))*NOISE
                vobs=rtest_signal(plant,validation)+np.random.default_rng(970+si).normal(size=(32,6))*NOISE
                off=offsets(pool,pige);voff=offsets(validation,pige);models={};scores={}
                for name in MODELS:
                    z=fits[name]['pinv']@((obs[ids]-off[ids])/NOISE).ravel()
                    models[name]=spline_from_z(z,pige,linear,int(name[6:])) if name.startswith('spline') else model_from_z(z,pige,linear,'wrapped_cubic' if name=='wrapped_cubic' else 'periodic',2 if name=='wrapped_cubic' else int(name[-1]))
                    scores[name]=float(np.mean(((np.einsum('nij,j->ni',vmatrices[name],z)+voff-vobs)/NOISE)**2))
                chosen=min(['fourier2','spline5','fourier4','spline9'],key=scores.get)
                selection.append(dict(config=config['name'],truth_family=family,truth_seed=831+si,chosen=chosen,**scores))
                models['validation_selected']=models[chosen]
                for turns in [1,3]:
                    _,p,o=helix(121,turns);p*=config['fixture_scale']
                    for name,model in models.items():
                        q=geometric_commands(model,p,o,plant.tool_home,plant.part_home)
                        true=[plant.fk(v) for v in q];pa=np.array([v[0] for v in true]);oa=np.array([v[1] for v in true])
                        error=1000*np.linalg.norm(pa-p,axis=1)
                        params=matrices[chosen if name=='validation_selected' else name].shape[-1]
                        rows.append(dict(config=config['name'],truth_family=family,truth_seed=831+si,turns=turns,model=name,
                            selected_model=chosen if name=='validation_selected' else name,parameters=params,
                            tcp_rms_um=float(np.sqrt(np.mean(error**2))),tcp_max_um=float(error.max()),axis_rms_urad=float(np.sqrt(np.mean(angle_error(oa,o)**2))*1e6)))
                        if ci==1 and si==0 and turns==3:examples[family,name]=error
                print('journal representation',config['name'],family,831+si,flush=True)
    write_csv('journal_representation_trials.csv',rows);write_csv('journal_model_selection.csv',selection)
    summary=[]
    for family,turns,name in itertools.product(['harmonic','localized'],[1,3],MODELS+['validation_selected']):
        rs=[r for r in rows if r['truth_family']==family and r['turns']==turns and r['model']==name]
        v=[np.mean([r['tcp_rms_um'] for r in rs if r['truth_seed']==831+si]) for si in range(seeds)]
        summary.append(dict(truth_family=family,turns=turns,model=name,mean_rms_um=float(np.mean(v)),seed_cluster_ci95=cluster_interval(v),
            parameters=rs[0]['parameters'] if len({r['parameters'] for r in rs})==1 else '43 or 67'))
    fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    for ax,family in zip(axes,['harmonic','localized']):
        rs=[r for r in summary if r['truth_family']==family and r['turns']==3]
        y=[r['mean_rms_um'] for r in rs];ci=np.array([r['seed_cluster_ci95'] for r in rs])
        ax.bar(range(6),y,color=['#777','#0072b2','#56b4e9','#009e73','#e69f00','#cc79a7'])
        ax.errorbar(range(6),y,yerr=np.array([np.array(y)-ci[:,0],ci[:,1]-np.array(y)]),fmt='none',color='black',capsize=3)
        ax.set_xticks(range(6),[r['model'].replace('_','\n') for r in rs],rotation=25,fontsize=8)
        ax.set_ylabel('Three-turn TCP RMS (µm)');ax.set_title(family+' truth; seed-cluster 95% intervals')
    savefig('journal_equal_capacity',fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,family in zip(axes,['harmonic','localized']):
        for name in MODELS+['validation_selected']:
            ax.plot(np.linspace(0,1080,121),examples[family,name],label=name,lw=.9)
        ax.set_title(family+' reference-seed example');ax.set_xlabel('Unwrapped C (deg)');ax.set_ylabel('TCP error (µm)');ax.legend(fontsize=7,ncol=2)
    savefig('journal_family_sensor_traces',fig)
    fig,ax=plt.subplots(figsize=(7,4));bottom=np.zeros(2)
    for name in MODELS[1:]:
        counts=[sum(r['truth_family']==f and r['chosen']==name for r in selection) for f in ['harmonic','localized']]
        ax.bar(['harmonic','localized'],counts,bottom=bottom,label=name);bottom+=counts
    ax.set_ylabel('Selected models across seed/configuration pairs');ax.legend(fontsize=8)
    savefig('journal_model_selection',fig)
    return dict(records=len(rows),selection_records=len(selection),aggregate=summary,selection_counts={f:{name:sum(r['truth_family']==f and r['chosen']==name for r in selection) for name in MODELS[1:]} for f in ['harmonic','localized']},
        protocol='12 seeds x 3 configurations x 2 truth families x 2 turns x 6 schemes. 60 common training poses + 32 disjoint validation poses, all methods see same observations; one paired sensor-noise draw per seed; 121 points per held-out TCP path')

def random_commands(rng,n):
    return np.c_[rng.uniform(-80,80,(n,2)),rng.uniform(-180,-110,n),rng.uniform(-np.pi/3,np.pi/3,n),rng.uniform(0,2*np.pi,n)]

def uncertainty(seeds=12,trials=100):
    pool=candidates();ids=select_poses(pool,48);M=matrix(pool);f=solver(M,ids)
    probe_sd=.0005;cache={}
    for si in range(seeds):
        rng=np.random.default_rng(22801+si);q=np.r_[random_commands(rng,96),random_commands(rng,64)]
        cov=tcp_covariance(q,pool[ids],f);observed_cov=cov['full']+np.eye(3)[None]*probe_sd**2
        cache[si]=(q,cov,observed_cov)
        print('TCP covariance geometry',831+si,flush=True)
    rows=[];sdrows=[];samples={};crit=7.814727903
    for case,cls in [('matched',MatchedPlant),('harmonic_mismatch',IndependentPlant),('localized_mismatch',LocalizedPlant)]:
        for si in range(seeds):
            q,cov,observed_cov=cache[si]
            plant=cls(831+si);sig=rtest_signal(plant,pool[ids]);true=np.array([plant.fk(v)[0] for v in q]);errors=[]
            for trial in range(trials):
                pige,linear=auxiliary_calibration(plant,seed=380000+1000*si+trial)
                noise=np.random.default_rng(180000+1000*si+trial)
                y=sig+noise.normal(size=sig.shape)*NOISE
                z=f['pinv']@((y-offsets(pool[ids],pige))/NOISE).ravel()
                model=model_from_z(z,pige,linear);machine=Machine(model,tool_home=plant.tool_home)
                pred=np.array([machine.fk(v)[0] for v in q]);err=pred-true;errors.append(err[96:])
                measured=true+noise.normal(0,probe_sd,true.shape);observed=measured-pred
                scores=np.sqrt(np.maximum(squared_distance(observed,observed_cov),0));threshold=conformal_threshold(scores[:96])
                e=err[96:];r2=squared_distance(e,cov['rotary'][96:]);full2=squared_distance(e,cov['full'][96:])
                oc=scores[96:]<=threshold
                # Volume-equivalent radius, not a simultaneous-path guarantee.
                radius=1000*threshold*np.linalg.det(observed_cov[96:])**(1/6)
                rows.append(dict(case=case,truth_seed=831+si,noise_trial=trial,
                    rotary_latent_coverage=float(np.mean(r2<=crit)),full_latent_coverage=float(np.mean(full2<=crit)),
                    full_observed_coverage=float(np.mean(scores[96:]**2<=crit)),conformal_observed_coverage=float(np.mean(oc)),
                    conformal_threshold=threshold,mean_region_radius_um=float(np.mean(radius)),tcp_prediction_rms_um=float(1000*np.sqrt(np.mean(np.sum(e*e,axis=1))))))
                if si==0 and trial==0:samples[case]=dict(pred=pred,measured=measured,threshold=threshold,errors=err,observed_cov=observed_cov)
            E=np.array(errors);sd=np.std(E,axis=0,ddof=1);target=np.sqrt(np.diagonal(cov['full'][96:],axis1=1,axis2=2))
            sdrows.append(dict(case=case,truth_seed=831+si,relative_sd_error=float(np.linalg.norm(sd-target)/np.linalg.norm(target))))
            print('journal coverage',case,831+si,flush=True)
    write_csv('journal_tcp_coverage_trials.csv',rows);write_csv('journal_tcp_sd_control.csv',sdrows)
    summary=[]
    for case in ['matched','harmonic_mismatch','localized_mismatch']:
        rs=[r for r in rows if r['case']==case];entry={'case':case}
        for key in ['rotary_latent_coverage','full_latent_coverage','full_observed_coverage','conformal_observed_coverage','mean_region_radius_um','tcp_prediction_rms_um']:
            v=[np.mean([r[key] for r in rs if r['truth_seed']==831+si]) for si in range(seeds)]
            entry[key]=float(np.mean(v));entry[key+'_ci95']=cluster_interval(v)
        entry['relative_sd_error']=float(np.mean([r['relative_sd_error'] for r in sdrows if r['case']==case]));summary.append(entry)
    fig,axes=plt.subplots(1,2,figsize=(12,4.2));x=np.arange(3)
    for j,(key,label) in enumerate([('rotary_latent_coverage','Rotary only: latent TCP'),('full_latent_coverage','Full propagation: latent TCP'),('full_observed_coverage','Full + probe: observed TCP'),('conformal_observed_coverage','Split conformal: observed TCP')]):
        y=np.array([100*r[key] for r in summary]);ci=100*np.array([r[key+'_ci95'] for r in summary])
        axes[0].bar(x+(j-1.5)*.18,y,.18,label=label)
        axes[0].errorbar(x+(j-1.5)*.18,y,yerr=np.array([y-ci[:,0],ci[:,1]-y]),fmt='none',color='black',capsize=2)
    axes[0].axhline(95,color='black',ls='--');axes[0].set_ylim(0,103);axes[0].set_ylabel('Coverage (%)');axes[0].set_xticks(x,['Matched','Harmonic\nmismatch','Localized\nmismatch']);axes[0].legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.22),ncol=2)
    axes[1].bar(x,[r['mean_region_radius_um'] for r in summary]);axes[1].set_xticks(x,['Matched','Harmonic\nmismatch','Localized\nmismatch']);axes[1].set_ylabel('Conformal volume-equivalent radius (µm)')
    savefig('journal_tcp_coverage',fig)
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for ax,case in zip(axes,['matched','harmonic_mismatch','localized_mismatch']):
        sample=samples[case];res=1000*(sample['measured'][96:]-sample['pred'][96:]);width=1000*sample['threshold']*np.sqrt(sample['observed_cov'][96:,0,0])
        ax.fill_between(np.arange(64),-width,width,alpha=.2,label='Projected 95% conformal region, x')
        ax.plot(res[:,0],'.-',lw=.7,label='Independent noisy TCP x residual')
        ax.set_ylabel('Residual (µm)');ax.set_title(case);ax.legend(fontsize=8)
    axes[-1].set_xlabel('IID held-out pose index (not mechanical time)');savefig('journal_prediction_regions',fig)
    q,cov,observed_cov=cache[0]
    np.savez_compressed(RES/'journal_tcp_covariance.npz',q=q,rotary_cov=cov['rotary'],full_cov=cov['full'],observed_cov=observed_cov,H=cov['H'],G=cov['G'],K=cov['K'],Dw=cov['Dw'],
        q_all=np.array([cache[si][0] for si in range(seeds)]),full_cov_all=np.array([cache[si][1]['full'] for si in range(seeds)]))
    np.savez_compressed(RES/'journal_prediction_examples.npz',**{case+'__'+name:sample[name] for case,sample in samples.items() for name in ['pred','measured','threshold']})
    return dict(records=len(rows),seeds=seeds,trials_per_seed=trials,calibration_poses=96,test_poses=64,aggregate=summary,
        scope='First-order TCP covariance: 43 rotary coefficients + 11 shared PIGEs + 54 separately fitted linear coefficients. Gaussian known sensor variances; no frame uncertainty/dynamics. Split conformal covers future noisy XYZ probe responses marginally under exchangeability, not latent TCP or simultaneous trajectories. Reference configuration only; commands IID from disclosed distribution.')

def run(seeds=12,trials=100):
    start=time.time();out=dict(seeds=seeds,synthetic=True,representation=representation(seeds),uncertainty=uncertainty(seeds,trials),wall_seconds=time.time()-start)
    (RES/'journal_metrics.json').write_text(json.dumps(out,indent=2,allow_nan=False));return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=12);p.add_argument('--trials',type=int,default=100);a=p.parse_args();run(a.seeds,a.trials)
