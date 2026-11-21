"""Focused uncertainty audit: 12 seeds, 3 truths, 2 noise laws, 100 fits each.

Same records across all region methods. Nonlinear parametric Monte Carlo uses
only a fitted reference model, never the true coefficients. Drift is a declared
3 micrometre x shift after calibration; recalibration measures 96 fresh responses at calibration poses.
"""
from pathlib import Path
import argparse,json,time,csv,itertools
import numpy as np
from scipy.stats import chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_models import *
from journal_models import tcp_covariance
from journal_studies import random_commands
from validation_plant import MatchedPlant,LocalizedPlant
from independent_plant import IndependentPlant
from periodic_model import candidates,select_poses,rtest_signal,auxiliary_calibration
from publication_studies import matrix,solver,cluster_interval
ROOT=Path(__file__).resolve().parent;RES=ROOT/'results';FIG=ROOT/'report/figures'
METHODS=['rotary_only','joint_iid_assumption','joint_noise_aware','independent_parameters','nonlinear_MC','conformal_whitened','conformal_spherical']
LABELS=['Rotary only','Joint, IID assumed','Joint, known correlation','No parameter cross terms','Nonlinear MC covariance','Conformal ellipsoid','Conformal sphere']

def write(name,rows):
    with (RES/name).open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def covariance(x):
    d=x-x.mean(axis=0);return np.einsum('sni,snj->nij',d,d)/(len(x)-1)
def figsave(name,f):
    f.tight_layout();f.savefig(FIG/(name+'.png'),dpi=250,bbox_inches='tight');f.savefig(FIG/(name+'.pdf'),bbox_inches='tight');plt.close(f)

def run(seeds=12,trials=100,mc_draws=1024):
    start=time.time();pool=candidates();ids=select_poses(pool,48);pool=pool[ids];f=solver(matrix(pool),np.arange(len(pool)));P=f['pinv']
    C=auxiliary_covariance();Leta=np.linalg.cholesky(C);probe_sd=.0005;cut=np.sqrt(chi2.ppf(.95,3));rows=[];mcrows=[];shiftrows=[];examples={}
    cases=[('matched',MatchedPlant),('harmonic',IndependentPlant),('localized',LocalizedPlant)]
    for si in range(seeds):
        q=random_commands(np.random.default_rng(20261008+si),160);base=tcp_covariance(q,pool,f)
        for family,cls in cases:
            truth=cls(831+si);signal=rtest_signal(truth,pool);actual=np.array([truth.fk(v)[0] for v in q])
            eta_center=np.zeros(65);eta_center[:11]=truth.pige/ETA_SCALE[:11]
            u=np.linspace(-1,1,51);B=np.c_[u,u*u,u*u*u]
            coeff=np.array([np.linalg.lstsq(B,np.array([truth.errors(np.eye(5)[axis]*v*200)[axis] for v in u]),rcond=None)[0].T for axis in range(3)])
            eta_center[11:]=coeff.ravel()/ETA_SCALE[11:]
            for law in ['iid','correlated']:
                R=np.eye(6)
                if law=='correlated':
                    R[:3,:3]=.6+.4*np.eye(3);R[3:,3:]=.6+.4*np.eye(3)
                Rp=R[:3,:3];L=np.linalg.cholesky(R);Lp=np.linalg.cholesky(Rp);probe_cov=probe_sd**2*Rp
                covs=covariance_variants(base,P,R)
                # Paired random variates for a fair comparison, and independent across seeds.
                rng=np.random.default_rng(301000+si)
                eta=eta_center+rng.normal(size=(trials,65))@Leta.T
                noise=(rng.normal(size=(trials,len(pool),6))@L.T)*NOISE
                y=signal[None]+noise;z=((y-batch_offset(eta,pool))/NOISE).reshape(trials,-1)@P.T
                pred=batch_fk(z,eta,q);error=pred-actual[None]
                probe=(rng.normal(size=(trials,len(q),3))@Lp.T)*probe_sd
                residual=actual[None]+probe-pred
                # Parametric MC about first fitted realization, without truth access.
                mrng=np.random.default_rng(501000+si);emc=eta[0]+mrng.normal(size=(mc_draws,65))@Leta.T
                sig_hat=batch_signal(z[:1],eta[:1],pool)[0]
                ym=sig_hat+(mrng.normal(size=(mc_draws,len(pool),6))@L.T)*NOISE
                zm=((ym-batch_offset(emc,pool))/NOISE).reshape(mc_draws,-1)@P.T
                pm=batch_fk(zm,emc,q);cm=covariance(pm)
                half=covariance(pm[:mc_draws//2]);other=covariance(pm[mc_draws//2:])
                empirical=covariance(error)
                mcrows.append(dict(truth_seed=831+si,family=family,noise_law=law,mc_draws=mc_draws,
                    relative_MC_vs_first_order=float(np.linalg.norm(cm-covs['joint'])/np.linalg.norm(covs['joint'])),
                    relative_split_MC_difference=float(np.linalg.norm(half-other)/np.linalg.norm(cm)),
                    relative_empirical_vs_first_order=float(np.linalg.norm(empirical-covs['joint'])/np.linalg.norm(covs['joint'])),
                    ratio_naive_to_joint_trace=float(np.trace(covs['naive'],axis1=1,axis2=2).mean()/np.trace(covs['joint'],axis1=1,axis2=2).mean()),
                    parametric_refit_mean_shift_um=float(1000*np.sqrt(np.mean(np.sum((pm.mean(axis=0)-pred[0])**2,axis=1))))))
                V={'rotary_only':base['rotary']+np.eye(3)*probe_sd**2,
                   'joint_iid_assumption':base['full']+np.eye(3)*probe_sd**2,
                   'joint_noise_aware':covs['joint']+probe_cov,
                   'independent_parameters':covs['naive']+probe_cov,
                   'nonlinear_MC':cm+probe_cov,
                   'conformal_whitened':covs['joint']+probe_cov,
                   'conformal_spherical':np.broadcast_to(np.eye(3),(len(q),3,3))}
                for method in METHODS:
                    scores=many_scores(residual,V[method]);cp=method.startswith('conformal')
                    threshold=many_threshold(scores[:,:96]) if cp else np.full(trials,cut)
                    covered=scores[:,96:]<=threshold[:,None];radius=1000*threshold[:,None]*np.linalg.det(V[method][96:])[None]**(1/6)
                    for trial in range(trials):
                        rows.append(dict(truth_seed=831+si,family=family,noise_law=law,noise_trial=trial,method=method,
                            observed_coverage=float(covered[trial].mean()),mean_radius_um=float(radius[trial].mean()),
                            point_prediction_rms_um=float(1000*np.sqrt(np.mean(np.sum(error[trial,96:]**2,axis=1))))))
                # Out-of-distribution stress and a separately measured recalibration.
                Vcp=V['conformal_whitened'];shift=np.array([.003,0.,0.]);original=many_scores(residual,Vcp)
                shifted=many_scores(residual+shift,Vcp);oldq=many_threshold(original[:,:96])
                refreshed_noise=(rng.normal(size=(trials,96,3))@Lp.T)*probe_sd
                recal=actual[None,:96]+shift+refreshed_noise-pred[:,:96]
                newq=many_threshold(many_scores(recal,Vcp[:96]))
                for trial in range(trials):
                    for state,qt in [('before_shift',oldq),('shift_stale_calibration',oldq),('shift_recalibrated',newq)]:
                        sc=original[trial,96:] if state=='before_shift' else shifted[trial,96:]
                        rr=1000*qt[trial]*np.linalg.det(Vcp[96:])**(1/6)
                        shiftrows.append(dict(truth_seed=831+si,family=family,noise_law=law,noise_trial=trial,state=state,
                            observed_coverage=float(np.mean(sc<=qt[trial])),mean_radius_um=float(rr.mean())))
                if si==0 and family=='localized' and law=='correlated':
                    examples=dict(q=q,residual=residual[0],shifted_residual=residual[0]+shift,cov=Vcp,
                                  old_threshold=oldq[0],new_threshold=newq[0])
                print('audit',831+si,family,law,flush=True)
    write('audit_region_trials.csv',rows);write('audit_nonlinear_MC.csv',mcrows);write('audit_shift_trials.csv',shiftrows)
    summary=[]
    for family,law,method in itertools.product([c[0] for c in cases],['iid','correlated'],METHODS):
        sub=[r for r in rows if (r['family'],r['noise_law'],r['method'])==(family,law,method)];r=dict(family=family,noise_law=law,method=method)
        for key in ['observed_coverage','mean_radius_um','point_prediction_rms_um']:
            values=[np.mean([x[key] for x in sub if x['truth_seed']==831+si]) for si in range(seeds)]
            r[key]=float(np.mean(values));r[key+'_seed_ci95']=cluster_interval(values)
        summary.append(r)
    shifts=[]
    for family,law,state in itertools.product([c[0] for c in cases],['iid','correlated'],['before_shift','shift_stale_calibration','shift_recalibrated']):
        sub=[r for r in shiftrows if (r['family'],r['noise_law'],r['state'])==(family,law,state)];r=dict(family=family,noise_law=law,state=state)
        for key in ['observed_coverage','mean_radius_um']:
            values=[np.mean([x[key] for x in sub if x['truth_seed']==831+si]) for si in range(seeds)];r[key]=float(np.mean(values));r[key+'_seed_ci95']=cluster_interval(values)
        shifts.append(r)
    # Pair differences at seed level: no false independence of poses/noise trials.
    paired=[]
    for family,law in itertools.product([c[0] for c in cases],['iid','correlated']):
        values=[]
        for si in range(seeds):
            means={method:np.mean([r['mean_radius_um'] for r in rows if r['family']==family and r['noise_law']==law and r['truth_seed']==831+si and r['method']==method]) for method in ['conformal_whitened','conformal_spherical']}
            values.append(means['conformal_whitened']-means['conformal_spherical'])
        paired.append(dict(family=family,noise_law=law,ellipsoid_minus_sphere_radius_um=float(np.mean(values)),seed_ci95=cluster_interval(values)))
    out=dict(synthetic=True,seeds=seeds,noise_trials=trials,mc_draws=mc_draws,raw_fits=seeds*3*2*trials,
             parametric_MC_refits=seeds*3*2*mc_draws,region_records=len(rows),shift_records=len(shiftrows),aggregate=summary,shift_aggregate=shifts,paired_radius=paired,
             mean_MC_covariance_relative_difference=float(np.mean([r['relative_MC_vs_first_order'] for r in mcrows])),
             mean_split_MC_relative_difference=float(np.mean([r['relative_split_MC_difference'] for r in mcrows])),wall_seconds=time.time()-start,
             protocol='48 six-channel training poses; 96 independent response-calibration poses, 64 assessment poses per seed. Probe SD 0.5 um. Within-channel-group correlation 0.6 in correlated law; no across-pose dependence. Fixed +3 um x drift violates calibration/assessment exchangeability. Recalibration uses 96 fresh responses at the original independently drawn calibration poses. Same point predictions for all region methods. First-order derivatives about zero maps; nonlinear MC about first noisy fitted realization.')
    (RES/'audit_metrics.json').write_text(json.dumps(out,indent=2,allow_nan=False));np.savez_compressed(RES/'audit_example.npz',**examples)
    plot(out);return out

def plot(m):
    plt.rcParams.update({'font.size':12,'pdf.fonttype':42})
    fig,axes=plt.subplots(3,2,figsize=(10,8.8),sharey=True)
    for ax,(fam,law) in zip(axes.flat,itertools.product(['matched','harmonic','localized'],['iid','correlated'])):
        r=[next(x for x in m['aggregate'] if x['family']==fam and x['noise_law']==law and x['method']==k) for k in METHODS]
        y=np.array([100*x['observed_coverage'] for x in r]);ci=100*np.array([x['observed_coverage_seed_ci95'] for x in r]);ax.bar(range(7),y,color=['#aaa','#888','#0072b2','#cc79a7','#009e73','#e69f00','#56b4e9'])
        ax.errorbar(range(7),y,yerr=np.maximum(np.array([y-ci[:,0],ci[:,1]-y]),0),fmt='none',color='k',capsize=2)
        ax.axhline(95,color='black',ls='--',lw=.8);ax.set(ylim=(0,102),title=f'{fam}, {law}',xticks=range(7),xticklabels=['Rot','IID','Joint','No cross','MC','CP ell.','CP sph.']);ax.tick_params(axis='x',rotation=35)
    [ax.set_ylabel('XYZ coverage (%)') for ax in axes[:,0]];figsave('audit_region_comparison',fig)
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.7))
    for ax,law in zip(axes,['iid','correlated']):
        for j,method in enumerate(['conformal_whitened','conformal_spherical']):
            r=[next(x for x in m['aggregate'] if x['family']==fam and x['noise_law']==law and x['method']==method) for fam in ['matched','harmonic','localized']]
            ax.bar(np.arange(3)+(j-.5)*.32,[x['mean_radius_um'] for x in r],.32,label='Ellipsoid' if j==0 else 'Sphere')
        ax.set(xticks=range(3),xticklabels=['Matched','Harmonic','Localized'],ylabel='Mean equivalent radius (um)',title=law+' noise');ax.legend()
    figsave('audit_region_efficiency',fig)
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.7),sharey=True)
    for ax,law in zip(axes,['iid','correlated']):
        for j,state in enumerate(['before_shift','shift_stale_calibration','shift_recalibrated']):
            r=[next(x for x in m['shift_aggregate'] if x['family']==fam and x['noise_law']==law and x['state']==state) for fam in ['matched','harmonic','localized']]
            ax.bar(np.arange(3)+(j-1)*.24,[100*x['observed_coverage'] for x in r],.24,label=['Before drift','Stale calibration','Fresh calibration'][j])
        ax.axhline(95,color='k',ls='--',lw=.8);ax.set(xticks=range(3),xticklabels=['Matched','Harmonic','Localized'],ylabel='Observed XYZ coverage (%)',ylim=(0,103),title=law+' noise')
    axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3,fontsize=10);figsave('audit_drift_coverage',fig)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=12);p.add_argument('--trials',type=int,default=100);p.add_argument('--mc-draws',type=int,default=1024);a=p.parse_args();run(a.seeds,a.trials,a.mc_draws)
