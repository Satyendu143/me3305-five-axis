"""Run every report experiment; deterministic seeds and export all numerical data."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fiveaxis import *
from calibration import *
import remedies

ROOT=Path(__file__).resolve().parent
FIG=ROOT/'report'/'figures'; DATA=ROOT/'results'

def save(fig,name):
    FIG.mkdir(parents=True,exist_ok=True); fig.tight_layout()
    fig.savefig(FIG/(name+'.png'),dpi=170,bbox_inches='tight'); plt.close(fig)

def summary(r):
    return {k:{'rms':float(np.sqrt(np.mean(r[k]**2))),'max':float(np.max(r[k]))} for k in ('pos_um','ori_urad','contour_um')}

def measured_model(true,noise_um=.4,noise_urad=.4,seed=80):
    rng=np.random.default_rng(seed); estimate=ErrorModel(); datasets=[]
    for a in range(5):
        u=np.linspace(-1,1,31) if a<4 else np.linspace(0,2,61)
        B=np.c_[u,u*u,u*u*u]
        for j in range(6):
            exact=B@true.coeff[a,j,1:]
            sig=noise_um/1000 if j<3 else noise_urad/1e6
            observed=exact+rng.normal(0,sig,len(u))
            estimate.coeff[a,j,1:]=np.linalg.lstsq(B,observed,rcond=None)[0]
            datasets.append((a,j,u,exact,observed,B@estimate.coeff[a,j,1:]))
    pnoise=np.array([noise_urad/1e6]*3+[noise_um/1000]*2+[noise_urad/1e6]*2+[noise_um/1000]*2+[noise_urad/1e6]*2)
    # Independent virtual alignment measurements; not inferred from motor encoders.
    estimate.pige=true.pige+rng.normal(0,pnoise)
    return estimate,datasets

def run_all(n=121,mc=100):
    DATA.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.22})
    metrics={'provenance':'All data synthetic; no raw data from Xiang & Altintas. mm/rad internally; um/urad on plots.','seed':20261007}
    model=synthetic_model(); machine=Machine(model); t,p,o=helix(n)
    model_json={'coeff':model.coeff.tolist(),'pige':model.pige.tolist(),'convention':model.convention,'q_scale':Q_SCALE.tolist(),'tool_home':machine.tool_home.tolist(),'part_home':machine.part_home.tolist()}
    (DATA/'model.json').write_text(json.dumps(model_json,indent=2))
    runs={s:trajectory(machine,p,o,s) for s in ['none','positioning','full']}
    metrics['geometry']={s:summary(r) for s,r in runs.items()}
    fig=plt.figure(figsize=(9,6)); ax=fig.add_subplot(projection='3d')
    ax.plot(*p.T,label='Target'); ax.plot(*(p+500*(runs['none']['p']-p)).T,label='Uncompensated (error x500)')
    ax.plot(*(p+500*(runs['full']['p']-p)).T,label='Full map (error x500)')
    ax.quiver(*p[::12].T,*o[::12].T,length=12,color='k',label='Target tool axes')
    ax.set(xlabel='X [mm]',ylabel='Y [mm]',zlabel='Z [mm]',title='Synthetic helix: displacement errors exaggerated x500'); ax.legend(fontsize=8)
    save(fig,'helix_3d')
    fig,axs=plt.subplots(3,1,figsize=(9,8),sharex=True)
    for s,r in runs.items():
        for ax,k,l in zip(axs,['pos_um','ori_urad','contour_um'],['TCP following [um]','Tool-axis angle [urad]','Local normal error [um]']):
            ax.plot(t,r[k],label=s); ax.set_ylabel(l)
    axs[0].legend(); axs[-1].set_xlabel('Normalized path parameter'); save(fig,'geometry_compensation')
    for s,r in runs.items(): np.savez_compressed(DATA/f'helix_{s}.npz',target=p,target_o=o,t=t,**{k:v for k,v in r.items() if k!='history'})

    # Large-error test, not scaled to manufacture a match to the paper's 5 mm.
    large=Machine(synthetic_model(scale=25)); histories=[]
    for pi,oi in zip(p[::6],o[::6]):
        _,h=compensate(large,pi,oi,iterations=5); padded=np.tile(h[-1],(6,1)); padded[:len(h)]=h; histories.append(padded)
    maxima=np.max(histories,axis=0); metrics['large_errors']={'max_position_um_by_iteration':(1000*maxima[:,0]).tolist(),'max_orientation_urad_by_iteration':(1e6*maxima[:,1]).tolist()}
    fig,ax=plt.subplots(figsize=(8,4)); ax.semilogy(np.arange(6),np.maximum(1000*maxima[:,0],1e-9),'o-',label='Position [um]'); ax.semilogy(np.arange(6),np.maximum(1e6*maxima[:,1],1e-9),'s-',label='Axis angle [urad]'); ax.set(xlabel='Compensation updates',ylabel='Max error (different units)',title='Large synthetic errors: iterative contraction'); ax.legend(); save(fig,'iteration')
    np.savez(DATA/'iteration.npz',maxima=maxima)

    # All component channels present; realistic estimated maps have nonzero floor.
    estimated,dsets=measured_model(model); predictor=Machine(estimated)
    np.savez(DATA/'component_measurements.npz',**{f'axis_{a}_channel_{j}':np.c_[u,exact,obs,fit] for a,j,u,exact,obs,fit in dsets})
    q=[]; pa=[]; oa=[]
    for pi,oi in zip(p,o):
        qi,_=compensate(predictor,pi,oi); ai,bi=machine.fk(qi); q.append(qi);pa.append(ai);oa.append(bi)
    estimated_pos=1000*np.linalg.norm(np.asarray(pa)-p,axis=1); estimated_ori=1e6*angle_error(np.asarray(oa),o)
    missing=ErrorModel(coeff=estimated.coeff.copy(),pige=estimated.pige.copy());
    for a in range(3): missing.coeff[a,3+a]=0
    qmiss=[]; pmiss=[]
    for pi,oi in zip(p,o):
        qi,_=compensate(Machine(missing),pi,oi); ai,_=machine.fk(qi);pmiss.append(ai)
    missed=1000*np.linalg.norm(np.asarray(pmiss)-p,axis=1)
    metrics['estimated_maps']={'rms_position_um':remedies.rms(estimated_pos),'max_position_um':float(max(estimated_pos)),'rms_axis_urad':remedies.rms(estimated_ori),'missing_roll_rms_um':remedies.rms(missed)}
    fig,axs=plt.subplots(2,3,figsize=(12,7));
    for j,ax in enumerate(axs.flat):
        _,_,u,exact,obs,fit=dsets[j]; fact=1000 if j<3 else 1e6
        ax.plot(200*u,exact*fact,label='True map');ax.scatter(200*u,obs*fact,s=12,label='Virtual sensor');ax.plot(200*u,fit*fact,'--',label='Cubic fit')
        ax.set(xlabel='X drive [mm]',ylabel='um' if j<3 else 'urad',title=PDGE_NAMES[j])
    axs[0,0].legend(fontsize=8);save(fig,'linear_sensors')
    fig,axs=plt.subplots(2,1,figsize=(9,6));axs[0].plot(t,estimated_pos,label='All channels measured');axs[0].plot(t,missed,label='Three linear roll channels omitted');axs[0].set_ylabel('TCP error [um]');axs[0].legend();axs[1].plot(t,estimated_ori);axs[1].set(xlabel='Path parameter',ylabel='Axis angle [urad]');save(fig,'roll_and_map_noise')
    np.savez(DATA/'estimated_map.npz',pos_um=estimated_pos,ori_urad=estimated_ori,missing_roll_um=missed,estimated_coeff=estimated.coeff,estimated_pige=estimated.pige)
    # 41-channel sensitivity, physical error unit 1um or 1urad.
    sensitivities=[]
    for k in range(41):
        e=ErrorModel(); unit=.001
        if k<30:
            a,j=divmod(k,6); unit=.001 if j<3 else 1e-6; e.coeff[a,j,1]=unit
        else:
            j=k-30; unit=.001 if j in [3,4,7,8] else 1e-6; e.pige[j]=unit
        r=trajectory(Machine(e),p[::8],o[::8],scheme='none'); sensitivities.append(remedies.rms(r['pos_um']))
    fig,ax=plt.subplots(figsize=(13,5));ax.bar(np.arange(41),sensitivities);ax.set_xticks(np.arange(41),error_catalogue(),rotation=90);ax.set(ylabel='RMS TCP [um] per 1um / 1urad amplitude',title='All 41 geometric channels: path-specific sensitivity');save(fig,'sensitivity_41');metrics['sensitivity_um']=sensitivities

    # Eq.29 audit with literal printed expression versus corrected algebra.
    e=np.array([.012,-.008,.005,30e-6,-20e-6,15e-6]);angles=np.deg2rad(np.linspace(-60,30,181));errlit=[];errgood=[];rank=[]
    for a in angles:
        read=a_readings(a,50,100,100,e);errlit.append((a_printed_eq29(read,a,50,100,100)-e)/np.array([.001]*3+[1e-6]*3));errgood.append((a_corrected(read,a,50,100,100)-e)/np.array([.001]*3+[1e-6]*3))
    fig,axs=plt.subplots(2,1,figsize=(9,6));axs[0].plot(np.rad2deg(angles),np.asarray(errlit)[:,3],label='Printed eps_xa error [urad]');axs[0].plot(np.rad2deg(angles),np.asarray(errgood)[:,3],label='Corrected');axs[0].legend();axs[0].set_ylabel('Recovered - true [urad]');axs[1].plot(np.rad2deg(angles),np.asarray(errlit)[:,2],label='Printed delta_za error [um]');axs[1].plot(np.rad2deg(angles),np.asarray(errgood)[:,2],label='Corrected');axs[1].legend();axs[1].set(xlabel='A angle [deg]',ylabel='Recovered - true [um]');save(fig,'eq29_audit')
    metrics['eq29_corrected_max_error_scaled']=float(np.max(np.abs(errgood)));np.savez(DATA/'eq29_audit.npz',angles=angles,error_printed=errlit,error_corrected=errgood)

    # Rotary simultaneous/separate/alternating identification with full scalar data.
    A,records=design(n=25); z=truth(); y=readings(records,z,noise_um=.2)
    fits={s:solve(A,y,.2,s) for s in ['joint','separate','alternating']}
    fits['nonlinear']=refine(records,y,fits['joint']['z'],noise_um=.2)
    holdA,hold=design(n=32); yh=readings(hold,z,noise_um=0)
    metrics['calibration']={s:{'rank':f['rank'],'condition':f['condition'],'parameter_scaled_rmse':remedies.rms(f['z']-z),'training_residual_um':remedies.rms(f['residual_um']),'heldout_sensor_um':remedies.rms(1000*(exact_predictions(hold,f['z'])-yh))} for s,f in fits.items()}
    # Complete calibration-to-compensation chain, keeping the SAME relative gauge.
    # Squareness and linear maps are independently calibrated with virtual optics.
    reltrue=ErrorModel(coeff=model.coeff.copy(),pige=np.zeros(11),convention='relative')
    reltrue.pige[:3]=model.pige[:3];reltrue.pige[3:]=z[:8]*S8
    reltrue.coeff[3:,:,1:]=(z[8:]*S44[8:]).reshape(2,6,3)
    relfit=ErrorModel(coeff=estimated.coeff.copy(),pige=np.zeros(11),convention='relative')
    relfit.pige[:3]=estimated.pige[:3];relfit.pige[3:]=fits['nonlinear']['z'][:8]*S8
    relfit.coeff[3:,:,1:]=(fits['nonlinear']['z'][8:]*S44[8:]).reshape(2,6,3)
    ph=2*np.pi*t;aa=np.deg2rad(20+7*np.sin(ph));orot=np.c_[np.sin(ph)*np.sin(aa),-np.cos(ph)*np.sin(aa),np.cos(aa)]
    relmachine=Machine(reltrue);baseline=trajectory(relmachine,p,orot,'none');calpos=[];calori=[];pred=Machine(relfit)
    for pi,oi in zip(p,orot):
        qi,_=compensate(pred,pi,oi);ai,bi=relmachine.fk(qi);calpos.append(1000*np.linalg.norm(ai-pi));calori.append(1e6*angle_error(bi,oi))
    metrics['calibration_to_compensation']={'before_rms_um':remedies.rms(baseline['pos_um']),'after_rms_um':remedies.rms(calpos),'after_max_um':float(max(calpos)),'before_axis_rms_urad':remedies.rms(baseline['ori_urad']),'after_axis_rms_urad':remedies.rms(calori)}
    fig,axs=plt.subplots(2,1,figsize=(9,7));axs[0].plot(t,baseline['pos_um'],label='Before');axs[0].plot(t,calpos,label='From noisy optical + ballbar calibration');axs[0].set_ylabel('TCP [um]');axs[0].legend();axs[1].plot(t,baseline['ori_urad'],label='Before');axs[1].plot(t,calori,label='After');axs[1].set(xlabel='Held-out path parameter',ylabel='Axis angle [urad]');axs[1].legend();save(fig,'calibration_to_compensation')
    np.savez(DATA/'calibration_to_compensation.npz',t=t,before_um=baseline['pos_um'],after_um=calpos,before_urad=baseline['ori_urad'],after_urad=calori,true_coeff=reltrue.coeff,true_pige=reltrue.pige,estimated_coeff=relfit.coeff,estimated_pige=relfit.pige)
    fig,axs=plt.subplots(2,1,figsize=(9,7));ballbar_out={};rngbb=np.random.default_rng(88)
    for axis,ax in zip(['A','C'],axs):
        angles=np.linspace(-np.pi/3,np.pi/6,81) if axis=='A' else np.linspace(0,2*np.pi,81)
        b=np.array([0,0,50.]) if axis=='A' else np.array([100.,0,50.]);nvec=np.array([1.,0,0]) if axis=='A' else np.array([0.,0,1.])
        traces={'none':[],'positioning':[],'measured_full':[]}
        for angle in angles:
            a,c=(angle,0.) if axis=='A' else (0.,angle)
            R=(rz(c)@rx(a))[:3,:3];target=b-100*R@nvec;orientation=R@np.array([0,0,1.]);q0=relmachine.ideal().ik(target,orientation,branch=1 if a>=0 else -1)
            for scheme in traces:
                if scheme=='none':qi=q0
                else:
                    pr=pred if scheme=='measured_full' else Machine(reltrue.subset('positioning'))
                    qi,_=compensate(pr,target,orientation,q0=q0)
                actual,_=relmachine.fk(qi);traces[scheme].append(1000*(np.linalg.norm(b-actual)-100)+rngbb.normal(0,.2))
        for scheme,yy in traces.items():ax.plot(np.rad2deg(angles),yy,label=scheme)
        ax.set(xlabel=f'{axis} [deg]',ylabel='Axial ballbar signal [um]');ax.legend(fontsize=8)
        ballbar_out[axis+'_angle']=angles
        for scheme,yy in traces.items():ballbar_out[axis+'_'+scheme]=yy
    save(fig,'ballbar_compensation');np.savez(DATA/'ballbar_compensation.npz',**ballbar_out)
    np.savez(DATA/'calibration.npz',A=A,y=y,truth_scaled=z,**{s:f['z'] for s,f in fits.items()})
    with (DATA/'ballbar_training.csv').open('w') as fp:
        w=csv.writer(fp);w.writerow(['axis','a_rad','c_rad','L_mm','H_mm','nx','ny','nz','direction_index','reading_mm'])
        for rec,yi in zip(records,y): axis,a,c,L,H,nvec,k=rec;w.writerow([axis,a,c,L,H,*nvec,k,yi])
    for axid,label in enumerate(['A','C']):
        ua=np.linspace(-2/3,1/3,101) if axid==0 else np.linspace(0,2,101);basis=np.c_[ua,ua**2,ua**3]
        fig,axs=plt.subplots(2,3,figsize=(12,7)); trueco=(z[8:]*S44[8:]).reshape(2,6,3)
        for j,ax in enumerate(axs.flat):
            fact=1000 if j<3 else 1e6;ax.plot(ua*(90 if axid==0 else 180),basis@trueco[axid,j]*fact,'k',label='Synthetic truth')
            for s in ['separate','alternating','nonlinear']:
                co=(fits[s]['z'][8:]*S44[8:]).reshape(2,6,3);ax.plot(ua*(90 if axid==0 else 180),basis@co[axid,j]*fact,label=s)
            ax.set(xlabel=f'{label} [deg]',ylabel='um' if j<3 else 'urad',title=('delta_' if j<3 else 'eps_')+'xyz'[j%3]+label)
        axs[0,0].legend(fontsize=7);save(fig,f'rotary_pdge_{label}')
    fig,axs=plt.subplots(2,1,figsize=(9,7));names=P8;phys=np.where(S8==.001,1000,1e6)
    for s,f in fits.items(): axs[0].plot(np.arange(8),(f['z'][:8]-z[:8])*S8*phys,'o-',label=s)
    axs[0].set_xticks(np.arange(8),names);axs[0].set_ylabel('PIGE estimation error [um / urad]');axs[0].legend(fontsize=8)
    axs[1].semilogy(fits['alternating']['convergence']);axs[1].axhline(np.linalg.norm(fits['joint']['residual_um'])/1000,label='Joint LS residual',ls='--');axs[1].set(xlabel='Alternating updates',ylabel='Residual norm [mm]');axs[1].legend();save(fig,'pige_decoupling')

    # Observability: a scalar direction versus 3 directions, same poses/noise.
    mask=np.array([r[-1]==1 for r in records]);rank_radial=solve(A[mask],y[mask])['rank']
    AR,recR=design('rtest',n=25);fr=solve(AR,readings(recR,z,'rtest',noise_um=.2),.2)
    fig,ax=plt.subplots(figsize=(9,5));ax.semilogy(np.linalg.svd(A,compute_uv=False),'o-',label='All 3 ballbar directions');ax.semilogy(np.linalg.svd(A[mask],compute_uv=False),'s-',label='Only radial ballbar');ax.semilogy(fr['singular'],'--',label='3D R-test same poses/noise');ax.set(xlabel='Singular-value index',ylabel='Scaled sensitivity singular value',title='Geometry and direction coverage determine observability');ax.legend();save(fig,'observability')
    metrics['observability']={'full_rank':44,'radial_rank':rank_radial,'rtest_rank':fr['rank'],'condition_full':fits['joint']['condition']}
    # Known linear model MC isolates sensor-noise uncertainty; no thermal contamination.
    rng=np.random.default_rng(91);pinv=np.linalg.pinv(A,rcond=1e-10);Z=z[:,None]+pinv@rng.normal(0,.2/1000,(len(y),mc))
    predicted_std=np.sqrt(np.diag(fits['joint']['covariance']));empirical=np.std(Z-z[:,None],axis=1,ddof=1)
    fig,ax=plt.subplots(figsize=(9,4));ax.plot(predicted_std,label='Linear covariance prediction');ax.plot(empirical,'--',label=f'Monte Carlo {mc} trials');ax.set(xlabel='Scaled parameter index',ylabel='Standard deviation [scaled units]',title='Uncertainty, not just least-squares residual');ax.legend();save(fig,'uncertainty');metrics['uncertainty']={'mc_trials':mc,'relative_std_difference':float(np.linalg.norm(empirical-predicted_std)/np.linalg.norm(predicted_std))};np.savez(DATA/'uncertainty.npz',samples=Z,predicted_std=predicted_std)
    # Height/radius degeneracy of pointwise A-axis recovery.
    gaps=np.geomspace(.01,100,100);conds=[]
    scales=np.diag([.001]*3+[1e-5]*3)
    for gap in gaps: conds.append(np.linalg.cond(np.vstack([a_design(.2,50,0),a_design(.2,50+gap,0),a_design(.2,50,100)])@scales))
    fig,ax=plt.subplots(figsize=(8,4));ax.loglog(gaps,conds);ax.set(xlabel='Height separation H1-H [mm]',ylabel='Scaled condition number',title='Remedy: separate measurement lever arms');save(fig,'calibration_geometry');metrics['geometry_condition']={'gap_001mm':conds[0],'gap_100mm':conds[-1]}

    # Sensor traces of each extension; truth curves and readings are distinct.
    for name,fn in [('thermal',remedies.thermal),('backlash',remedies.backlash),('servo',remedies.servo),('abbe',remedies.abbe),('sensor_faults',remedies.sensor_faults)]:
        r=fn();metrics[name]={k:v for k,v in r.items() if isinstance(v,(int,float))};np.savez_compressed(DATA/(name+'.npz'),**{k:v for k,v in r.items() if not isinstance(v,str)})
        if name=='thermal':
            fig,axs=plt.subplots(2,1,figsize=(9,7));axs[0].plot(r['t']/60,r['sensors']);axs[0].set(ylabel='RTD temperature rise [K]',title='Synthetic independent temperature channels')
            axs[1].plot(r['t']/60,r['before'][:,2],label='Z drift truth');axs[1].plot(r['t']/60,r['after'][:,2],label='After learned feedforward');axs[1].set(xlabel='Time [min]',ylabel='Z error [um]');axs[1].legend()
        elif name=='abbe':
            fig,axs=plt.subplots(2,1,figsize=(9,7));axs[0].plot(r['x'],r['theta'],label='Angular truth');axs[0].plot(r['x'],r['angle_sensor'],label='Autocollimator signal',alpha=.6);axs[0].set_ylabel('Angle [urad]');axs[0].legend()
            for k in ['before','after','collocated']: axs[1].plot(r['x'],r[k],label=k)
            axs[1].set(xlabel='Travel [mm]',ylabel='Displacement measurement error [um]');axs[1].legend()
        elif name=='sensor_faults':
            fig,axs=plt.subplots(2,1,figsize=(9,7));axs[0].plot(r['t'],r['sensors']);axs[0].plot(r['t'],r['truth'],'k--',label='Truth');axs[0].set_ylabel('Sensor readings [um]');axs[0].legend()
            for k in ['before','after']: axs[1].plot(r['t'],r[k],label=k)
            axs[1].set(xlabel='Time [s]',ylabel='Error [um]');axs[1].legend()
        else:
            fig,axs=plt.subplots(2,1,figsize=(9,7));axs[0].plot(r['t'],r['sensor_before'],label='External scale/encoder before');axs[0].plot(r['t'],r['sensor_after'],label='After estimated remedy');axs[0].set_ylabel('Measured following error [um]');axs[0].legend()
            axs[1].plot(r['t'],r['before'],label='Physical before');axs[1].plot(r['t'],r['after'],label='Physical after');axs[1].set(xlabel='Time [s]',ylabel='Physical following error [um]');axs[1].legend()
        save(fig,name)
    # Rotary IK singularity: no extra sensor repairs a missing actuator direction.
    angles=np.geomspace(1e-4,60,100);cond=[];gain=[]
    ideal=Machine()
    for aa in angles:
        a=np.deg2rad(aa);J=ideal.jacobian(np.array([0.,0.,-80.,a,.5]))@np.diag(Q_SCALE);cond.append(np.linalg.cond(J));gain.append(1/max(abs(np.sin(a)),1e-12))
    fig,ax=plt.subplots(figsize=(8,4));ax.loglog(angles,cond,label='Scaled full-task Jacobian condition');ax.loglog(angles,gain,label='Azimuth sensitivity 1/sin(A)');ax.set(xlabel='Tilt A [deg]',ylabel='Condition / amplification');ax.legend();save(fig,'singularity');metrics['singularity']={'condition_at_0001deg':cond[0],'condition_at_60deg':cond[-1]}
    # Explicit MCS-to-PCS datum mapping experiment.
    x=np.linspace(-100,100,200);coef=np.array([.02,5e-5,1e-7,-2e-10]);offset=80.;f=np.polynomial.polynomial.polyval(offset+x,coef);mapped=pcs_map(coef,offset,x)
    fig,ax=plt.subplots(figsize=(8,4));ax.plot(x,1000*f,label='Absolute MCS error');ax.plot(x,1000*mapped,label='Datum-subtracted PCS');ax.axvline(0,color='k',lw=.7);ax.set(xlabel='PCS position [mm]',ylabel='Error [um]');ax.legend();save(fig,'pcs_mapping')
    (DATA/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
    volume_study(machine)
    return metrics

def volume_study(machine=None):
    if machine is None:machine=Machine(synthetic_model())
    pp=[];ee=[];qq=[]
    for x in np.linspace(-100,100,7):
        for y in np.linspace(-100,100,7):
            for z in np.linspace(-130,30,7):
                q=np.array([x,y,z,np.deg2rad(30),np.deg2rad(45)])
                p,_=machine.ideal().fk(q);pe,_=machine.fk(q);pp.append(p);ee.append(pe-p);qq.append(q)
    pp=np.asarray(pp);ee=np.asarray(ee);norm=1000*np.linalg.norm(ee,axis=1)
    fig=plt.figure(figsize=(9,6));ax=fig.add_subplot(projection='3d');sc=ax.scatter(*pp.T,c=norm,cmap='viridis',s=20)
    fig.colorbar(sc,ax=ax,shrink=.7,label='TCP geometric error [um]');ax.set(xlabel='PCS X [mm]',ylabel='PCS Y [mm]',zlabel='PCS Z [mm]',title='Volume slice: fixed A=30 deg, C=45 deg (343 drive-grid poses)');save(fig,'volume_map')
    np.savez(DATA/'volume_map.npz',drive_commands=qq,nominal=pp,error_mm=ee,norm_um=norm)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--points',type=int,default=121);parser.add_argument('--mc',type=int,default=100);args=parser.parse_args()
    print(json.dumps(run_all(args.points,args.mc),indent=2))
