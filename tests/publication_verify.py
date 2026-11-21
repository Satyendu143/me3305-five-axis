"""Checks for new baselines, task design, matched uncertainty, and real clock."""
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from publication_studies import *

def run():
    checks={};rng=np.random.default_rng(191)
    pool=candidates();held=pool[::7];ids=task_select(pool,held,48)
    assert len(set(ids))==48;assert np.array_equal(ids[:16],task_select(pool,held,16))
    checks['task_design_unique_nested']='pass'
    f=solver(matrix(pool),ids);assert f['rank']==43
    checks['task_design_full_rank']=f['rank']
    from dynamic_sim import servo_truth,TAU1,TAU2
    # Exact cascade matches the analytical step response and is a semigroup
    # when the held input and pole constants are unchanged.
    for dt in [.01,.003]:
        u=np.ones((101,5));u[0]=0;out=servo_truth(u,dt,integrator='exact_zoh');t=100*dt
        analytical=1-(TAU1*np.exp(-t/TAU1)-TAU2*np.exp(-t/TAU2))/(TAU1-TAU2)
        assert np.max(abs(out[-1]-analytical))<1e-12
    u=np.ones((101,5));u[0]=0;v=np.ones((201,5));v[0]=0
    assert np.max(abs(servo_truth(u,.001,integrator='exact_zoh')[-1]-servo_truth(v,.0005,integrator='exact_zoh')[-1]))<1e-12
    checks['exact_two_pole_step_and_semigroup']='pass'
    F=rng.normal(size=(8,8));F=F.T@F+np.eye(8);C=rng.normal(size=(8,8));C=C.T@C;B=rng.normal(size=(3,8))
    inv=np.linalg.inv(F);U=inv@B.T;G=np.eye(3)+B@U
    delta=np.trace(C@inv)-np.trace(C@np.linalg.inv(F+B.T@B))
    woodbury=np.trace(np.linalg.solve(G,U.T@C@U))
    assert abs(delta-woodbury)<1e-12;checks['task_variance_gain_vs_direct_inverse']=abs(delta-woodbury)
    for kind,order in [('wrapped_cubic',2),('periodic',2),('periodic',4),('cubic',2)]:
        row=np.array([.31,7.2,80.,90.]);M=matrix(row[None],kind,order)[0]
        z=rng.normal(size=M.shape[-1])*1e-5;m=model_from_z(z,np.zeros(11),np.zeros((3,6,4)),kind,order)
        plant=IndependentPlant();a,c,L,H=row
        p,R=plant.table(a,c,L,H,errors=m.evaluate([0,0,0,a,c]),pige=np.zeros(11),return_rotation=True)
        R0=(rx(-a)@rz(-c))[:3,:3];obs=np.r_[p-R0@np.array([L,0,H]),LEVER*rotation_vector_small(R@R0.T)]
        assert np.max(abs(obs-M@z))<1e-10
        checks[kind+str(order)+'_exact_vs_sensitivity']=float(np.max(abs(obs-M@z)))
    m=WrappedCubic();m.rotary=rng.normal(size=(6,2))*.001;m.scale=1e-6
    for c in [.2,1.7,5.1]:
        e0=m.evaluate([0,0,0,.4,c])[4];e1=m.evaluate([0,0,0,.4,c+2*np.pi])[4]
        delta=e1-e0;delta[5]-=m.scale*2*np.pi
        assert max(abs(delta))<1e-14
    checks['wrapped_baseline_bounded_plus_scale']='pass'
    assert solver(matrix(pool,'wrapped_cubic'),ids)['rank']==31
    left=m.evaluate([0,0,0,.4,2*np.pi-1e-9])[4];right=m.evaluate([0,0,0,.4,2*np.pi+1e-9])[4]
    assert max(abs(left-right))<1e-11
    checks['wrapped_seam_continuity_and_scale_identifiability']='pass'
    plant=IndependentPlant(mismatch=0);assert not np.any(plant.c_coeff[:,4:])
    checks['omitted_harmonics_zero_control']='pass'
    from research_experiments import load_models
    r=run_integrated(plant,load_models()['periodic'],n=301,thermal_mode='physical',full_factorial=True)
    assert abs(r['minutes'][-1]-r['minutes'][0]-.5)<1e-12
    assert len(r['schemes'])==9;checks['physical_clock_and_full_factorial']='pass'
    assert r['RTD'].shape==(301,3)
    # Covariance = pseudoinverse product, but only full rank is accepted.
    assert np.allclose(f['cov'],np.linalg.inv(f['A'].T@f['A']),rtol=1e-7,atol=1e-7)
    checks['full_rank_covariance_formula']='pass'
    singular=solver(matrix(pool),np.array([0,1,2]));assert singular['cov'] is None
    checks['rank_deficient_covariance_not_zero']='pass'
    low,high=cluster_interval([1,2,3,4]);assert low<=2.5<=high
    checks['bootstrap_centred_on_seed_clusters']='pass'
    from publication_dashboard import launch_preprint
    tabs=launch_preprint();button=tabs.children[0].children[1].children[2];button.click()
    preview=tabs._preprint_cache['run']
    assert len(preview['t'])==1501 and len(preview['schemes'])==9
    assert abs(preview['minutes'][-1]-preview['minutes'][0]-.5)<1e-12
    checks['final_dashboard_exact_zoh_button_callback']='pass'
    root=Path(__file__).resolve().parents[1];(root/'results/publication_verification.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2));return checks

if __name__=='__main__':run()
