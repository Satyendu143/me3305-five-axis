"""Regression tests for independent truth, periodicity, design and coupling."""
import sys,json,ast
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fiveaxis import Machine,ErrorModel,synthetic_model
from independent_plant import IndependentPlant
from periodic_model import *
from dynamic_sim import *
from research_experiments import load_models

def run():
    checks={};plant=IndependentPlant();rng=np.random.default_rng(512)
    # Static import audit proves implementation separation, not hardware validity.
    tree=ast.parse((Path(__file__).resolve().parents[1]/'independent_plant.py').read_text())
    imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
    assert imports==['numpy'];checks['independent_plant_import_audit']='NumPy only'
    model=ErrorModel(convention='relative');model.coeff=synthetic_model().coeff;model.pige=plant.pige
    machine=Machine(model);error=0
    for _ in range(100):
        q=rng.uniform([-100,-100,-100,-1,-10],[100,100,0,1,12]);p,o=machine.fk(q)
        pi,oi=plant.fk(q,errors=model.evaluate(q),pige=model.pige)
        error=max(error,np.linalg.norm(p-pi),np.linalg.norm(o-oi))
    assert error<2e-12;checks['100_pose_direct_vs_homogeneous_max_mm']=error
    m=load_models()['periodic']
    for c in [-10.,-.2,.7,6.4,18.]:
        e0=m.evaluate([0,0,0,.5,c])[4];e1=m.evaluate([0,0,0,.5,c+2*np.pi])[4]
        delta=e1-e0;delta[5]-=m.scale*2*np.pi
        assert np.max(abs(delta))<1e-15
    checks['Fourier_wrap_and_separate_scale']='pass'
    assert np.max(abs(m.evaluate(np.zeros(5))))==0;checks['zero_datum_maps']='pass'
    # Exact full-frame sensitivity finite differences with independent plant.
    pool=candidates();row=pool[137];base=RotaryModel();base.pige=np.zeros(11)
    A=sensitivity(*row);maxerr=0
    for j in range(43):
        z=np.zeros(43);z[j]=1e-4;unpack(z,base)
        a,c,L,H=row;pp,RR=plant.table(a,c,L,H,errors=base.evaluate([0,0,0,a,c]),pige=base.pige,return_rotation=True)
        R0=(rx(-a)@rz(-c))[:3,:3];obs=np.r_[pp-R0@np.array([L,0,H]),LEVER*rotation_vector_small(RR@R0.T)]
        maxerr=max(maxerr,np.max(abs(obs/1e-4-A[:,j])))
    assert maxerr<1e-7;checks['43_parameter_full_frame_sensitivity']=maxerr
    ix=select_poses(pool,48);assert len(set(ix))==48
    assert np.array_equal(ix[:16],select_poses(pool,16));checks['equal_budget_unique_nested_design']='pass'
    S=np.concatenate([sensitivity(*r) for r in pool[ix]])
    assert np.linalg.matrix_rank(S)==43;checks['mixed_tilt_full_rank']=43
    single=np.concatenate([sensitivity(*r) for r in pool[pool[:,0]==0]])
    assert np.linalg.matrix_rank(single)<43;checks['orientation_cannot_fix_missing_tilts']=int(np.linalg.matrix_rank(single))
    Sp=np.concatenate([sensitivity(*r,orientation=False) for r in pool[ix]])/np.tile(NOISE[:3],len(ix))[:,None]
    Sa=S/np.tile(NOISE,len(ix))[:,None]
    assert np.linalg.eigvalsh(Sa.T@Sa-Sp.T@Sp).min()>-1e-5
    checks['orientation_information_positive_semidefinite']='pass'
    # First-order inverse proof checked in a matched control, not two-pole truth.
    desired=rng.normal(size=(100,5));alpha=np.array([.4,.5,.6,.7,.8]);u=desired.copy()
    u[1:]=desired[:-1]+np.diff(desired,axis=0)/alpha;y=desired[0].copy();err=[]
    for k in range(1,len(u)):y=y+alpha*(u[k]-y);err.append(np.max(abs(y-desired[k])))
    assert max(err)<1e-14;checks['first_order_inverse_matched_control']=max(err)
    cal=calibrate_dynamic(dt=.05);cal2=calibrate_dynamic(dt=.05)
    assert np.array_equal(cal['alpha'],cal2['alpha']);checks['calibration_reproducibility']='pass'
    assert np.max(abs(cal['width']-WIDTH)/WIDTH)<.02;checks['sensor_estimated_play_width_relative_error']=float(np.max(abs(cal['width']-WIDTH)/WIDTH))
    r=run_integrated(plant,m,n=601,calibration=cal)
    before=np.sqrt(np.mean(r['schemes']['uncompensated']['pos_um']**2));after=np.sqrt(np.mean(r['schemes']['all_remedies']['pos_um']**2))
    assert 0<after<before;checks['coupled_remedy_ratio']=float(after/before)
    assert after>1.;checks['mismatched_plant_not_numerical_zero']='pass'
    u=rng.normal(size=(100,5));v=servo_truth(u,.01,True);u2=u.copy();u2[60:]+=100;v2=servo_truth(u2,.01,True)
    assert np.array_equal(v[:60],v2[:60]);checks['servo_state_causality']='pass'
    b=backlash_truth(u);b2=backlash_truth(u2);assert np.array_equal(b[:60],b2[:60]);checks['play_state_causality']='pass'
    from research_dashboard import sampling_view
    assert sampling_view()['rank']==43;assert sampling_view(single_tilt=True)['rank']<43
    checks['dashboard_pure_callbacks']='pass'
    output=Path(__file__).resolve().parents[1]/'results/research_verification.json';output.write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
    return checks

if __name__=='__main__':run()
