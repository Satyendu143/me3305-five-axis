"""Meaningful independent checks: algebra, matrix exponential, signs, inverse,
calibration observability, numerical convergence, estimator holdout behavior.
Run: python tests/verify.py
"""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.linalg import expm
import sympy as sp
from fiveaxis import *
from calibration import *
from remedies import *

def run():
    checks={};rng=np.random.default_rng(8)
    errors=[]
    for _ in range(30):
        v=rng.normal(size=3);w=rng.normal(size=3);th=rng.uniform(-2,2)
        hat=np.zeros((4,4));hat[:3,:3]=skew(w);hat[:3,3]=v
        errors.append(np.max(abs(exp_twist(v,w,th)-expm(hat*th))))
    assert max(errors)<5e-14;checks['SE3_vs_scipy_expm']=max(errors)
    for w in [np.zeros(3),np.array([1e-12,0,0])]:
        v=np.array([.1,.2,.3]);g=exp_twist(v,w);hat=np.zeros((4,4));hat[:3,:3]=skew(w);hat[:3,3]=v
        assert np.allclose(g,expm(hat),atol=1e-14)
    checks['small_angle_and_translation_limit']='pass'
    ideal=Machine(tool_home=np.array([2.,3.,150.]),part_home=np.array([10.,-5.,20.]))
    maxerr=0
    for _ in range(100):
        q=np.r_[rng.uniform(-100,100,3),rng.uniform(-1,1),rng.uniform(-6,6)]
        p,o=ideal.fk(q);Ri=(rz(q[4])@rx(q[3]))[:3,:3]
        assert np.allclose(p,Ri@(q[:3]+ideal.tool_home)-ideal.part_home)
        qr=ideal.ik(p,o,previous=q);pr,orr=ideal.fk(qr);maxerr=max(maxerr,np.linalg.norm(pr-p))
        assert np.linalg.norm(pr-p)<1e-10 and np.linalg.norm(orr-o)<1e-12
    checks['IK_FK_100_poses_with_home_offsets_mm']=maxerr
    ref=np.array([0.,0.,1.]);a,c=.7,1.3;v=(rz(c)@rx(a))[:3,:3]@ref
    solutions=paden_kahan_orientation(ref,v,[1,0,0],[0,0,1])
    assert len(solutions)==2
    for aa,cc in solutions:assert np.linalg.norm((rz(cc)@rx(aa))[:3,:3]@ref-v)<1e-12
    checks['general_Paden_Kahan_two_branches']='pass'
    p,o=ideal.fk([1,2,3,0,1.2]);q=ideal.ik(p,o,previous=np.array([1,2,3,0,1.2]));assert abs(q[4]-1.2)<1e-12
    checks['singularity_branch_keeps_previous_C']='pass'
    true=synthetic_model();m=Machine(true);q=np.array([25,40,-80,.6,1.3]);g=m.matrix(q)
    assert np.allclose(g[:3,:3].T@g[:3,:3],np.eye(3),atol=1e-14);assert abs(np.linalg.det(g[:3,:3])-1)<1e-14
    checks['exact_error_transform_stays_in_SE3']='pass'
    # Symbolically prove the corrected inverse without dividing by sin/cos.
    c,s,H,H1,L=sp.symbols('c s H H1 L',nonzero=True)
    dx,dy,dz,ex,ey,ez=sp.symbols('dx dy dz ex ey ez')
    ra1=dx+H*(c*ey+s*ez);rr1=c*dz-s*dy;rt1=s*dz+c*dy-H*ex
    rr2=rr1;rt2=s*dz+c*dy-H1*ex
    rr3=rr1-L*(c*ey+s*ez);rt3=rt1+L*(-s*ey+c*ez)
    recovered_ex=-(rt2-rt1)/(H1-H)
    recovered_dy=-s*rr1+c*(rt1+H*recovered_ex)
    recovered_dz=c*rr1+s*(rt1+H*recovered_ex)
    recovered_ey=-c*(rr3-rr1)/L-s*(rt3-rt1)/L
    recovered_ez=-s*(rr3-rr1)/L+c*(rt3-rt1)/L
    recovered_dx=ra1-H*(c*recovered_ey+s*recovered_ez)
    for expr,expected in zip([recovered_dx,recovered_dy,recovered_dz,recovered_ex,recovered_ey,recovered_ez],[dx,dy,dz,ex,ey,ez]):
        angle=sp.symbols('angle',real=True)
        assert sp.trigsimp(sp.simplify(expr-expected).subs({c:sp.cos(angle),s:sp.sin(angle)}))==0
    checks['symbolic_corrected_Eq29']='pass'
    maxerr=0
    for a in [0.,np.pi/2,-np.pi/2,.3,-.5]:
        e=rng.normal(size=6)*[.01,.01,.01,1e-5,1e-5,1e-5];r=a_readings(a,50,100,100,e)
        maxerr=max(maxerr,np.max(abs(a_corrected(r,a,50,100,100)-e)))
    assert maxerr<1e-16;checks['corrected_inverse_at_trigonometric_edge_cases']=maxerr
    e=np.array([.012,-.008,.005,30e-6,-20e-6,15e-6]);a=np.deg2rad(-30);r=a_readings(a,50,100,100,e)
    lit=a_printed_eq29(r,a,50,100,100)
    assert abs(lit[3]+e[3])<1e-15 and abs(lit[2]-(e[2]-100*e[3]*np.sin(-a)))<1e-15
    checks['printed_Eq29_counterexample']={'eps_xa_true_urad':30,'printed_urad':float(lit[3]*1e6),'delta_za_true_um':5,'printed_um':float(lit[2]*1000)}
    A,rec=design(n=13);maxerr=0
    for j in range(44):
        z=np.zeros(44);z[j]=1e-3
        numerical=(exact_predictions(rec,z)-exact_predictions(rec,np.zeros(44)))/1e-3
        maxerr=max(maxerr,np.max(abs(numerical-A[:,j])))
    assert maxerr<2e-7;checks['44_parameter_sensitivity_vs_exact_finite_difference']=maxerr
    y=A@truth();fit=solve(A,y,noise_um=0);assert fit['rank']==44;assert np.max(abs(fit['z']-truth()))<1e-8
    checks['noiseless_linear_calibration_recovery_scaled']=float(max(abs(fit['z']-truth())))
    y=readings(rec,truth(),noise_um=0);fit=refine(rec,y,solve(A,y,noise_um=0)['z'],noise_um=0)
    assert max(abs(fit['z']-truth()))<1e-3 and max(abs(fit['residual_um']))<1e-5
    checks['noiseless_exact_ballbar_recovery_scaled']=float(max(abs(fit['z']-truth())))
    mask=np.array([r[-1]==1 for r in rec]);assert np.linalg.matrix_rank(A[mask])<44
    checks['radial_only_unidentifiability_rank']=int(np.linalg.matrix_rank(A[mask]))
    S=np.vstack([a_design(.3,50,0),a_design(.3,50,0),a_design(.3,50,100)])
    assert np.linalg.matrix_rank(S)<6
    checks['equal_height_degeneracy_rank']=int(np.linalg.matrix_rank(S))
    AR,_=design('rtest',n=13);assert np.allclose(A.T@A,AR.T@AR,atol=1e-13)
    checks['ballbar_three_directions_Rtest_information_equivalence']='pass'
    t,p,o=helix(31);r=trajectory(m,p,o,'full');assert max(r['pos_um'])<1e-3 and max(r['ori_urad'])<1e-3
    checks['full_compensation_max_um']=float(max(r['pos_um']))
    qi,hi=compensate(m,p[2],o[2],method='damped',iterations=8);assert hi[-1,0]<1e-6
    checks['damped_compensation_mm']=float(hi[-1,0])
    assert np.isclose(pcs_map([.1,.02,.003],35,0),0);checks['PCS_datum_zero']='pass'
    from io_tools import load_ballbar_csv
    path=Path(__file__).resolve().parents[1]/'results'/'ballbar_training.csv'
    if path.exists():
        AA,rr,yy=load_ballbar_csv(path);assert AA.shape==(675,44);checks['CSV_sensor_roundtrip']='pass'
    for name,fn in [('thermal',thermal),('backlash',backlash),('servo',servo),('abbe',abbe),('sensor_faults',sensor_faults)]:
        r=fn();assert r['rms_after']<r['rms_before'];checks[name+'_remedy_ratio']=r['rms_after']/r['rms_before']
    output=Path(__file__).resolve().parents[1]/'results'/'verification.json';output.write_text(json.dumps(checks,indent=2));return checks

if __name__=='__main__':print(json.dumps(run(),indent=2))
