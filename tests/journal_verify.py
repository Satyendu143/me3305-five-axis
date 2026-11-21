"""Independent checks of journal spline, nuisance and prediction-region additions."""
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from journal_models import *
from publication_studies import solver
from periodic_model import candidates,select_poses,rtest_signal,auxiliary_calibration
from validation_plant import LocalizedPlant,MatchedPlant

def run():
    checks={};rng=np.random.default_rng(1208);pool=candidates();ids=select_poses(pool,60)
    for knots in [5,9]:
        b=PeriodicBasis(knots)
        assert np.max(abs(b(0)))==0
        assert np.max(abs(b(.37)-b(.37+2*np.pi)))<1e-14
        for derivative in [0,1,2]:
            assert np.max(abs(b.curve(0,derivative)-b.curve(2*np.pi,derivative)))<1e-13
        checks[f'spline{knots}_periodic_C2_and_datum']='pass'
        M=spline_matrix(pool,knots);fit=solver(M,ids);assert fit['rank']==18+6*(knots-1)+1
        checks[f'spline{knots}_full_rank']=fit['rank']
        z=rng.normal(size=M.shape[-1])*1e-5;model=spline_from_z(z,np.zeros(11),np.zeros((3,6,4)),knots)
        p=MatchedPlant();row=pool[47];a,c,L,H=row
        point,R=p.table(*row,errors=model.evaluate([0,0,0,a,c]),pige=np.zeros(11),return_rotation=True)
        R0=(rx(-a)@rz(-c))[:3,:3]
        from periodic_model import rotation_vector_small
        obs=np.r_[point-R0@np.array([L,0,H]),LEVER*rotation_vector_small(R@R0.T)]
        assert np.max(abs(obs-M[47]@z))<1e-10
        checks[f'spline{knots}_exact_vs_sensitivity']='pass'
    plant=LocalizedPlant()
    for c in [.1,1.4,5.7]:
        a=plant.errors([0,0,0,.3,c]);b=plant.errors([0,0,0,.3,c+2*np.pi]);delta=b[4]-a[4];delta[5]-=2*np.pi*plant.c_scale
        assert max(abs(delta))<1e-14
    checks['localized_numpy_truth_periodicity']='pass'
    source=(Path(__file__).resolve().parents[1]/'validation_plant.py').read_text()
    assert not any(s in source for s in ['from scipy','from journal_models','from fiveaxis','from periodic_model'])
    checks['second_truth_no_estimator_imports']='pass'
    q=np.array([[20,-30,-150,.3,.4],[-40,25,-125,-.5,4.5]])
    H,G=tcp_jacobians(q);H2,G2=tcp_jacobians(q,step=.0005)
    assert np.max(abs(H-H2))<1e-7 and np.max(abs(G-G2))<1e-7
    checks['108_column_TCP_derivative_step_stability']='pass'
    f=solver(matrix(pool),ids[:48]);cov=tcp_covariance(q,pool[ids[:48]],f)
    assert np.linalg.eigvalsh(cov['full']).min()>0
    assert np.linalg.eigvalsh(cov['full']-cov['rotary']).min()>-1e-16
    checks['TCP_covariance_positive_and_auxiliary_PSD']='pass'
    # Monte Carlo with independent raw rotary/auxiliary metrology; retained
    # covariance between the shared PIGE subtraction and TCP prediction.
    eta=rng.multivariate_normal(np.zeros(65),auxiliary_covariance(),12000).T
    eps=rng.normal(size=(len(ids[:48])*6,12000));z=f['pinv']@(eps-cov['Dw']@eta)
    E=np.einsum('nij,jt->nit',cov['H'],z)+np.einsum('nij,jt->nit',cov['G'],eta)
    empirical=np.einsum('nit,njt->nij',E,E)/12000
    rel=np.linalg.norm(empirical-cov['full'])/np.linalg.norm(cov['full']);assert rel<.035
    checks['shared_nuisance_covariance_vs_raw_MC_relative_error']=float(rel)
    wrong=cov['rotary']+np.einsum('nij,jk,nlk->nil',cov['G'],auxiliary_covariance(),cov['G'])
    assert np.linalg.norm(wrong-cov['full'])>1e-10
    checks['incomplete_auxiliary_covariance_differs_from_joint']='pass'
    scores=np.arange(96.);assert conformal_threshold(scores)==92
    assert np.isinf(conformal_threshold(np.arange(4.),alpha=.01))
    checks['conformal_finite_sample_order_statistic']='pass'
    S=abs(rng.normal(size=(15000,97)));coverage=np.mean(S[:,-1]<=np.partition(S[:,:96],92,axis=1)[:,92])
    assert .949<coverage<.969;checks['conformal_exchangeable_rank_control']=float(coverage)
    ROOT=Path(__file__).resolve().parents[1]
    if (ROOT/'results/journal_prediction_examples.npz').exists():
        from journal_dashboard import region_preview,launch_journal
        a=region_preview();b=region_preview(rotary_factor=2,auxiliary_factor=2)
        assert np.all(b['gaussian_width_um']>a['gaussian_width_um'])
        checks['live_covariance_sensor_SD_scaling']='pass'
        assert np.isinf(region_preview(confidence=.99)['threshold'])
        checks['99_percent_region_requires_more_calibration']='pass'
        box=launch_journal();box.children[1].value='localized_mismatch'
        assert box._journal_preview['conformal_observed_coverage']==region_preview('localized_mismatch')['conformal_observed_coverage']
        box.children[4].value=.99;assert np.isinf(box._journal_preview['threshold'])
        checks['journal_widget_callbacks_and_unbounded_remedy']='pass'
    (ROOT/'results/journal_verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
    return checks
if __name__=='__main__':run()
