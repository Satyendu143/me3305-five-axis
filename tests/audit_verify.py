"""Checks against scalar kinematics and independent covariance calculations."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit_models import *
from journal_models import perturb_model,tcp_covariance,conformal_threshold
from fiveaxis import Machine
from publication_studies import matrix,solver,offsets
from periodic_model import candidates,rtest_signal
from independent_plant import IndependentPlant
class ModelPlant(IndependentPlant):
    def __init__(self,model):super().__init__();self.model=model;self.pige=model.pige
    def errors(self,q):return self.model.evaluate(q)
def run():
    rng=np.random.default_rng(101008);checks={}
    z=rng.normal(size=(4,43));eta=rng.normal(size=(4,65));q=np.c_[rng.uniform(-100,100,(8,2)),rng.uniform(-180,-110,8),rng.uniform(-1,1,8),rng.uniform(0,10,8)]
    pred=batch_fk(z,eta,q);ref=np.array([[Machine(perturb_model(a,b)).fk(v)[0] for v in q] for a,b in zip(z,eta)])
    assert np.max(abs(pred-ref))<2e-12;checks['batch_exact_fk_vs_scalar_max_mm']=float(np.max(abs(pred-ref)))
    pool=candidates()[::5];sig=batch_signal(z,eta,pool)
    ref=np.array([rtest_signal(ModelPlant(perturb_model(a,b)),pool) for a,b in zip(z,eta)])
    assert np.max(abs(sig-ref))<2e-12;checks['batch_measurement_vs_independent_frame_max_mm']=float(np.max(abs(sig-ref)))
    off=batch_offset(eta,pool);ref=np.array([offsets(pool,(b*ETA_SCALE)[:11]) for b in eta]);assert np.max(abs(off-ref))<2e-12
    checks['batch_nuisance_offset_vs_scalar']='pass'
    f=solver(matrix(pool),np.arange(len(pool)));base=tcp_covariance(q[:3],pool,f)
    R=np.eye(6);R[:3,:3]=.6+np.eye(3)*.4
    out=covariance_variants(base,f['pinv'],R);P=f['pinv'];D=base['Dw'];C=auxiliary_covariance();A=P@D
    Cz=P@np.kron(np.eye(len(pool)),R)@P.T+A@C@A.T
    full=np.block([[Cz,-A@C],[-C@A.T,C]])
    J=np.concatenate([base['H'],base['G']],axis=-1);ref=np.einsum('nij,jk,nlk->nil',J,full,J)
    assert np.max(abs(out['joint']-ref))<1e-18;checks['correlated_joint_covariance_vs_full_block']='pass'
    naive=np.block([[Cz,np.zeros((43,65))],[np.zeros((65,43)),C]])
    ref=np.einsum('nij,jk,nlk->nil',J,naive,J)
    assert np.max(abs(out['naive']-ref))<1e-18;checks['naive_ablation_retains_both_marginal_covariances']='pass'
    eta_noise=rng.multivariate_normal(np.zeros(65),C,18000);eps=rng.normal(size=(18000,len(pool),6))@np.linalg.cholesky(R).T
    dz=eps.reshape(18000,-1)@P.T-eta_noise@A.T
    E=np.einsum('nij,sj->sni',base['H'],dz)+np.einsum('nij,sj->sni',base['G'],eta_noise)
    emp=np.einsum('sni,snj->nij',E,E)/len(E);rel=np.linalg.norm(emp-out['joint'])/np.linalg.norm(out['joint']);assert rel<.04
    checks['correlated_raw_MC_relative_covariance_error']=float(rel)
    for scores,alpha in [([],0.05),([float('nan')],.05),([1],0),([1],1),([[1]],.05)]:
        try:conformal_threshold(scores,alpha)
        except ValueError:pass
        else:raise AssertionError('invalid score/alpha accepted')
    checks['conformal_rejects_invalid_inputs']='pass'
    assert np.isinf(conformal_threshold(np.arange(96.),.01));checks['conformal_finite_sample_unbounded_branch']='pass'
    empirical=np.linalg.eigvalsh(out['joint']);assert empirical.min()>0
    checks['correlated_covariance_positive_definite']='pass'
    (Path(__file__).resolve().parents[1]/'results/audit_verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2));return checks
if __name__=='__main__':run()
