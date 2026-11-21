"""Colab-compatible live uncertainty-assumption and prediction-region explorer."""
from pathlib import Path
import numpy as np
from scipy.stats import chi2
from journal_models import squared_distance,conformal_threshold
ROOT=Path(__file__).resolve().parent

def region_preview(case='matched',rotary_factor=1.,auxiliary_factor=1.,confidence=.95):
    with np.load(ROOT/'results/journal_tcp_covariance.npz') as a:
        rotary=a['rotary_cov'].copy();full=a['full_cov'].copy()
    with np.load(ROOT/'results/journal_prediction_examples.npz') as a:
        residual=a[case+'__measured']-a[case+'__pred']
    covariance=rotary*rotary_factor**2+(full-rotary)*auxiliary_factor**2+np.eye(3)[None]*.0005**2
    score=np.sqrt(np.maximum(squared_distance(residual,covariance),0))
    threshold=conformal_threshold(score[:96],alpha=1-confidence)
    full_threshold=np.sqrt(chi2.ppf(confidence,3))
    scale=1000*np.sqrt(covariance[96:,0,0])
    return dict(residual_um=1000*residual[96:,0],gaussian_width_um=scale*full_threshold,
        conformal_width_um=scale*threshold,gaussian_observed_coverage=float(np.mean(score[96:]<=full_threshold)),
        conformal_observed_coverage=float(np.mean(score[96:]<=threshold)),threshold=threshold)

def launch_journal():
    import ipywidgets as W
    import matplotlib.pyplot as plt
    from IPython.display import display,clear_output
    case=W.Dropdown(options=['matched','harmonic_mismatch','localized_mismatch'],description='Case')
    rot=W.FloatSlider(value=1,min=.25,max=3,step=.25,description='Rotary SD',continuous_update=False)
    aux=W.FloatSlider(value=1,min=.25,max=3,step=.25,description='Auxiliary SD',continuous_update=False)
    conf=W.FloatSlider(value=.95,min=.80,max=.99,step=.01,description='Coverage',continuous_update=False)
    out=W.Output();box=W.VBox([W.HTML('<b>Journal uncertainty audit</b><br>Measurements stay fixed. Sliders change assumed SD multipliers; conformal regions are recalibrated on 96 separate noisy poses. This is an assessment tool, not controller feedback.'),case,rot,aux,conf,out])
    def draw(change=None):
        result=region_preview(case.value,rot.value,aux.value,conf.value);box._journal_preview=result
        with out:
            clear_output(wait=True);fig,ax=plt.subplots(figsize=(10,4));x=np.arange(64)
            g=result['gaussian_width_um'];c=result['conformal_width_um']
            if np.all(np.isfinite(c)):
                ax.fill_between(x,-c,c,alpha=.2,label='Projected conformal XYZ region')
            else:
                print('Conformal set is unbounded at this confidence with 96 calibration poses. Add calibration measurements (at least 99 for a finite 99% set) or request a lower confidence.')
            ax.plot(x,g,'--',color='orange',label='Projected Gaussian XYZ region');ax.plot(x,-g,'--',color='orange')
            ax.plot(x,result['residual_um'],'.-',lw=.7,label='Held-out probe x residual')
            ax.set_xlabel('IID assessment-pose index');ax.set_ylabel('Residual (µm)');ax.legend(fontsize=8);fig.tight_layout();display(fig);plt.close(fig)
            print('Observed XYZ coverage for this single saved example: Gaussian',round(100*result['gaussian_observed_coverage'],1),'%, conformal',round(100*result['conformal_observed_coverage'],1),'%. This is not a new Monte Carlo run.')
    for control in [case,rot,aux,conf]:control.observe(draw,names='value')
    draw();display(box);return box
