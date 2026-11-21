"""Colab-native research controls; release sliders or click Run to recompute."""
import numpy as np
from pathlib import Path
from research_experiments import load_models,static_test
from independent_plant import IndependentPlant
from periodic_model import candidates,select_poses,sensitivity,NOISE
from dynamic_sim import run_integrated

def sampling_view(budget=28,method='d_optimal',orientation=True,single_tilt=False):
    pool=candidates()
    if single_tilt:pool=pool[pool[:,0]==0]
    ids=select_poses(pool,min(budget,len(pool)),method)
    A=np.concatenate([sensitivity(*r,orientation=orientation) for r in pool[ids]])
    noise=NOISE if orientation else NOISE[:3];A=A/np.tile(noise,len(ids))[:,None]
    s=np.linalg.svd(A,compute_uv=False);rank=int(np.sum(s>s[0]*1e-10))
    return dict(pool=pool,selected=ids,singular=s,rank=rank,readings=len(ids)*(6 if orientation else 3),
                condition=float(s[0]/s[-1]) if rank==43 else float('inf'))

def launch_research():
    import ipywidgets as W
    import matplotlib.pyplot as plt
    from IPython.display import display,clear_output
    models=load_models();plant=IndependentPlant()
    intro=W.HTML('<h2>Calibration fit → trajectory generalization</h2><p>Synthetic independent plant. Event-driven recomputation; not a real-time controller. Original 41-channel and standalone sensor controls remain in the legacy laboratory.</p>')
    turns=W.FloatSlider(value=3,min=.5,max=3,step=.25,description='C turns',continuous_update=False)
    mapout=W.Output();maptext=W.HTML()
    def map_update(*_):
        with mapout:
            clear_output(wait=True);fig,axes=plt.subplots(1,2,figsize=(11,3.5));rms={}
            for kind in ['uncompensated','cubic','periodic']:
                r=static_test(plant,models.get(kind,models['periodic']),turns.value,n=121,compensated=kind!='uncompensated')
                axes[0].plot(r['c_deg'],r['pos_um'],label=kind);axes[1].plot(r['c_deg'],r['ori_urad'],label=kind)
                rms[kind]=np.sqrt(np.mean(r['pos_um']**2))
            axes[0].set(xlabel='C (deg)',ylabel='TCP error (µm)');axes[1].set(xlabel='C (deg)',ylabel='Axis error (µrad)');axes[0].legend()
            display(fig);plt.close(fig)
            maptext.value='<p>TCP RMS (µm): '+', '.join(f'{k}: {v:.3f}' for k,v in rms.items())+'</p><p>Remedy: periodic runout plus a separate scale term for continuous C rotation. Keep finite-travel A cubic. Higher-harmonic mismatch remains; calibration covers only 0–360°.</p>'
    turns.observe(map_update,names='value');maps=W.VBox([turns,maptext,mapout])

    budget=W.IntSlider(value=28,min=12,max=60,step=4,description='Poses',continuous_update=False)
    method=W.Dropdown(options=['uniform','random','d_optimal'],value='d_optimal',description='Design')
    orientation=W.Checkbox(value=True,description='Orientation triad')
    tilt=W.Checkbox(value=False,description='A=0 only')
    calout=W.Output();caltext=W.HTML()
    def cal_update(*_):
        r=sampling_view(budget.value,method.value,orientation.value,tilt.value)
        with calout:
            clear_output(wait=True);fig,axes=plt.subplots(1,2,figsize=(11,3.5));v=r['pool'][r['selected']]
            axes[0].scatter(np.rad2deg(v[:,1]),np.rad2deg(v[:,0]),c=v[:,3],s=v[:,2]/2)
            axes[0].set(xlabel='C (deg)',ylabel='A (deg)',title='Size=radius; color=height')
            axes[1].semilogy(np.arange(1,44),np.maximum(r['singular']/r['singular'][0],1e-16),'o-',markersize=3)
            axes[1].set(xlabel='Mode',ylabel='Normalized singular value');display(fig);plt.close(fig)
            caltext.value=f'<p>Rank {r["rank"]}/43; {r["readings"]} scalar readings; scaled condition {r["condition"]:.3g}.</p><p>Remedy: vary tilt, radius and height before purchasing a more accurate sensor. Orientation sensing adds information but cannot recover unsampled A functions. D-optimal is greedy and model-based; it does not guarantee smaller bias.</p>'
    for w in [budget,method,orientation,tilt]:w.observe(cal_update,names='value')
    sampler=W.VBox([W.HBox([budget,method]),W.HBox([orientation,tilt]),caltext,calout])

    duration=W.FloatSlider(value=30,min=20,max=60,step=10,description='Duration s',continuous_update=False)
    revolutions=W.FloatSlider(value=2.5,min=.5,max=3,step=.5,description='Turns',continuous_update=False)
    scheme=W.Dropdown(options=['uncompensated','geometry','geometry_thermal','geometry_thermal_backlash','all_remedies'],value='all_remedies',description='Remedy')
    sensor=W.Dropdown(options=['TCP and contour','RTD and thermal','Motor/output encoders'],description='Graph')
    button=W.Button(description='Run coupled simulation',button_style='primary');dynout=W.Output();dyntext=W.HTML();cache={}
    def dyn_plot(*_):
        if not cache:return
        r=cache['result'];v=r['schemes'][scheme.value]
        with dynout:
            clear_output(wait=True);fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
            if sensor.value=='TCP and contour':
                for ax,field,label in zip(axes,['pos_um','contour_um'],['TCP following (µm)','Local contour (µm)']):
                    ax.plot(r['t'],r['schemes']['uncompensated'][field],label='Before');ax.plot(r['t'],v[field],label=scheme.value);ax.set_ylabel(label);ax.legend(fontsize=8)
            elif sensor.value=='RTD and thermal':
                axes[0].plot(r['t'],r['RTD']);axes[0].set_ylabel('RTD change (K)')
                axes[1].plot(r['t'],1000*r['thermal_true']);axes[1].plot(r['t'],1000*r['thermal_estimate'],'--');axes[1].set_ylabel('Truth / RTD map (µm)')
            else:
                axes[0].plot(r['t'],1000*(v['q_command'][:,0]-v['q_motor'][:,0]));axes[0].set_ylabel('Command − motor X (µm)')
                axes[1].plot(r['t'],1000*(v['output_encoder'][:,0]-v['q_motor'][:,0]));axes[1].set_ylabel('Output − motor X (µm)')
            axes[-1].set_xlabel('Mechanical time (s)');display(fig);plt.close(fig)
            dyntext.value=f'<p>Selected TCP RMS {np.sqrt(np.mean(v["pos_um"]**2)):.3f} µm; contour RMS {np.sqrt(np.mean(v["contour_um"]**2)):.3f} µm.</p><p>RTDs → thermal map; output encoder → reversal-width estimate; motor encoder → lag estimate. Calibration uses disjoint records. Two-pole/time-varying servo, nonlinear thermal drift and variable play deliberately violate estimator assumptions. Thermal history is accelerated. Initial transients included; no saturation/collision limits modeled.</p>'
    def dyn_run(*_):
        button.disabled=True
        try:cache['result']=run_integrated(plant,models['periodic'],duration=duration.value,n=601,turns=revolutions.value);dyn_plot()
        finally:button.disabled=False
    button.on_click(dyn_run);scheme.observe(dyn_plot,names='value');sensor.observe(dyn_plot,names='value')
    dynamics=W.VBox([W.HBox([duration,revolutions]),button,W.HBox([scheme,sensor]),dyntext,dynout])
    tabs=W.Tab(children=[maps,sampler,dynamics])
    for i,title in enumerate(['Rotary generalization','Calibration design','Coupled errors and sensors']):tabs.set_title(i,title)
    display(W.VBox([intro,tabs]));map_update();cal_update();dyn_run()
    return tabs
