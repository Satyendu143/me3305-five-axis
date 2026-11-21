"""Colab-friendly interactive window for the final preprint protocol."""
import json
from pathlib import Path
import numpy as np
import ipywidgets as W
from IPython.display import display,clear_output
import matplotlib.pyplot as plt
from publication_studies import ROOT,CONFIGS,METHODS
from research_experiments import load_models
from independent_plant import IndependentPlant
from dynamic_sim import run_integrated

def launch_preprint():
    output=W.Output();mode=W.Dropdown(options=['physical','accelerated'],value='physical',description='Heat clock')
    seconds=W.Dropdown(options=[30,60],value=30,description='Duration [s]')
    run=W.Button(description='Run exact-ZOH preview',button_style='primary')
    schemes=W.SelectMultiple(options=['uncompensated','geometry','geometry_ff','geometry_backlash',
        'geometry_thermal','geometry_thermal_ff','geometry_thermal_backlash','all_remedies'],
        value=('uncompensated','geometry','all_remedies'),description='Schemes',rows=8)
    note=W.HTML('<b>Numerical preview, not a real-time CNC controller.</b> Exact-ZOH servo; '
        '0.02 s preview sampling (paper factorial uses 0.01 s). Accelerated heat is a stress clock. '
        'Assessment probes do not feed back. Full nine-scheme records are computed when Run is clicked.')
    cache={}
    def plot():
        if not cache:return
        with output:
            clear_output(wait=True);r=cache['run'];fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
            for name in schemes.value:
                v=r['schemes'][name];axes[0].plot(r['t'],v['pos_um'],label=name)
                print(name,'TCP RMS [um]',round(float(np.sqrt(np.mean(v['pos_um']**2))),3),
                    'normal-plane RMS [um]',round(float(np.sqrt(np.mean(v['contour_um']**2))),3))
            for j in range(3):axes[1].plot(r['t'],r['RTD'][:,j],label=f'RTD {j+1}')
            final=r['schemes']['all_remedies']
            axes[2].plot(r['t'],1000*(final['output_encoder'][:,0]-final['q_motor'][:,0]),label='X output − motor encoder')
            axes[0].set_ylabel('TCP error (um)');axes[1].set_ylabel('Delta temperature (K)');axes[2].set_ylabel('Encoder difference (um)')
            for ax in axes:ax.legend(fontsize=7);ax.grid(alpha=.2)
            axes[-1].set_xlabel('Time (s)');fig.tight_layout();display(fig);plt.close(fig)
    def compute(_):
        run.disabled=True
        try:
            with output:clear_output(wait=True);print('Running synthetic exact-ZOH plant...')
            duration=seconds.value;cache['run']=run_integrated(IndependentPlant(),load_models()['periodic'],
                duration=duration,n=int(duration/.02)+1,turns=2.5 if duration==30 else 1.25,
                thermal_mode=mode.value,full_factorial=True,servo_integrator='exact_zoh')
            plot()
        except Exception as exc:
            with output:print(type(exc).__name__,str(exc))
        finally:run.disabled=False
    run.on_click(compute);schemes.observe(lambda change:plot(),names='value')
    design=W.Output();strength=W.Dropdown(options=[0.,1.,3.],value=1.,description='Mismatch')
    budget=W.Dropdown(options=[16,28,48],value=48,description='Pose budget')
    def design_plot(_=None):
        with design:
            clear_output(wait=True);m=json.loads((ROOT/'results/publication_metrics.json').read_text())
            rs=[r for r in m['design']['aggregate'] if r['mismatch']==strength.value and r['poses']==budget.value]
            fig,ax=plt.subplots(figsize=(9,4));x=np.arange(len(rs));ax.bar(x,[r['mean_rms_um'] for r in rs],label='Total held-out RMS')
            ax.plot(x,[r['noise_only_rms_um'] for r in rs],'ko',label='Conditional noise only')
            ax.set_xticks(x,[r['method'].replace('_',' ') for r in rs],rotation=12);ax.set_ylabel('Displacement RMS (um)');ax.legend();fig.tight_layout();display(fig);plt.close(fig)
            print('Recorded 12-seed / 3-configuration panel; sliders inspect saved evidence, not new fits.')
    strength.observe(design_plot,names='value');budget.observe(design_plot,names='value');design_plot()
    tabs=W.Tab(children=[W.VBox([note,W.HBox([mode,seconds,run]),schemes,output]),W.VBox([W.HBox([strength,budget]),design])])
    tabs._preprint_cache=cache
    tabs.set_title(0,'Coupled sensors/remedies');tabs.set_title(1,'Noise versus total error');display(tabs)
    return tabs
