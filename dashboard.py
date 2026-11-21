"""Colab-native interactive Python window: ipywidgets, Matplotlib, Plotly.
No desktop tkinter dependency and no remote service or account required.
"""
from pathlib import Path
import numpy as np
from fiveaxis import *
from calibration import design,solve,truth,readings,a_design
from remedies import REMEDY_TEXT,thermal,backlash,servo,abbe,sensor_faults

def animation_figure(machine=None,n=81,exaggeration=100):
    import plotly.graph_objects as go
    if machine is None: machine=Machine(synthetic_model())
    t,p,o=helix(n);raw=trajectory(machine,p,o,'none');fixed=trajectory(machine,p,o,'full')
    rawp=p+exaggeration*(raw['p']-p);fixedp=p+exaggeration*(fixed['p']-p)
    def trace(points,name,color):
        return go.Scatter3d(x=points[:,0],y=points[:,1],z=points[:,2],mode='lines',name=name,line=dict(color=color,width=5))
    fig=go.Figure([trace(p,'Target','#1f77b4'),trace(rawp,f'Uncompensated displacement x{exaggeration}','#d62728'),trace(fixedp,'Compensated','#2ca02c'),trace(np.vstack([p[0],p[0]+15*o[0]]),'Tool axis','#111111')])
    frames=[]
    for k in range(n):
        tool=np.vstack([rawp[k],rawp[k]+15*raw['o'][k]])
        frames.append(go.Frame(name=str(k),data=[trace(tool,'Tool axis','#111111')],traces=[3]))
    fig.frames=frames
    fig.update_layout(title='Synthetic XYZ-AC helix | displacement magnification stated in legend',scene=dict(xaxis_title='X [mm]',yaxis_title='Y [mm]',zaxis_title='Z [mm]',aspectmode='data'),height=580,
        updatemenus=[dict(type='buttons',buttons=[dict(label='Play',method='animate',args=[None,{'frame':{'duration':70,'redraw':True},'transition':{'duration':0}}]),dict(label='Pause',method='animate',args=[[None],{'mode':'immediate','frame':{'duration':0,'redraw':False}}])])],
        sliders=[dict(steps=[dict(method='animate',label=str(k),args=[[str(k)],{'mode':'immediate','frame':{'duration':0,'redraw':True}}]) for k in range(0,n,4)])])
    return fig

def export_animation(path=None):
    if path is None: path=Path(__file__).resolve().parent/'results'/'helix_animation.html'
    animation_figure().write_html(str(path),include_plotlyjs=True)
    return str(path)

def launch():
    import ipywidgets as W
    import matplotlib.pyplot as plt
    from IPython.display import display,HTML,clear_output
    title=W.HTML('<h2>Five-axis error laboratory</h2><p>All signals are synthetic. Change controls to recompute; this is an event-driven visualization, not a 1 ms CNC controller.</p>')
    scale=W.FloatSlider(value=1,min=0,max=30,step=.25,description='Error scale',continuous_update=False)
    channel=W.Dropdown(options=[('All 41 channels',-1)]+[(n,k) for k,n in enumerate(error_catalogue())],value=-1,description='Channel')
    amplitude=W.FloatSlider(value=20,min=-100,max=100,step=1,description='um / urad',continuous_update=False)
    scheme=W.Dropdown(options=[('All 41 map','full'),('Positioning LUT only','positioning'),('No compensation','none')],value='full',description='Scheme')
    method=W.Dropdown(options=['paper','damped'],description='Solver')
    exaggeration=W.IntSlider(value=100,min=1,max=1000,description='Visual gain',continuous_update=False)
    out=W.Output(); status=W.HTML()
    def geometry_update(*_):
        with out:
            clear_output(wait=True)
            e=synthetic_model(scale.value)
            if channel.value>=0:
                e=ErrorModel();k=channel.value
                if k<30:
                    a,j=divmod(k,6);e.coeff[a,j,1]=amplitude.value/(1000 if j<3 else 1e6)
                else:
                    j=k-30;e.pige[j]=amplitude.value/(1000 if j in [3,4,7,8] else 1e6)
            m=Machine(e);t,p,o=helix(61);raw=trajectory(m,p,o,'none');r=trajectory(m,p,o,scheme.value,method.value)
            fig,axs=plt.subplots(1,2,figsize=(11,3.5));axs[0].plot(t,raw['pos_um'],label='Before');axs[0].plot(t,r['pos_um'],label='After');axs[0].set(xlabel='Path',ylabel='TCP error [um]');axs[0].legend();axs[1].plot(t,raw['ori_urad'],label='Before');axs[1].plot(t,r['ori_urad'],label='After');axs[1].set(xlabel='Path',ylabel='Axis angle [urad]');display(fig);plt.close(fig)
            import plotly.graph_objects as go
            f=go.Figure()
            for name,pts in [('Target',p),('Before (displacement exaggerated)',p+exaggeration.value*(raw['p']-p)),('After (displacement exaggerated)',p+exaggeration.value*(r['p']-p))]:
                f.add_trace(go.Scatter3d(x=pts[:,0],y=pts[:,1],z=pts[:,2],mode='lines',name=name))
            f.update_layout(height=430,scene=dict(aspectmode='data',xaxis_title='X mm',yaxis_title='Y mm',zaxis_title='Z mm'),title=f'Displacement magnification x{exaggeration.value}')
            display(f)
            status.value=f'<b>After:</b> max TCP {max(r["pos_um"]):.4g} um; max axis {max(r["ori_urad"]):.4g} urad. <p>{REMEDY_TEXT["Geometry"]}</p>'
    for w in [scale,channel,amplitude,scheme,method,exaggeration]: w.observe(geometry_update,names='value')
    geom=W.VBox([W.HBox([scale,channel]),W.HBox([amplitude,scheme,method]),exaggeration,status,out])

    problem=W.Dropdown(options=['Thermal','Backlash','Servo lag','Abbe offset','Sensor fault'],description='Problem')
    noise=W.FloatSlider(value=.5,min=0,max=3,step=.1,description='Sensor um',continuous_update=False)
    magnitude=W.FloatSlider(value=24,min=1,max=100,description='Magnitude',continuous_update=False)
    sensorout=W.Output();advice=W.HTML()
    def sensor_update(*_):
        name=problem.value
        if name=='Thermal': r=thermal(noise_um=noise.value);x=r['t']/60;before=r['before'][:,2];after=r['after'][:,2];extra=r['sensors'];extra_unit='RTD rise [K]';xlabel='Time [min]'
        elif name=='Backlash': r=backlash(magnitude.value,noise.value);x=r['t'];before=r['sensor_before'];after=r['sensor_after'];extra=None;xlabel='Time [s]'
        elif name=='Servo lag': r=servo(magnitude.value,noise.value);x=r['t'];before=r['sensor_before'];after=r['sensor_after'];extra=None;xlabel='Time [s]'
        elif name=='Abbe offset': r=abbe(magnitude.value,noise.value);x=r['x'];before=r['before'];after=r['after'];extra=r['angle_sensor'];extra_unit='Angle sensor [urad]';xlabel='Travel [mm]'
        else: r=sensor_faults(noise.value);x=r['t'];before=r['before'];after=r['after'];extra=r['sensors'];extra_unit='Redundant signals [um]';xlabel='Time [s]'
        with sensorout:
            clear_output(wait=True);fig,ax=plt.subplots(2 if extra is not None else 1,1,figsize=(10,6 if extra is not None else 3.5),squeeze=False)
            ax[0,0].plot(x,before,label='Before signal error');ax[0,0].plot(x,after,label='After signal error');ax[0,0].set(xlabel=xlabel,ylabel='Error [um]');ax[0,0].legend()
            if extra is not None: ax[1,0].plot(x,extra);ax[1,0].set(xlabel=xlabel,ylabel=extra_unit)
            display(fig);plt.close(fig)
        advice.value=f'<p>{REMEDY_TEXT[name]}</p><b>Physical RMS (all axes for thermal):</b> {r["rms_before"]:.3g} → {r["rms_after"]:.3g} um. Magnitude units: backlash um, servo ms, Abbe mm; fixed illustrative temperature profile for thermal.'
    for w in [problem,noise,magnitude]: w.observe(sensor_update,names='value')
    sensors=W.VBox([W.HBox([problem,noise,magnitude]),advice,sensorout])

    sensor=W.Dropdown(options=['ballbar','rtest'],description='Sensor')
    coverage=W.Dropdown(options=['All directions','Radial only'],description='Coverage')
    gap=W.FloatLogSlider(value=50,base=10,min=-2,max=2,description='H1-H mm',continuous_update=False)
    radius=W.FloatSlider(value=100,min=0,max=150,step=5,description='L mm',continuous_update=False)
    calout=W.Output();caltext=W.HTML()
    def calibration_update(*_):
        A,rec=design(sensor.value,H=50,H1=50+gap.value,L=radius.value,n=13)
        if coverage.value=='Radial only': mask=np.array([r[-1]==1 for r in rec]);A=A[mask];rec=[r for r,b in zip(rec,mask) if b]
        f=solve(A,readings(rec,truth(),sensor.value,noise_um=noise.value),noise.value)
        with calout:
            clear_output(wait=True);fig,ax=plt.subplots(figsize=(9,3.5));ax.semilogy(np.maximum(f['singular'],1e-18),'o-');ax.set(xlabel='Singular-value index',ylabel='Scaled sensitivity');display(fig);plt.close(fig)
        caltext.value=f'<b>Rank {f["rank"]}/44; condition {f["condition"]:.3g}.</b><p>{REMEDY_TEXT["Poor calibration"]}</p><p>With equal noise and complete directional coverage, 3D R-test and three ballbar projections contain equivalent first-order position information; simultaneous acquisition changes timing sensitivity. R-test radial-only here means only its Y channel.</p>'
    for w in [sensor,coverage,gap,radius,noise]:w.observe(calibration_update,names='value')
    calibration=W.VBox([W.HBox([sensor,coverage]),W.HBox([gap,radius]),caltext,calout])

    tilt=W.FloatLogSlider(value=10,base=10,min=-4,max=1.77,description='Tilt A deg',continuous_update=False)
    singout=W.Output();singtext=W.HTML()
    def singular_update(*_):
        a=np.deg2rad(tilt.value);J=Machine().jacobian(np.array([0,0,-100,a,0.]))@np.diag(Q_SCALE);ss=np.linalg.svd(J,compute_uv=False)
        with singout:
            clear_output(wait=True);fig,ax=plt.subplots(figsize=(8,3.5));ax.bar(np.arange(5),ss);ax.set(yscale='log',xlabel='Task Jacobian mode',ylabel='Singular value');display(fig);plt.close(fig)
        singtext.value=f'<b>Condition {ss[0]/ss[-1]:.3g}; azimuth gain {1/np.sin(a):.3g}.</b><p>{REMEDY_TEXT["Singularity"]}</p>'
    tilt.observe(singular_update,names='value');sing=W.VBox([tilt,singtext,singout])
    tabs=W.Tab(children=[geom,sensors,calibration,sing])
    for i,n in enumerate(['41 errors and compensation','Sensor remedies','Calibration observability','IK singularity']): tabs.set_title(i,n)
    display(W.VBox([title,tabs]));geometry_update();sensor_update();calibration_update();singular_update()
    return tabs
