"""Explicit data import, rather than guessing sensor units or coordinates."""
import csv
from pathlib import Path
import numpy as np
from calibration import table_sensitivity,S44,solve,refine
from fiveaxis import trajectory,Machine,synthetic_model

def load_ballbar_csv(path):
    records=[]; readings=[]; rows=[]
    with open(path) as fp:
        for row in csv.DictReader(fp):
            axis=row['axis'];a,c,L,H=[float(row[k]) for k in ['a_rad','c_rad','L_mm','H_mm']]
            n=np.array([float(row[k]) for k in ['nx','ny','nz']])
            if abs(np.linalg.norm(n)-1)>1e-6: raise ValueError('Sensor direction must be unit length')
            if axis not in ['A','C']: raise ValueError('axis must be A or C')
            records.append((axis,a,c,L,H,n,int(row['direction_index'])))
            readings.append(float(row['reading_mm']));rows.append(n@table_sensitivity(a,c,L,H)*S44)
    return np.asarray(rows),records,np.asarray(readings)

def fit_ballbar_csv(path,noise_um=.2):
    A,r,y=load_ballbar_csv(path);fit=solve(A,y,noise_um)
    if fit['rank']<44: return fit # do not report uniquely identified parameters
    return refine(r,y,fit['z'],noise_um=noise_um)

def load_cl_csv(path):
    """Cutter locations in PCS: X_mm,Y_mm,Z_mm,I,J,K. No controller G-code."""
    with open(path) as fp: rows=list(csv.DictReader(fp))
    p=np.array([[float(r[k]) for k in ['X_mm','Y_mm','Z_mm']] for r in rows])
    o=np.array([[float(r[k]) for k in ['I','J','K']] for r in rows])
    if np.any(np.linalg.norm(o,axis=1)<1e-12): raise ValueError('Nonzero tool direction required')
    o/=np.linalg.norm(o,axis=1,keepdims=True)
    return p,o

def export_cl_commands(input_csv,output_csv,machine=None):
    if machine is None: machine=Machine(synthetic_model())
    p,o=load_cl_csv(input_csv);r=trajectory(machine,p,o,'full')
    with open(output_csv,'w') as fp:
        w=csv.writer(fp);w.writerow(['x_mm','y_mm','z_mm','a_rad','c_rad','predicted_tcp_error_um','predicted_axis_error_urad'])
        for q,ep,eo in zip(r['q'],r['pos_um'],r['ori_urad']):w.writerow([*q,ep,eo])
    return r
