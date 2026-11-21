"""Inspection companion checks; no detector-performance acceptance gate."""
from pathlib import Path
import sys,json,csv,tempfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inspection_dashboard import *

def run():
    checks={}
    assert threshold(np.arange(96.))==92 and np.isinf(threshold(np.arange(96.),.01))
    try: threshold(np.array([np.nan]))
    except ValueError: pass
    else: raise AssertionError('Nonfinite calibration accepted')
    checks['rank_and_invalid_calibration']='pass'
    r=np.array([[1.,2.,3.],[np.nan,0.,0.]])
    V=np.broadcast_to(np.diag([1.,4.,9.]),(2,3,3)).copy()
    s=inspect(r,V,2);assert np.isclose(s['score'][0],np.sqrt(3)) and not s['alarm'][0] and s['quality_error'][1]
    checks['quadratic_score_and_missing_measurement']='pass'
    q=np.zeros((3,5));q[:,4]=np.radians([350,10,180])
    assert np.array_equal(pose_roi(q,c_start=330,c_end=30),[True,True,False])
    assert not pose_roi(q,a_limits=(5,10)).any()
    checks['circular_ROI_and_empty_selection']='pass'
    c=confusion([True,True,False,False],[True,False,True,False]);assert c['tp']==c['fp']==c['tn']==c['fn']==1 and c['f1']==.5
    assert confusion([False],[False])['precision'] is None
    checks['confusion_and_undefined_metrics']='pass'
    a=example();b=example('offset');assert a['cutoff']==b['cutoff']
    assert np.allclose(b['residual_mm'][:,0]-a['residual_mm'][:,0],.003)
    d=example('dropout');assert d['quality_error'].sum()==8
    assert example('harmonic_mismatch')['labels'] is None
    assert np.isinf(example(confidence=.99)['cutoff'])
    import matplotlib.pyplot as plt
    empty=example(a_limits=(80,90));assert not empty['mask'].any()
    fig=plot_inspection(empty);plt.close(fig)
    with tempfile.TemporaryDirectory() as folder:
        p=export_log(d,Path(folder)/'inspection.csv')
        with p.open() as h: rows=list(csv.DictReader(h))
        assert len(rows)==64 and rows[0]['ry_um']=='' and rows[0]['quality_error']=='True'
    checks['reference_frozen_injection_and_export']='pass'
    box=launch_inspection();box.children[1].value='dropout';assert box._inspection_result['quality_error'].sum()==8
    box.children[7].value=10;assert box._inspection_result['mask'].sum()==10
    checks['widget_callbacks']='pass'
    (ROOT/'results/inspection_verification.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2));return checks

if __name__=='__main__':run()
