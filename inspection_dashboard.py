"""Inspection-style sensor replay, inspired by MathWorks workflow features.

Not a MATLAB implementation, camera detector, causal diagnosis, live DAQ or
new research algorithm. Normal-reference scores use held-out XYZ metrology.
"""
from pathlib import Path
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
CASES = ['normal', 'offset', 'sparse_spikes', 'dropout',
         'harmonic_mismatch', 'localized_mismatch']


def threshold(scores, alpha=.05):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Calibration scores must be a nonempty finite vector')
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must be strictly between zero and one')
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return float(np.partition(scores, k - 1)[k - 1]) if k <= len(scores) else np.inf


def inspect(residual, covariance, cutoff):
    """Flag missing values separately; never impute them as normal responses."""
    residual = np.asarray(residual, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if residual.ndim != 2 or residual.shape[1] != 3:
        raise ValueError('Residual shape must be [N,3], in mm')
    if covariance.shape != (len(residual), 3, 3) or not np.isfinite(covariance).all():
        raise ValueError('Covariance shape must be [N,3,3], in mm squared')
    if not np.allclose(covariance, covariance.swapaxes(-1, -2), atol=1e-14):
        raise ValueError('Covariance must be symmetric')
    if np.linalg.eigvalsh(covariance).min() <= 0:
        raise ValueError('Covariance must be positive definite')
    if np.isnan(cutoff) or cutoff < 0:
        raise ValueError('Cutoff must be nonnegative, possibly infinite')
    valid = np.isfinite(residual).all(axis=1)
    score = np.full(len(residual), np.nan)
    d = residual[valid]
    score[valid] = np.sqrt(np.maximum(
        np.einsum('ni,ni->n', d, np.linalg.solve(covariance[valid], d[..., None])[..., 0]), 0))
    # Marginal standardized coordinates help locate symptoms, not causes.
    coordinate = residual / np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
    anomaly = valid & (score > cutoff)
    return dict(valid=valid, score=score, coordinate=coordinate,
                anomaly=anomaly, quality_error=~valid, alarm=anomaly | ~valid)


def pose_roi(q, a_limits=(-60., 60.), c_start=0., c_end=360.):
    q = np.asarray(q)
    if q.ndim != 2 or q.shape[1] != 5 or not np.isfinite(q).all():
        raise ValueError('Commands must be a finite [N,5] array, mm/rad')
    lo, hi = a_limits
    if lo > hi or not (0 <= c_start <= 360 and 0 <= c_end <= 360):
        raise ValueError('Invalid A or C region bounds')
    a = np.degrees(q[:, 3]); c = np.degrees(q[:, 4]) % 360
    cmask = ((c >= c_start) & (c <= c_end)) if c_start <= c_end else ((c >= c_start) | (c <= c_end))
    return (a >= lo) & (a <= hi) & cmask


def confusion(labels, alarm):
    labels = np.asarray(labels, dtype=bool); alarm = np.asarray(alarm, dtype=bool)
    if labels.shape != alarm.shape or labels.ndim != 1:
        raise ValueError('Labels and alarms must be equal-length vectors')
    tp = int(np.sum(labels & alarm)); fp = int(np.sum(~labels & alarm))
    fn = int(np.sum(labels & ~alarm)); tn = int(np.sum(~labels & ~alarm))
    ratio = lambda x, y: float(x / y) if y else None
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=ratio(tp, tp+fp),
                recall=ratio(tp, tp+fn), false_positive_rate=ratio(fp, fp+tn),
                f1=ratio(2*tp, 2*tp+fp+fn))


def example(case='normal', confidence=.95, offset_um=3.,
            a_limits=(-60., 60.), c_start=0., c_end=360., through=64):
    if case not in CASES:
        raise ValueError('Unknown inspection case')
    if not np.isfinite(offset_um) or not 1 <= through <= 64:
        raise ValueError('Invalid offset or replay length')
    data = ROOT/'results' if (ROOT/'results/journal_tcp_covariance.npz').exists() else ROOT/'data'
    with np.load(data/'journal_tcp_covariance.npz') as z:
        q = z['q'].copy(); V = z['observed_cov'].copy()
    with np.load(data/'journal_prediction_examples.npz') as z:
        normal = z['matched__measured'] - z['matched__pred']
        selected = case if case.endswith('mismatch') else 'matched'
        residual = (z[selected+'__measured'] - z[selected+'__pred'])[96:].copy()
    cutoff = threshold(inspect(normal[:96], V[:96], 0)['score'], 1-confidence)
    labels = np.zeros(64, dtype=bool)
    if case == 'offset': residual[:, 0] += offset_um/1000; labels[:] = offset_um != 0
    if case == 'sparse_spikes': residual[::8, 0] += offset_um/1000; labels[::8] = offset_um != 0
    if case == 'dropout': residual[::8, 1] = np.nan; labels[::8] = True
    if case.endswith('mismatch'): labels = None
    result = inspect(residual, V[96:], cutoff)
    mask = pose_roi(q[96:], a_limits, c_start, c_end) & (np.arange(64) < through)
    return dict(**result, q=q[96:], residual_mm=residual, covariance_mm2=V[96:],
                cutoff=cutoff, mask=mask, labels=labels, case=case,
                confidence=confidence, offset_um=offset_um)


def guidance(result):
    mask = result['mask']; n = int(mask.sum())
    if not n: return 'No records in this region. Widen A/C bounds or replay more poses.'
    if result['quality_error'][mask].any():
        return ('Missing/nonfinite readings: inspect acquisition, synchronization and sensor connection; '
                'use independent reference metrology. Missing data is not evidence of normal operation.')
    if result['anomaly'][mask].any():
        return ('A response differs from the normal reference. Repeat measurements with independent XYZ '
                'metrology; inspect reference frames and sensor bias. Check RTDs/time history for thermal '
                'association and angular metrology for pose-dependent error. These are candidate checks, '
                'not an identified root cause. Recalibrating a threshold alone does not fix accuracy.')
    return ('No statistical alarm in this view. This does not certify machining tolerance, latent TCP '
            'accuracy, or the absence of an undetected fault.')


def export_log(result, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    columns = ['pose_index','a_deg','c_deg','rx_um','ry_um','rz_um','score',
               'threshold','selected','valid','statistical_alarm','quality_error']
    with path.open('w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=columns); w.writeheader()
        for i in range(len(result['q'])):
            residual = [float(v*1000) if np.isfinite(v) else '' for v in result['residual_mm'][i]]
            w.writerow(dict(zip(columns, [i, float(np.degrees(result['q'][i,3])),
                float(np.degrees(result['q'][i,4]) % 360), *residual,
                float(result['score'][i]) if np.isfinite(result['score'][i]) else '',
                float(result['cutoff']) if np.isfinite(result['cutoff']) else 'unbounded',
                bool(result['mask'][i]), bool(result['valid'][i]),
                bool(result['anomaly'][i]), bool(result['quality_error'][i])])))
    return path


def plot_inspection(result):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    index = np.arange(64); mask = result['mask']; good = mask & result['valid']
    axes[0,0].plot(index[good], result['score'][good], '.', label='XYZ score')
    if np.isfinite(result['cutoff']): axes[0,0].axhline(result['cutoff'], color='red', ls='--', label='Normal-reference cutoff')
    axes[0,0].set(xlabel='Assessment pose index (not physical time)', ylabel='Quadratic-distance score'); axes[0,0].legend(fontsize=8)
    d = np.ma.masked_invalid(np.where(mask[:,None], result['coordinate'], np.nan).T)
    im = axes[0,1].imshow(d, aspect='auto', cmap='coolwarm', vmin=-6, vmax=6)
    axes[0,1].set(yticks=[0,1,2], yticklabels=['X','Y','Z'], xlabel='Assessment pose index', title='Marginal standardized residuals')
    fig.colorbar(im, ax=axes[0,1], label='Residual / marginal SD')
    q = result['q']; color = result['score'][good]
    im = axes[1,0].scatter(np.degrees(q[good,4]) % 360, np.degrees(q[good,3]), c=color, cmap='viridis', s=30)
    alerts = mask & result['alarm']
    axes[1,0].scatter(np.degrees(q[alerts,4]) % 360, np.degrees(q[alerts,3]), marker='x', c='red', label='Alarm')
    axes[1,0].set(xlabel='C command (deg)', ylabel='A command (deg)', title='Pose-space localization, not a camera image'); axes[1,0].legend(fontsize=8)
    fig.colorbar(im, ax=axes[1,0], label='XYZ score')
    for j, axis in enumerate('XYZ'): axes[1,1].plot(index[mask], 1000*result['residual_mm'][mask,j], '.-', lw=.6, label=axis)
    axes[1,1].set(xlabel='Assessment pose index', ylabel='Noisy-response residual (um)'); axes[1,1].legend()
    fig.suptitle(f"{result['case']} — fixed normal reference; {mask.sum()} selected poses")
    fig.tight_layout(); return fig


def launch_inspection():
    import ipywidgets as W
    import matplotlib.pyplot as plt
    from IPython.display import display, clear_output
    case = W.Dropdown(options=CASES, description='Case')
    confidence = W.FloatSlider(value=.95, min=.80, max=.99, step=.01, description='Reference level', continuous_update=False)
    offset = W.FloatSlider(value=3, min=0, max=8, step=.5, description='Offset (um)', continuous_update=False)
    a = W.FloatRangeSlider(value=(-60,60), min=-60, max=60, step=5, description='A ROI (deg)', continuous_update=False)
    start = W.FloatSlider(value=0, min=0, max=360, step=15, description='C start', continuous_update=False)
    end = W.FloatSlider(value=360, min=0, max=360, step=15, description='C end', continuous_update=False)
    replay = W.IntSlider(value=64, min=1, max=64, description='Replay through', continuous_update=False)
    button = W.Button(description='Export inspection CSV'); out = W.Output(); saved = W.Output()
    box = W.VBox([W.HTML('<b>Sensor inspection companion</b><br>Saved synthetic response replay, not live DAQ. '
        'Statistical anomalies are not causal diagnoses or dimensional accept/reject decisions. '
        'Threshold uses only 96 matched-reference calibration responses; ROI is descriptive.'),
        case,confidence,offset,a,start,end,replay,button,out,saved])
    def draw(change=None):
        r = example(case.value, confidence.value, offset.value, a.value, start.value, end.value, replay.value)
        box._inspection_result = r
        with out:
            clear_output(wait=True); fig = plot_inspection(r); display(fig); plt.close(fig)
            n = int(r['mask'].sum()); alarms = int(r['alarm'][r['mask']].sum())
            print(f'Alarms: {alarms}/{n}; invalid: {int(r["quality_error"][r["mask"]].sum())}.')
            if not np.isfinite(r['cutoff']): print('Unbounded statistical threshold: 96 calibration points are insufficient for a finite 99% region.')
            if r['labels'] is not None and n: print('Known injected-label metrics:', confusion(r['labels'][r['mask']], r['alarm'][r['mask']]))
            else: print('No fault labels for this case: precision/recall are not inferred.')
            print(guidance(r))
    def save(event):
        p = export_log(box._inspection_result, ROOT/'inspection/inspection_log.csv')
        with saved:
            clear_output(); print('Saved:', p)
            try:
                from google.colab import files
                files.download(str(p))
            except ImportError: pass
    for control in [case,confidence,offset,a,start,end,replay]: control.observe(draw, names='value')
    button.on_click(save); draw(); display(box); return box


def demo_report():
    """Single saved example, not an independent detection benchmark."""
    folder = ROOT/'inspection'; folder.mkdir(exist_ok=True); rows = []
    for case in CASES:
        r = example(case); row = dict(case=case, records=64, alarms=int(r['alarm'].sum()),
            invalid=int(r['quality_error'].sum()), labels_available=r['labels'] is not None)
        if r['labels'] is not None: row.update(confusion(r['labels'], r['alarm']))
        rows.append(row)
    record = dict(scope='One saved synthetic example; no repeated-trial or physical detection claim',
                  alpha=.05, normal_calibration_records=96, assessment_records=64, cases=rows)
    (folder/'inspection_metrics.json').write_text(json.dumps(record, indent=2, allow_nan=False))
    import matplotlib.pyplot as plt
    r = example('sparse_spikes'); fig = plot_inspection(r)
    fig.savefig(folder/'inspection_spike_example.png', dpi=180, bbox_inches='tight'); plt.close(fig)
    export_log(r, folder/'inspection_example.csv'); return record


if __name__ == '__main__': print(json.dumps(demo_report(), indent=2))
