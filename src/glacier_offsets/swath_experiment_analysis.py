"""Frozen physical comparison, empirical validation, closures, and reports."""
import json
from pathlib import Path

import numpy as np

from glacier_offsets.products import write_json
from glacier_offsets.swath_models import evaluate, error_metrics, paired_metrics
from glacier_offsets.swath_physics import metadata, COMPONENTS
from glacier_offsets.swath_experiment_io import read_json, load_npz, cohorts, COHORTS


def observation_rows(experiment, cohort):
    rows, ledger = [], {}
    for link in experiment.links:
        root = experiment.output/'measurements'/link['pair']
        report = root/'report.json'
        if not report.exists() or read_json(report).get('status') != 'complete':
            ledger[link['pair']] = {'status': 'unavailable'}
            continue
        experiment.checked_report(report)
        data = load_npz(root/'observations.npz')
        mask = cohorts(data)[cohort]
        physical_root = experiment.output/'physics'/link['pair']
        if (physical_root/'predictions.npz').exists():
            experiment.checked_report(physical_root/'report.json', complete=False)
            predictions = load_npz(physical_root/'predictions.npz')
        else:
            predictions = {}
        difference = data['iw2_total']-data['iw1_total']
        owner1, owner2 = data['iw1_owner'], data['iw2_owner']
        groups = sorted(set(zip(owner1[mask].tolist(), owner2[mask].tolist())))
        counts = {'input_supported': int(data['dual_support'].sum()),
                  'geometry_eligible': int((data['dual_support'] & data['geometry_eligible']).sum()),
                  'interior': int((data['dual_support'] & data['burst_interior']).sum()),
                  'native_valid': int(data['dual_valid'].sum()), 'selected': int(mask.sum())}
        ledger[link['pair']] = {'status': 'complete', 'counts': counts}
        for a, b in groups:
            selected = mask & (owner1 == a) & (owner2 == b)
            values = difference[..., 0][selected]
            row = dict(link, burst1=str(data['source_ids'][a-1]), burst2=str(data['source_ids'][b-1]),
                       mean=float(values.mean()), values=values.tolist(), range_values=difference[..., 1][selected].tolist(),
                       center_line=data['center_line'][selected].tolist(),
                       center_sample=data['center_sample'][selected].tolist(),
                       physical={c: predictions.get(f'{c}_prediction_px', np.full(mask.shape, np.nan))[selected].tolist()
                                 for c in COMPONENTS})
            rows.append(row)
    return rows, ledger


def closure_report(experiment, cohort):
    observations = {}
    for link in experiment.links:
        path = experiment.output/'measurements'/link['pair']/'observations.npz'
        if path.exists() and read_json(path.parent/'report.json').get('status') == 'complete':
            observations[link['pair']] = load_npz(path)
    result = {}
    for a, b, c in zip(experiment.dates, experiment.dates[1:], experiment.dates[2:]):
        keys = [f'{a}_{b}', f'{b}_{c}', f'{a}_{c}']
        if any(key not in observations for key in keys):
            result[f'{a}_{b}_{c}'] = {'status': 'unavailable'}
            continue
        views = [observations[key] for key in keys]
        mask = np.logical_and.reduce([cohorts(view)[cohort] for view in views])
        for other in views[1:]:
            for key in ('center_line', 'center_sample'):
                np.testing.assert_array_equal(views[0][key], other[key])
            for swath in ('iw1', 'iw2'):
                mask &= views[0][f'{swath}_owner'] == other[f'{swath}_owner']
        delta = [view['iw2_total']-view['iw1_total'] for view in views]
        closure = delta[0]+delta[1]-delta[2]
        result[f'{a}_{b}_{c}'] = {'status': 'complete', 'n': int(mask.sum()),
            'azimuth_px': closure[..., 0][mask].tolist(), 'range_px': closure[..., 1][mask].tolist(),
            'center_line': views[0]['center_line'][mask].tolist(),
            'center_sample': views[0]['center_sample'][mask].tolist(),
            'interpretation': 'raw same-owner temporal closure; date-additive fitted fields cancel by construction'}
    return result


def predictions_for_rows(rows, model):
    if model == 'baseline':
        return [0.0]*len(rows)
    return [np.array(row['physical'][model]) for row in rows]


def fit(experiment, draws=1000):
    physics_status = experiment.output/'physics_status.json'
    if not physics_status.exists():
        raise ValueError('Run physics first; empirical fitting cannot precede physical publication')
    physics_identity = experiment.inputs.file(physics_status)
    specification = {'experiment': experiment.key, 'physics_sha256': physics_identity, 'draws': draws,
                     'implementation': experiment.implementation('swath_models.py', 'swath_experiment_analysis.py',
                                                                 'swath_experiment_io.py'),
                     'physics_reports': {x['pair']: experiment.inputs.hashes(experiment.output/'physics'/x['pair']/'report.json')
                        for x in experiment.links if (experiment.output/'physics'/x['pair']/'report.json').exists()},
                     'measurements': {x['pair']: experiment.inputs.hashes(experiment.output/'measurements'/x['pair']/'report.json')
                        for x in experiment.links if (experiment.output/'measurements'/x['pair']/'report.json').exists()}}
    if experiment.current(experiment.output/'fit.json', specification):
        return experiment.checked_report(experiment.output/'fit.json')
    missions = {d: experiment.inventory[d]['mission'] for d in experiment.dates}
    meta = {d: metadata(experiment, d) for d in experiment.dates}
    report = {'status': 'complete', 'cohorts': {}, 'metadata': meta,
              'physical_status': read_json(physics_status)['status'],
              'physical_predictions_frozen_sha256': physics_identity,
              'empirical_target': 'raw observations; no physical residual fitting'}
    for cohort in COHORTS:
        rows, ledger = observation_rows(experiment, cohort)
        models = evaluate(rows, experiment.dates, missions, draws=draws)
        physical = {c: error_metrics(rows, predictions_for_rows(rows, c)) for c in COMPONENTS}
        paired = paired_metrics(rows, {c: predictions_for_rows(rows, c) for c in ('baseline', *COMPONENTS)})
        for fold in models['folds']:
            testing = [rows[i] for i in fold['test_rows']]
            predictions = {name: entry['prediction_px'] for name, entry in fold['models'].items()}
            predictions.update({c: predictions_for_rows(testing, c) for c in ('baseline', *COMPONENTS)})
            # Physical coverage cannot silently remove windows from empirical-vs-baseline scores.
            fold['empirical_common'] = paired_metrics(testing, {k: predictions[k] for k in
                                                                ('baseline', 'spacecraft', 'per_acquisition')})
            fold['all_methods_common'] = paired_metrics(testing, predictions)
            fold['physical_metrics'] = {c: error_metrics(testing, predictions[c]) for c in COMPONENTS}
        # Stratifications are declared before observing their effect on residuals.
        breakdowns = {}
        selectors = {
            'spacecraft_order': lambda r: missions[r['earlier']]+'->'+missions[r['later']],
            'ipf': lambda r: meta[r['earlier']]['ipf']+'->'+meta[r['later']]['ipf'],
            'burst_group': lambda r: r['burst1']+'|'+r['burst2']}
        for name, grouping in selectors.items():
            breakdowns[name] = {}
            for label in sorted({grouping(r) for r in rows}):
                subset = [r for r in rows if grouping(r) == label]
                breakdowns[name][label] = {c: error_metrics(subset, predictions_for_rows(subset, c))
                                         for c in ('baseline', *COMPONENTS)}
        models.update(rows=rows, ledger=ledger, physical_metrics=physical,
                      physical_common=paired, raw_closures=closure_report(experiment, cohort), breakdowns=breakdowns)
        for kind in ('leave_acquisition_out', 'leave_link_out', 'leave_burst_group_out'):
            models.setdefault('validation_breakdowns', {})[kind] = {}
            for name, grouping in selectors.items():
                grouped = {}
                for label in sorted({grouping(r) for r in rows}):
                    selected_rows, empirical = [], {m: [] for m in ('spacecraft', 'per_acquisition')}
                    for fold in models['folds']:
                        if fold['kind'] != kind:
                            continue
                        for j, i in enumerate(fold['test_rows']):
                            if grouping(rows[i]) != label:
                                continue
                            selected_rows.append(rows[i])
                            for m in empirical:
                                empirical[m].append(fold['models'][m]['prediction_px'][j])
                    predictions = dict(empirical, baseline=[0]*len(selected_rows),
                                       physical=predictions_for_rows(selected_rows, 'combined'))
                    grouped[label] = {'individual': {m: error_metrics(selected_rows, p) for m, p in predictions.items()},
                                      'common': paired_metrics(selected_rows, predictions)}
                models['validation_breakdowns'][kind][name] = grouped
        report['cohorts'][cohort] = models
    products = []
    for cohort, record in report['cohorts'].items():
        path = experiment.output/'analysis'/f'{cohort}.json'
        write_json(path, record)
        products.append(path)
    plot_comparison(experiment, report['cohorts']['primary'])
    products.append(experiment.output/'analysis/comparison.png')
    return experiment.finish(experiment.output/'fit.json', specification,
        {k: v for k, v in report.items() if k != 'cohorts'}, products)


def plot_comparison(experiment, record):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
    rows = record['rows']
    dates = experiment.dates
    x = np.array([dates.index(r['later']) for r in rows])
    observed = np.array([r['mean'] for r in rows])
    physical = np.array([np.mean(r['physical']['combined']) for r in rows])
    axes[0, 0].scatter(x, observed, s=15, label='Observed burst-pair means')
    axes[0, 0].scatter(x, physical, s=15, label='Unfitted physical candidate')
    axes[0, 0].set(xlabel='Later acquisition index', ylabel='IW2 − IW1 azimuth [px]')
    axes[0, 0].legend()
    axes[0, 1].scatter(physical, observed, s=15)
    axes[0, 1].set(xlabel='Physical candidate [px]', ylabel='Observed burst-pair mean [px]')
    if np.isfinite(physical).any():
        low, high = np.nanmin(physical), np.nanmax(physical)
        axes[0, 1].plot([low, high], [low, high], color='black', linewidth=.7)
    for row in rows:
        axes[1, 0].scatter(row['center_line'], np.array(row['values'])-np.array(row['physical']['combined']),
                           s=4, alpha=.2, color='tab:blue')
    axes[1, 0].set(xlabel='Original reference azimuth line', ylabel='Observed − physical [px]')
    model = record['models']['per_acquisition']
    terms = np.array(model['parameters'])
    covariance = model['covariance']
    sigma = np.sqrt(np.maximum(0, np.diag(covariance))) if covariance is not None else np.zeros(len(dates))
    axes[1, 1].errorbar(np.arange(len(dates)), terms, yerr=1.96*sigma, fmt='o', markersize=3,
                        label='All-data per-date fit (descriptive)')
    axes[1, 1].set(xlabel='Acquisition index', ylabel='Relative date term [px]')
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.axhline(0, color='gray', linewidth=.5)
        ax.grid(alpha=.2)
    fig.suptitle('Unspliced production range-2x overlap — primary population; physical candidates unvalidated')
    fig.savefig(experiment.output/'analysis/comparison.png', dpi=140)
    plt.close(fig)


def cross_validated_summary(record, kind):
    rows = record['rows']
    testing, prediction = [], {k: [] for k in ('baseline', 'physical', 'spacecraft', 'per_acquisition')}
    for fold in record['folds']:
        if fold['kind'] != kind:
            continue
        for j, index in enumerate(fold['test_rows']):
            row = rows[index]
            testing.append(row)
            prediction['baseline'].append(0)
            prediction['physical'].append(np.asarray(row['physical']['combined']))
            for model in ('spacecraft', 'per_acquisition'):
                prediction[model].append(fold['models'][model]['prediction_px'][j])
    return {'per_model': {k: error_metrics(testing, p) for k, p in prediction.items()},
            'common': paired_metrics(testing, prediction)}


def report(experiment):
    root = experiment.output
    lines = ['# Larsen unspliced range-2x swath experiment', '',
             f'Production source: `{experiment.stack}` (read only).',
             f'Planned acquisitions: {len(experiment.dates)}; links: {len(experiment.links)}.', '',
             'Physical candidates were generated before fitting the raw-data empirical models.',
             'Timing conventions remain candidate hypotheses unless independently validated.', '']
    summary = {'experiment': experiment.key, 'status': 'complete', 'stages': {}, 'conclusions': {}}
    for stage, file in (('audit', 'audit.json'), ('measure', 'measurement_status.json'),
                        ('physics', 'physics_status.json'), ('fit', 'fit.json'), ('previews', 'preview_status.json')):
        if not (root/file).exists():
            summary['stages'][stage] = 'pending'
            summary['status'] = 'incomplete'
            continue
        data = read_json(root/file)
        if stage == 'measure':
            complete = sum(v.get('status') == 'complete' for v in data.values())
            status = 'complete' if complete == len(experiment.links) else 'incomplete'
            lines += [f'Measured links: {complete}/{len(experiment.links)}.', '']
        else:
            status = data.get('status', 'incomplete')
        summary['stages'][stage] = status
        if status != 'complete':
            summary['status'] = 'incomplete'
    for cohort in COHORTS:
        path = root/'analysis'/f'{cohort}.json'
        if not path.exists():
            continue
        record = read_json(path)
        lines += [f'## {cohort}', '',
                  f"Usable graph: {record['graph']['links']} links; {record['graph']['cycles']} cycles; "
                  f"{len(record['graph']['bridges'])} bridges.", '',
                  '| Validation | Model | Windows | Links | MAE (px) | RMSE (px) |',
                  '| --- | --- | ---: | ---: | ---: | ---: |']
        summary['conclusions'][cohort] = {}
        for kind in ('leave_acquisition_out', 'leave_link_out', 'leave_burst_group_out'):
            scores = cross_validated_summary(record, kind)
            summary['conclusions'][cohort][kind] = scores
            for name, metrics in scores['per_model'].items():
                fmt = lambda v: f'{v:.5f}' if v is not None else 'unavailable'
                lines.append(f"| {kind} | {name} | {metrics['n']} | {metrics['links']} | "
                             f"{fmt(metrics['mae_px'])} | {fmt(metrics['rmse_px'])} |")
        conclusions = {}
        for model, kinds in (('spacecraft', ('leave_acquisition_out', 'leave_burst_group_out')),
                             ('per_acquisition', ('leave_link_out', 'leave_burst_group_out'))):
            scores = [record['validation_score_intervals'].get(kind, {}).get(model) for kind in kinds]
            if all(s and s['descriptive_interval95_px'] and s['descriptive_interval95_px'][1] < 0 for s in scores):
                conclusion = 'supported improvement within the declared descriptive validation'
            elif all(s and s['mae_change_vs_baseline_px'] >= 0 for s in scores):
                conclusion = 'no improvement'
            else:
                conclusion = 'inconclusive'
            conclusions[model] = conclusion
            lines += ['', f'{model}: **{conclusion}**.']
        comparison = record['physical_common']['models']
        baseline_mae = comparison.get('baseline', {}).get('mae_px')
        physical_mae = comparison.get('combined', {}).get('mae_px')
        conclusions['physical'] = ('no improvement in differential residuals; physical conventions remain unvalidated'
            if baseline_mae is not None and physical_mae is not None and physical_mae >= baseline_mae else
            'inconclusive physical attribution; candidates retain unresolved historical conventions')
        lines += ['', f"physical: **{conclusions['physical']}**."]
        summary['conclusions'][cohort]['classification'] = conclusions
        lines += ['', 'Pairwise comparisons use common finite windows; full coverage and attrition are retained in JSON.',
                  'Per-acquisition predictions for held-out dates are unavailable, not interpolated.', '']
    lines += ['## Interpretation', '',
              'A lower held-out error is descriptive evidence of improvement on this dataset. '
              'Unsupported physical conventions, inadequate support, or inconsistent cycles make the result inconclusive. '
              'No improvement is a valid experiment outcome. Spatially overlapping windows are not independent samples.', '',
              'Map previews use symmetric relative allocation and a spatially constant swath contrast outside the overlap. '
              'They do not establish absolute motion accuracy. Raw observations remain unchanged.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(root/'summary.json', summary)
    return summary
