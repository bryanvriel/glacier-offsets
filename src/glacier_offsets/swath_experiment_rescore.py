"""Post hoc link-mean evaluation of immutable swath-experiment observations.

No matching, physical prediction, or production correction is performed here.
The prespecified window-level experiment remains a separate, unchanged result.
"""
from collections import Counter, defaultdict
from pathlib import Path
import logging

import numpy as np

from glacier_offsets.products import identity, write_json
from glacier_offsets.swath_models import (bootstrap_parameters, design, graph_audit,
    link_weights, multiplier_draws, paired_metrics, predict, train, validation_folds)
from glacier_offsets.swath_experiment_io import COHORTS, read_json

METHOD = 'posthoc-link-means-v1'
MINIMUM_SUPPORT = (0, 20, 30, 50)
MODELS = ('physical', 'spacecraft', 'per_acquisition')
VALIDATIONS = ('leave_acquisition_out', 'leave_link_out', 'leave_burst_group_out')
ORDERS = ('AB', 'BA', 'AA', 'BB')


def select_rows(rows, minimum):
    """Keep complete links by selected-window count, never by offset values.

    The threshold is evaluated before folds, on the original quality cohort.
    Actual counts after spatial withholding are published separately.
    """
    counts = Counter()
    for row in rows:
        counts[row['pair']] += len(row['values'])
    indexes = [i for i, row in enumerate(rows) if counts[row['pair']] >= minimum]
    return indexes, [rows[i] for i in indexes]


def summarize_links(rows, metadata):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['pair']].append(row)
    result = []
    for pair, group in sorted(grouped.items()):
        a, b = group[0]['earlier'], group[0]['later']
        result.append({'pair': pair, 'earlier': a, 'later': b,
            'order': metadata[a]['mission'][-1]+metadata[b]['mission'][-1],
            'ipf': metadata[a]['ipf']+'→'+metadata[b]['ipf'],
            'windows': sum(len(r['values']) for r in group), 'burst_groups': len(group),
            'mean_px': float(np.mean([np.mean(r['values']) for r in group])),
            'physical_px': {c: float(np.mean([np.mean(r['physical'][c]) for r in group]))
                            for c in ('bistatic', 'fm_rate', 'combined')}})
    return result


def _quantile(values, weights, p):
    order = np.argsort(values)
    index = min(np.searchsorted(np.cumsum(weights[order]), p), len(order)-1)
    return float(np.asarray(values)[order[index]])


def score_instances(instances):
    """Score matched link means, equally weighting links across repeated folds."""
    if not instances:
        return {'links': 0, 'evaluation_instances': 0, 'unique_test_windows': 0,
                'baseline_mae_px': None, 'model_mae_px': None, 'mae_change_px': None,
                'baseline_rmse_px': None, 'model_rmse_px': None, 'rmse_change_px': None}
    weights = link_weights(instances)
    weights /= weights.sum()
    target = np.array([x['observed_mean_px'] for x in instances])
    residual = np.array([x['residual_px'] for x in instances])
    unique_windows = set()
    for item in instances:
        for source, indexes in item['window_selection']:
            unique_windows.update((source, j) for j in indexes)
    baseline_mae, model_mae = float(weights @ abs(target)), float(weights @ abs(residual))
    baseline_rmse, model_rmse = float(np.sqrt(weights @ target**2)), float(np.sqrt(weights @ residual**2))
    return {'links': len({x['pair'] for x in instances}), 'evaluation_instances': len(instances),
            'unique_test_windows': len(unique_windows), 'signed_residual_px': float(weights @ residual),
            'baseline_mae_px': baseline_mae, 'model_mae_px': model_mae,
            'mae_change_px': model_mae-baseline_mae, 'baseline_rmse_px': baseline_rmse,
            'model_rmse_px': model_rmse, 'rmse_change_px': model_rmse-baseline_rmse,
            'median_absolute_residual_px': _quantile(abs(residual), weights, .5),
            'p90_absolute_residual_px': _quantile(abs(residual), weights, .9)}


def link_instances(rows, source_indexes, point, samples, multipliers):
    """Average matched windows within groups, then groups within each test link.

    Shared-factor draws reweight groups inside links and links themselves.
    Overlapping windows are never bootstrapped as independent observations.
    """
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        target = np.asarray(row['values'], dtype=float)
        prediction = np.broadcast_to(np.asarray(point[i], dtype=float), target.shape)
        valid = np.isfinite(target) & np.isfinite(prediction)
        if not valid.any():
            continue
        draw = np.asarray(samples[:, i], dtype=float)
        if not np.isfinite(draw).all():
            continue
        grouped[row['pair']].append((i, float(target[valid].mean()), float(prediction[valid].mean()),
                                    np.flatnonzero(valid).tolist()))
    instances, draw_records = [], []
    for pair, group in sorted(grouped.items()):
        indices = [i for i, _, _, _ in group]
        target = np.array([y for _, y, _, _ in group])
        prediction = np.array([p for _, _, p, _ in group])
        factors = multipliers[:, indices]
        denominator = factors.sum(axis=1)
        target_draw = (factors*target[None, :]).sum(axis=1)/denominator
        prediction_draw = (factors*samples[:, indices]).sum(axis=1)/denominator
        observed, predicted = float(target.mean()), float(prediction.mean())
        instances.append({'pair': pair, 'observed_mean_px': observed, 'prediction_mean_px': predicted,
            'residual_px': observed-predicted, 'windows': sum(len(w) for _, _, _, w in group),
            'burst_groups': len(group), 'source_rows': [source_indexes[i] for i in indices],
            'window_selection': [[source_indexes[i], w] for i, _, _, w in group]})
        draw_records.append({'pair': pair, 'target': target_draw,
                             'prediction': prediction_draw, 'weight': factors.mean(axis=1)})
    return instances, draw_records


def score_intervals(instances, records, sufficient=True):
    if not instances or not sufficient:
        return {'mae_change_interval95_px': None, 'rmse_change_interval95_px': None}
    base = link_weights(instances)
    weights = np.stack([r['weight'] for r in records], axis=1)*base[None, :]
    weights /= weights.sum(axis=1, keepdims=True)
    target = np.stack([r['target'] for r in records], axis=1)
    residual = target-np.stack([r['prediction'] for r in records], axis=1)
    mae = np.sum(weights*(abs(residual)-abs(target)), axis=1)
    rmse = np.sqrt(np.sum(weights*residual**2, axis=1))-np.sqrt(np.sum(weights*target**2, axis=1))
    return {'mae_change_interval95_px': np.quantile(mae, [.025, .975]).tolist(),
            'rmse_change_interval95_px': np.quantile(rmse, [.025, .975]).tolist()}


def coefficient_summary(rows, dates, missions, multipliers):
    fit = train(rows, dates, missions, 'spacecraft')
    if fit['rank'] != 1:
        return {'beta_px': None, 'interval95_px': None, 'links': len({r['pair'] for r in rows})}
    draws = bootstrap_parameters(fit, multipliers)[:, 0]
    enough = len({(r['burst1'], r['burst2']) for r in rows}) >= 2
    return {'beta_px': float(fit['parameters'][0]), 'links': len({r['pair'] for r in rows}),
            'interval95_px': np.quantile(draws, [.025, .975]).tolist() if enough else None,
            'std_px': float(draws.std(ddof=1)) if enough else None}


def order_summary(links):
    summary = {}
    for order in ORDERS:
        selected = [x for x in links if x['order'] == order]
        values = np.array([x['mean_px'] for x in selected])
        summary[order] = {'links': len(values), 'mean_px': float(values.mean()) if values.size else None,
                          'median_px': float(np.median(values)) if values.size else None}
    mixed = [x for x in links if x['order'] in ('AB', 'BA')]
    matches = [x for x in mixed if x['mean_px']*(1 if x['order'] == 'AB' else -1) > 0]
    summary['sign_consistency'] = {'consistent': len(matches), 'mixed_links': len(mixed),
        'exceptions': [x['pair'] for x in mixed if x not in matches],
        'interpretation': 'descriptive count; no independent-link binomial p-value'}
    ab, ba = summary['AB']['mean_px'], summary['BA']['mean_px']
    summary['mixed_order_even_component_px'] = (ab+ba)/2 if ab is not None and ba is not None else None
    return summary


def evaluate_variant(original_rows, metadata, minimum=0, draws=1000, seed=0):
    source_indexes, rows = select_rows(original_rows, minimum)
    dates = sorted(metadata)
    missions = {d: metadata[d]['mission'] for d in dates}
    multipliers = multiplier_draws(rows, draws, seed)
    links = summarize_links(rows, metadata)
    result = {'minimum_windows': minimum, 'links': links, 'order': order_summary(links),
        'graph': graph_audit(rows, dates), 'source_rows': source_indexes,
        'coefficient': coefficient_summary(rows, dates, missions, multipliers), 'ipf': {},
        'folds': [], 'validation': {}, 'selection_timing': 'total cohort windows per link before any withholding'}
    version_draws = {}
    for version in sorted({v['ipf'] for v in metadata.values()}):
        mask = np.array([metadata[r['earlier']]['ipf'] == metadata[r['later']]['ipf'] == version for r in rows], bool)
        subset = [r for r, take in zip(rows, mask) if take]
        result['ipf'][version] = coefficient_summary(subset, dates, missions, multipliers[:, mask])
        fitted = train(subset, dates, missions, 'spacecraft')
        if fitted['rank'] == 1:
            version_draws[version] = bootstrap_parameters(fitted, multipliers[:, mask])[:, 0]
    versions = sorted(version_draws)
    if len(versions) == 2:
        delta = version_draws[versions[1]]-version_draws[versions[0]]
        result['ipf_difference'] = {'later_minus_earlier': versions,
            'beta_difference_px': result['ipf'][versions[1]]['beta_px']-result['ipf'][versions[0]]['beta_px'],
            'interval95_px': np.quantile(delta, [.025, .975]).tolist(),
            'method': 'joint draws preserve source-burst factors across regimes; cross-regime links excluded'}
    accumulators = {(kind, model): ([], [], True) for kind in VALIDATIONS for model in MODELS}
    loo_coefficients = []
    folds = validation_folds(rows, dates) if rows else []
    for kind, label, training_mask, testing_mask in folds:
        if not testing_mask.any():
            continue
        training = [r for r, take in zip(rows, training_mask) if take]
        testing = [r for r, take in zip(rows, testing_mask) if take]
        test_indexes = [i for i, take in zip(source_indexes, testing_mask) if take]
        fold = {'kind': kind, 'label': label,
            'train_rows': [i for i, take in zip(source_indexes, training_mask) if take],
            'test_rows': test_indexes, 'training_graph': graph_audit(training, dates), 'models': {}}
        enough = len({(r['burst1'], r['burst2']) for r in training}) >= 2
        for model in MODELS:
            if model == 'physical':
                point = [np.asarray(r['physical']['combined']) for r in testing]
                # Draws change group weights, but never calibrate physical fields.
                matched_means = []
                for row, prediction in zip(testing, point):
                    valid = np.isfinite(prediction) & np.isfinite(row['values'])
                    matched_means.append(float(prediction[valid].mean()) if valid.any() else np.nan)
                sample = np.broadcast_to(np.array(matched_means),
                                         (draws, len(testing))).copy()
                sufficient = True  # Physical fields have no fitted training parameters.
            else:
                fitted = train(training, dates, missions, model)
                point = predict(fitted, testing, dates, missions)
                sample = bootstrap_parameters(fitted, multipliers[:, training_mask]) @ design(testing, dates, missions, model).T
                sample[:, ~np.isfinite(point)] = np.nan
                sufficient = enough
                if model == 'spacecraft' and kind == 'leave_link_out' and fitted['rank'] == 1:
                    loo_coefficients.append(float(fitted['parameters'][0]))
            instances, score_draws = link_instances(testing, test_indexes, point, sample, multipliers[:, testing_mask])
            covered = {x['pair'] for x in instances}
            fold['models'][model] = {'instances': instances,
                'unavailable_links': sorted({r['pair'] for r in testing}-covered),
                'insufficient_training_groups': not sufficient,
                'prediction_availability': 'unseen dates and disconnected endpoint contrasts are unavailable'}
            previous, previous_draws, previous_sufficient = accumulators[kind, model]
            previous.extend(instances)
            previous_draws.extend(score_draws)
            accumulators[kind, model] = previous, previous_draws, previous_sufficient and (sufficient or not instances)
        result['folds'].append(fold)
    for kind in VALIDATIONS:
        result['validation'][kind] = {}
        for model in MODELS:
            instances, records, sufficient = accumulators[kind, model]
            source_rows = {i for item in instances for i in item['source_rows']}
            sufficient &= len({(original_rows[i]['burst1'], original_rows[i]['burst2']) for i in source_rows}) >= 2
            result['validation'][kind][model] = dict(score_instances(instances),
                **score_intervals(instances, records, sufficient),
                insufficient_groups=not sufficient,
                unavailable_link_instances=sum(len(f['models'][model]['unavailable_links'])
                    for f in result['folds'] if f['kind'] == kind))
    result['leave_link_out_beta_range_px'] = [min(loo_coefficients), max(loo_coefficients)] if loo_coefficients else None
    return result


def original_matched_window_scores(record):
    """Correct the display populations while preserving all original estimates."""
    result = {}
    for kind in VALIDATIONS:
        testing, predictions = [], {m: [] for m in MODELS}
        for fold in record['folds']:
            if fold['kind'] != kind:
                continue
            for j, index in enumerate(fold['test_rows']):
                row = record['rows'][index]
                testing.append(row)
                predictions['physical'].append(np.asarray(row['physical']['combined']))
                for model in ('spacecraft', 'per_acquisition'):
                    predictions[model].append(fold['models'][model]['prediction_px'][j])
        result[kind] = {}
        for model, prediction in predictions.items():
            result[kind][model] = paired_metrics(testing, {'baseline': [0.]*len(testing), 'model': prediction})
    return result


def rescore(experiment, minimum_support=MINIMUM_SUPPORT, draws=1000):
    """Read only frozen numerical artifacts; write a separate post hoc report."""
    if draws < 2 or any(type(n) is not int or n < 0 for n in minimum_support):
        raise ValueError('Use at least two draws and nonnegative integer support thresholds')
    root = experiment.output
    frozen = read_json(root/'plan.json')
    fit = read_json(root/'fit.json')
    if (frozen['identity'] != experiment.key or fit.get('status') != 'complete' or
            fit['specification']['experiment'] != experiment.key or
            fit['identity'] != identity(fit['specification'])):
        raise ValueError('A complete matching frozen experiment is required')
    original = {}
    for filename in ('plan.json', 'fit.json', 'summary.json', 'REPORT.md', 'analysis/comparison.png'):
        original[filename] = experiment.inputs.file(root/filename)
    records = {}
    for cohort in COHORTS:
        path = root/'analysis'/f'{cohort}.json'
        original[f'analysis/{cohort}.json'] = experiment.inputs.file(path, fit['products'][str(path.resolve())])
        records[cohort] = read_json(path)
    spec = {'experiment': experiment.key, 'method': METHOD, 'original_artifacts': original,
        'minimum_support': sorted(set(minimum_support)), 'draws': draws, 'seed': 0,
        'post_hoc': True, 'implementation': {
            str(Path(__file__).resolve()): experiment.inputs.hashes(__file__),
            str(Path(__file__).with_name('swath_models.py')):
                experiment.inputs.hashes(Path(__file__).with_name('swath_models.py'))}}
    out = root/'rescoring'/METHOD
    if experiment.current(out/'record.json', spec):
        return read_json(out/'record.json')
    out.mkdir(parents=True, exist_ok=True)
    result = {'method': METHOD, 'post_hoc': True, 'metadata': fit['metadata'], 'cohorts': {},
        'policy': {'target': 'equal mean of burst-pair group means per link',
            'score': 'loss after averaging; equal total weight per link across repeated folds',
            'support': 'selected windows per complete link, evaluated before folds; no residual or agreement selection',
            'uncertainty': f'{draws} positive shared acquisition/source-burst factor draws, refitting training data only',
            'interpretation': 'descriptive sensitivity; overlapping windows and adjacent links are dependent',
            'native': 'diagnostic only; no model-selection verdict',
            'original': 'frozen window-based estimates and artifacts are preserved unchanged'}}
    for cohort, record in records.items():
        logging.info('Link rescoring: %s', cohort)
        entry = {'original_matched_window_scores': original_matched_window_scores(record), 'variants': {}}
        if cohort == 'native':
            links = summarize_links(record['rows'], fit['metadata'])
            entry.update(role='diagnostic_only', links=links, order=order_summary(links))
        else:
            entry['role'] = 'post_hoc_sensitivity'
            for minimum in spec['minimum_support']:
                logging.info('%s: minimum %d selected windows/link', cohort, minimum)
                entry['variants'][str(minimum)] = evaluate_variant(record['rows'], fit['metadata'], minimum, draws)
        result['cohorts'][cohort] = entry
    products = []
    for cohort, entry in result['cohorts'].items():
        path = out/f'{cohort}.json'
        write_json(path, entry)
        products.append(path)
    products.extend(plot_results(out, result))
    report_file = out/'REPORT.md'
    report_file.write_text(render_report(result))
    products.append(report_file)
    return experiment.finish(out/'record.json', spec,
        {'status': 'complete', 'post_hoc': True, 'original_results_unchanged': True,
         'policy': result['policy']}, products)


def _format(value):
    return 'unavailable' if value is None or not np.isfinite(value) else f'{value:.5f}'


def render_report(result):
    lines = ['# Link-level swath rescoring — post hoc sensitivity analysis', '',
        'The original experiment is preserved unchanged. These support thresholds and link-mean scores were chosen '
        'after inspecting its results; they are not a replacement prespecified analysis.', '',
        'Each link target is the equal mean of its burst-pair group means. Loss is calculated after averaging. '
        'Each model is compared with a zero-correction baseline on exactly the same held-out data.', '',
        'Thresholds use total selected windows per link before folds; counts can decrease after spatial withholding. '
        'Windows overlap and are not independent replicates. All intervals below are descriptive shared-factor '
        'sensitivity intervals, not independent-link significance tests.', '', '## Primary spacecraft pattern', '']
    primary = result['cohorts']['primary']['variants']
    lines += ['| Minimum windows/link | Links | β (px) | Descriptive 95% interval | Consistent mixed-link signs |',
              '| ---: | ---: | ---: | --- | --- |']
    for minimum, variant in primary.items():
        coefficient = variant['coefficient']
        interval = coefficient['interval95_px']
        interval_text = 'unavailable' if interval is None else f'[{interval[0]:.5f}, {interval[1]:.5f}]'
        sign = variant['order']['sign_consistency']
        lines.append(f"| {minimum} | {len(variant['links'])} | {_format(coefficient['beta_px'])} | {interval_text} | "
                     f"{sign['consistent']}/{sign['mixed_links']} |")
    baseline = primary.get('0', next(iter(primary.values())))
    interval = baseline['coefficient']['interval95_px']
    lines += ['', ('The overall coefficient has a positive descriptive interval: evidence for a repeatable '
        'spacecraft-associated relative swath pattern in this dataset.' if interval and interval[0] > 0 else
        'The descriptive coefficient interval includes zero; interpret the order plots and sensitivities together.'),
        'This association does not identify hardware timing as its cause or establish absolute motion accuracy.', '',
        f"Leaving one link out gives β in {baseline['leave_link_out_beta_range_px']} px.", '',
        '### Spacecraft model: held-out whole-link means', '',
        '| Minimum windows/link | Matched links | Baseline MAE (px) | Spacecraft MAE (px) | Point reduction |',
        '| ---: | ---: | ---: | ---: | ---: |']
    for minimum, variant in primary.items():
        score = variant['validation']['leave_link_out']['spacecraft']
        base, corrected = score['baseline_mae_px'], score['model_mae_px']
        reduction = f'{100*(base-corrected)/base:.1f}%' if base and corrected is not None else 'unavailable'
        lines.append(f"| {minimum} | {score['links']} | {_format(base)} | {_format(corrected)} | {reduction} |")
    lines += ['', 'These are held-out point estimates. Their shared-factor uncertainty intervals, reported below, '
        'can include zero improvement even when the coefficient interval is positive. Do not interpret either '
        'quantity as a test of the physical cause.', '',
        '| Order | Links | Mean link contrast (px) | Median (px) |', '| --- | ---: | ---: | ---: |']
    for order in ORDERS:
        values = baseline['order'][order]
        lines.append(f"| {order[0]}→{order[1]} | {values['links']} | {_format(values['mean_px'])} | {_format(values['median_px'])} |")
    lines += ['', f"Mixed-order even component: {_format(baseline['order']['mixed_order_even_component_px'])} px. "
        'Different support across orders can contribute to this asymmetry.', '', '## Processing regimes', '',
        '| Minimum windows/link | IPF | Within-regime links | β (px) | Descriptive 95% interval |',
        '| ---: | --- | ---: | ---: | --- |']
    for minimum, variant in primary.items():
        for version, coefficient in variant['ipf'].items():
            interval = coefficient['interval95_px']
            text = 'unavailable' if interval is None else f'[{interval[0]:.5f}, {interval[1]:.5f}]'
            lines.append(f"| {minimum} | {version} | {coefficient['links']} | {_format(coefficient['beta_px'])} | {text} |")
    lines += ['', 'Cross-regime links are excluded from regime-specific fits. JSON retains jointly resampled '
        'regime differences; spacecraft, processing regime, season and acquisition date remain partly confounded.', '',
        '## Held-out link-mean errors', '',
        'Each row contains its own matched baseline. Rows for different models may cover different links; '
        'compare model against baseline within a row. Negative changes indicate improvement.', '']
    for cohort, entry in result['cohorts'].items():
        if entry['role'] == 'diagnostic_only':
            lines += ['### Native observations: diagnostic only', '',
                'No scientific model-selection verdict is assigned to this population, which includes poor correlation peaks.', '']
            continue
        lines += [f'### {cohort}', '',
            '| Min windows | Validation | Model | Matched links | Test instances | Baseline MAE | Model MAE | ΔMAE | ΔMAE interval | Baseline RMSE | Model RMSE |',
            '| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |']
        for minimum, variant in entry['variants'].items():
            for kind, models in variant['validation'].items():
                for model, score in models.items():
                    ci = score['mae_change_interval95_px']
                    text = 'unavailable' if ci is None else f'[{ci[0]:.5f}, {ci[1]:.5f}]'
                    lines.append(f"| {minimum} | {kind} | {model} | {score['links']} | {score['evaluation_instances']} | "
                        f"{_format(score['baseline_mae_px'])} | {_format(score['model_mae_px'])} | {_format(score['mae_change_px'])} | "
                        f"{text} | {_format(score['baseline_rmse_px'])} | {_format(score['model_rmse_px'])} |")
        lines += ['', '| Min windows | Usable links | Graph rank | Cycles | Bridges | Isolated dates |',
                  '| ---: | ---: | ---: | ---: | ---: | ---: |']
        for minimum, variant in entry['variants'].items():
            graph = variant['graph']
            lines.append(f"| {minimum} | {graph['links']} | {graph['rank']} | {graph['cycles']} | "
                         f"{len(graph['bridges'])} | {len(graph['isolated_dates'])} |")
        lines += ['', 'An unrestricted date model can be poorly constrained even when full rank. Unseen dates '
            'and disconnected endpoint contrasts remain unavailable; no interpolation or prior fills them.', '']
    lines += ['## Original window scores with corrected comparison populations', '',
        'These reproduce the original predictions and window-based scoring. Both columns in each row use '
        'identical finite windows. The original files and estimates were not overwritten.', '',
        '| Population | Validation | Model | Matched windows | Matched links | Baseline MAE | Model MAE |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for cohort, entry in result['cohorts'].items():
        for kind, models in entry['original_matched_window_scores'].items():
            for model, comparison in models.items():
                base, corrected = comparison['models']['baseline'], comparison['models']['model']
                lines.append(f"| {cohort} | {kind} | {model} | {base['n']} | {base['links']} | "
                             f"{_format(base['mae_px'])} | {_format(corrected['mae_px'])} |")
    lines += ['', '## Interpretation limits', '',
        'A stable nonzero spacecraft coefficient and an uncertain reduction of individual-window errors answer '
        'different questions. Link-level results assess average swath mismatch, not the accuracy of every match. '
        'The physical fields remain the original unfitted predictions; a satellite-specific swath timing term is not modeled here. '
        'These post hoc results do not authorize a production correction.', '',
        'Figures: [spacecraft order](spacecraft_order.png), [support sensitivity](support_sensitivity.png), '
        '[held-out scores](link_validation.png).']
    return '\n'.join(lines)+'\n'


def plot_results(out, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    primary = result['cohorts']['primary']['variants']
    colors = dict(zip(ORDERS, ('#d55e00', '#0072b2', '#009e73', '#aa4499')))
    figures = []
    all_variant = primary.get('0', next(iter(primary.values())))
    supported = primary.get('20', all_variant)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for ax, variant in zip(axes, (all_variant, supported)):
        rng = np.random.default_rng(0)
        for i, order in enumerate(ORDERS):
            points = [x for x in variant['links'] if x['order'] == order]
            x = i+rng.uniform(-.15, .15, len(points))
            for xx, point in zip(x, points):
                ax.scatter(xx, point['mean_px'], s=20+min(point['windows'], 180)*.55,
                    facecolors=colors[order] if point['windows'] >= 20 else 'none',
                    edgecolors=colors[order], alpha=.85)
            if points:
                ax.scatter(i, np.mean([p['mean_px'] for p in points]), marker='_', s=350, c='black', linewidths=2)
            beta = variant['coefficient']['beta_px']
            if beta is not None:
                prediction = beta if order == 'AB' else -beta if order == 'BA' else 0
                ax.plot([i-.3, i+.3], [prediction, prediction], '--', c=colors[order], linewidth=1)
        ax.axhline(0, color='gray', linewidth=.7)
        ax.set_xticks(range(4), [f'{s[0]}→{s[1]}\n(n={variant["order"][s]["links"]})' for s in ORDERS])
        ax.set(ylabel='IW2 − IW1 azimuth link mean [original pixels]',
               title=f'Minimum {variant["minimum_windows"]} selected windows/link')
        ax.grid(axis='y', alpha=.2)
    axes[0].text(.02, .03, 'Hollow: fewer than 20 windows\nSize: selected windows; black bar: order mean\nDashed: descriptive spacecraft fit',
                 transform=axes[0].transAxes, fontsize=8, bbox={'facecolor': 'white', 'alpha': .8, 'edgecolor': 'none'})
    fig.suptitle('Spacecraft-order pattern — primary cohort; post hoc support sensitivity\nEach point is one link; panels have separate vertical scales')
    path = out/'spacecraft_order.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append(path)
    thresholds = list(map(int, primary))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    beta = [v['coefficient']['beta_px'] for v in primary.values()]
    for x, value, variant in zip(thresholds, beta, primary.values()):
        ci = variant['coefficient']['interval95_px']
        if value is not None:
            axes[0, 0].plot([x, x], ci if ci else [value, value], color='tab:blue')
            axes[0, 0].scatter(x, value, color='tab:blue')
    axes[0, 0].set(title='Overall coefficient; descriptive 95% intervals', ylabel='β [pixels]')
    for model, color in zip(MODELS, ('tab:orange', 'tab:blue', 'tab:green')):
        scores = [v['validation']['leave_link_out'][model] for v in primary.values()]
        axes[0, 1].plot(thresholds, [np.nan if s['rmse_change_px'] is None else s['rmse_change_px'] for s in scores],
                        'o-', label=model, color=color)
    axes[0, 1].set(title='Held-out link RMSE change vs matched baseline', ylabel='ΔRMSE [pixels]; negative improves')
    axes[0, 1].legend(fontsize=8)
    for label, orders in (('Mixed spacecraft', ('AB', 'BA')), ('Same spacecraft', ('AA', 'BB'))):
        axes[1, 0].plot(thresholds, [sum(v['order'][o]['links'] for o in orders) for v in primary.values()], 'o-', label=label)
    axes[1, 0].set(title='Support attrition', ylabel='Eligible links')
    axes[1, 0].legend()
    versions = sorted(all_variant['ipf'])
    for k, version in enumerate(versions):
        xs = np.array(thresholds)+(k-(len(versions)-1)/2)*1.7
        values = [v['ipf'][version]['beta_px'] for v in primary.values()]
        line, = axes[1, 1].plot(xs, values, 'o-', label=version)
        for x, variant in zip(xs, primary.values()):
            ci = variant['ipf'][version]['interval95_px']
            if ci:
                axes[1, 1].plot([x, x], ci, color=line.get_color())
    axes[1, 1].set(title='Within-IPF spacecraft coefficients', ylabel='β [pixels]')
    axes[1, 1].legend(title='IPF')
    for ax in axes.flat:
        ax.set(xlabel='Minimum selected windows/link', xticks=thresholds)
        ax.grid(alpha=.2)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
        ax.axhline(0, color='gray', linewidth=.7)
    fig.suptitle('Post hoc minimum-support sensitivity — primary cohort')
    path = out/'support_sensitivity.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append(path)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    for col, kind in enumerate(VALIDATIONS):
        for model, color in zip(MODELS, ('tab:orange', 'tab:blue', 'tab:green')):
            scores = [v['validation'][kind][model] for v in primary.values()]
            for row, metric in enumerate(('mae', 'rmse')):
                points = [np.nan if s[f'{metric}_change_px'] is None else s[f'{metric}_change_px'] for s in scores]
                axes[row, col].plot(thresholds, points, 'o-', label=model, color=color)
                for x, score in zip(thresholds, scores):
                    interval = score[f'{metric}_change_interval95_px']
                    if interval:
                        axes[row, col].plot([x, x], interval, color=color, alpha=.5)
                axes[row, col].axhline(0, color='gray', linewidth=.7)
                axes[row, col].set(xlabel='Minimum windows/link', ylabel=f'Δ{metric.upper()} [pixels]', xticks=thresholds)
                axes[row, col].grid(alpha=.2)
        axes[0, col].set_title(kind.replace('_', ' '))
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].text(.03, .05, 'Per-date predictions unavailable\nfor unseen dates', transform=axes[1, 0].transAxes, fontsize=8)
    fig.suptitle('Held-out link-mean errors — matched baselines; negative improves\nPost hoc analysis; intervals describe shared acquisition/source-burst sensitivity')
    path = out/'link_validation.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append(path)
    return figures
