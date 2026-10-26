"""Small identifiable swath-contrast models and explicit withheld predictions.

No regularization, correction application, spatial fit, or missing-date prior.
One row is a burst-pair mean; each link has equal total fitting weight.
"""
import numpy as np


def graph_audit(rows, dates):
    index = {d: i for i, d in enumerate(dates)}
    edges = sorted({(r['earlier'], r['later']) for r in rows})
    adjacency = {d: set() for d in dates}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    components, visited = [], set()
    for start in dates:
        if start in visited:
            continue
        pending, component = [start], []
        while pending:
            node = pending.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            pending.extend(adjacency[node]-visited)
        components.append(sorted(component))
    labels = {d: k for k, group in enumerate(components) for d in group}
    bridges = []
    # At most 59 distinct links, so explicit edge removal is clearer than a graph dependency.
    for a, b in edges:
        seen, pending = set(), [a]
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            pending.extend(x for x in adjacency[node] if {node, x} != {a, b} and x not in seen)
        if b not in seen:
            bridges.append(f'{a}_{b}')
    return {'nodes': len(dates), 'links': len(edges), 'components': components,
            'rank': len(dates)-len(components), 'cycles': len(edges)-len(dates)+len(components),
            'bridges': bridges, 'isolated_dates': [d for d in dates if not adjacency[d]],
            'labels': labels}


def link_weights(rows):
    counts = {}
    for row in rows:
        counts[row['pair']] = counts.get(row['pair'], 0)+1
    return np.array([1/counts[r['pair']] for r in rows], dtype=float)


def design(rows, dates, missions, model):
    if model == 'spacecraft':
        return np.array([[int(missions[r['later']] == 'S1B')-int(missions[r['earlier']] == 'S1B')]
                         for r in rows], dtype=float).reshape(-1, 1)
    result = np.zeros((len(rows), len(dates)))
    index = {d: i for i, d in enumerate(dates)}
    for i, row in enumerate(rows):
        result[i, index[row['earlier']]] = -1
        result[i, index[row['later']]] = 1
    return result


def train(rows, dates, missions, model, multipliers=None):
    matrix = design(rows, dates, missions, model)
    weights = link_weights(rows)
    y = np.array([r['mean'] for r in rows])
    if multipliers is not None:
        weights *= multipliers
    if not len(rows):
        parameter = np.full(matrix.shape[1], np.nan)
    else:
        root = np.sqrt(weights)
        parameter = np.linalg.lstsq(matrix*root[:, None], y*root, rcond=None)[0]
    graph = graph_audit(rows, dates)
    identifiable = np.linalg.matrix_rank(matrix) if matrix.size else 0
    return {'parameters': parameter, 'graph': graph, 'rank': int(identifiable),
            'model': model, 'matrix': matrix, 'weights': weights, 'y': y}


def prediction_mask(fit, rows):
    if fit['model'] == 'spacecraft':
        return np.full(len(rows), fit['rank'] == 1)
    graph = fit['graph']
    return np.array([r['earlier'] not in graph['isolated_dates'] and
                     r['later'] not in graph['isolated_dates'] and
                     graph['labels'][r['earlier']] == graph['labels'][r['later']] for r in rows])


def predict(fit, rows, dates, missions):
    matrix = design(rows, dates, missions, fit['model'])
    result = matrix @ fit['parameters']
    result[~prediction_mask(fit, rows)] = np.nan
    return result


def validation_folds(rows, dates):
    """Explicit masks; spatial folds purge either shared source burst."""
    folds = []
    for day in dates:
        test = np.array([day in (r['earlier'], r['later']) for r in rows])
        folds.append(('leave_acquisition_out', day, ~test, test))
    for pair in sorted({r['pair'] for r in rows}):
        test = np.array([r['pair'] == pair for r in rows])
        folds.append(('leave_link_out', pair, ~test, test))
    for group in sorted({(r['burst1'], r['burst2']) for r in rows}):
        test = np.array([(r['burst1'], r['burst2']) == group for r in rows])
        shared = set(group)
        train_mask = np.array([not shared.intersection((r['burst1'], r['burst2'])) for r in rows])
        folds.append(('leave_burst_group_out', '|'.join(group), train_mask, test))
    return folds


def multiplier_draws(rows, draws=1000, seed=0):
    """Positive shared factors preserve graph rank in descriptive refits."""
    rng = np.random.default_rng(seed)
    dates = sorted({d for r in rows for d in (r['earlier'], r['later'])})
    bursts = sorted({s for r in rows for s in (r['burst1'], r['burst2'])})
    factors = {('date', d): rng.exponential(size=draws) for d in dates}
    factors.update({('burst', s): rng.exponential(size=draws) for s in bursts})
    out = np.ones((draws, len(rows)))
    for i, row in enumerate(rows):
        for domain, keys in (('date', ('earlier', 'later')), ('burst', ('burst1', 'burst2'))):
            for key in keys:
                out[:, i] *= factors[domain, row[key]]
    return out


def bootstrap_parameters(fit, multipliers):
    a, y = fit['matrix'], fit['y']
    if not len(y) or fit['rank'] == 0:
        return np.full((len(multipliers), a.shape[1]), np.nan)
    # Use a fixed full-rank basis, retaining a zero-sum datum per component.
    _, _, vt = np.linalg.svd(a, full_matrices=False)
    basis = vt[:fit['rank']].T
    reduced = a @ basis
    weights = multipliers*fit['weights'][None, :]
    gram = np.einsum('ni,dn,nj->dij', reduced, weights, reduced, optimize=True)
    rhs = np.einsum('ni,dn,n->di', reduced, weights, y, optimize=True)
    try:
        solved = np.linalg.solve(gram, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError:
        # Positive multipliers preserve rank mathematically; extremely unequal
        # weights can still make normal equations lose precision. No ridge/prior.
        solved = np.array([np.linalg.lstsq(reduced*np.sqrt(w)[:, None], y*np.sqrt(w), rcond=None)[0]
                           for w in weights])
    return solved @ basis.T


def error_metrics(rows, predictions):
    """Equal link, then equal burst group, then equal window weighting."""
    values, weights, available_rows = [], [], []
    for row, predicted in zip(rows, predictions):
        residual = np.asarray(row['values'])-predicted
        finite = np.isfinite(residual)
        if finite.any():
            available_rows.append((row, residual[finite]))
    grouped_weights = link_weights([r for r, _ in available_rows])
    for (_, residual), weight in zip(available_rows, grouped_weights):
        values.extend(residual.tolist())
        weights.extend([weight/len(residual)]*len(residual))
    if not values:
        return {'n': 0, 'links': 0, 'signed_bias_px': None, 'mae_px': None, 'rmse_px': None,
                'median_absolute_px': None, 'p90_absolute_px': None}
    v, w = np.array(values), np.array(weights)
    w /= w.sum()
    order = np.argsort(abs(v))
    quantile = lambda p: float(abs(v)[order][np.searchsorted(np.cumsum(w[order]), p)])
    return {'n': len(v), 'groups': len(available_rows),
            'links': len({r['pair'] for r, _ in available_rows}),
            'signed_bias_px': float(w @ v), 'mae_px': float(w @ abs(v)),
            'rmse_px': float(np.sqrt(w @ v**2)), 'median_absolute_px': quantile(.5),
            'p90_absolute_px': quantile(.9)}


def paired_metrics(rows, predictions):
    """All methods use the same finite evaluation windows; attrition stays explicit."""
    common = [np.logical_and.reduce([np.isfinite(np.broadcast_to(p[i], np.shape(r['values'])))
                                   for p in predictions.values()]) for i, r in enumerate(rows)]
    selected, selected_predictions = [], {name: [] for name in predictions}
    for i, (row, mask) in enumerate(zip(rows, common)):
        if not mask.any():
            continue
        selected.append(dict(row, values=np.asarray(row['values'])[mask].tolist()))
        for name, p in predictions.items():
            selected_predictions[name].append(np.broadcast_to(p[i], mask.shape)[mask])
    return {'common_n': sum(int(x.sum()) for x in common),
            'models': {name: error_metrics(selected, p) for name, p in selected_predictions.items()}}


def evaluate(rows, dates, missions, draws=1000, seed=0):
    """Fit descriptive models and every fixed validation fold, with no test leakage."""
    multipliers = multiplier_draws(rows, draws, seed)
    report = {'graph': graph_audit(rows, dates), 'models': {}, 'folds': [],
              'uncertainty': {'draws': draws, 'seed': seed,
                 'method': 'positive Exp(1) multipliers shared by each acquisition and source burst; training-only refits',
                 'interpretation': 'descriptive dependence sensitivity, not independent-window confidence',
                 'insufficient_groups': len({(r['burst1'], r['burst2']) for r in rows}) < 2}}
    score_draws, score_insufficient = {}, {}
    for model in ('spacecraft', 'per_acquisition'):
        fit = train(rows, dates, missions, model)
        samples = bootstrap_parameters(fit, multipliers)
        covariance = (np.cov(samples, rowvar=False) if draws > 1 and np.isfinite(samples).all()
                      and not report['uncertainty']['insufficient_groups'] else None)
        parameters = fit['parameters'].copy()
        if model == 'per_acquisition':
            parameters[[dates.index(d) for d in fit['graph']['isolated_dates']]] = np.nan
        elif fit['rank'] == 0:
            parameters[:] = np.nan
        report['models'][model] = {'parameters': parameters.tolist(), 'rank': fit['rank'],
             'datum': 'zero sum within each connected component' if model == 'per_acquisition' else 'A=0, B=beta',
             'parameter_names': dates if model == 'per_acquisition' else ['beta_B_minus_A_px'],
             'covariance': np.atleast_2d(covariance).tolist() if covariance is not None else None,
             'isolated_parameters': fit['graph']['isolated_dates'],
             'training_metrics': error_metrics(rows, predict(fit, rows, dates, missions))}
    for kind, label, train_mask, test_mask in validation_folds(rows, dates):
        if not test_mask.any():
            continue
        training = [r for r, take in zip(rows, train_mask) if take]
        testing = [r for r, take in zip(rows, test_mask) if take]
        entry = {'kind': kind, 'label': label, 'train_rows': np.flatnonzero(train_mask).tolist(),
                 'test_rows': np.flatnonzero(test_mask).tolist(), 'models': {},
                 'graph': graph_audit(training, dates),
                 'insufficient_training_groups': len({(r['burst1'], r['burst2']) for r in training}) < 2}
        for model in ('spacecraft', 'per_acquisition'):
            fit = train(training, dates, missions, model)
            prediction = predict(fit, testing, dates, missions)
            samples = bootstrap_parameters(fit, multipliers[:, train_mask])
            test_design = design(testing, dates, missions, model)
            predicted_samples = samples @ test_design.T
            predicted_samples[:, ~prediction_mask(fit, testing)] = np.nan
            finite = np.isfinite(prediction)
            intervals = np.full((len(testing), 2), np.nan)
            sigma = np.full(len(testing), np.nan)
            if not entry['insufficient_training_groups'] and finite.any():
                intervals[finite] = np.quantile(predicted_samples[:, finite], [.025, .975], axis=0).T
                sigma[finite] = np.std(predicted_samples[:, finite], axis=0, ddof=1)
            entry['models'][model] = {'prediction_px': prediction.tolist(),
                'interval95_px': intervals.tolist(), 'prediction_std_px': sigma.tolist(),
                'metrics': error_metrics(testing, prediction), 'unavailable_n': int((~finite).sum())}
            accumulated = score_draws.setdefault((kind, model), [])
            score_insufficient[(kind, model)] = (score_insufficient.get((kind, model), False) or
                                                (finite.any() and entry['insufficient_training_groups']))
            for j, index in enumerate(np.flatnonzero(test_mask)):
                if not finite[j]:
                    continue
                values = np.asarray(rows[index]['values'])
                accumulated.append((index, np.mean(abs(values[None, :]-predicted_samples[:, j, None]), axis=1),
                                    float(np.mean(abs(values))), float(np.mean(abs(values-prediction[j])))))
        report['folds'].append(entry)
    report['validation_score_intervals'] = {}
    for (kind, model), records in score_draws.items():
        if not records:
            continue
        # A link can occur twice in leave-acquisition-out testing. Divide its
        # total weight across both appearances and all its burst groups.
        selected = [rows[i] for i, _, _, _ in records]
        weights = link_weights(selected)
        indexes = [i for i, _, _, _ in records]
        weighted = multipliers[:, indexes]*weights[None, :]
        loss = np.stack([v for _, v, _, _ in records], axis=1)
        baseline = np.array([v for _, _, v, _ in records])
        draw_delta = np.sum(weighted*(loss-baseline[None, :]), axis=1)/weighted.sum(axis=1)
        point_delta = float(np.average([v-b for _, _, b, v in records], weights=weights))
        interval = (np.quantile(draw_delta, [.025, .975]).tolist()
                    if not report['uncertainty']['insufficient_groups'] and not score_insufficient[kind, model] else None)
        report['validation_score_intervals'].setdefault(kind, {})[model] = {
            'mae_change_vs_baseline_px': point_delta, 'descriptive_interval95_px': interval,
            'insufficient_training_groups': bool(score_insufficient[kind, model]),
            'unique_test_groups': len(set(indexes)), 'evaluation_instances': len(indexes)}
    return report
