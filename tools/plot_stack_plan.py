#!/usr/bin/env python
"""Plot planned temporal networks and geographic-burst connectivity from JSON.

    glacier-offsets stack plan configs/stack.yaml > plan.json
    python tools/plot_stack_plan.py plan.json -o plan_plots

Only the saved JSON is read; SAFE, orbit, DEM and processing outputs are not
opened. NumPy/Matplotlib are sufficient; native ISCE3 and NetworkX are not used.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE/'.runtime/deps'))
os.environ.setdefault('MPLCONFIGDIR', str(WORKSPACE/'.runtime/matplotlib'))


def utc(value):
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f'Invalid UTC timestamp: {value!r}') from exc


def components(nodes, edges):
    """Find every undirected component, including isolates and disjoint chains."""
    adjacent = {node: set() for node in nodes}
    for a, b in edges:
        adjacent[a].add(b); adjacent[b].add(a)
    result, unseen = [], set(nodes)
    while unseen:
        pending, group = [min(unseen)], []
        unseen.remove(pending[0])
        while pending:
            node = pending.pop(); group.append(node)
            following = adjacent[node] & unseen
            unseen.difference_update(following); pending.extend(sorted(following))
        result.append(sorted(group))
    return result


def gaps(dates, acquisitions, threshold):
    result = []
    for a, b in zip(dates, dates[1:]):
        seconds = (utc(acquisitions[b]['start']) - utc(acquisitions[a]['start'])).total_seconds()
        result.append({'earlier': a, 'later': b, 'days': seconds/86400,
                       'above_threshold': seconds > threshold*86400})
    return result


def analyze_plan(plan, gap_days=12):
    """Summarize planned links separately from links with usable shared bursts."""
    if not math.isfinite(gap_days) or gap_days <= 0:
        raise ValueError('gap-days must be a finite positive number')
    if not isinstance(plan, dict) or not isinstance(plan.get('inventory'), list) or not isinstance(plan.get('pairs'), dict):
        raise ValueError('Expected stack plan JSON with inventory and pairs (not a download plan or status report)')
    selected = plan.get('selected_burst_ids', [])
    if not isinstance(selected, list) or any(not isinstance(b, str) for b in selected) or len(set(selected)) != len(selected):
        raise ValueError('selected_burst_ids must contain unique strings')
    acquisitions, warnings = {}, []
    for row in plan['inventory']:
        if not isinstance(row, dict):
            raise ValueError('Invalid acquisition inventory row')
        if not row.get('eligible', False):
            continue
        day = row.get('date')
        if not isinstance(day, str) or utc(day).date().isoformat() != day:
            raise ValueError(f'Invalid acquisition date: {day!r}')
        if day in acquisitions:
            raise ValueError(f'Ambiguous duplicate acquisition date: {day}')
        start = utc(row.get('start', day))
        if start.date().isoformat() != day:
            raise ValueError(f'Acquisition UTC date disagrees with start: {day}')
        available = row.get('available_burst_ids', row.get('burst_ids'))
        if available is None and 'missing_burst_ids' in row:
            available = [b for b in selected if b not in row['missing_burst_ids']]
        if available is not None and (not isinstance(available, list) or any(not isinstance(b, str) for b in available)):
            raise ValueError(f'Invalid burst availability: {day}')
        available = [] if row.get('error') else available
        acquisitions[day] = {'start': start.isoformat(), 'mission': str(row.get('mission', 'Unknown')),
            'available_burst_ids': None if available is None else [b for b in selected if b in available],
            'error': row.get('error')}
        if 'start' not in row:
            warnings.append(f'{day}: no scene start; using midnight UTC for acquisition gaps')
    dates = sorted(acquisitions)
    if not dates:
        raise ValueError('No eligible acquisition dates in this plan')
    pairs, seen = [], set()
    for key, row in sorted(plan['pairs'].items()):
        if not isinstance(row, dict):
            raise ValueError(f'Invalid pair record: {key}')
        if not row.get('active', True):
            continue
        a, b = row.get('earlier'), row.get('later')
        if a not in acquisitions or b not in acquisitions or a >= b:
            raise ValueError(f'Pair endpoints must be distinct chronological eligible dates: {key}')
        if (a, b) in seen:
            raise ValueError(f'Duplicate pair: {a}, {b}')
        seen.add((a, b))
        interval = row.get('scene_interval_seconds', (utc(acquisitions[b]['start']) - utc(acquisitions[a]['start'])).total_seconds())
        if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval <= 0:
            raise ValueError(f'Invalid scene interval: {key}')
        shared = row.get('shared_burst_ids', [])
        if not isinstance(shared, list) or any(not isinstance(burst, str) for burst in shared):
            raise ValueError(f'Invalid shared bursts: {key}')
        if not set(shared) <= set(selected):
            raise ValueError(f'Pair has bursts outside selected_burst_ids: {key}')
        if not isinstance(row.get('burst_intervals', {}), dict):
            raise ValueError(f'Invalid burst intervals: {key}')
        usable = []
        for burst in sorted(set(shared)):
            for day in (a, b):
                available = acquisitions[day]['available_burst_ids']
                if available is not None and burst not in available:
                    raise ValueError(f'Pair claims unavailable burst {burst} on {day}')
            entry = row.get('burst_intervals', {}).get(burst, {})
            if not isinstance(entry, dict):
                raise ValueError(f'Invalid interval for {burst}: {key}')
            seconds = entry.get('interval_seconds')
            if 'burst_intervals' not in row or (isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
                                               and math.isfinite(seconds) and seconds > 0):
                usable.append(burst)
        pairs.append({'earlier': a, 'later': b, 'days': interval/86400,
                      'shared_burst_ids': sorted(set(shared)), 'usable_burst_ids': usable})
    edges = [(p['earlier'], p['later']) for p in pairs]
    groups = components(dates, edges)
    degree = Counter(day for edge in edges for day in edge)
    temporal = gaps(dates, acquisitions, gap_days)
    bursts = {}
    reference = plan.get('reference_date')
    if reference is not None and reference not in acquisitions:
        raise ValueError('Reference date is not an eligible acquisition')
    for burst in selected:
        available = [d for d in dates if acquisitions[d]['available_burst_ids'] is not None and
                     burst in acquisitions[d]['available_burst_ids']]
        unknown = [d for d in dates if acquisitions[d]['available_burst_ids'] is None]
        links = [(p['earlier'], p['later']) for p in pairs if burst in p['usable_burst_ids']]
        # Explicit pair metadata can identify availability absent from older inventory rows.
        available = sorted(set(available) | {day for link in links for day in link})
        unknown = [d for d in unknown if d not in available]
        burst_groups = components(available, links)
        intervals = gaps(available, acquisitions, gap_days)
        bursts[burst] = {'available_dates': available, 'unknown_dates': unknown,
            'missing_dates': [d for d in dates if d not in available and d not in unknown],
            'pairs': [{'earlier': a, 'later': b} for a, b in links], 'components': burst_groups,
            'reference_component': next((g for g in burst_groups if reference in g), []),
            'largest_availability_gap_days': max((g['days'] for g in intervals), default=None),
            'large_availability_gaps': [g for g in intervals if g['above_threshold']]}
    unknown_count = sum(a['available_burst_ids'] is None for a in acquisitions.values())
    if unknown_count:
        warnings.append(f'{unknown_count} acquisitions have incomplete burst availability metadata')
    metadata_errors = [d for d in dates if acquisitions[d]['error']]
    for day in metadata_errors[:5]:
        warnings.append(f'{day}: {acquisitions[day]["error"]}')
    if len(metadata_errors) > 5:
        warnings.append(f'{len(metadata_errors)-5} more metadata failures; see summary.json')
    if plan.get('errors'):
        warnings.append(f'Input plan reports {len(plan["errors"])} preflight errors; inspect plan_errors in summary.json')
    return {'reference_date': reference, 'dates': dates, 'acquisitions': acquisitions, 'pairs': pairs,
        'components': groups, 'isolated_dates': [d for d in dates if degree[d] == 0],
        'usable_shared_components': components(dates, [(p['earlier'], p['later']) for p in pairs if p['usable_burst_ids']]),
        'degree': {d: degree[d] for d in dates}, 'consecutive_gaps': temporal,
        'gap_threshold_days': gap_days, 'large_gaps': [g for g in temporal if g['above_threshold']],
        'metadata_error_dates': metadata_errors,
        'bursts': bursts, 'omitted_links': plan.get('omitted_links', []),
        'excluded_acquisitions': [r for r in plan['inventory'] if not r.get('eligible', False)],
        'plan_errors': plan.get('errors', []), 'warnings': warnings,
        'interpretation': 'Connectivity is based on planned metadata, not measured matching quality or valid pixel overlap. '
            'Availability gaps use exact scene-start UTC times. Burst graphs omit links with missing/nonpositive burst intervals. '
            'Missing burst coverage includes dates with inventory/metadata errors; it does not imply absent files.'}


def _date_axis(ax):
    import matplotlib.dates as mdates
    locator = mdates.AutoDateLocator(minticks=4, maxticks=10, tz=timezone.utc)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=timezone.utc))
    ax.grid(axis='x', color='#e2e6ed', linewidth=.6)
    ax.set_axisbelow(True)


def _coordinates(summary):
    import matplotlib.dates as mdates
    return {day: mdates.date2num(utc(row['start'])) for day, row in summary['acquisitions'].items()}


def plot_overview(summary):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.path import Path as ArtistPath
    from matplotlib.patches import PathPatch
    import numpy as np
    fig = plt.figure(figsize=(16, 10), layout='constrained')
    grid = fig.add_gridspec(3, 2, height_ratios=[1.35, 1, 1])
    network, gap_ax = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, :])
    baselines, degrees = fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1])
    dates, pairs, groups = summary['dates'], summary['pairs'], summary['components']
    x = _coordinates(summary)
    colors = plt.get_cmap('tab20')
    group_of = {d: i for i, group in enumerate(groups) for d in group}
    usable_group = {d: i for i, group in enumerate(summary['usable_shared_components']) for d in group}
    span = max(x.values()) - min(x.values()) or 1
    for pair in pairs:
        a, b = pair['earlier'], pair['later']; lane = group_of[a]
        lift = .2 + min(.4, (x[b]-x[a])/span*2)
        path = ArtistPath([(x[a], lane), ((x[a]+x[b])/2, lane+lift), (x[b], lane)],
                          [ArtistPath.MOVETO, ArtistPath.CURVE3, ArtistPath.CURVE3])
        known = bool(pair['usable_burst_ids'])
        network.add_patch(PathPatch(path, fill=False, edgecolor=colors(usable_group[a] % 20) if known else '#b73a3a',
                                    linewidth=1.15, alpha=.85, linestyle='-' if known else '--'))
    partial = []
    markers = {'S1A': 'o', 'S1B': '^', 'S1C': 's', 'S1D': 'D'}
    for day in dates:
        row = summary['acquisitions'][day]; lane = group_of[day]
        network.scatter(x[day], lane, s=27, marker='x' if row['error'] else markers.get(row['mission'], 'o'),
                        color='#b73a3a' if row['error'] else colors(usable_group[day] % 20), zorder=3)
        available = row['available_burst_ids']
        if not row['error'] and available is not None and len(available) < len(summary['bursts']):
            partial.append(day)
            network.scatter(x[day], lane, s=68, facecolors='none', edgecolors='#db7b00', linewidths=1.3, zorder=4)
        if day in summary['isolated_dates'] or row['error']:
            network.annotate(day, (x[day], lane), xytext=(0, -14), textcoords='offset points', ha='center', fontsize=7)
    reference = summary['reference_date']
    if reference:
        network.scatter(x[reference], group_of[reference], s=145, marker='*', color='#172a45',
                        edgecolors='white', linewidths=.5, zorder=5)
    network.set_ylim(-.25, max(len(groups)-1, 0)+.45)
    network.set_yticks(range(len(groups)), [f'Component {i+1}' for i in range(len(groups))], fontsize=8)
    network.set_title('Planned links · colors separate components with usable shared bursts', loc='left', fontweight='bold')
    handles = [Line2D([], [], marker='*', color='#172a45', linestyle='none', markersize=11, label='Reference'),
               Line2D([], [], marker='o', markerfacecolor='none', color='#db7b00', linestyle='none', label='Partial burst coverage'),
               Line2D([], [], color='#b73a3a', linestyle='--', label='No usable shared burst'),
               Line2D([], [], color='#b73a3a', marker='x', linestyle='none', label='Metadata/orbit error')]
    for mission in sorted({r['mission'] for r in summary['acquisitions'].values()}):
        handles.append(Line2D([], [], marker=markers.get(mission, 'o'), color='#63718a', linestyle='none', label=mission))
    network.legend(handles=handles, fontsize=8, loc='upper left', ncol=min(len(handles), 6))
    intervals = summary['consecutive_gaps']
    gap_ax.bar([x[g['later']] for g in intervals], [g['days'] for g in intervals], width=max(.5, span/len(dates)*.55),
               color=['#c84545' if g['above_threshold'] else '#4579a7' for g in intervals])
    gap_ax.axhline(summary['gap_threshold_days'], color='#a33434', linestyle='--', linewidth=1,
                   label=f'Flag gaps > {summary["gap_threshold_days"]:g} days')
    for g in sorted(summary['large_gaps'], key=lambda g: g['days'], reverse=True)[:5]:
        gap_ax.annotate(f'{g["days"]:.6g} d', (x[g['later']], g['days']), xytext=(0, 5),
                        textcoords='offset points', ha='center', fontsize=8)
    gap_ax.set_title('Consecutive acquisition gaps · exact scene-start times', loc='left', fontweight='bold')
    gap_ax.set_ylabel('Gap [days]'); gap_ax.set_ylim(bottom=0); gap_ax.legend(fontsize=8)
    nominal = Counter(round(p['days']) for p in pairs)
    baselines.bar(list(nominal), list(nominal.values()), color='#4579a7', width=.7)
    baselines.set_title('Pair separations · rounded to whole days', loc='left', fontweight='bold')
    baselines.set_xlabel('Approximate separation [days]'); baselines.set_ylabel('Planned pairs')
    baselines.set_xticks(sorted(nominal)); baselines.yaxis.get_major_locator().set_params(integer=True)
    if not pairs:baselines.text(.5, .5, 'No planned pairs', ha='center', transform=baselines.transAxes)
    degrees.plot([x[d] for d in dates], [summary['degree'][d] for d in dates], '.-', color='#40775d', linewidth=.9)
    degrees.set_title('Matching consumers per acquisition', loc='left', fontweight='bold')
    degrees.set_ylabel('Pair count'); degrees.set_ylim(bottom=0); degrees.yaxis.get_major_locator().set_params(integer=True)
    for ax in (network, gap_ax, degrees):
        _date_axis(ax); ax.set_xlim(min(x.values())-span*.025-1, max(x.values())+span*.025+1)
    degrees.set_xlabel('Acquisition UTC date'); gap_ax.set_xlabel('Later acquisition UTC date')
    longest = max((g['days'] for g in intervals), default=0)
    fig.suptitle(f'{len(dates)} acquisitions  ·  {len(pairs)} pairs  ·  {len(groups)} planned components  ·  '
                 f'{len(summary["isolated_dates"])} isolated dates\n'
                 f'{dates[0]} to {dates[-1]}  |  largest acquisition gap: {longest:.6g} d  |  '
                 f'{len(partial)} partial / {len(summary["metadata_error_dates"])} metadata-error dates  |  '
                 f'any-shared-burst graph: {len(summary["usable_shared_components"])} components',
                 fontsize=13, fontweight='bold')
    return fig


def plot_bursts(summary, burst_ids):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, (ax, stats) = plt.subplots(1, 2, figsize=(16, max(3.5, 2.1+.43*len(burst_ids))),
                                   gridspec_kw={'width_ratios':[5, 1.2]}, layout='constrained')
    x = _coordinates(summary); colors = plt.get_cmap('tab20')
    for index, burst in enumerate(burst_ids):
        row = summary['bursts'][burst]
        component = {d:i for i, group in enumerate(row['components']) for d in group}
        for pair in row['pairs']:
            a, b = pair['earlier'], pair['later']
            ax.plot([x[a], x[b]], [index, index], color=colors(component[a] % 20), linewidth=1.8, zorder=1)
        for day in row['available_dates']:
            ax.scatter(x[day], index, s=11, color=colors(component[day] % 20), zorder=3)
        ax.scatter([x[d] for d in row['missing_dates']], [index]*len(row['missing_dates']),
                   marker='x', s=15, color='#c84545', linewidths=.8, zorder=2)
        ax.scatter([x[d] for d in row['unknown_dates']], [index]*len(row['unknown_dates']),
                   marker='|', s=25, color='#7f8995', zorder=2)
        gap = row['largest_availability_gap_days']
        detail = f'{len(row["components"]):>2} components  |  {len(row["available_dates"])}/{len(summary["dates"])} dates'
        detail += f'\nmax gap {gap:.6g} d' if gap is not None else '\nmax gap —'
        stats.text(0, index, detail, va='center', fontsize=8,
                   color='#a33434' if len(row['components']) > 1 else '#33465c')
    for panel in (ax, stats):panel.set_ylim(len(burst_ids)-.45, -.65)
    labels = [b.split('_')[-1].upper()+' '+b.split('_')[-2] if len(b.split('_')) >= 3 else b for b in burst_ids]
    ax.set_yticks(range(len(burst_ids)), labels, fontsize=9); ax.set_xlabel('Acquisition UTC date'); _date_axis(ax)
    span = max(x.values()) - min(x.values()) or 1
    ax.set_xlim(min(x.values())-span*.02-1, max(x.values())+span*.02+1)
    stats.axis('off'); stats.set_title('Per-burst connectivity', loc='left', fontsize=10)
    ax.legend(handles=[Line2D([], [], color='#4579a7', marker='o', markersize=3, label='Shared-burst pair / available date'),
                       Line2D([], [], color='#c84545', marker='x', linestyle='none', label='Missing/unusable burst'),
                       Line2D([], [], color='#7f8995', marker='|', linestyle='none', label='Unknown availability')],
              loc='upper left', bbox_to_anchor=(0, 1.10), ncol=3, fontsize=8)
    fig.suptitle('Geographic-burst networks · component colors restart on each row\n'
                 'Gaps use scene-start UTC times; connectivity is metadata-based, not measured matching quality.', fontsize=12)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('plan', help='Saved stack plan JSON, or - to read standard input')
    parser.add_argument('-o', '--output-dir', type=Path, help='Default: <plan-stem>_plots beside the input')
    parser.add_argument('--gap-days', type=float, default=12, help='Flag scene-start gaps strictly above this many days (default: 12)')
    parser.add_argument('--burst', action='append', help='Show only this geographic burst in the burst figure; repeat as needed')
    parser.add_argument('--formats', nargs='+', choices=('png', 'pdf', 'svg'), default=['png'])
    parser.add_argument('--show', action='store_true', help='Also open Matplotlib windows')
    args = parser.parse_args(argv)
    try:
        source = None if args.plan == '-' else Path(args.plan).expanduser()
        plan = json.load(sys.stdin) if source is None else json.loads(source.read_text())
        summary = analyze_plan(plan, args.gap_days)
        requested = list(dict.fromkeys(args.burst or summary['bursts']))
        if set(requested) - summary['bursts'].keys():
            raise ValueError(f'Unknown burst IDs: {sorted(set(requested) - summary["bursts"].keys())}')
        output = (args.output_dir or (source.parent/(source.stem+'_plots') if source else Path('stack_plan_plots'))).expanduser()
        if source and output.resolve() == source.resolve():
            raise ValueError('Output must be a directory, not the plan file')
        import matplotlib
        if not args.show:matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        output.mkdir(parents=True, exist_ok=True)
        paths = []
        figures = [('overview', plot_overview(summary))]
        # Keep labels legible on large AOIs by paginating instead of dropping rows.
        for first in range(0, len(requested), 24):
            name = 'burst_network' if len(requested) <= 24 else f'burst_network_{first//24+1:02d}'
            figures.append((name, plot_bursts(summary, requested[first:first+24])))
        for name, fig in figures:
            for fmt in dict.fromkeys(args.formats):
                path = output/f'{name}.{fmt}'; fig.savefig(path, dpi=180); paths.append(str(path.resolve()))
        summary['source_plan'] = str(source.resolve()) if source else 'stdin'
        (output/'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
        print(f'{len(summary["dates"])} acquisitions | {len(summary["pairs"])} pairs | '
              f'{len(summary["components"])} planned components | {len(summary["isolated_dates"])} isolated dates')
        print(f'Any-shared-burst graph: {len(summary["usable_shared_components"])} components | '
              f'{sum(not p["usable_burst_ids"] for p in summary["pairs"])} pairs without usable shared bursts | '
              f'{len(summary["metadata_error_dates"])} metadata-error dates')
        print(f'Gaps > {args.gap_days:g} days: {len(summary["large_gaps"])} | '
              f'burst locations with multiple components: {sum(len(r["components"])>1 for r in summary["bursts"].values())}/{len(summary["bursts"])}')
        for warning in summary['warnings']:print('WARNING: '+warning, file=sys.stderr)
        for path in [*paths, str((output/'summary.json').resolve())]:print('Saved '+path)
        if args.show:plt.show()
        for _, fig in figures:plt.close(fig)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
