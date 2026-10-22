"""Metadata-only acquisition inventory, pinned reference selection and pair graph."""
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path

from .aoi import footprint, load_aoi, project_footprint
from .initialization import OrbitCatalog, read_scene
from .inputs import inventory
from .products import identity
from .stack_config import iso_date


class StackPreflightError(ValueError):
    """Invalid inventory with its inspectable decisions attached."""
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


@dataclass
class Acquisition:
    scene: object
    orbit: object
    bursts: dict
    error: str | None = None

    @property
    def date(self):
        return self.scene.start.date().isoformat()


@dataclass
class StackPlan:
    acquisitions: tuple
    reference: Acquisition
    bursts: tuple
    pairs: tuple
    report: dict
    anchor: dict


def scan_safes(directory):
    """Visit visible SAFE containers, pruning hidden and measurement trees."""
    result = []
    for root, directories, files in os.walk(directory):
        directories[:] = sorted(name for name in directories if not name.startswith('.'))
        for name in list(directories):
            if name.lower().endswith('.safe'):
                result.append(Path(root) / name)
                directories.remove(name)
        result.extend(Path(root) / name for name in sorted(files)
                      if not name.startswith('.') and name.lower().endswith('.zip'))
    return sorted(result)


def read_catalog(root):
    path = Path(root) / 'stack_manifest.json'
    catalog = json.loads(path.read_text()) if path.is_file() else {}
    geometry_path = Path(root) / 'geometry/manifest.json'
    if not catalog.get('anchor') and geometry_path.is_file():
        geometry = json.loads(geometry_path.read_text())
        if geometry.get('geometry_identity'):
            # Recover a pin if interrupted after geometry commit but before catalog commit.
            catalog.update(anchor=geometry['specification']['anchor'],
                           reference_specification_identity=geometry['geometry_identity'],
                           layout_identity=geometry['layout_identity'])
    return catalog


def pair_graph(acquisitions, settings):
    by_date = {a.date: a for a in acquisitions}
    omitted = []
    if settings['pairs_file']:
        links = set()
        with Path(settings['pairs_file']).open(newline='') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ['earlier', 'later']:
                raise ValueError('pairs_file must have exactly the CSV columns earlier,later')
            for row in reader:
                if None in row or any(v is None for v in row.values()):
                    raise ValueError('Malformed pairs_file row')
                earlier, later = (iso_date(row[k], f'pairs_file.{k}') for k in ('earlier', 'later'))
                if earlier not in by_date or later not in by_date:
                    raise ValueError(f'Unknown or ineligible acquisition in pairs_file: {earlier},{later}')
                if earlier >= later:
                    raise ValueError('pairs_file requires distinct chronological earlier,later dates')
                links.add((earlier, later))
        links = sorted(links)
    else:
        links = []
        n, cap = settings['neighbors'], settings['max_separation_days']
        for i, a in enumerate(acquisitions):
            for b in acquisitions[i+1:i+1+n]:
                seconds = (b.scene.start - a.scene.start).total_seconds()
                if cap is None or seconds <= cap * 86400:
                    links.append((a.date, b.date))
                else:
                    omitted.append({'earlier': a.date, 'later': b.date, 'separation_seconds': seconds,
                                    'reason': 'exceeds max_separation_days'})
    connected = {d for pair in links for d in pair}
    return tuple(links), omitted, sorted(set(by_date) - connected)


def plan_stack(cfg, previous=None):
    """Plan from local manifests/annotations only; never hash full SAFE contents."""
    settings, inp = cfg['stack'], cfg['inputs']
    root = cfg.get('output', {}).get('directory')
    if not root:
        raise ValueError('output.directory is required for stack planning')
    previous = read_catalog(root) if previous is None else previous
    pin = previous.get('anchor')
    report = {'inventory': [], 'errors': [], 'reference_date': None, 'selected_burst_ids': [],
              'pairs': {}, 'omitted_links': [], 'disconnected_acquisitions': []}
    errors = report['errors']
    scenes = []
    for path in scan_safes(settings['safe_directory']):
        try:
            scene = read_scene(path)
            row = {'path': str(scene.path), 'date': scene.start.date().isoformat(),
                   'start': scene.start.isoformat(), 'stop': scene.stop.isoformat(),
                   'mission': scene.mission, 'relative_orbit': scene.relative_orbit,
                   'direction': scene.direction, 'polarizations': scene.polarizations}
            scenes.append((scene, row))
        except ValueError as exc:
            row = {'path': str(path), 'eligible': False, 'reason': str(exc)}
            errors.append(str(exc))
        report['inventory'].append(row)
    try:
        catalog = OrbitCatalog(settings['orbit_directory'])
    except ValueError as exc:
        errors.append(str(exc))
        raise StackPreflightError(str(exc), report) from exc
    aoi = load_aoi(cfg['aoi']) if cfg.get('aoi') is not None else None
    wanted = inp.get('burst_id')
    if not aoi and not wanted:
        raise ValueError('inputs.burst_id or aoi is required for stack selection')
    candidates = []
    dates = {}
    for scene, row in sorted(scenes, key=lambda item: item[0].start):
        day = row['date']
        if ((settings['start_date'] and day < settings['start_date']) or
                (settings['end_date'] and day > settings['end_date'])):
            row.update(eligible=False, reason='outside inclusive UTC date filters')
            continue
        if inp['polarization'] not in scene.polarizations:
            row.update(eligible=False, reason=f"missing polarization {inp['polarization']}")
            continue
        dates.setdefault(day, []).append(str(scene.path))
        try:
            orbit = catalog.resolve(scene)
            bursts = inventory(scene.path, orbit.path, inp['polarization'], inp['subswaths'])
            indexed = {}
            for burst in bursts:
                if burst.id in indexed:
                    raise ValueError(f'Ambiguous duplicate geographic burst {burst.id}')
                indexed[burst.id] = burst
            selected = [b for b in bursts if (project_footprint(footprint(b), aoi.crs).intersection(aoi.geometry).area > 0
                                             if aoi else b.id == wanted)]
            row.update(orbit=str(orbit.path), burst_ids=sorted(indexed),
                       selected_burst_ids=sorted(b.id for b in selected), eligible=bool(selected),
                       reason='eligible' if selected else 'no AOI/explicit-burst coverage')
            if selected:
                candidates.append(Acquisition(scene, orbit, indexed))
        except (ValueError, RuntimeError) as exc:
            row.update(eligible=True, reason='metadata/orbit failure', error=str(exc))
            # Keep the date in the graph: a failed date must not be bridged over.
            candidates.append(Acquisition(scene, None, {}, str(exc)))
    for day, paths in dates.items():
        if len(paths) > 1:
            errors.append(f'Ambiguous duplicate acquisition date {day}: {paths}')
    groups = {(a.scene.relative_orbit, a.scene.direction) for a in candidates}
    if len(groups) > 1:
        errors.append(f'Mixed relative orbit/direction groups: {sorted(groups)}')
    eligible = [a for a in candidates if not a.error]
    requested = settings['reference_date']
    if pin:
        if requested and requested != pin['reference_date']:
            errors.append('Reference is pinned; changing it requires a new output root (including with --force)')
        requested = pin['reference_date']
    reference = next((a for a in eligible if a.date == requested), None) if requested else next(iter(eligible), None)
    if reference is None:
        errors.append(f'No eligible reference acquisition{": " + requested if requested else ""}')
    if errors:
        raise StackPreflightError('; '.join(errors), report)
    if aoi:
        bursts = tuple(sorted((b for b in reference.bursts.values()
                              if project_footprint(footprint(b), aoi.crs).intersection(aoi.geometry).area > 0),
                              key=lambda b: (b.swath, b.start is None, b.start, b.id)))
    else:
        bursts = (reference.bursts[wanted],)
    # AOI selects complete reference bursts; date coverage is a subset of this fixed list.
    ids = [b.id for b in bursts]
    if any(b.start is None for b in bursts):
        raise StackPreflightError('Reference bursts require exact sensing timestamps', report)
    selection = {'aoi': aoi.report if aoi else None, 'burst_id': wanted, 'selected_ids': ids,
                 'subswaths': inp['subswaths'], 'polarization': inp['polarization']}
    anchor = {'reference_date': reference.date, 'reference_safe': str(reference.scene.path),
              'reference_orbit': str(reference.orbit.path), 'dem': inp.get('dem'), 'selection': selection,
              'alignment_tolerance_px': .001}
    if pin and pin != anchor:
        raise StackPreflightError('Pinned reference inputs/selection changed; use a new output root', report)
    try:
        pairs, omitted, disconnected = pair_graph(candidates, settings['pairing'])
    except ValueError as exc:
        raise StackPreflightError(str(exc), report) from exc
    by_date = {a.date: a for a in candidates}
    for row in report['inventory']:
        if row.get('eligible'):
            a = by_date[row['date']]
            row.update(available_burst_ids=[b for b in ids if b in a.bursts],
                       missing_burst_ids=[b for b in ids if b not in a.bursts])
    report.update(reference_date=reference.date, selected_burst_ids=ids, anchor=anchor,
                  omitted_links=omitted, disconnected_acquisitions=disconnected)
    for earlier, later in pairs:
        a, b = by_date[earlier], by_date[later]
        shared = [key for key in ids if key in a.bursts and key in b.bursts]
        report['pairs'][f'{earlier}_{later}'] = {'earlier': earlier, 'later': later, 'active': True,
            'status': 'planned', 'shared_burst_ids': shared,
            'scene_interval_seconds': (b.scene.start-a.scene.start).total_seconds(),
            'burst_intervals': {key: {'acquisition_times': [x.start.isoformat() if x.start else None
                                                          for x in (a.bursts[key], b.bursts[key])],
                                     'interval_seconds': ((b.bursts[key].start-a.bursts[key].start).total_seconds()
                                                          if a.bursts[key].start and b.bursts[key].start else None)}
                                for key in shared}}
    report['plan_identity'] = identity(report)
    return StackPlan(tuple(candidates), reference, bursts, pairs, report, anchor)


def validate_stack(cfg, plan, check_dates=True):
    """Check reference placement, native sampling, DEM coverage and stage settings."""
    from .assembly import validate_native_layout, settings
    from .batch import dem_extent, effective_config, validate_burst
    from .geocode import validate_geocoding
    from .environment import select_backend
    merge = settings(cfg)
    if not merge['enabled']:
        raise ValueError('Stack processing requires burst_merge.enabled: true')
    if merge['alignment_tolerance_px'] != .001:
        raise ValueError('Stacks require burst_merge.alignment_tolerance_px: 0.001')
    validate_geocoding(cfg['geocoding'])
    extent = dem_extent(cfg)
    reference = {b.id: b for b in plan.bursts}
    validate_native_layout([(b, b) for b in plan.bursts], cfg)
    for b in plan.bursts:
        validate_burst(effective_config(cfg, b.id), (b, b), extent)
    if check_dates:
        for acquisition in plan.acquisitions:
            if acquisition.error:
                raise ValueError(f'{acquisition.date}: {acquisition.error}')
            for key in reference.keys() & acquisition.bursts.keys():
                validate_burst(effective_config(cfg, key), (reference[key], acquisition.bursts[key]), extent)
    return select_backend(cfg['matching']['backend'])[0]
