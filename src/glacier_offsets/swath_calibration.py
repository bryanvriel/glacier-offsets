"""Estimate and export a relative IW1/IW2 spacecraft azimuth calibration."""
import argparse
from contextlib import ExitStack
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

RUN_STAGES = ('plan', 'audit', 'measure', 'physics', 'fit', 'previews', 'report')
STAGES = (*RUN_STAGES, 'export', 'rescore', 'run')


def _parser(legacy):
    parser = argparse.ArgumentParser(prog=None if legacy else 'glacier-offsets stack calibrate-swaths',
                                     description=__doc__)
    if legacy:
        parser.add_argument('stage', choices=STAGES)
        parser.add_argument('config', type=Path)
        parser.add_argument('--output', type=Path, default=Path.cwd()/'swath-calibration')
    else:
        parser.add_argument('config', nargs='?', type=Path,
                            help='Stack YAML; optional for export of results with saved reference metadata')
        parser.add_argument('--stage', choices=STAGES, default='run')
        parser.add_argument('--output', required=True, type=Path, help='Separate calibration results directory')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--pilot-only', action='store_true', help='Measure only the first three-date triangle; no export')
    parser.add_argument('--retained-previews', action='store_true',
                        help='Refresh verified retained previews without SLC reads (previews stage only)')
    parser.add_argument('--calibration-output', type=Path, help='Calibration JSON path; default RESULTS_DIR/calibration.json')
    parser.add_argument('--force', action='store_true', help='Replace a differing calibration JSON/checksum; does not force matching')
    return parser


def _run(args, *, legacy=False):
    from .swath_experiment_io import Experiment, result_lock
    from .products import write_json
    if args.stage == 'export':
        from .stack_config import load_status_config
        from .stack_lock import stack_lock
        from .swath_calibration_export import export_calibration
        cfg = load_status_config(args.config) if args.config else None
        with ExitStack() as locks:
            if cfg is not None:
                locks.enter_context(stack_lock(cfg['output']['directory']))
            locks.enter_context(result_lock(args.output))
            return export_calibration(args.output, args.calibration_output, cfg=cfg, force=args.force)
    experiment = Experiment(args.config, args.output, args.workers)
    with experiment.lock():
        if args.stage != 'rescore' and not args.retained_previews:
            experiment.plan()
        stages = (*RUN_STAGES, 'export') if args.stage == 'run' and not legacy else (
            RUN_STAGES if args.stage == 'run' else (args.stage,))
        result = None
        for stage in stages:
            logging.info('Swath calibration stage: %s', stage)
            if stage == 'plan':
                result = experiment.plan()
            elif stage == 'audit':
                from .swath_physics import audit
                result = audit(experiment)
            elif stage == 'measure':
                from .swath_observations import measure
                result = measure(experiment, args.pilot_only)
                write_json(experiment.output/'measure_stage.json', result)
            elif stage == 'physics':
                from .swath_physics import physics
                result = physics(experiment)
                write_json(experiment.output/'physics_status.json', result)
            elif stage == 'fit':
                from .swath_experiment_analysis import fit
                result = fit(experiment)
            elif stage == 'previews':
                from .swath_experiment_previews import previews
                result = previews(experiment, retained=args.retained_previews)
            elif stage == 'rescore':
                from .swath_experiment_rescore import rescore
                result = rescore(experiment)
            elif stage == 'export':
                from .swath_calibration_export import export_calibration
                result = export_calibration(experiment.output, args.calibration_output,
                                            cfg=experiment.cfg, force=args.force)
            else:
                from .swath_experiment_analysis import report
                result = report(experiment)
            logging.info('%s: %s', stage, result.get('status', 'complete'))
            if (args.pilot_only and stage == 'measure') or (not legacy and result.get('status', 'complete') != 'complete'):
                break
        experiment.inputs.unchanged()
    return result


def main(argv=None, *, legacy=False):
    parser = _parser(legacy)
    args = parser.parse_args(argv)
    if args.config is None and args.stage != 'export':
        parser.error('CONFIG.yaml is required except for --stage export')
    if args.retained_previews and args.stage != 'previews':
        parser.error('--retained-previews requires --stage previews')
    if args.pilot_only and args.stage not in ('run', 'measure'):
        parser.error('--pilot-only requires the run or measure stage')
    if (args.force or args.calibration_output) and (args.stage not in ('run', 'export') or args.pilot_only):
        parser.error('--force and --calibration-output require a run that exports or --stage export')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    os.environ.setdefault('MPLCONFIGDIR', str(args.output.resolve()/'.cache/matplotlib'))
    # Native Pyre initialization interprets CLI options; restore argv for callers.
    original_argv = sys.argv
    sys.argv = sys.argv[:1]
    try:
        result = _run(args, legacy=legacy)
        response = {'stage': args.stage, 'status': result.get('status', 'complete'),
                    'output': str(args.output.resolve())}
        if 'calibration' in result:
            response.update(result)
        print(json.dumps(response, indent=2))
        return 0 if response['status'] == 'complete' else 1
    except KeyboardInterrupt:
        logging.error('Interrupted; completed calibration measurements are preserved')
        return 130
    except (ValueError, KeyError, TypeError, OSError, ImportError, RuntimeError, subprocess.CalledProcessError) as exc:
        logging.error('%s', exc)
        return 2
    finally:
        sys.argv = original_argv


def legacy_main(argv=None):
    """Keep the historical stage-first research command, without auto-export."""
    return main(argv, legacy=True)


if __name__ == '__main__':
    raise SystemExit(main())
