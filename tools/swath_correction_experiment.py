"""Standalone read-only Larsen swath experiment; never runs production stages.

Example (use the ISCE3 environment):
  python -m tools.swath_correction_experiment run configs/stack.larsen.production.yaml \
      --output work/larsen/swath-experiment-2017h1
"""
import argparse
import json
import logging
import os
from pathlib import Path
import sys


RUN_STAGES = ('plan', 'audit', 'measure', 'physics', 'fit', 'previews', 'report')
STAGES = (*RUN_STAGES, 'rescore', 'run')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('config', type=Path)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1]/'work/larsen/swath-experiment-2017h1')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--pilot-only', action='store_true', help='Measure only the first three-date triangle')
    parser.add_argument('--retained-previews', action='store_true',
                        help='Refresh previews from verified retained radar arrays and frozen fit; no SLC reads')
    args = parser.parse_args(argv)
    if args.retained_previews and args.stage != 'previews':
        parser.error('--retained-previews requires the previews stage')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    os.environ.setdefault('MPLCONFIGDIR', str(args.output.resolve()/'.cache/matplotlib'))
    # Pyre interprets --config-like options during native initialization.
    sys.argv = sys.argv[:1]
    from tools.swath_experiment_io import Experiment
    from glacier_offsets.products import write_json
    experiment = Experiment(args.config, args.output, args.workers)
    with experiment.lock():
        if args.stage != 'rescore' and not args.retained_previews:
            experiment.plan()
        stages = RUN_STAGES if args.stage == 'run' else (args.stage,)
        result = None
        for stage in stages:
            logging.info('Experiment stage: %s', stage)
            if stage == 'plan':
                result = experiment.plan()
            elif stage == 'audit':
                from glacier_offsets.swath_physics import audit
                result = audit(experiment)
            elif stage == 'measure':
                from glacier_offsets.swath_observations import measure
                result = measure(experiment, args.pilot_only)
                write_json(experiment.output/'measure_stage.json', result)
            elif stage == 'physics':
                from glacier_offsets.swath_physics import physics
                result = physics(experiment)
                write_json(experiment.output/'physics_status.json', result)
            elif stage == 'fit':
                from tools.swath_experiment_analysis import fit
                result = fit(experiment)
            elif stage == 'previews':
                from tools.swath_experiment_previews import previews
                result = previews(experiment, retained=args.retained_previews)
            elif stage == 'rescore':
                from tools.swath_experiment_rescore import rescore
                result = rescore(experiment)
            else:
                from tools.swath_experiment_analysis import report
                result = report(experiment)
            logging.info('%s: %s', stage, result.get('status', 'complete'))
            if args.pilot_only and stage == 'measure':
                break
        experiment.inputs.unchanged()
    print(json.dumps({'stage': args.stage, 'status': result.get('status', 'complete'),
                      'output': str(experiment.output)}, indent=2))
    return 0 if result.get('status', 'complete') == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
