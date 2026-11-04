"""Portable installed CLI for the exploratory pixel inversion.

Version-2 YAMLs drive the complete netCDF-to-results workflow (``run``) or its
preparation stages (``prepare``). Version-1 YAMLs keep their prepared-directory
behavior under ``fit``. Export, verification, plotting and reports accept a
result directory or the YAML that names it.
"""
import argparse
import logging
from pathlib import Path

from .velocity_solid_earth import Interrupted


def _output(path):
    """Resolve a result directory from a directory or a version-1/2 YAML."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        return path
    from .velocity_workflow_config import config_version, load_workflow_config
    if config_version(path) == 2:
        return Path(load_workflow_config(path)['output'])
    from .velocity_pixel_config import load_config
    return Path(load_config(path)['output'])


def _v2(out):
    from .velocity_workflow import is_v2_output
    return is_v2_output(out)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='glacier-offsets velocity-pixels', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', help='Version-2 YAML: validate, prepare, calibrate noise, fit, export, verify and report')
    prepare = commands.add_parser('prepare', help='Version-2 YAML: extraction, solid-Earth correction and noise only')
    for command in (run, prepare):
        command.add_argument('config', type=Path)
    run.add_argument('--batch-size', type=int, help='Override execution.batch_size; no effect on scientific parameters')
    fit = commands.add_parser('fit', help='Fit from the inversion YAML (version 1: prepared directory; version 2: through fitting)')
    fit.add_argument('config', type=Path)
    fit.add_argument('--batch-size', type=int)
    for command in (fit, run):
        command.add_argument('--remove-mixed-windows', action=argparse.BooleanOptionalAction, default=None,
                             help='Override screening.remove_mixed_windows (YAML default: false)')
    export = commands.add_parser('export', help='Export maps using the recorded settings')
    export.add_argument('output', type=Path, help='Result directory or inversion YAML')
    export.add_argument('--regions', type=Path, help='Optional reviewed ROI overlay; otherwise use staged/configured regions')
    verify = commands.add_parser('verify', help='Check numerical replay and native maps')
    verify.add_argument('output', type=Path, help='Result directory or inversion YAML')
    verify.add_argument('--regional-support', type=Path, help='Version 1: optional frozen cohort; otherwise use staged support')
    plot = commands.add_parser('plot', help='Plot measured and modeled point time series for reviewed targets')
    plot.add_argument('output', type=Path, help='Result directory or inversion YAML')
    plot.add_argument('--regions', type=Path, help='Reviewed target regions; otherwise use staged/configured regions')
    maps = commands.add_parser('maps', help='Plot amplitude/phase from saved fits or exported NetCDF')
    from .velocity_pixel_maps import add_arguments
    add_arguments(maps)
    report = commands.add_parser('report', help='Write a report inside the result directory')
    report.add_argument('output', type=Path, help='Result directory or inversion YAML')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    try:
        if args.command in ('run', 'prepare') or (args.command == 'fit' and _is_v2_config(args.config)):
            from .velocity_workflow import run_workflow
            from .velocity_workflow_config import config_version
            if config_version(args.config) != 2:
                raise ValueError(f'{args.command} needs a version-2 YAML; use "velocity-pixels fit" for version-1 prepared directories')
            stages = {'run': 'run', 'prepare': 'prepare', 'fit': 'fit'}[args.command]
            options = {} if args.command == 'prepare' else {'batch_size': args.batch_size, 'remove_mixed_windows': args.remove_mixed_windows}
            result = run_workflow(args.config, stages, **options)
            print(Path(result['config']['output']))
        elif args.command == 'fit':
            from .velocity_pixel_inversion import main as fit_main
            options = ['--config', str(args.config)]
            if args.batch_size is not None:
                options += ['--batch-size', str(args.batch_size)]
            if args.remove_mixed_windows is not None:
                options += ['--remove-mixed-windows' if args.remove_mixed_windows else '--no-remove-mixed-windows']
            fit_main(options)
        elif args.command == 'export':
            from .velocity_pixel_export import export
            export(_output(args.output), args.regions)
        elif args.command == 'verify':
            out = _output(args.output)
            if _v2(out):
                from .velocity_workflow_products import RunView, verify_v2
                verify_v2(RunView.v2(out))
            else:
                from .velocity_pixel_verify import verify
                verify(out, args.regional_support)
        elif args.command == 'plot':
            from .velocity_pixel_plot import run as plot_run
            plot_run(_output(args.output), args.regions)
        elif args.command == 'maps':
            from .velocity_pixel_maps import run_cli
            run_cli(args)
        else:
            out = _output(args.output)
            if _v2(out):
                import json
                from .velocity_workflow_products import RunView, report_v2
                points = out/'point_timeseries/points.json'
                report_v2(RunView.v2(out), json.loads(points.read_text()) if points.is_file() else None)
                print(out/'report.md')
            else:
                from .velocity_pixel_report import run as report_run
                report_run(out)
        return 0
    except (KeyboardInterrupt, Interrupted) as error:
        logging.error('Interrupted (%s); completed checkpoints are kept and the next run resumes them', error or 'signal')
        return 130
    except (ValueError, OSError, ImportError, KeyError) as error:
        parser.error(str(error))


def _is_v2_config(path):
    from .velocity_workflow_config import config_version
    try:
        return config_version(path) == 2
    except (OSError, ValueError):
        return False


if __name__ == '__main__':
    raise SystemExit(main())
