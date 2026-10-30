"""Portable installed CLI for the two-year exploratory pixel inversion."""
import argparse
import logging
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog='glacier-offsets velocity-pixels', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('fit', help='Fit from the separate inversion YAML')
    fit.add_argument('config', type=Path)
    fit.add_argument('--batch-size', type=int)
    fit.add_argument('--remove-mixed-windows', action=argparse.BooleanOptionalAction, default=None,
                     help='Override screening.remove_mixed_windows (YAML default: false)')
    export = commands.add_parser('export', help='Export maps using the recorded settings')
    export.add_argument('output', type=Path)
    export.add_argument('--regions', type=Path, help='Optional reviewed ROI overlay; otherwise use staged regions')
    verify = commands.add_parser('verify', help='Check numerical replay and native maps')
    verify.add_argument('output', type=Path)
    verify.add_argument('--regional-support', type=Path, help='Optional frozen cohort; otherwise use staged support')
    plot = commands.add_parser('plot', help='Plot measured and modeled shelf-point time series')
    plot.add_argument('output', type=Path)
    plot.add_argument('--regions', type=Path, help='Reviewed target regions; otherwise use staged regions')
    maps = commands.add_parser('maps', help='Plot amplitude/phase from saved fits or exported NetCDF')
    from .velocity_pixel_maps import add_arguments
    add_arguments(maps)
    report = commands.add_parser('report', help='Write a report inside the experiment directory')
    report.add_argument('output', type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    try:
        if args.command == 'fit':
            from .velocity_pixel_inversion import main as fit_main
            options = ['--config', str(args.config)]
            if args.batch_size is not None:
                options += ['--batch-size', str(args.batch_size)]
            if args.remove_mixed_windows is not None:
                options += ['--remove-mixed-windows' if args.remove_mixed_windows else '--no-remove-mixed-windows']
            fit_main(options)
        elif args.command == 'export':
            from .velocity_pixel_export import export
            export(args.output.resolve(), args.regions)
        elif args.command == 'verify':
            from .velocity_pixel_verify import verify
            verify(args.output.resolve(), args.regional_support)
        elif args.command == 'plot':
            from .velocity_pixel_plot import run
            run(args.output.resolve(), args.regions)
        elif args.command == 'maps':
            from .velocity_pixel_maps import run_cli
            run_cli(args)
        else:
            from .velocity_pixel_report import run
            run(args.output.resolve())
        return 0
    except (ValueError, OSError, ImportError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
