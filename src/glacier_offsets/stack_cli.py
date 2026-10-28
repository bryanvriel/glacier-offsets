"""Public stack command group, keeping metadata-only operations read-only."""
import argparse
import json
import logging
import subprocess

from .stack import StackPreflightError, plan_stack, validate_stack
from .stack_config import load_stack_config, load_status_config
from .stack_stages import STAGE_DESCRIPTIONS


def main(argv=None):
    if argv and argv[0] == 'init':
        from .stack_initialization import main as initialize
        return initialize(argv[1:])
    parser = argparse.ArgumentParser(prog='glacier-offsets stack',
        epilog='Use glacier-offsets stack init --help to initialize a stack YAML.')
    parser.add_argument('command', choices=['stages', 'status', 'plan', 'validate', 'coregister', 'assemble', 'match',
                                           'geocode', 'diagnostics', 'correct-swaths', 'run', 'rebuild-layout', 'export-reference'])
    parser.add_argument('config', nargs='?')
    parser.add_argument('--output', help='New reference bundle directory (export-reference only)')
    parser.add_argument('--backend', choices=['cpu', 'gpu', 'auto'])
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Verify and preview rebuild-layout without changing files')
    parser.add_argument('--json', action='store_true', help='Machine-readable stack status')
    parser.add_argument('--pair', action='append', help='Saved pair to correct (repeatable; correct-swaths only)')
    args = parser.parse_args(argv)
    if args.output and args.command != 'export-reference':
        parser.error('--output is supported by export-reference')
    if args.command == 'export-reference' and (not args.output or args.force or args.backend):
        parser.error('export-reference requires --output and does not accept --force or --backend')
    if args.pair and args.command != 'correct-swaths':
        parser.error('--pair is supported by correct-swaths')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    if args.json and args.command != 'status':
        parser.error('--json is supported by status')
    if args.dry_run and args.command != 'rebuild-layout':
        parser.error('--dry-run is supported by rebuild-layout')
    if args.command == 'rebuild-layout' and args.force:
        parser.error('rebuild-layout preserves coregistration; --force is not supported')
    if args.command in ('stages', 'status') and args.force:
        parser.error('--force is not supported by read-only inspection commands')
    if args.command == 'stages':
        for stage, description in STAGE_DESCRIPTIONS.items():
            print(f'{stage:12} {description}')
        print('\nrun executes all stages; plan and validate inspect inputs without processing.')
        print('Input downloads: glacier-offsets prepare-inputs --help')
        return 0
    if not args.config:
        parser.error('config is required for this command')
    try:
        if args.command == 'status':
            from .stack_status import inspect_stack, format_status
            cfg = load_status_config(args.config)
            if args.backend:
                cfg['matching']['backend'] = args.backend
            report = inspect_stack(cfg)
            print(json.dumps(report, indent=2) if args.json else format_status(report))
            return 0
        if args.command == 'export-reference':
            from .reference_bundle import export_reference
            cfg = load_status_config(args.config)
            print(json.dumps(export_reference(cfg, args.output), indent=2))
            return 0
        if args.command == 'correct-swaths':
            from .swath_correction import execute_corrections
            cfg = load_status_config(args.config)
            if args.backend:
                cfg['matching']['backend'] = args.backend
            return execute_corrections(cfg, force=args.force, pairs=args.pair)
        cfg = load_stack_config(args.config, discovery=args.command == 'plan')
        if args.backend:
            cfg['matching']['backend'] = args.backend
        if args.command == 'rebuild-layout':
            from .stack_rebuild import rebuild_layout
            print(json.dumps(rebuild_layout(cfg, dry_run=args.dry_run), indent=2))
            return 0
        if args.command in ('plan', 'validate'):
            plan = plan_stack(cfg)
            if args.command == 'validate':
                plan.report['validated_backend'] = validate_stack(cfg, plan)
            print(json.dumps(plan.report, indent=2, sort_keys=True))
            return 0
        from .stack_workflow import execute_stack
        return execute_stack(cfg, args.command, force=args.force)
    except StackPreflightError as exc:
        print(json.dumps(exc.report, indent=2, sort_keys=True))
        logging.error('%s', exc)
        return 2
    except KeyboardInterrupt:
        logging.error('Interrupted; completed stack products are preserved')
        return 130
    except (ValueError, RuntimeError, OSError, NotImplementedError, ImportError, subprocess.CalledProcessError) as exc:
        logging.error('%s', exc)
        return 2
