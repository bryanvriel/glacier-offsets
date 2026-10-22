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
                                           'geocode', 'diagnostics', 'run'])
    parser.add_argument('config', nargs='?')
    parser.add_argument('--backend', choices=['cpu', 'gpu', 'auto'])
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--json', action='store_true', help='Machine-readable stack status')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    if args.json and args.command != 'status':
        parser.error('--json is supported by status')
    if args.command in ('stages', 'status') and args.force:
        parser.error('--force is not supported by read-only inspection commands')
    if args.command == 'stages':
        for stage, description in STAGE_DESCRIPTIONS.items():
            print(f'{stage:12} {description}')
        print('\nrun executes all stages; plan and validate inspect inputs without processing.')
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
        cfg = load_stack_config(args.config, discovery=args.command == 'plan')
        if args.backend:
            cfg['matching']['backend'] = args.backend
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
