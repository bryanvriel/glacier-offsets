"""Stage CLI."""
import argparse
import logging
import subprocess
import sys
from .config import load_config
from .inputs import discover, select_pair


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == 'prepare-inputs':
        from .input_preparation import main as prepare_inputs
        return prepare_inputs(argv[1:])
    if argv and argv[0] == 'stack':
        from .stack_cli import main as stack_main
        return stack_main(argv[1:])
    parser = argparse.ArgumentParser(prog="glacier-offsets")
    parser.epilog = ('Stack workflows: glacier-offsets stack --help. '
                     'Download inputs: glacier-offsets prepare-inputs --help')
    parser.add_argument("command", choices=["list-bursts", "validate", "coregister", "assemble", "match", "geocode", "diagnostics", "run"])
    parser.add_argument("config")
    parser.add_argument("--backend", choices=["cpu", "gpu", "auto"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        cfg = load_config(args.config, discovery=args.command == "list-bursts")
        if args.command == "list-bursts":
            selection = None
            if cfg.get('aoi') is not None:
                from .aoi import select_aoi
                pairs, selection = select_aoi(cfg)
            else:
                pairs = discover(cfg)
            print("BURST ID            IW  REFERENCE SENSING START       SECONDARY SENSING START")
            for a, b in sorted(pairs, key=lambda pair: (pair[0].swath, pair[0].id)):
                print(f"{a.id:20} {a.swath}   {a.start.isoformat():28} {b.start.isoformat()}")
            print(f"{len(pairs)} selected / {selection['common_count']} common bursts" if selection else f"{len(pairs)} common bursts")
            return 0 if pairs else 2
        if args.command in {'validate','run'} or (cfg.get('aoi') is not None and args.command in {'geocode','diagnostics'}):
            from .geocode import validate_geocoding
            validate_geocoding(cfg.get('geocoding',{}))
        if args.backend:
            cfg["matching"]["backend"] = args.backend
        if args.command == "validate":
            if cfg.get('aoi') is not None:
                from .batch import validate_batch
                pairs = validate_batch(cfg)
            else:
                pairs = [select_pair(cfg)]
            from .environment import select_backend
            from osgeo import gdal
            gdal.UseExceptions()
            d = gdal.Open(cfg["inputs"]["dem"])
            if not d.GetProjection():
                raise ValueError("DEM must declare a CRS and ellipsoidal heights")
            backend = select_backend(cfg['matching']['backend'])[0]
            for pair in pairs:
                print(f"Validated {pair[0].id}; backend={backend}; DEM={d.RasterXSize}x{d.RasterYSize}")
            return 0
        if cfg.get('aoi') is not None:
            from .batch import execute_batch
            return execute_batch(cfg, args.command, force=args.force)
        if args.command=='assemble':raise ValueError('assemble requires an AOI configuration')
        from .pipeline import execute
        execute(cfg, args.command, force=args.force)
        return 0
    except KeyboardInterrupt:
        logging.error('Interrupted; completed burst products are preserved')
        return 130
    except (ValueError, RuntimeError, OSError, NotImplementedError, ImportError, subprocess.CalledProcessError) as exc:
        logging.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
