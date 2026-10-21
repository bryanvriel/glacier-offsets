"""Invoke unmodified COMPASS public radar stages with its 0.5.6 path adapter."""
from pathlib import Path
import sys
from compass.utils.runconfig import RunConfig
from compass import s1_rdr2geo, s1_geo2rdr, s1_resample


def run(path):
    cfg = RunConfig.load_from_yaml(str(path), 's1_cslc_radar')
    if len(cfg.bursts) != 1:
        raise ValueError(f'Exactly one burst/polarization required; COMPASS loaded {len(cfg.bursts)}')
    if cfg.is_reference:
        s1_rdr2geo.run(cfg)
    else:
        s1_geo2rdr.run(cfg)
        for paths in cfg.output_paths.values():
            for name in ('range.off', 'azimuth.off'):
                source = Path(paths.output_directory) / name
                target = Path(paths.scratch_directory) / name
                if not source.is_file():
                    raise FileNotFoundError(f'COMPASS geo2rdr did not write {source}')
                if target.is_symlink():
                    target.unlink()
                if not target.exists():
                    target.symlink_to(source)
                for suffix in ('.hdr', '.xml', '.vrt'):
                    ancillary = Path(str(source) + suffix)
                    link = Path(str(target) + suffix)
                    if ancillary.exists() and not link.exists():
                        link.symlink_to(ancillary)
        s1_resample.run(cfg)


if __name__ == '__main__':
    run(sys.argv[1])
