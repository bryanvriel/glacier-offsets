"""Compatibility entry point; prefer glacier-offsets stack calibrate-swaths."""
from glacier_offsets.swath_calibration import RUN_STAGES, STAGES, legacy_main as main


if __name__ == '__main__':
    raise SystemExit(main())
