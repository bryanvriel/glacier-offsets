#!/usr/bin/env python3
"""Compatibility entry point for glacier-offsets prepare-inputs."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from glacier_offsets.input_preparation import main

if __name__ == '__main__':
    sys.exit(main())
