#!/usr/bin/env python
"""Initialize reference-geometry stack processing from local acquisitions."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from glacier_offsets.stack_initialization import main

if __name__ == '__main__':
    sys.exit(main())
