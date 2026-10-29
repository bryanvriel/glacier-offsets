"""Compatibility import; use glacier_offsets.swath_experiment_analysis."""
import sys
from glacier_offsets import swath_experiment_analysis as _implementation

# Preserve legacy imports and monkeypatch targets for existing research scripts.
sys.modules[__name__] = _implementation
