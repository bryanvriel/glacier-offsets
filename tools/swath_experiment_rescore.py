"""Compatibility import; use glacier_offsets.swath_experiment_rescore."""
import sys
from glacier_offsets import swath_experiment_rescore as _implementation

# Preserve legacy imports and monkeypatch targets for existing research scripts.
sys.modules[__name__] = _implementation
