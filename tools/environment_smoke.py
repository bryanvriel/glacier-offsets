"""Run using the isce3 interpreter with PYTHONPATH=src."""
import json
from pathlib import Path
from glacier_offsets.environment import inspect_environment
result = inspect_environment()
Path('docs/environment.json').write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != 'conda_packages'}, indent=2))
