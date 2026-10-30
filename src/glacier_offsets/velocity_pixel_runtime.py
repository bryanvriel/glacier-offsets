"""Installed-module provenance and explicit auxiliary data for pixel runs."""
from importlib.util import find_spec
from pathlib import Path

from .velocity_io import digest


NUMERICAL_MODULES = (
    'velocity_basis', 'velocity_pixel', 'velocity_pixel_config',
    'velocity_systematics', 'velocity_quality', 'velocity_vertical',
    'velocity_pixel_inversion', 'velocity_pixel_runtime', 'velocity_pixel_mixed',
)
AUXILIARY_FILES = {'regions': 'regions.reviewed.json', 'regional_support': 'regional-support.npz'}


def module_fingerprints(names):
    """Use importable module identities, independent of checkout/install paths."""
    result = {}
    for name in names:
        module = name if name.startswith('glacier_offsets.') else 'glacier_offsets.' + name
        spec = find_spec(module)
        if spec is None or spec.origin is None:
            raise ValueError(f'Cannot locate installed module {module}')
        result[module] = digest(spec.origin)
    return result


def verify_code(record):
    for name, expected in record.items():
        if name.startswith('src/glacier_offsets/') and name.endswith('.py'):
            name = 'glacier_offsets.' + Path(name).stem
        if not name.startswith('glacier_offsets.'):
            raise ValueError('Historical recipe uses source-tree fingerprints; rerun with the installed pixel command')
        if module_fingerprints((name,))[name] != expected:
            raise ValueError(f'Numerical implementation changed: {name}')


def auxiliary_path(out, name, override=None, required=False):
    """Prefer an explicit path, otherwise use a copy staged in the run directory."""
    path = Path(override).expanduser().resolve() if override is not None else Path(out) / AUXILIARY_FILES[name]
    if path.is_file():
        return path
    if override is not None or required:
        raise ValueError(f'Missing {name}: {path}; supply its path in YAML or on the command line')
    return None


def auxiliary_fingerprints(out):
    return {filename: digest(Path(out) / filename) for filename in AUXILIARY_FILES.values()
            if (Path(out) / filename).is_file()}
