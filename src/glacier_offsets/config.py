"""Stage-specific configuration validation; discovery is metadata-only."""
from copy import deepcopy
import math
from pathlib import Path
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).with_name('defaults.yaml')
PATH_FIELDS = (
    *(('inputs', key) for key in ('reference_safe', 'secondary_safe',
                                 'reference_orbit', 'secondary_orbit', 'dem')),
    ('output', 'directory'),
    ('aoi', 'path'),
    ('gross_offsets', 'velocity', 'vx', 'path'),
    ('gross_offsets', 'velocity', 'vy', 'path'),
)


def resolve_config_paths(cfg: dict, directory: Path) -> dict:
    """Resolve declared paths against their YAML directory, before inheritance."""
    for field in PATH_FIELDS:
        section = cfg
        for key in field[:-1]:
            section = section.get(key) if isinstance(section, dict) else None
        if not isinstance(section, dict) or field[-1] not in section:
            continue
        value = section[field[-1]]
        if value is None:
            continue  # Leave missing required values to stage-specific validation.
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{".".join(field)} must be a nonempty local path string')
        local = Path(value).expanduser()
        section[field[-1]] = str(local if local.is_absolute() else (directory / local).resolve())
    return cfg


def _read_mapping(path: Path) -> dict:
    try:
        cfg = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f'Cannot read configuration {path}: {exc}') from exc
    if not isinstance(cfg, dict):
        raise ValueError(f'Configuration must be a YAML mapping: {path}')
    return cfg


def load_defaults() -> dict:
    """Read the single packaged defaults file, also linked from configs/."""
    return resolve_config_paths(_read_mapping(DEFAULT_CONFIG_PATH), DEFAULT_CONFIG_PATH.resolve().parent)


def merge_config(base: dict, overrides: dict) -> dict:
    """Recursively merge mappings; scalar values and entire lists replace base."""
    result = deepcopy(base)
    for key, value in overrides.items():
        result[key] = (merge_config(result[key], value)
                       if isinstance(result.get(key), dict) and isinstance(value, dict)
                       else deepcopy(value))
    return result


def config_overrides(cfg: dict, defaults: dict | None = None) -> dict:
    """Keep only differences, for compact YAMLs that inherit future defaults."""
    defaults = load_defaults() if defaults is None else defaults
    result = {}
    for key, value in cfg.items():
        if key not in defaults:
            result[key] = deepcopy(value)
        elif isinstance(value, dict) and isinstance(defaults[key], dict):
            nested = config_overrides(value, defaults[key])
            if nested:
                result[key] = nested
        elif value != defaults[key]:
            result[key] = deepcopy(value)
    return result


def _load_overrides(path: Path, ancestors: tuple[Path, ...] = ()) -> dict:
    path = path.expanduser().resolve()
    if path in ancestors:
        raise ValueError('Configuration inheritance cycle: ' + ' -> '.join(map(str, (*ancestors, path))))
    cfg = resolve_config_paths(_read_mapping(path), path.parent)
    parent = cfg.pop('extends', 'defaults')
    if not isinstance(parent, str) or not parent.strip():
        raise ValueError(f'extends must be a nonempty YAML path or "defaults": {path}')
    if parent == 'defaults':
        return cfg
    parent_path = Path(parent).expanduser()
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return merge_config(_load_overrides(parent_path, (*ancestors, path)), cfg)


DEFAULT_MATCHING = load_defaults()['matching']  # Compatibility for native API clients.


def gross_values(cfg: dict) -> tuple[int, int]:
    defaults = load_defaults()['gross_offsets']
    gross = cfg.get("gross_offsets", defaults)
    if not isinstance(gross,dict):raise ValueError("gross_offsets must be a YAML mapping")
    mode = gross.get("mode", defaults['mode'])
    if mode == "velocity":
        raise ValueError("velocity gross offsets require reference geometry; use the velocity-prior adapter")
    if mode == "zero":
        return (0, 0)
    if mode != "constant":
        raise ValueError("gross_offsets.mode must be zero, constant or velocity")
    const = gross.get("constant", {})
    if not isinstance(const,dict):raise ValueError("gross_offsets.constant must be a YAML mapping")
    values = tuple(const.get(k) for k in ("azimuth_px", "range_px"))
    if any(type(v) is not int or not -(2**31) <= v < 2**31 for v in values):
        raise ValueError("constant azimuth_px and range_px must be integer full-resolution reference SLC pixels")
    return values


def load_config(path: str | Path, discovery: bool = False) -> dict:
    """Load a pair; local paths are absolute internally and YAML-relative on disk."""
    cfg = merge_config(load_defaults(), _load_overrides(Path(path)))
    inp = cfg.get("inputs", {})
    if not isinstance(inp,dict):raise ValueError("inputs must be a YAML mapping")
    for key in ("reference_safe", "secondary_safe", "reference_orbit", "secondary_orbit"):
        value = inp.get(key)
        if not value or not Path(value).is_absolute() or not Path(value).exists():
            raise ValueError(f"inputs.{key} must be an existing local path (resolved: {value!r})")
    if not isinstance(inp.get('polarization'),str):raise ValueError('inputs.polarization must be a string')
    inp["polarization"] = inp["polarization"].upper()
    if inp["polarization"] not in {"HH", "HV", "VV", "VH"}:
        raise ValueError("polarization must be HH/HV/VV/VH")
    swaths = inp['subswaths']
    if not swaths or any(type(x) is not int or x not in {1, 2, 3} for x in swaths) or len(set(swaths)) != len(swaths):
        raise ValueError("subswaths must be distinct integers from 1,2,3")
    inp["subswaths"] = swaths
    aoi = cfg.get('aoi')
    if aoi is not None:
        from .aoi import validate_specification
        validate_specification(aoi)
        if inp.get('burst_id'):
            raise ValueError('Choose inputs.burst_id or aoi exclusively; clear inherited burst_id with null')
    if discovery:
        return cfg
    from .assembly import settings as assembly_settings
    assembly_settings(cfg)
    for key in ("dem",):
        if not inp.get(key) or not Path(inp[key]).is_absolute() or not Path(inp[key]).is_file():
            raise ValueError(f"inputs.{key} must be an existing local file (resolved: {inp.get(key)!r})")
    if not inp.get("burst_id") and aoi is None:
        raise ValueError("inputs.burst_id or aoi is required; choose one with glacier-offsets list-bursts")
    if not isinstance(cfg.get('output',{}),dict):raise ValueError('output must be a YAML mapping')
    if not isinstance(cfg.get('geocoding'),dict):raise ValueError('geocoding must be a YAML mapping')
    output = cfg.get("output", {}).get("directory")
    if not output or not Path(output).is_absolute():
        raise ValueError("output.directory must be a nonempty local path")
    if isinstance(cfg.get('gross_offsets'),dict) and cfg['gross_offsets'].get('mode')=='velocity':
        from .velocity_prior import validate_velocity
        validate_velocity(cfg['gross_offsets'].get('velocity'))
    else:
        gross_values(cfg)
    matching = cfg['matching']
    if not isinstance(matching,dict):raise ValueError('matching must be a YAML mapping')
    if matching["backend"] not in {"cpu", "gpu", "auto"}:
        raise ValueError("backend must be cpu, gpu, or auto")
    for key in ("window", "search", "stride", "batch"):
        v = matching[key]
        if not isinstance(v,(list,tuple)) or len(v) != 2 or any(type(x) is not int or x <= 0 for x in v):
            raise ValueError(f"matching.{key} needs two positive integers [azimuth,range]")
    unknown=set(cfg.get('matching',{}))-set(DEFAULT_MATCHING)-{'first_center'}
    if unknown: raise ValueError(f'Unknown matching options: {sorted(unknown)}')
    for key in ('raw_oversampling','oversampling','zoom'):
        v=matching[key]
        if type(v) is not int or v<=0 or v&(v-1):
            raise ValueError(f'matching.{key} must be a positive power of two')
    if matching['zoom']<2*matching['raw_oversampling'] or matching['zoom']>2*min(matching['search'])*matching['raw_oversampling']:
        raise ValueError('zoom must span at least one raw refinement pixel and fit within the residual search')
    if type(matching['deramp']) is not int or matching['deramp'] not in (0,1,2):
        raise ValueError('matching.deramp must be 0,1,2')
    if any(v<8 for v in matching['window']):raise ValueError('Matching windows must be at least 8 pixels per axis')
    for key in ('snr_threshold','correlation_threshold'):
        if type(matching[key]) not in (int,float) or not math.isfinite(matching[key]) or matching[key]<0:
            raise ValueError(f'matching.{key} must be nonnegative')
    if matching['correlation_threshold']>1:raise ValueError('Normalized correlation threshold cannot exceed one')
    if 'first_center' in matching and (len(matching['first_center'])!=2 or any(not isinstance(v,(int,float)) for v in matching['first_center'])):
        raise ValueError('matching.first_center needs [azimuth line, range sample] in zero-based SLC coordinates')
    cfg["matching"] = matching
    return cfg
