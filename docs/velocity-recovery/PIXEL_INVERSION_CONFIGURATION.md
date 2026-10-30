# Configuring the pixel inversion

Two configuration versions exist. **Version 2** (below) runs the complete
netCDF-to-results workflow from raw displacement packages for one or more
calendar years. **Version 1** (from [Version 1](#version-1-prepared-directories)
onward) reuses a prepared directory and remains supported unchanged.

## Version 2: netCDF to results

```bash
glacier-offsets velocity-pixels run inversion.yaml      # validate → prepare → noise → fit → export/maps → verify/report
glacier-offsets velocity-pixels prepare inversion.yaml  # extraction, solid-Earth correction and noise only
```

See [`configs/velocity-pixel.v2.example.yaml`](../../configs/velocity-pixel.v2.example.yaml)
and the Larsen configuration [`configs/velocity-pixel.larsen.v2.yaml`](../../configs/velocity-pixel.larsen.v2.yaml).
Install the package into `ice` as in the version-1 instructions and run from any
directory. Activate the environment (or set `PROJ_DATA`): the workflow checks
pyproj, Python dependencies, the PySolid worker and auxiliary files before it
opens any input. `fit` with a version-2 YAML stops after fitting; `export`,
`verify`, `plot`, `maps` and `report` accept the result directory or the YAML.

### Schema

| Key | Meaning |
| --- | --- |
| `version: 2` | Required |
| `inputs` | Displacement packages (`glacier-offsets-displacements-v1`), any number, any order |
| `cache_directory` | Shared immutable artifacts; may serve several result directories |
| `output` | Result directory; must differ from the cache |
| `corrections.ocean_tide` | Only `require_existing`: CATS provenance and `tidal_correction_available` must be present; CATS is not computed here |
| `corrections.solid_earth` | `true` (default) applies full-ENU PySolid once; `false` records that none was applied |
| `noise.mode` | `supplied` or `controls`; `group_by: year` |
| `noise.files` | Supplied mode: list of paths or `{path, years}` |
| `noise.regions`, `noise.block_size_m` | Controls mode: reviewed regions JSON; spatial group size (default 3000 m) |
| `coverage.min_span_days` | **Required.** Other `screening`, `coverage`, `uncertainty` and `display` keys keep the version-1 defaults and checks |
| `diagnostics.regions` | Optional overlays and point series; defaults to `noise.regions` |
| `execution.solid_earth_python` | Interpreter for the PySolid worker; default is the current one |
| `execution.batch_size`, `preparation_chunk_pairs`, `solid_earth_spacing_deg` | 512, 8, 0.1; the first two are computational only |

Paths resolve relative to the YAML. Unknown or duplicate keys, invalid values
and settings the collected pairs can never meet (for example a span longer than
the inputs, or too few pairs in a year) are reported together before any
preparation. `remove_mixed_windows` stays `false` by default; `run`/`fit` accept
`--remove-mixed-windows`/`--no-remove-mixed-windows` and `--batch-size`.

### Virtual collection

Packages must share EPSG/CRS definition, `x`/`y`, geometry, reference-date and
layout identities, burst IDs, source associations and ownership, reference
centre line/sample and observation longitude/latitude exactly. Range/azimuth
ENU sensitivities may differ by at most the documented `1e-5` export tolerance;
each package keeps its own vectors. Internal content checksums are verified once
per file content. Duplicate pair IDs, incomplete packages, unsupported spacecraft,
missing ocean-tide provenance or an already recorded solid-Earth correction are
rejected. Pairs are sorted by scene acquisition time (then pair ID), so input
order does not matter; nothing is merged, reprojected or bridged, and exact
owning-burst endpoints are never replaced by scene times.
`collection-manifest.json` records every pair's file, slot, calendar-midpoint
year, spacecraft order and whether it crosses a calendar year.

### Cache artifacts and the run manifest

`cache_directory` holds three independently identified, read-only artifacts:

- `observations/<id>/observations.h5`: unique radar sources, exact endpoints,
  range/azimuth metres, SNR/peak, `valid_including_mixed`, per-pair
  `mixed_window`, `known_mixed_source`, per-file geometry, CRS, coordinates,
  map association, height and CATS response. Identity: input checksums, pair
  mapping and extraction code.
- `solid_earth/<id>/solid_earth.h5`: PySolid corrections. Identity: observation
  artifact, worker code, spacing and the worker's pysolid/numpy/h5py versions.
- `noise/<id>/noise.json`: annual covariances and diagnostics. Supplied mode is
  bound to the files and labels; controls mode to observations, corrections,
  screening policy, reviewed regions, selected control sources and estimator.

Artifacts are built under an exclusive lock in a partial directory, checksummed,
made read-only and published by one rename; every reuse re-verifies checksums.
The result directory never copies them: `run-manifest.json` records their
directories and checksums.

The PySolid worker (`glacier_offsets/solid_earth_worker.py`) imports only numpy,
h5py and pysolid and runs as a file (`python -I`) under
`execution.solid_earth_python`, typically the `isce3` environment. It retains the
validated method: bilinear 0.1° grid, bracketing UTC seconds at exact burst
endpoints, projection through each package's geometry and sampled exact-point
checks within 10 µm.

### Annual noise

- **supplied**: JSON with `covariance_ra_m2_by_year: {"2017": [[...],[...]], ...}`,
  or the legacy unlabeled `covariance_by_year` list with explicit `years` in the
  YAML. Covariances must be finite, symmetric and positive definite. Each
  observed year needs exactly one; extra years are recorded as unused.
- **controls**: sources whose chip-centre positions lie inside reviewed
  `role: control` regions (never chosen from fitted amplitudes). For each year,
  the full-year record (not January–June) gives order-aware, 60-day
  acquisition-purged held-out residuals, then the spatially balanced robust
  covariance estimator with its minimum of 40 residuals. `noise.json` reports
  samples, control source IDs per region, training dates and diagnostics. A year
  that cannot be calibrated fails; pooled or other-year noise is never used.

### Model with any number of years

Parameters are named: `east_velocity_<year>`, `north_velocity_<year>`,
`order_<year>` (only where both spacecraft orders occur), shared
`vertical_M2_C/S`, `vertical_O1_C/S` and `Msf_C/S`; the azimuth comparator uses
`azimuth_velocity_<year>`. Year regimes start on 1 January of each observed year
(the first and last are open-ended). Secular displacement integrates each
year's velocity over the pair's exact burst-endpoint overlap with that year;
annual noise and order offsets follow the recorded calendar-midpoint year.
Within-year pairs use the version-1 screen unchanged. Cross-year pairs are
screened against integrated annual robust backgrounds trained on within-year
pairs with every shared acquisition day purged. The flow axis is the equal-year
mean; `year_direction_change_deg` is the maximum pairwise annual change (zero
for one year). For two within-year years the design reproduces version 1 column
for column.

### Outputs and restart

`source_coefficients.h5` (with map association, CRS and year labels),
`coefficient_maps.nc`, GeoTIFFs (amplitude/phase, residual RMS, retained
pairs, speed), the quicklook, amplitude/phase, azimuth, residual-vertical, fit
and residual/coverage diagnostic figures, `amplitude_phase_maps/` for all four
components, `map-summary.json`, `verification.json` and `report.md`. Reviewed
`role: target` regions (any number) add `point_timeseries/`. Axis labels and
GeoTIFF CRS come from the inputs.

Preparation and fitting run in bounded chunks with checkpoints; an interrupted
command (exit 130) resumes on rerun when inputs, settings and code match, and
produces the same result as an uninterrupted run. `stages/*.json` records each
result stage's identity: display changes redraw exports only; screening changes
recompute controls noise and fits; correction changes create new correction,
noise and fit identities. Writers are exclusive per result directory and per
cache artifact.

### Validation (2026-10-30)

On the Larsen packages with the frozen supplied covariances
(`noise.json` labeled 2017, 2018) and `remove_mixed_windows: true`, the version-2
run reproduced `.runtime/velocity-recovery-larsen/maps-2017-2018` exactly: identical
observations, solid-Earth corrections (difference 0.0 m), masks, flow axes,
coefficients and RMS (maximum differences 0.0), and all 61 shared NetCDF
variables. With mixed windows retained it reproduced the earlier check: 575
mixed sources, 554 estimable (0 when removed), 18,224/19,751 retained mixed-source
pairs by year, unchanged unmixed fits and exact agreement with the historical
validation subset. An installed copy run from outside the checkout with an
empty cache and controls noise generated every default output in 2.6 minutes.
`tests/test_velocity_workflow.py` covers single-, multi-, split-file and
cross-year cases, input-order and virtual/physical equivalence, geometry and
projection tolerance, duplicates, provenance, supplied and correlated control
noise, insufficient controls, both mixed-window policies, interruptions,
invalidation and locking.

## Version 1: prepared directories

The exploratory pixel inversion has its own YAML file:
[`configs/velocity-pixel.larsen.yaml`](../../configs/velocity-pixel.larsen.yaml).
It is separate from stack creation and from the earlier regional
`velocity-recovery.*.yaml` workflow. Its defaults reproduce the original
two-year model: common along-flow Msf and residual vertical M2/O1, with separate
EN secular velocities and azimuth spacecraft-order offsets per year. The
direct-azimuth comparator is also fitted.

### Install and run from any directory

Install the package once into the existing `ice` environment (which already
contains the scientific/GIS dependencies):

```bash
/Users/briel/miniforge3/envs/ice/bin/python -m pip install \
  --no-deps --no-build-isolation /Users/briel/data/insar/glacier_offsets
```

This is a regular installation: the inversion and diagnostic modules are
included in the package and do not import scripts from `tools/`. After
activating `ice`, run from any working directory, without `PYTHONPATH`:

```bash
glacier-offsets velocity-pixels fit /path/to/experiment.yaml
glacier-offsets velocity-pixels export /path/to/output
glacier-offsets velocity-pixels maps /path/to/output
glacier-offsets velocity-pixels maps /path/to/experiment.yaml
glacier-offsets velocity-pixels verify /path/to/output
glacier-offsets velocity-pixels plot /path/to/output
glacier-offsets velocity-pixels report /path/to/output
```

Without activation, use the full command path
`/Users/briel/miniforge3/envs/ice/bin/glacier-offsets`. The equivalent module
entry point is `python -m glacier_offsets.velocity_pixel_cli`. Use
`glacier-offsets velocity-pixels --help` for the available operations.

For the supplied Larsen configuration:

```bash
/Users/briel/miniforge3/envs/ice/bin/glacier-offsets velocity-pixels fit \
  /Users/briel/data/insar/glacier_offsets/configs/velocity-pixel.larsen.yaml

/Users/briel/miniforge3/envs/ice/bin/glacier-offsets velocity-pixels export \
  /Users/briel/data/insar/glacier_offsets/.runtime/velocity-recovery-larsen/maps-2017-2018-configured
```

`input_directory`, `output`, `regions` and `regional_support` resolve relative
to the YAML file, not the working directory. Copy the YAML for another
experiment and give it a new output path. If the YAML moves, adjust relative
paths or use absolute ones. The input and output directories must differ.

The runner copies `inputs.json`, `observations.h5`, `solid_earth_tides.json`,
`solid_earth_tides.h5` and `noise.json` from `input_directory` to `output` if
needed, checking hashes and rejecting conflicting destination files. The
prepared observation and tide caches occupy about 246 MiB for the current
dataset. Input netCDF paths recorded in `inputs.json` must remain accessible
for export and diagnostics. No stack creation or PySolid calculation is repeated.
The inversion requires prepared frozen weights; it does not implicitly train
new weights from files in the checkout.

Mixed-window sources are retained by default. Set the following option to
restore the earlier source-wide exclusion in both years:

```yaml
screening:
  remove_mixed_windows: true
```

For a single run, `glacier-offsets velocity-pixels fit experiment.yaml
--remove-mixed-windows` enables removal; `--no-remove-mixed-windows` overrides
an enabled YAML setting. Both overrides are recorded in the resolved YAML and
fit recipe.

The default is `false`. This changes the observations used by the fit, so use
a new experiment output directory and rerun the inversion; replotting an old
fit does not restore discarded observations. All other validity, finite-data,
tide-availability, peak, residual and coverage checks remain in force. A flagged
mixed window may retain its raw azimuth value when the swath correction is
unavailable; this is recorded rather than treated as a corrected observation.

New prepared caches preserve `valid_including_mixed` independently of this
choice. Older caches discarded the original validity flags for mixed sources.
On a new run with `remove_mixed_windows: false`, the runner recovers those
flags from the checksum-verified original input packages and writes a small
`mixed_window_support.h5` beside the copied cache. It does not modify the old
cache or infer validity from finite displacements alone. Original packages
must be accessible for this one-time recovery. Once recovered, the support
file is reused and copied with the prepared inputs. The recipe records the
policy and support-file hash. Historical fits with no such option continue
to replay their original exclusion.

Optional auxiliary paths in the YAML are:

```yaml
regions: /path/to/regions.reviewed.json
regional_support: /path/to/fitted-support.npz
```

They are copied to `regions.reviewed.json` and `regional-support.npz` in the
output. If the YAML omits a path, the runner reuses the corresponding file
from the prepared input directory when present. Export works without regions
and simply omits ROI outlines; point-series plots need at least one reviewed
target region (any number is supported). Verification always checks source replay, physical radar prediction
and native maps; the additional descriptive regional fits run only when a
frozen support file is supplied. Missing regional checks are recorded as such.
Export/plot accept `--regions FILE`; verification accepts
`--regional-support FILE` to supply these explicitly for older outputs.
These inputs are never inferred from the working directory.

Point selection retains its fixed geographic/geometry rules, separate from
map display filters. Very restrictive settings can leave no eligible points.
The existing `tools/*pixel*.py` scripts remain compatibility wrappers for an
installed package. The source-tree `bin/glacier-offsets` launcher also works
from another directory when called by its full path and uses the same ABI
isolation as the earlier temporal fitter.

### Settings

#### Plot amplitude and phase without rerunning inversion or export

```bash
glacier-offsets velocity-pixels maps /path/to/experiment.yaml

# All four components, using all stored estimable fits:
glacier-offsets velocity-pixels maps /path/to/experiment.yaml --component all --mask estimable

# Change only the display:
glacier-offsets velocity-pixels maps /path/to/experiment.yaml \
  --mask phase-quality --amplitude-max 0.15 --phase-range positive
```

The script is also available as
`python -m glacier_offsets.velocity_pixel_maps DIRECTORY_OR_YAML` or
`python /path/to/tools/plot_velocity_amplitude_phase.py DIRECTORY_OR_YAML` with the
installed package. It works from any directory in the `ice` environment.

When given YAML, the command resolves its `output` relative to the YAML's
directory. An output directory can still be supplied directly. The YAML is
used to locate the data; masks and thresholds come from the saved fit recipe
or NetCDF, so editing the YAML after fitting does not change the interpretation
of existing coefficients. No original input cache is needed for plotting.

It reads `observations.h5` and `source_coefficients.h5` directly when available,
using the run's recipe for display thresholds. It also accepts a directory
containing only `coefficient_maps.nc`, in which case it uses the saved map
fields and masks. It never needs the original displacement stacks, tide cache
or an inversion rerun. Source tables take precedence over an older map export.

PNG/PDF pairs and `plot-settings.json` are saved in `DIRECTORY/amplitude_phase_maps/`
by default; `--output-dir` selects another location. Source products are opened
read-only. Color limits clip colors, not stored coefficients. Gray is excluded
source footprint; white is outside it.

`--component` accepts `msf-along` (default), `msf-azimuth`, `vertical-m2`,
`vertical-o1` or `all`. Horizontal amplitudes are in m/day; residual vertical
**height** amplitudes are in metres. Phase is `atan2(S,C)` with the existing
2000-01-01 UTC origin. `--phase-range signed` uses −180..180° (default);
`positive` uses 0..360°. Satellite-azimuth phase retains its signed projection
convention, and the residual vertical phases describe height rather than velocity.

`--mask` accepts `default`, `estimable`, `standard`, `along`, `phase-quality`,
or `flow-proxy`. The default is the ordinary along-flow mask for joint-model
components and standard coverage for direct azimuth. Additional quality masks
are display diagnostics rather than calibrated detections. `--amplitude-max`
sets the upper color limit in the component's units; with `--component all`,
the same numeric limit is applied to every component. `--dpi` controls PNG
resolution (default 180). Optional reviewed regions come from the output
directory or `--regions FILE`.

#### Inversion settings

All scientific sections support partial overrides; omitted keys take the
documented defaults. Unknown keys, duplicate YAML keys, wrong value types,
nonfinite values and inconsistent thresholds fail before output creation.

| YAML setting | Default | Effect |
| --- | --- | --- |
| `screening.remove_mixed_windows` | `false` | Optional removal of known mixed-window sources in both years |
| `screening.min_peak` | 0.08 | Pair acceptance and robust screen/flow-axis training |
| `screening.max_range_residual_m` | 2 | Maximum absolute purged range residual, metres |
| `screening.max_azimuth_residual_m` | 5 | Maximum absolute purged azimuth residual, metres |
| `coverage.min_pairs_per_year` | 20 | Retained pairs required in **each** year |
| `coverage.min_pairs_total` | 40 | Retained pairs required overall for fit support |
| `coverage.min_span_days` | 540 | First retained early to last retained late endpoint |
| `coverage.phase_bins` | 12 | Number of equal bins over one Msf cycle |
| `coverage.min_phase_bins` | 9 | Occupied midpoint-phase bins required for fit support |
| `coverage.min_pairs_standard` | 60 | Additional total-pair requirement for standard map coverage |
| `uncertainty.cluster_days` | 30 | Calendar-cluster width for diagnostic covariance |
| `display.min_speed_m_day` | 0.1 | Ordinary moving-ice display threshold |
| `display.min_abs_azimuth_projection` | 0.3 | Minimum absolute flow projection in both years |
| `display.max_direction_change_deg` | 20 | Additional diagnostic/proxy direction filter |
| `display.diagnostic_min_abs_azimuth_projection` | 0.5 | Additional diagnostic/proxy projection filter |
| `display.max_phase_sigma_deg` | 30 | Diagnostic phase-SE filter |
| `display.min_amplitude_over_sigma` | 2 | Diagnostic amplitude/SE filter |
| `display.proxy_min_speed_m_day` | 0.5 | Additional fast-flow proxy speed filter |
| `execution.batch_size` | 512 | Sources processed per batch; computational only |

Pair counts refer to retained displacement **pairs**, each with range and
azimuth, not the number of scalar channel observations. Screening is performed
separately within each year using acquisition-purged secular/order residuals.
There is no SNR bypass. Full model rank and target estimability remain required
even if coverage thresholds are relaxed.

For example, to show fits with 45 retained pairs overall, keeping the original
fit-support criteria, change:

```yaml
coverage:
  min_pairs_standard: 45
```

This changes the standard-coverage mask. It does not remove observations from
a fit or alter coefficients already supported by the other coverage criteria.
By contrast, changing `screening` changes the fitted data and may change both
the inferred flow axis and the harmonic coefficients. Changing fit-support
criteria changes which computed fits are retained. Changing cluster width
changes diagnostic covariance, while leaving GLS coefficients unchanged.
`display` only controls exported masks and figures; all supported, estimable
fits remain in HDF5 and NetCDF.

Frozen yearly H1 control covariances are reused **even if screening changes**;
the YAML does not recalibrate weights or introduce quality-dependent weighting.
The model constituents, common harmonic across years, exact endpoint handling,
CATS/body-tide corrections and phase convention remain fixed. The YAML is scoped
to this two-year exploratory inversion, not arbitrary years or a model bank.

### Provenance and reuse

Each run records the full effective settings in `inversion-config.resolved.yaml`
and the scientific parameters in `recipe.json`. Recipe descriptions are generated
from the actual values. Input, noise, auxiliary data and numerical code hashes bind the fits to
their recipe. Code fingerprints use installed module names rather than paths
relative to the source tree. Export uses that recipe for masks, labels and NetCDF metadata;
verification and time-series plots replay its screening settings.

A repeat command reuses a completed fit only when the recipe matches. Changing
scientific settings, noise or numerical code requires a new output directory.
Do not delete the old recipe to force reuse. Incomplete checkpoints are rejected
for inspection. Batch size may be overridden with `--batch-size 256` without
changing the scientific recipe; the resolved YAML retains the first run's
execution setting.

The old positional command remains available for prepared directories with
default settings. Historical results without a `parameters` recipe section
use the original defaults when exported or plotted. Their recorded code hashes
remain historical; the strict verification command can therefore reject an old
recipe after numerical code changes. Reports are written inside their own output directory; the historical
checkpoint document in the checkout is never overwritten.
