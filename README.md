# Sentinel-1 glacier offsets

A small, restartable Python workflow for local Sentinel-1 TOPS SAFE pairs and acquisition stacks, one polarization and either one geographic burst or complete bursts intersecting an AOI. COMPASS performs geometry-only radar coregistration; native ISCE3 Ampcor tracks **amplitudes of complex SLCs**. Offset products retain applied gross, native residual and total separately. Only total motion is converted to displacement/velocity and geocoded. Local georeferenced velocity fields can guide searches. Pair AOI processing assembles compatible SLCs on one radar grid before tracking beneath `merged/`. Stack processing prepares each acquisition once on a pinned reference frame and reuses its SLC across pairs. There are no downloads, NESD or custom correlators.

Git tracks Python source (including audit scripts), the launcher, package/environment files, this README and selected reference configs. Data, processing outputs, images, plans, validation reports and other working documents stay local; links to those records below describe the working installation and may not exist in a fresh clone. New reference configs or non-Python source assets must be added to the explicit allowlist in `.gitignore`.

## Environment and commands

The tested interpreter is `/Users/briel/miniforge3/envs/isce3/bin/python`. Use `conda activate isce3` on this machine. On a new CPU host, create the pinned environment with `conda env create -f environment.yml` and activate it, then `python -m pip install --no-deps --no-build-isolation -e .`. Native ISCE3, COMPASS, s1-reader and GDAL are conda dependencies, not PyPI substitutes. See [API audit](docs/m0-api.md) and [runtime/build provenance](docs/environment.json).

For a flat package list, [requirements.txt](requirements.txt) contains the same CPU dependencies and version/build constraints as [environment.yml](environment.yml):

```bash
conda install -n isce3 -c conda-forge --strict-channel-priority --file requirements.txt
conda activate isce3
python -m pip install --no-deps --no-build-isolation -e .
```

To create a new environment instead, use `conda create -n glacier-offsets -c conda-forge --strict-channel-priority --file requirements.txt`, then activate `glacier-offsets` and install the local package as above. The YAML already works with `conda env create -f environment.yml`; the Conda version inspected on this machine also supports `conda install --file environment.yml`. The flat list is compatible with older Conda package-list workflows and requires the channel on the command line. Keep the two files' dependency specifications synchronized. `requirements.txt` uses Conda syntax and should not be passed to `pip install -r`; `pyproject.toml` installs the workflow itself, while `requirements-diagnostics.txt` only supplies workspace-local plotting dependencies. Both full environment files select the CPU ISCE3 build; CUDA hosts need an appropriate supported build instead.

The workspace launcher uses the active Python, source tree and workspace-local plotting dependencies. On this machine:

```bash
conda activate isce3
export PATH="$PWD/bin:$PATH"
glacier-offsets list-bursts configs/pair.example.yaml
glacier-offsets validate configs/pair.yaml
glacier-offsets run configs/pair.yaml --backend auto
glacier-offsets diagnostics configs/pair.yaml
```

If the environment is not activated, prefix a launcher command with `GLACIER_OFFSETS_PYTHON=/Users/briel/miniforge3/envs/isce3/bin/python`. `PYTHONPATH=src:.runtime/deps python -m glacier_offsets ...` is also supported. Plotting dependencies were installed under `.runtime/deps` without changing the user's conda packages; their pins are in `requirements-diagnostics.txt`. Environment.yml includes the plotting stack for a new install.

Each processing stage is independently runnable:

```bash
glacier-offsets coregister configs/pair.yaml
glacier-offsets match configs/pair.yaml
glacier-offsets geocode configs/pair.yaml
glacier-offsets diagnostics configs/pair.yaml
```

CPU geometry/coregistration uses OpenMP where implemented by ISCE3. Set its thread limit before launching Python; COMPASS subprocesses inherit it:

```bash
OMP_NUM_THREADS=4 glacier-offsets run configs/pair.pig.yaml --backend cpu
```

The inspected native **PyCPUAmpcor 0.25.21 does not parallelize its chunk loop** and exposes no CPU thread-count setting. Its Ampcor FFTW code does not initialize threaded FFT plans. Setting `OMP_NUM_THREADS` therefore does not make this CPU matcher use four cores. `matching.batch` changes the number of windows processed together and memory use, not worker count; CPU `nStreams` is also not a CPU thread pool. OpenMP/FFTW capabilities elsewhere in the linked ISCE3 library do not imply parallel Ampcor. CUDA Ampcor is a separate backend. See [configuration and threading audit](docs/configuration/README.md) for the installed-library/source checks and the executed synthetic check. No process pool or parallel matching extension was added.

`--force` repeats the requested computation (`run --force` repeats all stages). Run identity includes inputs' SHA256 hashes, software versions, polarization, selected burst, matching parameters and gross mode/values. A change in matching/gross configuration invalidates matching and geocoding, while retaining geometric coregistration. Later stage commands reject stale preceding stages. Input hashes are checked even on restart; this takes time for large SAFE files. A failed stage clears its completion identity. A source SHA256 snapshot is recorded at validation; after editing workflow code, explicitly force the affected stages. Successful unchanged `run` reuses coregistration, matching and geocoding and refreshes diagnostics.

## Local inputs

### Reference-geometry stacks

Use [stack.example.yaml](configs/stack.example.yaml) or initialize a separate stack
configuration; no pair-specific SAFE/orbit fields are required:

```bash
glacier-offsets stack init --safe-dir /path/safes --orbit-dir /path/orbits \
  --dem /path/dem.tif --polarization HH --aoi /path/glacier.kml --epsg 3031 \
  --output configs/stack.generated.yaml --output-directory work/my_stack --relative-paths
glacier-offsets stack plan configs/stack.generated.yaml
glacier-offsets stack validate configs/stack.generated.yaml
OMP_NUM_THREADS=4 glacier-offsets stack run configs/stack.generated.yaml --backend cpu
```

`python tools/init_stack_config.py` accepts the same initialization options.

List processing stages or inspect completion without running them:

```bash
glacier-offsets stack stages
glacier-offsets stack status configs/stack.pig.interior.yaml --backend cpu
```

`status` shows an overall stage checklist and per-acquisition/per-pair checkmarks:
`✓` complete, `○` pending, `~` stale, `!` incomplete, `−` blocked/no work, and
`?` unknown. It checks saved completion identities, relevant configuration and
key output-file presence. Leftover files from invalidated stages do not count
as completed. Use the same backend override as your processing command when
comparing matching settings. Add `--json` for a machine-readable report.

Inspection is read-only and works with raw-input volumes disconnected. It uses
the last saved active plan (or existing output directories if there is no
catalog); it does not discover newly added SAFE files or rehash inputs/products.
Use `stack plan` to inspect a changed inventory. Missing stages are displayed
normally; a successfully read status exits zero even when processing is incomplete.

Choose `--burst-id`, `--bbox` or `--aoi` (including KML); an AOI selects complete
bursts. `--start-date`, `--end-date` and `--reference-date` use `YYYY-MM-DD`.
All paths follow the same YAML-relative inheritance rules as pair configurations.
`plan` reports the inventory, exclusions, reference, partial burst coverage,
pair graph, omitted links and disconnected dates. `validate` additionally checks
native metadata, reference-grid placement, DEM coverage and processing settings.
Both commands are read-only and avoid complete SAFE hashing.

V1 accepts one SAFE per UTC acquisition date, one relative orbit/direction group,
and one polarization. It scans directories recursively without descending inside
unpacked SAFEs, and selects local precise orbits with interpolation margins.
The earliest eligible date is the initial reference unless explicitly selected.
Automatic pairs connect each date to its next `stack.pairing.neighbors` dates
(default 1), retaining exact scene-start gaps of at most 12 days. Set
`max_separation_days: null` to disable the cap. A `pairs_file` CSV with exactly
`earlier,later` columns and ISO dates replaces automatic pairing and its cap;
duplicate links are removed, and unknown/reversed/self links are rejected.

Stages can also be run individually:

```bash
glacier-offsets stack coregister configs/stack.generated.yaml
glacier-offsets stack assemble configs/stack.generated.yaml
glacier-offsets stack match configs/stack.generated.yaml --backend cpu
glacier-offsets stack geocode configs/stack.generated.yaml
glacier-offsets stack diagnostics configs/stack.generated.yaml
```

`coregister` prepares each reference burst's SLC/topology once, and aligns every
available date/burst directly from its original SLC onto that reference grid.
`assemble` writes one stripped complex SLC and validity mask per date. The GDAL
mask is stored externally as `slc.tif.msk` to keep SLC strips contiguous for
native CPU Ampcor on macOS; keep this file beside `slc.tif`. Reference
annotation and actual SLC support determine fixed overlap ownership. Shared
regions switch at their azimuth midpoint, at least eight SLC lines inside both
valid edges; overlaps narrower than 16 lines are rejected. Unshared coverage is
preserved. The cut and its policy are recorded in the geometry and pair manifests.
Support uses the same connected-zero-fill
policy as matching: isolated complex zeros remain valid observations, while
explicit invalid masks, nonfinite samples and connected zero fill remain nodata.
Missing date bursts and holes remain nodata
even where another burst overlaps. Reference bursts must align within 0.001
pixels; fractional placement is rejected before processing.

Pairs reference the prepared date SLCs without copying them or invoking COMPASS.
Offsets are later-minus-earlier. Per-burst sensing times determine exact pair
durations; velocity-prior projection and physical pixel scales always use the
stack-reference burst geometry/orbit, including pairs between two other dates.
All pairs share a matching template and a map grid derived from reference
geometry, the configured EPSG and spacing. Missing coverage cannot change map
shape or transform. Pairs without shared support produce explicit empty products
without running native Ampcor.

Products are organized as follows:

```text
<output>/stack_manifest.json
<output>/geometry/                  # burst topology, fixed layout/ownership, map grid
<output>/acquisitions/<YYYY-MM-DD>/  # aligned bursts, slc.tif, slc.tif.msk, valid.tif, manifest.json
<output>/pairs/<earlier>_<later>/    # manifest.json, radar/, geo/, diagnostics/
```

Reference preparation pins the reference inputs, burst selection and layout.
Adding earlier dates preserves that pin. Changing the reference, selection,
reference geometry inputs or relevant geometry software requires a new output
root, including with `--force`. New dates and pairing changes reuse unrelated
SLCs and pair rasters; removed links remain on disk and become inactive catalog
entries. Matching/gross and geocoding changes invalidate their downstream stages.
`--force` refreshes the requested stage and invalidates dependents. Input hashes
are shared within an invocation; prepared SLCs and geometry are verified before reuse. Reports are
atomic, interrupted/failed stages lose completion, and independent dates/pairs
continue after individual failures. Exit codes are 0 complete, 1 processing
failure, 2 invalid preflight, and 130 interrupted. The catalog records coverage,
stage errors, elapsed time, storage and process memory high-water marks.

Synthetic native and regression validation, plus the outstanding real-data
gates, are recorded in [stack validation](docs/stack-processing/README.md).
Stacks built with earlier ownership policies require a new output root.
[stack.pig.interior.yaml](configs/stack.pig.interior.yaml) inherits the PIG
configuration and uses `work/pig/stack-interior` for the interior-cut policy:

```bash
glacier-offsets stack run configs/stack.pig.interior.yaml --backend cpu
```

Multi-slice assembly, NESD, time-series inversion, raster cubes, parallel stack
execution and fractional alignment of reference bursts remain outside V1.

### Initialize a pair YAML

Given two SAFE directories or ZIP files, a DEM and a directory of local precise orbits:

```bash
conda activate isce3
python tools/init_pair_config.py /path/first.zip /path/second.zip \
  --dem /path/dem.tif --orbit-dir /path/orbits \
  --output configs/my_pair.yaml --output-directory work/my_pair
bin/glacier-offsets list-bursts configs/my_pair.yaml
```

Set `inputs.burst_id` to one reported common geographic ID, then run:

```bash
bin/glacier-offsets validate configs/my_pair.yaml
bin/glacier-offsets run configs/my_pair.yaml --backend auto
```

The initializer reads `manifest.safe` without extracting SLC imagery, orders the acquisitions by exact UTC sensing start, checks relative orbit/direction compatibility, and writes absolute input/output paths by default. Add `--relative-paths` to write paths relative to the generated YAML's directory for syncing between machines. Command-line input paths and `--output-directory` are interpreted relative to the current working directory before being converted for the YAML. It recursively indexes `.EOF`/`.xml` files in the orbit directory. Only precise `AUX_POEORB` files with matching mission, valid increasing state-vector times, and full padded acquisition coverage qualify. The padding follows the installed s1-reader: one Sentinel-1 orbital period plus 60 seconds before acquisition start, and 60 seconds after stop. If multiple precise files qualify, selection prefers the latest XML creation date, then file version, with path as a deterministic tie-breaker. The generated YAML records the selected files explicitly. No orbits are downloaded.

Polarization is inferred when exactly one is common; otherwise supply `--polarization HH` (or VV/HV/VH). `--burst-id t038_080975_iw2` selects and validates an explicit burst during initialization; otherwise it is left unset for discovery. Matching uses the current glacier defaults, with automatic first-center placement. Geocoding defaults to the DEM's EPSG and 200 m spacing; supply a projected metre `--epsg` when the DEM is geographic or lacks a suitable EPSG identifier, and use `--spacing-m` to change spacing. DEM heights must already be above the WGS84 ellipsoid. Existing YAML files are preserved unless `--force` is supplied.

### Shared defaults and pair overrides

Edit **[configs/defaults.yaml](configs/defaults.yaml)** to change workflow defaults. It is a symlink to the single packaged source file `src/glacier_offsets/defaults.yaml`, so there is only one copy in the workspace. The installed wheel includes that file. Pair YAMLs contain only local inputs/output, geocoding EPSG, and values differing from the defaults:

```yaml
extends: defaults.yaml  # path relative to THIS YAML, not the working directory
inputs:
  reference_safe: ../data/safe/reference.zip
  secondary_safe: ../data/safe/secondary.zip
  reference_orbit: ../data/orbits/reference.EOF
  secondary_orbit: ../data/orbits/secondary.EOF
  dem: ../data/dem/elevation.tif
  polarization: HH
  burst_id: t038_080975_iw2
output:
  directory: ../work/my_pair
geocoding:
  epsg: 3031
matching:
  window: [96, 384]  # overrides this setting; other matching values are inherited
```

Mappings merge recursively; lists such as window/stride/subswaths replace the whole list. Absolute parent paths and further parent/child layers are supported. Cycles, missing parents and malformed inheritance raise clear errors. `extends: defaults` selects the packaged default file without a filesystem-relative path; the initializer writes this portable form, and omitting `extends` also inherits the packaged defaults for compatibility with earlier YAMLs. `extends` is removed from the resolved processing configuration. If the example YAML is copied to another directory, adjust its `extends` path or use `extends: defaults`.

Local paths may be absolute or relative to the directory of the YAML that declares them, regardless of where you run the command. This applies to SAFE files/directories, orbit files, the DEM, `output.directory`, and `gross_offsets.velocity.vx.path` / `vy.path`. Inherited paths remain relative to their parent YAML; a child override is relative to the child YAML. `~` expands to the current user's home directory. For a YAML in `configs/`, `../data/...` and `../work/...` refer to the project's `data/` and `work/` directories. Keep that layout when syncing between machines. Input paths must exist; the output directory may be created during processing. `load_config()` returns absolute paths for downstream processing.

All maintained pair YAMLs now inherit from the shared file. Changing a default affects pairs that have not explicitly overridden it. Existing result files are not rewritten on edit; processing identities use resolved settings when stages are next run. Archived validation configs/results retain their as-run parameters. The initializer writes only overrides, including an explicit EPSG inferred from the DEM; it does not duplicate matching defaults or a default map spacing. To freeze a configuration for archival reproducibility, save the resolved config from `load_config()` rather than a live inheritance document.

`--max-separation-days 12` optionally rejects a pair whose exact manifest start-time gap exceeds 12 days; a gap of exactly 12 days is accepted. This is an initialization check; processing still uses the selected burst's exact interval. The scene reader, orbit catalog and pair checks are reusable functions in `initialization.py`, so a future stack planner can index scenes/orbits once and filter candidate pairs by temporal separation. The current initializer generates one pair config and supports AOI selection; date-stack orchestration remains outside scope. Default output directories include both full SAFE product names to distinguish acquisition pairs before the existing per-burst subdirectory is added.

`configs/pair.example.yaml` contains the actual Jan 10/22 2020 S1A HH SAFE ZIPs, their precise orbits and the ellipsoidal-height REMA DEM. It intentionally omits burst_id, so `list-bursts` works before selection. This command needs only local SAFE/orbit paths, polarization, and optional integer subswaths; no DEM, AOI, matching settings or output directory is required for discovery. Geographic IDs, subswaths and UTC sensing times are printed. No common IDs exits 2; duplicated IDs or parse failures give an explicit error. No burst is automatically selected.

The user selected `t038_080975_iw2`; `configs/pair.yaml` contains this selection. `configs/pair.constant.yaml` uses the same pair/grid with constant azimuth +2, range −3 pixels. The configured output CRS is EPSG:3031 at 200 m. Input paths must be existing and local; absolute and YAML-relative paths are supported. DEM heights must be above the WGS84 ellipsoid; height datum cannot be inferred safely from an arbitrary GeoTIFF. These local data declare ellipsoidal heights in data/manifest.json.

```yaml
gross_offsets:
  mode: zero        # default; zero | constant
  constant:
    azimuth_px: 0   # integer full-resolution reference SLC pixels
    range_px: 0
```

In zero mode the constant values are ignored. Constant mode requires both integer components; search half-widths are residual widths around that prediction. `velocity` samples declared local horizontal velocity components at reference chip centers and projects predicted motion with native ISCE3 geometry; see [M5 inputs and validation](docs/m5.md) and [the PIG example](configs/pair.pig.velocity.yaml). Missing priors are masked, without a zero fallback. Gross offsets never enter COMPASS resampling and are added to residual offsets exactly once, with native `mergeGrossOffset=0`. COMPASS geometric offsets are preserved separately and are not glacier motion.

Matching arrays are `[azimuth, range]`: window, residual search half-width, stride and batch. `first_center` is optional `[zero-based SLC line, sample]`, not sparse-grid indices. Geometric centers are start+(window−1)/2; even windows have half-pixel centers. The two selected configs use the same first center for sensitivity comparison. Ampcor oversampling/zoom/deramp, backend, SNR and peak thresholds are configurable. `auto` probes CUDA and falls back to CPU; an explicitly requested unavailable GPU gives an informative error. GPU numerical checks are conditional on hardware.

The default glacier tracking settings are 256×64 windows, 40×10 search half-widths and 128×32 skips in **range × azimuth** order. YAML arrays use **[azimuth, range]**, so these settings are written as:

```yaml
matching:
  window: [64, 256]       # azimuth lines, range samples
  search: [10, 40]        # residual search half-widths: azimuth ±10, range ±40 pixels
  stride: [32, 128]       # skips between chip starts; 50% overlap in both axes
  oversampling: 32        # native correlation-surface oversampling
  raw_oversampling: 2
  zoom: 16
```

Oversampling controls subpixel refinement, not the size of the search or map cells. Raw-data oversampling and the refinement zoom remain separate parameters. The selected zero/constant configs start at center `[47.5, 175.5]` in [azimuth, range] order, which fits the larger windows and both search centers. The October 11 M4 closeout reran and validated these settings. Earlier 64×64-window results remain archived as a historical baseline.

## Products and scientific interpretation

Outputs are under `work/jan10_jan22/t038_080975_iw2/`:

- `manifest.json`, COMPASS YAMLs/logs, isolated orbit links, original reference complex SLC, resampled secondary complex SLC, reference grid, x/y/z geometry and layover/shadow.
- `radar/offsets.h5`: raw applied gross/residual/total for range and azimuth, exact center line/sample, input/quality masks, rejection flags, SNR, correlation peak and native covariance. `matching_grid.h5` carries sparse-grid associations. `radar/native/` preserves unmodified native binary outputs; float32 BIP offsets are azimuth then range, covariance has three components. No Earth affine is assigned to radar products.
- `geo/`: total range/azimuth displacement (m), velocity (m/day), validity and quality GeoTIFFs, formal uncertainty GeoTIFFs, and source-center geometry/azimuth scales in HDF5. `valid_mask.tif` marks available cells under the selected source mask; `quality_valid_mask.tif` separately marks quality-accepted cells. Both use 0/1. Missing displacement/velocity values are NaN.
- `diagnostics/`: amplitude/mask overview, window locations, raw gross/residual/total and filtered-total QC, source geometry and velocity figures, `uncertainty.png` with displacement/velocity formal 1σ maps, and a JSON coverage/rejection/limitations report. Empty results are explicitly labeled.
- `validation/m4-closeout/zero` and `validation/m4-closeout/constant`: retained products using the final settings. `validation/zero` and `validation/constant` preserve the earlier baseline.

Positive range is increasing slant range, away from the sensor. Positive azimuth is increasing reference sensing time along the ground track. These are **radar observables**, not east/north components. Azimuth scale uses local COMPASS DEM-intersecting zero-Doppler positions at fixed slant range, separated by one full-resolution line, and their WGS84 geodesic distance. No platform-speed constant is used. Velocity divides by the exact burst-start interval in seconds/86400, with UTC timestamps, not a rounded 12-day interval.

Geocoding uses the PLAN-approved geometry alternative: bilinear COMPASS geometry at exact SLC chip centers and an explicit projected-metre output grid. Geometry/units are checked independently with native ISCE3 geo2rdr. Configure resampling separately from quality selection:

```yaml
geocoding:
  epsg: 3031
  spacing_m: 200
  resampling: nearest  # nearest (default) | linear (earlier conservative method)
  mask: input         # input (default, retain observations) | quality
```

`nearest` maps output pixel centers back to fractional sparse radar row/column using the complete finite-center geometry mesh, rounds to the original radar sample, then applies that sample's mask. This preserves isolated good observations. A rejected sample keeps its own region as nodata; the method never substitutes a more distant good observation. Geometry interpolation is separate from displacement resampling. `linear` requires three adjacent selected triangle vertices and can remove isolated good samples. Both methods stop at the center mesh's boundary; neither extrapolates beyond it.

The default is to retain observations for later analysis: `matching.snr_threshold: 0.0`, `matching.correlation_threshold: 0.0`, and `geocoding.mask: input`. Input mode bypasses SNR/peak exclusions while retaining input-window, nonfinite, search-boundary and geometry exclusions. `snr.tif` and `correlation_peak.tif` use the same source sample as displacement, so thresholds can be applied later without repeating matching. Zero cutoffs are a retention policy, not an accuracy assessment.

Optional `quality` selects the existing Ampcor quality mask. `quality_valid_mask.tif` marks observations passing the thresholds recorded in the matching run; with zero cutoffs it primarily represents basic eligibility, not scientifically validated motion. Positive thresholds can still be configured as advisory flags while input mode preserves the wider footprint. Changing only geocoding options leaves radar observations and matching untouched. Metadata records the selected policies and the actual matching thresholds. Formal uncertainty is copied from the selected sample for nearest resampling; linear propagation uses squared weights assuming independent vertices. Bias and spatial correlations, including overlapping-chip correlations, remain excluded.

A matching window is a reference SLC chip yielding one observation at its center. A geocoded cell is one projected map pixel, here 200×200 m. Multiple cells can represent one radar observation; greater map coverage does not add independent measurements. Raw radar arrays include rejected matches, and the map rectangle includes space outside the burst. For a quality-masked radar view, use `--mask valid`; the default input geocoding retains more observations. In the earlier selected-pair comparison with SNR 5 and peak 0.2, linear/quality mapped 701 cells; nearest/quality mapped **1,234**, and nearest/input mapped **41,085**, of which **1,234** passed those thresholds. The same 229 threshold-passing radar windows were used in all comparisons. These historical counts predate the new zero-cutoff defaults. Details and the AutoRIFT Geogrid audit are in [geocoding strategies](docs/geocoding-strategies.md).

Changing only these geocoding settings recomputes maps without repeating COMPASS or Ampcor:

```bash
bin/glacier-offsets geocode configs/pair.yaml
bin/glacier-offsets diagnostics configs/pair.yaml
```

Input masks distinguish connected zero-fill from isolated quantized zero speckle observations: boundary zeros and 3×3 all-zero interiors seed connected-component propagation. Nonfinite samples are invalid. COMPASS complex nodata=0 derived GDAL masks are superseded by this complex-zero policy because they reject valid zero real components. Chips or complete secondary searches intersecting missing data are rejected. Search-boundary peaks are flagged; observations below configured SNR/peak cutoffs remain raw and are mapped in input mode, but fail the quality mask. There is no guarantee that every decorrelated or out-of-search false match is identified.

## Burst footprints in KML

Export burst footprints for one SAFE directory or ZIP:

```bash
conda activate isce3
python tools/export_bursts_kml.py /path/acquisition.zip --output bursts.kml
```

The KML has colored IW1/IW2/IW3 folders, one placemark per geographic burst ID, annotation-derived polygons, a center point for its label, and polarization/UTC/source metadata. It uses the actual annotation geolocation-grid borders, including s1-reader's split polygons at the antimeridian. These footprints show nominal burst coverage, not the SLC valid-data mask. No DEM, coregistration, or image extraction is required. Without an orbit, s1-reader derives geographic IDs from the annotation ascending-node time. Add `--orbit-dir data/orbits` to auto-select a precise orbit using the same resolver as the YAML initializer, or `--orbit /path/orbit.EOF` to specify one, for consistency with processing burst IDs. Polarization is inferred if unique; otherwise supply `--polarization HH` (or VV/HV/VH). `--subswaths 1 2` limits the export. The default output is `<SAFE stem>.bursts.kml` in the current directory; existing files require `--force` to replace. Output follows [KML 2.2 coordinate and polygon conventions](https://developers.google.com/kml/documentation/kmlreference).

## Standalone radar-offset plot

The plotting script reads an existing burst output directory (or its `radar` directory) without rerunning processing:

```bash
conda activate isce3
python tools/plot_radar_offsets.py work/jan10_jan22/t038_080975_iw2 --field total
python tools/plot_radar_offsets.py work/jan10_jan22/t038_080975_iw2/radar --field residual --mask valid
```

It saves three vertically stacked panels (3×1) for azimuth offset, range offset and SNR. Axes use full-resolution reference range samples and azimuth lines at matching-window centers, with increasing azimuth downward. Offsets are in SLC pixels. The default is unfiltered observations; `--mask input` excludes invalid inputs, and `--mask valid` selects quality-valid observations. Color limits use the 98th percentile; `--percentile 100` shows the full range. `--output figure.png` selects the output file, and `--show` also opens an interactive window. By default the PNG is saved beside `offsets.h5`.

## Validation status

M4 is closed with the conditional limitations documented in the [milestone/validation report](docs/validation.md). Following the geocoding strategy update, the suite ran 43 tests: 42 passed and one CUDA consistency test was skipped. Fresh zero/constant real runs, units/nodata/uncertainty audits, native geometry checks, restart checks and installed-wheel smoke passed at closeout. The subsequent [geocoding comparison](docs/geocoding-strategies.md) also verifies unchanged radar/coregistration products and native map-to-radar sample selection. This real 12-day pair has sparse quality-valid coverage. No independently identified stationary-ground mask or independent displacement truth has been supplied, so absolute accuracy and stationary-ground bias are not established. CUDA is absent on this macOS ARM CPU build. M5 has since been authorized, implemented and CPU-validated using the supplied MEaSUREs v2 Antarctic velocity mosaic; see [the M5 report](docs/m5.md).

```bash
PYTHONPATH=src:.runtime/deps OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
GLACIER_OFFSETS_PYTHON=/Users/briel/miniforge3/envs/isce3/bin/python \
  PYTHONPATH=src:.runtime/deps OMP_NUM_THREADS=4 python tools/validate_pair.py \
  --snapshot-name my-validation --report-directory docs/my-validation
PYTHONPATH=src python tools/geometry_check.py --config configs/pair.yaml --report docs/my-validation/geometry.json
```

The validation script forces fresh matching in zero and constant modes, retains snapshots, restores zero products, and checks unchanged restarts without recomputing COMPASS SLCs. Use a new snapshot name for each validation set; `--overwrite` explicitly refreshes that named set. Supply `--config` and `--constant-config` for a different pair; both configs must have identical inputs, output, matching grid/settings and geocoding settings. Real mode differences are sensitivity measurements; known-shift correctness is established by the synthetic tests.


## AOI-selected multi-burst processing (M6–M8)

Select common bursts for **one SAFE/ZIP pair** with a polygon vector file or
WGS84 bounds. Clear a burst ID inherited from a parent YAML explicitly:

```yaml
extends: pair.test.yaml
inputs:
  burst_id: null
aoi:
  path: ../docs/multi-burst-aoi/aoi.geojson
  # layer: glaciers  # Required for a dataset with multiple layers.
output:
  directory: ../work/multi-burst-aoi/zero
```

Alternatively use `aoi: {bbox: [-135, -77, -125, -74]}`. Bounds are west, south,
east, north in longitude/latitude degrees. `west > east` crosses the dateline.
AOIs require exactly one non-null path or bbox, and conflict with a non-null
`inputs.burst_id`. Paths belong to the YAML that declares them, including across
inheritance. To return to a single inherited burst, clear the AOI with `aoi: null`.

GeoJSON, Shapefile, GeoPackage and KML polygon layers are supported. Layers require a
horizontal CRS; standard GeoJSON and KML WGS84 conventions apply. Multiple features are
unioned, preserving holes and disconnected components. Invalid, empty,
nonpolygon and nonfinite geometry is rejected without repair. Geographic AOIs
and footprints use GDAL dateline wrapping; projected vector AOIs retain their
native polygon boundaries. Footprint edges are densified to at most 0.01 degrees
before projection with explicit XY axis order.

For a Google Earth KML polygon, use the same AOI configuration:

```yaml
aoi:
  path: ../data/glacier-aoi.kml
  # layer: glaciers  # Required when GDAL exposes multiple folders/layers.
```

KML altitude is ignored for this horizontal AOI. Polygon-only `MultiGeometry`
placemarks are supported; points and paths in the selected layer are rejected.
If a file has multiple layers, the error lists their names for `aoi.layer`.
The pair initializer also accepts `--aoi glacier-aoi.kml` and optional
`--aoi-layer glaciers`. KML reading uses the existing GDAL KML/LIBKML driver.

```bash
bin/glacier-offsets list-bursts configs/pair.aoi.example.yaml
bin/glacier-offsets validate configs/pair.aoi.example.yaml
bin/glacier-offsets run configs/pair.aoi.example.yaml --backend cpu
# Stage-only commands use the same per-burst prerequisites as single-burst runs:
bin/glacier-offsets match configs/pair.aoi.example.yaml
bin/glacier-offsets geocode configs/pair.aoi.example.yaml
bin/glacier-offsets diagnostics configs/pair.aoi.example.yaml
```

Discovery remains metadata-only and needs no DEM: selected IDs are ordered by
IW1/IW2/IW3, then geographic ID, followed by selected/common counts. Discovery
without either selector still lists all common bursts. Validation checks every
selected burst without processing imagery or creating reports. Scene order,
relative orbit, direction, precise orbit compatibility, and burst sampling are
validated. A selection requires **positive-area AOI ∩ reference footprint ∩
secondary footprint**. Touching boundaries and separate overlaps with different
AOI components do not qualify. Missing or invalid candidate footprints abort
selection and identify the affected IDs.

The initializer supports mutually exclusive selectors while preserving orbit
auto-detection and relative-path output:

```bash
python tools/init_pair_config.py REF.zip SEC.zip --dem DEM.tif \
  --orbit-dir ORBITS --aoi AOI.gpkg --aoi-layer glaciers \
  --relative-paths --output configs/pair.generated.yaml
# Or replace --aoi/--aoi-layer with --bbox WEST SOUTH EAST NORTH or --burst-id ID.
```

The AOI selects work; it does not clip imagery, restrict matching windows, or
guarantee full AOI coverage. By default, selected bursts are coregistered
independently, then their compatible reference/secondary SLCs are assembled on
one radar grid **before** native ISCE3 matching and geocoding. Overlapping valid
rectangles use ISCE2-style later-source precedence, with the same source ownership
for both acquisitions and geometry. Complex samples are not averaged. Per-center
geometry, exact pair intervals and velocity-prior projection come from the owning
source burst. Fractional placements exceeding `burst_merge.alignment_tolerance_px`
(default `0.001`) and incompatible sampling are rejected; this stage does not
resample incompatible TOPS grids.
The DEM extent must cover the **entire reference burst footprint**, including
areas outside the AOI. Internal DEM nodata is handled by geometry masks and
coverage diagnostics. Annotation borders do not establish valid SLC coverage or
independent offset accuracy. CPU/GPU execution is sequential across bursts.

Source coregistration stays beneath `<output.directory>/<burst_id>/`. The final
assembled `coreg/`, `radar/`, `geo/`, `diagnostics/`, `manifest.json` and
`assembly.json` are beneath `<output.directory>/merged/`, with one GeoTIFF per
offset/displacement/velocity field. Additional maps record source-burst index,
pair interval and chip/search seam crossings. Seam-crossing observations remain
flagged for inspection; assembly does not establish their accuracy. Set
`burst_merge.enabled: false` to retain independent per-burst matching/geocoding.
Existing per-burst matching and map products are preserved when enabling assembly.
Pair-level atomic
`selection.json` records the AOI, CRS, normalized geometry identity, vector-source
checksums (including Shapefile sidecars), both candidate footprints, overlap areas,
decisions and ordered selected IDs. Atomic `batch_manifest.json` records the
command, selection identity, per-burst status/errors, merged status and product
paths, matching/map coverage and aggregate outcome. Shared SAFE/orbit/DEM hashes and software
inspection are computed once per invocation, and selected native bursts are
reused for coregistration preparation and M5 projection.

`run` performs source coregistration, assembly, matching, geocoding and diagnostics.
`coregister` prepares the source bursts only; `assemble` builds the merged SLCs
from complete source products. `match`, `geocode` and `diagnostics` resume their
merged stages and check the prerequisites. For the completed PIG AOI:

```bash
conda activate isce3
bin/glacier-offsets match configs/pair.pig.aoi.yaml --backend cpu
bin/glacier-offsets geocode configs/pair.pig.aoi.yaml
bin/glacier-offsets diagnostics configs/pair.pig.aoi.yaml
```

Normal commands still read the configured raw inputs for provenance and native
metadata. The [assembled-workflow validation record](docs/merged-slc-aoi/README.md)
includes an explicit completed-product validator for environments where those
inputs cannot be read, along with the PIG boundary comparisons and memory use.

Individual burst failures are recorded and remaining selected bursts continue.
Assembly requires all currently selected source bursts to complete; a failed
source blocks the merged output and invalidates its completion identities.
Rerun the same command to retry incomplete work and reuse valid completed stages.
Changing the AOI updates selection reports without invalidating unchanged burst
products; newly selected bursts run, while deselected directories remain on disk
and leave the current summary. Changed assembly ownership or source checksums
rebuild the merged stages. `--force` retains the requested stage's semantics;
`run --force` also refreshes the selected source coregistration. Empty accepted matching or map coverage is
reported explicitly even when the stage completes successfully.

Exit codes are `0` for all selected bursts completed, `1` for individual burst
failures, `2` for shared preflight errors or no selected bursts, and `130` for
keyboard interruption. Processing writes an empty selection and batch report
before returning `2`; discovery and validation remain read-only. Completed
products survive interruption.

The reproducible example AOI targets two IW2 bursts in archived reference
metadata. Restore its configured SAFE/orbit/DEM files before running it. See
[validation record](docs/multi-burst-aoi/README.md) for executed checks and the
original implementation-time real-data gate. The comparison tool explicitly uses
`burst_merge.enabled: false`: it runs independent batch and equivalent standalone
workflows in separate directories, compares all
coordinates, masks, motion and center geometry exactly, and checks AOI restart
checksums and modification times. Only named SNR/correlation/covariance and formal
uncertainty fields permit bounded numerical differences (`rtol=1e-5`, `atol=1e-6`
in stored units); their measured differences are retained in `comparison.json`:

```bash
python tools/validate_multi_burst.py configs/pair.aoi.example.yaml \
  --output-directory work/multi-burst-validation/zero --backend cpu
```

To inspect already generated products without processing or requiring the original
SAFE/orbit/DEM files, append `--compare-only`. This reports product agreement and
explicitly marks AOI restart checks as untested. Add `--strict` to require exact
agreement for ancillary fields too. A failure still writes per-field differences;
it does not stop after the first array. Both modes require matching NaN/nodata
patterns, schemas, metadata identities and map georeferencing. The real zero-mode
case and the observed upstream coregistration differences are documented in the
[comparison investigation](docs/multi-burst-aoi/comparison-investigation/README.md).

Velocity matching now recycles a sequential native worker after four calls to
bound native CPU allocations. Sparse geometry and SLC-mask reads also use bounded
raster windows/cache. FFTW plans are carried between worker replacements. The v7
matching identity causes one matching/geocoding refresh while reusing valid
coregistration; no `--force` is needed. See the
[RAM investigation](docs/multi-burst-aoi/memory-investigation/README.md) for measured
RAM, reproducibility checks and the distinction from legacy v6 products.

Repeat with a velocity-mode AOI config using the same velocity source/conventions
as its standalone runs. GPU validation requires a supported CUDA host; successful
GPU execution alone does not establish CPU/GPU numerical agreement. Multiple
SAFE slices per acquisition, date-stack orchestration, burst parallelism,
incompatible-grid resampling, AOI clipping and restricted AOI matching windows
remain outside scope.
