# Sentinel-1 glacier offsets

Estimate glacier motion from Sentinel-1 TOPS SLC amplitude offset tracking.
Process one acquisition pair or reuse aligned acquisitions across a date stack,
selecting one geographic burst or complete bursts intersecting an area of
interest (AOI). COMPASS supplies geometry-only coregistration; native ISCE3
Ampcor performs matching, with an optional ISCE2 CPU matcher.

Stack processing produces radar offsets, geocoded displacements, quality metrics
and diagnostics. Packaging creates a self-contained displacement netCDF for
subsequent correction and analysis. Standalone pairs also produce velocity
GeoTIFFs. Range/azimuth motion follows radar directions; horizontal conversion
produces local east/north components.

**[Quick start](#quick-start)** · **[Detailed reference](README_DETAILED.md)**

## Installation

From the repository root, create and activate the CPU environment:

```bash
conda env create -f environment.yml
conda activate glacier-offsets
python -m pip install --no-deps --no-build-isolation -e .
export OMP_NUM_THREADS=4
```

If already installed, activate your existing processing environment instead.
The environment supplies Python 3.12, ISCE3, COMPASS, s1-reader and GDAL; native
processing dependencies are installed through Conda. GPU matching requires a
supported Linux/NVIDIA host and a CUDA-enabled ISCE3 build. See
[environment and execution details](README_DETAILED.md#environment-and-commands).

## Quick start

The examples below run from the repository root. Installed `glacier-offsets`
commands also work elsewhere; use the appropriate config path. The `tools/`
utilities shown here require the checkout. Antarctic examples use `HH`
polarization and `EPSG:3031`; choose values appropriate to your data.

You need SAFE directories or ZIPs, precise orbits, a DEM with WGS84 ellipsoidal
heights, and an AOI. The DEM must cover complete selected bursts, including areas
outside the AOI. Replace these paths:

```bash
SAFES=/path/to/safes
ORBITS=/path/to/orbits
DEM=/path/to/dem.tif
AOI=/path/to/glacier.kml
```

Need inputs first? Follow [burst downloading](README_DETAILED.md#download-bursts-for-an-aoi)
and [orbit/DEM preparation](README_DETAILED.md#prepare-orbits-and-a-dem). Finish
downloading before preparation. For partial collections, review
`download-bursts --allow-partial --dry-run` before accepting omissions.

### One pair

Choose the two actual SAFE paths, generate a config, and inspect it before running:

```bash
python tools/init_pair_config.py "$SAFES/earlier.SAFE" "$SAFES/later.SAFE" \
  --orbit-dir "$ORBITS" --dem "$DEM" --aoi "$AOI" \
  --polarization HH --epsg 3031 \
  --output configs/pair.generated.yaml --output-directory work/pair

glacier-offsets list-bursts configs/pair.generated.yaml
glacier-offsets validate configs/pair.generated.yaml --backend cpu
glacier-offsets run configs/pair.generated.yaml --backend cpu
```

AOI results are under `work/pair/merged/`, including `radar/`, `geo/` and
`diagnostics/`. Replace `--aoi` with `--burst-id ID` for one burst; its results
go directly under `work/pair/`.

### A date stack

Use one SAFE per UTC date with a consistent track, direction and polarization:

```bash
glacier-offsets stack init \
  --safe-dir "$SAFES" --orbit-dir "$ORBITS" --dem "$DEM" \
  --aoi "$AOI" --polarization HH --epsg 3031 \
  --output configs/stack.generated.yaml --output-directory work/stack

glacier-offsets stack plan configs/stack.generated.yaml
glacier-offsets stack validate configs/stack.generated.yaml --backend cpu
glacier-offsets stack run configs/stack.generated.yaml --backend cpu
```

If you used `prepare-inputs`, replace `--safe-dir`, `--orbit-dir` and `--dem`
with `--prepared-inputs /path/to/prepared/manifest.json` at initialization.

Review the plan before processing. Defaults pair each date with its next neighbor
when the exact separation is at most 12 days. The earliest eligible date supplies
the reference geometry; use `--reference-date YYYY-MM-DD` to choose another.
For partial collections, choose a reference covering all desired bursts. Keep
the reference pinned when extending a stack. See [stack configuration and stages](README_DETAILED.md#reference-geometry-stacks).

For merge-first matching and swath calibration, review the
[configuration example](configs/stack.merge-first.example.yaml) before processing.
It includes `matching.range_oversampling: 2`; matching changes can require
rematching existing pairs. The [reference](README_DETAILED.md#merge-first-stacks)
explains ownership, matching settings and the optional ISCE2 engine.

To inspect pair coverage visually:

```bash
glacier-offsets stack plan configs/stack.generated.yaml > plan.json
python tools/plot_stack_plan.py plan.json
```

### Resume and check progress

Rerun the same `run` command after an interruption to reuse completed stages.
Use `--force` when you intend to repeat processing. Inspect saved stack progress:

```bash
glacier-offsets stack status configs/stack.generated.yaml --backend cpu
```

By default, merged SLCs are released after their last matching use and rebuilt
from retained coreg products when needed. Set
`storage: {keep_merged_slcs: true}` to retain them. CPU stack matching uses up to
`OMP_NUM_THREADS` separate single-threaded burst workers within each pair;
acquisition preparation and date pairs run sequentially. More workers need more
RAM. See [storage and restart behavior](README_DETAILED.md#retained-products-and-storage)
and [troubleshooting](README_DETAILED.md#troubleshooting).

### Reuse a reference on another computer

Export a completed stack's reference without copying old acquisition/pair products:

```bash
glacier-offsets stack export-reference configs/stack.generated.yaml \
  --output /path/reference-bundle
```

In the new config, set `stack.reference_directory` to the transferred bundle and
`inputs.dem: null` to use its DEM. The reference may precede `stack.start_date`.
Keep compatible matching/map-grid settings and point the SAFE, orbit and output
paths to the new computer. See [portable references](README_DETAILED.md#transfer-a-prepared-reference-to-another-computer).

## Correct and package a processed stack

These steps are independently invoked. Use the corrections appropriate to your
dataset and retain the source products needed by each step. Swath correction
reads retained stack products; tidal correction and horizontal conversion read
the packaged netCDF.

### Optional swath calibration and correction

With swath correction disabled in the estimation YAML, estimate the relative
IW1/IW2 spacecraft coefficient from independent overlap matching:

```bash
glacier-offsets stack calibrate-swaths configs/stack.generated.yaml \
  --output /path/to/swath-calibration --workers 4
```

This requires compatible merge-first processing, ISCE3 CPU matching,
`deramp: 0`, `range_oversampling: 2`, zero gross offsets and nearest geocoding,
plus retained aligned bursts, geometry, SAFE inputs and orbits. Review
`REPORT.md`, then copy the generated `calibration-settings.yaml` settings into
your processing YAML. They supply the calibration JSON path, checksum and
`enabled: true`.

For completed experimental results, use `--stage export` with their results
directory to create the calibration without rematching. The
[estimation reference](README_DETAILED.md#estimate-swath-calibration-factors)
explains scope and validation. Apply the selected calibration:

```bash
glacier-offsets stack correct-swaths configs/stack.generated.yaml
```

Corrected products are written separately from raw pairs. See the
[configuration template](configs/stack.swath-corrected.example.yaml) for settings.

### Package displacements as netCDF

```bash
glacier-offsets package configs/stack.generated.yaml
```

The default creates `work/stack/packages/displacements_raw.nc`. To package
completed swath-corrected products, set `packaging.source: swath_corrected`;
the filename becomes `displacements_swath_corrected.nc`. The command packages
completed active pairs and records omissions. See the
[netCDF schema and options](README_DETAILED.md#package-displacement-stacks-as-netcdf).

### Correct vertical ocean tides

```bash
python -m pip install -e '.[tides]'
glacier-offsets correct-tides /path/displacements.nc \
  --model /path/CATS2008_v2023.nc --output /path/displacements_tide_corrected.nc
```

The command uses the netCDF and external CATS2008-v2023 model, including its
embedded grounding/flexure response. Range remains in slant-range metres;
azimuth tidal correction remains disabled. See
[tidal correction conventions](README_DETAILED.md#correct-vertical-ocean-tides).

### Convert to local east/north displacements

```bash
glacier-offsets to-horizontal /path/displacements_tide_corrected.nc \
  --vertical-motion zero --output /path/displacements_horizontal.nc
```

This uses the netCDF alone and preserves its grid and acquisition endpoints.
Use `horizontal_valid_mask` for the converted fields. East/north means local ENU;
the grid's EPSG code describes spatial coordinates. The output retains pair
displacements for subsequent time-series fitting. See
[conversion assumptions and geometry checks](README_DETAILED.md#convert-to-local-eastnorth-displacements).

## Outputs and interpretation

A stack writes:

```text
work/stack/
  geometry/                     # Pinned reference and common radar/map grids
  acquisitions/<date>/          # Aligned bursts and temporary merged SLCs
  pairs/<earlier>_<later>/       # Radar offsets, geo/ displacements, diagnostics
  pairs_swath_corrected/        # Optional separate corrected products
  stack_manifest.json           # Active plan, provenance and stage status
  packages/displacements_raw.nc # Created by the independent package command
```

Per-burst aligned SLCs, resampling offsets and geometry remain on disk, so storage
still grows with acquisition count. New stack `geo/` products contain
displacements; standalone pair outputs also include velocities. Quality metrics,
valid masks and formal uncertainties require assessment for the dataset and
scientific application. Multiple map cells can represent one radar observation.
See [products, signs and quality conventions](README_DETAILED.md#products-and-scientific-interpretation).

The [detailed reference](README_DETAILED.md#contents) contains configuration,
command options, troubleshooting and a separate validation/history section.
Start new configurations from the [pair](configs/pair.example.yaml) or
[stack](configs/stack.example.yaml) template.

## Tests

With the processing environment activated and the package installed:

```bash
OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
```
