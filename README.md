# Sentinel-1 glacier offsets

Estimate glacier displacement and velocity from Sentinel-1 TOPS SLC amplitude
offset tracking. Process a single acquisition pair or a stack of dates, selecting
one geographic burst or complete bursts intersecting a polygon area of interest.

The workflow uses COMPASS for geometry-only coregistration and native ISCE3
Ampcor for amplitude matching. An optional ISCE2-style stack strategy uses common-valid
overwrite merging with either ISCE3 Ampcor or the ISCE2 CPU magnitude kernel. It produces radar offsets, geocoded displacement
GeoTIFFs, quality metrics, and diagnostic plots. Standalone pairs also include
velocity GeoTIFFs. Range and azimuth
motion are reported in radar directions; they are not east/north components.

**[Getting started](GETTING_STARTED.md)** ·
**[Detailed reference](README_DETAILED.md)**

## Features

- Pair processing and reference-geometry stacks that reuse aligned acquisitions
  across multiple date pairs.
- AOI selection from KML, GeoJSON, Shapefile, GeoPackage or geographic bounds.
- Restartable ASF burst downloads, precise-orbit/DEM preparation, and optional
  selection of compatible partial acquisitions.
- Zero, constant or velocity-guided gross offsets, with gross, residual and total
  motion retained separately.
- CPU matching with burst workers for stacks, plus GPU matching with a supported
  CUDA build of ISCE3.
- Checkpointed processing, input checksums, stage status, and network plots for
  inspecting planned pair coverage and connectivity.
- Self-contained displacement netCDF stacks compatible with `iceutils.Stack`,
  with raw or swath-corrected observations, quality metrics, timing and LOS geometry.
- CATS2008-v2023 vertical ocean-tide correction directly from a netCDF stack,
  retaining the model's flexure response and writing a separate corrected stack.

## Installation

From the repository root, create the CPU environment and install the package:

```bash
conda env create -f environment.yml
conda activate glacier-offsets
python -m pip install --no-deps --no-build-isolation -e .
```

The environment supplies Python 3.12, ISCE3, COMPASS, s1-reader and GDAL. Native
processing dependencies must be installed through Conda. Optional download and
input-preparation dependencies are described in the
[detailed reference](README_DETAILED.md#local-inputs).

## Quick start

You need local Sentinel-1 SLC SAFE directories or ZIPs, precise orbit files,
a DEM with WGS84 ellipsoidal heights, and an AOI. The DEM must cover the complete
selected bursts. Run these examples from the repository root and replace the
input paths. `HH` and `EPSG:3031` below are example settings for Antarctic data.

### One pair

```bash
python tools/init_pair_config.py /path/earlier.SAFE /path/later.SAFE \
  --orbit-dir /path/orbits --dem /path/dem.tif --aoi /path/glacier.kml \
  --polarization HH --epsg 3031 \
  --output configs/pair.generated.yaml --output-directory work/pair

glacier-offsets list-bursts configs/pair.generated.yaml
glacier-offsets validate configs/pair.generated.yaml --backend cpu
glacier-offsets run configs/pair.generated.yaml --backend cpu
```

AOI pair results are written beneath `work/pair/merged/`, including `radar/`,
`geo/` and `diagnostics/`. Use `--burst-id ID` instead of `--aoi` to process a
single burst.

### A date stack

```bash
glacier-offsets stack init \
  --safe-dir /path/safes --orbit-dir /path/orbits --dem /path/dem.tif \
  --aoi /path/glacier.kml --polarization HH --epsg 3031 \
  --output configs/stack.generated.yaml --output-directory work/stack

glacier-offsets stack plan configs/stack.generated.yaml
OMP_NUM_THREADS=4 glacier-offsets stack run configs/stack.generated.yaml --backend cpu
glacier-offsets stack status configs/stack.generated.yaml --backend cpu
glacier-offsets package configs/stack.generated.yaml
```

The default plan pairs each date with its next neighbor when the exact separation
is at most 12 days. Each stack uses one SAFE per UTC date and a consistent track,
direction and polarization. The earliest eligible date supplies the reference
geometry; `--reference-date YYYY-MM-DD` selects another date. Completed stacks
keep their reference geometry pinned when extended.

A separate stack can reuse a reference on another computer without copying old
pair products:

```bash
glacier-offsets stack export-reference configs/stack.generated.yaml \
  --output /path/reference-bundle
```

Set `stack.reference_directory` to the transferred bundle in the new config;
its reference may precede `stack.start_date`. Keep the original matching grid and
map-grid settings. See [portable references](README_DETAILED.md#transfer-a-prepared-reference-to-another-computer)
for contents, disk requirements, and compatibility checks.

To visualize planned pairs and missing burst coverage:

```bash
glacier-offsets stack plan configs/stack.generated.yaml > plan.json
python tools/plot_stack_plan.py plan.json
```

### Estimate and apply swath calibration after processing

First estimate the factors from independently matched IW1/IW2 overlaps. Run from
the repository root in the ISCE3 environment, using your processed stack YAML
with swath correction disabled and a separate experiment output directory:

```bash
python -m tools.swath_correction_experiment run configs/stack.yaml \
  --output /path/to/swath-calibration-experiment --workers 4
```

The `tools` directory is not installed with the package. To run from another
directory, add the checkout root and its `src` directory to `PYTHONPATH` and use
an absolute config path; see the [outside-the-checkout example](README_DETAILED.md#estimate-swath-calibration-factors).

This requires retained aligned burst SLCs, reference geometry, SAFE inputs and
orbits. The current estimator supports ISCE3 CPU matching with `deramp: 0`,
`range_oversampling: 2`, zero gross offsets and nearest geocoding. Review
`REPORT.md` and `analysis/primary.json` in the experiment directory. The tool
estimates coefficients but does not export a production calibration JSON;
see the [estimation and publication steps](README_DETAILED.md#estimate-swath-calibration-factors).

Then apply the reviewed, published calibration to the completed stack:

```bash
glacier-offsets stack correct-swaths configs/stack.yaml
```

Use your stack YAML with `stack.swath_correction.enabled: true`, `calibration`
pointing to the applicable calibration JSON, and `calibration_sha256` set to its
SHA-256 checksum. See the [configuration template](configs/stack.swath-corrected.example.yaml)
and [swath correction details](README_DETAILED.md#optional-production-spacecraft-swath-correction).
This writes separate corrected pair products using the completed offsets. It
does not estimate calibration factors or rerun matching. To package them, set
`packaging.source: swath_corrected` and run:

```bash
glacier-offsets package configs/stack.yaml
```

## Outputs and storage

A stack writes:

```text
work/stack/
  geometry/                     # Shared reference products and fixed radar layout
  acquisitions/<date>/          # Aligned bursts, masks and temporary merged SLCs
  pairs/<earlier>_<later>/       # Radar offsets, geocoded maps and diagnostics
  stack_manifest.json           # Plan, provenance and processing status
  packages/displacements_raw.nc # Created independently by glacier-offsets package
```

Rerun the same `run` command after an interruption to reuse completed products.
By default, merged SLCs are removed after their last matching use. Per-burst
aligned SLCs, coregistration offset rasters and reference geometry remain on disk,
so storage still grows with acquisition count. See the
[storage policy](README_DETAILED.md#reference-geometry-stacks) before planning
a large stack.

New stack maps contain displacements without velocities. The independent
`package` command preserves the common grid and packages completed active pairs,
recording omissions. Set `packaging.source: swath_corrected` to use completed
corrected products. See the [netCDF schema and examples](README_DETAILED.md#package-displacement-stacks-as-netcdf).

To remove vertical ocean tides using the model's embedded grounding/flexure response:

```bash
python -m pip install -e '.[tides]'
glacier-offsets correct-tides /path/displacements.nc \
  --model /path/CATS2008_v2023.nc --output /path/displacements_tide_corrected.nc
```

The corrected primary range remains in slant-range metres. This command needs
the input netCDF and local tide model, and reads no other processing outputs.
See [tidal correction conventions](README_DETAILED.md#correct-vertical-ocean-tides).

To convert the supplied radar components to local east/north displacements:

```bash
glacier-offsets to-horizontal /path/displacements_tide_corrected.nc \
  --vertical-motion zero --output /path/displacements_horizontal.nc
```

This uses only the input netCDF, preserves the spatial grid and pair endpoints,
and writes displacements in metres. Azimuth tidal correction remains disabled.
Use `horizontal_valid_mask` for the converted fields. See
[horizontal conversion assumptions and geometry checks](README_DETAILED.md#convert-to-local-eastnorth-displacements).

Default ISCE3 stack CPU matching uses up to `OMP_NUM_THREADS` separate burst workers within
each date pair. Acquisition preparation and date pairs run sequentially; more
matching workers require more RAM. GPU matching requires a supported Linux/NVIDIA
environment and a CUDA-enabled ISCE3 build.

For merge-first stacks (stack-common validity, later-burst overwrites and the
ISCE3-backed compositor), see the
[configuration example](configs/stack.merge-first.example.yaml) and
[details](README_DETAILED.md#merge-first-stacks). The example also enables
`matching.range_oversampling: 2`, which interpolates the merged SLCs twice in
range before matching and removes the whole-pixel range bias of magnitude
matching ([details](README_DETAILED.md#range-2x-matching)). The optional ISCE2
matcher has its own [example](configs/stack.isce2.example.yaml).

## Documentation

- [Getting started](GETTING_STARTED.md): pair and stack walkthroughs.
- [Detailed reference](README_DETAILED.md): downloading, configuration, motion
  conventions, processing stages, outputs and validation notes. Some linked
  investigation records describe local datasets and are not included in a clone.
- [Stack processing notes](docs/stack-processing/README.md): ownership,
  restart behavior and validation records.
- [Pair configuration template](configs/pair.example.yaml) and
  [stack configuration template](configs/stack.example.yaml).

The workflow estimates pairwise motion. Time-series inversion and NESD are
outside its current scope. Diagnostic quality metrics and formal uncertainties
should be assessed for the dataset and scientific application.

## Tests

With the processing environment activated and the package installed:

```bash
OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
```
