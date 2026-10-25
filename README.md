# Sentinel-1 glacier offsets

Estimate glacier displacement and velocity from Sentinel-1 TOPS SLC amplitude
offset tracking. Process a single acquisition pair or a stack of dates, selecting
one geographic burst or complete bursts intersecting a polygon area of interest.

The workflow uses COMPASS for geometry-only coregistration and native ISCE3
Ampcor for amplitude matching. An optional ISCE2-style stack strategy uses common-valid
overwrite merging with either ISCE3 Ampcor or the ISCE2 CPU magnitude kernel. It produces radar offsets, geocoded displacement
and velocity GeoTIFFs, quality metrics, and diagnostic plots. Range and azimuth
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
```

The default plan pairs each date with its next neighbor when the exact separation
is at most 12 days. Each stack uses one SAFE per UTC date and a consistent track,
direction and polarization. The earliest eligible date supplies the reference
geometry; `--reference-date YYYY-MM-DD` selects another date. Completed stacks
keep their reference geometry pinned when extended.

To visualize planned pairs and missing burst coverage:

```bash
glacier-offsets stack plan configs/stack.generated.yaml > plan.json
python tools/plot_stack_plan.py plan.json
```

## Outputs and storage

A stack writes:

```text
work/stack/
  geometry/                     # Shared reference products and fixed radar layout
  acquisitions/<date>/          # Aligned bursts, masks and temporary merged SLCs
  pairs/<earlier>_<later>/       # Radar offsets, geocoded maps and diagnostics
  stack_manifest.json           # Plan, provenance and processing status
```

Rerun the same `run` command after an interruption to reuse completed products.
By default, merged SLCs are removed after their last matching use. Per-burst
aligned SLCs, coregistration offset rasters and reference geometry remain on disk,
so storage still grows with acquisition count. See the
[storage policy](README_DETAILED.md#reference-geometry-stacks) before planning
a large stack.

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
