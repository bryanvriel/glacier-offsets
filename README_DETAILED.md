# Sentinel-1 glacier offsets: detailed reference

[Quick start and workflow](README.md#quick-start) · [Project overview](README.md)

Use this reference for configuration, command options, scientific conventions,
schemas and troubleshooting. The [README](README.md) provides the installation
and the common processing sequence.

A small, restartable Python workflow for local Sentinel-1 TOPS SAFE pairs and acquisition stacks, one polarization and either one geographic burst or complete bursts intersecting an AOI. COMPASS performs geometry-only radar coregistration; native ISCE3 Ampcor tracks **amplitudes of complex SLCs**. Offset products retain applied gross, native residual and total separately. Only total motion is converted to displacement/velocity and geocoded. Local georeferenced velocity fields can guide searches. Pair AOI processing assembles compatible SLCs on one radar grid before tracking beneath `merged/`. Stack processing prepares each acquisition once on a pinned reference frame and reuses its SLC across pairs. Optional input preparation downloads precise orbits and a cropped DEM; processing stages use those local files. There is no NESD or custom correlator.

## Contents

- [Environment and commands](#environment-and-commands)
- [Local inputs](#local-inputs)
  - [Download bursts for an AOI](#download-bursts-for-an-aoi)
  - [Prepare orbits and a DEM](#prepare-orbits-and-a-dem)
- [Configuration and pair processing](#configuration-and-pair-processing)
  - [Shared defaults and pair overrides](#shared-defaults-and-pair-overrides)
  - [Initialize a pair YAML](#initialize-a-pair-yaml)
  - [AOI-selected multi-burst processing](#aoi-selected-multi-burst-processing)
- [Stack processing](#stack-processing)
  - [Reference-geometry stacks](#reference-geometry-stacks)
    - [Pair graph and processing stages](#pair-graph-and-processing-stages)
    - [Matching workers and FFTW plans](#matching-workers-and-fftw-plans)
    - [Retained products and storage](#retained-products-and-storage)
    - [Restart, extension and immutable references](#restart-extension-and-immutable-references)
    - [Rebuild legacy ownership layouts](#rebuild-legacy-ownership-layouts)
    - [Transfer a prepared reference to another computer](#transfer-a-prepared-reference-to-another-computer)
    - [Visualize a stack plan](#visualize-a-stack-plan)
  - [Merge-first stacks](#merge-first-stacks)
    - [Swath ownership and layout rebuilds](#swath-ownership-and-layout-rebuilds)
    - [Common validity and compositor contract](#common-validity-and-compositor-contract)
    - [Range-2x matching](#range-2x-matching)
    - [Optional ISCE2 matcher](#optional-isce2-matcher)
- [Swath calibration and correction](#swath-calibration-and-correction)
  - [Optional production spacecraft swath correction](#optional-production-spacecraft-swath-correction)
    - [Estimate swath calibration factors](#estimate-swath-calibration-factors)
    - [Apply a published swath calibration](#apply-a-published-swath-calibration)
- [Displacement stack post-processing](#displacement-stack-post-processing)
  - [Package displacement stacks as netCDF](#package-displacement-stacks-as-netcdf)
  - [Correct vertical ocean tides](#correct-vertical-ocean-tides)
  - [Convert to local east/north displacements](#convert-to-local-eastnorth-displacements)
- [Products and scientific interpretation](#products-and-scientific-interpretation)
  - [Burst footprints in KML](#burst-footprints-in-kml)
  - [Standalone radar-offset plot](#standalone-radar-offset-plot)
- [Troubleshooting](#troubleshooting)
- [Validation and investigation history](#validation-and-investigation-history)
  - [Validation status](#validation-status)
  - [Matcher comparison and range-oversampling experiments](#matcher-comparison-and-range-oversampling-experiments)
  - [Compositor parity checks](#compositor-parity-checks)
  - [ISCE2 integration tests](#isce2-integration-tests)
  - [Isolated half-year swath experiment](#isolated-half-year-swath-experiment)
  - [Audit ENU coefficient stability](#audit-enu-coefficient-stability)
    - [Larsen coefficient and scale results (2026-10-27)](#larsen-coefficient-and-scale-results-2026-10-27)
  - [Horizontal conversion validation (2026-10-27)](#horizontal-conversion-validation-2026-10-27)
  - [Ownership-layout validation](#ownership-layout-validation)
  - [AOI-selected pair validation](#aoi-selected-pair-validation)
  - [Repository and local validation records](#repository-and-local-validation-records)

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

The inspected native **PyCPUAmpcor 0.25.21 does not parallelize its chunk loop** and exposes no CPU thread-count setting. Its Ampcor FFTW code does not initialize threaded FFT plans. Setting `OMP_NUM_THREADS` therefore does not make an individual CPU matcher use four cores. `matching.batch` changes the number of windows processed together and memory use, not worker count; CPU `nStreams` is also not a CPU thread pool. OpenMP/FFTW capabilities elsewhere in the linked ISCE3 library do not imply parallel Ampcor. CUDA Ampcor is a separate backend. See [configuration and threading audit](docs/configuration/README.md) for the installed-library/source checks and the executed synthetic check. The stack workflow uses `OMP_NUM_THREADS` to run separate CPU burst-matching processes as described below; standalone pair/AOI matching retains its sequential native worker.

`--force` repeats the requested computation (`run --force` repeats all stages). Run identity includes inputs' SHA256 hashes, software versions, polarization, selected burst, matching parameters and gross mode/values. A change in matching/gross configuration invalidates matching and geocoding, while retaining geometric coregistration. Later stage commands reject stale preceding stages. Input hashes are checked even on restart; this takes time for large SAFE files. A failed stage clears its completion identity. A source SHA256 snapshot is recorded at validation; after editing workflow code, explicitly force the affected stages. Successful unchanged `run` reuses coregistration, matching and geocoding and refreshes diagnostics.

Within a stack invocation, reference, alignment and assembly verification
reuse SHA-256 digests for unchanged product files. Each lookup still checks file identity, size and
modification/change timestamps; changed or replaced files are hashed again, and
files changing during hashing are rejected. The cache is discarded between
invocations, so restart still verifies existing products and raw inputs afresh.
Verification logs identify the current acquisition, and the catalog's `metrics`
record product files/bytes hashed and checksum-cache hits. This optimization
does not invalidate existing products or require `--force`.

Stack acquisition restart checks use four read-only threads by default. They
hash different dates' raw inputs and saved products concurrently; native
validation, alignment, assembly and catalog updates remain sequential. Set the
limit independently of matching's `OMP_NUM_THREADS` in any stack YAML:

```yaml
stack:
  verification_workers: 4  # Positive integer; 1 restores sequential checking.
```

Checks are prefetched at most this many dates ahead. Shared files are hashed
once, and product revisions are checked again before reuse. This changes no
scientific identities or products and needs no `--force`. Concurrent reads may
help an SSD, but the speedup is limited by drive bandwidth; try 2 or 1 if the
drive slows down under concurrent reads. An already-running process keeps its
existing behavior until the next invocation.

## Local inputs

### Download bursts for an AOI

Install the optional downloader in the processing environment:

```bash
python -m pip install -e '.[burst-download]'
```

The download command uses ASF search and [burst2safe](https://github.com/ASFHyP3/burst2safe),
with `all_anns=True` for s1-reader compatibility. Supply existing Earthdata
credentials in `.netrc`, or set `EARTHDATA_TOKEN`. The command never creates a
credential file. Search and `--dry-run` need no credentials.

Download, prepare local inputs, then initialize the existing stack workflow:

```bash
glacier-offsets download-bursts --aoi data/larsen/LAR2_small.kml \
  --start-date 2018-01-08 --end-date 2018-01-20 \
  --relative-orbit 38 --polarization HH --subswaths 2 \
  -o data/larsen/burst-downloads --dry-run
  
# Remove --dry-run to download the planned acquisition groups.
glacier-offsets download-bursts --aoi data/larsen/LAR2_small.kml \
  --start-date 2018-01-08 --end-date 2018-01-20 \
  --relative-orbit 38 --polarization HH --subswaths 2 \
  -o data/larsen/burst-downloads
  
glacier-offsets prepare-inputs data/larsen/burst-downloads \
  -o data/larsen/prepared-bursts \
  --dem-source /Volumes/TU200Pro/glaciers/antarctica/rema/Gapless-REMA100.tif
  
glacier-offsets stack init \
  --prepared-inputs data/larsen/prepared-bursts/manifest.json \
  --aoi data/larsen/LAR2_small.kml --polarization HH --epsg 3031 \
  --neighbors 2 --max-separation-days 12 \
  --output configs/stack.lar.bursts.yaml \
  --output-directory work/larsen/stack-bursts --relative-paths
  
glacier-offsets stack validate configs/stack.lar.bursts.yaml

glacier-offsets stack run configs/stack.lar.bursts.yaml --backend cpu
```

AOI files use the same polygon validation, CRS and antimeridian handling as
processing. `--aoi-layer` selects a vector layer; `--bbox W S E N` replaces
`--aoi`. Start/end dates include the entire specified UTC days. Omit
`--subswaths` to consider IW1–IW3. The default output summarizes acquisition and
burst-product counts, unique geographic bursts, date coverage, spacecraft, swaths,
added context bursts, and up to five acquisitions with issues. Burst-product
counts include each date's product; geographic counts identify distinct burst
locations across dates. This summary stays compact for long date ranges.
Add `--json` for the full plan with every acquisition, burst and footprint;
`--dry-run --json > plan.json` saves it explicitly. Dry runs create no files
unless you redirect the output yourself.
Positive-area intersection preserves AOI holes. Intervening bursts are included
within each swath even when they fall outside the AOI. By default, missing interior bursts,
ambiguous duplicate products, incompatible swath combinations, and multiple
groups on a UTC date are explicit errors.

Add `--allow-partial` to accept a valid contiguous subset when an acquisition
has interior gaps or incompatible swath extents. Review it first with
`--allow-partial --dry-run`. The planner retains AOI imagery in **every observed
swath**, prioritizes the number of retained IW1 AOI bursts, then maximizes total
AOI bursts, then total retained products including context. Ties favor the lowest
start/end burst IDs in swath order. Adjacent included swaths must satisfy
burst2safe's compatibility checks. If no such subset exists, or products are
ambiguous or metadata inconsistent, the acquisition still fails. Imagery is never
fabricated to fill a gap. Only available bursts within the original AOI-selected
along-track span are considered; this mode does not extend swath endpoints.

The summary distinguishes full, accepted partial, and failed acquisitions, and
shows omission counts and up to five partial dates with their retained ranges.
`--json` and the manifest retain each unavailable burst ID, every available-but-
discarded granule/parent/footprint, and the policy/version used. Trimming one
swath can require discarding valid bursts in another swath. Accepted partial
acquisitions complete normally and pass through `prepare-inputs`; genuine failures
continue to block it. Preparation preserves the omission provenance. The stack's
existing masks leave missing coverage as nodata for affected pairs, preserving
unaffected dates/bursts and the common map grid. A location missing on consecutive
dates may still have an 18-day or longer gap despite retaining those dates elsewhere.
Choose a full-coverage reference date with `stack init --reference-date` so that
the pinned reference burst selection includes all desired locations.

You may enable this policy on an existing strict collection with the same AOI,
track, polarization, swaths and date range (or an extension). Unchanged completed
acquisitions are verified and reused; only groups whose retained granules change
are rebuilt. Keep `--allow-partial` on subsequent downloader invocations for that
collection. Switching it back to strict requires a separate collection. Rerun
preparation after the collection changes before initializing a stack.

Collections contain `download_manifest.json`, visible completed SAFEs beneath
`safe/<spacecraft>_<absolute-orbit>/`, and hidden `.staging/` work. Downloads
validate XML, complex TIFF dimensions/readability and required helper annotations
before publishing a complete acquisition directory. The manifest records
granules/parents, coverage, context, checksums, package versions and group status.
Rerunning the same request verifies checksums and reuses unchanged files;
incomplete/corrupt groups are retried and independent groups continue after a
failure. Resume checkpoints preserve completed acquisition records even when
interrupted again before reaching those acquisitions or during checksum
verification; each resumed reuse still checks file integrity. Concurrent writers
to one collection are rejected. Keep AOI, track,
polarization and swaths fixed; date ranges may expand, but narrowing dates or
changing selection requires a new collection.

`prepare-inputs` accepts either the collection root or its `safe/` directory and
refuses running, failed or inconsistent collections. It carries download
provenance into its manifest. Empty helper annotations remain available for
s1-reader corrections but contribute no imagery bursts or DEM footprint.
Full-SAFE inputs and downstream commands retain their existing behavior. Use
separate preparation/configuration/stack output roots when comparing reconstructed
and full SAFEs, since stack reference geometry is pinned. Exit codes are 0 for
success, 1 for acquisition failures, 2 for invalid preflight/empty dry runs, and
130 for interruption.

Reconstructed SAFEs can contain overlapping geolocation records at internal
burst joins. During inventory, KML export and COMPASS processing, these records
are interpreted in memory: for a repeated line/pixel, use the record nearest
the following burst's annotated start. Normalization requires an assembled
product, a regular burst-boundary grid, timestamps within 1.1 azimuth samples,
and coordinates within two ground azimuth pixels. Conflicting records and
invalid resulting footprints still fail validation. Downloaded XML and imagery
remain unchanged; existing collections do not need downloading or preparation
again for this correction. Regenerate saved stack plans to update coverage.

### Prepare orbits and a DEM

Install optional preparation dependencies in the existing ISCE3 environment:

```bash
python -m pip install -e '.[preparation]'
```

Given a directory of SAFE ZIPs or unpacked SAFEs:

```bash
glacier-offsets prepare-inputs data/pig/safe -o data/pig/prepared --dem-source REMA
glacier-offsets stack init --prepared-inputs data/pig/prepared/manifest.json \
  --polarization HH --aoi data/pig/pig_isce3_poly.kml --epsg 3031 \
  --output configs/stack.generated.yaml --output-directory work/pig/stack-prepared \
  --relative-paths
glacier-offsets stack validate configs/stack.generated.yaml
glacier-offsets stack run configs/stack.generated.yaml --backend cpu
```

`tools/s1_prepare.py` remains an equivalent source-tree entry point. Preparation
recursively scans visible SAFE containers, ignoring dot-prefixed files and
directories. An unpacked SAFE takes precedence over its same-named ZIP in the
same directory; distinct products on the same date still trigger the stack's
duplicate-date checks. Use `--no-recursive` for a shallow preparation scan.

Precise orbits are downloaded through `sentineleof` using its ASF backend by
default; `--use-cdse` selects CDSE and its authentication requirements. Scene
mission/times come from `manifest.safe`. Cached and downloaded orbits must pass
the same XML, state-vector, interpolation-margin and latest-creation selection
rules as processing. RESORB fallback is rejected. Downloads are staged and
validated before cached precise files are replaced. `--refresh-orbits` queries
the provider even if the cache is valid.

DEM defaults are one shared GeoTIFF, a 5 km footprint buffer, and automatic
source selection: REMA for buffered bounds entirely south of 55°S, otherwise
COP. REMA defaults to 32 m and native EPSG:3031; `--rema-resolution 10` selects
10 m. COP uses `sardem` with geoid-to-ellipsoid conversion enabled; the NISAR
option uses its pre-converted ellipsoidal DEM and requires Earthdata access.
REMA uses unsigned public S3 access without AWS credentials. The unsigned-access
and retry settings apply to GDAL's worker threads and are restored after the warp.
Sources supply WGS84 ellipsoidal heights; this datum is recorded as provenance,
not inferred from elevation values. `--dem-format ENVI` also works through GDAL;
the optional `--isce2-xml` flag additionally requires ISCE2 and sardem.

`--dem-source` also accepts a local raster path, including the published
Gapless-REMA100 mosaic:

```bash
glacier-offsets prepare-inputs data/larsen/safe -o data/larsen/prepared \
  --dem-source /Volumes/TU200Pro/glaciers/antarctica/rema/Gapless-REMA100.tif
```

Local rasters are cropped to the buffered SAFE footprints or `--bbox` without
downloading a DEM. GeoTIFF output retains the source CRS, pixel spacing and grid
alignment; this mosaic therefore stays at 100 m in EPSG:3031. `--rema-resolution`
applies only to the remote REMA provider. ENVI output is reprojected to EPSG:4326.
The local raster must be single-band, north-up, unrotated, have an identifiable
EPSG CRS, and already contain unpacked WGS84 ellipsoidal elevations in metres.
No vertical-datum conversion or gap filling is performed. The Gapless-REMA100
README explicitly specifies ellipsoidal metre heights. Its nodata value is
honored and normalized to -9999 in the prepared crop.

Preparation hashes the local source and its GDAL-listed dependencies once per
invocation and records their paths and SHA256 values alongside the cropped DEM.
Changing source contents changes the cache key. A large mosaic can take some
time to fingerprint, but its pixels are only loaded for cropping/validation,
not as one in-memory array. Local DEM preparation needs GDAL, without the
optional DEM-downloader packages. Existing orbit downloads still use sentineleof.

Use `--min-dem-valid-fraction 1` when the requested crop must be completely free
of nodata; the default fraction below is unchanged. Existing stacks pin their
DEM: initialize a new configuration/output root using the updated preparation
manifest when switching DEMs, then regenerate geometry and aligned acquisitions.

New and cached DEMs are opened with GDAL and checked for CRS, geotransform,
buffered extent, real-valued metre elevations and valid pixels. The default
`--min-dem-valid-fraction 0.01` rejects rasters with less than 1% finite,
unmasked elevation coverage, including entirely nodata results. This is a
whole-output raster fraction, not a promise that every glacier pixel is valid.
Nodata is retained without filling. `--max-dem-pixels` defaults to 200 million;
REMA estimates this before downloading, and every output is checked afterward.
GeoTIFFs are validated in temporary files before atomic replacement. Validation
statistics are saved beside each DEM in `<filename>.preparation.json`.

Use `--bbox W S E N` to override DEM bounds. This can be smaller than a SAFE
scene, but must still cover complete bursts selected for processing. When
consuming a prepared manifest, `stack init` checks selected reference-burst
extent coverage before writing its YAML. `--dem-mode per-safe`, `--skip-dem`
and single-SAFE preparation need explicit path overrides at initialization
where no shared DEM or SAFE directory is recorded.

The preparation output contains `orbits/`, `dem/`, per-scene input reports and
`manifest.json`. A failed scene, invalid ZIP or DEM failure gives a nonzero exit
code and a failure record; `stack init --prepared-inputs` refuses failed
manifests. Re-run preparation to retry while reusing validated products. AUX
downloads default to `--aux none`; optional CAL/INS downloads are retained for
external workflows and are not applied by this processor. Preparation never
downloads SAFE imagery, runs COMPASS, or changes an existing stack's pin.

Interrupted preparation leaves a `running` manifest, which initialization also
rejects until preparation finishes successfully.

## Configuration and pair processing

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

`--max-separation-days 12` optionally rejects a pair whose exact manifest start-time gap exceeds 12 days; a gap of exactly 12 days is accepted. This is an initialization check; processing still uses the selected burst's exact interval. The scene reader, orbit catalog and pair checks in `initialization.py` are shared with stack planning and input preparation. This initializer generates one pair config and supports AOI selection; use `glacier-offsets stack init` for acquisition stacks. Default output directories include both full SAFE product names to distinguish acquisition pairs before the existing per-burst subdirectory is added.

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

<a id="aoi-selected-multi-burst-processing-m6m8"></a>

### AOI-selected multi-burst processing

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

## Stack processing

### Reference-geometry stacks

If you have only SAFE scenes, first use the [input preparation command](#prepare-orbits-and-a-dem).

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

#### Pair graph and processing stages

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
regions within a swath switch at their azimuth midpoint, at least eight SLC lines
inside both valid edges; overlaps narrower than 16 lines are rejected. Between
swaths, the switch lies at the range midpoint of their shared support, at least
eight samples inside both valid edges; overlaps narrower than 16 samples are
rejected. Along-track cuts are resolved first, then swath support is partitioned
in range, preserving coverage at staggered burst junctions. Unshared coverage is
preserved. Cuts and their policy are recorded in the geometry and pair manifests.
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

#### Matching workers and FFTW plans

Stack stages, acquisition preparation and date pairs execute sequentially. Within
each pair, CPU matching runs up to `OMP_NUM_THREADS` spawned workers, capped by
the number of bursts with shared valid support. With the variable unset, it uses
one worker. An OpenMP list such as `4,1` uses its first value as the worker budget.
Each matching child is limited to one OpenMP/BLAS thread, while other stages
retain the original environment and native threading. For example:

```bash
OMP_NUM_THREADS=4 glacier-offsets stack run configs/stack.generated.yaml --backend cpu
```

Matching windows are grouped by their original burst ownership on the assembled
radar grid; neither the matching template nor the source SLCs are cropped or
shifted. Full chip/search support, exact burst intervals and seam flags retain
their existing definitions. At most one native call per burst is in flight, and
workers recycle after four native calls to bound unreleased Ampcor allocations.
For full-grid zero/constant matching, raw results in unowned or unavailable
regions are retained. These regions use a shared queue instead of one burst's
lane, with tasks bounded to at most half the average worker window count. Idle
workers can process them concurrently while each real burst still has at most
one native call in flight. Scientific source ownership and masks are unchanged.
The parent combines results and publishes the pair product only after all work
succeeds. Worker limits, PIDs, thread limits, native timings and wall time are
recorded in the matching metadata. Extra workers increase RAM and concurrent SLC
reads; the limit is a CPU budget, not an automatic memory or storage tuner.
CPU stack matching pins FFTW's measured plans. ISCE3's CPU Ampcor compiles in
`FFTW_MEASURE`, so workers that start together otherwise time FFT algorithms
independently and can choose plans that round differently. That moves
near-tied, low-SNR correlation peaks by one oversampled sample (1/64 px with
raw oversampling 2 and oversampling 32). The first match in a stack runs one
quiet spawned process that plans every native chunk shape on synthetic rasters
(about 5 s) and writes `<stack>/.cache/fftw/<key>.wisdom`, with the key
inputs in a matching `.json`. The key covers the FFTW version and library
checksum, the ISCE3 version, the machine and CPU model, and `window`,
`search`, `batch`, `raw_oversampling`, `oversampling`, `zoom` and `deramp`.
Workers import that file before planning and never rewrite it; a missing file
is an error. Repeated matching of the same pair on the same host and build is
then bit-identical at any worker count. Segment metadata records
`fftw_wisdom_sha256` and `fftw_wisdom_extended` (true if a call still had to
measure a plan), and `native_parameters.fftw_wisdom` records the file. The
wisdom enters no cache identity, so completed pairs are not rematched; pairs
matched before pinning can differ from new ones by 1/64 px in about 0.1% of
valid windows. Wisdom is not portable across hosts or builds: compare such
runs with a tolerance of 1/(raw_oversampling × oversampling) px in valid
windows. Deleting `.cache/fftw` remeasures it on the next match. Standalone
pair/AOI matching and tools that call the matcher directly keep the temporary
per-invocation plan cache. GPU (cuFFT) matching is unaffected.

Shared tasks record `work_kind: unowned_or_inactive` and a null
`scheduling_burst_id`; each segment includes its native elapsed time. Scheduling
strategy v2 preserves completed matching products. Restart a running process to
use the new scheduler for remaining pairs; no `--force` is needed.

GPU matching remains sequential on the selected GPU. `auto` uses the CPU worker
policy when it selects CPU; a runtime GPU-to-CPU fallback remains sequential for
that pair. Changing only `OMP_NUM_THREADS` reuses completed matching products.
The v8 stack matching identity refreshes older matching/downstream products once,
while retaining prepared geometry and acquisition SLCs; no new output root or
`--force` is needed.

#### Retained products and storage

The default storage policy retains per-burst **coreg products** (aligned SLCs,
reference geometry/topography and COMPASS metadata), offsets, maps, masks and
provenance. Merged SLCs are disposable: AOI pair processing releases its merged
reference/secondary TIFFs after offsets commit; stacks assemble merged dates on
demand during `run` and release each date's TIFF and external mask after its last
active matching consumer completes. Failed matching consumers retain their
required SLCs. The planned pair graph determines first and last use: each merged
date is assembled once and kept through all its planned matching consumers.
Geometry and per-burst sources remain available for geocoding,
reassembly, new pairs and stack extension without repeating COMPASS.

Cleanup is journaled in the manifests so `stack status`, restart and downstream
commands recognize intentional release. Small amplitude previews preserve
diagnostic plots after cleanup. Matching changes or new pairs automatically
rebuild needed merged SLCs; completed offset fields remain reusable. Explicit
`assemble` commands materialize SLCs and keep them until a later matching command.
To retain merged SLCs throughout processing, add this override to your YAML:

```yaml
storage:
  keep_merged_slcs: true
```

The policy applies on the next processing invocation, including existing output
roots. Raw SAFE files, input DEM/orbits and per-burst `coreg` products are never
removed by merged-SLC cleanup. Single-burst pair processing retains its coreg SLCs.

Products are organized as follows:

```text
<output>/stack_manifest.json
<output>/geometry/                  # burst topology, fixed layout/ownership, map grid
<output>/acquisitions/<YYYY-MM-DD>/  # aligned bursts, slc.tif, slc.tif.msk, valid.tif, manifest.json
<output>/pairs/<earlier>_<later>/    # manifest.json, radar/, geo/, diagnostics/
```

#### Restart, extension and immutable references

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

#### Rebuild legacy ownership layouts

Stacks using `reference-overlap-midpoint-v1` can update ownership **in the same
output directory**, retaining their verified COMPASS reference and aligned burst
products. Stop any running stack process first, then use the same configuration:

```bash
glacier-offsets stack rebuild-layout configs/stack.generated.yaml --dry-run
glacier-offsets stack rebuild-layout configs/stack.generated.yaml
OMP_NUM_THREADS=4 glacier-offsets stack run configs/stack.generated.yaml --backend cpu
```

The dry run verifies retained-product checksums and reports reuse counts without
writing files; verification can take time on a large stack. The rebuild permits
only this ownership-policy update with unchanged reference inputs, selection and
software. It replaces the ownership raster and geometry mosaic, removes date
assemblies/previews and all registered pair directories (matching, geocoding and
diagnostics), and updates cache identities for every saved date, including dates
outside the current filters. The next `run` regenerates merged SLCs on demand and
recomputes pair outputs. Completed COMPASS products are reused; missing or
unfinished bursts are prepared normally. Use `run` without `--force`.

Metadata backups and a resumable journal are stored under `.layout-rebuild/`.
After an interrupted rebuild, rerun the same rebuild command; normal processing
is blocked until it finishes. Writer locking prevents concurrent processing or
rebuilds in the same root. Stop processes launched with older software too,
because those processes do not acquire the lock. Other reference, software or
support-policy changes still require a new output root.

Multi-slice assembly, NESD, time-series inversion, concurrent date
preparation/pairs and fractional alignment of reference bursts remain outside V1.

#### Transfer a prepared reference to another computer

Export the reference from a completed stack before transferring it:

```bash
glacier-offsets stack export-reference configs/stack.larsen.production.yaml \
  --output /data/references/larsen-20170101
```

The export takes the existing stack lock, verifies its recorded products and
reference inputs, and publishes a new directory atomically. The output directory
must not exist and must be outside the source stack. An unfinished layout rebuild
or an active stack writer prevents export. No acquisition or pair processing runs.

Copy that directory and the new year's SAFE/orbit inputs to the other computer.
Create a config using the original processing, polarization, burst/AOI selection,
matching grid (`window`, `search`, `stride`, `first_center`), and geocoding EPSG and
spacing, with these stack/output overrides:

```yaml
stack:
  reference_directory: /data/references/larsen-20170101
  safe_directory: /data/larsen/2018/safes
  orbit_directory: /data/larsen/2018/orbits
  start_date: '2018-01-01'
  end_date: '2018-12-31'
  processing: isce3_merge_first
  swath_merge: midpoint
output:
  directory: /data/stacks/larsen-2018
```

A standalone Larsen example is [`configs/stack.larsen.2018-reference.example.yaml`](configs/stack.larsen.2018-reference.example.yaml).

Paths obey the usual YAML-relative inheritance rules. The bundle supplies the
reference date, DEM, and AOI; `inputs.dem` and `aoi` can be omitted. An inherited
original DEM/AOI path is also replaced with its bundled counterpart. Explicitly
incompatible inputs or settings are rejected. A specified `reference_date` must
agree with the bundle. Do not carry an absolute `extends` path from another
computer without updating it.

```bash
glacier-offsets stack plan configs/larsen-2018.yaml
glacier-offsets stack validate configs/larsen-2018.yaml
OMP_NUM_THREADS=4 glacier-offsets stack run configs/larsen-2018.yaml --backend cpu
glacier-offsets package configs/larsen-2018.yaml
```

Planning and validation inspect metadata; processing verifies the bundled file
checksums before preparing acquisitions. For a separate full integrity check:

```python
from glacier_offsets.reference_bundle import read_bundle
read_bundle('/data/references/larsen-20170101', verify=True)
```

The 2017 reference is loaded independently of the date filters. Only selected
2018 acquisitions enter the pair graph; there is no long-baseline displacement
pair to the reference. If the reference is inside the date filters, it can enter
the graph normally.

The versioned `reference.json` manifest authenticates each bundled file and records
a path-independent bundle fingerprint. Contents include prepared reference SLCs,
per-burst topology and radar grids, fixed support/ownership, the common map grid,
exact burst timing, the single reference SAFE, its precise orbit, the required DEM,
and any AOI sidecars. VRT dependencies and symlink targets are copied into the
bundle, with relative VRT references. No other year's acquisition directories or
pair products are included. The bundle retains original reference identities and
software/input provenance; machine paths and hardware need not match.

On the first processing invocation, the destination copies the verified bundle
into `<output.directory>/.reference/`. Reserve disk space for this copy in addition
to processing products. The external bundle may then be disconnected; retain the
configured reference path, which is pinned in `.reference-source.json`. Inspection,
subsequent processing and packaging use the managed assets. Imported reference
fingerprints and provenance are carried into packaged netCDF attributes.

Imported radar layout, observation ownership, matching centers and map grid remain
fixed. Each new acquisition's validity further masks its own support; gaps are
not filled from another burst. The supplied bundle is never changed. `--force`
reprocesses destination products without regenerating its reference. Reference
replacement and `stack rebuild-layout` require a new output directory when using
an imported reference. Configurations without `reference_directory` retain their
existing reference preparation and common-validity behavior.

A shared reference does not extend an empirical correction's application scope.
The shipped 2017 Larsen spacecraft/swath calibration must be independently
validated for 2018 before applying it there.

#### Visualize a stack plan

```bash
glacier-offsets stack plan configs/stack.generated.yaml > plan.json
python tools/plot_stack_plan.py plan.json --formats png pdf
```

The standalone tool reads only the saved JSON and writes to `plan_plots/`:
an overview of network components, acquisition gaps, pair separations and pair
counts per date; per-geographic-burst networks showing partial coverage and
disconnections; and `summary.json` with exact gaps, component memberships and
metadata errors. A connected date graph does not guarantee connected coverage
at every burst location. Missing/nonpositive burst intervals and pairs with no
usable shared bursts are distinguished from viable links; this is a metadata
check, not evidence of matching quality. Use `--gap-days 9` to change the gap
highlight threshold, `--burst ID` (repeatable) to focus on particular bursts,
`-o DIRECTORY` to choose an output folder, or `--show` for Matplotlib windows.
No native processing libraries or mounted input volumes are needed to plot.

### Merge-first stacks

Set `stack.processing: isce3_merge_first` in a stack YAML to use stack-common
valid rectangles and later-source overwrite merging, assembled by the
ISCE3-backed compositor. This selector uses the ISCE3 matcher unless
`matching.engine: isce2` is given explicitly. See
[the example configuration](configs/stack.merge-first.example.yaml):

```yaml
stack:
  processing: isce3_merge_first
  swath_merge: midpoint
matching:
  engine: isce3
  backend: cpu
  deramp: 0
  range_oversampling: 2
  zoom: 16
```

These are the recommended production settings. Leave `snr_threshold` and
`correlation_threshold` at zero and filter afterwards, for example with
`(peak >= 0.3 & SNR >= 8) | SNR >= 20`. `range_oversampling` is described in
[Range-2x matching](#range-2x-matching) below.

`isce2_merge_first` is the earlier, deprecated name for the same merge policy;
loading it logs a deprecation warning. Both selectors use the ISCE3
compositor, which replaced the original GDAL merger after the Larsen parity
gates (October 24). Omitting the engine under the deprecated name still
selects the ISCE2 matcher. Both selectors share policy records and cache
identities with stacks the GDAL merger produced, so those roots keep their
geometry, aligned bursts, assemblies and pair products without `--force`;
released merged SLCs are rebuilt by the ISCE3 compositor with identical pixels
and masks. Manifests record `layout_implementation` and
`assembly_implementation` for products the ISCE3 compositor wrote; these
records enter no cache identity. The existing
`stack.processing: isce3` interior-ownership policy is unchanged.

The ISCE3 matcher needs no ISCE2 installation or runtime probe. The existing
ISCE3 CPU worker scheduling and GPU selection still apply. Changing only the
matcher preserves pinned geometry, common validity, and aligned acquisitions;
it invalidates offsets, geocoding and diagnostics. Rerunning in the same root
replaces those pair results. Released merged SLCs are reassembled from retained
aligned bursts. Preserve the earlier pair outputs before comparing engines.

To keep the common valid rectangles and later-burst overwrites but move the
cross-swath switch into the overlap interior, add:

```yaml
stack:
  processing: isce3_merge_first
  swath_merge: midpoint
matching:
  engine: isce3
  deramp: 0
```

#### Swath ownership and layout rebuilds

`swath_merge` defaults to `overwrite`. The `midpoint` option resolves burst
overwrites within each swath first, then cuts adjacent swath support at the
range midpoint for each azimuth interval. It retains unshared coverage and
requires at least 16 samples of overlap (eight per side). The same fixed owner
supplies SLCs and geometry on every date; unavailable samples remain masked.

For an existing common-valid stack, use the controlled rebuild before running:

```bash
glacier-offsets stack rebuild-layout configs/stack.swath-midpoint.yaml --dry-run
glacier-offsets stack rebuild-layout configs/stack.swath-midpoint.yaml
OMP_NUM_THREADS=8 glacier-offsets stack run configs/stack.swath-midpoint.yaml --backend cpu
```

This verifies and preserves COMPASS reference/aligned bursts, resampling offsets
and per-date validity metadata. It replaces ownership geometry, deletes pair
outputs and temporary assemblies, and journals the change for restart. Reference
inputs, selection and software must match the existing stack. Switching back
to `swath_merge: overwrite` uses the same rebuild command.

#### Common validity and compositor contract

This strategy retains COMPASS geometry-only coregistration. For each aligned
secondary burst it reads the actual COMPASS azimuth/range offset extrema and
ports ISCE2's `adjustValidSampleLine_V2` validity adjustment, including its
four/eight-pixel interpolation margins and sign cases. COMPASS offsets already
refer to secondary-local indices; ISCE2's separate relative-burst shift must
not be added a second time. Rectangles are clipped to the physical raster.
It then intersects reference and complete-date burst rectangles across all
active dates before assembling any SLC. Partial dates are excluded from this
intersection, following ISCE2's incomplete-acquisition handling, but retain
their available imagery. A failed alignment blocks common-layout preparation
instead of silently changing the inventory.

With `swath_merge: overwrite`, SLCs are merged in ascending swath/time order:
later valid rectangles overwrite earlier ones. With `midpoint`, this overwrite
rule applies within each swath before the range cuts described above. Neither
option interpolates, blends, or substitutes sources for missing imagery. Geometry and source indices use the same
ownership.

The compositor plans source placements, common validity and ownership once as
an immutable plan. The categorical ownership raster, geometry VRTs, date SLCs
and pair intervals all consume that plan. Placement uses ISCE3 radar grids with
the same epoch reconciliation, rounding and 0.001-pixel tolerance as before;
incompatible sampling and fractional placement are still rejected. Aligned
bursts are read and `slc.tif`, its external mask and `valid.tif` are written in
64-line `isce3.io.Raster` blocks, without a full-scene array. GDAL remains only
for TIFF creation options, external masks, VRTs and ancillary-file lists.

#### Range-2x matching

Native magnitude matching (`deramp: 0`) detects amplitude at the original
range sampling. Detection doubles the signal bandwidth, so the amplitude
spectrum aliases and range offsets are pulled toward whole pixels. On a
known-shift test with real Larsen speckle the bias amplitude was about 0.1 px
in IW1 and 0.06 px in IW2, for ISCE2 and `deramp: 0` alike. `deramp: 1` removes
it but adds burst-join artifacts in merged IW1. With
`matching.range_oversampling: 2`, stack CPU matching interpolates both merged
SLCs twice in range as complex data before native detection; that known-shift
bias was then below 0.003 px, with no join artifacts.

```yaml
matching:
  engine: isce3
  backend: cpu      # required
  deramp: 0         # required
  range_oversampling: 2
```

For each pair, complete rows of both merged SLCs are Fourier-interpolated
(periodic, no taper or fill) into temporary stripped TIFFs, about twice each
merged SLC's size (about 10 GB per Larsen pair), and removed after matching.
They go under `pairs/<pair>/radar/native/range2x/` unless
`storage.scratch_directory` names an existing local directory. Use a fast
local disk there: on the external Larsen volume, sustained writes of about
20 MB/s made preparation take about 190 s per date, against about 16 s on the
internal SSD. The scratch location enters no cache identity. Native Ampcor runs with window, search and stride
doubled in range on those inputs; raw and correlation oversampling and zoom are
unchanged, so the physical range refinement extent halves. Chip centers,
input support, ownership and seam flags stay on the original grid, and gross,
residual and total offsets and covariance are reported in original reference
pixels. On the full Larsen 2017-01-19_2017-01-25 pair (8 workers, inputs on the
internal SSD), range-2x took 164 s against 93 s for `deramp: 0`: 33 s of
preparation and 131 s of matching. Peak worker RSS was 1.2 GB against 0.85 GB. Segment metadata and `native_parameters.range_oversampling`
record the treatment.

The default, 1, keeps existing matching identities, so completed pairs are not
rematched. Setting 2 changes the matching identity: rerunning a root rematches
its pairs, reusing geometry, aligned bursts and assemblies. Standalone pair/AOI
matching, GPU matching and the ISCE2 engine reject `range_oversampling: 2`.

#### Optional ISCE2 matcher

Matching workers always read these **merged** SLCs, including across
worker boundaries. `OMP_NUM_THREADS` is the maximum number of independent CPU
matching processes; date preparation and pairs remain sequential. Constant and
velocity-guided gross offsets retain the existing sign convention and are
subtracted from ISCE2's returned total to record the residual exactly once.

The adapter calls `mroipac.ampcor.Ampcor`, the same Fortran kernel used by
ISCE2 DenseAmpcor, with both input types set to `mag`. It preserves our explicit
matching grid instead of DenseAmpcor's automatic grid and fork wrapper.
Raw oversampling is fixed at two; `oversampling` and `zoom` map to ISCE2's native
refinement settings. `deramp` and `batch` are ISCE3 settings and do not control
this engine. Native covariance is converted from range/azimuth order into our
azimuth/range convention. This implements ISCE2 validity/merge/matching on
COMPASS-aligned inputs; it does not replace COMPASS with ISCE2 resampling or NESD.

ISCE2 can run in a separate environment. All paths resolve relative to the
YAML that declares them. For this machine's existing installation:

```yaml
stack:
  processing: isce3_merge_first
matching:
  engine: isce2
  backend: cpu
  zoom: 8
  correlation_threshold: 0.0
  isce2:
    python: /Users/briel/miniforge3/envs/isce/bin/python
    packages: /Users/briel/apps/isce2/packages
    library_path: /Users/briel/miniforge3/envs/gmt/lib
output:
  directory: /path/to/new/stack-isce2
```

`python: null` uses the workflow's interpreter when ISCE2 is installed there.
`packages` and `library_path` are optional; the latter is needed only when the
native installation needs additional library resolution. Environment overrides
apply only to the child process. The local library path above supplies the GDAL
ABI required by this particular ISCE2 build; it is not a general installation
requirement. Runtime import errors fail before stack mutation. Engine version,
native binary and Python wrapper checksums enter matching provenance and cache
identities. There is no ISCE3 fallback for an ISCE2 request.

ISCE2 does not expose a refined normalized correlation peak. That field and
its geocoded raster are explicitly NaN; they do not reject otherwise accepted
offsets. Use `snr_threshold` for additional quality selection and leave
`correlation_threshold: 0`. GPU requests are rejected for this CPU engine.

Use a new output root when switching an existing midpoint stack to common
validity. `rebuild-layout` supports the earlier midpoint-policy repair and
switches between the two common-valid swath merge policies described above.
On an ISCE2 stack, extending/filtering the active date inventory recomputes
common validity. Aligned burst products remain cached, but changed common
validity invalidates affected assembly and pair identities. The map extent
is retained from initial reference preparation. Completed merged SLCs retain
the normal bounded-storage cleanup/rebuild behavior.

```bash
glacier-offsets stack validate configs/stack.isce2.example.yaml --backend cpu
OMP_NUM_THREADS=4 glacier-offsets stack run configs/stack.isce2.example.yaml --backend cpu
```

## Swath calibration and correction

### Optional production spacecraft swath correction

Use `configs/stack.larsen.swath-corrected.yaml` to enable a separately published
relative IW1/IW2 azimuth calibration. It inherits the unchanged production
matching/geocoding configuration and pins the January–June 2017 spacecraft
coefficient, **β = 0.0257650993 original reference azimuth pixels**, by SHA-256.
There is no fitting, physical timing correction, or per-acquisition model in
this production stage. The ordinary production config keeps it disabled.
The Larsen YAML is local, like its production parent; the versioned generic
template is `configs/stack.swath-corrected.example.yaml`. The calibration JSON
is versioned separately from machine-specific configurations.

#### Estimate swath calibration factors

The installed CLI includes overlap estimation and calibration export. Run this
in the ISCE3 environment, from any working directory, using the processed
stack's YAML with `stack.swath_correction.enabled: false`:

```bash
glacier-offsets stack calibrate-swaths /path/to/configs/stack.yaml \
  --output /path/to/swath-calibration --workers 4
```

The results directory must be outside the production stack and its ancestors.
The estimator uses eligible acquisitions in the saved stack catalog and builds
links between each date and its next two neighbors. It needs completed reference
geometry, retained aligned per-burst SLCs and alignment products, SAFE inputs,
precise orbits and the configured DEM. It requires ISCE3 CPU matching,
`deramp: 0`, `range_oversampling: 2`, zero gross offsets and nearest geocoding.
It performs unspliced overlap matching and physical comparisons, then fits and
validates empirical models. Stack and calibration locks protect retained inputs;
production pairs remain unchanged. Estimation is independently invoked and is
not added to `stack run`.

The full command runs `plan`, `audit`, `measure`, `physics`, `fit`, `previews`,
`report` and `export`. Select one with `--stage STAGE`. Use `--pilot-only` to
measure just the first three-date triangle; pilots never export a calibration.
Incomplete stages stop a full run. Rerunning unchanged work uses verified
checkpoints. Support-threshold sensitivity rescoring remains explicitly opt-in
with `--stage rescore` and never changes the exported primary-cohort estimate.

Review `REPORT.md`, `summary.json` and `analysis/primary.json` in the results
directory. The full run also creates:

- `calibration.json`: the coefficient, descriptive uncertainty, actual reference
  PRF, geometry/matching/date scope, training dates, validation outcome and source
  checksums, in the format consumed by `correct-swaths`.
- `calibration.json.sha256`: the checksum pin for that exact JSON.
- `calibration-settings.yaml`: the `stack.swath_correction` settings to copy into
  your processing YAML; its calibration path is absolute.

The primary model's `models.spacecraft.parameters[0]` supplies `beta_px`;
`sqrt(models.spacecraft.covariance[0][0])` supplies `beta_std_px`. Unidentifiable
coefficients, missing covariance, incomplete measurements or physics stages,
changed inputs and incompatible reference/matching settings block export.
Inconclusive or negative predictive validation is recorded explicitly and is
not relabeled as validated accuracy. Application is restricted to the fitted
acquisition interval; the command does not widen an existing calibration or
apply corrections automatically. Model error and absolute accuracy remain
outside the descriptive coefficient uncertainty.

With an imported reference, the estimator uses the local bundled DEM resolved
by `stack.reference_directory` and verifies its original checksum. Laptop paths
retained in reference provenance need not exist; leave that provenance unchanged
and use `inputs.dem: null` to select the bundled DEM.

To export an already completed experiment, including outputs of the historical
`tools.swath_correction_experiment` command, without rematching:

```bash
glacier-offsets stack calibrate-swaths /path/to/configs/stack.yaml \
  --output /path/to/existing-swath-experiment --stage export
```

Legacy exports need the YAML and retained stack manifests for reference/track
metadata, but do not discover SAFE files, load native burst geometry, or require
the original experimental source checkout. Transferred result paths are resolved
inside the supplied results directory and registered fit products/dependency
reports are checksum-verified. Newly planned results also save portable reference
metadata, allowing `--stage export --output RESULTS_DIR` with no YAML and the
stack disconnected. Existing experimental measurements retain their provenance;
use export-only to consume them without rerunning the implementation after its
move into the package. Legacy `tools.*` imports remain compatibility shims for
research scripts and are not required by the installed estimator.

Use `--calibration-output /path/to/calibration.json` to choose another JSON
location. Identical exports are reused. A different existing JSON is preserved
unless `--force` is supplied; this option replaces only calibration export
files, not upstream matches. Recalculate a pin after editing any calibration:

```bash
sha256sum /path/to/calibration.json       # Linux
shasum -a 256 /path/to/calibration.json   # macOS
```

The separate frozen Larsen 2018 estimate is also available as
[`configs/calibrations/larsen-2018-spacecraft.json`](configs/calibrations/larsen-2018-spacecraft.json).
It records its inconclusive original held-out improvement tests. The
[method description](docs/stack-processing/SWATH_CORRECTION_METHOD.md) defines
the signs, weights and symmetric allocation. Copy the exported JSON to the
processing computer and use its path/checksum to apply the correction below.

#### Apply a published swath calibration

```yaml
stack:
  swath_correction:
    enabled: true
    calibration: calibrations/larsen-2017h1-spacecraft.json
    calibration_sha256: ed6ab421174d8941559b0fd2decc8ba43c318c28e1e79b568d8630b6420878dc
    pairs_directory: ../work/larsen/stack/pairs_swath_corrected
```

Paths resolve relative to the YAML declaring them, including inherited paths.
Omitting `pairs_directory` uses `<output.directory>/pairs_swath_corrected`.
The corrected destination must be separate from raw pairs, geometry and
acquisitions; symlinks cannot redirect publication into original products.

For already completed production pairs, run only the new stage:

```bash
glacier-offsets stack correct-swaths configs/stack.yaml
```

Add `--pair 2017-01-19_2017-01-25` to process one saved active pair, repeating
`--pair` to select several. This command uses saved metadata and completed
offset/geometry products, without SAFE discovery, rematching or modifying raw
pair manifests. The raw-data directories may be offline. An active writer or
unfinished layout rebuild blocks it. Missing/stale prerequisites are reported
per pair and never trigger upstream processing.

`stack run` with the opt-in YAML applies the correction after its ordinary
stages. `stack status` includes `correct-swaths` when enabled. A correction
failure does not invalidate completed raw stages. `--force` on
`correct-swaths` rebuilds only the derived products; as before, `--force` on
`run` rebuilds the upstream workflow too.

The coefficient gives `Δ = β × (is_B(later) − is_B(earlier))`. Subtract
`−Δ/2` from IW1 and `+Δ/2` from IW2. Same-spacecraft pairs get zero correction.
Complete chip/search footprints must belong to one swath; same-swath burst
joins qualify. Mixed or unsupported footprints retain their **raw** values
with `correction_available=0`. This is the continuous-footprint presentation
used in the clean comparison maps, with no seam outlines, filling or smoothing.

Each `<pairs_directory>/<earlier>_<later>/` contains:

- `geo/azimuth_displacement_m.tif`:
  corrected where supported, original observations elsewhere.
- Raw azimuth comparison rasters, bias-to-subtract layers in m,
  correction availability, cross-swath mixing, and same-swath join masks.
- Unchanged range rasters, validity layers and native formal uncertainties.
  Separate `azimuth_correction_std_*` layers describe calibration sensitivity;
  they are shared across observations and exclude model error. They are not
  added to native covariance as if independent pixel noise.
- `radar/offsets.h5`: original gross/residual/total offsets and native covariance,
  plus `corrected_total/*`, `bias_to_subtract/azimuth_px`, availability and
  correction uncertainty arrays. **The original `total/*` datasets remain raw.**
- `diagnostics/report.json`, a clean `diagnostics/comparison.png`, the exact
  `calibration.json`, and a checksummed `manifest.json`.

The stage converts pixel corrections using the retained local azimuth ground
scale and exact per-source interval before applying the **same production
nearest-source mapping and mask**. It verifies that the mapped raw field is
bit-identical to production. Raw inputs are checksum/revision checked before
and after use. Completed jobs are reused only when calibration, implementation,
source identities/checksums and all output checksums match; failed jobs retry
independently. The changing global progress catalog is not a cache dependency.
`correction_status.json` in the corrected directory records the latest selected
jobs; per-pair completion manifests are authoritative. Metadata-only `status`
checks pins, identities, consumed-input revisions and output existence, while
execution additionally verifies scientific checksums.
macOS `._` sidecars are excluded from raster inputs and completion products.

The shipped calibration is scoped to the pinned Larsen geometry, orbit 38,
descending HH, midpoint ownership, matching treatment and calendar 2017. Its
provenance distinguishes the January–June calibration period from July–December
application, which is explicitly labelled **unvalidated** in the output.
Unknown spacecraft, changed matching/geometry, invalid calibration or a checksum
mismatch fail rather than becoming zero predictions. A new stack needs its own
compatible, explicitly pinned calibration. The constant extension across each
swath and the symmetric allocation remain relative-calibration assumptions;
this stage does not establish absolute motion accuracy or fix noisy matching.

## Displacement stack post-processing

### Package displacement stacks as netCDF

Package the **saved active pair catalog**, independently of `stack run`:

```bash
glacier-offsets package configs/stack.generated.yaml
glacier-offsets package configs/stack.generated.yaml --output /path/displacements.nc
glacier-offsets package configs/stack.generated.yaml --force
```

```yaml
packaging:
  source: raw  # raw | swath_corrected; independent of whether correction is enabled
  output: null # default: <output.directory>/packages/displacements_<source>.nc
```

Output paths follow the usual YAML-relative inheritance rules; `--output` is
relative to the current directory and overrides the YAML. Packaging uses
`h5netcdf` (included in the environment specifications), without requiring
iceutils in the processing environment. It never runs upstream processing or
discovers newly downloaded acquisitions. Rerun the upstream workflow first to
change the saved active catalog.
Packaging validates the saved matching backend without probing CPU/CUDA; a
processing-time `--backend` override need not be written back into the YAML.

Every newly generated **stack** geocode has the `displacement-v1` product
profile: displacements, displacement uncertainties, SNR/peak metrics, masks,
timing and geometry. It produces no velocity, velocity uncertainty, or swath
correction field in m/day. Stack diagnostics show displacements. Existing
completed maps remain reusable and their velocity files are ignored by
packaging; explicit geocoding regeneration removes the four workflow-owned
velocity rasters. Standalone pair/AOI commands retain their existing products.
Product inventory versions do not change unchanged displacement identities.

Only complete, compatible active pairs enter the package. Missing, stale,
damaged or incompatible pairs are listed with reasons in the command report
and `omitted_pairs_json`; inactive pairs are excluded. Empty completed pairs
remain as all-invalid slices. At least one eligible pair gives exit 0, including
partial packages; no eligible pairs or invalid shared prerequisites give exit 2.
Interruption gives 130 and preserves any previous complete package. The command
holds the stack writer lock and rejects unfinished layout rebuilds.

`source: swath_corrected` requires an enabled correction configuration and
already completed corrected products matching its pinned calibration and raw
identities. It accepts checksum-verified legacy corrections with their recorded
implementation provenance. Missing corrected pairs are omitted; they never
fall back to raw pairs. Unsupported **pixels** within a completed correction
retain raw values, as before, and are identified by `correction_available=0`.

Packaging preserves the common projected grid without resampling. The initial
release requires `geocoding.resampling: nearest`. For a linear stack, change
that setting and run `stack geocode`; matching need not be repeated. The chosen
`geocoding.mask` and original matching thresholds are retained without new
filtering. Unavailable quality metrics, including ISCE2 correlation peaks, stay
NaN without removing otherwise valid observations.

The file uses schema `glacier-offsets-displacements-v1`, dimensions
`(time, y, x)`, CF time/CRS metadata and a global `EPSG` recognized by iceutils.
Its five primary variables are:

| Variable | Type | Units/convention |
| --- | --- | --- |
| `range_displacement_m` | float32 | metres, positive away from sensor |
| `azimuth_displacement_m` | float32 | metres, positive increasing reference sensing time |
| `snr` | float32 | dimensionless, original matcher SNR |
| `correlation_peak` | float32 | dimensionless, original peak or NaN if unavailable |
| `valid_mask` | uint8 | 0 invalid, 1 valid; no `_FillValue` |

Floating variables are NaN outside valid support. Variables are losslessly
compressed with chunks no larger than `(1, 256, 256)`. `time` labels pair
midpoints, sorted with pair ID as a tie-breaker; duplicate midpoint labels are
allowed. `pair_id` and `names` uniquely identify each slice. These are pair
observations, not cumulative displacements from a common acquisition.

Auxiliary variables keep the file self-contained:

- `earlier_time`, `later_time`, `time_bounds`, spacecraft/orbit metadata and
  exact `earlier_burst_time`, `later_burst_time`, `pair_interval_seconds`
  tables indexed by `(time, burst)`. Durations retain the existing exact
  **burst-start** convention, not a rounded day count or a claim of per-pixel
  secondary sensing times.
- `source_burst_index(y,x)` is 1-based into `burst_id`; zero means no source.
  `source_observation_index` records the zero-based flattened sparse observation
  copied to each cell, with -1 outside geometry. Multiple map cells can therefore
  be recognized as one measurement rather than independent observations.
- `observation_longitude`, `observation_latitude`, `height_m` (WGS84 ellipsoid),
  `reference_center_line`, `reference_center_sample`, local azimuth ground scale,
  per-burst reference radar grids and `source_radar_offset`. Observation positions
  are the original chip centers; `x/y` are output cell centers.
- `reference_los_enu(enu,y,x)` is the unit vector **ground to reference sensor**
  in local east/north/up coordinates. `range_sensitivity_enu` and
  `azimuth_sensitivity_enu` are observable displacement metres per metre of ENU
  motion, evaluated by native reference geo2rdr with symmetric ±1 m ECEF
  perturbations. Azimuth sensitivity uses the saved local ground scale and
  reference PRF; it is not an assumed horizontal unit heading vector.
- `reference_orbit_time/position/velocity` preserve the pinned ECEF state vectors.
  Provenance includes input checksums, matching/geocoding settings, software,
  geometry/layout identities and, for corrected products, complete calibration,
  application scope and per-pixel correction availability/mixing/join flags.

New reference preparation retains a separately checksummed
`geometry/reference_orbit.h5` snapshot. Existing stacks need their **pinned
reference EOF** connected once for enrichment; its SHA-256 must match the saved
reference specification. No SAFE, original DEM or velocity-prior file is read.
Supplemental sparse viewing geometry is cached separately beneath `geometry/`,
without changing coregistration/matching identities or raw pair manifests.
Later packaging can reuse those snapshots offline. Existing saved outputs must
still be available to build a new package.

The package is atomically published only after validation and input revision
checks. Unchanged packages are verified and reused; changed sources or omissions
rebuild them. `--force` affects packaging only. Internal variable checksums and
`validate_package(path)` allow standalone integrity checks without opening any
stack outputs. `correct-tides` consumes this file plus a local CATS2008 model.
Future ENU/projected-velocity tools can also use this self-contained format.

```python
import iceutils as ice

stack = ice.Stack('/path/displacements.nc')
displacement = stack.slice(0, key='range_displacement_m')
mask = stack.slice(0, key='valid_mask')  # remains uint8
print(stack.names, stack.hdr.epsg)
stack.close()
```

There is no duplicate `data` variable: pass the desired measurement name as
`key=`. Use `stack.ds` for timing tables, geometry and provenance. Mean fields
or inversions must explicitly account for each pair's endpoints and duration.

For an exact comparison against all source rasters and an independent iceutils
check on a standalone copy, use:

```bash
python tools/validate_packaging.py configs/stack.generated.yaml /path/displacements.nc \
  --ice-python /path/to/ice/bin/python
```

### Correct vertical ocean tides

Install the optional predictor in the processing environment, preserving the
project's NumPy/ISCE3 constraints:

```bash
python -m pip install -e '.[tides]'
glacier-offsets correct-tides /path/displacements.nc \
  --model /path/CATS2008_v2023.nc \
  --output /path/displacements_tide_corrected.nc
```

This is a standalone **netCDF-to-netCDF** operation. It reads no stack YAML,
SAFE, orbit, DEM, GeoTIFF, pair manifest or acquisition directory. All observation
locations, burst acquisition times, associations and viewing geometry come from
the input displacement package; the only external scientific input is the local
CATS2008-v2023 model. Raw and swath-corrected packages are both accepted.
The default output is `INPUT_tide_corrected.nc` beside the source. Inputs and
the model are never modified. The command is independent of `stack run` and
`package` and never downloads model files.

The vertical-motion term follows Sun, Riel and Minchew (2023), *Disintegration
and Buttressing Effect of the Landfast Sea Ice in the Larsen B Embayment,
Antarctic Peninsula*, Supporting Information Text S1, equation (1). For each
source observation, predict upward-positive heights at its exact earlier/later
**burst-start** times and compute:

```text
delta_z = eta(later) - eta(earlier)
range_bias = delta_z * range_sensitivity_enu[up]
range_corrected = range_input - range_bias
```

With positive range away from the sensor, the up sensitivity is approximately
`-cos(incidence)`, giving `range_corrected = range_input + delta_z*cos(incidence)`.
The primary range stays in **slant-range metres**. The paper's additional
division by `sin(incidence)` is deferred to a future horizontal-velocity step.
Azimuth is unchanged. This corrects modeled vertical ocean-tide motion only;
it does not remove horizontal tidal flow or add solid-Earth/load-tide terms.

The first implementation supports the local **CATS2008-v2023 TMD3 netCDF**
(`CATS2008_v2023.nc`), not the older OTIS binary distribution. Default
`--response model` uses its embedded BedMachine-v2-based flexure/grounding
response: constituent fields are multiplied by the flexure percentage divided
by 100 **before spatial interpolation**, as in pyTMD. The small modeled
overshoots above 100% are retained. Zero-response grounded cells get a known
zero correction. A missing flexure field is an error, without a freely-floating
fallback. `--response floating` is an explicit alternative for full ocean-tide
response, restricted to cells with ocean-model support.

Coordinates are transformed into the model's own polar stereographic grid
(including its kilometre units and central meridian). Only the required height
constituent window is read; Antarctic current fields are not loaded. Complex
harmonics are linearly interpolated at actual chip-center locations, without
extrapolation or hole filling. Predictions use pyTMD's TMD3 nodal corrections,
minor-constituent inference, UTC days since 1992-01-01 and zero delta-T according
to the OTIS/TMD3 convention. Repeated map cells sharing one radar observation
reuse the same prediction, with a bounded cache of burst/endpoint results.

The output retains the five primary variables, original map grid, pair labels,
timing, metadata, integer `valid_mask`, SNR, correlation peak, azimuth and
reference geometry. It adds:

- `range_displacement_before_tides_m(time,y,x)`: exact input range for audit.
- `tide_height_earlier_m`, `tide_height_later_m`,
  `tidal_vertical_displacement_m` and `tidal_range_bias_m`, all in metres.
- `tidal_correction_available(time,y,x)`: uint8 0/1, without a fill value.
- `tidal_response_factor(y,x)` and `tidal_model_ocean_mask(y,x)` for spatial
  provenance. The response is informational; it must not be multiplied into
  the already corrected observations a second time.
- `tidal_correction_provenance_json` and a processing-history entry containing
  input/model checksums, method, software version, response policy and counts.

Unavailable model predictions retain the original displacement and validity,
with correction availability zero and auxiliary tide fields NaN. The measurement
mask and tide availability therefore answer different questions. Counts include
known zero-response cells; map cells copied from one observation remain dependent.
The processing uses the package's existing burst-start timing convention, not
scene midpoints or a claim of exact per-pixel secondary sensing times.

Outputs are verified and atomically replaced after input/model revision checks.
An unchanged result is reused. A different recipe at the same output path needs
`--force`, which rebuilds from the uncorrected source. Already tide-corrected
inputs are rejected even with `--force`, preventing double application. An
interruption preserves both the source and any previously completed output.
Exit codes are 0 successful/reused, 2 invalid prerequisites or processing failure,
and 130 interrupted. `validate_tidal_stack(path)` validates the output in
isolation, including its checksums, availability and projection arithmetic.

Model/predictor references:
[CATS2008-v2023](https://doi.org/10.15784/601772) and
[pyTMD ocean/load-tide prediction](https://pytmd.readthedocs.io/en/stable/getting_started/Getting-Started.html).

For an audit using only the two netCDF files, optionally including an independent
pyTMD model-driver comparison and iceutils read of a standalone copy:

```bash
PYTHONPATH=src:. python tools/validate_tidal_stack.py INPUT.nc CORRECTED.nc \
  --model /path/CATS2008_v2023.nc --ice-python /path/to/ice/bin/python
```

### Convert to local east/north displacements

```bash
glacier-offsets to-horizontal /path/displacements_tide_corrected.nc \
  --vertical-motion zero --output /path/displacements_horizontal.nc
```

`to-horizontal` is an independently invoked netCDF-to-netCDF operation. Raw,
swath-corrected and range-tide-corrected displacement packages are accepted.
It needs no model, orbit file, DEM, SAFE, YAML or other stack outputs. Default
output is `INPUT_horizontal.nc`; `--force` rebuilds conversion from the radar
parent. It does not modify the input. Identical recipes reuse a validated output;
changed inputs/settings need a new output path or `--force`. Writers share the
derived-output directory lock. Temporary outputs are validated, source revisions
checked again, and published by atomic replacement.

For each reference observation, the stored horizontal sensitivities define:

```text
range_displacement   = g_range_east * east  + g_range_north * north
azimuth_displacement = g_azimuth_east * east + g_azimuth_north * north
```

The converter solves this linearized 2x2 system in double precision and stores
`east_displacement_m(time,y,x)` and `north_displacement_m(time,y,x)` as float32
metres. Positive values mean eastward/northward, later minus earlier. It uses
the native reference-geometry sensitivities and existing azimuth scale, without
an extra incidence correction or division by pair duration. Geometry is evaluated
at the pinned reference observation, not recomputed along the displaced trajectory.

`--vertical-motion zero` (the only currently supported constraint) assumes zero
residual vertical displacement in the supplied radar observations. Two radar
components cannot resolve unconstrained 3D motion. Surface-parallel motion and
other vertical-motion constraints are not implemented. Apply range tidal
correction **before** conversion. **Azimuth tidal correction remains disabled**:
azimuth is used exactly as supplied, including any residual apparent azimuth
motion from vertical tides. No azimuth tidal term is subtracted inside conversion.
For tide-corrected inputs, cells without an available range tide correction are
excluded from horizontal valid support. Known grounded zero corrections remain
available. Swath-correction fallback flags and all existing correction history
are retained.

The existing map CRS, shape, transform, pixel-center coordinates and source
observation associations are unchanged. No reprojection or interpolation occurs.
The new components are physical **local geographic east/north**, even when stored
on an EPSG:3031 grid; they are not projected grid-x/grid-y coordinate differences.
All original radar variables and masks remain available unchanged.

Geometry checks are separate from the original measurement mask. Defaults are:

- `--max-condition 3`: maximum 2-norm condition number of the horizontal matrix.
  A minimum singular value of `1e-6` m/m also excludes effectively zero sensitivity.
- `--max-azimuth-scale-ratio 2`: local azimuth scale must lie between one half
  and twice its burst's median. Each unique source observation contributes once
  to that median, regardless of how many map cells copy it. This conservative
  check excludes all 226 Larsen outliers identified in the audit below; it does
  not repair retained geometry or establish DEM accuracy.

These settings and medians are recorded in the conversion recipe. Raising the
limits changes the recipe and admits more geometry. Additional variables are:

- `horizontal_geometry_flags(y,x)`: uint8 bits 1 missing geometry, 2 ill-conditioned
  or weak sensitivity, 4 azimuth-scale outlier; zero means accepted.
- `horizontal_geometry_valid_mask(y,x)`, `horizontal_condition_number(y,x)` and
  `horizontal_azimuth_scale_ratio(y,x)`: static diagnostics.
- `horizontal_valid_mask(time,y,x)`: uint8 1 for a usable horizontal observation.
- `horizontal_validity_flags(time,y,x)`: uint8 bits 1 input invalid, 2 geometry
  rejected, 4 requested range tide correction unavailable, 8 numerical overflow.
  Reasons can overlap. Floating horizontal measurements are NaN wherever this
  flag is nonzero. All integer masks/flags have no fill value, preserving zeros.
- `horizontal_uncertainty_available(time,y,x)`: uint8 0/1, independent of the
  measurement mask. SNR and correlation peak are never treated as uncertainties.
- `horizontal_conversion_provenance_json`: input checksum/identity, software,
  settings, assumptions, preserved-variable hashes and coverage counts.

If the input supplies a complete formal covariance via
`range_displacement_sigma_m`, `azimuth_displacement_sigma_m` (metres), and
`range_azimuth_displacement_covariance_m2` (m2), all with `(time,y,x)` dimensions,
the converter propagates `C_EN = inverse(A) C_RA inverse(A).T`. It writes
`east_displacement_sigma_m`, `north_displacement_sigma_m` and
`east_north_displacement_covariance_m2`. Invalid/non-PSD covariance produces
unavailable uncertainty without discarding the displacement. Existing packages
without the complete triplet have unavailable uncertainty; no zero cross
covariance is assumed. This propagation covers formal matching uncertainty only,
excluding tide-model/calibration errors and correlations between shared acquisitions.

Empty completed slices remain in the output. A nonempty input with no usable
horizontal observations is an error; partial coverage succeeds and is reported.
`validate_horizontal_stack(path)` checks hashes, masks, inversion/forward
equations and optional covariance using the output file alone. For a separate
closed-form check of every accepted slice and an optional standalone iceutils read:

```bash
PYTHONPATH=src:. python tools/validate_horizontal_stack.py RADAR.nc HORIZONTAL.nc \
  --ice-python /path/to/ice/bin/python
```

```python
import iceutils as ice
stack = ice.Stack('/path/displacements_horizontal.nc')
east = stack.slice(0, key='east_displacement_m')
valid = stack.slice(0, key='horizontal_valid_mask').astype(bool)
early = stack.ds['earlier_burst_time']  # (time, burst), exact UTC endpoints
late = stack.ds['later_burst_time']
stack.close()
```

The output contains pair displacements, not a velocity fit. For a later fit with
displacement basis `B_j(t)`, use rows `B_j(later) - B_j(earlier)` at each pixel's
source-burst endpoints and differentiate the fitted displacement function to
obtain velocity. Alternatively, integrate velocity basis functions over each
pair interval. Midpoints are labels, not instantaneous observations. The constant
displacement term is unobservable and must be omitted or anchored. Temporal gaps,
shared-acquisition correlations, regularization and fit diagnostics remain the
responsibility of that future fitting step.

## Products and scientific interpretation

Standalone pair outputs, illustrated by `work/jan10_jan22/t038_080975_iw2/`, include:

- `manifest.json`, COMPASS YAMLs/logs, isolated orbit links, original reference complex SLC, resampled secondary complex SLC, reference grid, x/y/z geometry and layover/shadow.
- `radar/offsets.h5`: raw applied gross/residual/total for range and azimuth, exact center line/sample, input/quality masks, rejection flags, SNR, correlation peak and native covariance. `matching_grid.h5` carries sparse-grid associations. `radar/native/` preserves unmodified native binary outputs; float32 BIP offsets are azimuth then range, covariance has three components. No Earth affine is assigned to radar products.
- `geo/`: total range/azimuth displacement (m), velocity (m/day), validity and quality GeoTIFFs, formal uncertainty GeoTIFFs, and source-center geometry/azimuth scales in HDF5. `valid_mask.tif` marks available cells under the selected source mask; `quality_valid_mask.tif` separately marks quality-accepted cells. Both use 0/1. Missing displacement/velocity values are NaN.
- `diagnostics/`: amplitude/mask overview, window locations, raw gross/residual/total and filtered-total QC, source geometry and velocity figures, `uncertainty.png` with displacement/velocity formal 1σ maps, and a JSON coverage/rejection/limitations report. Empty results are explicitly labeled.
- `validation/m4-closeout/zero` and `validation/m4-closeout/constant`: retained products using the final settings. `validation/zero` and `validation/constant` preserve the earlier baseline.

New stack `geo/` products contain displacements and their uncertainties, quality, masks and geometry; standalone pair products also contain velocities. See [netCDF packaging](#package-displacement-stacks-as-netcdf) for the stack schema.

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

### Burst footprints in KML

Export burst footprints for one SAFE directory or ZIP:

```bash
conda activate isce3
python tools/export_bursts_kml.py /path/acquisition.zip --output bursts.kml
```

The KML has colored IW1/IW2/IW3 folders, one placemark per geographic burst ID, annotation-derived polygons, a center point for its label, and polarization/UTC/source metadata. It uses the actual annotation geolocation-grid borders, including s1-reader's split polygons at the antimeridian. These footprints show nominal burst coverage, not the SLC valid-data mask. No DEM, coregistration, or image extraction is required. Without an orbit, s1-reader derives geographic IDs from the annotation ascending-node time. Add `--orbit-dir data/orbits` to auto-select a precise orbit using the same resolver as the YAML initializer, or `--orbit /path/orbit.EOF` to specify one, for consistency with processing burst IDs. Polarization is inferred if unique; otherwise supply `--polarization HH` (or VV/HV/VH). `--subswaths 1 2` limits the export. The default output is `<SAFE stem>.bursts.kml` in the current directory; existing files require `--force` to replace. Output follows [KML 2.2 coordinate and polygon conventions](https://developers.google.com/kml/documentation/kmlreference).

### Standalone radar-offset plot

The plotting script reads an existing burst output directory (or its `radar` directory) without rerunning processing:

```bash
conda activate isce3
python tools/plot_radar_offsets.py work/jan10_jan22/t038_080975_iw2 --field total
python tools/plot_radar_offsets.py work/jan10_jan22/t038_080975_iw2/radar --field residual --mask valid
```

It saves three vertically stacked panels (3×1) for azimuth offset, range offset and SNR. Axes use full-resolution reference range samples and azimuth lines at matching-window centers, with increasing azimuth downward. Offsets are in SLC pixels. The default is unfiltered observations; `--mask input` excludes invalid inputs, and `--mask valid` selects quality-valid observations. Color limits use the 98th percentile; `--percentile 100` shows the full range. `--output figure.png` selects the output file, and `--show` also opens an interactive window. By default the PNG is saved beside `offsets.h5`.

## Troubleshooting

Use `stack status CONFIG.yaml` to inspect the saved active plan and stage
prerequisites; use `stack plan CONFIG.yaml` after changing the input inventory.
The checks below address configuration and restart errors without editing saved
scientific identities.

| Symptom | Check |
| --- | --- |
| `Configured DEM conflicts with reference bundle` | An inherited DEM may still be set. Use `inputs.dem: null` to select the bundled DEM, or supply an identical checksum-verified DEM. See [portable references](#transfer-a-prepared-reference-to-another-computer). |
| Laptop paths appear in a transferred reference | Original reference paths remain provenance. Acquisition SAFE/orbit paths and the destination output directory must resolve on the processing computer; runtime reference assets use the managed bundle. |
| `Unsupported swath calibration method or units` | Point `stack.swath_correction.calibration` at the exported `calibration.json`. `fit.json` and `analysis/primary.json` are analysis outputs. See [estimation and export](#estimate-swath-calibration-factors). |
| The calibration checksum differs | Calculate SHA-256 for the exact calibration JSON being used and update `calibration_sha256`; a changed file requires a new pin. |
| A legacy `tools.*` estimation command cannot be imported | Use the installed `glacier-offsets stack calibrate-swaths` command. For completed experimental results, use `--stage export` to avoid rematching. |
| A stage is stale or blocked | Check the saved prerequisite and resolved configuration in `stack status`. Run the required stage; reserve `--force` for an intentional rebuild. An unfinished layout rebuild must be resumed before normal processing. |

A successful empty pair records that no valid observations were available; it
is different from an incomplete stage. A finite displacement or a zero quality
threshold does not establish measurement accuracy. Consult [quality and mask
conventions](#products-and-scientific-interpretation) before applying filters.

## Validation and investigation history

The sections above describe current operation. The records below retain
dataset-specific experiments, implementation milestones and historical
acceptance gates; some linked local artifacts are not distributed with
the source checkout. Historical scope statements describe their original
experiment or milestone.

### Validation status

M4 is closed with the conditional limitations documented in the [milestone/validation report](docs/validation.md). Following the geocoding strategy update, the suite ran 43 tests: 42 passed and one CUDA consistency test was skipped. Fresh zero/constant real runs, units/nodata/uncertainty audits, native geometry checks, restart checks and installed-wheel smoke passed at closeout. The subsequent [geocoding comparison](docs/geocoding-strategies.md) also verifies unchanged radar/coregistration products and native map-to-radar sample selection. This real 12-day pair has sparse quality-valid coverage. No independently identified stationary-ground mask or independent displacement truth has been supplied, so absolute accuracy and stationary-ground bias are not established. CUDA is absent on this macOS ARM CPU build. M5 has since been authorized, implemented and CPU-validated using the supplied MEaSUREs v2 Antarctic velocity mosaic; see [the M5 report](docs/m5.md).

```bash
PYTHONPATH=src:.runtime/deps OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
GLACIER_OFFSETS_PYTHON=/Users/briel/miniforge3/envs/isce3/bin/python \
  PYTHONPATH=src:.runtime/deps OMP_NUM_THREADS=4 python tools/validate_pair.py \
  --snapshot-name my-validation --report-directory docs/my-validation
PYTHONPATH=src python tools/geometry_check.py --config configs/pair.yaml --report docs/my-validation/geometry.json
```

The validation script forces fresh matching in zero and constant modes, retains snapshots, restores zero products, and checks unchanged restarts without recomputing COMPASS SLCs. Use a new snapshot name for each validation set; `--overwrite` explicitly refreshes that named set. Supply `--config` and `--constant-config` for a different pair; both configs must have identical inputs, output, matching grid/settings and geocoding settings. Real mode differences are sensitivity measurements; known-shift correctness is established by the synthetic tests.

### Matcher comparison and range-oversampling experiments

For an isolated full-pair test that preserves the baseline, use a config with
the settings above and a new comparison directory outside the source stack:

```bash
OMP_NUM_THREADS=8 python tools/compare_stack_engines.py configs/stack.isce3.yaml \
  /path/to/stack/pairs/2017-01-01_2017-01-07 --output /path/to/comparison
```

The tool verifies retained aligned inputs, reconstructs the saved ownership,
runs matching/geocoding/diagnostics, checks identical grid and input support,
and writes `comparison.json` and `matcher_comparison.png`. It releases its
temporary merged SLCs after completion. It requires a saved ISCE2 baseline
with zero or constant gross offsets; velocity-prior comparisons are excluded.

To compare ISCE3 `deramp: 0` and `deramp: 1` independently in IW1 and IW2,
use the same-patch overlap diagnostic on a saved zero-gross, `deramp: 0` pair:

```bash
PYTHONPATH=src:.runtime/deps MPLCONFIGDIR=/tmp/glacier-mpl OMP_NUM_THREADS=4 \
  /Users/briel/miniforge3/envs/isce3/bin/python tools/compare_swath_overlap.py \
  /path/to/stack/pairs/2017-01-19_2017-01-25 \
  --output /path/to/new-deramp-comparison --workers 4 --compare-deramp
```

The command verifies retained aligned burst SLCs, selects each swath's latest
fully supported source before matching, and runs both modes at identical saved
global centers. Both workers read unspliced aligned bursts. Chip/search sizes,
stride, native refinement settings, thresholds, geometry and support are fixed;
only `deramp` changes. Source data and production results remain read-only.
For triangles or new date pairs in the same stack, add
`--acquisition-dates 2017-05-31 2017-06-12 --compare-deramp`. This reads those
aligned acquisitions directly on the saved grid, verifies their input checksums,
and omits comparisons against the template pair's merged offsets. The template
may use either deramp mode; both unspliced modes are run for the requested dates.
No merged SLC assembly or full-field matching is needed for this diagnostic.
The output must be a new directory outside the source stack; there is no resume
mode. Omit `--compare-deramp` for the original magnitude-first diagnostic.

`deramp-0/` and `deramp-1/` retain all per-burst raw observations, native files,
quality/failure masks, effective parameters, combined `observations.npz`, reports
and individual plots. The root `selection.npz` records source owners and support
before either mode runs. Root `report.json`, `comparison_masks.npz` and
`deramp_comparison.png` compare both offset components, with fixed-support,
burst-interior, baseline-quality and common-quality groups. Inspect validity
transitions and sample counts alongside disagreement statistics: a smaller
disagreement on fewer observations does not establish improvement or accuracy.
No observations are selected for agreement, and failed owners never fall back
to another burst. Native job exceptions are retained in reports while independent
jobs continue; incomplete comparisons exit 1. `progress.json` records completed
jobs, including failures. Initial integrity checks can take time before matching
workers start. This diagnostic does not change production defaults or establish
absolute glacier-motion accuracy.

To isolate range input preparation further, use a completed paired-deramp
diagnostic as a read-only baseline:

```bash
PYTHONPATH=src:.runtime/deps MPLCONFIGDIR=/tmp/glacier-mpl OMP_NUM_THREADS=4 \
  /Users/briel/miniforge3/envs/isce3/bin/python -m tools.compare_swath_range_oversampling \
  /path/to/completed-deramp-comparison \
  --output /path/to/new-range-control --workers 4
```

This verifies the exact aligned inputs used by the baseline, interpolates full
complex rows at twice the range sampling, and retains only strips around the
original matching patches. It performs no azimuth interpolation or deramping
before native magnitude detection. A quarter-original-pixel origin adjustment
keeps physical chip centers and pixel-cell footprints fixed as window, search
and stride double in range. Offsets and covariance are converted back to original
pixel units; original support and source ownership are retained without fallback.
Native raw/refinement factors and zoom stay fixed, so range quantization is finer
and the physical native range-refinement extent is smaller. These differences,
SNR normalization and full-row periodic interpolation are explicit limitations
of this control. Reports compare unchanged baseline-quality cohorts and include
the earlier `deramp: 1` observations. `range_control.png`, `report.json`, per-burst
NPZ/native files, and prepared strip TIFFs are retained in the new output root.

For the full merged pair, compare both treatments against its saved magnitude
baseline without writing to the production stack:

```bash
PYTHONPATH=src:.runtime/deps MPLCONFIGDIR=/tmp/glacier-mpl OMP_NUM_THREADS=4 \
  /Users/briel/miniforge3/envs/isce3/bin/python -m tools.compare_merged_preprocessing \
  configs/stack.larsen.swath-midpoint.yaml /path/to/stack/pairs/2017-01-19_2017-01-25 \
  --output /path/to/new-full-merged-comparison --workers 4
```

The tool reconstructs both merged SLCs from retained aligned bursts using the
saved layout and requires their checksums to match the baseline. Native workers
read complete merged images across scheduling boundaries in both treatments.
The range-2x adapter transforms each segment's coordinates and converts its
results back to original units before normal offset publication, geocoding and
diagnostics. Original input support, source ownership and seam flags must agree
exactly. Both variants reuse saved center geometry and the pinned map grid.
`comparison.json`, `radar_comparison.png`, `radar_changes.png`, and
`geo_comparison.png` retain full-field results and common-valid comparisons,
including seam/interior groups and quality attrition. Temporary merged and
range-oversampled images are released only after a successful comparison;
native results, radar offsets, maps, previews and provenance remain. Choose a
new output directory; this utility has no resume option. The range-only control
limitations above also apply to the full merged experiment.

For a scalar refinement-zoom sensitivity experiment, add `--zoom 32` and
`--zoom-baseline /path/to/completed-full-merged-comparison`, using another new
output directory. Only zoom may differ from the saved treatment settings.
Requested zooms that would be silently clipped by the native search cap are
rejected. With raw oversampling two, zoom 16 gives refinement half-widths
`[4, 4]` original pixels for native `deramp: 1` and `[4, 2]` for range-2x;
zoom 32 gives `[8, 8]` and `[8, 4]`, respectively. Thus range-2x/zoom-32
restores the range extent of native/zoom-16 while expanding azimuth extent.
The tool verifies actual native zoom and writes `zoom_sensitivity.json`, shared
four-treatment masks, `zoom_changes.png` and `zoom_field_comparison.png`.
The report separates validity transitions, seam/interior changes, large tails,
native pool wall time and maximum individual-worker RSS; that RSS excludes
the parent and does not measure concurrent process-tree memory.

### Compositor parity checks

The
frozen GDAL merger is kept only in `tools/reference_stack_merge.py`, for parity
tests and the gate tool. To compare both on a saved stack without modifying it:

```bash
PYTHONPATH=src:.runtime/deps OMP_NUM_THREADS=8 \
  /Users/briel/miniforge3/envs/isce3/bin/python -m tools.compare_stack_mosaic \
  configs/stack.larsen.swath-midpoint.yaml --output /path/to/new-mosaic-parity
```

This verifies retained reference and aligned burst checksums, rebuilds both
swath policies with each implementation, compares every layout and date raster
block by block, and matches the selected pairs (`--pairs`, default January 1–7
and 19–25) on each implementation's SLCs plus a reference replay control.
`comparison.json` records per-field differences, validity transitions, wall
time, block I/O, per-process peak RSS and production size/timestamp/checksum
checks. Temporary SLCs are removed unless `--keep-slcs` is given.

### ISCE2 integration tests

Native integration tests can be enabled without installing ISCE2 into the
COMPASS environment:

```bash
GLACIER_ISCE2_PYTHON=/Users/briel/miniforge3/envs/isce/bin/python \
GLACIER_ISCE2_PACKAGES=/Users/briel/apps/isce2/packages \
GLACIER_ISCE2_LIBRARY_PATH=/Users/briel/miniforge3/envs/gmt/lib \
OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
```

### Isolated half-year swath experiment

The standalone experiment reads the retained **unspliced** IW1/IW2 bursts from
`configs/stack.larsen.production.yaml`. It measures the production matching
centers with the production complex range-2x kernel, then evaluates unfitted
bistatic/FM-rate candidates before fitting two small models to raw azimuth
contrasts: one spacecraft coefficient and one relative term per acquisition.
It does not add production correction stages or modify the source stack.

```bash
glacier-offsets stack calibrate-swaths configs/stack.larsen.production.yaml \
  --output work/larsen/swath-experiment-2017h1
```

The historical 31-date January–June plan had 30 adjacent links plus 29 skip-one links.
The first three dates form a pilot gate before full measurement expansion.
`--pilot-only` stops after that triangle. `--workers` accepts 1–4; pairs remain
sequential. Individual `plan`, `audit`, `measure`, `physics`, `fit`, `previews`
and `report` stages use the same arguments. Rerunning stages verifies inputs
and reuses current per-burst matching jobs and products. Changed stage code
invalidates the affected artifacts; a changed frozen configuration or geometry
requires a separate experiment directory. Missing or incomplete inputs are
reported explicitly. No production processing is launched.

Artifacts include a resolved plan, source/runconfig audit, per-link and per-burst
NPZ observations, sparse physical predictions in seconds and original pixels,
all quality/support counts, empirical parameter covariance, withheld-fold ledgers,
raw same-owner triangle closures, analysis plots, `REPORT.md` and `summary.json`.
Validation leaves out acquisitions, complete links, and burst groups (purging
shared source bursts). A per-date model cannot predict an unseen date or a
contrast across disconnected components. The 1,000 shared acquisition/source-burst
multiplier refits describe dependence sensitivity rather than independent-window
accuracy. No residual or agreement clipping is used.

The physical audit records the executed COMPASS radar path, original-mid-burst
coefficient selection, saved resampling offsets/Jacobians, native center/corner
geometry checks and discrepancies from the installed full-burst-span LUT routine.
Historical timing conventions remain explicitly unvalidated physical hypotheses;
missing predictions are never replaced by zero.

For completed production pairs, separate preview GeoTIFFs and derived radar arrays
use leave-one-link-out empirical contrasts and the mean unfitted physical
contrast. All use the same symmetric convention (subtract −Δ/2 from IW1 and
+Δ/2 from IW2) and explicitly assume a constant relative bias outside the measured
overlap. Mixed-swath footprints are unavailable for correction; same-swath burst
joins are marked separately. Maps retain raw observations wherever a correction
is unsupported, using the same production nearest-source mapping for every
panel. Common scientific availability remains a separate diagnostic mask.
Plots have no overlap outline or artificial masked strip; displacement color
limits are ±5 m. `comparison_clean.png` and `corrected_clean.png` provide the
clean spacecraft comparison and single corrected map. The pinned map grid, local physical
scales and exact source intervals are preserved. Native covariance and correction
uncertainty remain separate. These are diagnostic relative previews, not calibrated
production motion products or evidence of absolute accuracy.

To refresh existing previews from checksum-verified retained radar arrays and
production center geometry, without reading aligned SLCs or refitting models:

```bash
glacier-offsets stack calibrate-swaths configs/stack.larsen.production.yaml \
  --output /path/to/swath-calibration --stage previews --retained-previews
```

To evaluate the **mean swath mismatch per link**, run the separate post hoc stage:

```bash
glacier-offsets stack calibrate-swaths configs/stack.larsen.production.yaml \
  --output /path/to/swath-calibration --stage rescore
```

This reads the checksum-verified frozen analysis files without repeating matching,
physics or geocoding. Outputs go under `rescoring/posthoc-link-means-v1/`; the
original plan, estimates, reports and maps remain unchanged. The stage computes
error **after** averaging burst-pair means within a link, and refits both empirical
models for minimum supports of 0, 20, 30 and 50 selected windows/link. These thresholds
were chosen after inspecting the data and are explicitly labeled sensitivity analyses.
They use whole-link counts before withholding; actual held-out counts and coverage
remain in the fold ledgers. Overlapping windows are not independent replicates.

The new report includes spacecraft-order figures, support attrition, IPF-specific
coefficients with jointly resampled regime differences, all three withheld protocols,
and corrected displays of the original window-level scores. Every model is compared
against its own baseline on identical samples. Native observations are diagnostic
only, with no model-selection verdict. Spacecraft association and predictive error
reduction are reported separately; rescoring itself enables no production correction.

### Audit ENU coefficient stability

Before extending tidal correction to azimuth or converting radar offsets to
horizontal components, audit the packaged coefficients in the ISCE3 environment:

```bash
PYTHONPATH=src:. python tools/audit_enu_geometry.py /path/displacements.nc \
  --output .runtime/enu-geometry-audit/coefficients.json
```

This read-only tool reconstructs the pinned orbit from the netCDF alone. It
tests central differences at 0.01–30 m perturbations, geo2rdr time tolerances
of `1e-6`–`1e-12` seconds, and additional iteration/Doppler-derivative settings.
An independent derivative of the zero-Doppler equation checks both radar
components; a one-sided 3 m vertical move checks the finite tidal response.
The selection includes every declared condition/scale/vertical-response outlier,
spatial samples from every burst, and neighbors of the strongest outliers.
JSON records errors and solver failures; the companion NPZ retains all tested
coefficients. The original acceptance window is 0.1–10 m and `1e-8`–`1e-10`
seconds with an absolute coefficient error budget of `1e-5` m/m. A narrower
0.3–10 m practical plateau is reported separately. Exit status 1 means the
original window did not pass; it does not discard the report or stress failures.

Numerical convergence does not establish the accuracy of the retained local
azimuth scale. An optional independent check reads the retained reference
manifest and its checksum-pinned DEM, tests every declared outlier plus a
typical observation from each burst, and varies the native rdr2geo line span
and solver tolerance:

```bash
PYTHONPATH=src:. python tools/audit_azimuth_scale.py /path/displacements.nc \
  /path/stack/geometry/manifest.json \
  --output .runtime/enu-geometry-audit/scales.json
```

#### Larsen coefficient and scale results (2026-10-27)

The Larsen audit on 2026-10-27 tested 1,069 unique observations, including all
226 declared outliers across ten bursts. The stored production coefficients
(±1 m, `1e-8` seconds) matched the independent derivatives within `5.13e-7`
m/m. The 0.3–10 m practical plateau passed; the 0.1 m extreme narrowly exceeded
the original error budget, centimetre perturbations lost precision, and
`1e-12` seconds sometimes failed to converge even with more iterations.
The separate DEM check tested 236 observations. All ten typical scales agreed
within 0.02 m/line plus 1%, but only 16 of the 226 outlier scales did. At the
largest outlier, source observation 61479, the retained scale was 332.48 m/line
and fresh native intersections gave 14.59 m/line. Some terrain intersections
also failed or depended strongly on line span. These discrepancies require
investigating the retained geometry and scale before enabling azimuth tidal
correction. Neither audit changes geometry, displacements, or correction policy;
`correct-tides` continues to change range only.

### Horizontal conversion validation (2026-10-27)

The 2026-10-27 Larsen check converted all 60 slices (`60 x 873 x 978`). It retained
23,000,167 of 23,094,016 source-valid observations; 93,849 were excluded by the
geometry checks. Every accepted observation agreed with a separate closed-form
inversion within `7.63e-6` m, consistent with float32 storage precision. All source
variable values were preserved. A standalone copy opened in the installed
iceutils with uint8 horizontal masks and burst endpoints intact; unchanged reuse
also passed. The input had no full formal covariance, so horizontal uncertainty
was explicitly unavailable.

### Ownership-layout validation

Synthetic native and regression validation, plus the outstanding real-data
gates, are recorded in [stack validation](docs/stack-processing/README.md).

Retained aligned bursts can also be checked in isolation without rerunning
COMPASS or modifying a production stack:

```bash
python tools/validate_swath_ownership.py /path/to/old-stack \
  --pair 2017-01-01_2017-01-07 --output /path/to/new-validation \
  --native-samples 19700 21100
```

The tool assembles both dates in the new directory, reports full chip/search
support and optional native matching controls around the requested range strip.
Its output is a validation artifact, not a resumable production stack.
[stack.pig.interior.yaml](configs/stack.pig.interior.yaml) inherits the PIG
configuration and uses `work/pig/stack-interior` for the interior-cut policy:

```bash
glacier-offsets stack run configs/stack.pig.interior.yaml --backend cpu
```

### AOI-selected pair validation

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

### Repository and local validation records

Git tracks Python source (including audit scripts), the launcher, package/environment files, this README and selected reference configs. Data, processing outputs, images, plans, validation reports and other working documents stay local; links to those records below describe the working installation and may not exist in a fresh clone. New reference configs or non-Python source assets must be added to the explicit allowlist in `.gitignore`.
