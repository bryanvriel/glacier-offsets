# Sentinel-1 glacier offsets: detailed reference

[Project overview](README.md) · [Getting started](GETTING_STARTED.md)

This document contains the detailed command reference, processing conventions,
implementation notes and validation history.

A small, restartable Python workflow for local Sentinel-1 TOPS SAFE pairs and acquisition stacks, one polarization and either one geographic burst or complete bursts intersecting an AOI. COMPASS performs geometry-only radar coregistration; native ISCE3 Ampcor tracks **amplitudes of complex SLCs**. Offset products retain applied gross, native residual and total separately. Only total motion is converted to displacement/velocity and geocoded. Local georeferenced velocity fields can guide searches. Pair AOI processing assembles compatible SLCs on one radar grid before tracking beneath `merged/`. Stack processing prepares each acquisition once on a pinned reference frame and reuses its SLC across pairs. Optional input preparation downloads precise orbits and a cropped DEM; processing stages use those local files. There is no NESD or custom correlator.

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
  zoom: 16
```

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
for TIFF creation options, external masks, VRTs and ancillary-file lists. The
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
checks. Temporary SLCs are removed unless `--keep-slcs` is given. Matching workers always read these **merged** SLCs, including across
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

Native integration tests can be enabled without installing ISCE2 into the
COMPASS environment:

```bash
GLACIER_ISCE2_PYTHON=/Users/briel/miniforge3/envs/isce/bin/python \
GLACIER_ISCE2_PACKAGES=/Users/briel/apps/isce2/packages \
GLACIER_ISCE2_LIBRARY_PATH=/Users/briel/miniforge3/envs/gmt/lib \
OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
```

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

Multi-slice assembly, NESD, time-series inversion, raster cubes, concurrent date
preparation/pairs and fractional alignment of reference bursts remain outside V1.

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
