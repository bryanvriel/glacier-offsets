# NISAR GOFF packages

The importer consumes completed, enriched `nisar_subset.py` HDF5 subsets offline.
It does not download imagery, coregister acquisitions or repeat matching. Older
subsets without surface heights, covariance, or processing/orbit metadata must
be regenerated. Directory discovery ignores `.part` files and is nonrecursive.

```bash
# Processing environment (ISCE3 required for native geometry)
glacier-offsets package --nisar-subsets /data/nisar/subsets \
  --layer 1 --output /data/nisar/displacements.nc
glacier-offsets correct-tides /data/nisar/displacements.nc \
  --model /data/CATS2008_v2023.nc
glacier-offsets to-horizontal /data/nisar/displacements_tide_corrected.nc

# ice environment; use the same horizontal packages for both fitting families
glacier-offsets fit-velocity configs/velocity-recovery.nisar.example.yaml --stage inspect
# Review independent control/target regions before pilot.
glacier-offsets fit-velocity configs/velocity-recovery.nisar.example.yaml --stage pilot
glacier-offsets fit-velocity configs/velocity-recovery.nisar.example.yaml --stage map
glacier-offsets velocity-pixels run configs/velocity-pixel.nisar.example.yaml
```

`--nisar-subsets` also accepts explicit files and multiple directories. It is
mutually exclusive with the existing Sentinel-1 stack CONFIG positional argument.
`--output` is required; `--force` replaces only a recognized displacement package.
Publication is atomic and includes input revision/checksum checks and standalone
validation. Identical validated outputs are reused.

If a directory contains both `_001.h5` and `_002.h5` revisions of a product,
the default is to report a conflict. To select the highest numeric final counter:

```bash
glacier-offsets package --nisar-subsets /data/nisar/subsets \
  --nisar-revisions latest --layer 1 --output /data/nisar/displacements.nc
```

This applies only when granule IDs differ solely in that final counter. Different
frames remain separate inputs; processing releases and other granule differences
are not ranked automatically. Each selection is logged and recorded with the
superseded input checksums in package provenance and the command report. Source
files are left intact. The highest counter is an explicit revision-selection
policy, not a claim of better measurement accuracy.

### Packaging speed and restart

Native geometry runs in separate processes, using the available CPU count capped
at four by default. `--workers N` sets the number explicitly; `--workers 1` uses
the serial path. Work is queued in bounded 256x256 tiles and only the parent
process writes the package. Mask-invalid cells, nonfinite offsets, missing heights
and unavailable geometry cubes are screened before native solves. Invalid formal
covariance alone does not exclude a cell. The symmetric +/-1 m calculation is
unchanged for eligible cells; deliberately skipped cells have unavailable native
geometry. Original masks, measurements and heights remain in the package.

Completed geometry tiles are cached in `.nisar-geometry-cache` beside the output.
The cache survives an interrupted import and `--force` package rebuilds. It is
keyed by source checksum, matching layer, tile, method, implementation and geometry
dependency versions; tiles are checksum-verified on every reuse and damaged tiles
are rebuilt. Changes to source data or the implementation invalidate old entries.
Atomic tile publication permits concurrent packages to share a cache safely;
concurrent writers may redundantly compute the same tile. Final package locking,
checksums, validation and atomic publication still apply.

```bash
glacier-offsets package --nisar-subsets /data/nisar/subsets \
  --nisar-revisions latest --workers 8 --geometry-cache /scratch/nisar-geometry \
  --output /data/nisar/displacements.nc
```

`--no-geometry-cache` disables both reading and writing the cache. The cache can
be removed when no import is using it; the standalone package does not depend on
it. Worker count and cache location do not change package identity. INFO logs
show input inspection/checksumming, tile progress and approximate remaining time,
computed/skipped/cached cell counts, product timings, output hashing, validation,
and final source verification. The command report includes performance counters.
Cache hits avoid native solves; importing arrays, compression and validation still
take time. Small imports may run faster serially because process startup has a
fixed cost.

### Optional integer decimation

`--decimation N` retains every Nth native cell in both map axes before geometry
calculation. The default is 1 (80 m); factors 2 and 3 give 160 m and 240 m:

```bash
glacier-offsets package --nisar-subsets /data/nisar/subsets \
  --layer 1 --decimation 3 --output /data/nisar/displacements_240m.nc
```

Both axes use the fixed EPSG:3031 center lattice `40 + 80*N*k` metres. The global
anchor keeps different subsets, frames and tracks at common locations, independent
of input order or bounding-box edges. Coordinates are cropped to retained centers;
the output transform uses half the new posting around those centers. Products
with no retained cell keep an empty granule slice. A selection with no retained
cells anywhere fails explicitly.

This is point decimation, with no additional interpolation, filtering or averaging.
Retained offsets, source values, masks, heights, covariance and per-cell sensing
times keep their original values and units. Azimuth ground spacing per SLC line
also retains its native meaning. Native geometry is calculated only at retained
eligible cells. Sampling settings enter package provenance, geometry cache keys,
package identity and validation. Embedded source metadata still describes the
original input subset. `geocoding_resampling: bilinear` describes upstream GOFF
geocoding; `sampling.additional_resampling: none` describes this importer.

Corrections, control selection and both fitting readers accept the coarse grid.
Collections require a consistent posting/factor and global alignment; native and
coarse versions of a product are alternative samplings, not extra observations.
Regenerate grounded-control regions and empirical-noise calibration for the coarse
package. Physical control-block sizes remain in metres; fewer retained locations
may affect minimum-support requirements. Exact 200 m posting is unavailable through
integer decimation of 80 m inputs.

Each output contains one selected matching layer, frequency A and HH polarization.
It may include different tracks and frames, but all inputs must have consistent
maturity and processing release. Aligned EPSG:3031 inputs are placed on the union
of the selected lattice with missing support padded; offsets are not resampled.
Layers are separate experiments, never independent observations to pool. In the checked P05023 data,
matching windows are 32/64/128 pixels, with search half-widths **32/32/16 pixels**
and a 15-pixel skip. The 80 m posting is not the independent spatial resolution.

## Standalone v2 profile

`schema_version = glacier-offsets-displacements-v2`, `mission = NISAR`,
`observation_support = geocoded_cell`, `geocoding_resampling = bilinear`.
Sentinel-1 retains its v1 native-source profile and existing behavior.

| Fields | Dimensions | Convention |
| --- | --- | --- |
| range/azimuth displacement, SNR, correlation peak, valid mask | time,y,x | metres; later minus earlier; positive range away from sensor and azimuth increasing reference sensing time |
| earlier/later observation time | time,y,x | UTC seconds since 1970-01-01; each source cube's epoch decoded independently |
| LOS and range/azimuth ENU sensitivities | time,enu,y,x | ground-to-sensor unit LOS; observable metres per metre of local ENU motion |
| height, ground-track velocity, azimuth scale, geometry flags | time,y,x | WGS84 ellipsoidal height; scale in metres per RSLC line |
| range/azimuth sigma and cross covariance | time,y,x | metres and m2; uncertainty availability separate from measurement validity |
| input mask and source_* fields | time,y,x | unaltered source values, including invalid support |
| track, frame, reference/secondary absolute orbits | time | source product identities |
| source_metadata_json | time | embedded orbit, attitude, identification, grid and processing metadata |

`time` is a sorted pair-midpoint label; `pair_id`/`names` are full granule IDs.
Map coordinates are cell centers. Longitude/latitude identify those map locations.
No original radar-source index or burst ownership is invented.

Native reference geo2rdr uses symmetric +/-1 m local ENU perturbations at each
cell's supplied height. Range differences give range sensitivity; azimuth-time
differences are multiplied by interpolated ground-track velocity. These are
center-based linearizations, not a reconstruction of the original chips or
bilinear interpolation weights. The original GOFF values already contain
spatial correlation from overlapping windows and bilinear geocoding. The copied
formal covariance does not characterize all those correlations or systematic
errors and is not automatically used as fitting noise.

The encoded mask's fill value, both subswath validity digits and anomaly bits
8-23 determine input validity alongside finite offsets and usable geometry.
The water flag is retained and does **not** reject floating ice. Non-PSD formal
covariance removes uncertainty availability without removing valid motion.

## Corrections and fitting

`tools/build_grounded_control_regions.py` accepts a tide-corrected horizontal
NISAR package with the same command-line arguments as Sentinel-1. It uses map
locations, each cell's exact pair interval, horizontal validity, and the fitting
collection's deterministic frame-overlap selection. Counts in `sources` and
`supported_sources` refer to correlated map cells for NISAR, not independent
radar observations. The existing grounded/slow thresholds apply, including at
least 20 peak-supported observations per supported location; short records can
therefore produce an empty regions file. Package provenance records the
observation-support convention.

Control screening reads coverage masks in HDF5 chunk-aligned tiles, rejects
non-grounded boxes first, then reads only horizontal displacements, endpoint
times and quality/validity fields for possible controls. It does not load native
geometry or reconstruct horizontal vectors. Progress logs distinguish coverage,
time-series scanning and the final sequential SHA-256 read of the package.
By default this proposal tool checks package metadata and records that file
checksum; `--verify-package` additionally verifies every NISAR array before
screening, which can be expensive. The JSON records the verification mode.
Fitting retains its independent full package validation.

If `velocity-pixels run` stops after reporting completed fits and exported maps,
keep the result/cache directories and rerun the same command after updating the
installation. Completed fits are checksum-verified and reused; unfinished plot,
verification and report stages resume. Version-2 plots can read the map association
stored in `source_coefficients.h5` without a local observation-cache copy. Exported
NISAR maps use `analysis_location_index`; Sentinel-1 maps use
`source_observation_index`. NISAR reports summarize track/year noise covariance
groups directly, including older cached noise records without annual diagnostics.

### Pixel-workflow performance

`velocity-pixels` verifies the complete package once per file SHA-256 and
validator implementation, storing an atomic validation receipt under
`cache_directory/verified-nisar`. Subsequent collections still checksum the
input bytes; they reuse full array validation only for a matching receipt.
Preparation reuses the collection's just-verified checksum only while its file
revision remains unchanged. Changed inputs or validator code require validation
again. The initial verification can take minutes for a large package.

Observation caches use a versioned compact layout (`nisar-observations-v4-compact`),
retaining only locations with at least one usable horizontal observation. Original
row-major map IDs remain in `source_id`; `map_source_index` maps grid cells to
compact cache rows, with -1 outside support. Every product retains its own
geometry, endpoints and overlap-suppression flags. Coordinates, exports and
point selection preserve the original map locations. Dense GOFF coverage can
leave almost no empty locations to remove.

Extraction uses 256-cell spatial tiles and contiguous cache writes, with the
existing completed-pair checkpoints. A single package's tiles align with its
native HDF5 chunks; shifted packages may straddle neighboring source chunks.
Radar extraction avoids horizontal matrix inversion when the package already
provides horizontal validity. Old caches are not interpreted as compact caches.
PySolid reads bounded blocks of pair geometry, caps NISAR time blocks at eight
products, and streams progress; its independent sampled accuracy check remains.

Acquisition-purged screening batches well-conditioned robust solves across
locations. The same date/acquisition exclusions, rank threshold, five IRLS
updates, MAD floor and residual cuts apply. Systems with poor conditioning
replay the scalar SVD least-squares implementation. Screening omits predictions
for target observations already excluded by validity/peak screening; it retains
all permitted training observations. Both control calibration and map fitting
use this path. Scientific coverage gates and per-track comparators are unchanged.

Benchmark representative controls without fitting a map:

```bash
PYTHONPATH=src python tools/benchmark_nisar_velocity.py \
  --package /data/nisar/displacements_tide_corrected_horizontal.nc \
  --regions /data/nisar/reviewed-regions.json \
  --cache /data/nisar/velocity-cache --output /data/nisar/benchmark.json
```

The benchmark validates its input, samples up to 128 locations in each of three
control regions, compares screening against the scalar reference, and checks
aligned extraction against the previous tile traversal. Reuse that cache directory
in the fitting YAML to reuse validation. Large observation/correction caches
still require space, and millions of location fits remain substantial work.

CATS uses each location's own two endpoint times and pair geometry, changing
range only. Missing modeled corrections retain input radar values with a separate
availability mask; they are excluded from horizontal fitting support. Horizontal
conversion uses the pair's 2x2 matrix with zero residual vertical motion, retaining
the original radar variables and propagating full formal covariance. The pixel
workflow fits those retained radar components, including its residual M2/O1
vertical model. PySolid uses spatially varying endpoints and each pair's geometry;
it is applied exactly once during pixel preparation.

Collections can combine NISAR tracks at common aligned map locations; mixed
NISAR/Sentinel-1 collections are rejected. Adjacent frames sharing a track and
reference/secondary acquisition-orbit pair are not independent repeats. At a
location, the valid frame with the lowest frame number wins; full granule ID
breaks ties. Collection manifests record overlap groups and this policy.

For `velocity-pixels`, controls/supplied noise must set `group_by: track_year`.
Group labels are `T075:2026`, for example. Supplied JSON uses
`covariance_ra_m2_by_group`. Every observed group needs a positive-definite
empirical radar covariance; there is no other-track/year fallback. Alternatively,
use `noise.mode: pair` with a NISAR `fit-velocity` pair-noise calibration bound to
the exact same input package hashes. Pair estimates shrink within their own
track/year group. Reported control biases are not subtracted from glacier motion.

Joint fits retain annual EN secular motion, along-flow Msf and residual vertical
M2/O1 terms, evaluated at exact endpoints. No Sentinel-1 spacecraft-order term is
assigned to NISAR. Azimuth-only comparators are stored separately as
`azimuth_track_NNN`; the compatibility `azimuth_direct` group is explicitly
unavailable. Maps export per-track comparator summaries. Joint displays use joint
estimability rather than one track's azimuth heading as a veto.

Control screening, cross validation and recovery retain physical map blocks and
acquisition-purged temporal sampling. Spatially adjacent GOFF cells must not be
interpreted as independent native measurements. Coverage/rank gates remain in
force: a few real pairs can test packaging and geometry but cannot establish
multi-year harmonic recovery or calibrated scientific accuracy.
