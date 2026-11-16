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

Each output contains one selected matching layer, frequency A and HH polarization.
It may include different tracks and frames, but all inputs must have consistent
maturity and processing release. Aligned EPSG:3031 80 m grids are placed on their
union with missing support padded; offsets are not resampled. Layers are separate
experiments, never independent observations to pool. In the checked P05023 data,
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
