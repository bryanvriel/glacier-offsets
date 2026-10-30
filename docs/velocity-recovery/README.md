# Exact-pair Msf velocity recovery

The `fit-velocity` workflow reads self-contained horizontal netCDF displacement
packages and compares free EN and along-flow Msf models. It runs in the existing
`ice` environment. It does not import packaging, GDAL, COMPASS or ISCE3, discover
raw SAR inputs, or load the processing environment's `.runtime/deps` directory.

```bash
GLACIER_OFFSETS_PYTHON=/Users/briel/miniforge3/envs/ice/bin/python \
  bin/glacier-offsets fit-velocity configs/velocity-recovery.example.yaml --stage inspect
# Review/edit the proposed regions, then set regions in the fit YAML.
GLACIER_OFFSETS_PYTHON=/Users/briel/miniforge3/envs/ice/bin/python \
  bin/glacier-offsets fit-velocity CONFIG.yaml --stage pilot
GLACIER_OFFSETS_PYTHON=/Users/briel/miniforge3/envs/ice/bin/python \
  bin/glacier-offsets fit-velocity CONFIG.yaml --stage map
```

The workspace launcher and main console/module dispatcher remove the workspace
`.runtime/deps` path before importing numerical fitting modules. A caller that
already imported numerical modules from that overlay must start a clean process.
The selected interpreter supplies NumPy/SciPy/h5netcdf; packages
are not installed or changed. The code uses SciPy for the coupled EN solve,
because scalar iceutils time-series inversions cannot represent a full 2×2
observation covariance. Scalar weights retain iceutils' **1/sigma** convention;
Huber factors multiply those weights by their square roots.

## Inputs and preservation

Copy the example fit YAML and provide one or several horizontal netCDF paths.
Paths resolve relative to that YAML. `start` is inclusive UTC and `end` is
exclusive UTC. Source data variable checksums are verified before fitting;
complete file hashes, revisions and correction history enter the experiment
fingerprint. Inputs must share grids, pinned reference geometry, layout, ENU axes,
source-burst ownership, sparse observation associations and sensitivity arrays.
Duplicate pair IDs are errors. Duplicate midpoint labels are allowed. Missing
burst endpoints remain unavailable; scene times are never substituted.
Unsupported swath-correction fallback pixels remain in the input for audit but
are excluded from calibrated temporal observations.

Baseline freezing and two-year processing preparation are separate from temporal
inference. Run in the ice environment:

```bash
PYTHONPATH=src /path/to/ice/bin/python tools/prepare_velocity_experiment.py \
  BASELINE_RAW.nc BASELINE_SWATH.nc BASELINE_TIDES.nc BASELINE_HORIZONTAL.nc \
  --freeze-directory .runtime/velocity-recovery/frozen-2017 \
  --stack-config configs/stack.larsen.production.yaml \
  --extended-config configs/stack.larsen.velocity-2017-2018.yaml
```

This verifies internal data hashes, copies files, verifies copied SHA256s and
source revisions, marks copies read-only and writes `frozen.json`. Existing
snapshots are reused only when their hashes match. The new stack YAML inherits
the current processing settings, pins 2017-01-01, extends through 2018-12-31 and
disables spacecraft correction. It does not launch processing or alter the
existing calibration. The existing stack workflow handles aligned-product reuse,
common-validity changes and retention of its pinned map grid.

Run stack planning, validation and processing in the ISCE3 environment. Extending
common validity can invalidate saved pair products, so finish freezing first.
Use the same extended-grid products for the eventual 2017/2018/joint comparison;
the frozen old-grid record is an audit baseline, not the record-length control.

## Region review

`inspect` writes `inspection.json`, `inspection_recipe.json`,
`regions.proposed.json` and `regions.proposed.png`. Candidate squares are about
8 km across. Selection uses coverage, geometry, median displacement/duration,
height and stored tidal response. It does not fit an Msf harmonic. If the supplied
fields cannot support all six candidates, the report states the shortfall.

The proposal JSON contains an EPSG and a `regions` list. Each entry has a unique
`id`, a `role` (`target` or `control`) and projected metre `bounds` in
`[west,south,east,north]` order. After reviewing geography and independent bedrock
information, set `reviewed: true` and record the basis in `review_notes`. The pilot
requires at least four targets and two bedrock controls. Low apparent motion and
zero modeled tide response alone do not establish bedrock truth.

Pilot samples are unique radar sources, with one source per approximately 3 km
block within a region and at least 3 km between selected source centers.
Physical source positions define spatial blocks when
available. Copied map cells never increase regional sample counts or the map
multiple-testing count. All proposal and sampling choices precede harmonic fits.

## Observation and temporal models

Displacements satisfy `d = integral(v(t), early, late)` in local geographic EN.
Every pixel uses its owning burst's exact UTC endpoints. Velocity harmonics use
`P = 14.765294` days and phase origin 2000-01-01 00:00:00 UTC:

```text
v_Msf(t) = C cos(omega tau) + S sin(omega tau)
amplitude = sqrt(C² + S²)       # instantaneous velocity, m/day
phase = atan2(S,C)             # radians; cos(omega tau - phase)
```

Harmonic integrals are analytic, with a cancellation-resistant sinc form. No
post-fit attenuation correction is applied. Free EN has four Msf coefficients;
along-flow has two. Both leave secular and trend EN components free. A shared
robust secular fit supplies the local axis; pilot resamples reassess it. Bedrock
has no physical flow direction, so control summaries explicitly use east as the
projection axis; full EN control residuals determine noise.

The default bank contains constant velocity, a linear velocity trend, and the
same cubic integrated B-spline dictionary with L2 or L1 penalties. Spline scales
are 60/120/240 days. Displacement columns are differences of analytic primitives;
velocity evaluation uses their B-spline derivatives. Columns are standardized
within each training fit, and only spline coefficients are penalized. The bank
uses nine penalties `10^-4` through `10^4`. L1 uses one weighting iteration and a
convex epigraph solve with an explicit feasible duality-gap certificate, using
the existing ice environment's CVXOPT. Secular/Msf columns are eliminated by
orthogonal projection before that solve and remain unpenalized. The numerical
library retains a coordinate-descent fallback for lightweight callers; the full
pilot command requires CVXOPT. Primary models include no horizontal M2/O1
or automatic annual term.

## Noise, selection and uncertainty

Controls supply cross-fitted robust secular residuals. Noise estimation balances
3 km control groups and fits full range/azimuth covariance, including correlation.
The quality model fits nonpositive slopes of log variance against log SNR/peak;
observed high-quality residual scatter supplies component-specific noise floors.
Missing peaks select the SNR-only calibration. Missing/insufficient quality or
values outside its calibration domain use constant covariance. Precision is
capped in every covariance direction at 100 times baseline, corresponding to ten
times baseline inverse standard deviation.

The four weighting candidates are constant covariance, constant covariance with
robust weights, quality covariance with robust weights, and the existing quality
rule `(peak >= .3 and SNR >= 8) or SNR >= 20`. SNR and peak are not inverse
variances. Covariance is transformed through the stored horizontal inverse and
whitened jointly; a single Huber factor applies to both components of a pair,
using Mahalanobis threshold 2.45.

Predictive selection uses 60-day held-out blocks and purges every training pair
sharing an acquisition with a test pair. A test control region is excluded from
noise learning. Candidates are scored on the same held-out observations, including
observations rejected during quality-rule training. Fold losses balance regions
and spatial blocks. Within one predictive standard error, selection prefers
simpler weights/backgrounds and stronger penalties, subject to tuning recovery.
The full fold ledger and candidate scores are retained.

Default pilot uncertainty uses 1,000 paired draws of approximately 30-day temporal
and 3 km spatial blocks. Both parameterizations share the draws; flow direction
is reassessed. `paired_bootstrap.json` retains regional cosine/sine draws and
`paired_intervals.json` gives circular intervals and paired coefficient
differences. NetCDF covariance is conditional on the temporal model, estimated
noise and fitted robust/active-set weights. For L1 it is conditional on the active
set. These matrices do not include correction/model-selection uncertainty; inspect
resampling and sensitivity reports separately.

Rank-revealing SVD reports global conditioning and Msf information after nuisance
projection. `target_data_rank` describes information without the spline penalty;
`target_rank` includes the specified regularization. A regularized spline result
can therefore be estimable only conditional on that background penalty. The global
condition number alone is not a detection criterion.

## Calibration validation

The January–June 2017 calibration's existing scope is preserved. Applying it to
July–December 2017 was already labeled unvalidated; it cannot become evidence for
2018 by changing a date field. Extend the independent same-ground swath experiment
with the existing `tools.swath_correction_experiment` workflow after aligned 2018
inputs exist, using a separate experiment directory.

`tools/validate_velocity_calibration.py` reads its checksummed primary overlap
analysis and measurement evidence and compares split-year spacecraft estimates.
Held-out acquisition and shared-burst-purged folds compare common and year-specific
corrections, with paired acquisition/source-burst multiplier draws. A common
coefficient needs consistent split-year estimates and predictive performance
within one SE of the year-specific alternative. Otherwise each year must pass
independent validation, and unsupported cross-year pairs must be excluded.

```bash
PYTHONPATH=src /path/to/ice/bin/python tools/validate_velocity_calibration.py \
  /path/to/extended-overlap-experiment \
  --catalog /path/to/extended-stack/stack_manifest.json \
  --output .runtime/velocity-recovery/calibration-comparison.json
```

This publishes recommendations and estimates, not a production calibration.
Publish new, scoped calibration files using those estimates and existing production
correction conventions. Regenerate corrected packages and horizontal components.
Re-run the validator with `--inputs HORIZONTAL_2017.nc HORIZONTAL_2018.nc` (or one
joint input). It can then produce schema
`glacier-offsets-calibration-validation-v1`, binding exact input hashes and
checksummed evidence. Applied coefficients, overlap geometry/layout and matching
treatment must agree. Point `calibration_validation` in the fit YAML at this
report. A failed report remains a useful diagnostic and blocks mapping.

## Recovery and scientific gates

Tuning, validation and paired sampling use separate `SeedSequence` branches
`[12345,0]`, `[12345,1]` and `[12345,2]`. Validation injects 0/.025/.05/.10/.20 m/day
Msf amplitudes with stratified uniform phases and empirical paired control noise
resampled in approximately 30-day blocks. Each control residual is first whitened
by its own quality-dependent covariance, then recolored at the target's quality;
quality variation is not applied twice. Cases retain actual timing, geometry,
quality and missing-data patterns. They compare available 2017, 2018 and joint
records and include the controlled 90°/105° comparison, ±5°/10°/20° direction
mismatches, horizontal M2 (.03 m/day) and O1 (.01 m/day), residual vertical M2/O1
(.60/.20 m displacement at 5/10/20%), range-tide scales .8/1/1.2, shared calibration
uncertainty, trends, abrupt changes and unremoved azimuth vertical-tide response.
Controlled flow azimuths follow the supplied reference: degrees clockwise from
geographic north, with EN unit vector `[sin(azimuth), cos(azimuth)]`.
Unavailable correction sensitivities are labeled explicitly. Synthetic tests
also cover six/twelve-day attenuation and exact/near alias degeneracy.

`validation_recovery.json` retains every point/case/period/model score. A valid
scientific result can fail the .10 m/day benchmark. Maps require median relative
amplitude error <=20%, circular phase error <=20°, null false positives <=5%, and
95% coefficient-ellipse coverage between 90% and 99% in every tested clean target.
At least 200 successful validation draws and 1,000 paired uncertainty draws are
required. Failure or reduced experimental banks block maps, with reasons in
`pilot.json` and `REPORT.md`. Nuisance/correction sensitivity is reported separately
and must accompany any interpretation; passing the clean test does not establish
robustness to all correction assumptions.

Tuning stops a candidate at its first failed required recovery check and records
that failure; the independent validation stage still evaluates the full prescribed
case bank for the provisional or shortlisted models. Identical calendar/joint
cohorts reuse the same paired validation draws and are explicitly labeled as
equivalent, rather than being counted as independent records. Checksummed pilot
stage and tuning-attempt checkpoints are bound to the full source/configuration
recipe, so unchanged interrupted pilots can resume.

`tools/validate_existing_velocity_calibration.py` can assess an existing published
coefficient against unseen links within 2017, purging its published training
acquisitions and testing source-burst-purged generalization. It reports split-period
consistency separately from evidence of held-out error reduction. It never rewrites
the input calibration or asserts support for 2018.

## Products, restart and evaluation

Pilot products include separate `pilot_free_en.nc` and `pilot_along_flow.nc`,
noise/CV/tuning/validation reports, paired uncertainty, year-specific and joint
coefficients on the same local axis, and daily EN curves/plots for each sampled
ice point. Map products are separate `map_free_en.nc` and `map_along_flow.nc`.
All store coefficients, their covariance, basis definitions, Msf amplitude/phase,
ellipse/cross-flow diagnostics, target information/rank, residual/weight summaries,
coverage and distinct `fit_valid`, `detection_valid`, `phase_valid` masks.

Default coverage gates are 40 pairs/270 days or 60 pairs/540 days, plus at least
nine of twelve Msf phase bins. The period selected from the input controls which
coverage gate applies. A fit can be estimable without a significant detection.
Map significance tests the full Msf target (four coefficients for free EN, two
for along-flow); the two-coefficient along-flow projection p-value is also stored.
Map significance uses Benjamini–Yekutieli FDR <=.05 across unique source observations
in **both** models, allowing dependence from overlapping windows. Unsupported map
phase is NaN; `phase_valid` additionally requires supported detection and phase
standard deviation <=20°.

Map tiles are atomic netCDFs indexed by `tiles.json`. Restarts verify their SHA256s
and exact source/configuration/implementation/region/calibration fingerprints;
changed experiments require a new output directory. Final maps are assembled
atomically per file; `map_manifest.json` marks a completed pair and records hashes.
Source revisions are checked before publication. An exclusive directory lock
rejects competing fit writers.

The evaluator needs only a coefficient file:

```python
from glacier_offsets.velocity_io import evaluate_file, export_daily
velocity_en = evaluate_file('map_along_flow.nc', ['2017-01-01', '2017-01-02'], row=100, col=200)
# Full-grid daily cubes are an explicit, potentially large export:
export_daily('map_along_flow.nc', 'daily_velocity.nc', '2017-01-01', '2019-01-01')
```

No full-grid daily cube is written by inspection, pilot fitting or mapping.

## Local execution status

The Larsen inspection and frozen 2017 baselines live under
`.runtime/velocity-recovery-larsen/`. The user approved all four ice regions and
both stable-bedrock controls on October 28, 2026; that approval is recorded in
`regions.reviewed.json`. The completed one-year pilot writes beneath `2017/`,
with 80 candidates per model, six predictive folds, 200 validation draws per case
and 1,000 paired uncertainty draws. No model passed all null/coverage/recovery
requirements, so no production map was generated. The report distinguishes
provisional diagnostic fits from validated detections. The user will handle
extension processing separately. Scientific recovery acceptance and appropriate
correction validation remain required before any production map claim.
The unit/integration tests exercise synthetic pilot completion, map gates,
checksummed tile reuse, covariance semantics and standalone evaluation in the ice
environment; they do not constitute real-data scientific acceptance.

## Revised quality and regional validation

The separate 2017 reassessment uses first-half controls to select a failed-match
screen and recalibrate noise, then tests second-half controls with acquisition
purging and control-region exclusions. It fits shared regional along-flow Msf
coefficients with source-specific secular motion and exact source-burst times.
Temporal blocks are shared across the spatial field; unique sources and physical
spatial blocks are retained. Observed coefficients are compared in joint,
range-only and azimuth-only fits, including bedrock controls.

Run from the repository root in the existing ice environment:

```bash
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /Users/briel/miniforge3/envs/ice/bin/python tools/revalidate_velocity_quality.py
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /Users/briel/miniforge3/envs/ice/bin/python tools/assess_velocity_quality_sensitivity.py
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /Users/briel/miniforge3/envs/ice/bin/python tools/check_velocity_period_conditioning.py
PYTHONPATH=src:. /Users/briel/miniforge3/envs/ice/bin/python tools/verify_velocity_quality_outputs.py
```

The default output is `.runtime/velocity-recovery-larsen/2017-quality-regional/`.
`FINAL_REPORT.md` explains the completed results. The main experiment uses 1,000
regional bootstrap draws and 200 validation draws per case. Its paired empirical
sampler carries quality labels and failure residuals together; legacy and clean
samplers provide comparisons. Simulations repeat response-dependent screening.
Coverage is tested for frozen regional bootstrap coefficient ellipses, separately
from the original pilot's per-source covariance checks. Sensitivity outputs vary
peak threshold and temporal/spatial dependence scales. The guarded frequency scan
excludes ill-conditioned harmonic designs near the six-day sampling limit.

Regional benchmark recovery improves substantially, but no observed region/channel
test survives the primary multiple-testing adjustment. Uncertainty is sensitive
to block length, several injected cases overcover, and range systematics remain
unresolved. These are exploratory results on an already-inspected record. This
workflow changes no production inputs, corrections, original pilot outputs,
map gates, or 2018 processing. The new regional sampler is not automatically
substituted into the original `fit-velocity --stage pilot` path.

## Residual vertical M2/O1 experiment

The 2017 joint model keeps the existing swath/ocean-tide correction and fits
source-specific residual vertical M2/O1 height coefficients alongside a shared
regional along-flow Msf velocity harmonic and source-specific EN secular motion.
It uses the same screened radar observations for fixed-vertical joint,
azimuth-only, and residual-vertical comparisons. It does not introduce a free
secular vertical velocity or horizontal M2/O1 components.

```bash
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /Users/briel/miniforge3/envs/ice/bin/python tools/run_residual_vertical_pilot.py
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /Users/briel/miniforge3/envs/ice/bin/python tools/stress_residual_vertical_pilot.py
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 \
  /Users/briel/miniforge3/envs/ice/bin/python tools/summarize_residual_vertical_pilot.py
```

Results are written under
`.runtime/velocity-recovery-larsen/2017-residual-vertical/`. The main run uses
1,000 paired temporal/spatial bootstrap draws and 200 validation draws for 11
cases. Two additional larger-vertical stress cases use 200 draws each. The
summary preserves per-source coefficients and verifies standalone radar
prediction, checksums, source signatures, and completed draw counts.

`INTERPRETATION.md` is the result summary; `REPORT.md` contains the main tables.
The vertical terms denote residual displacement-height harmonics in metres,
while horizontal Msf coefficients denote instantaneous velocity in m/day.
Estimated vertical terms can absorb other errors with matching sampled patterns;
they do not uniquely identify ocean-tide error. Observed-data bootstrap masks
remain fixed; injections repeat the screen. The experiment changes no
production input, correction, map eligibility, or 2018 processing.


## Spacecraft order and full ENU solid-Earth tide follow-up (2017)

The [2026-10-29 experiment report](ORDER_SET_EXPERIMENT_2026-10-29.md) records
matched corrected-data fits, frequency scans, recovery tests, and nested
synthetic bootstrap calibration. The isolated outputs are in
`.runtime/velocity-recovery-larsen/2017-order-set/`; reproduction commands,
source coefficients, support, figures, and verification hashes are included.

The primary estimator fits regional satellite-azimuth Msf and a displacement
coefficient per owning burst with sign `B_later - B_earlier`. Along-flow and
residual-vertical estimates are comparisons. Full ENU solid-Earth motion is
subtracted at exact burst endpoints, and the existing CATS ocean-height
correction remains applied. Ocean loading is unavailable locally and remains
uncorrected. The order-aware, acquisition-purged screen is repeated inside
observed and synthetic bootstrap resamples.

All four ice targets pass the exploratory specified-frequency Holm test, with
along-flow amplitudes about 0.060, 0.073, 0.056, and 0.048 m/day; controls remain
consistent with zero. Recovery meets the 0.10 m/day amplitude/phase benchmark.
The uncertainty acceptance gate still fails: frozen ellipses under-cover
transferred control fields, while nested bootstrap intervals are conservative
under that generator. These results do not certify physical tidal attribution
or a production map. No 2018 observations are examined by this experiment.


## Cross-fitted two-half generator revision (2017)

The [generator revision report](GENERATOR_REVISION_2026-10-29.md) evaluates
opposite-half control predictions and spatial permutations that exhaust donor
pools before recycling. The original order/SET experiment and observed fits
remain unchanged. New results are in
`.runtime/velocity-recovery-larsen/2017-order-set-generator/`.

Four generator variants use 200 null fields each. The primary revision repeats
seven recovery cases and uses 100 nested null fields plus 20 benchmark fields
per region. It substantially reduces the earlier noise-transfer mismatch.
Target 4 retains a 10% rejection rate with a frozen observed ellipse, whereas
recomputed paired uncertainty gives 1/100 null rejections in each ice target
and none in the controls. The four-target Holm family has no false rejections
in 100 shared fields. Paired intervals remain conservative under this generator.

All 59 relevant tests pass. There are 37,200 complete recovery fits and 143,274
successful nested resamples; six rank-deficient resamples in one synthetic
null dataset are recorded and skipped by the unchanged fitter. Frozen inputs
and original experiment artifacts are checksum unchanged. No 2018 data are read.


## Combined 2017/2018 exploratory pixel maps

For configurable reruns, use the separate
[`velocity-pixel.larsen.yaml`](../../configs/velocity-pixel.larsen.yaml).
The [pixel inversion configuration guide](PIXEL_INVERSION_CONFIGURATION.md)
documents commands, retained-pair thresholds, screening, map filters and
recorded settings. This YAML is independent of stack creation.

The [pixel-map report](PIXEL_MAPS_2017_2018_2026-10-29.md) describes the full
source-wise inversion of 119 pairs on the native 200 m EPSG:3031 grid. Products
are in `.runtime/velocity-recovery-larsen/maps-2017-2018/`, including
`coefficient_maps.nc`, amplitude/phase GeoTIFFs, PNG/PDF figures, source tables,
quality flags and verification hashes.

The along-flow Msf + residual vertical M2/O1 model has separate EN secular
velocities and azimuth spacecraft-order offsets for 2017 and 2018. It fits a
common harmonic across years and keeps a direct-azimuth comparator. Every
unique source is solved separately; copied map cells inherit that source's
fit, with no spatial smoothing or regional harmonic borrowing. Both CATS
corrections remain applied, and full ENU solid-Earth motion is removed.

The 2018 projection vectors are retained despite small numerical export
differences; no 2017 empirical calibration is extrapolated. Data/geometry
support and phase-quality display filters are explicit. Pixelwise calendar
cluster covariance is diagnostic and has not inherited the regional bootstrap
calibration. These user-requested coefficient maps do not alter the original
production-map gate or assert physical-Msf detections.

64 relevant tests pass. Source replay, physical radar prediction, native-cell
replication and GeoTIFF coordinate/data checks pass. The two source packages
and all prior experimental results remain unchanged.
