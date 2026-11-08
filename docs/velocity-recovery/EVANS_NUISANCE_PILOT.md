# Evans matched nuisance pilot

These tools compare baseline, spacecraft-order displacement, full ENU solid-Earth
tide, and both on the existing Evans source observations. They read the original
GPS-audit source ledger and horizontal package; no rematching is required.

The preparation tool selects unique sources in the four original target regions
and two control regions, retaining the original 12 pilot targets as a separate
reporting cohort. It checks native RA against the prior EN-to-RA round trip to
10 micrometres, allowing the package's float32 storage precision. The correction
worker runs in an interpreter with PySolid, evaluates the exact owning-burst
endpoints, and checks its spatial interpolation against direct point evaluations.

Run from the code repository in the numerical `ice` environment:

```bash
export PYTHONPATH=src
export MPLCONFIGDIR=/tmp/evans-mpl-cache
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

python tools/prepare_evans_nuisance_pilot.py \
  /Users/briel/data/glaciers/antarctica/evans \
  --output .runtime/evans-nuisance-pilot \
  --solid-earth-python /Users/briel/miniforge3/envs/isce3/bin/python

python tools/run_evans_nuisance_pilot.py \
  /Users/briel/data/glaciers/antarctica/evans \
  --output .runtime/evans-nuisance-pilot --draws 500 --workers 4

python tools/assess_evans_nuisance_pilot.py \
  --output .runtime/evans-nuisance-pilot
```

On another machine, use its Evans workspace, output directory and PySolid
interpreter. The source workspace must have the existing `gps_informed_audit`
and `velocity_fit_robust` products under `work/stack/packages`.

Preparation checks input/code identities before reuse; the solid-Earth worker
checkpoints incomplete corrections. Analysis directories are keyed by the
observation/correction hashes, numerical source hashes and settings. Completed
per-field bootstrap files can be reused within an identical analysis recipe.
`latest_analysis.json` identifies the completed analysis. These tools expect a
single writer per experiment directory.

## Interpretation

The four ablations use the same observed support and original pooled RA
covariance. Quality-dependent weighting is intentionally absent from this
comparison. Their coefficient changes therefore isolate the fitted nuisance
terms rather than changes in quality weights. All variants retain existing
CATS range corrections; only the solid-Earth variants subtract the body tide.
Ocean loading is not supplied.

The provisional common screen uses peak >=0.1 and acquisition-purged,
two-component order/secular residual limits of 2/5 m. Sources need 20 retained
pairs. The same screen is recomputed from the corrected observations in every
bootstrap draw; it is shared across ablations. Source-cohort eligibility is
selected on the full observed record and is retrospective.

Direct satellite-azimuth Msf is primary. The joint RA along-flow comparison
assumes zero remaining vertical motion. The assessment also fits azimuth-only
along-flow and joint residual vertical M2/O1 models on the same support. Fitted
vertical coefficients are sensitivity diagnostics, not measurements of ocean
tide error. Annual fits test the stability of nuisance coefficients and motion
without supplying calibrated annual intervals.

Thirty-day temporal and 3 km physical spatial bootstrap blocks give conditional
uncertainties. Target flow axes and screening are recomputed, and observations
retain their original acquisition/order identities. This does not calibrate
interval coverage or replace an acquisition-shared stochastic error model.

Control-noise diagnostics use acquisition-purged predictions of secular/order
motion without removing control harmonics. Peak-supported held-out tails are
retained even where the supplementary fit screen would reject them. The robust
covariance estimator winsorizes those tails, so inspect `control_tails.json`
and source/block support in `control_pair_medians.csv` as well as pooled SDs.
More sources in the same two control regions do not establish geographic
transferability to target ice.

Read `ASSESSMENT.md` first, then `REPORT.md` and the analysis CSV/JSON files.
The experiment does not update production corrections, masks or fits and does
not establish a recovery, false-positive, uncertainty-coverage or map gate pass.
