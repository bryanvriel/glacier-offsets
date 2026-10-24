# Proposed enhancement: physical and empirical swath corrections

**Recorded:** 2026-10-24

**Status:** design for later implementation; no production correction enabled.

**Scope:** Sentinel-1 stack azimuth offsets, initially IW1/IW2. Range corrections require separate evidence.

This note records two complementary strategies: predicting swath-dependent
offsets from acquisition physics, and estimating relative swath offsets from
an overlap-measurement network. They should remain separate, auditable products.
An optional empirical residual fit may follow a physical prediction.

The aim is to explain and, where validated, correct a repeatable differential
azimuth component. Making the swath boundary look continuous is not sufficient
validation. Native ISCE3 matching, the pinned reference grid and raw observations
remain the baseline.

## 1. Evidence and constraints at the design date

The following are measured findings, not assumed calibration constants:

| Control | deramp=0 mean, px | deramp=1 mean, px | Quality-selected windows |
|---|---:|---:|---:|
| May 31–June 12 B→B | +0.00177 | +0.00065 | 115 / 121 |
| May–June B–A–B closure | +0.00480 | −0.00286 | 114 / 120 |
| March 14–26 A→A | −0.01736 | −0.00830 | 54 / 64 |
| March A–B–A closure | −0.01125 | −0.01283 | 50 / 56 |

Selection required native validity, burst-interior support and both swath
correlation peaks ≥0.3, with no agreement or outlier trimming. The June
order-reversing component on identical patches was about +0.026 px in both
deramp modes. Its B→B descriptive intervals were inside a declared ±0.01 px
reference band; its closure intervals were not. March A→A and closure medians
were zero, with wide intervals and retained tails. These results support an
order-associated component without identifying a hardware timing cause.

The new pairs provided 211 actual dual-supported centers and 186 supported
burst-interior centers before quality cuts. The earlier March 14–20 pair
provided 364 and 319 respectively. Denser sampling may improve coverage, but
it cannot turn unsupported footprints into observations or overlapping chips
into independent samples.

March and June used IPF 2.72 and 2.82 respectively, constant within each
triangle. Preserve processing-baseline and date distinctions in any model.
The explicitly tested analytic bistatic rank/PRI term canceled temporally in
the January audit; this did not test every timing or FM-rate effect.

The annual production plan had 61 acquisitions and 60 links forming a single
chain: zero independent cycles. Some saved production results were interrupted
or stale. A proposed solve must inspect the usable measurement graph, rather
than assuming that all planned links are valid.

Local evidence, containing full intervals, raw observations and support audits:

- [High-correlation control assessment](/Volumes/thinkplus/insar/glacier_offsets/work/larsen/experiments/spacecraft-high-coherence-20261024/README.md).
- [Earlier spacecraft-order investigation](/Volumes/thinkplus/insar/glacier_offsets/work/larsen/experiments/range2x-multipair-20261024/azimuth-spacecraft-order/README.md).
- [Stack hand-off](/Users/briel/data/insar/glacier_offsets/docs/stack-processing/HANDOFF_2026-10-24.md).

These absolute paths describe the current installation. The design does not
depend on retaining its directory names; an implementation should record
artifact identities and checksums.

## 2. Common mathematical conventions

All offsets below are later minus earlier, in original stack-reference SLC
pixels. Arrays use [azimuth, range]. A source center is the zero-based chip
start plus (window − 1)/2.

Let `o_s,ij(x)` be a measured azimuth offset in swath `s` for dates `i < j`,
and let `b_s,t(x)` be an apparent azimuth bias for acquisition `t`, expressed
on the same reference grid. The candidate additive model is:

```text
o_s,ij(x) ≈ u_ij(x) + b_s,j(x) − b_s,i(x) + e_s,ij(x)

y_ij(x) = o_2,ij(x) − o_1,ij(x)
d_t(x)  = b_2,t(x) − b_1,t(x)

y_ij(x) ≈ d_j(x) − d_i(x) + ε_ij(x)
```

The common motion term cancels to first order for genuinely corresponding
patches. Differences in scattering, finite chip support, geometry, motion
transport and peak selection can remain in `ε`. Cancellation is a modeling
assumption to check, not proof that every overlap difference is calibration.

Define triangle closure as:

```text
closure(i,j,k,x) = y_ij(x) + y_jk(x) − y_ik(x)
```

Every purely additive date-dependent `d_t(x)` cancels around an ideal
same-ground triangle. Nonzero measured closure can reject that complete
measurement model, but cannot by itself identify FM-rate error. Matching at
fixed chip centers also differs from following an advected material patch.

| | Strategy A: physical prediction | Strategy B: empirical network |
|---|---|---|
| Main input | Annotations, orbit, DEM, processing/resampling conventions | Independent IW1/IW2 overlap observations |
| Main output | Predicted apparent bias per swath/date/burst/location | Estimated relative swath term and its covariance |
| Main question | Can a specified physical effect predict the observations? | Is an additive relative calibration supported by the observations? |
| Spatial scope | A physically specified field; validation still needed | The overlap, plus any explicitly assumed continuation beyond it |
| Principal limitation | Incorrect physics, sign, mapping or double correction | Weak graphs, changing samples, non-additivity and unobserved common motion |

## 3. Shared overlap-observation stage

### Measurement contract

Use the retained aligned, unspliced complex SLC bursts. Run a small native
match in each swath at identical global chip centers with fixed physical
chip/search footprints. Preserve the production matcher and preprocessing
settings for a given comparison. Select the latest fully supported source
from input support before inspecting matching quality; a failed owner must
never fall back to a more favorable source.

The existing
[overlap diagnostic](/Users/briel/data/insar/glacier_offsets/tools/compare_swath_overlap.py)
is the prototype. Its `--acquisition-dates EARLIER LATER --compare-deramp`
route uses verified aligned dates on a saved grid and omits unrelated merged
offset comparisons. An enhancement should separate reusable measurement logic
from plotting and from the current requirement for a completed template pair.

Measurements can run once the relevant aligned dates and reference sampling
grid are available. They need not wait for every full-field match. A batch
network solve follows completion of the selected calibration measurements.

Retain, per window:

- Date pair, actual sensing times, spacecraft, IPF and relevant processing baseline.
- Global centers, source burst IDs in both swaths, chip/search/preprocessing settings.
- Geometry checks for centers and corners, input support and reasons for missing support.
- Both raw offsets and their gross/residual/total components, peak, SNR, covariance,
  finite/native-completed/search-boundary flags and failures.
- `y = IW2 − IW1`, burst-pair group, distance to source boundaries and phase-coverage diagnostics.

Keep raw observations separate from analysis selections. Report counts through
geometry eligibility, actual support, interiors, native validity and quality
cuts. Do not select for small swath disagreement, method agreement or closure.

### Sampling, aggregation and cost

Denser range centers may improve spatial and fractional-pixel coverage.
Record their locations and the resulting dependence between overlapping chips.
Keep burst-pair measurements before calculating a robust pair summary.

Changing spatial support can change a pair mean without a temporal calibration
change. Use common geographical subsets or prescribed spatial weights when
comparing scalar pair means; alternatively fit the spatial/group structure
explicitly. Diagnose range, along-track and quality dependence before
collapsing the data to one scalar per pair.

Use spatial blocks or burst groups for uncertainty, and account for groups
that share source bursts. Joint resampling across pairs should preserve
dependence from shared acquisitions and patches where possible. A bootstrap
with one group must not produce an effectively infinite fitting weight.
Require a documented variance floor/model-error term or mark the estimate
insufficiently constrained.

Few windows do not guarantee a cheap stage. The prototype paid substantial
full-burst I/O, geometry-sampling and native-initialization costs. Measure
wall time, bytes read and parent/worker RSS. Cache verified geometry and
consider cropped input reads only after demonstrating unchanged native
observations, support and global centers.

## 4. Strategy A — physical forward modeling

### Predict before fitting

For each selected date, swath and source burst, evaluate candidate physical
effects at the observed ground patches. Initial candidates include residual
azimuth timing/bistatic effects and azimuth FM-rate mismatch. Preserve separate
component fields, their units and their provenance.

The physical components are described in
[ESA's ETAD processing documentation](https://sentiwiki.copernicus.eu/web/s1-processing).
The implementation still needs the processing-baseline and mapping audit below.

Implementation requirements:

1. Audit what the actual COMPASS radar alignment/resampling path already applies.
   Do not infer applied corrections solely from available configuration flags.
2. Use the original acquisition annotations, orbit and DEM/height convention.
   Match the actual IPF processing assumptions and coefficient-selection rules.
3. Map reference centers through retained native resampling offsets to the
   appropriate acquisition/burst coordinates. Do not add a relative-burst shift
   twice; COMPASS offsets already use secondary-local coordinates here.
4. Preserve physical timing fields in seconds and separately derive apparent
   biases in reference SLC pixels. Establish the sign by tracing the actual
   coordinate mapping and by controlled tests; do not assume that a tagged-time
   correction and an observed image bias have the same sign.
5. Predict each swath's pair contribution and the observed contrast:

```text
q_s,ij(x) = b_s,j(x) − b_s,i(x)
y_pred,ij(x) = q_2,ij(x) − q_1,ij(x)
residual_ij(x) = y_observed,ij(x) − y_pred,ij(x)
```

The installed s1-reader exposes candidate implementations, including
`bistatic_delay` and `az_fm_rate_mismatch_from_llh`. Their availability is not
validation of their suitability for these historical products. Audit numerical
sampling, coefficient conventions, software version and processing baseline.

### Validation and possible application

Compare predictions with the retained observations at window and burst-group
level. Examine amplitude, sign, range/azimuth structure, spacecraft order,
IPF regime and residuals on held-out dates/pairs. Report all quality populations.
A fitted scale factor or timing constant is a separate calibrated parameter,
not an independent physical prediction.

A successful model should predict reproducible structure beyond a fitted
pair mean and improve held-out residuals without changing support to obtain
agreement. Retain rejected/near-null components as diagnostic results.

The physical model can predict both individual-swath contributions, including
a common component that overlap differences cannot observe. Applying that
common component requires independent physical or external validation.
Overlap agreement alone validates only the differential prediction.

A post-match layer is appropriate for a validated additive offset bias.
Effects that change image geometry, defocusing or peak selection may require
tests with corrected resampling and re-matching. A post-match shift cannot
repair a wrongly selected correlation peak.

## 5. Strategy B — empirical overlap-network calibration

### Network and identifiability

For a scalar relative swath term, assemble `y ≈ H d`, where each row of
`H` has −1 at the earlier date and +1 at the later date. Use weighted robust
least squares or a documented probabilistic equivalent.

For `n` acquisitions, `m` usable links and `c` connected components, the
number of independent graph cycles is `m − n + c`. A tree has none and can
fit every edge exactly. Robust loss alone cannot identify a bad edge in that
unconstrained model. Additional overlap-only links can supply cycles without
requiring additional full-field production matches.

Assess connectivity after quality/support selection and weighting. Fix one
relative datum per connected component, or use an equivalent constraint.
Changing that reference adds a constant to `d` and leaves within-component
pair differences unchanged. Relative levels between disconnected components
are unobservable without an explicit prior or additional measurements.

Short, strong links can constrain a weak long-interval pair only if its
endpoints are connected through sufficiently informative alternative paths
and the additive model is supported. Nominal connectivity through a weak
bridge does not give a precise correction.

### Candidate models and priors

Start with the simplest identifiable candidates and compare predictive
performance:

| Model | Purpose and constraint |
|---|---|
| Spacecraft term | `d_t = μ_spacecraft(t)`; for A/B only their difference is identifiable |
| Per-acquisition term | Free `d_t` within a sufficiently constrained component |
| Hierarchical term | `d_t = μ_spacecraft(t) + a_t`, with explicit residual constraints/shrinkage |
| Spatial/burst extension | Used only when reproducible spatial structure and adequate data support it |

Allow processing-regime effects where warranted. Spacecraft, date and IPF can
be confounded. A plot of fitted terms clustering by spacecraft, especially
under a spacecraft prior, does not select or validate the physical explanation.

A prior may constrain gaps or acquisitions with weak measurements. Store
which corrections are data-constrained, weakly constrained or prior-dominated.
Do not silently substitute zero for an unsupported correction.

For burst-dependent modeling, retain the actual pair of source bursts.
An observation group `g` may involve different IW1/IW2 burst indices:

```text
d_t,g = b_IW2,t,k2(g) − b_IW1,t,k1(g)
```

Along-track variation may reflect physical timing, geometry, texture,
fractional-pixel matching bias or changing sample selection. It does not
automatically establish a burst-cycle calibration term. Avoid freely adding
one parameter per acquisition/burst until rank, support and held-out tests
justify the additional freedom.

### Solve diagnostics and uncertainty

Retain observed and fitted edges, residuals, weights, robust-loss influence,
components, graph rank/cycle counts, priors and covariance. Cycles expose
inconsistency; one inconsistent triangle does not uniquely identify its bad edge.
Preserve raw closure diagnostics rather than reporting only the fitted,
necessarily additive network.

Reserve redundant edges/dates for validation where the remaining graph
permits it. Compare against the uncorrected baseline and simpler models.
A flexible solve fitting its own training edges is not validation.

For pair `i,j`, propagate the full fitted covariance:

```text
Var(d_j − d_i) = C[j,j] + C[i,i] − 2 C[i,j]
```

Record model error and uncertainty in priors as well as measurement error.
The overlap fit and production offsets can share imagery, so adding their
variances as if independent requires an explicit approximation or a justified
joint covariance treatment.

## 6. Shared correction application and geocoding

### Signs and the unobserved common component

Store a field named, or explicitly defined as, `bias_to_subtract`:

```text
corrected_total_s,ij = raw_total_s,ij − bias_to_subtract_s,ij
```

For an empirical relative contrast `Δ = d_j − d_i`:

| Allocation | IW1 bias to subtract | IW2 bias to subtract |
|---|---:|---:|
| IW1 anchored | 0 | Δ |
| Symmetric | −Δ/2 | +Δ/2 |

Subtracting the same Δ from both swaths leaves their difference unchanged.
The allocation convention changes the common azimuth level of the corrected
pair. It is distinct from the harmless network reference-date constraint.
Overlap data alone cannot choose the absolutely correct allocation.
Use an independently constrained motion reference if absolute calibration is
required; grounded or slow ice is not automatically stationary.

For validated physical predictions, individual `q_s,ij` may be used directly.
If only the differential component is validated, expose a relative-only
allocation mode and label the remaining common uncertainty.

An empirical overlap estimate does not establish that its value applies
uniformly across the entire swath. Any continuation outside the measured
strip is an explicit spatial-model assumption requiring validation. Do not
default to blending, tapering or smoothing merely to hide a seam.

### Ownership and mixed footprints

For a pure-swath footprint, apply the relevant field at the original radar
center with exact source ownership. Preserve original source IDs and masks.

The existing `seam_crossing` flag covers chip/search crossings of source
boundaries, including along-track burst boundaries. Distinguish cross-swath
mixing from a same-swath burst join. A center owner alone does not describe
the contents of the full matching footprint.

A mixed-swath correlation peak is nonlinear. Pixel-area or amplitude-weighted
averaging of swath corrections is not validated by the current experiments.
Initially flag such windows as requiring special treatment; make their
correction availability explicit. Validate any effective weighted correction
through controlled perturbations/native matching before enabling it.

### Derived radar fields and map-space equivalence

Preserve raw `gross`, `residual` and `total = gross + residual`. Write an
explicitly derived corrected total and its correction/uncertainty layers;
do not relabel it as the original native total.

Prefer correction before geocoding. Convert pixel bias to meters using the
actual source-center azimuth ground spacing, and use exact source intervals
for any velocity correction.

Let `G` be the original geocoding operator and `a` the source-center meters
per azimuth pixel. With unchanged mapping weights and masks:

```text
G(a * corrected_total) = G(a * raw_total) − G(a * bias_to_subtract)
```

Thus post-geocode application can be equivalent if it geocodes the
meter-valued correction with precisely the original operator. Applying a pixel
Δ directly to a displacement raster is dimensionally wrong. A categorical
source-index GeoTIFF alone is insufficient where linear interpolation mixes
sources or the physical scale varies.

Amplitude overlap measurements avoid interferometric phase wrapping but
retain finite-search and peak ambiguities. ESD/NESD can incorporate stationary
control or motion information, as illustrated by
[this TOPS ice-velocity study](https://www.mdpi.com/2072-4292/12/12/2014).
Avoid claiming universal superiority for this measurement merely because
common motion approximately cancels.

## 7. Proposed integration, artifacts and cache behavior

The architecture should support:

```text
aligned acquisitions + pinned geometry + matching grid
    -> independent overlap observations
    -> physical predictions and their validation
    -> optional empirical fit of observations OR physical-model residuals
    -> explicit per-swath bias-to-subtract fields
    -> versioned corrected radar/map products
```

Raw matching and raw geocoding may retain their current execution path.
Correction application is a separate, opt-in path after its selected model
and observations are ready. No correction strategy is a production default
at this design stage.

### Responsibilities in the current code

| Existing component | Enhancement responsibility |
|---|---|
| [Overlap diagnostic](/Users/briel/data/insar/glacier_offsets/tools/compare_swath_overlap.py) | Extract reusable observation/selection logic; retain a diagnostic CLI |
| [Stack workflow](/Users/briel/data/insar/glacier_offsets/src/glacier_offsets/stack_workflow.py) | Schedule independent observation, prediction, solve and derived-product stages |
| [Configuration](/Users/briel/data/insar/glacier_offsets/src/glacier_offsets/stack_config.py) | Validate explicit strategy, graph, sampling, model and allocation policies |
| [Status inspection](/Users/briel/data/insar/glacier_offsets/src/glacier_offsets/stack_status.py) | Distinguish unavailable, incomplete, prior-dominated, current and stale calibration |
| [Geocoding](/Users/briel/data/insar/glacier_offsets/src/glacier_offsets/geocode.py) | Read a derived field provider and include correction identity in corrected-map identity |
| [Product identities](/Users/briel/data/insar/glacier_offsets/src/glacier_offsets/products.py) | Checksums, complete specifications and atomic publication |
| [Stack locking](/Users/briel/data/insar/glacier_offsets/src/glacier_offsets/stack_lock.py) | Prevent incompatible concurrent publication and inventory/layout changes |

Candidate new modules are `swath_observations`, `swath_physics`,
`swath_network` and `swath_corrections`. These are proposed responsibilities,
not implemented APIs or commands.

An illustrative artifact layout, also not a current schema:

```text
calibration/swath/
  observations/<measurement-id>/<pair>/...
  physics/<prediction-id>/...
  fits/<solve-id>/...
  corrections/<correction-id>/<pair>/radar/...
  corrections/<correction-id>/<pair>/geo/...
```

Keep large arrays/tables in dedicated artifacts. The stack manifest should
index their paths, identities, status and selected active version rather
than duplicating every measurement or silently overwriting earlier tables.

Identity inputs must include the relevant source/image/geometry checksums,
grid and footprint definitions, preprocessing, selection and spatial weights,
physical model/version/baseline, graph and component membership, priors,
solver/loss settings, covariance method, allocation, mixed-footprint policy,
base match identity and software provenance. Record available information
separately from an estimate's uncertainty and correction eligibility.

Restart and invalidation rules:

- New/changed observations invalidate dependent fit comparisons, solves and
  derived corrections while preserving raw matches. Physical fields remain
  reusable unless their inputs, model or requested sampling coordinates change.
- A model or allocation change invalidates its derived products, not native matching.
- A corrected geocode identity includes the correction identity, original
  mapping settings/mask and base match identity. Raw-map cache reuse remains valid.
- Adding acquisitions may alter old corrections when new redundant information
  or priors change the solve. Changes are not guaranteed to be small. A new leaf
  in an unconstrained tree need not change old pair differences at all.
- Interrupted stages retain diagnostics, publish no false completion identity,
  and never replace the last complete calibration version.

A hybrid must state whether the empirical fit uses raw observations or
`observed − physical prediction`. Sum physical and residual contributions
once, with compatible signs, spatial coordinates, allocation and uncertainty.

## 8. Implementation sequence and acceptance gates

| Milestone | Deliverable | Gate before proceeding |
|---|---|---|
| 1. Observation service | Reusable sparse matching, support/quality ledger and benchmark | Native controls agree at fixed centers; failures retained; source data unchanged |
| 2. Physical diagnostics | Separate component predictions and held-out comparisons | Verified sign/mapping/baseline; no double correction; null cases reproduced |
| 3. Empirical diagnostics | Cycle-aware graph audit, candidate fits and covariance | Identifiability explicit; tree/gap/prior cases handled; predictive validation reported |
| 4. Derived radar layer | Immutable raw outputs plus explicit bias/corrected totals | Allocation signs, ownership, mask and uncertainty behavior verified |
| 5. Corrected geocoding | Separate corrected-map identity and products | Correct units and mapping; no false equivalence for mixed-source interpolation |
| 6. Real-data evaluation | More independent acquisition episodes and regimes | Improvement supported beyond training means and cosmetic seam continuity |

Required verification should target scientific contracts:

- Known native translations with injected differential swath/date biases:
  recover the observable contrast without removing known common motion.
- Reference-date changes leave within-component pair corrections unchanged.
- A tree fits arbitrary edges but supplies no additivity test; an inconsistent
  redundant cycle remains visible; weak bridges and disconnected components
  expose uncertainty and prior dependence.
- Static and date-additive physical fields cancel in ideal closure; a
  pair-specific perturbation does not.
- Pixel/time/meter conversion and source-burst mapping preserve sign and units.
  Validate field sampling at centers and corners, not only scene means.
- Changing density or spatial coverage does not silently inflate independent
  sample counts or change the analysis population without recorded attrition.
- Injected failed/native-boundary/low-quality observations remain inspectable
  and cannot become valid merely through a correction.
- Mixed-swath, along-track-join and pure footprints are distinguished.
  Preserve native peak failures rather than treating a scalar shift as repair.
- Radar correction followed by mapping equals subtraction of the identically
  mapped meter correction where the operator/mask assumptions hold.
- Append, re-solve, interruption, checksum changes and restart preserve raw
  products and correctly invalidate only dependent derived versions.

Real-data reports must show uncorrected and predicted/corrected observations,
all selection counts, burst/spatial structure, uncertainty, raw closures,
held-out performance, spacecraft/IPF/date dependence and actual runtime/RSS.
Retain weak and null controls. Agreeing swaths do not establish absolute motion
accuracy without independent validation.

Decisions still requiring evidence or an explicit policy include the physical
components to enable, an independent absolute reference or relative allocation,
the spatial continuation beyond the overlap, model complexity/shrinkage,
mixed-footprint treatment and thresholds for correction eligibility. Keep all
choices in the specification; do not infer them from favorable output maps.

## 9. Physical-method references

- [ESA/Copernicus Sentinel-1 processing and ETAD description](https://sentiwiki.copernicus.eu/web/s1-processing).
- [ETAD Algorithm Technical Baseline Document, issue 2.1](https://sentinels.copernicus.eu/documents/247904/4629150/Sentinel-1-ETAD-Algorithm-Technical-Baseline-Document.pdf).
  Consult applicable revisions and the historical IPF conventions before implementation.
- [ESA notice on the 2024 FM-rate coefficient-selection update](https://sentinels.copernicus.eu/-/copernicus-sentinel-1-etad-products-new-processing-baseline).
- [A Network-Based Enhanced Spectral Diversity Approach for TOPS Time-Series Analysis](https://ieeexplore.ieee.org/document/7637021/).
- [Improved Ice Velocity Measurements with Sentinel-1 TOPS Interferometry](https://www.mdpi.com/2072-4292/12/12/2014).
