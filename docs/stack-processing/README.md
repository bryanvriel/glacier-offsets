# Reference-geometry stack validation

Start with the [October 24 hand-off](HANDOFF_2026-10-24.md) for the current
Larsen configuration, selectable merge/matcher strategies, completed diagnostics,
and unresolved range-swath differences. The sections below retain the earlier
validation history; their dated production recommendations are historical.

The [proposed swath-correction enhancement](SWATH_CORRECTION_ENHANCEMENT.md)
records physical forward modeling and empirical overlap-network calibration as
separate strategies, with application conventions, validation gates and an
implementation sequence. It is a design note; no production correction is enabled.

The [October 22 burst-boundary investigation](boundary-investigation/README.md)
uses corrected real three-date products and controlled native matching to trace
the January 18–24 stripes to edge-amplitude notches at the SLC ownership cuts.
Interior ownership is now the stack default, with a versioned pin and an
[isolated full-AOI validation](boundary-investigation/interior-validation.json).
The original stack remains unchanged. The historical input-access notes below
describe earlier checks, not current local availability.

## ISCE3 merge-first compositor parity (October 24)

`stack.processing: isce3_merge_first` selects the ISCE3-backed compositor in
`stack_mosaic`. It was checked against the frozen GDAL merger
(`tools/reference_stack_merge.py`) on the six-date Larsen pilot using
`tools.compare_stack_mosaic`. The production root was read only. Its layout
checksums still match their records, and no watched file was modified. A
compact record is in [mosaic-parity.json](mosaic-parity.json). The full report
and retained layouts are in
`/Volumes/thinkplus/insar/glacier_offsets/work/larsen/experiments/isce3-mosaic-parity-20261024/`.

- **Layouts.** For both swath policies, source placements, ownership windows,
  cuts, the radar grid, `source.tif` and all four geometry VRTs are identical.
  The midpoint layout also reproduces production's ownership raster and
  geometry exactly. The `source.tif` *bytes* differ even though the pixels are
  equal. Existing layouts are reused, so their byte-based identities stay the
  same; a newly written layout gets a new identity.
- **Dates.** Under both policies, all 12 date compositions (7336 × 44088) have
  zero differing complex samples, external-mask values or validity samples,
  with identical counts, including unsupported regions. The midpoint SLCs from
  both implementations are byte-identical to the checksums production recorded
  before releasing its merged SLCs. Native CPU Ampcor can memory-map every
  output.
- **Matching.** Both pairs were matched with the unchanged ISCE3 matcher and the
  production settings (8 workers, `deramp: 0`, zoom 16, zero gross). Input-
  validity, validity, ownership, seam and center grids are identical.
  - January 19–25: the reference, ISCE3 and replay runs are equal in every
    dataset.
  - January 1–7: the ISCE3 run equals the replay control bit for bit. Both
    differ from the first reference run in 28 azimuth and 47 range windows of
    77,634, each by exactly 1/64 pixel.
- **Native run-to-run variation.** This is independent of the compositor. It
  occurs on byte-identical inputs, in low-SNR windows (median SNR 2.3, against
  10.5 for all valid windows). The saved production offsets differ from
  today's runs in 45–138 windows per component, by up to 29 pixels where a
  different near-tied peak is chosen; validity is unchanged. This existing
  matcher behavior should be investigated separately; agreement filtering was
  not applied.
- **Cost.** Alternating controlled runs on one date, writing to the internal
  disk: median 13.3 s for the reference and 13.9 s for ISCE3. Peak RSS was
  1.13–1.23 GB for the reference and 1.28–1.30 GB for ISCE3, about 0.1 GB more
  for the ISCE3 library. That is below the 2.6 GB of a single full-scene
  complex array. Gate-run timings on the exFAT volume include write-back of
  retained outputs. The first draft of the compositor kept an SLC and its
  external `.msk` open as two writers, which made closing take about 260 s on
  exFAT. It now writes the mask in a separate pass, and a regression test
  guards against concurrent SLC/mask writers.

After review, both merge-first selectors were cut over to the ISCE3
compositor, with `isce2_merge_first` kept as a deprecated alias. The in-source
GDAL merge-first path was removed; its frozen copy remains only in
`tools/reference_stack_merge.py` for the parity tests and
`tools.compare_stack_mosaic`. Policy records and cache identities are
unchanged, so existing stacks need no rebuild. Production configurations and
completed pair products were not changed.

Implementation follows M9–M12 in `plans/stack-processing.md`. Existing pair
configuration, output paths and scientific cache specifications remain intact.

The new automated tests cover metadata-only inventory, recursive SAFE scanning,
precise-orbit selection, inclusive UTC filters, explicit/pinned references,
AOI/KML eligibility, partial coverage, duplicate dates, mixed groups, exact
12-day limits, nearest-N graphs and explicit CSV overrides. They check independent
YAML inheritance and initialization, with read-only planning and validation.

Scientific tests use three synthetic dates (January 1, 4 and 9, 2020), two
overlapping bursts on a 320 by 384 radar grid, and translations (azimuth, range)
of (1, -2), (0, 0) and (4, -5) pixels. The middle date pins the geometry. Tests
compare native matching against direct native controls using identical SLCs and
centers, including the non-reference January 1–9 pair. They check sign, gross plus
residual accounting, unequal exact burst intervals, physical scaling, formal
uncertainties, seams, identical map grids, empty-support fields, fixed overlap
ownership, isolated holes and the stack-reference orbit used by velocity priors.
COMPASS preparation/alignment are injected with synthetic SLC/geometry products;
these tests do not establish real COMPASS registration accuracy.

Incremental tests append partial coverage and insert a date, recompute the graph,
retain inactive pair products, compare checksums and nanosecond modification
times, inject date/pair failures and interruptions, test forced invalidation,
recover a reference pin from committed geometry, and verify successful retries.

Run the tests using the installed native CPU environment:

```bash
PYTHONPATH=src:.runtime/deps MPLCONFIGDIR=.runtime/matplotlib OMP_NUM_THREADS=2 \
  /Users/briel/miniforge3/envs/isce3/bin/python -m unittest discover -s tests -v
```

Executed validation and synthetic timing/storage/memory measurements are recorded
in [validation.json](validation.json). RSS values are process lifetime high-water
marks; parent and child/native workers are reported separately. Synthetic sizes
and timings are not representative of full SAFE processing.

## Outstanding real-data gates

On October 22, 2026, `/Volumes` contained only `Macintosh HD`. The workspace's
`data` link targets `/Volumes/TU200Pro/insar/glacier-offsets/data/`, which was not
available. Neither the two-date PIG check nor the required real three-date zero
and velocity-guided runs could be executed. Once the data are available, run
`stack plan`, `stack validate` and the stages with `configs/stack.pig.yaml`.
A real stack with at least three acquisitions must include a non-reference pair;
inspect date validity masks, offsets, stationary terrain where available and seam
gradients, and record RAM, disk use, timings and restart behavior. Velocity
validation also requires declared local velocity component sources/conventions.
These gates remain open. GPU checks require supported CUDA hardware and were not
run on this CPU environment.

## macOS native memory mapping fix

After connecting TU200Pro, the user's three-date PIG run (January 12, 18 and 24)
completed reference/date preparation for three bursts and reached matching, then
aborted with `CPLVirtualMemNew() unsupported on this operating system / configuration`.
The inspected native CPU controller does not pass `useMmap` to its image reader;
it always requests a GDAL memory map. With internal TIFF masks, cache flushes can
interleave mask blocks and image strips, preventing direct mapping. The generic
GDAL virtual-memory fallback is unavailable in this macOS build. A local 2048 by
1024 complex raster reproduced the exact error with an internal mask, while an
external mask mapped successfully under the same cache pressure.

Date assembly now uses an external `slc.tif.msk`. Its storage identity is versioned
separately from reference preparation and alignment, so `stack run` automatically
rebuilds old merged dates and their dependent pairs without rerunning COMPASS or
changing the pinned frame. A Python mapping probe now reports unsupported CPU
inputs as worker exceptions before the native bare throw can abort the worker.
Regression tests force mid-write cache flushes, check direct memory mapping and
explicit masks, verify storage migration reuses preparation, and exercise an
unmappable compressed input through the native worker.

The app was denied direct access to the external SLC by macOS (`Operation not
permitted`), including outside the execution sandbox. The fix is validated locally;
the user's real stack still needs to be rerun to close the real-data gates.

## Isolated zeros incorrectly treated as missing support

Inspection of the locally copied January 18–24 pair found all 23,607 raw offsets
finite, but only 5,846 input-valid windows. Of the rejected windows, 1,092 had
existing normalized correlation peaks above 0.2 and SNR above 10. Twelve sampled
windows failed because of one or a few isolated masked pixels, rather than
missing image regions. At radar line/sample `(613, 21987)`, the January 12 source
contains one isolated zero in an otherwise finite 3×3 neighborhood. January 18
and January 24 contain nonzero samples there, but the reference ownership raster
marked that pixel unowned, causing date assembly to zero and mask both values.

The stack writer's per-pixel `abs(SLC) > 0` test conflicted with the matcher's
established connected-zero-fill rule. Reference ownership and date validity now
use that same rule, retaining isolated zeros while excluding explicit invalid
masks, nonfinite samples, and connected zero-fill components seeded by a 3×3
zero region or an image boundary. Genuine isolated holes remain invalid when
identified by an explicit mask. The broad seam flag was also inflated by the
incorrect unowned pixels; it is not solely a geographic burst-boundary flag.

A read-only audit using the existing prepared reference/date bursts and the
unchanged radar layout recovers 17,141 windows: corrected eligible support is
22,987/23,607, with no previously input-valid windows lost. All 1,092 of the
aforementioned higher-correlation rejected windows recover support. The audit
changes 10,598 ownership pixels, including 10,028 previously unowned pixels.
See [validity-audit.json](validity-audit.json). These are support counts, not a
claim that all recovered offsets are accurate; corrected SLCs and offsets must
be recomputed. Artifacts at genuine burst joins require separate reassessment.

The ownership method is now `stack-reference-support-connected-zero-fill-v2`.
The old pinned frame cannot be changed in place, including with `--force`.
Use the provided fresh-root configuration:

```bash
glacier-offsets stack run configs/stack.pig.validity.yaml --backend cpu
```

It writes to `work/pig/stack-validity` and preserves the old stack for comparison.

## Restart verification

Stack product verification shares an invocation-local checksum cache across
reference, alignment and assembly checks. An unchanged file is hashed once even
when it belongs to multiple manifests or is checked by both the workflow and
its preparation functions. Every lookup checks device/inode, file type, size,
mtime and ctime; changed files are rehashed and changes during hashing are
rejected. Missing files still require a matching cleanup journal. The cache
does not persist between commands, and directory hashes are not cached by parent
timestamps. Raw input verification retains its existing behavior. Logs report
acquisition progress; catalog metrics record product hash counts, bytes and hits.

## CPU burst matching workers

Stack CPU matching now uses `OMP_NUM_THREADS` as its maximum spawned process
count within a date pair, capped by bursts with shared eligible support. Each
child has one OpenMP/BLAS thread and is replaced after four native calls. There
is no nested process pool. Stages, dates and pairs stay sequential, and GPU
matching retains its existing single-worker path. Workers import pinned FFTW
wisdom from the stack; see below.

Full-grid zero/constant matching retains raw invalid-window results. Unowned
or unavailable regions use a shared queue of tasks, each with at most half the
average worker window count. Idle workers consume this queue without assigning
its workload to the first burst. Real bursts retain their individual lanes and
one-call-at-a-time limit. Segment metadata identifies shared work separately and
records native elapsed time. This scheduling-only change keeps completed offsets
reusable and leaves scientific ownership, chip coordinates and masks unchanged.

[Native scheduling tests](../../tests/test_stack_matching.py) compare zero and
constant gross offsets, edge-bounded constant offsets, and fragmented varying
priors against the prior direct/row-segment matcher on the same synthetic
three-date SLCs. One- and two-worker outputs agree in sparse coordinates, raw
gross/residual/total, SNR, correlation, covariance, masks, seam flags and exact
intervals (floating arrays within `rtol=atol=1e-5`). The known non-reference
translation remains (+3, -3) pixels. Tests exercise distinct worker PIDs,
single-thread limits, worker recycling, partial/empty coverage, GPU routing,
child exceptions, interruption cleanup, retry, and unchanged cache reuse when
only the process budget changes. These tests establish numerical agreement on
the synthetic fixture, not full-size speedup, aggregate RAM bounds or GPU
numerical equivalence.

Stack matching uses the v8 implementation identity, refreshing older matching
and downstream products while reusing valid reference geometry and acquisition
SLCs. Completed products do not depend on the requested worker count.

## Pinned FFTW wisdom (October 24)

Repeated CPU matching of one pair on byte-identical merged SLCs differed by
1/64 px in roughly 0.1% of valid, low-SNR windows; input-invalid windows could
jump by up to about 33 px. The cause is ISCE3's compiled-in `FFTW_MEASURE`:
concurrently started workers time FFT algorithms on a loaded machine and pick
different, differently rounding plans. The investigation report and scripts
are on branch `claude/quirky-solomon-762f26` (`docs/ampcor-determinism/`).

Stack matching now measures every chunk shape once in a quiet process and
pins the result in `<stack>/.cache/fftw/<key>.wisdom`, which workers import
without republishing (see [README_DETAILED.md](../../README_DETAILED.md)).
Real-data check on the Larsen 2017-01-01_2017-01-07 pair (77,634 windows,
`deramp: 0`, zoom 16, zero gross, regenerated byte-identical merged SLCs),
through the unchanged production scheduler with one stack wisdom file:

| Run | Workers | Wall (s) | Equal to first run |
| --- | ---: | ---: | --- |
| pinned 1 (measures the file) | 8 | 96 | — |
| pinned 2 | 8 | 93 | residual, SNR, peak, covariance, validity bitwise |
| pinned 3 | 4 | 116 | residual, SNR, peak, covariance, validity bitwise |

Every segment imported the file without measuring new plans. Against runs
with other plans (the investigation's unpinned and separately pinned runs),
70–78 valid windows differ, each by exactly 1/64 px. Wall times include a
concurrent test run. The wisdom enters no cache identity, so completed pairs
are kept.

## Range-2x matching option (October 24)

`matching.range_oversampling: 2` turns the validated range-2x preprocessing
prototype (`tools.compare_merged_preprocessing`) into a stack matching option;
see [README_DETAILED.md](../../README_DETAILED.md#range-2x-matching). The
production implementation interpolates complete merged images instead of
retained strips, so its Fourier lengths differ from the prototype's.

Real-data check on the Larsen 2017-01-19_2017-01-25 pair (77,634 windows,
zoom 16, zero gross, midpoint layout). Both merged SLCs were rebuilt from the
retained bursts and matched the production release checksums.

| Comparison | Result |
| --- | --- |
| Input support, validity vs the prototype | identical; 69,534 valid windows, no transitions |
| Total offsets vs the prototype | 99.97% (azimuth) and 99.85% (range) bit-identical; the rest differ by exactly 1/64 px, mean 0 |
| Range shift vs the `deramp: 0` production pair | mean +0.040 px, as for the prototype |
| Repeated runs with one stack wisdom file | bit-identical (three runs, inputs on either disk) |
| Cost, 8 workers, inputs on the internal SSD | 164 s (33 s preparation, 131 s matching) against 93 s for `deramp: 0`; peak worker RSS 1.2 GB against 0.85 GB |
| Preparation with inputs on the external volume | about 190 s per date, limited by sustained writes of about 20 MB/s |

The last row motivated `storage.scratch_directory`. A synthetic known-shift
test ([test_range_oversampling.py](../../tests/test_range_oversampling.py))
checks that range-2x recovers a 0.3 px range shift to better than 0.02 px while
plain magnitude matching is pulled more than 0.1 px toward whole pixels.

## Disposable merged SLCs

The default `storage.keep_merged_slcs: false` retains per-burst aligned SLCs,
COMPASS reference products/geometry and provenance. `stack run` materializes a
merged date at its first needed match and releases its TIFF/external mask after
all planned matching consumers have committed offsets. Each date is assembled
once within the current run; failures keep SLCs needed by unsuccessful consumers.
Later matching requests can restore released caches from retained aligned bursts.
Small nearest-sample previews support diagnostics after cleanup. Cleanup journals
distinguish intentional release from missing/corrupt products, including partial
unlink operations interrupted before the final checkpoint. Read-only status
continues to report completed scientific stages after cache release.

[Storage tests](../../tests/test_storage.py) cover native pair and stack matching,
geocoding and diagnostic plots after release; lazy first use and final-consumer
release; unchanged retries without assembly/matching; failed consumers; new pairs;
retention overrides; safe file boundaries; and interruption during cleanup without
invalidating committed offsets. `storage.keep_merged_slcs: true` retains merged
SLCs. Explicit `assemble` commands still materialize them for inspection.
