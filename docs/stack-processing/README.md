# Reference-geometry stack validation

The [October 22 burst-boundary investigation](boundary-investigation/README.md)
uses corrected real three-date products and controlled native matching to trace
the January 18–24 stripes to edge-amplitude notches at the SLC ownership cuts.
Interior ownership is now the stack default, with a versioned pin and an
[isolated full-AOI validation](boundary-investigation/interior-validation.json).
The original stack remains unchanged. The historical input-access notes below
describe earlier checks, not current local availability.

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
