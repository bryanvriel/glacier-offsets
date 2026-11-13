# Azimuth projection and along-flow Msf reliability

With one Sentinel-1 track, along-flow Msf is identified from azimuth offsets
alone. How reliable it is depends on the azimuth projection of the flow
direction, *p*. Noise grows as 1/|p|. Any cross-flow or systematic azimuth
term at the Msf frequency grows faster, and the reported uncertainties
cannot see it. Where |p| is below about 0.3, a block's along-flow amplitude
mainly reflects the assumption that Msf motion is purely along-flow, not
the data.

## Why only azimuth carries Msf

Each track is acquired at a fixed solar time. The Msf frequency is S2 − M2, and
S2 has a constant phase at that time, so M2 aliases exactly onto Msf. Range sees
vertical motion through the incidence angle. A residual vertical M2 (CATS error)
therefore appears in range at the Msf frequency, indistinguishable from
horizontal Msf. The pixel model keeps a free residual vertical M2/O1 term,
because tying the vertical to CATS biases Msf (at LAR2 by about 0.05 m/day and
25°), and that term absorbs range's Msf information. What remains is
azimuth.

The along-flow coefficient is then the azimuth Msf coefficient divided by

*p* = (azimuth line-of-sight row, horizontal part) · (unit flow axis),

stored per source and year as `azimuth_flow_projection_by_year`. |p| is near 1
when ice flows along the flight direction, and near 0 when it flows along range.

## How errors scale

Azimuth senses cross-flow motion through √(1 − p²), since its horizontal
sensitivity is close to a unit vector. Dividing by *p* gives:

| \|p\| | Noise, relative to \|p\| = 1 (1/\|p\|) | Along-flow error per unit of cross-flow or systematic azimuth signal (√(1 − p²)/\|p\|) |
|---|---|---|
| 0.8 | 1.25× | 0.75 |
| 0.5 | 2.0× | 1.7 |
| 0.3 | 3.3× | 3.2 |
| 0.23 | 4.3× | 4.2 |

Two kinds of error, and only one is reported:

- **Noise** is in the uncertainties. The pixel and block covariances come from
  the fit residuals and the calendar-cluster sandwich, propagated through the
  division, so they grow as 1/|p|. Larger blocks and regional means average it
  down.
- **Systematic error** is not. A coherent signal at the Msf frequency lies in
  the same direction as the model's Msf column, so the fit absorbs it into the
  Msf coefficient and leaves no residual. Candidates are cross-flow Msf, aliased
  cross-flow M2 (for example from tidal tilt), and residual azimuth patterns
  from orbits, ionosphere or the spacecraft order term at that periodicity. All
  blocks in a region share the bias, so averaging does not reduce it: a
  regional mean can be formally significant and still be wrong.

The split-period check does not catch it either if the systematic term is
present in both halves.

## Larsen C example

From the Larsen C proposal run: 128×512 windows, 64×256 stride, 2 km blocks,
`--min-sources 3`.

| | NE corner (near the front) | Shelf reference |
|---|---|---|
| Median \|p\| | 0.23 (90% of sources below 0.3) | 0.79 |
| Direct-azimuth Msf, regional mean | 0.032 m/day | 0.068 m/day |
| Along-flow Msf, regional mean of blocks | 0.140 ± 0.018 m/day at 129° ± 7° | 0.085 ± 0.007 m/day at 144° ± 3° |
| 2 km block amplitude SE (median) | 0.063 m/day | 0.015 m/day |
| 2 km blocks detected | 24% | 99% |

- **The radar sees less Msf in the corner.** Its high along-flow amplitude
  comes entirely from dividing by 0.23.
- **Statistically, the excess is significant:** T = 24, p ≈ 5 × 10⁻⁶.
  Split-period block differences there are consistent with their SEs, so the
  noise part is calibrated.
- **Physically, it is ambiguous.** Only about 0.013 m/day of extra azimuth
  signal is needed to produce the whole excess. That is a cross-flow Msf about
  15% of the along-flow one, or about the size of the predicted M2 alias at
  LAR2 (0.0097 m/day).

GPS context (King, Makinson & Gudmundsson, 2011, GRL):

- **Along-flow Msf is uniform.** It is 87–97 mm/day with the same phase at
  LAR1–3, including LAR3, which lies near this corner.
- **Cross-flow motion differs between sites.** It is negligible at LAR2. At the
  front sites LAR1 and LAR3 it reaches a few centimetres and follows tidal
  tilt.
- **A tilt-driven M2 alias could explain the corner.** If a few centimetres of
  cross-flow M2 were present, its alias (about 0.01 m/day) at |p| ≈ 0.23 would
  give about 0.04–0.05 m/day of apparent along-flow Msf. That is the corner's
  excess, but it is a plausible mechanism, not a demonstrated one.

## Practice

- **Mask low projection in amplitude products.** The pixel maps already do this
  through the `along_display` display rule (|p| ≥ 0.3 in every year).
- **Use the same cut for blocks:** `tools/aggregate_msf_blocks.py
  --min-abs-projection 0.3`. The split-period tool inherits it. On the Larsen
  run it halved the 90th-percentile floating block SE (0.063 → 0.029 m/day) and
  kept 97% of floating detections (2,176 → 2,102), with calibration unchanged.
- **Report by projection range:**
  - |p| ≥ 0.5: along-flow Msf as is;
  - 0.3–0.5: lower confidence;
  - below 0.3: azimuth Msf only, or "unconstrained (low projection)".
- **When interpreting regional means,** consider a systematic allowance
  alongside the statistical SE: δ_sys · √(1 − p²)/|p|, with δ_sys of about
  0.01 m/day (the M2-alias scale at LAR2). Report it as a separate term, not
  folded into the SE.

## What would recover low-projection areas

- **A second viewing geometry.** Another track (ascending, or another relative
  orbit) sees the flow at a different projection, so the horizontal Msf vector
  can be solved without the along-flow assumption. A different acquisition time
  also rotates the M2 alias, which separates range Msf from vertical M2.
  Sentinel-1 has no second track at Larsen C. NISAR passes would rotate the
  alias by about 50–85°.
- **A vertical M2 prior** with realistic CATS uncertainty, instead of a free
  residual term. This partly restores range information at the cost of some
  CATS leakage; the Evans vertical-prior pilot explored it.
- **Borrowing along flow lines** from well-projected ice. This is possible, but
  it imposes a spatial model.
