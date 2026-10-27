# Relative IW1/IW2 azimuth correction

This empirical correction adjusts for a repeatable S1A/S1B-associated difference
between IW1 and IW2 azimuth offsets.

**Calibration.** Match corresponding ground patches independently in unspliced
IW1 and IW2 bursts, using the production ISCE3 range-2x treatment and sampling.
Retain native-valid burst interiors satisfying, independently in both swaths,
`(peak >= 0.3 and SNR >= 8) or SNR >= 20`. The January–June 2017 experiment
measured 59 six-/twelve-day links; 53 had usable primary observations.

**Model and least squares.** Estimate one shared coefficient from many links.
Let `i` and `j` be earlier/later dates, `g` a burst-pair group, and `k` a selected
window. All offsets below are total azimuth offsets in original reference pixels.
Average the raw IW2−IW1 differences within each group:

$$
y_{ijgk}=o^{(2)}_{ijgk}-o^{(1)}_{ijgk},
\qquad
\bar y_{ijg}=\frac{1}{n_{ijg}}\sum_{k=1}^{n_{ijg}}y_{ijgk}.
$$

Here `n_ijg` counts windows and `G_ij` counts available groups. Define spacecraft
order and weights so every link has unit total weight:

$$
c_{ij}=\mathbf{1}_{\{s_j=\mathrm{S1B}\}}-\mathbf{1}_{\{s_i=\mathrm{S1B}\}},
\qquad w_{ijg}=\frac{1}{G_{ij}}.
$$

Over the usable training links, weighted least squares gives:

$$
\widehat\beta=\operatorname*{arg\,min}_{\beta}
\sum_{(i,j)\in\mathcal L}\sum_{g=1}^{G_{ij}}
w_{ijg}\left(\bar y_{ijg}-c_{ij}\beta\right)^2,
$$

$$
\widehat\beta=
\frac{\sum_{(i,j)\in\mathcal L}\sum_{g=1}^{G_{ij}}w_{ijg}c_{ij}\bar y_{ijg}}
{\sum_{(i,j)\in\mathcal L}\sum_{g=1}^{G_{ij}}w_{ijg}c_{ij}^{2}},
\qquad
\widehat\Delta_{ij}=c_{ij}\widehat\beta.
$$

The 29 usable mixed-spacecraft links constrain the coefficient; 24 same-spacecraft
links serve as controls. No residual-based trimming or regularization is used.
The frozen estimate is:

$$
\widehat\beta=0.0257651\ \text{pixels}.
$$

Same-spacecraft pairs receive zero correction; mixed-spacecraft pairs receive
opposite signs. The relative contrast is about 0.36 m at representative Larsen sampling.

**Application.** Freeze the coefficient and apply a symmetric allocation:

$$
\begin{aligned}
o^{(1),\mathrm{corr}}_{ij}&=o^{(1),\mathrm{raw}}_{ij}+\widehat\Delta_{ij}/2,\\
o^{(2),\mathrm{corr}}_{ij}&=o^{(2),\mathrm{raw}}_{ij}-\widehat\Delta_{ij}/2.
\end{aligned}
$$

Extend this constant pixel bias across each swath. Correct only complete
single-swath chip/search footprints; mixed or unsupported windows retain raw
values and availability flags. Convert using local ground spacing and exact
pair intervals before the unchanged production map operator. Range is unchanged.
Publish companion rasters and diagnostics separately from raw products.

**Validation and limits.** Withhold acquisitions, links, and burst groups,
excluding shared source bursts from spatial training. Held-out mean overlap
mismatch improved for 23 of 29 usable mixed-spacecraft links. Calibration
uncertainty uses shared acquisition/source-burst bootstrap multipliers and is
kept separate from native formal uncertainty. The physical cause is unknown.
Uniform extension beyond the overlap is an assumption; absolute motion accuracy
is unvalidated. July–December 2017 is outside the calibration period.

[Pinned calibration](../../configs/calibrations/larsen-2017h1-spacecraft.json)
· [Production implementation](../../src/glacier_offsets/swath_correction.py)
