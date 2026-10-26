"""Ordered public processing stages shared by execution and inspection."""

STAGE_DESCRIPTIONS = {
    'coregister': 'Prepare reference geometry and align acquisition bursts',
    'assemble': 'Build one fixed-ownership SLC and validity mask per acquisition',
    'match': 'Measure offsets for active pairs',
    'geocode': 'Map offsets, velocities, masks and uncertainties',
    'diagnostics': 'Generate pair quality reports and figures',
    'correct-swaths': 'Apply optional pinned spacecraft calibration to separate pair products',
}
STAGES = tuple(STAGE_DESCRIPTIONS)
