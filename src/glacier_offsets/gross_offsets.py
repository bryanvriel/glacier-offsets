"""M2 search centers; geometric coregistration offsets never enter this module."""
from .config import gross_values


def apply_gross(ampcor, cfg):
    gross=gross_values(cfg)
    ampcor.mergeGrossOffset=0
    ampcor.setConstantGrossOffset(*gross)
    return gross
