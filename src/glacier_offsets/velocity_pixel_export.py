"""Export native-grid coefficient NetCDF, GeoTIFFs and exploratory figures.

Works on version-1 prepared directories and version-2 workflow results. Field
names follow the stored parameter names and year labels; CRS labels come from
the inputs.
"""
import argparse
from pathlib import Path

from .velocity_workflow_products import RunView, export_maps


def export(out, regions=None):
    return export_maps(RunView.open(Path(out), regions))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--regions', type=Path)
    a = p.parse_args(argv)
    export(a.output.resolve(), a.regions)


if __name__ == '__main__':
    main()
