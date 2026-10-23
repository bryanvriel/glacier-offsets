# Getting started

Use the **pair workflow** for two acquisitions, or the **stack workflow** to reuse
aligned acquisitions across many date pairs. Run commands from this repository.
The examples use Antarctic data: `HH` polarization and `EPSG:3031`.

## 1. Set up

For a new CPU installation:

```bash
conda env create -f environment.yml
conda activate glacier-offsets
python -m pip install --no-deps --no-build-isolation -e .
```

If already installed, activate your existing environment instead (on this
machine, `conda activate isce3`). Set a CPU budget before processing:

```bash
export OMP_NUM_THREADS=4
```

## 2. Have your inputs ready

You need SAFE directories or SAFE ZIPs, precise orbit files, a DEM with WGS84
ellipsoidal heights, and a polygon AOI. The DEM must cover the whole selected
bursts, including areas outside the AOI. Replace these paths:

```bash
SAFES=/path/to/safes
ORBITS=/path/to/orbits
DEM=/path/to/dem.tif
AOI=/path/to/glacier.kml
```

Need data first? Follow [burst downloads](README_DETAILED.md#download-bursts-for-an-aoi)
and [orbit/DEM preparation](README_DETAILED.md#prepare-orbits-and-a-dem). Use
`--allow-partial` to accept compatible partial burst acquisitions; review their
omissions with `--dry-run`. Finish downloading before preparing inputs.

## 3A. Run one pair

Choose the two actual SAFE paths, then generate and check a configuration:

```bash
python tools/init_pair_config.py \
  "$SAFES/earlier.SAFE" "$SAFES/later.SAFE" \
  --orbit-dir "$ORBITS" --dem "$DEM" --aoi "$AOI" \
  --polarization HH --epsg 3031 \
  --output configs/pair.generated.yaml --output-directory work/pair

glacier-offsets list-bursts configs/pair.generated.yaml
glacier-offsets validate configs/pair.generated.yaml --backend cpu
glacier-offsets run configs/pair.generated.yaml --backend cpu
```

The AOI selects complete bursts. Results are in **`work/pair/merged/geo/`**,
with radar offsets and diagnostics beside it. For one burst, replace `--aoi`
with `--burst-id ID`; results then go directly under `work/pair/`.

## 3B. Run a stack

Use one SAFE per UTC date, with a consistent track, direction and polarization:

```bash
glacier-offsets stack init \
  --safe-dir "$SAFES" --orbit-dir "$ORBITS" --dem "$DEM" \
  --aoi "$AOI" --polarization HH --epsg 3031 \
  --output configs/stack.generated.yaml --output-directory work/stack

glacier-offsets stack plan configs/stack.generated.yaml
glacier-offsets stack validate configs/stack.generated.yaml --backend cpu
glacier-offsets stack run configs/stack.generated.yaml --backend cpu
```

If you used `prepare-inputs`, replace the three `--safe-dir`/`--orbit-dir`/`--dem`
options with `--prepared-inputs /path/to/prepared/manifest.json`.

Review the plan before running. Defaults pair each date with its next neighbor
when their exact separation is at most 12 days. The earliest eligible date
becomes the reference; use `--reference-date YYYY-MM-DD` at initialization to
choose another. For partial collections, choose a reference with full desired
coverage. Keep the reference fixed when extending an existing stack.

To inspect time gaps and connected components visually, save `stack plan` output
to `plan.json`, then run `python tools/plot_stack_plan.py plan.json`.
See [network plots](README_DETAILED.md#visualize-a-stack-plan) for options.

Pair maps are in **`work/stack/pairs/<earlier>_<later>/geo/`**. Per-burst aligned
SLCs are retained under `work/stack/acquisitions/`; merged date SLCs are temporary
and released after their last matching use. CPU matching uses up to
`OMP_NUM_THREADS` single-threaded burst workers within each pair; other stages
and date pairs run sequentially. More workers need more RAM.

## 4. Resume or check progress

Rerun the same `run` command after an interruption; completed stages are reused.
Merged SLCs are rebuilt from retained coreg products only when needed again.
Set `storage: {keep_merged_slcs: true}` in your YAML to retain them instead.
Use `--force` only when you intend to repeat processing. Inspect a stack with:

```bash
glacier-offsets stack status configs/stack.generated.yaml --backend cpu
```

For matching settings, individual stages, GPU support and output definitions,
see the [detailed reference](README_DETAILED.md).
