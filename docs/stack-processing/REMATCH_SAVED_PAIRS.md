# Isolated Evans matching trials

Copy `rematch_saved_pairs.py` into the Evans `tools/` directory on the server. Run in the environment used for native stack processing, with the current `glacier_offsets` package importable. This is a separate experimental runner, not a new option on `glacier-offsets stack match`.

Required server products: the existing stack's `geometry/`, acquisition manifests and retained aligned per-burst SLCs under `acquisitions/`, and the selected pair manifests plus `radar/offsets.h5`. Original merged SLCs may have been released. No raw SAFE discovery or COMPASS coregistration is performed. Paths in retained manifests must still resolve on the server.

From the Evans project root, inspect one trial without writes:

```bash
python tools/rematch_saved_pairs.py configs/stack.yaml \
  --pair 2018-06-09_2018-06-15 \
  --output work/matching_trials/w128x512 \
  --window 128 512 \
  --first-center 137.5 551.5 \
  --workers 8 --dry-run
```

A successful dry run lists no missing inputs. It validates the matching grid and available product paths; actual execution additionally verifies retained product checksums and matching layout. It does not estimate matching quality.

Run three window sizes on two bad pairs and one good comparison pair:

```bash
for trial_spec in '64 256' '128 512' '256 1024'; do
  read -r trial_az trial_range <<< "$trial_spec"
  python tools/rematch_saved_pairs.py configs/stack.yaml \
    --pair 2018-06-09_2018-06-15 \
    --pair 2019-05-11_2019-05-17 \
    --pair 2017-11-23_2017-11-29 \
    --output "work/matching_trials/w${trial_az}x${trial_range}" \
    --window "$trial_az" "$trial_range" \
    --first-center 137.5 551.5 \
    --workers 8 || break
done
```

Arrays are `[azimuth, range]` in original SLC pixels. The first center is a zero-based full-resolution SLC coordinate, not an offset-grid index. The common first center fits the largest window with the saved search `[10,40]`, and stays on the original `[32,128]` center lattice. The runner retains the YAML's stride, search, oversampling, deramp, quality thresholds, map grid and gross-offset settings unless explicitly overridden. These window-size runs change only window size and the shared initial center.

Use `--pair` repeatedly for any existing saved pair ID. No changes to stack start/end dates or pair planning are needed. Acquisition assemblies are rebuilt only in the experiment's `_inputs/` directory using the saved full-stack ownership. This avoids changing stack-common validity by replanning a reduced date inventory.

The original stack is protected by a shared read lock for the duration of the trial. Multiple rematching experiments may read the same stack concurrently, provided each uses a fresh output directory. Production processing, packaging, correction, and other exclusive stack operations remain blocked while a trial is reading. A production writer also blocks new trials. The scratch directory does not determine this lock; it protects the source stack resolved from the YAML's `output.directory`. Error messages include the locked path.

Older versions of this runner took an exclusive source lock, so even separate experiment outputs and scratch directories could not run concurrently. Update both `tools/rematch_saved_pairs.py` and `src/glacier_offsets/stack_lock.py`, ensuring the processing environment imports the updated package. An already-running older process retains its exclusive lock until it finishes. The lock is an OS lock on the directory inode, not a lock file to delete.

Production inputs are opened read-only and their sizes/mtimes are checked afterward. Each experiment must use a fresh output directory; reruns into an existing directory are refused. A failed run leaves diagnostics/checkpoints for inspection but has no resume interface.

The experiment verifies the retained reference and aligned input checksums; this initial read can take time on large stacks. It creates merged SLCs only for the selected dates and releases experimental merged SLCs after their matching consumers finish. Native range-2x processing also needs temporary storage. Saved input manifests and production pair products remain in place.

Outputs for each pair, e.g.:

```text
work/matching_trials/w128x512/
  experiment.json
  pairs/2018-06-09_2018-06-15/
    radar/offsets.h5
    geo/
    diagnostics/offset_quality.png
    diagnostics/overview.png
    comparison.json
```

`comparison.json` joins the old and new results by physical center coordinates. Larger windows change edge support, seam flags and flattened observation IDs. Do not compare array indices or IDs directly across runs. The geographic map grid is retained, but native center geometry is recomputed. Peak statistics also depend on window size; improved peaks or smoother maps alone are not accuracy validation.

A subsequent search-only sensitivity can add `--search 4 12` with a fresh output directory. The search is centered on the gross-offset mode selected in the YAML. The current zoom 16 fits this search with raw oversampling 2; smaller searches may require reducing `matching.zoom` in the YAML.

## Velocity gross offsets

The runner supports `gross_offsets.mode: velocity`, including comparisons against an originally zero-gross pair. Create `configs/stack.velocity-prior.yaml` alongside `stack.yaml`. For a MEaSUREs VX/VY map in physical map-axis m/year, use the following and replace the file paths with the actual server paths:

```yaml
extends: stack.yaml
gross_offsets:
  mode: velocity
  velocity:
    vx:
      path: /absolute/path/to/antarctica_ice_velocity_450m_v2.nc
      variable: VX
    vy:
      path: /absolute/path/to/antarctica_ice_velocity_450m_v2.nc
      variable: VY
    basis: map_xy
    map_units: ground
    units: m/year
    year_days: 365.25
    interpolation: bilinear
    product: MEaSUREs InSAR-Based Antarctica Ice Velocity Map (NSIDC-0484)
    version: '2'
```

Use the basis, units and product provenance of your actual velocity input. East/north components require `basis: east_north`; GeoTIFF components use `band: 1` instead of `variable`. A velocity raster is required; GPS harmonic coefficients are not a direct input to this interface.

```bash
python tools/rematch_saved_pairs.py configs/stack.velocity-prior.yaml \
  --pair 2018-06-09_2018-06-15 \
  --output work/matching_trials/velocity_w128x512_search4x12 \
  --window 128 512 --search 4 12 \
  --first-center 137.5 551.5 --workers 8 --dry-run
```

Remove `--dry-run` to run. For a controlled test, first run velocity gross offsets with the original `--search 10 40`, then with `--search 4 12`, using separate output directories and the same window/centers. The latter searches ±4 azimuth and ±12 range pixels around the per-source integer velocity prediction. Reported total displacement remains gross plus measured residual, exactly once. Missing velocity coverage is masked, with no zero fallback.

Projection uses each source burst's pinned reference radar grid and the retained `geometry/reference_orbit.h5` snapshot, authenticated against its companion JSON and the pinned orbit record. It uses the matched burst's exact interval, including pairs that do not contain the geometry reference date. No matched-date orbit is substituted. If an old stack lacks the snapshot, the original pinned EOF must be available; a snapshot is created only inside the fresh experiment's `_inputs/geometry/`. Production geometry is never enriched or modified by the runner. Velocity component files and orbit inputs enter the experiment checksum ledger.

Validation performed locally: control tests for dry-run no-write behavior, fresh/output-path guards, settings isolation and search bounds, common center lattices, physical-center HDF5 joins, retained orbit selection, reference adapters and velocity dispatch. Targeted native tests check reference-orbit/exact-burst timing, recovery outside a zero-centered search, and range-2x gross-offset scaling. Local dry-run resolves the Evans settings and reports the missing reference/acquisition products. Full Evans rematching is untested locally because those retained products exist only on the server.
