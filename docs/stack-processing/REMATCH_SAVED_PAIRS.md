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

Arrays are `[azimuth, range]` in original SLC pixels. The first center is a zero-based full-resolution SLC coordinate, not an offset-grid index. The common first center fits the largest window with the saved search `[10,40]`, and stays on the original `[32,128]` center lattice. The runner retains the YAML's stride, search, oversampling, deramp, quality thresholds, map grid and zero/constant gross-offset settings unless explicitly overridden. These window-size runs change only window size and the shared initial center.

Use `--pair` repeatedly for any existing saved pair ID. No changes to stack start/end dates or pair planning are needed. Acquisition assemblies are rebuilt only in the experiment's `_inputs/` directory using the saved full-stack ownership. This avoids changing stack-common validity by replanning a reduced date inventory.

The original stack is locked against concurrent writers for the duration of the trial. Run after production processing has stopped; the script fails if another writer holds the lock. Production inputs are opened read-only and their sizes/mtimes are checked afterward. Each experiment must use a fresh output directory; reruns into an existing directory are refused. A failed run leaves diagnostics/checkpoints for inspection but has no resume interface.

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

A subsequent search-only sensitivity can add `--search 4 12` with a fresh output directory. This remains a zero/constant-centered search; the runner deliberately does not implement a velocity prior. Test windows first, then search width independently. The current zoom 16 fits this search with raw oversampling 2. Do not interpret the narrower search as a GPS-guided search.

Validation performed locally: five control tests for dry-run no-write behavior, fresh/output-path guards, settings isolation and search bounds, common center lattices, and HDF5 comparisons joined by physical centers. Local dry-run resolves the Evans settings and reports the missing reference/acquisition products. Full native rematching is untested locally because those retained products exist only on the server.
