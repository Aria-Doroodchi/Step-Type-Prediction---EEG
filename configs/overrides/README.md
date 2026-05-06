# Per-participant overrides

One YAML file per participant: `Pxx.yaml`. Default runs apply only the
`raw_assembly` section from this file when that participant is being
processed. This keeps preprocessing parameters uniform across the cohort
while preserving manual `.bdf` crops/appends that reflect data-collection
issues.

Set `participant_overrides.mode: full` in a config, or pass
`--participant-override-mode full`, to apply the full participant YAML for
fine-tuning. Set the mode to `none` to ignore participant YAMLs entirely.

The most common reason to create one of these files is that the participant's
raw recording needed manual surgery (cuts, multiple files concatenated)
during the original preprocessing. The new pipeline preserves that surgery
declaratively rather than in a one-off Python script.

## Schema

```yaml
# configs/overrides/P03.yaml

# -------------------------------------------------------------------
# Raw assembly: how to load this participant's raw recording.
# Defaults to a single file at `{raw_root}/{pid}/{pid}_CNV.bdf` if omitted.
# -------------------------------------------------------------------
raw_assembly:
  files:
    # Simple form — load this file in full.
    - "P03/P03_CNV_part1.bdf"
    # Cropped form — load and crop to [tmin, tmax] seconds before concat.
    - { path: "P03/P03_CNV_part2.bdf", tmin: 100.0, tmax: 500.0 }
    # If a single file needs multiple keep-windows, list it multiple times
    # with different tmin/tmax — each entry is concatenated in order.
    - { path: "P03/P03_CNV_part2.bdf", tmin: 600.0, tmax: 1100.0 }

# -------------------------------------------------------------------
# Bad channels the auto-detector tends to miss. Added on top of PyPREP's
# detected list; final bads = pyprep ∪ bads_extra.
# -------------------------------------------------------------------
bads_extra: [P10, PO8]

# -------------------------------------------------------------------
# Override the conservative ICLabel auto-exclusion. Use sparingly — only
# when the auto-classifier visibly misses an artifact you know is there.
# -------------------------------------------------------------------
ica:
  manual_exclude: []          # extra component indices to also exclude
  manual_keep: []             # indices the auto-classifier flagged but you want to keep

# -------------------------------------------------------------------
# Epoch repair/rejection is AutoReject-local by default. It CV-tunes
# per-channel thresholds plus n_interpolate/consensus, repairing epochs by
# interpolation where possible before dropping them.
#
# Legacy preprocessing.reject.thresholds entries from the old scripts are
# intentionally redundant for default runs. Keep them only as commented
# provenance, or opt into a threshold fallback explicitly in full override
# mode if you are doing a side-by-side legacy comparison.
# -------------------------------------------------------------------
# preprocessing:
#   reject:
#     method: autoreject
#     n_interpolate: [1, 4, 8, 16]
#     consensus: [0.2, 0.4, 0.6, 0.8]
#     random_state: 42

# -------------------------------------------------------------------
# Anything else from default.yaml can also be overridden here, e.g.:
# -------------------------------------------------------------------
# preprocessing:
#   ica:
#     n_components: 25
```

## What gets merged where

`configs/default.yaml`  ←  `configs/local.yaml`  ←  selected participant override

Right-hand wins. In the default `raw_assembly_only` mode, "selected
participant override" means only `raw_assembly`. In `full` mode, it means the
entire `configs/overrides/Pxx.yaml` file.
