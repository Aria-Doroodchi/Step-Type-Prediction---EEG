# Preprocessing profiles

Each YAML in this folder is a **named preprocessing profile** — a complete
specification of the preprocessing stage that applies *globally to every
participant* (per-participant overrides in `configs/overrides/Pxx.yaml`
still layer on top for participant-specific deviations).

Why this exists: when you tune ML models, you want the preprocessing held
fixed so that a metric improvement is attributable to the model change,
not a silent change in upstream filtering. Conversely, when you tune
preprocessing (e.g. tighten the ICLabel threshold), you want a clean way
to A/B-compare the downstream model accuracy. Named profiles give you
exactly that.

## Selecting a profile

`configs/default.yaml` has one line that picks the active profile:

```yaml
preprocessing_profile: default     # → loads configs/preprocessing/default.yaml
```

Override on the CLI for a single run:

```bash
python run.py --preprocessing-profile aggressive
```

(Or set it in `configs/local.yaml` for your machine.)

## Creating a new profile

1. Copy an existing one as a starting point:

   ```bash
   cp configs/preprocessing/default.yaml configs/preprocessing/strict_ica.yaml
   ```

2. Edit. Common knobs:

   - `filter.bandpass`         narrow or widen the final filter
   - `ica.n_components`        more components = finer decomposition, slower
   - `ica.iclabel_artifact_prob_threshold`  lower = aggressive (more dropped),
                                            higher = conservative (more kept)
   - `reject.method`           `autoreject` (per-channel CV) vs
                               `threshold` (single voltage cutoff)

3. Reference it from `configs/default.yaml`:

   ```yaml
   preprocessing_profile: strict_ica
   ```

4. Re-run preprocessing. **Note:** the `data/interim/epochs/` cache was
   built with the *previous* profile. To produce clean comparisons,
   either:

   ```bash
   make clean preprocess src features train       # full rebuild
   ```

   or scope the cache by profile manually:

   ```bash
   mv data/interim data/interim_default
   python scripts/01_preprocess.py --force
   # … now data/interim/ holds the new profile's epochs …
   ```

## How merging works

Final config = profile + cohort-level config + machine config + per-participant override.

```
configs/preprocessing/<profile>.yaml   (THIS FILE — global preprocessing knobs)
                  ↓ merged into ↓
configs/default.yaml                   (paths, participants, modeling, src_loc)
                  ↓ merged into ↓
configs/local.yaml                     (per-machine paths)
                  ↓ merged into ↓
configs/overrides/Pxx.yaml             (per-participant tweaks, applied last)
```

Right-hand wins on conflicts. So a per-participant override `bads_extra`
or `ica.manual_exclude` always wins over the profile setting; that's by
design — those are the lab-curated decisions you don't want erased by a
profile switch.

## What's NOT in here

- **Source localization** (eLORETA settings) — `configs/default.yaml` →
  `source_localization:`. Treated as a stable assumption, not a profile.
- **Feature extraction** (bin width, frequency bands) — also in
  `default.yaml`. These are downstream of preprocessing.
- **Modeling** (hyperparameter grids, CV folds) — also in `default.yaml`.

If you find yourself wanting to A/B those too, the same profile pattern
works for them; for now we only need it for preprocessing.

## Provided profiles

| Name | Purpose |
|---|---|
| `default` | The cohort default. Auto bads (PyPREP), auto ICA (ICLabel @ p>0.9), autoreject epochs, 0.1–40 Hz bandpass. |
| `smoke`   | Fast variant used by `configs/smoke.yaml`. Skips PyPREP, half-size ICA, threshold rejection, narrow band. |
