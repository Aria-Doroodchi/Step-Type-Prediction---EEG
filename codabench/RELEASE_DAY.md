# Release day: Graz + BrainHero → a sealed-phase candidate

DRAFT (sprint 2026-09-28; commands verified on the mock study at the end of
the sprint, see § 9). The runbook for the day the Track 2 training data is
released. It takes you from download to a zipped candidate submission and never
uploads: uploading is the user's action. The recipe itself is in
[SEALED_RECIPE.md](SEALED_RECIPE.md). The pipeline was rehearsed end to end on
a mock study with the sealed structure (§ 8).

All commands run in WSL (`wsl.exe bash -s <<'EOF' … EOF` from the Bash tool;
`source ~/codabench/env.sh` first). Run anything longer than ~10 min as a
resumable lane script from a background Bash call (see `scripts/sealed_lib.sh`),
never `nohup … &`.

## 0. Before anything: freeze and regress (15 min)

The organisers will ship the new study in updated packages: a new `neuralbench`
(or `neuralfetch`) version and new commits in `2026-competition`. Any of these
can shift our numbers, so record what we had, update, and prove that nothing moved.

```bash
source ~/codabench/env.sh; D=~/codabench/logs/release_$(date +%F); mkdir -p $D
pip freeze > $D/freeze_before.txt
cd ~/codabench/2026-competition
git stash list; git diff --stat          # local edit: bci_studies.py _OVERLAYS
git stash && git pull && git stash pop   # re-apply our overlay registrations
pip install -U neuralbench neuralfetch   # only if the release notes ask for it
pip freeze > $D/freeze_after.txt; diff $D/freeze_before.txt $D/freeze_after.txt
bash ~/codabench/scripts/install_xsess_overlays.sh   # re-link overlays into the new package
# regression: must reproduce 0.770 (clean, w = 0.75) exactly, in under 5 min
R=regress_$(date +%F); rm -rf ~/codabench/logs/$R ~/codabench/logs/sealed_$R
RUN_NAME=$R bash ~/codabench/scripts/train_sealed.sh ~/neuralbench/benchopt_data zhou2016_xsess
cat ~/codabench/logs/$R/RESULTS.md
```

If `git stash pop` conflicts in `bci_studies.py`, keep upstream's version and
re-run `install_xsess_overlays.sh`, which re-adds our entries. If the regression
does not give 0.770, stop and find out why before trusting any release number.

## 1. Download (pre-approved)

The Graz + BrainHero data is pre-approved. It goes to
`Z:\Projects\codabench\neural_compet\` (master copy) and then to the WSL disk,
where training runs. The expected size is ~15–30 GB (80 h × 47 ch × 500 Hz).

- If the organisers add a benchopt selector (`BCI[study=<name>]`):
  `bash ~/codabench/scripts/prepare_track2_data.sh <name>` (idempotent, retries,
  log in `logs/prepare_track2_*.log`).
- Otherwise use `neuralbench eeg <task> --dataset <name> --download`, as in the
  start kit.
- Then copy the study folder from `~/neuralbench/benchopt_data/neural_compet/` to Z:
  with `robocopy` from PowerShell (the pattern is in
  `logs/download_z_2026-09-25/`).

## 2. Inspect what the official loader delivers (decides half of § 5)

The solver sees only what the official loader delivers, so read its meta first:

```bash
cd ~/codabench/2026-competition
python - <<'PY'
import sys; sys.path.insert(0, "tracks/bci_decoding")
from datasets.bci_studies import Dataset            # or the organisers' new dataset class
d = Dataset(study="<name>"); data = d.get_data()     # adapt to the released API
m = {k: data[k] for k in ("n_classes", "sfreq", "n_chans", "n_times")}
print(m, data["ch_names"])
X, y, info = next(iter(data["train_loader"])); print(X.shape, y[:10], info.keys())
PY
```

Read off:
- **`sfreq` and `n_chans`.** NeuralBench's default EEG extractor has
  `picks: [eeg]` and `frequency: 120.0`
  (`neuralbench/defaults/config.yaml`), so expect **43 EEG channels at 120 Hz**.
  In that case the EMG/EOG channels never reach `predict()`, and their ablation
  (§ 5) is EDA only. If 47 channels or 500 Hz arrive, the 500 Hz rows of § 8
  apply.
- **The class codes** and their order (LabelEncoder sorts the names).
- **Whether sessions and contexts are in the trigger table**
  (`train_loader.dataset.seg_ds.triggers` columns). The solver's online mode
  and `blend_w="auto"` need session ids there; the cell metric needs context.
- **How the test split is defined**, i.e. which subjects and sessions.

## 3. Overlay + cross-session cache

1. Write `config/nb_overlays/<task>/<name>_xsess.yaml` following
   `config/nb_overlays/motor_imagery/zhou2016_xsess.yaml`. The cache keeps every
   window with its split code and the harness makes its own splits, so the
   overlay only has to load the full study with every labelled window. Add it to
   `STUDIES`-free use with `--task`/`--overlay` (below). Register it for benchopt
   by adding its name to `install_xsess_overlays.sh` and re-running that script.
2. Build the cross-session cache (every window + subject/session/run/context):

   ```bash
   python ~/codabench/analysis/xsess_cache.py <name> --task eeg/<task> --overlay <name>_xsess
   ```

   Check the `[data]`/final line: 20 subjects; 6 sessions for the ten training
   participants and 3 for the ten evaluation participants; 3 classes; the channel
   count from § 2; `context` in meta (`context_column`). If the study has no
   `context`/`condition`/`paradigm` column, find the column that holds
   Graz/BrainHero and add it to `xsess_cache.py`'s list before building.
3. EMG/EOG EDA only (not for the submission unless § 2 showed them in the
   official loader): build a second cache with the non-EEG channels kept, via
   `--picks eeg,emg,eog`. It records `ch_types` in meta.

## 4. EDA

TBD

## 5. Replica validation and the § 3 ablations

TBD

## 6. Pre-registered release-day rules

TBD

## 7. Train the candidate, replay, zip

TBD

## 8. Dress-rehearsal timings (mock, 2026-09-28/29)

TBD

## 9. Verification log

TBD
