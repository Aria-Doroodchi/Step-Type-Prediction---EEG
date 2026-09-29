# Release day: Graz + BrainHero → a sealed-phase candidate

**DRAFT** (sprint 2026-09-28/29; § 9 records the literal run-through on the mock).
This runbook covers the day the Track 2 training data is released, from download
to a zipped, checked candidate. It never uploads: uploading is the user's action
(1 submission per day in the sealed phase). The recipe itself is in
[SEALED_RECIPE.md](SEALED_RECIPE.md).

**What was rehearsed and what was not.** A full-size mock with the sealed
structure (`analysis/mock_sealed.py`, § 8) was used to rehearse:
- `train_sealed.sh` on the organisers' split;
- the 500 Hz solver sizing;
- 6 of the 10 ablations;
- the EDA script;
- the zip check (§ 7 step 3).

Since 2026-09-29 the mock also serves the release-day replica through benchopt
(`split=replica_full`), so § 7 can be rehearsed end to end (block at the end of
§ 7). Three things can only be exercised on the real release:
1. the organisers' loader (§ 2): sampling rate, channels, session and context
   columns, whether the hidden sessions are included;
2. our NeuralBench overlays `<name>_xsess` / `<name>_all` and their registration
   (§ 3);
3. the cache build through NeuralBench at release size.

The commands for these were written against today's API (neuralbench 0.3.1,
benchopt 1.10.0) and tested on the proxies where possible. The organisers may
change that API.

## How to run the commands

Everything runs in WSL, one Bash-tool call per block:

```bash
wsl.exe bash -s <<'EOF'
source ~/codabench/env.sh >/dev/null; export PYTHONUTF8=1
# the variable block below, then the block's commands
EOF
```

Shell variables do not survive between calls, so paste the **variable block** at
the top of every call. Fill it in as the day goes on:

```bash
S=<name>                        # study key: cache name and overlay prefix, e.g. graz2026
DH=~/neuralbench/benchopt_data  # benchopt data home (= $BENCHOPT_DATA_HOME, set by env.sh)
EVAL=<comma list from § 2>      # the 10 evaluation participants' cache indices
FULL=$(python -c "import json,os; print(','.join(map(str, json.load(open(os.path.expanduser('~/neuralbench/xsess_cache/$S/meta.json')))['full_subjects'])))" 2>/dev/null)  # set once § 3.3 is done
DEC="CHANS=eeg RECIPE_SPEC=riemann:xd=1,fb=1 RECIPE_ALIGN=router-psd:riemann WCV=loso"; BW=auto  # § 6 decisions (recipe defaults until then)
```

- **Long jobs.** Anything longer than ~10 min (cache builds, ablations, § 7 runs)
  goes in a background Bash call (`run_in_background: true`) that runs the
  command in the foreground of its WSL shell. Never use `nohup … &`.
- **Watching.** Use the Monitor tool on
  `bash ~/codabench/scripts/monitor_loop.sh <log dir> 300` and read the step
  logs in that folder.
- **Resuming.** `train_sealed.sh` and `release_ablations.sh` are resumable: the
  same command again skips finished steps.
- **Line endings.** Shell scripts must keep LF endings: `grep -c $'\r' <file>`
  must print 0.

## 0. Freeze, update, regress (~15 min)

The organisers will ship the study in updated packages: a new `neuralbench` /
`neuralfetch` and new commits in `2026-competition`. Record what we had, update,
and prove that nothing moved.

```bash
D=~/codabench/logs/release_$(date +%F); mkdir -p $D
pip freeze > $D/freeze_before.txt
cd ~/codabench/2026-competition
git -c core.autocrlf=true status --short   # expect only tracks/bci_decoding/datasets/bci_studies.py
git -c core.autocrlf=true checkout -- tracks/bci_decoding/datasets/bci_studies.py   # drops our 4 _OVERLAYS lines
git -c core.autocrlf=true pull --ff-only
# pip install -U neuralbench neuralfetch    # only if the release notes ask for it
pip freeze > $D/freeze_after.txt; diff $D/freeze_before.txt $D/freeze_after.txt
bash ~/codabench/scripts/install_xsess_overlays.sh   # re-links our overlays, re-adds our _OVERLAYS entries
```

Why `-c core.autocrlf=true`:
- The clone was checked out on Windows with CRLF endings, and WSL's git has no
  autocrlf. Plain `git status` can therefore list ~82 files that differ only by
  CR (seen 2026-09-28), and a `git stash` would stash all of them. With the flag,
  only `bci_studies.py` differs (+4 lines: our `_OVERLAYS` entries).
- Upstream changed that file (d49d741, default study → dreyer2023; already
  fetched, `main` is 2 commits behind `origin/main`), so it is checked out before
  the pull.
- The installer re-adds our entries after the `"dreyer2023": "dreyer2023",`
  line, which `origin/main` still has. Its last output line must list every
  `*_xsess` name. If it does not, the anchor line has changed: add the entries to
  `_OVERLAYS` by hand.

**Regression** (~2 min; must reproduce the committed flow exactly):

```bash
R=regress_$(date +%F); rm -rf ~/codabench/logs/$R ~/codabench/logs/sealed_$R
RUN_NAME=$R bash ~/codabench/scripts/train_sealed.sh $DH zhou2016_xsess
grep "gate:" ~/codabench/logs/$R/STATUS.md
```

Expected, as on 2026-09-28 20:06:00–20:08:01
(`logs/sprint0928_D_zhou_regress/STATUS.md`): `gate: train 0.770000 replay
0.770000 EQUAL; harness blend_calib pooled 0.778333 (w=0.75), gap 0.0083 OK`.

If you get anything else, stop and find out why before trusting any release
number. The harness rows come from the cache built before the update, so a
changed package shows up in the benchopt score.

## 1. Download (pre-approved)

The Graz + BrainHero download is pre-approved (sprint brief § 8). The expected
size is ~15–30 GB (80 h × 47 ch × 500 Hz). The master copy goes to
`Z:\Projects\codabench\neural_compet\`; training reads the WSL copy.

- **If the organisers' `bci_studies.py` has a key for the study** (check
  `_OVERLAYS` after § 0):
  `bash ~/codabench/scripts/prepare_track2_data.sh <key>`.
  - It runs `benchopt prepare` (download plus window extraction), is idempotent
    and retries 3 times.
  - Log: `~/codabench/logs/prepare_track2_*.log`.
  - The data lands in `$BENCHOPT_DATA_HOME/neural_compet/`, which is where
    `load_task`, `xsess_cache.py` and benchopt read it.
- **Otherwise**, the start kit's `neuralbench eeg motor_imagery --dataset <name>
  --download` saves to `DATA_DIR` of `config/neuralbench_config.json`
  (`/home/ali_d/neuralbench/data`), and nothing here reads that folder. Move the
  study folder into `~/neuralbench/benchopt_data/neural_compet/` afterwards.
- **Copy to Z:** from PowerShell, with the options of
  `logs/download_z_2026-09-25/robocopy_new.log`:
  ```powershell
  robocopy \\wsl.localhost\Ubuntu\home\ali_d\neuralbench\benchopt_data\neural_compet\<Study> Z:\Projects\codabench\neural_compet\<Study> /S /E /DCOPY:DA /COPY:DAT /NP /MT:8 /R:2 /W:5 /NDL /NFL
  ```

## 2. Inspect what the official loader delivers (decides half of § 3–§ 6)

The solver sees only what the official loader delivers. Load it once exactly as
`BCI[...]` does, through `benchmark_utils.nb_task.load_task`.
`Dataset(study=...).get_data()` fails outside a benchopt run: it sets no
parameter, and `get_seed()` raises.

```bash
cd ~/codabench/2026-competition
OV=<the organisers' overlay name for the study, or None for the task default>
python - "$OV" <<'PY'
import os, sys; from pathlib import Path; sys.path.insert(0, "tracks/bci_decoding")
import pandas as pd
from benchmark_utils.nb_task import load_task
ov = None if sys.argv[1] == "None" else sys.argv[1]
loaders, meta = load_task("eeg", "motor_imagery", dataset=ov,
                          data_dir=Path(os.environ["BENCHOPT_DATA_HOME"]) / "neural_compet",
                          device="cpu", batch_size=64, seed=0, num_workers=0,
                          target_transform=lambda y: y.argmax(-1))
print({k: meta[k] for k in ("sfreq", "n_chans", "n_times")}, meta["ch_names"])
X, y, info = next(iter(loaders["train"])); print(X.shape, y[:10].tolist(), list(info))
ts = []
for name in ("train", "val", "test"):
    t = loaders[name].dataset.seg_ds.triggers
    print(f"--- {name}: {len(t)} windows; columns {t.columns.tolist()}")
    print(t.groupby(["subject", "session"]).size().unstack(fill_value=0))
    ts.append(t)
t = pd.concat(ts)
print("context-like columns:", {c: sorted(t[c].astype(str).unique())
                                for c in ("context", "condition", "paradigm") if c in t})
print("event types:", t["type"].unique().tolist(),
      "| codes:", sorted(t["code"].astype(str).unique()) if "code" in t else "no code column")
ids = sorted({str(s) for s in t["subject"]}, key=lambda s: (len(s), s))
n_sess = t.groupby(t["subject"].astype(str))["session"].nunique()
print("cache index -> subject id (sessions):", {i: f"{s} ({n_sess[s]})" for i, s in enumerate(ids)})
print("EVAL guess (fewer sessions than the most):",
      ",".join(str(i) for i, s in enumerate(ids) if n_sess[s] < n_sess.max()) or "none")
PY
```

Tested 2026-09-29 00:06:34–00:06:43 on `OV=zhou2016_xsess` (9 s; 120 Hz, 14 ch, 3
sessions per subject, `EVAL guess: none`). Read off:

- **`sfreq` and `n_chans`.** NeuralBench's default EEG extractor picks `eeg` at
  120 Hz (`neuralbench/defaults/config.yaml`), so expect **43 EEG channels at
  120 Hz**.
  - Then the EMG/EOG channels never reach `predict()`. Rule 1 (§ 6) cannot adopt
    them, and the channel ablations (§ 5 b) are EDA only.
  - If 47 channels arrive, the default cache is itself the 47-channel cache: skip
    § 3.4 and § 5 b. If 500 Hz arrives, the 500 Hz column of § 8 applies.
- **The class codes and their order.** The label is the index of the code in the
  sorted code names (NeuralBench's LabelEncoder).
- **Session and context columns.**
  - The solver's `blend_w="auto"` needs `session`.
  - The cell metric, the `al_ctx` ablation and `align="subject_context"` need a
    context column named `context`, `condition` or `paradigm`.
  - If Graz / BrainHero sits in a column with another name, add that name in two
    places: the `for col in ("context", "condition", "paradigm")` loop in
    `analysis/xsess_cache.py` (`build`), and `_CTX_COLUMNS` in
    `solvers/bci_decoding/riemann_sealed.py`. With only the first, the harness
    scores context cells while the solver's `auto` weight does not, and the MATCH
    gate can fail. Then re-run the § 0 regression.
- **The subject index map, and EVAL.**
  - A cache index is the position of the subject id in the ids sorted by
    `(len(id), id)`, the order `xsess_cache.py` uses. This is not a plain sort.
  - EVAL is the ten evaluation participants' indices. Put them in the variable
    block. The `EVAL guess` line lists the subjects with fewer sessions than the
    most.
  - If every subject shows 6 sessions, the evaluation participants' hidden
    sessions 4–6 are in the release. Take EVAL from the organisers'
    documentation, and follow the hidden-session branch of § 3.1.
- **Which subjects and sessions are in train / val / test** (the tables), plus
  the event `type` values and the session labels. The overlay queries of § 3.1
  use them.

## 3. Overlays and caches

### 3.1 Two overlays

Write both into `~/codabench/config/nb_overlays/motor_imagery/`:
- `BCI[study=...]` loads only eeg/motor_imagery overlays: `bci_studies.py`
  hard-codes `load_task("eeg", "motor_imagery", ...)`.
- The installer links only that folder.
- If the organisers' study lives under another NeuralBench task, the installer's
  link loop and `bci_studies.py` must be extended first (not rehearsed).

Start from the organisers' overlay for the study, if they ship one. To find it:
`NB=$(python -c 'import neuralbench, os; print(os.path.dirname(neuralbench.__file__))'); ls $NB/tasks/eeg/motor_imagery/datasets/`.
Copy its `data.study.source` block and any `filter_stimuli`, then replace the
split as in `config/nb_overlays/motor_imagery/zhou2016_xsess.yaml`:

```bash
cat > ~/codabench/config/nb_overlays/motor_imagery/${S}_xsess.yaml <<'YAML'
# Release-day internal replica (RELEASE_DAY.md section 3.1): test = the ten fully
# labelled participants' sessions 4-6; train = everything else minus a 2 % val slice.
data:
  study:
    source:
      name: <SOURCE: copy the organisers' whole source block>
    split:
      =replace=: true
      name: PredefinedSplit
      test_split_query: "subject in [<FULL ids, quoted>] and session in [<their 4th-6th session labels, quoted>]"
      col_name: split
      valid_split_by: _index
      valid_split_ratio: 0.02
      valid_random_state: 33
brain_model_output_size: &brain_model_output_size 3
metrics: !!python/object/apply:neuralbench.defaults.metrics.get_classification_metric_configs
  - *brain_model_output_size
YAML
```

Then fill in the placeholders. The ids and labels are the strings of the § 2
tables, e.g. `subject in ['Graz/1', 'Graz/2'] and session in ['4', '5', '6']`.

- **`<name>_xsess`, the replica.** Test = the fully labelled participants'
  sessions 4–6. The cache (§ 3.3) and § 7 step 1 both use this overlay, so
  benchopt's test set is exactly the harness's replica test set.
- **`<name>_all`, the final training set.** Copy the file to `${S}_all.yaml` and
  replace `test_split_query` with a small smoke slice, e.g. one run:
  `"subject == '<one fully labelled id>' and session == '<its first session>' and run == '<its last run>'"`.
  - PredefinedSplit needs a non-empty test split, and there is no template for a
    random slice.
  - The loader also withholds `valid_split_ratio` (2 %) of the training windows
    as a validation slice that no solver sees. The candidate therefore trains on
    ~98 % of the labelled windows.
  - Skip this overlay if the organisers' own selector already trains on all
    public labelled data. Use that selector as `DATASET` in § 7 step 2 instead.

**If the evaluation participants' hidden sessions are in the release** (§ 2 shows
them with 6 sessions), their windows carry no true labels. They must never train,
and they must not enter the replica's test set either.
- **In `<name>_xsess`, drop them with a stimulus filter** under `data.study`,
  next to `split`, as in the zhou2016 template:
  ```yaml
      filter_stimuli:
        name: QueryEvents
        query: "type != '<stimulus type from § 2>' or not (subject in [<EVAL ids, quoted>] and session in [<4th-6th labels, quoted>])"
  ```
  - Combine it with `and` if the organisers' overlay already has a filter query.
  - The harness and benchopt then see the same windows, and the cache holds no
    hidden rows.
  - Without the filter, these windows are split 0 under `<name>_xsess` and train
    with placeholder labels in calib:3. The preflight's `hidden_in_train=0` does
    not catch that, because only split 2 counts as hidden.
- **In `<name>_all`, make them the test split:**
  `test_split_query: "subject in [<EVAL ids>] and session in [<4th-6th labels>]"`.
  They then never train. Step 2's benchopt score on them means nothing
  (placeholder labels); only train = replay counts there.
- These queries are pandas syntax on the events table and could not be tested
  without the release. Check them with § 2's snippet (next section).

### 3.2 Register the overlays

```bash
sed -i "s/\"scherer2015_xsess3\"\]/\"scherer2015_xsess3\", \"${S}_xsess\", \"${S}_all\"]/" ~/codabench/scripts/install_xsess_overlays.sh
grep -n "new = \[" ~/codabench/scripts/install_xsess_overlays.sh; grep -c $'\r' ~/codabench/scripts/install_xsess_overlays.sh
bash ~/codabench/scripts/install_xsess_overlays.sh   # "linked <S>_xsess.yaml", ..., "_OVERLAYS now has: [..., '<S>_xsess', '<S>_all']"
```

The names go into the Python list `new = [...]` inside the installer. The `sed`
is idempotent and keeps LF endings (tested on a copy, 2026-09-29).

Then check each overlay with § 2's snippet (`OV=${S}_xsess`, then `OV=${S}_all`):
- `_xsess`: the test table must hold exactly FULL × sessions 4–6, and no hidden
  sessions may appear anywhere.
- `_all`: the smoke slice (or the hidden sessions) is the test table, and train
  holds everything else.

### 3.3 The cross-session cache

```bash
python ~/codabench/analysis/xsess_cache.py $S --task eeg/motor_imagery --overlay ${S}_xsess \
    --sealed --eval_subjects $EVAL --calib_sessions 3 --hidden_labelled 2>&1 | tee ~/codabench/logs/cache_$S.log
```

- **Never omit `--eval_subjects`.** Under `<name>_xsess` its default (the subjects
  with split-2 rows) is the fully labelled participants, i.e. inverted.
- **`--hidden_labelled`:** the split-2 rows (here the fully labelled
  participants' sessions 4–6) carry true labels and may be tested on.
- **`--task eeg/motor_imagery`:** the same windows as `BCI[...]`.
- **Rebuilding** needs `rm -rf ~/neuralbench/xsess_cache/$S` first, because the
  build skips an existing `meta.json`. Afterwards, move away the log folders of
  every run on the old cache: `train_sealed.sh` and `release_ablations.sh` stop
  on a changed cache fingerprint.

Check two log lines:
- `release structure: hidden (split 2) rows=<the full participants' sessions 4–6> eval_subjects=[EVAL] full_subjects=[the other ten] calib_sessions=3`
- the final line, `X=(n, C, T) subjects=20 sessions/subject=[...] classes=... ch_types={'eeg': C} context=<column>`:
  - 6 sessions for the ten fully labelled participants and 3 for the ten
    evaluation participants;
  - 3 classes;
  - C from § 2;
  - a context column, not `None`.

Now `FULL` in the variable block reads `meta["full_subjects"]`. Check that
`echo $FULL` prints ten indices that do not overlap EVAL.

### 3.4 The 47-channel cache (EMG/EOG)

```bash
python ~/codabench/analysis/xsess_cache.py ${S}_x --task eeg/motor_imagery --overlay ${S}_xsess --picks eeg,emg,eog \
    --sealed --eval_subjects $EVAL --calib_sessions 3 --hidden_labelled 2>&1 | tee ~/codabench/logs/cache_${S}_x.log
```

- **What it is.** The same windows plus the non-EEG channels, under a second
  study key, with the same subjects and the same FULL. The channel ablations of
  § 5 need it: on the 43-channel cache they are identical to the recipe row.
- **Channel types come from the channel names** (EMG / EOG / ECG in the name,
  `ch_types_source=names`). That is the rule the solver applies on Codabench.
  - The final line should show `ch_types={'eeg': 43, 'emg': 2, 'eog': 2}`.
  - Channels whose names do not say EMG/EOG (e.g. `ExG1`) are typed EEG on both
    sides.
  - `--picks` takes MNE channel types. If the release types these channels
    otherwise (e.g. `misc`), use that type.
- **Skip this cache** if § 2 already showed 47 channels.

## 4. EDA (~3 min per cache at 120 Hz, plus ~30 min reading)

```bash
cd ~/codabench && python analysis/release_eda.py $S --split calib:3 --test_subjects $FULL --chans eeg --out logs/release_eda_$S.md
cd ~/codabench && python analysis/release_eda.py ${S}_x --split calib:3 --test_subjects $FULL --chans all --out logs/release_eda_${S}_x.md
```

On `mock_sealed_120` (calib:3, 3 threads) it took 2 min 38 s (2026-09-28
23:45:16–23:47:55) and peaked at 4,485,420 kB (4.3 GiB). It has not been run at
500 Hz. The estimate there is 10–12 GB peak, so run it alone.

It uses the harness's own split, channel and hidden-row logic, prints the same
`[data]` line, and writes a report (`.md` plus a `.json` sidecar). The report
opens with a Flags list, then six sections:

1. **Structure**, next to the tracks page's 20 × 6 sessions, 43 + 2 + 2
   channels and 3 classes.
2. **Windows and class counts per subject × session × context cell.** It flags
   cells with < 10 windows or an imbalance above 1.5:1. Uniform LDA priors assume
   balance within a cell.
3. **Context layout** (within sessions by run, or one context per session), and
   any (subject, context) pair missing from training. `router-psdctx` needs every
   pair. In the solver, a pair with fewer than `ctx_min` = 16 training windows
   keeps its subject's whitening.
4. **The log-PSD router, per test session (4, 5, 6) and per context**, with the
   fallback rate at the 0.5 cap and uncapped.
   - With one calibration session it fell from 0.718 to 0.353 across two later
     Zhou sessions (LOG 2026-09-28 Phase 3; the script reproduces this).
   - On the full-size mock with three calibration sessions it scored 0.997.
   - A drop by session 6 is the risk to note.
   - Per-window router assignments (`router_assign`) are saved only by
     `sealed_run.py` rows (the `run` ablation, in `logs/sealed_rel_${S}_B/probs/`).
5. **Drift** of each session's mean covariance from the calibration mean.
6. **Evoked response per class**, which is xDAWN's premise. A GFP peak ratio of
   ≥ 1.5 means an evoked response is present (zhou2016: 1.60–2.09; the mock,
   which has none: 0.98–1.01).

The flags are descriptive heuristics, not rules; § 6 decides. The drift section
reads the X of any hidden rows in the cache (never their labels). With the § 3.1
filter there are none.

## 5. The ablations on the replica

The internal replica of the sealed split is **`SPLIT=calib:3
TEST_SUBJECTS=$FULL`**: every participant's sessions 1–3 train, and the ten
fully labelled participants' sessions 4–6 are scored with the cell metric.

`replica:3` (the organisers' own split) is for the mock only:
- On the released data it either has an empty test set (the evaluation
  participants have no labelled sessions 4–6) or scores placeholder labels.
- Since 2026-09-29, `train_sealed.sh`, `release_ablations.sh` and the harness
  refuse `replica:K` on any cache whose meta has no `"mock"` key.
- `release_ablations.sh` also stops when SPLIT is unset on such a cache.

Always pass `SPLIT` and `TEST_SUBJECTS` explicitly. Run the two ablation sets
below, each from its own background call:

```bash
# (a) the recipe and every ablation except the channel sets, on the default cache (lanes A + B)
TAG=rel_$S STUDY=$S SPLIT=calib:3 TEST_SUBJECTS=$FULL XS_THREADS=6 \
    STEPS="base al_ctx xd0 wcv_loso pool_test online run" bash ~/codabench/scripts/release_ablations.sh
```

```bash
# (b) the channel sets, on the 47-channel cache (lane A only)
TAG=rel_${S}_x STUDY=${S}_x SPLIT=calib:3 TEST_SUBJECTS=$FULL XS_THREADS=6 LANES=A \
    STEPS="base ch_eeg_eog ch_eeg_emg ch_all" bash ~/codabench/scripts/release_ablations.sh
```

**How to run them.**
- At 120 Hz, run (a) and (b) side by side: three lanes × 6 threads. Each 120 Hz
  harness process peaked at ≤ 6.1 GiB in Phase 2.
- On a cache over 2 GiB (500 Hz), the script runs its own lanes one after the
  other. Run (b) after (a).
- If § 2 showed 47 channels, run (a) without `STEPS` (all 10 steps; RESULTS.md
  is then written automatically) and skip (b).

**What each step changes** (header of `scripts/release_ablations.sh`):

| Step | Change from `base` |
|---|---|
| `base` | none: the recipe (blend_calib with router ids, xDAWN + filter bank, router-psd, chans eeg, pool all, blend-weight CV `last`) |
| `ch_*` | `--chans eeg+eog / eeg+emg / all` |
| `al_ctx` | router-psdctx: routing and whitening per (subject, context) |
| `xd0` | no xDAWN |
| `wcv_loso` | the blend weight by leave-one-calibration-session-out |
| `pool_test` | pooled model and router trained on the test subjects only |
| `online` | online-64 (rule-dependent, reported apart) |
| `run` | `sealed_run.py`: MeanLogReg floor, pooled / per-subject, aligns none / router-psd / router-psdctx |

**Watching.**
- `logs/sealed_<TAG>/STATUS.md` gets the preflight `[data]` line within seconds.
  Check `split=calib:3`, `test_subjects=[FULL]`, `test_sessions=[3, 4, 5]`,
  `test_cells=`, `hidden_in_train=0` and, on (b), `chans=eeg(43/47 ch)`.
- The step logs are in the same folder. The rows go to
  `logs/sealed_<TAG>_A/` and `_B/`.
- Because `STEPS` is a subset, each run ends with `lanes … finished; <step> not
  done` instead of `ALL DONE`. That is expected: stop the Monitor there.

When both runs have finished:

```bash
python ~/codabench/analysis/release_summarize.py --tag rel_$S --study $S --split calib:3 --test_subjects $FULL \
    | tee ~/codabench/logs/sealed_rel_$S/RESULTS.md
python ~/codabench/analysis/release_summarize.py --tag rel_${S}_x --study ${S}_x --split calib:3 --test_subjects $FULL \
    | tee ~/codabench/logs/sealed_rel_${S}_x/RESULTS.md
```

Take the channel lines of the DECISIONS block from `rel_${S}_x` and everything
else from `rel_$S`, whose channel lines read `not run -> keep eeg`.

## 6. Pre-registered release-day rules (2026-09-28)

These rules were written before any Graz + BrainHero number existed.
- **What the summarizer does.** `release_summarize.py` applies the numeric part
  of rules 1–5 and prints a DECISIONS block. The parts marked *manual* below you
  check yourself.
- **Metric.** Every comparison uses the cell-averaged balanced accuracy on the
  replica test sessions (§ 5).
- **Uncertainty.** A paired subject bootstrap over the 10 fully labelled
  participants: 95 % CI, 10,000 resamples.

1. **Channels.** Keep a non-EEG channel set only if both conditions hold:
   - it beats EEG-only by ≥ 2 points and the CI excludes 0 (DECISIONS of
     `rel_${S}_x`);
   - *manual:* the official loader delivers those channels (§ 2). The summarizer
     does not check this.

   The drifting-EMG trap is exactly this case. On the full-size mock, EEG + EMG
   cost 3.7 points (CI −5.2 to −2.4; 0/10 subjects up) on this split (LOG
   2026-09-28 Phase 2). The organisers' split `replica:3`, whose training set
   holds the later sessions, had shown +0.8 on the small mock (review finding
   R1-3).
2. **Context alignment.** Adopt `router-psdctx` only if it beats `router-psd` by
   ≥ 2 points with the CI excluding 0.
   - **Deploying it.** Put `RECIPE_ALIGN=router-psdctx:riemann` in DEC.
     `train_sealed.sh` then bakes `align="subject_context"` into the candidate
     (since 2026-09-28, agent G). On `mock_sealed_s` this gave train = replay =
     harness = 0.547222.
   - **Small pairs.** A (subject, context) pair with fewer than `ctx_min` = 16
     training windows keeps its subject's whitening. Read the fit log's `pairs=`
     line (`train.log`).
   - **Not with ADAPT=online:** the solver raises NotImplementedError and
     `train_sealed.sh` refuses the combination. See rule 6.
3. **xDAWN.** Drop it (`RECIPE_SPEC=riemann:xd=0,fb=1`) if `xd=0` is ≥ 1 point
   better. Ties keep the recipe.
4. **Blend-weight CV.** Use `WCV=loso` with `BW=auto` unless LOSO is ≥ 2 points
   worse than `last`.
   - With `auto`, the solver chooses w by leave-one-calibration-session-out on all
     its own training data and stores it in the joblib.
   - Otherwise use `WCV=last` with `BW=harness`: the harness's weight on the
     replica is baked in.
5. **Pooling.** Keep "pooled on everyone" unless test-subjects-only is ≥ 2
   points better with the CI excluding 0.
   - The solver has no option to train on a subset. If the rule says
     `pool = test`, the candidate still trains on everyone (the summarizer prints
     a NOTE).
   - Record the missed gain in LOG as an open item.
6. **Online re-centring** (*manual*; rule-dependent). It stays off unless the
   organisers allow test-time statistics (SEALED_RECIPE § 5 #1). If they do:
   - **Choose the buffer N ∈ {32, 64, 128}** on the replica. The harness can only
     feed the test sessions in recording order. Run this in a background call,
     ~35 min at 120 Hz (untimed); use the decided `--family`, `--chans` and
     `--wcv`, and add `--mmap` on a 500 Hz cache:
     ```bash
     for N in 32 64 128; do
       python ~/codabench/analysis/sealed_personal.py --tag rel_${S}_online --study $S --split calib:3 \
           --test_subjects $FULL --router_cap 0.5 --wvariant calib --chans eeg \
           --family riemann:xd=1,fb=1 --align online-$N:riemann --wcv loso
     done > ~/codabench/logs/online_N_$S.log 2>&1
     grep "\[done\].*blend_calib|router-id" ~/codabench/logs/online_N_$S.log
     ```
     - If the organisers announce recording order, take the N with the best
       replica cell score.
     - If they announce interleaved or shuffled order, or say nothing, take
       N = 32. Under interleaving it kept the most gain on all four proxies (LOG
       2026-09-28 Phase 3), and no tool replays the replica in another order.
     - `analysis/sealed_stream.py` has the orders, but it cannot be used on the
       release cache as it stands. It has no `--test_subjects`, `--router_cap`
       or context cells, it ignores hidden rows in its own calib:K split, and it
       writes into the committed `logs/sprint0928_p3/`.
   - **Expected gains** (proxies, online minus clean):
     - at N = 64: +3.0 (Tangermann) and +4.4 (Scherer 3-class) with subjects
       interleaved, vs +3.5 and +7.5 in recording order;
     - at N = 32, interleaved: +3.3 and +6.0;
     - on Zhou, +0.5 to +3.5, never significant.
   - **Settings.** DEC gets `ADAPT=online RECIPE_ALIGN=online-<N>:riemann`.
   - **Precedence with rule 2** (added 2026-09-29, still before any release
     data): online and `router-psdctx` cannot be combined. If rule 2 adopted
     `router-psdctx`, keep it, unless both of the following hold:
     - the organisers confirm recording order;
     - the online-N row beats the `al_ctx` row on the replica by ≥ 2 points of
       cell score. There is no paired-bootstrap command for this pair:
       `release_summarize.py` compares each row with `base` only.
7. **Gates before a zip is kept.** `train_sealed.sh` enforces them through
   `GATE` (since 2026-09-29):
   - **`GATE=replica`** blocks the zip on any of: a failed replay, train ≠
     replay, a harness gap above 2 points (FLAG), or, with `BLEND_W=auto` and
     `WCV=loso`, a solver-vs-harness weight MISMATCH.
   - **`GATE=final`** blocks only on a failed replay or train ≠ replay. The gap
     and weight lines are recorded as information.
   - **A blocked run** writes `NO ZIP (GATE=…): <reason>` and `STOPPED: gate
     failed, no zip` to STATUS.md, renames any older zip in its folder to
     `*.zip.blocked`, and exits 1.
   - **`FORCE_ZIP=1`** overrides only a weight MISMATCH under `GATE=replica`, with
     a loud WARNING. Never use it for a candidate. Never record a blocked or forced
     zip in `SUBMISSIONS.md`.

   Check list (the `grep` lines in § 7):
   - **Step 1** (`logs/replica_$S/STATUS.md`, `GATE=replica`) needs all of:
     - `END replay rc=0`;
     - `gate: train X replay X EQUAL; … gap G OK`, i.e. the benchopt test score is
       within 2 points of the harness's blend_calib router-id pooled balanced
       accuracy;
     - with BW=auto, `blend weight: … MATCH`;
     - `zipped … (NOT uploaded)`.

     The solver trains without the loader's 2 % validation slice, which the
     harness keeps, so a MISMATCH by one grid step (0.25) can happen without a
     bug. Before calling it one, compare the solver's
     `blend_w=auto -> … cell scores [...]` line in `train.log` with the harness's
     `blend_calib: w_pooled=… (training CV [...])` line in `personal_*.log`.
   - **Step 2** (`logs/final_$S/STATUS.md`, `GATE=final`) needs:
     - `END replay rc=0`, `EQUAL` and `zipped … (NOT uploaded)`;
     - the `candidate …` line equal to DEC.

     The gap and weight lines compare a smoke slice and up-to-6-fold LOSO on all
     labelled data with the 3-fold calib:3 replica. FLAG or MISMATCH there is
     expected; record it, but it is not a failure.
   - **Step 3:** the unzipped candidate's read-only replay reproduces step 2's
     train score exactly, and the defaults in its `submission.py` are DEC.

   Any failure: stop and debug.

## 7. Train the candidate, replay, zip

Put the § 6 decisions into DEC and BW in the variable block first. Both runs set
`GATE` and `RUN_NAME` explicitly.

A run folder belongs to one set of settings. To rerun with a changed DEC (e.g.
after a failed gate), either use a new `RUN_NAME` (`replica2_$S`, and then
`HARNESS_FROM=replica2_$S` in step 2), or first move `logs/replica_$S` and
`logs/sealed_replica_$S` away. Otherwise the config guard stops with
`settings differ`.

**Step 1. Replica check through benchopt** (≤ ~80 min at 120 Hz). This is
like-for-like with the harness: the `_xsess` test set is the harness's replica
test set, and the same DEC goes in as in step 2, so the replica gates check the
settings you will ship.

```bash
env $DEC SPLIT=calib:3 TEST_SUBJECTS=$FULL BLEND_W=$BW GATE=replica RUN_NAME=replica_$S \
    DATASET="BCI[study=${S}_xsess]" bash ~/codabench/scripts/train_sealed.sh $DH $S
grep -E "gates blocking|candidate|END (train|replay)|gate:|blend weight:|NO ZIP|WARNING|zipped|ALL DONE" ~/codabench/logs/replica_$S/STATUS.md
grep -h '\[done\]\|blend_calib: w\|\[router\]' ~/codabench/logs/replica_$S/{validate,personal_*}.log   # the harness numbers
```

The zip of this run is not the candidate: its model never saw the fully labelled
participants' sessions 4–6.

**Step 2. Final candidate on all labelled data** (~30 min at 120 Hz):

```bash
env $DEC SPLIT=calib:3 TEST_SUBJECTS=$FULL BLEND_W=$BW GATE=final RUN_NAME=final_$S HARNESS_FROM=replica_$S \
    DATASET="BCI[study=${S}_all]" bash ~/codabench/scripts/train_sealed.sh $DH $S
grep -E "gates blocking|HARNESS_FROM|candidate|END (train|replay)|gate:|blend weight:|NO ZIP|WARNING|zipped|ALL DONE" ~/codabench/logs/final_$S/STATUS.md
```

- **`HARNESS_FROM=replica_$S`** seeds this run's harness folder
  (`logs/sealed_final_$S/`) with step 1's rows (`logs/sealed_replica_$S/`,
  results plus probs), so the harness steps skip every identical setting. They
  still run: `sealed_personal` refits its router, then skips.
  - Without it, `RUN_NAME` starts an empty harness folder, and preflight,
    validate and the personal step run again (≈ 55 min on the 120 Hz mock).
  - Without `RUN_NAME`, the config guard stops, because DATASET differs from
    step 1.
- **SPLIT and TEST_SUBJECTS** drive only the harness steps; benchopt trains on
  DATASET.
- **The weight.** With BW=auto the solver chooses w by LOSO over all its training
  sessions (up to 6 folds) and stores it in the joblib. STATUS's `blend weight:`
  line shows the value.
- **The zip** goes to `logs/final_$S/`.

**Step 3. Check the zip before the user uploads it** (~2 min):

```bash
Z=$(ls -t ~/codabench/logs/final_$S/riemann_sealed_${S}_*.zip | head -1); R=$(mktemp -d)
python -m zipfile -e "$Z" "$R" && chmod -R a-w "$R" && ls -l "$R"   # WSL has no unzip
grep -nE '^\s+"(personal|blend_w|adapt|use_xdawn|filterbank|kind|buffer|chans|align|ctx_min)": \[' "$R/submission.py"
cd ~/codabench/2026-competition
COMPET_SUBMISSION_DIR="$R" benchopt run tracks/bci_decoding -d "BCI[study=${S}_all]" \
    -s "$R/submission.py" --no-plot --no-html --no-cache --output zipcheck_$S
python -c "import glob, os, pandas as pd; p = max(glob.glob('tracks/bci_decoding/outputs/zipcheck_${S}*.parquet'), key=os.path.getmtime); print(os.path.basename(p), pd.read_parquet(p)['objective_balanced_accuracy'].iloc[-1])"
grep "gate:" ~/codabench/logs/final_$S/STATUS.md
```

Check each of these:
- **Two files at the zip root:** `riemann_sealed.joblib` and `submission.py`.
- **The grepped defaults equal DEC**, since Codabench runs the defaults:
  - `personal ["blend"]`;
  - `blend_w ["auto"]` or the baked number;
  - `use_xdawn` / `filterbank` from RECIPE_SPEC;
  - `kind`;
  - `align ["subject"]`, or `["subject_context"]` under router-psdctx;
  - `chans`;
  - `adapt ["none"]`, unless online is allowed.
- **The replay score equals step 2's `gate: train` score exactly.** This was
  rehearsed 2026-09-29 00:02:05–00:02:22 on the `mock_sealed_s` zip: 17 s,
  0.4833333 = `gate: train 0.483333`.

Then record the candidate in `SUBMISSIONS.md` as "not uploaded": date, DEC, zip
path, and the step-1 gate numbers. Uploading stays the user's action.

**Rehearsal on the mock** (for § 9). `split=replica_full` stands in for
`<name>_xsess`: its test set is the full participants' sessions ≥ 3, with hidden
rows excluded. The default split, which trains on every labelled window, stands
in for `<name>_all`. The DEC is the full-size mock's calib:3 DECISIONS (xDAWN
dropped, router-psdctx adopted; properties of the mock, not evidence) with
WCV=loso, so the `subject_context` bake and `auto` are exercised together.

```bash
S=mock_sealed_s; DH=~/neuralbench/benchopt_data; FULL=0,1,2,3,4,5,6,7,8,9
DEC="CHANS=eeg RECIPE_SPEC=riemann:xd=0,fb=1 RECIPE_ALIGN=router-psdctx:riemann WCV=loso"; BW=auto
env $DEC SPLIT=calib:3 TEST_SUBJECTS=$FULL BLEND_W=$BW GATE=replica RUN_NAME=replica_$S XS_THREADS=6 \
    DATASET="../datasets/mock_sealed.py[study=$S,split=replica_full]" bash ~/codabench/scripts/train_sealed.sh $DH $S
env $DEC SPLIT=calib:3 TEST_SUBJECTS=$FULL BLEND_W=$BW GATE=final RUN_NAME=final_$S HARNESS_FROM=replica_$S XS_THREADS=6 \
    DATASET="../datasets/mock_sealed.py[study=$S]" bash ~/codabench/scripts/train_sealed.sh $DH $S
```

Then run step 3 with `-d "../datasets/mock_sealed.py[study=$S]"` in place of
`-d "BCI[study=${S}_all]"`.

## 8. Dress-rehearsal timings (mock, 2026-09-28) and the release-day budget

**Setup.** The full-size mock study (`mock_sealed_120` / `mock_sealed_500`):
- 20 subjects × 6 sessions × 4 runs, 2 contexts: 14,400 windows of 4 s, 3
  classes;
- 47 channels in the cache, 43 EEG after the default pick (`chans=eeg`), 5,073
  features.

**Machine.** CPU-only i7-12700K (20 threads), 31 GB RAM in WSL.

**Contention.** Every run shared the CPU:
- the default flow ran beside the ablation lanes (22:11–22:52);
- the 500 Hz fit ran beside the auto flow (22:54–23:25);
- a 4-thread agent ran throughout.

The times are therefore upper bounds.

**Units.** RSS is /usr/bin/time's `Maximum resident set size` in kB (KiB) as
logged, then GiB. The solver's own `maxrss` prints are GiB too.

**Sources.**
- `logs/train_sealed_mock_sealed_120{,_auto}/STATUS.md`
- `logs/sealed_sprint0928_p2{,b}/`
- `logs/sealed_sprint0928_abl120/STATUS.md`
- LOG 2026-09-28 Phase 2

**Which split.** The `train_sealed.sh` rows ran on the mock's **organisers'
split** (`replica:3`): the harness and the solver trained on 10,800 windows, and
`auto` used 6 LOSO folds. The release-day replica `calib:3` trains the harness on
**7,200 windows with 3 LOSO folds** (the ablations' preflight: `train=7200`). § 5
and § 7 step 1 should therefore be faster than these rows. The final `_all`
solver fit (~10,800 windows, up to 6 folds) is the size measured here.

| Step | 120 Hz (the likely format) | 500 Hz (worst case) |
|---|---|---|
| **`train_sealed.sh` default flow** (fixed weight, WCV=last, 10 threads) | | |
| … preflight | 6 s | not run |
| … validate (`sealed_run`: MeanLogReg + recipe, pooled / per-subject, none / router) | 17 min 14 s | not run |
| … personal step (`sealed_personal`, WCV=last, `--wvariant calib`) | 15 min 32 s | not run |
| … benchopt training (solver fit on 10,800 windows: 399 s) | 7 min 31 s | not run |
| … read-only replay (3,600 test windows) | 48 s | 1:16.67 at 10 threads, 1:05.71 at 2 threads; 2,333,832 kB (2.2 GiB); 18.5–22.9 ms per window |
| **whole default flow** | **41 min 16 s** (22:11:13–22:52:29); peak 6,380,660 kB (6.1 GiB) | — |
| **`BLEND_W=auto` flow** (validate reused from the default flow) | | |
| … personal step with `--wcv loso` | 37 min 44 s (22:52:41–23:30:25) | not run |
| … benchopt training with `auto` | 22 min 13 s (23:30:25–23:52:38). Solver fit 1,290 s: stages 4.6 min + 6 LOSO folds 16.9 min (131–190 s each) | 31 min 41 s (22:54:07–23:25:48). Solver fit 1,826 s = 30.4 min: stages 8.5 min + 6 folds 21.9 min (178–222 s each) |
| … read-only replay | 41 s | as above |
| **whole auto flow** | **60 min 53 s** (22:52:31–23:53:24): ≈ 38 min harness `--wcv loso` + 22 min solver fit + replay; peak 6,346,756 kB (6.1 GiB) | peak 12,383,172 kB = 11.8 GiB (12.7 GB); solver print `maxrss 11.81` |
| **auto flow result** | train = replay = harness = 0.614722; solver w 0.75 = harness LOSO w 0.75, MATCH | train = replay (10 and 2 threads) = 0.612778; w 0.75 |
| **one ablation** (`release_ablations.sh`, calib:3, 5 threads per lane) | **10–14 min**: base 11.6, xd0 10.8, ch_eeg_eog 11.4, ch_eeg_emg 11.5, ch_all 13.6, al_ctx 10.1 min | not measured. The script runs its lanes one after the other above 2 GiB, because one 500 Hz `sealed_run` config was reported at 16.4 GiB in Phase 1 (agent L; no log kept) |
| **6 of the 10 ablations** | 58 min wall (21:55:44–22:54:00). Lane A ran 5 steps in series; lane B ran xd0 only (11 min). Peak 6,070,708 kB (5.8 GiB). Not timed at full size: wcv_loso, pool_test, online, run | — |
| EDA (`release_eda.py`, calib:3, 3 threads) | 2 min 38 s; 4,485,420 kB (4.3 GiB) | not run (estimate 10–12 GB peak) |
| zip check (§ 7 step 3; `mock_sealed_s`, 720 test windows) | 17 s | — |
| regression (§ 0; `zhou2016_xsess`, 2026-09-28) | 2 min 1 s | — |

**Sizing verdict** (brief rule: fit ≤ 45 min, peak RSS ≤ 20 GB, predict ≤ 10 min
at 10 threads and ≤ 30 min at 2 threads): the recipe **passes at 500 Hz** with a
wide margin: 30.4 min, 11.8 GiB, 77 s and 66 s.

The fast shrinkage LDA (sprint Phase 1) is what makes this feasible:
- one 5,073-feature fit took 38.7 s with sklearn's solve and 3.8 s with the
  Cholesky solve;
- the `auto` search refits 21 LDAs (the pooled one and one per subject) in each
  of its 6 folds.

**Release-day budget at 120 Hz**, from a finished download to a checked zip:

| Step | Compute | Basis |
|---|---|---|
| § 0 update + regression | ~15 min | regression flow 2 min 1 s |
| § 1 download + copy to Z: | not counted | 15–30 GB, network-bound |
| § 2 loader inspection (3×: organisers', `_xsess`, `_all`) | ~5 min | 9 s on zhou2016_xsess; release size untimed |
| § 3 two caches | 30–60 min (a guess) | not measured at release size; the four proxy caches took 2 min 16 s together |
| § 4 EDA, two caches | ~6 min | 2 min 38 s per cache |
| § 5 ablations, (a) and (b) side by side | 1.5–2 h | 6 of 10 steps timed (10–14 min each); (a) lane B holds 5 steps, 4 of them untimed (wcv_loso, pool_test, online, run) |
| § 7 step 1 (replica, `auto`) | 40–80 min | upper bound = the replica:3 flow (17 + 38 + 22 min + replay); calib:3 is smaller |
| § 7 step 2 (final, `HARNESS_FROM`) | ~30 min | solver fit with `auto` on ~10,800 windows: 22 min 13 s; harness rows reused (without `HARNESS_FROM`, +≈ 55 min) |
| § 7 step 3 zip check | ~5 min | 17 s replay on `mock_sealed_s` |
| rule 6 online N (only if allowed) | ~35 min | 3 `sealed_personal` runs, untimed |

That is **about 4–5.5 h of compute plus ~1 h of reading and deciding: plan
5.5–6.5 h**. This is still inside a day, and it assumes the cache build is not
much slower than guessed.

At 500 Hz, add time for serial lanes and larger harness steps (only the solver
fit and the replay were timed there): plan most of a day.

## 9. Verification log

Filled in by agent V1, who follows this runbook literally on the mock (sprint
2026-09-29). On the mock, set `S=mock_sealed_s` and `FULL=0,1,2,3,4,5,6,7,8,9`
(its meta `full_subjects`), and map the sections as follows:

| Section | On the mock |
|---|---|
| § 0 | as written |
| § 1 | skip: nothing to download |
| § 2 | run the snippet with `OV=zhou2016_xsess`; the mock is not a NeuralBench study |
| § 3 | skip: the cache exists (`python ~/codabench/analysis/mock_sealed.py mock_sealed_s` builds it) and already holds 47 channels, so `${S}_x` is `$S` itself |
| § 4 | both commands on `$S`: `--chans eeg`, then `--chans all` |
| § 5 | the "§ 2 showed 47 channels" branch: (a) without `STEPS`, no (b) |
| § 7 | the rehearsal block at the end of § 7, then step 3 on its zip |

| Step | Command (as run) | Result |
|---|---|---|
| | | |
