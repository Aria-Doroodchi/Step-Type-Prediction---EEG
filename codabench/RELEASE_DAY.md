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

1. **Two overlays**, written into `config/nb_overlays/<task>/` following
   `config/nb_overlays/motor_imagery/zhou2016_xsess.yaml`. Register both for
   benchopt: add their names to `scripts/install_xsess_overlays.sh`, then re-run
   that script.
   - **`<name>_xsess`, the replica.** `test_split_query` selects the ten fully
     labelled participants' sessions 4–6 (in the study's own session and subject
     labels). The cross-session cache and the benchopt replica check (§ 7 step 1)
     both use this overlay, so benchopt's test set is exactly the harness's
     replica test set.
   - **`<name>_all`, the final training set.** Every labelled window goes to
     train, and the test split is a 2 % random slice (a smoke check only).
     Skip this overlay if the organisers' own selector already trains on all
     public labelled data.
2. **Build the cross-session cache** (every window + subject/session/run/context +
   the release structure):

   ```bash
   python ~/codabench/analysis/xsess_cache.py <name> --task eeg/<task> --overlay <name>_xsess \
       --sealed --eval_subjects <the 10 evaluation participants' indices> --calib_sessions 3 \
       --hidden_labelled
   ```

   Subject indices are positions in the cache's sorted subject list
   (`meta["subjects"]`). The evaluation participants are the ones with only 3
   labelled sessions. `--hidden_labelled` tells the harness that split-2 rows
   (here: the fully labelled participants' sessions 4–6) carry real labels and
   may be tested on. The build logs a
   `release structure: hidden (split 2) rows=... eval_subjects=... full_subjects=...`
   line. Check it, plus the final line:
   - 20 subjects;
   - 6 sessions for the ten training participants, 3 for the ten evaluation
     participants;
   - 3 classes;
   - the channel count from § 2;
   - `context_column` in meta.

   If the study has no `context`/`condition`/`paradigm` column, find the column
   that holds Graz/BrainHero and add it to `xsess_cache.py`'s list before building.
3. **EMG/EOG EDA only.** This is not for the submission unless § 2 showed those
   channels in the official loader. Build a second cache under another study key,
   same command plus `--picks eeg,emg,eog` (e.g. `<name>_x`). Channel types in meta
   come from the channel names (EMG/EOG/ECG in the name: `ch_types_source=names`),
   which is the same rule the solver applies on Codabench. Compare the `[data]`
   line's `ch_types=` counts with the expected 43/2/2: channels whose names do not
   say EMG/EOG (e.g. `ExG1`) are typed EEG on both sides.

## 4. EDA (30 min)

From the cache:
- windows per subject × session × context cell, and the class counts per cell
  (uniform LDA priors assume balance within a cell);
- how contexts are distributed: within sessions (runs), or one context per
  session;
- router accuracy **per test session** on the replica. The log-PSD router
  falls from 0.72 to 0.35 across two later sessions when there is one calibration
  session (LOG 2026-09-28 Phase 3). With three calibration sessions it should
  hold better, but check sessions 4, 5 and 6 separately. The `router_acc` of the
  § 5 rows is the average; the per-window assignments are in the probs files
  (`router_assign`);
- the evoked response to the cue per class (xDAWN's premise), as in
  `analysis/dreyer_eda.py`.

## 5. Replica validation and the § 3 ablations

The fully labelled participants provide the only labelled later sessions, so the
internal replica of the sealed split is **`SPLIT=calib:3
TEST_SUBJECTS=<the ten fully labelled participants' indices>`**: every subject's
sessions 1–3 are training, and the fully labelled participants' sessions 4–6 are
scored. `replica:3` (the organisers' own split) is refused on the released data;
it is only for gates (iii)/(iv) on the mock. `train_sealed.sh` refuses a
sealed-looking cache when `SPLIT` is unset. It never falls back to the proxy
split silently.

```bash
S=<name>; FULL=<comma list of the 10 fully labelled participants' indices>
DH=~/neuralbench/benchopt_data
# 1. the recipe on the replica; stop to read the harness numbers first
SPLIT=calib:3 TEST_SUBJECTS=$FULL STOP_AFTER=personal_none \
    bash ~/codabench/scripts/train_sealed.sh $DH $S
cat ~/codabench/logs/train_sealed_$S/STATUS.md
# 2. every SEALED_RECIPE section-3 ablation, two resumable lanes
#    (a background Bash call keeps WSL attached; see scripts/sealed_lib.sh)
TAG=rel_$S STUDY=$S SPLIT=calib:3 TEST_SUBJECTS=$FULL bash ~/codabench/scripts/release_ablations.sh
python ~/codabench/analysis/release_summarize.py --tag rel_$S --study $S \
    --split calib:3 --test_subjects $FULL
```

What each ablation answers, and the rule that decides it, is in
`scripts/release_ablations.sh` (header) and § 6. The ablations cover:
- channel sets;
- subject vs (subject, context) alignment;
- xDAWN on/off;
- blend-weight CV last vs LOSO;
- pooled on everyone vs on the test subjects only;
- clean vs online-64 (rule-dependent, reported apart).

Every harness run starts with a preflight `[data]` line (`hidden_in_train=0` is
enforced). Read it.

## 6. Pre-registered release-day rules (2026-09-28)

These were written before any Graz + BrainHero number existed.
`release_summarize.py` applies rules 1–5 itself and prints them as a
`DECISIONS` block. All comparisons are cell-averaged balanced accuracy on the
replica test sessions (§ 5), using a paired subject bootstrap: 10 fully labelled
participants, 95 % CI.

1. **Channels.** Keep a non-EEG channel set only if both hold:
   - it beats EEG-only by ≥ 2 points *and* the CI excludes 0;
   - the official loader delivers those channels (§ 2).

   The drifting-EMG trap is exactly this: on the mock, EEG+EMG costs 4.2 points
   on this split while looking harmless on another.
2. **Context alignment.** Adopt `router-psdctx` only if it beats `router-psd` by
   ≥ 2 points with the CI excluding 0. The solver then trains with
   `align="subject_context"` (`RECIPE_ALIGN=router-psdctx:riemann`).
3. **xDAWN.** Drop it if `xd=0` is ≥ 1 point better; ties keep the recipe.
4. **Blend-weight CV.** Use `WCV=loso` / `BLEND_W=auto` (the solver chooses w by
   leave-one-calibration-session-out, stores it, and the harness's LOSO weight
   must match it) unless LOSO is ≥ 2 points worse than `last`.
5. **Pooling.** Keep "pooled on everyone" unless test-subjects-only is ≥ 2 points
   better with the CI excluding 0.
6. **Online re-centring** stays off unless the organisers allow it. If they do:
   - choose the buffer N ∈ {32, 64, 128} on the replica under the order the
     organisers announce (interleaved → N = 32 kept the most gain on all proxies);
   - expect +3 to +4 points interleaved and up to +7.5 in recording order.
7. **Gates before zipping:**
   - benchopt train = read-only replay, exactly;
   - on the `<name>_xsess` overlay, the benchopt test score is within 2 points
     of the harness's blend_calib (router-id) pooled balanced accuracy;
   - with `BLEND_W=auto`, STATUS says `MATCH` for solver vs harness weight on the
     replica.

   Any failure means stop and debug; never zip a candidate that failed a gate.

## 7. Train the candidate, replay, zip

1. **Replica check through benchopt** (like-for-like with the harness):

   ```bash
   SPLIT=calib:3 TEST_SUBJECTS=$FULL BLEND_W=auto DATASET="BCI[study=${S}_xsess]" \
       bash ~/codabench/scripts/train_sealed.sh $DH $S
   cat ~/codabench/logs/train_sealed_${S}_auto/RESULTS.md   # gate table
   ```

2. **Final candidate on all labelled data.** Use the § 6 decisions as env, e.g.
   `CHANS=eeg RECIPE_SPEC=riemann:xd=1,fb=1 RECIPE_ALIGN=router-psd:riemann`, and
   `ADAPT=online` only if allowed. Give it its own `RUN_NAME`.

   ```bash
   RUN_NAME=final_$S SPLIT=calib:3 TEST_SUBJECTS=$FULL BLEND_W=auto \
       DATASET="BCI[study=${S}_all]" bash ~/codabench/scripts/train_sealed.sh $DH $S
   ```

   The harness steps are the same as in step 1 (they are skipped as done if the
   settings match). Training uses every labelled window. The zip lands in
   `logs/final_$S/`, and the solver's chosen weight is in STATUS.md.
3. **Before the user uploads it:**
   - unzip it into a fresh folder;
   - run one read-only replay from that folder (as `train_sealed.sh`'s replay
     step does), and check that `submission.py`'s defaults are the baked ones
     (Codabench runs defaults);
   - record the candidate in `SUBMISSIONS.md` as "not uploaded".

   Upload stays the user's action: 1 submission/day in the sealed phase.

## 8. Dress-rehearsal timings (mock, 2026-09-28/29)

All runs used the full-size mock study: 20 subjects × 6 sessions × 2 contexts,
14,400 windows of 4 s, 43 EEG channels after the default pick, and 5,073
features. The machine is a CPU-only i7-12700K. Every run shared the CPU with one
other 10-thread job and a 4-thread agent, so the times are upper bounds. Logs are
in `logs/sealed_sprint0928_p2*/`, `logs/train_sealed_mock_sealed_120*/` and
`logs/sealed_sprint0928_abl120/`.

| Step | 120 Hz (the likely format) | 500 Hz (worst case) |
|---|---|---|
| `train_sealed.sh`, harness weight: preflight + validate | 17 min | — |
| … personal step (blend weight, `--wvariant calib`) | 16 min | — |
| … benchopt training (solver fit 10,800 windows) | 7.5 min | — |
| … read-only replay (3,600 test windows) | 50 s | 77 s at 10 threads, 66 s at 2 threads |
| **whole default flow** | **41 min** | — |
| solver fit with `blend_w="auto"` (6 LOSO folds) | see LOG | **30.4 min** (weight search 21.9 min), peak RSS 12.4 GB |
| one release ablation (`sealed_personal`, 5 threads) | 9–12 min each | ~2× the harness memory: one sealed_run config peaked at 16.4 GB, so run lanes one after another |
| 6 ablations, two lanes of 5 threads | 58 min | — |

**Sizing verdict** (brief rule: fit ≤ 45 min, peak RSS ≤ 20 GB, predict ≤ 10 min
at 10 threads and ≤ 30 min at 2 threads): **passes at 500 Hz** with a wide margin.
Inference needs 2.3 GB and ~20 ms per window. Train = replay held exactly in every
run (0.612778 at 500 Hz; 0.614722 at 120 Hz). The fast shrinkage LDA (sprint
Phase 1) is what makes this feasible: with sklearn's SVD solve, one fit at 5,073
features took 39–86 s, and the LOSO search needs ~150 of them.

**Release-day budget at 120 Hz:** freeze + regress 15 min, download (unknown),
cache ~10 min, EDA 30 min, replica recipe 35 min, ablations ~1.5–2 h (all 11
steps, two lanes), candidate training with `auto` ~45 min, replay + zip 5 min.
That is **about 4–5 hours of compute** plus reading time, well inside a day.

## 9. Verification log

TBD
