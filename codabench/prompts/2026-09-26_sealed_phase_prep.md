# Weekend session: prepare for the Track 2 sealed phase (cross-session, 3-class)

You are working unattended over a weekend. The user has approved the plan below and
will not answer questions until Monday. Work autonomously, decide the next run from the
previous run's result using the decision rules in this prompt, and never let a job go
stale (§ 6). Read this whole prompt before doing anything.

## 1. Context (all verified on 2026-09-25 unless marked)

- Competition: EEG/EMG Foundation Challenge 2026, Track 2 (BCI decoding). Work lives in
  `codabench/` of the repo `C:\Users\Ali D\Documents\ML`, branch `feat/codabench-track2`.
  Read first (10 min max): `codabench/TRACK2_BCI.md`, `codabench/REUSING_MY_MODELS.md`
  §§ 3b, 4, 6, and the two 2026-09-25 entries at the end of `codabench/LOG.md`.
- Compute is CPU-only in WSL Ubuntu (i7-12700K, 20 threads, 31 GB RAM). From the Bash
  tool run WSL with a heredoc:
  ```
  wsl.exe bash -s <<'EOF'
  source ~/codabench/env.sh >/dev/null
  cd ~/codabench/2026-competition
  ...
  EOF
  ```
  `~/codabench` is the repo's `codabench/` folder. Do not use the PowerShell tool for
  WSL commands (quoting breaks); do use PowerShell for robocopy and `powercfg`.
- **The sealed phase (Oct 28 – Nov 21) decides the ranking.** Same 20 people across six
  sessions; sessions 1–3 labelled (calibration), 4–6 hidden. 3 classes: kinesthetic
  motor imagery, mental calculation, word association. 43 EEG + 2 EMG + 2 EOG at
  500 Hz. Metric: balanced accuracy averaged over subject × session × context cells.
  `predict(X)` receives bare `(B, C, T)` windows: **no subject or session id at
  prediction time**. Its public training data (Graz + BrainHero 2026) is NOT released
  yet; check https://neural-interfaces26.github.io/tracks.html once per day and stop
  everything else if it appears (§ 5, Phase 6).
- The warm-up (Dreyer 2023, cross-subject left/right MI) is a proxy that does not
  count. Our warm-up model EEGNet-StepType-WU1 scores 0.820 (leaderboard 0.82; #1 is
  0.93). **Do not work on the warm-up leaderboard this weekend.**
- Already measured on 2026-09-25 (do not repeat): WU1 + Riemann probability ensemble
  = +0.8 points only; more training participants = no gain; REVE frozen probe likely
  below WU1 and full fine-tune 45 min/epoch on this CPU; Dreyer 2023 is in REVE's
  pretraining corpus; test windows reach `predict` in recording order (one contiguous
  block per participant, 81 % of 64-window batches single-subject on Dreyer).

## 2. Data on disk (verified 2026-09-25 17:31)

Master copy (network share, 45 MB/s, too slow for training):
`Z:\Projects\codabench\neural_compet\` = `/mnt/z/Projects/codabench/neural_compet/`
in WSL (mounted via `/etc/fstab`). Fast local copies in WSL:

| Study name (neuralfetch) | Local path | What it is | Cross-session? |
|---|---|---|---|
| `Dreyer2023Large` | `~/neuralbench/benchopt_data/neural_compet/` | L/R hand MI, 87 people, 27 ch, 120 Hz windows | no (cross-subject) |
| `Tangermann2012Review` (BNCI2014_001) | same | 4-class MI, 9 people × **2 sessions (different days)**, 22 ch | **yes** |
| `Scherer2015Individually` (BNCI2015_004) | `~/neuralbench/stage_z/neural_compet/` | 5 mental tasks, 9 people × **2 sessions**, 30 ch, 256 Hz | **yes** |
| `Zhou2016Fully` | same | 3-class MI (left, right, feet), 4 people × **3 sessions** × 2 runs | **yes** |
| `Zyma2019Electroencephalograms` | same | mental arithmetic vs rest, 36 people, 72 EDF (2 per person) | no (one session) |

Stieger 2021 is deliberately NOT downloaded (399 GB; the user wants to discuss it first).
Do not download it. Do not download pretrained model weights (REVE etc.) either: the
user has not approved that; leave those as documented next steps.

Phase 0 first copies the three `stage_z` studies into `benchopt_data/neural_compet/`
(they are small, 2.2 GB) so one `BENCHOPT_DATA_HOME` serves everything.

**Scherer 2015 is the closest sealed-phase proxy** if its classes contain word
association + mental subtraction + motor imagery. neuralfetch labels them
`math / letter / rotation / count / baseline`; the paper (Scherer et al. 2015, PLOS
ONE) describes word association, mental subtraction, spatial navigation, hand MI,
feet MI. Resolve the numeric-code → task mapping from `moabb/datasets/bnci/*.py`
(BNCI2015_004 loader and docstring) and the files under the study's `download/`
folder before using it. Record the mapping in `LOG.md`.

## 3. Goal and priorities

Produce, by the end of the weekend, `codabench/SEALED_RECIPE.md`: the evidence-ranked
training recipe for the sealed phase, plus the code that implements it on the proxy
datasets so that, when the Graz + BrainHero data drops, retraining is a one-command
job. Priority order (highest first):

1. A **cross-session evaluation harness** that mimics the sealed metric (§ 5 Phase 0).
2. **Per-subject / per-session alignment** without ids at predict time (Phase 2). This
   is the single most promising lever: the sealed phase is within-subject cross-session
   drift, the textbook case for re-centring, and it is the most plausible explanation
   of the 0.90+ cluster on the warm-up leaderboard.
3. **Pooling / personalisation** (Phase 3): pooled vs per-subject vs pooled + per-subject
   calibration vs blend, the thesis pooling question in the sealed setting.
4. **3-class mental-task generalisation** on Scherer 2015 (Phase 4).
5. **Cross-dataset pre-training** of the small models on the MI datasets, fine-tuned on
   Scherer per subject (Phase 5). This is what "pre-training" means here: CPU-feasible
   EEGNet-scale pre-training, not foundation-model training.

Warm-up-only cues (the slow lateralised left/right potential, xDAWN templates of it)
are not a priority; note where they inflate a proxy score.

## 4. Rules

- Branch `feat/codabench-track2`. Commit at every milestone (each phase's results),
  message prefix `codabench:`. Push to remote `personal` after each commit (OneDrive
  has reverted `.git` after power events before; a pushed commit cannot be lost). No
  tags. Never commit logs except `RESULTS.md`, `STATUS.md`, `HANDOFF.md`.
- Never touch `codabench/solvers/bci_decoding/eegnet_steptype_wu1.py` or anything under
  `codabench/submissions/`. Never upload to Codabench. Never write inside
  `2026-competition/tracks/bci_decoding/outputs/` by hand.
- Never tune on a test session; hyper-parameters come from earlier sessions or
  leave-one-subject-out over training subjects. A test-tuned number may appear only as
  a clearly labelled oracle upper bound.
- Every full run must print the data shape in its first fit log line; verify it matches
  the expected subjects × sessions × trials before believing a score.
- Differences below ~2 points on these small datasets (4–9 subjects) are noise: add
  seeds/folds before concluding, and say "no gain" plainly.
- Keep `PYTHONUTF8=1` for any Python on the Windows side. Keep runs under ~7 GB RSS
  each and never run more than two CPU-heavy jobs at once.
- Write `codabench/HANDOFF.md` (state, what is running, what is next) whenever the
  session has been going for a long time or before any long run: a later session must
  be able to resume from it alone.

## 5. Phases and decision rules

Each phase: estimate the wall time first, run in the background with the launch
pattern of § 6, time-log start/end from the log stamps, summarise into
`RESULTS.md` in the phase's log dir, append a dated entry to `codabench/LOG.md`
(estimate vs actual, what was learned), commit + push, then apply the decision rule.

### Phase 0 — harness (estimate 1.5–2 h, mostly code)

1. Machine prep: PowerShell `powercfg /change standby-timeout-ac 0`,
   `standby-timeout-dc 0`, `hibernate-timeout-ac 0`, `hibernate-timeout-dc 0`; record
   the previous values in `LOG.md`. Confirm WSL sees `/mnt/z` and the local data.
2. Copy `~/neuralbench/stage_z/neural_compet/*` into
   `~/neuralbench/benchopt_data/neural_compet/` (rsync; skip if present).
3. Add benchopt dataset overlays for cross-session use. Pattern: `_OVERLAYS` in
   `2026-competition/tracks/bci_decoding/datasets/bci_studies.py` names a yaml overlay
   in the neuralbench task's `datasets/` folder; `neuralbench/tasks/eeg/motor_imagery/
   datasets/zhou2016.yaml` already splits by session (`PredefinedSplit`, `col_name:
   split`), and `tasks/eeg/mental_imagery/config.yaml` (Scherer) and
   `tasks/eeg/mental_arithmetic/config.yaml` (Zyma) exist. Read them, then add overlays
   `tangermann2012_xsess`, `scherer2015_xsess`, `zhou2016_xsess`: earlier sessions →
   train (+ a session-based val where 3 sessions exist), last session → test.
   Keep window definitions identical to the task defaults so results are comparable.
4. Build a cross-session window cache per study, following
   `codabench/analysis/dreyer_eda.py` (X.npy, y.npy, subj.npy, plus **session.npy**
   and run.npy, meta.json with ch_names/sfreq). All alignment/pooling experiments run
   on these caches with plain Python (fast, no benchopt overhead); benchopt runs are
   only for final confirmation of a recipe.
5. Metric module: cell-averaged balanced accuracy (subject × session; there is no
   "context" in the proxies, note it) plus pooled balanced accuracy; per-subject table.
6. Runner: one script per phase in `codabench/scripts/`, `step` pattern from
   `scripts/riemann_blocks_2026-09-25.sh` (per-step `.done` marker, `STATUS.md` with
   `HH:MM:SS` rows, per-step log, timeout per step, resumable on re-run).
7. Smoke test everything on Zhou 2016 (smallest) before any long run.

Decision: Phase 0 is done when the Riemann baseline (xDAWN on, filter bank on) and
EEGNet-StepType each produce a cross-session score on all three proxies with the
expected data shapes. If Scherer's label mapping cannot be resolved, run Scherer with
all five classes and mark the 3-class question open.

### Phase 1 — cross-session baselines (estimate 1–1.5 h compute)

On Tangermann, Scherer, Zhou: MeanLogReg floor, Riemann-StepType (xDAWN on/off ×
filter bank on/off), EEGNet-StepType (100 epochs, patience 20, 3 seeds; early
stopping needs a held-out split: use the val session where one exists, else hold out
2 training subjects), upstream braindecode EEGNet. Two training modes each:
**pooled** (all subjects' early sessions) and **per-subject** (only that subject's
early sessions). Sizes are tiny (Tangermann ≈ 288 trials/subject/session), so a
Riemann fit is seconds and an EEGNet run minutes.

Decision: pick the stronger base family per dataset by cell-averaged score with seeds.
If pooled beats per-subject on ≥ 2 of 3 datasets, pooled is the default for Phases 2–3
(expected, per the thesis pooling result); otherwise carry both.

### Phase 2 — alignment (estimate 1.5–2 h)

Euclidean alignment (He & Wu 2020: R = mean over trials of X Xᵀ/T per group,
X ← R^(−1/2) X) and Riemannian re-centring (pyriemann `transfer.TLCenter` if it
imports; else implement). Groups = (subject, session). Conditions:
(a) none; (b) **oracle**: align train per (subject, session) and test per (subject,
session) using the test session's own unlabelled windows; (c) **train-side only**:
align train per group, test with the global training reference; (d) **router**: no
ids at test time; store one reference per training subject (their sessions pooled),
assign each test window (and, separately, each contiguous 64-window batch) to the
nearest training subject by Riemannian distance, re-centre with that reference,
fall back to the global reference when the distance exceeds a threshold set on
training data; report router accuracy (fraction of test windows assigned to the true
subject). (e) **batch-level**: align each contiguous 64-window test batch with its own
statistics (mechanically what `predict` sees; flag it as rule-dependent: check
`2026-competition/codabench/pages/terms.md` and note the verdict).

Decision rules: if (b) gains ≥ 2 points over (a) on ≥ 2 datasets, alignment is in
the recipe; then adopt (d) if router accuracy ≥ 90 % and (d) recovers ≥ 70 % of the
(b) gain, else adopt (c) and log why. If router accuracy is < 90 %, iterate at most
three times on the fingerprint (tangent-space log-var, per-band covariances, batch
voting) before falling back. If (b) gains < 2 points everywhere, alignment is out;
say so plainly and move on.

### Phase 3 — pooling / personalisation (estimate 1.5–2 h)

With the Phase 2 winner applied: (i) pooled; (ii) pooled + per-subject calibration
(Riemann: re-centre + refit LDA on the subject's own early sessions; EEGNet: fine-tune
the last layer on the subject's early sessions, 10–20 epochs); (iii) blend of pooled
and per-subject probabilities with the weight chosen by leave-one-subject-out on
training subjects (never on the test session); (iv) per-subject only. Personalised
variants need the router from Phase 2 at test time; report both oracle-id and
router-id versions.

Decision: the recipe takes the variant with the best cell-averaged score that wins on
≥ 2 of 3 datasets with router ids. Ties (< 2 points) go to the simpler variant.

### Phase 4 — 3-class mental tasks on Scherer 2015 (estimate 1 h)

Build the 3-class subset closest to the sealed classes (word association, mental
subtraction, hand MI, if the mapping allows) and run the Phase 3 recipe on it: are
the sealed classes separable cross-session with our features, and which feature
blocks carry it (ablate xDAWN / filter bank / log-var)? Also test Zyma 2019 as
"calculation vs rest" (36 people, cross-subject only) as a sanity check that the
calculation state is detectable at all.

Decision: if the filter-bank block carries most of the 3-class signal, make it the
default and drop xDAWN for the sealed recipe; if xDAWN still matters, keep both.

### Phase 5 — cross-dataset pre-training (estimate 3–4 h, the long one; run overnight)

Harmonise channels across Dreyer, Tangermann, Zhou, Scherer: intersect 10-20 names
(expect ~15–20 common ones; record them) or interpolate to a common montage with
MNE; resample to one rate. Pre-train EEGNet-StepType on the MI datasets (pooled,
all sessions; 3 seeds), then per Scherer subject: (a) train from scratch on their
early session, (b) fine-tune the pre-trained net on it, (c) fine-tune with the
Phase 2 alignment. Compare on the Scherer test session, cell-averaged. Do the
Riemann analogue: reference covariance from the pooled MI data vs from Scherer only.

Decision: if pre-training gains ≥ 2 points on Scherer with 3 seeds, it is in the
recipe and the harmonised channel set becomes the sealed default; otherwise record
"no gain from cross-dataset pre-training at this scale" and leave REVE/LaBraM as the
documented (not run) next step, with the 2026-09-25 feasibility numbers.

### Phase 6 — deliverable

`codabench/SEALED_RECIPE.md`: (1) the recipe, one paragraph; (2) the evidence table:
every phase's numbers with seeds, per dataset, cell-averaged and pooled; (3) what is
still untested until Graz + BrainHero arrives (EMG/EOG handling, 500 Hz, 47 ch, the
"context" cells) with an ablation plan; (4) `codabench/scripts/train_sealed.sh`
skeleton that takes a data path and reproduces the recipe end to end; (5) the
top-3 next steps ranked with expected gain, effort and risk. Update `HANDOFF.md`,
`LOG.md`, commit, push.

If the Graz + BrainHero data is released during the weekend: stop, download it to
`Z:\Projects\codabench\neural_compet\` (it is approved data), rerun Phase 0's cache
+ EDA (`dreyer_eda.py` parametrised by study) on it, and run the current best recipe
on it before anything else.

## 6. Keeping jobs alive and never stale (mandatory)

Known killers on this machine: WSL's VM shuts down when no `wsl.exe` stays attached
(a `nohup … &` inside a short `wsl.exe bash -s` heredoc died silently on 2026-09-25);
battery idle-sleep (10-min DC standby killed a run before); OneDrive reverting `.git`
after power events; Windows moving idle background work to E-cores (epochs slowed
12 s → 20 s overnight on 2026-09-24; harmless, but revise the ETA).

**Launch pattern** (this survived 13-min and 3-h runs): run the phase script in the
*foreground* of a Bash tool call with `run_in_background: true`, so `wsl.exe` stays
attached for the whole run:
```
wsl.exe bash -lc 'bash ~/codabench/scripts/<phase>.sh'
```
The script itself writes `STATUS.md` (`| HH:MM:SS | START/END step rc= |`), one log
per step, and `.done` markers. Before launching, write the ETA into `STATUS.md`
(`| HH:MM:SS | ETA <phase> ≈ N min, done by HH:MM |`).

**Watchdog** (build it in Phase 0 as `codabench/scripts/watch_run.sh <logdir>`; it
prints one line per step: state, elapsed vs ETA, seconds since the log's last write,
python PID alive + CPU %, free RAM/swap, and a verdict `OK | SLOW | STALLED | DEAD`):
- `STALLED` = the step's log file has not grown for 10 min (compare `stat -c %Y` to
  `date +%s`, the same check as watching a transcript's size/mtime).
- `SLOW` = elapsed > 1.5 × the step's ETA. `DEAD` = no python process for a step
  whose `.done` is missing and `STATUS.md` has no END row.

**Cadence:** after every launch, arm the `Monitor` tool on the watchdog for up to
30 min (its maximum), e.g. a loop that runs `watch_run.sh` every 5 min and prints a
line only when the verdict is not `OK` or a step ends; the loop must also match
`Traceback|Error|Killed|OOM|ALL DONE` in the logs so a crash is never silent. Re-arm
it when it expires, until `ALL DONE`. Do not wait passively between checks: write
docs, prepare the next phase's code, or analyse finished steps.

**When a check is not `OK`:** investigate before anything else. Read the last 20 log
lines; `ps -o pid,etime,%cpu,rss,cmd` for the python process; `free -g`; check that
WSL is still up (`wsl.exe -l --running`); check `powercfg` if the machine may have
slept; check the step is not silently on the wrong data (data shape in the fit line).
Fix the root cause (do not shrink the workload to dodge it), then re-run the phase
script: finished steps are skipped by their `.done` markers. If a step is legitimately
slower than estimated (steady progress, sane CPU), revise the ETA in `STATUS.md` and
`LOG.md` and continue. Log every incident in `LOG.md` (time, symptom, cause, fix).

**When a job is late but alive** (past its ETA, log still growing): apply the same
rule used for this prompt's own fact-finders: check the log's mtime and size, extend
the ETA once, and if it passes 2 × the original estimate, decide whether the finished
steps already answer the phase's question; if they do, proceed with them and let the
rest finish in the background rather than blocking on it.

## 7. Time budget and order

Phases 0 → 1 → 2 → 3 (Saturday), 4 → 5 (Saturday night, Phase 5 overnight), 6
(Sunday). Total compute ≈ 8–12 h; everything checkpointed. If you run out of session
context, `HANDOFF.md` + the `.done` markers must let the next session continue
without re-reading this conversation. Start with Phase 0 now.
