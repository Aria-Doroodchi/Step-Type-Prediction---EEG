# Project map

Durable facts about this repository: where each workstream's state lives, how to run
things, what not to touch, what has bitten us. Current numbers and "next steps" are
deliberately **not** copied here: read them live from the state files, because they
change every sprint. Verify anything here that looks off. It was written 2026-09-28.

## Workstreams

| Workstream | Where the work lives | State of record (read these) | Environment | Git |
|---|---|---|---|---|
| **Codabench Track 2** (EEG/EMG Foundation Challenge 2026, BCI decoding, cross-session 3-class) | `codabench/` | `codabench/HANDOFF.md`, `codabench/SEALED_RECIPE.md` (§ 5 ranked next steps), `codabench/RELEASE_DAY.md` (release-day runbook), tail of `codabench/LOG.md`, `codabench/TRACK2_BCI.md`, `codabench/SUBMISSIONS.md` (upload ledger) | WSL Ubuntu (see below) | branch `feat/codabench-track2`, commit prefix `codabench:` |
| **Thesis step-type pipeline** (straight vs diagonal step from CNV, per-participant nested CV; XGB / CNN / EEGNet) | `src/eeg_steptype/`, `scripts/0*_*.py`, `configs/` | `outputs/perf_loop/LEDGER.md` + `SUMMARY.md` (perf loop, complete), `README.md` § Results, `XGB_MODEL_SUMMARY.md`, `MODELS.md`, `SCRIPT_GUIDES.md`, `WORKFLOW.md` | Windows venvs | merged into `main`; new work goes on a fresh feature branch off `main` |
| **3-class motor-state module** (standing / straight / diagonal) | `src/eeg_statetype/`, `configs/state/`, `scripts/state_module/`, `run_state.py` | `outputs/state_module/LEDGER.md`, reports in `outputs/reports/state_3class_*` | Windows `.venv` | merged into `main` |
| **Stim / SEP module** (foot-sole e-stim evoked potentials) | `scripts/stim_module/`, `configs/stim.yaml` | `outputs/stim_module/LEDGER.md` (append-only) | Windows `.venv` | merged into `main` |
| **Public repo docs** (reviewers are sent to this public repo; numbers must match the user's resume) | `README.md`, `CHANGELOG.md`, `docs/` | `README.md` § Results, `CHANGELOG.md` | none | changes reach `main` only via PR, and only when the user asks |

Older notes (CLAUDE.md, memories) name branches such as `perf/agentic-improvements`
and `feat/stim-module`. Those were merged and deleted, so check `git branch -a`
before trusting a branch name.

Deliverable conventions: each workstream keeps an append-only LOG or LEDGER (dated
entries, estimate vs actual, decision) and one "state" doc that a new session can
resume from (`HANDOFF.md` in `codabench/`; the LEDGER plus its SUMMARY for the
thesis modules). Sprint briefs go to `codabench/prompts/` or `docs/sprints/`.

## External facts and sources (Track 2)

- Dates: registration closes **Oct 24 2026**, warm-up ends Oct 25, **sealed phase
  Oct 28 – Nov 21 2026** (the only phase that ranks). 5 submissions/day in warm-up,
  **1/day sealed**; the top 3 go through a reproducibility audit (± 2σ replay). Source:
  `codabench/COMPETITION.md`.
- Sealed data: Graz + BrainHero, 20 people × 6 sessions, 43 EEG + 2 EMG + 2 EOG at
  500 Hz; 10 training participants fully labelled, 10 evaluation participants with
  sessions 1–3 labelled. Release page (check at most once per day):
  https://neural-interfaces26.github.io/tracks.html
- Rules: `codabench/2026-competition/codabench/pages/terms.md` and
  `competition_bci.md`. Test-time use of unlabelled test windows is **not settled**
  (see SEALED_RECIPE § 5; the question is the user's to post).
- Leaderboard: public Codabench API; how it was read and the raw snapshot are in
  `codabench/logs/explore_2026-09-25/desk/`.

## Environments and how to run things

**WSL (all Codabench work).** Ubuntu in WSL2, user `ali_d`, CPU only: i7-12700K,
20 threads, 31 GB RAM, no NVIDIA GPU. From the Bash tool:

```bash
wsl.exe bash -s <<'EOF'
source ~/codabench/env.sh >/dev/null      # venv ~/neuralbench/.venv (py3.12, torch cpu, neuralbench, benchopt)
cd ~/codabench/2026-competition
...
EOF
```

- `~/codabench` is a symlink to the repo's `codabench/`, so in WSL the repo root is
  `/mnt/c/Users/Ali D/Documents/ML` (quote the space).
- Data: `BENCHOPT_DATA_HOME=~/neuralbench/benchopt_data` (fast local copy); master
  copy on the network share `Z:\Projects\codabench\neural_compet\` =
  `/mnt/z/Projects/codabench/neural_compet/` (45 MB/s, too slow to train from).
  Cross-session window caches live in `~/neuralbench/xsess_cache/<study>/`.
- The upstream competition repo is cloned (gitignored) at
  `codabench/2026-competition/`; local edits there are not versioned.
- Use the PowerShell tool only for Windows-side commands (`robocopy`, reading
  `powercfg`), never to drive WSL (its quoting breaks).

**Windows (thesis pipeline).** `.venv` = Python 3.14 (classical: XGB, sklearn);
`.venv312` = Python 3.12 + TensorFlow (neural only). Call them explicitly
(`.venv/Scripts/python.exe`) and always set `PYTHONUTF8=1`; a θ glyph crashes cp1252
console logging. Pipeline entry points: `scripts/00_preflight.py` …
`scripts/09_pooling_comparison.py`, `run_state.py`, Makefile targets (`smoke`,
`preflight`, `features`, `train`, `full-xgb`).

## Reusable tooling (look here before writing new code)

- Cross-session harness (Track 2): `codabench/analysis/xsess_cache.py` (window caches
  with subject/session/run/context), `xsess_lib.py` (cell-averaged metric, models,
  alignment, subject router), `sealed_run.py` (configs × modes × alignments →
  results_<study>.jsonl), `sealed_personal.py` (pooled/calib/blend variants),
  `sealed_pretrain.py`, `sealed_summarize.py`, `sealed_decide.py`,
  `sealed_decide_p3.py`, `sealed_bootstrap.py` (paired subject-level CIs),
  `router_eval.py`.
- Release day (sprint 2026-09-28):
  - runbook `codabench/RELEASE_DAY.md`;
  - mock of the sealed structure `analysis/mock_sealed.py` (caches `mock_sealed_s` /
    `_120` / `_500`) + benchopt dataset `datasets/mock_sealed.py` (path form only,
    `split=organisers|replica_full`);
  - `scripts/release_ablations.sh`, `analysis/release_summarize.py` (the
    pre-registered rules), `analysis/release_eda.py`;
  - `analysis/sealed_stream.py` (online re-centring under test-order variants).
- Feature blocks (sprint 2026-09-29):
  - harness `analysis/xfeat.py` + `xfeat_temporal.py` / `xfeat_spatial.py`,
    spec `riemann:xd=1,fb=1,x=<id>` (self-test `xfeat_selftest.py`);
  - `--split first`, the reverse-time replication;
  - `analysis/f0929_summarize.py` (screen/confirm rules, paired bootstrap);
  - solver opt-in `xblocks` (tseg<K>, bpt<K>, icoh);
  - `analysis/xblocks_gate.py` (defaults bit-identity + harness parity);
  - `analysis/xblocks_sizing.py` (500 Hz sizing through the mock dataset
    class);
  - `release_ablations.sh` steps `xb` / `xa` (env `XB`).
- Sprint 2026-10-01:
  - the dual (n < p) shrinkage LDA (`LDA_DUAL` in both solver files; checker
    `analysis/lda_dual_check.py`);
  - strict CV whitening references (`WREF=strict` in train_sealed.sh /
    release_ablations.sh, harness `--wref`, solver `wcv_ref`);
  - `analysis/compare_results.py` (two runs' rows key by key: scores,
    weights, |dP|; the gate for "numerically equivalent change");
  - `analysis/ctxmin_check.py`;
  - `analysis/joblib_compat_check.py` (a joblib under other
    sklearn / numpy / pyriemann versions in throwaway venvs);
  - `analysis/pc_screen.py` (template for a harness-only variant screen with
    a built-in baseline check);
  - lanes `scripts/sprint1001_*.sh`.
- Runners: `codabench/scripts/sealed_lib.sh`, `watch_run.sh`, `monitor_loop.sh`,
  `sealed_p*.sh` (examples of lane scripts), `train_sealed.sh` (release-day pipeline).
  This skill bundles generic copies in `scripts/`. Quote their path, which contains a
  space:
  - WSL: `"/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/"`
  - Git Bash, from the repo root: `.claude/skills/project-sprint/scripts/`

  The Monitor call, e.g.:
  `wsl.exe bash -lc 'bash "/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/monitor_loop.sh" ~/codabench/logs/<tag> 300'`.
- Solvers: `codabench/solvers/bci_decoding/` (`riemann_sealed.py` = the sealed
  recipe; `riemann_steptype.py`, `eegnet_steptype.py`).
- Thesis: the perf-loop harness and `scripts/09_pooling_comparison.py`, which is
  **not checkpointed**.

## Frozen artifacts: do not modify

- `codabench/submissions/` and `codabench/solvers/bci_decoding/eegnet_steptype_wu1.py`
  (the frozen warm-up submission).
- `codabench/2026-competition/tracks/bci_decoding/outputs/`: benchopt writes it; never
  edit it by hand.
- Delivered reports under `outputs/reports/` (supervisor report, the 3-class "SEP"
  report).
- The cited thesis run `outputs/runs/xgb_full_full_cnv_20260612_093700` (the 0.714
  headline). Never overwrite or rerun into its folder.

## Data and download approvals

- Pre-approved: the organisers' Track 2 datasets on the tracks page **except**
  Stieger 2021 (399 GB, the user wants to discuss it first), and Graz + BrainHero once
  released (download to `Z:\Projects\codabench\neural_compet\`, then copy to WSL).
- Not approved: pretrained model weights (REVE, LaBraM, …) and anything large not
  listed here. Ask first.
- Disk: C: (which holds the WSL disk) has ~100 GB free; keep ≥ 30 GB for Windows.

## Git conventions

- Commit on the workstream's feature branch, never on `main`. End messages with the
  co-author trailer from the current system reminder.
- Push to `personal` right after every commit (OneDrive can roll `.git` back after a
  power event; a pushed commit survives). Push to `org` only when the user says so
  ("sync both accounts").
- No tags, merges or PRs unless asked (tags are semver on `main` after a merge,
  bumped in `pyproject.toml`, `CITATION.cff` and the README together).
- Log directories are gitignored. Commit only `RESULTS.md`, `STATUS.md` and
  `HANDOFF.md` from them (`git add -f`). Never commit big regenerable artifacts
  (`outputs/runs/`, perf-loop logs, caches).

## Known traps (each has cost real time here)

| Trap | Fix |
|---|---|
| WSL's VM shuts down when no `wsl.exe` stays attached; `nohup … &` inside a short `wsl.exe bash -s` died silently | Run the phase script in the *foreground* of a Bash tool call with `run_in_background: true` |
| Two parallel jobs each spawning 20 OpenBLAS threads: a 9 s fit took > 150 s | Export `OMP/OPENBLAS/MKL_NUM_THREADS` per job (`step_lib.sh` does it) |
| Two lanes appending to one file on `/mnt/c`: lost rows, risk of interleaved JSON | One results file per lane or study; readers skip malformed lines |
| `python3` / `python` in Git Bash is the Microsoft Store alias and hangs | Use WSL's python or the explicit `.venv/Scripts/python.exe` |
| `pkill -f pattern` matched its own `bash -c` command line and killed itself | Kill by PID (`ps`, `pgrep`, then `kill <pid>`) |
| Multi-line heredocs with quotes inside the Bash tool broke parsing | Write files with the Write tool; keep shell heredocs simple |
| Watchdog reported STALLED because the job logged only at the end of each config | Print a heartbeat line every N epochs or items |
| `--config <missing file>` silently falls back to `default.yaml` (30 participants) | Check participant count and feature count in the first log lines |
| Feature cache key omits the block list | Bump `features.cache_tag` when the blocks change |
| loky `parallel >= 2` deadlocked the CNN overnight | `--parallel-participants 1` for neural models |
| `scripts/09_pooling_comparison.py` writes its CSV only at the end | Watch it closely; an overrun can lose the whole run |
| Machine sleep killed a 2 h run; OneDrive rolled `.git` back after a power event | Read `powercfg` before long runs; push after every commit |
| Codabench runs a solver's **default** parameters | Bake the chosen settings into a candidate copy (see `train_sealed.sh`) |
| A hyper-parameter chosen under one condition, deployed under another (blend weight: router 0.75 vs online 0.5, −3 points) | Choose hyper-parameters under the deployment condition |
| The same config run in two phases counted as two seeds | Deduplicate result rows by config key |
| Windows moves idle background work to E-cores overnight (epochs 12 s → 20 s) | Expect ~1.7× slower nights; revise ETAs instead of treating it as a stall |
| Bash reads a running script from disk as it goes: an agent editing `train_sealed.sh` while a lane ran it would corrupt the job (2026-09-28) | Never edit a `.sh` a running job uses; stage the change as a patch and apply it after the job ends |
| A lane step that wraps an inner lane script (`/usr/bin/time -v … release_ablations.sh`) logs nothing until it ends, so the watchdog said STALLED at 15 min (2026-09-28) | Watch the inner script's own log dir, or give the wrapper a heartbeat |
| Two lanes appending to one STATUS.md on `/mnt/c` lost START rows (2026-09-28) | Wait on `.done`/`.start` files, not on STATUS.md rows |
| benchopt caches results by dataset *parameters*, not data: after rebuilding a cache under the same study name a rerun silently returned the old score | Pass `--no-cache` after any cache rebuild (train_sealed.sh does) |
| A Workflow script interpolates `${VAR}` inside agent prompt template literals (a runbook's `${S}` crashed a launch) | Write shell variables in prompts as `<S>` or escape them as `\${S}` |
| WSL `git` in `2026-competition` (a Windows checkout) shows ~82 CRLF-only diffs; `git stash` would take them all | Use `git -c core.autocrlf=true` there |
| A Monitor pipe `wsl.exe … \| tr -d '\r' \| grep --line-buffered …` delivered no events for 30 min: `tr` block-buffers into a pipe (2026-09-29; a lane sat idle 10 min) | `stdbuf -oL tr -d '\r'` (or drop `tr`) in every Monitor pipe |
| Git Bash rewrites `/mnt/c/…` arguments passed to `wsl.exe bash -lc '…'` (MSYS path conversion), so a scratchpad path arrived empty (2026-09-30) | Pass commands through stdin: `wsl.exe bash -s <<'EOF' … EOF` |
| Non-ASCII text (—, ×, ≤) in a heredoc piped into `wsl.exe` did not match the file's UTF-8 bytes, and the Bash tool collapsed `\\` to `\` in a heredoc (2026-09-30) | Write patch scripts with the Write tool (UTF-8) and run them from WSL; use the Edit tool for docs with non-ASCII |
| `python3` / `python -` in Git Bash hung a tool call (the Store alias again) | Never call python outside WSL / the explicit venv path, even for a one-liner |
| `xsess_lib.read_results(path)` returns the rows of every `results_*.jsonl` in that folder, so a summary keyed without the study silently overwrote rows (2026-10-01) | Filter or key on `r["study"]` |
| Below 4,000 features the solver kept every LDA's p × p `covariance_`: a 52-subject Dreyer candidate was a 2.6 GB joblib (2026-10-01) | Fixed (dropped at every size); check the joblib size of any new candidate |
| A wrapper lane of an inner script that ends with a "not done" row returned rc 0 while one rule's row was missing (the summarizer filter bug, 2026-10-01) | Read the summarizer's MISSING lines after every ablation run, not only the rc |
| Edits to a `.py` / `.sh` that a lane will import later change what later steps test | Stage edits as patch scripts (scratchpad) and apply them between lanes; dry-run each patch on copies first |
