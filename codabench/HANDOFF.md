# HANDOFF — sealed-phase prep (weekend 2026-09-25 → 09-27)

Resume from this file alone. The plan is the weekend prompt (commit d3019b7,
pasted in the session); phases 0–6, decision rules, launch/watchdog pattern.
Branch `feat/codabench-track2`, push to `personal` after every commit.

## State (updated 2026-09-25 23:55): all phases done

| Phase | Status | Result (details: LOG.md 2026-09-25 entries) |
|---|---|---|
| 0 harness | ✅ 3c78929 | caches for 4 proxies, cross-session harness, overlays, watchdog |
| 1 baselines | ✅ 18af79d | Riemann xDAWN + filter bank is the base family; per-subject > pooled on 2 of 3 |
| 2 alignment | ✅ 08ea8c3 | clean router recovers little (affine invariance); online re-centring (rule-dep.) ≥ oracle |
| 3 personalisation | ✅ c22c19c | **blend_calib** best/tied on all 3 |
| 4 sealed-like 3 classes | ✅ 0816dc1 | separable: 0.49 clean / 0.56 online (chance 0.33); FB carries it; xDAWN not measurable there, kept |
| 5 pre-training | ✅ 43809a9 | no gain at this scale |
| 6 deliverable | ✅ b1ff2b8 | **[SEALED_RECIPE.md](SEALED_RECIPE.md)**, `solvers/bci_decoding/riemann_sealed.py`, `scripts/train_sealed.sh` (verified end to end on zhou2016_xsess, clean and online) |

Nothing is running. Nothing uploaded. `codabench/submissions/` and the WU1
solver untouched.

## What the user needs to do (Monday)

1. Read `SEALED_RECIPE.md` § 1 and § 5.
2. Decide whether to post the drafted organiser question (§ 5): it gates the
   biggest measured lever (+3.5 to +7.5 points).
3. Optional: approve REVE/LaBraM weight download (not needed for the recipe).

## If the weekend session continues

- Check https://neural-interfaces26.github.io/tracks.html once per day (last
  check 2026-09-25 18:53: "coming soon"). If Graz + BrainHero is released:
  download to `Z:\Projects\codabench\neural_compet\` (approved), copy/rsync into
  `~/neuralbench/benchopt_data/neural_compet/`, write a benchopt overlay (pattern
  in `config/nb_overlays/motor_imagery/`, install with
  `scripts/install_xsess_overlays.sh`), then
  `bash ~/codabench/scripts/train_sealed.sh ~/neuralbench/benchopt_data <study> eeg/<task> <overlay>`,
  and the § 3 ablations (EMG/EOG, contexts, replica split on the 10 training
  participants).
- Otherwise the cheapest useful experiment is next step 3 (Riemann + pooled
  EEGNet ensemble on Scherer 3-class, weight on held-out calibration data).

## How to run / resume

All in WSL (`wsl.exe bash -s <<'EOF' ... EOF` from the Bash tool; no PowerShell
for WSL, no `python3` on the Windows side: it is the Store alias and hangs).

- Caches: `bash ~/codabench/scripts/sealed_p0_caches.sh` (skips built ones) →
  `~/neuralbench/xsess_cache/{zhou2016,tangermann2012,scherer2015,zyma2019}/`
  (+ `zhou2016_xsess` from the end-to-end test).
- A phase: `wsl.exe bash -lc 'bash ~/codabench/scripts/sealed_pN.sh'` from the Bash
  tool with `run_in_background: true` (keeps wsl.exe attached). `LANES=A|B` runs
  one lane. Finished steps are skipped by `<step>.done`, finished configs by
  `results_<study>.jsonl` keys.
- Watchdog: `bash ~/codabench/scripts/watch_run.sh ~/codabench/logs/sealed_pN`;
  Monitor event stream: `bash ~/codabench/scripts/monitor_loop.sh <logdir> 300`.
- Summaries: `analysis/sealed_summarize.py --tag pN`, `sealed_decide.py` (Phase 2),
  `sealed_decide_p3.py` (Phase 3), `sealed_bootstrap.py` (subject-level CIs).
- One config by hand: `python ~/codabench/analysis/sealed_run.py --tag x --study
  zhou2016 --models riemann:xd=1,fb=1 --modes pooled --aligns router-psd:riemann`.
- Every step log starts with `[data] ... X=(n, C, T) subjects=...`: check it.

## Facts a resumer needs

- Scherer 2015 codes: 1 WORD, 2 SUB, 3 NAV, 4 HAND, 5 FEET (from the .mat
  `classes` field; MOABB names are wrong). Cache labels 0..4 in that order;
  sealed-like 3-class subset = `--classes 0,1,3`.
- Thread caps are mandatory for parallel lanes (`sealed_lib.sh` exports
  OMP/OPENBLAS/MKL_NUM_THREADS = XS_THREADS=10); without them two lanes thrash.
- `adapt="online"` and batch-level alignment are rule-dependent (test-time
  statistics); off by default until the organisers answer.
- The solver's online mode needs per-(subject, session) training centring; it
  reads session ids from the NeuralBench trigger table under the loader
  (`session_ids=yes` in the fit line). Without them it underperforms on
  multi-session data.
- Codabench runs a solver's DEFAULT parameters: `train_sealed.sh` bakes the
  chosen settings into a candidate copy (`logs/train_sealed_<study>/riemann_sealed_cand.py`).
- Do not touch `solvers/bci_decoding/eegnet_steptype_wu1.py`, `submissions/`, or
  `2026-competition/tracks/bci_decoding/outputs/` by hand. Do not upload.
