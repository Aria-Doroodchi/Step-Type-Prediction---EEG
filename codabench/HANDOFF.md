# HANDOFF — sealed-phase prep (weekend 2026-09-25 → 09-27)

Resume from this file alone. The plan is the weekend prompt (commit d3019b7,
pasted in the session); phases 0–6, decision rules, launch/watchdog pattern.
Branch `feat/codabench-track2`, push to `personal` after every commit.

## State (updated 2026-09-25 22:25)

| Phase | Status | Where |
|---|---|---|
| 0 harness | ✅ done (caches, harness, overlays, watchdog, smoke) | LOG.md 2026-09-25 evening entry |
| 1 baselines | ✅ done 19:37 (commit 18af79d): Riemann xDAWN+FB is the base family; per-subject > pooled on 2/3 | `logs/sealed_p1/RESULTS.md` |
| 2 alignment | ✅ done 21:07 (08ea8c3): clean router recovers little; online re-centring (rule-dep.) ≥ oracle | `logs/sealed_p2/RESULTS.md` |
| 3 pooling / personalisation | ✅ done 22:14 (c22c19c): **blend_calib** best/tied on all 3 | `logs/sealed_p3/RESULTS.md` |
| 4 Scherer 3-class | ⏳ lane A (ablation + personal) since 21:56, ~23:15; zyma already done (`logs/sealed_p4/zyma_early.log`); lane B (EEGNet 3-class) still to launch: `LANES=B bash scripts/sealed_p4.sh` | `logs/sealed_p4/` |
| 5 pre-training | ⏳ lane A (raw pre-train ×3 + fine-tune + Riemann ref) since 22:14, ~23:45; lane B (`LANES=B bash scripts/sealed_p5.sh`) when p4 lane A ends | `logs/sealed_p5/` |

Slot plan: 2 heavy jobs max. After p4 lane A → p5 lane B; after p5 lane A →
p4 lane B; then Phase 6 (end-to-end `train_sealed.sh` test on zhou2016_xsess,
SEALED_RECIPE.md).

Phase 6 (SEALED_RECIPE.md): draft exists; fill sections 1, 2, 5 from the RESULTS files.

Router (no ids at test): log-PSD LDA (`SubjectRouter("psd")`), per-window
cross-session subject accuracy Zhou 1.000, Tangermann 0.965, Scherer 0.866
(3 fingerprint iterations; see LOG). Phase 2 decision table:
`python ~/codabench/analysis/sealed_decide.py --tag p1 --tag p2`.

## How to run / resume

All in WSL (`wsl.exe bash -s <<'EOF' ... EOF` from the Bash tool; no PowerShell
for WSL, no `python3` on the Windows side: it is the Store alias and hangs).

- Caches: `bash ~/codabench/scripts/sealed_p0_caches.sh` (skips built ones) →
  `~/neuralbench/xsess_cache/{zhou2016,tangermann2012,scherer2015,zyma2019}/`.
- A phase: `wsl.exe bash -lc 'bash ~/codabench/scripts/sealed_pN.sh'` from the Bash
  tool with `run_in_background: true` (keeps wsl.exe attached). Finished steps
  are skipped by `<step>.done`; finished configs by `results_<study>.jsonl` keys.
- Watchdog: `bash ~/codabench/scripts/watch_run.sh ~/codabench/logs/sealed_pN`;
  Monitor event stream: `bash ~/codabench/scripts/monitor_loop.sh <logdir> 300`.
- Summary: `python ~/codabench/analysis/sealed_summarize.py --tag pN`.
- One config by hand: `python ~/codabench/analysis/sealed_run.py --tag x --study
  zhou2016 --models riemann:xd=0,fb=1 --modes pooled --aligns router:riemann`.
- Every step log starts with `[data] ... X=(n, C, T) subjects=...`: check it.

## Facts a resumer needs

- Scherer 2015 codes: 1 WORD, 2 SUB, 3 NAV, 4 HAND, 5 FEET (from the .mat
  `classes` field; MOABB names are wrong). Cache labels 0..4 in that order;
  sealed-like 3-class subset = `--classes 0,1,3`.
- Thread caps are mandatory for parallel lanes (`sealed_lib.sh` exports
  OMP/OPENBLAS/MKL_NUM_THREADS = XS_THREADS=10); without them two lanes thrash.
- Batch-level alignment (`batch:*`, `routerb:*`) is rule-dependent (see LOG.md);
  report it, do not adopt it without the organisers' answer.
- Do not touch `solvers/bci_decoding/eegnet_steptype_wu1.py`, `submissions/`, or
  `2026-competition/tracks/bci_decoding/outputs/` by hand. Do not upload.
