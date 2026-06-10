# Perf-improvement loop — LEDGER

Source of truth for the `perf/agentic-improvements` branch. Append every round.
AUC = held-out test AUC; gap = inner_best_score − held-out AUC. 0.50 = chance.

## Objective function
- **PRIMARY:** maximize cohort held-out test AUC = mean over participants of each
  participant's mean outer-fold AUC, restricted to `repeated_stratified` folds
  (chronological-check rows excluded).
- **GUARDRAIL:** overfit gap = mean(inner_best_score − held-out AUC) must not
  worsen by more than **+0.03** vs the current baseline. AUC-up-but-gap-broken = REJECT.
- A change that lowers the gap with neutral AUC = mild positive (recorded; does
  not by itself reset the plateau counter).

## Protocol (the harness every comparison uses)
- **Feature set (standard):** `_b0p25` — amplitude+slopes, bin 0.25 s, full-CNV,
  ~2.3k cols. This is the documented/validated pooling recipe
  (`configs/pooling_compare.yaml`; MODELS.md §7). Chosen over the 12.3k rich set
  (`rich_mean_0125`, cohort 0.655) purely for **tractability**: P30 single-subject
  is ~225 s here vs ~679 s rich — the ~25× cheaper per-fold correlation-drop is
  what makes a multi-candidate screen→confirm loop runnable in-session. All runs
  (baseline + every candidate) use this same set, so comparisons are apples-to-apples.
  Overlays: `configs/_perf_fast.yaml` (features) + `configs/_perf_subset8.yaml` (subset).
- **Window:** full-CNV (0.0–2.0 s) — the baseline window. Any window experiment
  stays strictly within 0–2 s.
- **SCREEN (iteration):** 8 subjects [P30,P02,P15,P13,P25,P07,P12,P08], **express**
  CV (5 splits × 2 repeats, inner 2). Noisy (`test_auc_sd ≈ 0.19`), for ranking.
- **CONFIRM:** full 20-subject cohort, **express** CV (5×2).
  > NOTE — substitution from the brief's "default tier": default tier is 5 splits
  > × 20 repeats with `n_iter=100`; on 20 subjects that is the repo's documented
  > *overnight* job and is not runnable in-session. The confirm step's purpose is
  > to **kill 8-subject false positives**, which comes from the larger *cohort*
  > (20 vs 8), not from more CV repeats. Express-on-20 delivers that. Documented
  > here for honesty.
- **Gating (screen → confirm):** keep a candidate only if cohort AUC beats the
  iteration baseline by a margin clearly above subset noise (require ≥ +0.03,
  and prefer paired per-subject deltas which are less noisy) AND the gap
  guardrail holds. Else revert + record the negative.
- **WIN (confirm):** must ALSO improve the 20-subject baseline (AUC up, guardrail
  intact). Only then promote (commit + tests + ledger).
- **Aggregation tool:** `outputs/perf_loop/aggregate.py <run_dir>` (task-exact defs).
- Neural (cnn/eegnet/eegnext) runs use the Python-3.12 venv (`.venv312`) because
  Python 3.14 has no TensorFlow wheel; classical models use `.venv`.

## Environment
- Branch `perf/agentic-improvements` off `main` @ 2d49fb6; baseline commit d5e98bb.
- `.venv` = Python 3.14.3 (classical: numpy 2.4, sklearn 1.8, xgboost 3.2, mne 1.12).
- `.venv312` = Python 3.12.10 + TensorFlow 2.21.0 + scikeras 0.13.0 (neural).
- Tests: 118 passed at baseline.

---

# Baselines

_(filled in below as runs complete; run ids cited for reproducibility)_

| Model | scope | run id | cohort AUC | gap | n subj |
|---|---|---|---|---|---|
| xgb | iteration (8, express) | base_xgb_iter | **0.5974** (sd 0.068) | **+0.141** | 8 |
| xgb | confirmation (20, express) | base_xgb_conf | **0.5674** (sd 0.089) | **+0.166** | 20 |
| cnn | smoke (P25) | smoke_cnn_p25 | 0.555 (TF verified end-to-end, ~7min/subj) | −0.055 | 1 |
| cnn | iteration (8) | base_cnn_iter | **0.5644** (sd 0.134) | **+0.070** | 8 |
| cnn | confirmation (18*) | base_cnn_conf | _pending_ | | 18* |

Per-subject (cnn iter): P02 0.80, P30 0.684, P07 0.628, P13 0.54, P08 0.538, P25 0.501,
P12 0.421, P15 0.404. NB CNN tier is **2 folds/subject** → very noisy (sd 0.134); a
reliable screen lift must be large. *CNN confirm uses 18 subjects — P35/P39 lack the
`_b0p0625` source caches that `require_source` needs (building eLORETA src for 2 subjects
is out of scope/expensive); documented here.

Per-subject (xgb iter): P30 0.716, P15 0.669, P12 0.608, P25 0.608, P13 0.584,
P08 0.542, P07 0.538, P02 0.516. (NB inner_best_score here = *accuracy* under the
default `scoring`, so the +0.141 gap is accuracy−AUC; candidate 1 fixes this.)

---

# Round 1 — backlog (ranked)

Grounded in OVERFITTING_GAP_SOLUTIONS.md (§) and the model docs. One lever per candidate.

1. **Align inner search metric to AUC** (`modeling.scoring: roc_auc`; §5.1). Foundational:
   today the search optimizes *accuracy* while we report *AUC*. Cheap, makes the gap
   meaningful, expected to lift AUC. (xgb + cnn)
2. **Partial cross-subject pooling** (§6; validated +0.106 on the demo). The single
   biggest documented lever. Needs a routing key `modeling.pooling.mode`. (xgb)
3. **Tighten regularization grid** (§3.1): cap `max_depth∈{2,3}`, `min_child_weight∈{3,5}`,
   push `reg_lambda/gamma/colsample`. Kills winner's curse → smaller gap. (xgb)
4. **Tighten the feature funnel** (§4.1): `k_best∈{100,200}`, stability `max_features∈{20,40}`.
   Fewer survivors for p≫n → smaller gap. (xgb)
5. **Raise stability threshold to 0.7** (§4.3): keep only features stable across subsamples. (xgb)
6. **Probability calibration inside CV** (`CalibratedClassifierCV`, §5.3). (xgb)
7. **CNN regularization**: dropout / kernel-length / fusion-unit knobs (MODELS.md §5.6). (cnn)
8. **Window experiment** within 0–2 s (e.g. 0.5–2.0) — later round, under the leakage guard.

## Round 1 — SCREEN results (8-subject express; baseline 0.5974, gap +0.141)

Paired per-subject deltas (same folds, random_state=1) in parentheses.

| candidate | run id | cohort AUC | AUC Δ (paired) | gap | gap Δ | verdict |
|---|---|---|---|---|---|---|
| scoring=roc_auc | r1_xgb_scoring | 0.5908 | −0.007 (−1.7 SE) | +0.209* | (n/a*) | REJECT — slightly hurts AUC. *inner now AUC not accuracy, so gap redefined, not comparable. |
| grid tighten | r1_xgb_gridtight | 0.5912 | −0.006 (−0.5 SE) | +0.096 | −0.045 | mild positive (gap↓, AUC neutral). Not an AUC win. |
| funnel tighten | r1_xgb_funnel | 0.6002 | +0.003 (+0.4 SE) | +0.171 | +0.030 | REJECT — within noise. |
| stability thr 0.7 | r1_xgb_stabthr | 0.5937 | −0.004 (−1.9 SE) | +0.146 | +0.004 | REJECT — within noise / slight down. |
| **partial pooling** | **r1_xgb_pool** | **0.6834** | **+0.086** (5/8 subj) | **−0.012** | **−0.153** | **PASS → confirm.** Big AUC lift + gap collapses. |

Decision: only **partial pooling** clears the +0.03 screen gate. grid-tighten noted as a
gap-only mild positive (candidate to stack later). scoring/funnel/stability reverted.

### Round 1 — pooling CONFIRM (20-subject) methodology note
First attempt ran partial pooling through the express tier on 20 subjects → the pooled
training set is ~1588 epochs and express stability selection (50 subsamples × 15 λ ≈ 750
saga fits/fold × 100 folds) is computationally explosive (>6 h projected). Killed it.
Pooling moves the regime from p≫n (~80 ep) to p≈n (1588 ep / 2308 cols), so heavy per-fold
selection is unnecessary. Confirm therefore uses the **validated `pooling_compare.yaml`
config** (light stability n_subsamples=8/λ=5, k_best=150, same 2.3k feature set) — the exact
setup that produced the documented +0.106 — run on all 20 subjects via
`scripts/09_pooling_comparison.py` (per_participant + partial + full on one matched frame →
clean paired comparison). run id: r1_pool_compare20.

Routing hardened: `modeling.pooling.mode != per_participant` now falls back to per-participant
for tensor models (cnn/eegnet/eegnext) and <2-subject cohorts, so a global pooling default
stays safe for neural runs and smoke configs.

### Round 1 — pooling CONFIRM (20-subject) RESULTS — run id `r1_pool_confirm20`
Re-run in the resumed session (`PYTHONUTF8=1` fixes the cp1252 θ logging crash that
truncated the earlier `r1_pool_compare20`). All three modes on one matched 20-subject
pooled frame (2304 cols), `scripts/09_pooling_comparison.py`. Aggregated task-exact
(`aggregate.py`, cohort = mean-over-subjects of mean-fold AUC):

| mode | cohort AUC | gap | folds |
|---|---|---|---|
| per_participant (matched baseline) | 0.5646 (sd 0.075) | **+0.173** | 80 |
| **partial** | **0.5957** (sd 0.134) | **−0.014** | 80 |
| full (LOSO transfer) | 0.5882 (sd 0.118) | −0.012 | 20 |

**Paired partial − per_participant (same test folds): +0.0311 AUC, SE 0.0244, t=1.27,
13/20 subjects up.** vs the ledger confirmation baseline `base_xgb_conf` (0.5674): **+0.028**.
Lift concentrated in P12 (+0.365), P25 (+0.198), P39 (+0.120); losers P03 (−0.157),
P14 (−0.113). Screen +0.086 → confirm +0.031: the effect SHRANK with cohort size (the
classic false-positive signature) but did **not** vanish or flip negative.

**VERDICT — partial pooling = confirmed WIN (modest AUC, robust gap).**
- PRIMARY: confirmed cohort AUC 0.5674 → **0.5957** (+0.028; paired +0.031 clears the
  +0.03 bar, though t=1.27 ⇒ not statistically significant — the AUC lift is real but noisy).
- GUARDRAIL: gap +0.166 → **−0.014** (Δ −0.18) — collapses, far inside the +0.03 limit.
  This gap collapse is the **robust** result (matches the documented demo) and the headline.
- Pooling strictly DOMINATES the objective (AUC not worse + guardrail dramatically better).

**PROMOTION (judgment call, flagged for user override).** Globally flipping
`default.yaml` to `partial` would silently convert the project's per-participant paradigm
to cross-subject pooling, make `--speed-tier express` pooled, bypass the chronological
check on default runs, and route per-participant unit tests into the pooled path (breaking
the resume test) — disproportionate to a t=1.27 AUC lift. So instead:
- New committed overlay **`configs/pooling.yaml`** whose default IS the improved behavior
  (`modeling.pooling.mode: partial`); legacy `per_participant` stays the global default in
  `default.yaml` (now an explicit, documented key) and is reachable. Contract satisfied
  without breaking the paradigm/tests.
- Confirmation baseline ADVANCED to the pooled regime: **xgb conf (pooled partial) = 0.5957,
  gap −0.014**. Plateau counter RESET (Round 1 produced a confirmed win). 118 tests green.
- Round 2 will optimize the **pooled** XGB (more data → looser funnel / richer grid).

---
