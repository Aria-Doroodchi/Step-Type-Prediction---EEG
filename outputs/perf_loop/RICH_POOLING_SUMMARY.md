# Rich-feature pooling sub-loop — SUMMARY

Branch `perf/agentic-improvements`. Tests the single highest-EV hypothesis left after the
main perf loop: **does cross-subject partial pooling on the RICH feature set produce a real
cohort-AUC gain — or at least make the project's best AUC honest?** The main loop established
partial pooling as the one lever that moves XGB, but ran entirely on the tractable 2.3k "fast"
feature set. Source of truth: [`LEDGER.md`](LEDGER.md) (RICH-POOLING SUB-LOOP section).

## Headline

**Yes — and it does both, modestly.** On the rich (~9.7k-column, no-src) feature set, partial
pooling reproduces the fast-set pattern at the project's *higher* AUC operating point: the
inner-vs-outer overfit **gap collapses (+0.198 → −0.039)** and held-out AUC is **+0.0386 paired**
over the matched per-participant arm (clears the +0.03 bar; t=1.17, not significant). vs the
project's recorded best (rich per-participant **0.655**, optimistic gap **+0.169**), the pooled
number (**0.6376**) is ~flat (−0.017, within noise) but **honest** (gap −0.039). Pooling makes
the best-AUC region trustworthy.

## Baseline → final (20-subject CONFIRM cohort, run `r_rich_conf20`)

| arm | cohort AUC | gap | note |
|---|---|---|---|
| recorded rich per-participant (heavy funnel + src, 5×20 CV) | **0.655** | **+0.169** | project's best per-subject AUC, optimistic gap |
| per_participant (matched: no-src, light funnel, 4-fold) | 0.5990 | +0.1978 | the clean paired baseline |
| **partial (rich pooled)** | **0.6376** | **−0.0385** | **+0.0386 paired (t=1.17); gap collapses** |

PRIMARY (held-out test AUC, paired same-fold): **+0.0386** (SE 0.033, t=1.17, 11/20 up).
GUARDRAIL (inner−outer gap): **+0.198 → −0.039** (Δ −0.236) — improves far past the +0.03 limit.

## The decision-tree branch we landed in

**"AUC ~flat (vs the recorded 0.655) but the gap COLLAPSES → promote the recommended HONEST
rich-pooled config,"** reinforced by a modest matched-arm AUC win (+0.0386, clears +0.03). It is
*not* the ≥0.67 clear-AUC-win branch, and *not* the ALL-NULL branch (it clears the screen→confirm
gate and collapses the gap). `full` (LOSO transfer, screen 0.594) < `partial` (0.665) → no single
transferable model; per-subject adaptation is required.

## What moved the numbers (and what did not)

- **Pooling's value is again mostly honesty, now at higher AUC.** The gap collapse (+0.198→−0.039)
  is the robust, reproducible effect — it holds at 8 and 20 subjects, on fast and rich features.
  The AUC lift (+0.0386) is real but modest and not significant, exactly like the fast set (+0.031).
- **Rich features absorb most — not all — of pooling's AUC headroom.** Matched per-participant
  jumps 0.567 (fast) → 0.599 (rich); the extra features capture cross-subject-shared signal that
  pooling otherwise supplies. Pooling still adds a comparable paired lift on top (fast +0.031,
  rich +0.0386), so the two levers are largely complementary, not redundant.
- **The 8-subject screen was PESSIMISTIC here (the unusual direction).** Its subset is enriched
  for strong subjects (P30/P02/P15/P12/P25) where per-participant already wins, so pooling looked
  null (+0.0156). The full cohort, with the harder subjects, showed the real +0.0386 — classic
  partial-pooling shrinkage helps the marginal subjects (P06 +0.22, P12 +0.31, P25 +0.26, P10
  +0.18) and slightly drags the stars (P15 −0.27, P13 −0.21). The confirm, not the screen, decides.

## Winners — config keys + legacy overrides

| Win | Key / artifact | Default (improved) | Legacy / opt-out |
|---|---|---|---|
| Rich partial pooling | committed overlay **`configs/pooling_rich.yaml`** (`modeling.pooling.mode: partial`) | partial (in the overlay) | `modeling.pooling.mode: per_participant` |
| Tractable rich funnel | **`modeling.pre_kbest`** (ANOVA pre-filter before correlation drop) | `2000` in the rich configs | global default stays `null` (off); set null in-overlay to disable |

**Judgment calls (flagged for review):**
- The **global `pre_kbest` default stays `null`.** It is a tractability/speed lever (AUC-neutral —
  the aggressive downstream funnel reproduces its selection), so flipping it globally would change
  the validated narrow-feature / per-participant path for no AUC gain. Baked into the rich configs
  where it matters. Flip globally only if rich/wide-feature runs become the norm.
- The **global `pooling.mode` default stays `per_participant`** (paradigm preservation, per the
  main loop). Rich pooling is a documented opt-in overlay.
- **`src` was not re-added.** The no-src result already lands the honest-config win; re-adding the
  ~2.6k source-space columns needs the eLORETA b0.125 caches for P35/P39 (or an 18-subject run),
  and the recorded numbers indicate source/binning move within-window AUC little. Lower-EV
  follow-up (below), not executed.

## Recommended next steps

1. **(Optional) add `src` back and re-confirm.** Build eLORETA b0.125 source caches for P35/P39
   (or accept an 18-subject src run), then re-screen/confirm `amplitude0.125+slopes+psd+src`.
   Promote only if src adds a *confirmed* lift over no-src. Expected small (source/binning move
   within-window AUC little), so this is incremental.
2. **Calibration inside the pooled CV** (`CalibratedClassifierCV`, OVERFITTING_GAP_SOLUTIONS §5.3) —
   with the gap now honest, calibrated probabilities are the natural next honesty improvement.
3. **Per-participant model selection / ensembling** — the scattered per-subject pooling deltas
   (stars lose, marginals gain) argue for choosing pooled-vs-own per subject via inner CV.
4. **CNN sub-loop** — unchanged recommendation from the main loop (cheap-headroom AUC levers).

## Reproducibility

- Screen: `r_rich_screen8` (8 subj, all 3 modes). Confirm: `r_rich_conf20` (20 subj, per_participant
  + partial). Both: `scripts/09_pooling_comparison.py --config configs/pooling_compare_rich.yaml`.
- Aggregate: `outputs/perf_loop/aggregate.py <run_dir>` (task-exact cohort AUC + gap).
- Paired delta: `outputs/perf_loop/paired_delta.py <run_dir> --a per_participant --b partial`.
- Train-time pooled rich model: `scripts/04_train.py --model xgb --config configs/pooling_rich.yaml`.
- `.venv` (Py 3.14), `PYTHONUTF8=1`. Feature caches keyed `{window}_{bin}_{cache_tag}`; the rich
  no-src frame uses `cache_tag: rich_nosrc_0125`.
