# Directions: Track 2 at a glance

**Updated 2026-10-01** after the 2026-09-29/30 sprint. This is the short version
for making decisions. Every number links to where it comes from. Read the long
documents only if you want the detail (§ 7). Kept current by every session that
changes a decision, a number or the plan (§ 8).

## 1. Where we stand

- **Calendar:**
  - registration closes **Oct 24**;
  - the warm-up (5 uploads/day, does not rank) ends **Oct 25**;
  - the **sealed phase, Oct 28 – Nov 21**, is the only phase that ranks:
    1 upload/day, top 3 audited.
- **Data:** the sealed training data (Graz + BrainHero) is still *"coming soon"*
  (checked 2026-09-30 01:10). Every number below comes from public stand-in
  datasets with 4–9 people each. Differences under ~2 points are noise.
- **Uploaded so far:** one warm-up model (EEGNet, 0.82 on the warm-up
  leaderboard; that phase doesn't rank).
- **Sealed-phase candidate:** **Riemann-Sealed + bpt4**. It retrains end to
  end on release day with the runbook (6–7 h, [RELEASE_DAY.md](RELEASE_DAY.md)).
- **Headline on the stand-in closest to the sealed tasks** (Scherer: word
  association / subtraction / hand imagery, 3 classes, chance 0.333):
  - **0.529** balanced accuracy, up from 0.486 before this week;
  - **0.592** if test-time re-centring is allowed (see decision 1).
  - Source: [SEALED_RECIPE § 2](SEALED_RECIPE.md).

## 2. Decisions waiting on you

| # | Decision | My recommendation | Why it matters | Your effort |
|---|---|---|---|---|
| 1 | **Ask the organisers** whether `predict()` may use statistics of unlabelled test windows, and in what order they arrive | Post it now (wording ready in [SEALED_RECIPE § 5](SEALED_RECIPE.md)) | Online re-centring is the largest lever we have: **+3 to +7.5 points**. We can't use it unless allowed | one forum post |
| 2 | **Deployment-test upload** during warm-up | Do it before Oct 25 ([SUBMISSIONS.md](SUBMISSIONS.md) has the command) | Our sealed model (sklearn/pyriemann) has never run on the scoring server. The sealed phase hides logs and allows 1 try/day | ~30 min training + 1 upload |
| 3 | **Keep bpt4 by default on release day** unless the real data shows ≥ 1 point better without it | Accept | It is the one solid gain this week (+4.4). Rule 3b in [RELEASE_DAY § 6](RELEASE_DAY.md) | read one rule |
| 4 | **icoh on 500 Hz data**: it fails the 1.5× compute gate (34 vs 19 min fit) but meets the older 45-min rule | Accept the 45-min rule: ~+15 min per training run is affordable | icoh may add ~2 points (+6.5 together with bpt4, post-hoc) | decision only |
| 5 | Optional: approve REVE/LaBraM weights; merge the thesis docs branch `docs/sprint0928-consistency` | Low priority | REVE/LaBraM are hours per epoch on this CPU | — |

## 3. The model in one picture (Riemann-Sealed + bpt4)

```
4-s EEG window, 43 channels (EMG/EOG dropped unless the release-day rule admits them)
 │
 ├─ 1 Router       log power 1–45 Hz → "which participant is this?"
 │                 (87–100 % correct on stand-ins; when unsure → shared path)
 ├─ 2 Re-centre    whiten with that participant's mean covariance
 │                 (removes person- and session-specific offsets)
 ├─ 3 Features     ~5,800 numbers per window (at 43 ch):
 │                   filter-bank covariances 4–8/8–13/13–30/30–45 Hz → tangent space  3,784   most of the signal
 │                   broadband covariance → tangent space                              946
 │                   xDAWN covariances (class templates) → tangent space               300
 │                   log-variance per channel                                           43
 │                   bpt4: band power per channel in 4 one-second bins                 688   new, +4.4
 └─ 4 Classifier   shrinkage LDA on everyone (pooled), blended with the participant's own LDA;
                   blend weight chosen on their calibration sessions
```

- **What is not in the sealed model:** the neural network (EEGNet), which
  scored 6–12 points below this pipeline on every stand-in.
- **Full explanation:** [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md). This
  draft from another session (2026-09-29, not yet committed) predates bpt4.

## 4. What we know

| Lever | Effect on the stand-ins | Status |
|---|---|---|
| Personal + pooled LDA blend | +1 to +12 points vs pooled only | in the model |
| Router + per-person re-centring (no test data) | up to +5 pooled, ~0 per-person | in the model |
| **Online re-centring** (uses unlabelled test windows) | +3.5 to +7.5; +3.0 to +4.4 if test windows arrive shuffled | **built, waiting on decision 1** |
| Filter bank (4 bands) | carries most of the 3-class signal | in the model |
| **bpt4**: when in the trial each band's power changes | +4.35 (CI +2.05, +6.67), positive on 4 of 4 stand-ins | **in the model since 09-29** |
| **icoh**: lagged connectivity between channels | +2.2 alone; +6.5 with bpt4 (post-hoc) | release-day test; compute issue at 500 Hz |
| xDAWN (class templates) | +4–5 on motor imagery, ~0 on the sealed-like tasks | kept; first to drop if the real data disagrees |
| No gain | EEGNet; pre-training on other datasets; more participants; re-referencing / Laplacian; CSP; regional covariances; band-power maps; time-segment covariances (under the deployed model); 8–30 Hz band-pass | dropped |

- **Where the brain-activation differences are** (training sessions only,
  [activation EDA](reports/features_0929/activation_eda.md)):
  - alpha, posterior-right, after ~1.5 s;
  - a word-association beta decrease at F3;
  - early frontal slow activity that looks like eye movement (kept out of
    the model: its bands start at 4 Hz).
- Differences are strong *within* each person but differ between people, which
  is why personalisation matters.

## 5. Bottlenecks and risks

1. **No real data yet.** Every choice is a proxy result on 4–9 people. The
   release-day ablations re-decide the open points on the real data with
   pre-registered rules.
2. **Rule uncertainty.** Online re-centring is neither allowed nor forbidden,
   and the test-window order is unknown (decision 1).
3. **Deployment.** The sealed model's file format has never run on the scoring
   server (decision 2).
4. **Compute (CPU only).** Each added feature makes the per-person LDAs slower
   (cubic in the feature count). At 500 Hz: the recipe fits in 19 min, + icoh
   in 34 min ([RELEASE_DAY § 8](RELEASE_DAY.md)). This caps how many features
   we can add.
5. **Few windows per person.** About 360–720 calibration windows (the mock's
   structure; the real counts are unknown) against ~6,000 features. Heavy shrinkage keeps it stable, but new blocks must be cheap and
   strong.
6. **Release-day time.** The runbook takes 6–7 h, and the release date is
   unknown with the sealed phase on Oct 28.

## 6. Next steps worth exploring (ranked)

| # | Step | Expected gain | Effort | Needs |
|---|---|---|---|---|
| 1 | Decisions 1 and 2 above | unlocks +3 to +7.5; removes the deployment risk | minutes | you |
| 2 | **Faster per-person LDA** (solve in the n < d form) | no accuracy change; makes icoh (and bigger feature sets) cheap at 500 Hz | ~3 h | nothing |
| 3 | **icoh restricted to 2 bands** (8–13, 13–30 Hz), screened and sized | keeps most of icoh's gain at ~half the cost | ~3 h | nothing |
| 4 | Strict per-fold blend-weight search | correctness: removes ~2 points of bias when choosing the weight | 2–3 h | nothing |
| 5 | Riemann + EEGNet probability ensemble | +1 to +2 | half a day | nothing |
| 6 | Release day (download → ablations → train → zip) | decides everything above on real data | 6–7 h | the data release |

Not worth it now: REVE/LaBraM on this CPU, more training participants, more
spatial-filter variants.

## 7. Where to read more

| Question | Document |
|---|---|
| What exactly is the model, and what is the evidence? | [SEALED_RECIPE.md](SEALED_RECIPE.md) § 1–2 (Phase 7 = this week's features) |
| What do I do when the data is released? | [RELEASE_DAY.md](RELEASE_DAY.md) |
| What happened in the last session, and what is running? | [HANDOFF.md](HANDOFF.md), top section |
| How do the models work, for EEG readers? | [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) |
| Every run, timing and decision | [LOG.md](LOG.md) |
| What was uploaded? | [SUBMISSIONS.md](SUBMISSIONS.md) |
| Dates and rules | [COMPETITION.md](COMPETITION.md) |

## 8. Keeping this file current

- **When:** update it at the end of every sprint or session that changes a
  decision, a number, the plan or the competition calendar.
- **Size:** keep it under ~150 lines. Detail goes to the documents in § 7, not
  here.
- **Numbers:** each one must match its source document.
- **Change log** (newest first, one line each):
  - 2026-10-01: created. bpt4 is in the model; icoh is a release-day test; 4
    decisions pending.
