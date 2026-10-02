# Directions: Track 2 at a glance

**Updated 2026-10-02** after the 2026-10-01 sprint. This is the short version
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
  (checked 2026-10-01 17:40). Every number below comes from public stand-in
  datasets with 4–9 people each. Differences under ~2 points are noise.
- **Uploaded so far** (warm-up, which doesn't rank;
  [SUBMISSIONS.md](SUBMISSIONS.md)):
  - EEGNet, 0.82;
  - the **sealed model's deployment test** on 2026-10-02: Finished, 0.62 in
    59 s. The Riemann-Sealed model runs on the scoring server.
- **Sealed-phase candidate:** **Riemann-Sealed + bpt4**. It retrains end to
  end on release day with the runbook: **~4–4.5 h** now, down from 6–7 h,
  rehearsed at full size on 10-01 ([RELEASE_DAY.md](RELEASE_DAY.md) § 8).
- **Headline on the stand-in closest to the sealed tasks** (Scherer: word
  association / subtraction / hand imagery, 3 classes, chance 0.333):
  - **0.529** balanced accuracy, up from 0.486 before this week;
  - **0.592** if test-time re-centring is allowed (see decision 1).
  - Source: [SEALED_RECIPE § 2](SEALED_RECIPE.md).

## 2. Decisions waiting on you

| # | Decision | My recommendation | Why it matters | Your effort |
|---|---|---|---|---|
| 1 | **Ask the organisers** whether `predict()` may use statistics of unlabelled test windows. Order and persistence are settled from the public code: one model object, batches of 64, unshuffled recording order | Post it on Discord, the only live channel: the listed email bounced (10-02), and GitHub issues get no replies. Without a yes it stays off, which is already the plan | Online re-centring is the largest lever: **+3.5 to +7.5 points** in recording order | one Discord message |
| 2 | ~~Deployment-test upload~~ **done 2026-10-02**: Finished, 0.62, 59 s | — | The sealed model's format runs on the scoring server | — |
| 3 | **Keep bpt4 by default on release day** unless the real data shows ≥ 1 point better without it | Accept | It is the one solid gain of 09-29 (+4.4). Rule 3b in [RELEASE_DAY § 6](RELEASE_DAY.md) | read one rule |
| 4 | **Review: strict CV references are now the release-day setting** (`WREF=strict`) | Accept | They remove a known bias in choosing the blend weight. Solver and harness agree exactly; the cost is +12 min at 500 Hz. Rule 4 in [RELEASE_DAY § 6](RELEASE_DAY.md) | read one rule |
| 5 | Optional: approve REVE/LaBraM weights; merge the thesis docs branch `docs/sprint0928-consistency` (`main`'s perf-loop SUMMARY and MODELS still show old numbers) | Low priority | REVE/LaBraM are hours per epoch on this CPU | — |

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
                   (since 10-01: per-person LDAs solved in a 50× faster, equivalent form)
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
| **icoh**: lagged connectivity between channels | +2.2 alone; +6.5 with bpt4 (post-hoc) | release-day test; **fits at 500 Hz since 10-01** (19.3 min, 1.23×) |
| xDAWN (class templates) | +4–5 on motor imagery, ~0 on the sealed-like tasks | kept; first to drop if the real data disagrees |
| Strict CV references for the blend weight | correctness, not accuracy: on Zhou w moved 0.5 → 0.75; the others unchanged | **release-day setting since 10-01** |
| No gain | EEGNet; pre-training on other datasets; more participants; re-referencing / Laplacian; CSP; regional covariances; band-power maps; time-segment covariances (under the deployed model); 8–30 Hz band-pass; **personal LDA with a shared (pooled) covariance** (10-01: −2.7 / −0.9 on Scherer 3-class) | dropped |

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
3. **Deployment: tested 2026-10-02**: the sealed model runs on the scoring
   server. Untested there: per-context references and the 47-ch input (Dreyer
   has neither). Upload the release-day candidate to the warm-up once it
   switches to Graz + BrainHero, if that is before Oct 25.
4. **Compute (CPU only): largely solved on 10-01.** The per-person LDAs now
   cost ~0.15 s instead of ~9 s each. At 500 Hz the recipe fits in 15.7 min,
   bpt4 in 17.7 and bpt4 + icoh in 19.3 ([RELEASE_DAY § 8](RELEASE_DAY.md)).
   What remains is the feature computation itself.
5. **Few windows per person.** About 360–720 calibration windows (mock
   structure; real counts unknown) against ~6,000 features. Heavy shrinkage
   keeps it stable, but new blocks must be cheap and strong.
6. **Release-day time.** The runbook now takes ~4–4.5 h at 120 Hz (rehearsed
   at full size: ablations 69 min, replica check 28 min). The release date is
   still unknown, with the sealed phase on Oct 28.

## 6. Next steps worth exploring (ranked)

| # | Step | Expected gain | Effort | Needs |
|---|---|---|---|---|
| 1 | Decision 1 above (send the organiser question) | unlocks +3 to +7.5 | minutes | you |
| 2 | Release day (download → ablations → train → zip) | decides bpt4, icoh, contexts, channels on real data | ~4–4.5 h | the data release |
| 3 | Riemann + EEGNet probability ensemble | +1 to +2 (EEGNet is now 6–12 points behind, so likely less) | half a day | nothing |
| 4 | Thesis workstream: check that the full-window AUC advantage (0.714 headline) is not a cue/response artefact ([MODELS.md](../MODELS.md) "Confirm window effect is not leakage") | integrity of a public number | ~3 h | nothing |

Done on 10-01 (no longer open): faster per-person LDA; icoh's 500 Hz cost;
strict per-fold blend-weight search. Not worth it now: REVE/LaBraM on this CPU,
more training participants, more spatial-filter variants, a restricted icoh,
a shared-covariance personal LDA.

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
  - 2026-10-02: deployment test passed (0.62, 59 s); decision 2 done. Organiser
    email bounced: decision 1 is now one Discord question (order/persistence from code).
  - 2026-10-02: sprint 10-01. Dual LDA (icoh fits at 500 Hz: old decision 4
    resolved); strict CV references adopted (new decision 4: review);
    deployment zip ready; release day ~4–4.5 h; shared-covariance LDA no gain.
  - 2026-10-01: created. bpt4 is in the model; icoh is a release-day test; 4
    decisions pending.
