# Value check: is this the right thing to spend the time frame on?

The user asked for two behaviours:
1. If they give a goal, check its value first. If it is not worthwhile, or there are
   clearly better candidates, tell them **before** choosing what to run.
2. If they give no goal, find the best area to work on for the time frame yourself.

The check must be honest in both directions. Flagging every goal is noise and wastes
their attention; quietly running a goal the evidence already answered wastes their
time frame. The standard is "would a well-informed collaborator raise this?"

## The rubric

Answer these for each candidate, the user's focus included. Keep evidence (file and
section, LOG date) next to each answer: the answers go into the message.

1. **Does it move what counts?** Name the objective the work serves: the Track 2
   sealed-phase score, an honest thesis AUC or result, the paper, the accuracy of the
   public repo. Work on something that does not count (e.g. the warm-up leaderboard,
   which does not rank) is worth little unless it validates a pipeline you will need.
2. **Is it already measured or decided?** Search before treating an idea as new: `grep -n -i "<keyword>"` over the workstream's
   LOG, LEDGER, SUMMARY, SEALED_RECIPE and SUBMISSIONS. "Already measured: no gain
   (LOG 2026-09-25)" is the most common reason a goal is not worthwhile. It stays
   worth doing only if something changed: new data, a bug found in the earlier
   test, or a clearly different variant.
3. **Expected gain, with a basis.** Give a number or range and where it comes from:
   an earlier measurement, a proxy result, literature. Compare it with the noise
   floor (~2 points balanced accuracy on the 4–9-subject Track 2 proxies; see the
   LEDGER for the thesis). A gain expected below the noise floor cannot be
   demonstrated in one sprint.
4. **Decision value.** Would the result change what we do, e.g. choose between recipe
   branches, retire an open question, unblock a deadline? Cheap experiments that
   settle a live decision rank high even with a modest expected gain.
5. **Cost against the frame.** Estimate coding plus compute plus write-up from known
   per-unit costs (LOG timings), then compare with `deadline − now − wrap-up
   reserve`. Work that cannot reach a usable result or a clean checkpoint in time is
   worth less. Often it can be scoped down.
6. **Blockers.** Does it need unreleased data, an approval (downloads, weights,
   settings), a GPU, the organisers, or another person? A blocked goal is not a
   sprint goal. Its unblocking step may be, or it goes under "needs you".
7. **Risk.** Rules and compliance (competition terms), irreversible or outward
   actions, the chance of breaking frozen artifacts or delivered numbers.
8. **Deadline pressure.** External dates (for Track 2 see the project map) make work
   that feeds them more valuable now than open-ended work that can wait.

Combine these into a rough **value per hour** and a short reason. Precision is not
the point: ranking the options and being able to say why is.

## Verdicts for a user-given focus

| Verdict | When | Action |
|---|---|---|
| **Proceed** | it counts, is not already answered, fits (or can be scoped to fit), is not blocked, and nothing clearly beats it | Start. In the kickoff, say in 2–4 lines what you checked and why it passes |
| **Proceed, scoped** | worthwhile but too big for the frame | Start the scoped version and say exactly what is in and out. Ask only if the scoping changes what the user actually wanted |
| **Flag: not worthwhile** | already measured with no new reason; does not count; expected gain below noise; blocked; breaks a standing constraint | Stop, tell, ask (below) |
| **Flag: clearly dominated** | another executable candidate has roughly **≥ 2× the value per hour**, or unblocks a hard near deadline the focus ignores | Stop, tell, ask (below) |

"Clearly" matters. Two candidates within a factor of about 1.5 of each other are a
judgment call, and the user's stated preference wins. Proceed and mention the
alternative in one line.

## How to flag (before anything runs)

Do the check fast, in 15 minutes or less, so the question reaches the user while they are still
there. Then send a short message, about 20 lines at most. The user is deciding
between options, not reading a report, so details belong in the plan file.

1. **Verdict in one sentence**, e.g. "I'd hold off on X: it was measured on 09-25 and
   gave no gain."
2. **Evidence**: 2–4 bullets with file references.
3. **Alternatives**: 2–3 candidates, each with expected gain, effort (fits the frame?)
   and risk, in one line apiece.
4. **Recommendation**, then AskUserQuestion with:
   - the recommended alternative, labelled "(Recommended)";
   - the user's goal as stated;
   - a scoped or combined version, if one makes sense.

   Put the reason in each option's description.

While waiting, do only read-only preparation: no launches, no commits, no downloads.
If the user picks their original goal, run it wholeheartedly. It is their project.

## When no focus is given

Rank the executable candidates by value per hour inside the frame, putting hard
deadlines first, and pick the top one. Then announce and start, without asking:

> **Sprint until <deadline> (<assumption if any>).** I'll work on **<pick>**: <why, 1–2
> lines>. Runners-up: <A> (<why not first>), <B> (…). Needs you: <items only the user
> can do>. The brief is in `<path>`; I'll report at the end.

For long frames it is fine to chain goals: the brief can list a second goal to start
once the first one's decision rule is satisfied, if time remains.

## Worked examples (from this project; re-check the current state before reusing)

- **"Push our warm-up leaderboard score; try REVE fine-tuning on Dreyer."** Flag: not
  worthwhile now.
  - The warm-up does not rank; only the sealed phase does (TRACK2_BCI, COMPETITION).
  - A REVE full fine-tune is ~45 min/epoch on this CPU, and a frozen probe is likely
    below WU1 (LOG 2026-09-25 exploration).
  - The weights need the user's approval.
  - Dreyer is in REVE's pretraining corpus, so any warm-up gain is inflated.

  Offer the SEALED_RECIPE § 5 items instead (e.g. the Riemann + pooled-EEGNet
  ensemble on the sealed-like 3 classes).
- **"Add more training participants."** Flag: already measured, no gain (LOG
  2026-09-25).
- **"Riemann + pooled-EEGNet ensemble on the Scherer 3-class proxy, 3 hours."**
  Proceed. It is ranked next step 3 in SEALED_RECIPE § 5, cheap because both models
  exist in the harness, and fits 3 h with a pre-registered rule (adopt only if ≥ 2
  points and the subject-bootstrap CI excludes 0, weight chosen on calibration data
  only).
- **"Rerun the perf loop on the fast feature set."** Flag unless something new: the
  loop plateaued after Round 4 (`outputs/perf_loop/LEDGER.md`, SUMMARY).
- **No focus, weekend, Track 2 data still unreleased.** Candidates come from
  SEALED_RECIPE § 5 and HANDOFF. Item 1 (the organisers' question) is the user's
  action and goes under "needs you". Pick the best executable item, and put a
  once-a-day data-release check with a contingency in the brief.
