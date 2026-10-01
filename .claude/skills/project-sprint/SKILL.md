---
name: project-sprint
description: Time-boxed autonomous work session ("sprint") on this EEG repo, which holds the thesis step-type pipeline and the Codabench Track 2 competition work in codabench/. The user gives a time frame ("3h", "until 6pm", "tonight", "this weekend") and optionally a focus area. Checks a given focus against better candidates and asks before spending the time on a weak one; with no focus, surveys the project state and picks the highest-value work that fits the time frame; then writes a brief with pre-registered decision rules, runs resumable, watchdog-monitored jobs, logs estimates against actuals, commits as it goes and hands off cleanly. Use whenever the user hands over a block of time to work on this project unattended or semi-attended, even without the word "sprint", e.g. "I'm away this weekend, keep improving the model", "work on the competition until 6pm", "overnight, see what's worth doing", "spend the next 3 hours on the ensemble idea".
argument-hint: "<time frame> [focus]   e.g.  3h  |  until 18:00  |  weekend ensemble on Scherer"
---

# Project sprint

Request: `$ARGUMENTS`
(If that is empty or unexpanded, take the time frame and focus from the user's message.)

A sprint turns a block of the user's time into the most valuable **finished and
documented** progress on this project. It follows the shape of the 2026-09-25
weekend session, which the user liked: the brief
`codabench/prompts/2026-09-26_sealed_phase_prep.md` produced
`codabench/SEALED_RECIPE.md` plus working, verified code by the next morning. The
shape is: orient, pick the right goal, write a brief with pre-registered decision
rules, run everything resumable and watched, write down every estimate, actual and
decision, commit as you go, hand off cleanly.

## Time budget

| Step | Share of the frame | Produces |
|---|---|---|
| 1 Intake | a few minutes | absolute deadline, focus, constraints |
| 2 Orient | ≤ 10 % (cap 30 min; ≤ 15 min when a focus may need the user's answer) | state snapshot, candidate list |
| 3 Value check and choice | inside step 2's budget | verdict; the user's answer if one is needed |
| 4 Brief | ≤ 5 % (cap 20 min) | the plan of record, committed |
| 5 Execute | the rest | phases, results, LOG entries, commits |
| 6 Wrap-up | reserve 10–15 % (min 15 min) | deliverable, HANDOFF, final summary |

The wrap-up reserve is not optional padding: a sprint that ends mid-run with nothing
written down leaves the user with less than a smaller sprint that finished. Plan the
work to end at `deadline − reserve`.

## 1. Intake

- If no time frame was given, ask for one in a single short question. It sizes
  everything else, and the user is present at invocation.
- Get the local time from the shell (`date`), not from memory, and turn the time
  frame into an absolute deadline. "3h" is now + 3 h. "Tonight" or "overnight" means
  until 08:00 the next morning, and "this weekend" until Monday 09:00, unless the
  user says otherwise. State the assumption in the kickoff message so they can
  correct it.
- Note the focus, if any (it may be vague, like "the competition", or precise).
- Note whether the user will be reachable ("I'm away until Monday" changes when a
  question can still be answered).
- Note any standing constraint the user explicitly lifts ("you may download X",
  "push to org too", "OK to change the power settings"). Only what is stated counts.

## 2. Orient

Read the state of record instead of trusting memory. Memories, CLAUDE.md and old
briefs age: branch names, "next steps" and numbers in them may be stale.

1. Read `references/project-map.md`: workstreams, where their state lives,
   environments, external sources, frozen artifacts, known traps.
2. For each workstream in scope, read its HANDOFF / LEDGER / next-steps sections and
   the tail of its LOG. Run `git status`, `git log --oneline -10` and
   `git branch -a`, and check for anything still running (log dirs with a STATUS.md
   but no ALL DONE row; `ps` inside WSL).
3. Check external sources the workstream depends on (data-release page, rules,
   leaderboard) when they could change the choice.
4. With a long frame (more than ~4 h) and no focus, fan out: one Explore agent per
   active workstream, each returning its state, open candidates with evidence and
   file references, and blockers in ≤ 250 words. With a short frame, read directly.
5. **Verify the claims the plan will rest on.** Trust the LOG and LEDGER for what
   was *measured*. Spot-check any claim about what the *code* does in the code
   itself, e.g. "the harness already scores context cells" or "the blend weight is
   chosen leave-one-session-out": open the function and follow the call. Docs
   describe intent and drift. In a 2026-09-28 test run, a two-minute check found
   that the first of those claims was false; planning on it would have produced
   wrong release-day numbers. A claim that turns out false is itself a candidate:
   fix it or correct the doc.
6. Build the candidate list: documented next steps, open questions, failures worth
   fixing, stale deliverables, anything an external change makes urgent. The
   user's focus, if given, is one candidate among them.

## 3. Value check and choice

Read `references/value-check.md` for the rubric, worked examples and message
templates. In short, judge each candidate on: whether it moves what actually counts;
whether it was already measured or decided (search the LOG and LEDGER before
treating an idea as new); expected gain against the noise floor, with a basis;
decision value; cost against the time left, including the wrap-up; blockers (data,
approvals, things only the user can do); risk; deadline pressure.

**The user gave a focus.**
- It is worthwhile and not clearly beaten: proceed. Say in 2–4 lines what you
  checked and why it passes, so the check is visible. If it does not fit the frame,
  scope it down and say what is in and out.
- It is not worthwhile (already measured, does not count, expected gain below
  noise, blocked, breaks a standing constraint) **or** clearly dominated (an
  alternative with roughly twice the value per hour, or one that unblocks a near
  deadline): stop and tell the user **before running anything**. Give the verdict
  with evidence and file references, 2–3 alternatives with gain, effort and risk,
  and your recommendation, then ask with AskUserQuestion: recommended option first,
  their goal as stated, a scoped or combined option. While you wait, only cheap
  read-only preparation: nothing launched, nothing committed.
  The user asked to hear about a weak goal before it consumes their time frame. A
  fast check (≤ 15 min) makes the question land while they are still at the
  keyboard; a slow one lands after they have left.

**No focus given.** Picking the work is your job. Choose the executable candidate
with the best value per hour inside the frame, putting hard external deadlines first.
Announce the pick, one-line reasons, the runners-up and anything that needs the user,
then start. No confirmation is needed.

Things only the user can do (post to a forum, approve a download, upload a
submission, merge to main) never become the sprint goal. List them under "needs you".

## 4. Write the brief

Write the plan of record from `references/sprint-brief-template.md`, which mirrors
the weekend brief. It must contain:
- verified, dated context, including what is already measured and must not be
  repeated;
- the goal and a priority order;
- rules: the standing constraints and the evidence rules;
- phases, each with a time estimate and a **pre-registered decision rule with
  numeric thresholds**;
- a keep-jobs-alive section for anything over ~10 min;
- a time budget with the launch cutoff and the wrap-up reserve;
- the deliverable;
- contingencies.

Save it as `<workstream>/prompts/<YYYY-MM-DD>_<slug>.md` (for the thesis workstreams,
use `docs/sprints/`), commit and push it before execution, and point HANDOFF at it.
A committed brief makes the plan reviewable, and it is what lets a later session
resume from the files alone.

## 5. Execute

Read `references/long-runs.md` before launching anything that takes more than a few
minutes, and `references/evidence.md` before drawing any conclusion. What matters
most, and why:

- **Estimate, time-log, investigate overruns.** State the ETA before launch and log
  start and end from the run's own time stamps. Past 1.5–2× the estimate, stop
  waiting and diagnose (CLAUDE.md rule). Silent stalls and fallbacks have cost this
  project whole runs.
- **Launch so the job survives.** Resumable steps with `.done` markers, WSL kept
  attached, BLAS thread caps, at most two heavy jobs at a time, per-lane result
  files. The bundled `scripts/step_lib.sh`, `scripts/watch_run.sh` and
  `scripts/monitor_loop.sh` implement this.
- **Watch actively, never wait passively.** Arm the watchdog in a Monitor, make long
  loops print heartbeat lines, and match crash signatures. Between checks, build the
  next phase, analyse finished steps and write docs.
- **Trust nothing unverified.** Smoke-test on the smallest dataset first, and check
  the data shape in every run's first log line. A silent config fallback to the
  wrong cohort has happened here.
- **Decide by the brief.** Apply each decision rule as written. When the evidence
  contradicts a rule's assumption (e.g. the prescribed fallback turns out to hurt),
  deviate in the open: write down the rule, the evidence and the choice.
- **After each phase:** write RESULTS.md, append a LOG entry (estimate vs actual,
  findings, decision in plain words), commit and push, refresh HANDOFF.
- **Launch cutoff.** Do not start a job unless 1.5× its ETA fits before
  `deadline − reserve`. The exception is a checkpointed job whose partial results are
  useful, and it must be reported as still running.
- **Stay unattended.** Do not interrupt the user mid-sprint except for an approval
  the brief cannot cover. When blocked, take the next-best item and report the
  blocked one at the end. Check external sources on the brief's cadence and follow
  its contingency if something big changes.

## 6. Wrap-up

Inside the reserve:
- finish or cleanly stop jobs;
- write or update the deliverable and the state docs (HANDOFF, LOG summary, next
  steps);
- update the user's decision brief `codabench/DIRECTIONS.md`: status, decisions
  pending with a recommendation each, what we know, bottlenecks, ranked next steps,
  one change-log line. It is the file the user actually reads, so keep it short;
- commit and push;
- record durable outcomes in memory.

Then send the final summary (template in `references/sprint-brief-template.md`):
- what was done, with numbers;
- decisions, saying "no gain" plainly where that is the result;
- deviations and incidents;
- what needs the user;
- what is still running (normally nothing);
- where everything is;
- next steps, ranked.

## Standing constraints

These hold unless the user explicitly lifts one in the request, because each one
protects something that is hard to undo or is the user's call:
- No uploads, submissions, public posts or messages to anyone. No merges, tags,
  force-pushes or PRs.
- No large downloads and no model weights without approval. The project map lists
  the pre-approved data.
- Leave frozen artifacts alone (listed in the project map).
- Never tune on test data. A test-tuned number appears only as a labelled oracle.
- Techniques whose legality under a competition's rules is unclear: flag them,
  report them separately, do not adopt them.
- Do not change system settings (power plans etc.) yourself. Read them, and if they
  would kill a run, ask the user or hold a keep-awake request instead.
- Commit only on a feature branch, and push to `personal` after every commit
  (OneDrive has reverted `.git` after power events).
