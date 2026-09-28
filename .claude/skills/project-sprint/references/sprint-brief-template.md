# Sprint brief template (and the other documents a sprint writes)

The brief is the plan of record. Write it for a reader who has only the repository:
a later session, or you after a context reset, must be able to resume from the
brief, HANDOFF and the `.done` markers alone. The model is
`codabench/prompts/2026-09-26_sealed_phase_prep.md`. Read it once if you have not.

Contents:
- Brief skeleton
- Writing good decision rules
- LOG entry
- HANDOFF
- Final summary to the user

---

## Brief skeleton

```markdown
# Sprint <YYYY-MM-DD>: <goal in a few words>

Time frame: <start> → <deadline> (<how the frame was interpreted>). The user is
<reachable | away until …>. Chosen by <the user's focus | the value check, see § 3>.
Work autonomously; decide each next step from the previous result with the rules
below; never let a job go stale (§ 6). Read the whole brief before starting.

## 1. Context (verified <date>, unless marked)
- Workstream, where the code lives, branch; what to read first (≤ 10 min).
- Environment recipe (WSL heredoc or Windows venv), machine limits.
- What counts and the deadlines that apply.
- **Already measured, do not repeat:** <result> (<file, date>); …
- Open questions the sprint must not silently decide (e.g. rule-dependent methods).

## 2. Data and assets (if relevant)
Table: name, local path, what it is, size, and whether it is pre-approved.
Anything that needs a download: approved or not, and where it goes.

## 3. Goal and priorities
The deliverable (file names), then the priority order of the work, highest first,
with one line of "why" each. If the value check flagged or scoped anything, record
the outcome here.

## 4. Rules
- Branch, commit prefix, push after every commit, no tags or merges; which log files
  may be committed.
- Frozen artifacts and standing constraints that apply to this sprint.
- Evidence rules: no tuning on test, the noise floor for this data, seeds/folds, the
  data-shape check in each run's first log line, "say no gain plainly".
- Resource limits: ≤ 2 heavy jobs, thread caps, RAM per job.
- When to update HANDOFF.

## 5. Phases and decision rules
Each phase: estimate the wall time first, run it with the launch pattern of § 6,
time-log start and end from the log stamps, write RESULTS.md, append a LOG entry,
commit + push, then apply the phase's decision rule.

### Phase 0: <setup / harness / smoke test> (estimate …)
Steps. **Done when:** <observable condition>.

### Phase 1: <…> (estimate …)
Steps (configs × seeds × datasets). **Decision:** <numeric rule, with what happens on
each branch>.

… (as many phases as the frame allows, the last one being the deliverable)

## 6. Keeping jobs alive and never stale
- Launch pattern (WSL attached in a background Bash call; resumable step script).
- ETA row in STATUS.md before launch.
- Watchdog + Monitor cadence; crash signatures; heartbeat lines.
- What to do on SLOW / STALLED / DEAD, and on late-but-alive.
- Machine hazards to check (sleep, OneDrive, E-cores).

## 7. Time budget and order
Phase order with start targets. Compute budget. **Launch cutoff:** no job starts
unless 1.5 × its ETA fits before <deadline − reserve>. **Wrap-up reserve:** <N> min.

## 8. Contingencies
- If <external event> happens (e.g. data released): <what to stop, what to do first>.
- If <approval needed> comes up: skip it, note it under "needs you", take the next item.
```

## Writing good decision rules

A decision rule is written *before* the numbers exist, so the numbers cannot talk you
into a story afterwards.

- Name the comparison, the metric, the threshold and the evidence required, e.g.
  "adopt the ensemble if it beats the best single model by ≥ 2 points cell-averaged,
  and the paired subject-bootstrap 95 % CI excludes 0, with the weight chosen on
  calibration sessions only".
- Say what happens on **each** branch, including the negative one ("otherwise record
  'no gain' and move to Phase 4").
- Tie the threshold to the noise floor of this data. Below it, require more seeds,
  folds or subjects before concluding.
- Prefer rules that pick a *simpler* variant on ties. Complexity has to earn its
  place.
- Allow one transparent escape hatch: when the evidence shows a rule's premise is
  wrong (e.g. the prescribed fallback lowers the score), deviate and write down why.

## LOG entry (append to the workstream's LOG, one per phase)

```markdown
## <YYYY-MM-DD> (<time of day>) — <Phase N: title>

<one-line purpose; scripts used; where the results table is>

| Time | Step | Estimate | Actual | Result |
|---|---|---|---|---|
| HH:MM:SS–HH:MM:SS | <step> | <est> | <actual>, <on / under / over> | <key number or ✅/❌> |

<results table: the numbers that matter, with seeds as mean ± SD>

**Decision.** <rule applied → outcome, in plain words; "no gain" when that is the
result; any deviation from the rule and why>

**Incident** (if any): time, symptom, cause, fix.
```

## HANDOFF (overwrite; a resumer reads only this)

```markdown
# HANDOFF — <sprint title>
## State (updated <date time>)
| Phase | Status | Result / where |
## What is running (log dir, ETA, how to check it)   ← "nothing" is a valid answer
## How to run / resume (commands, lane scripts, watchdog, summaries)
## Facts a resumer needs (mappings, traps met this sprint, constraints)
## What the user needs to do
```

## Final summary to the user (the last message of the sprint)

Lead with what they most need, then the rest:

1. **Outcome in two sentences**: what was built or learned, and the headline number.
2. **Decisions needed from the user**, with a recommendation each (these block
   progress).
3. **What was done**: phases with key numbers and decisions ("no gain" plainly).
4. **Deviations from the brief and incidents**, one line each with the reason.
5. **What is still running** (normally nothing) and **where everything is** (files,
   commits, pushed or not).
6. **Next steps**, ranked with expected gain, effort and risk.

Keep it scannable: short paragraphs and a table or two. The detail lives in the LOG.
