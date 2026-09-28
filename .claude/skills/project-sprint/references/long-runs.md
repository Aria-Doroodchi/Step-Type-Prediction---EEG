# Long runs: launch, watch, recover

Everything here comes from runs on this machine that died, stalled or thrashed. The
aim: a job either finishes, or its failure is noticed within minutes and fixed at the
root, and nothing is lost when it is relaunched.

## 1. Before launching

- **Estimate.** Ground the ETA in measured per-unit costs (earlier LOG timings, a
  smoke run) and write it down: `| HH:MM:SS | ETA <phase> ≈ N min, done by HH:MM |` in
  the log dir's STATUS.md. After the run, record on / under / over.
- **Smoke test** on the smallest dataset with tiny epochs first. It catches most bugs
  in seconds.
- **Check the machine.** Read the sleep settings (PowerShell
  `powercfg /query SCHEME_CURRENT SUB_SLEEP`). Do not change them yourself. If sleep
  would kill a long run, ask the user, or keep the system awake with a background
  process holding `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`.
  Check disk space for large outputs.
- **Resources.** At most **two CPU-heavy jobs at once** (20 threads). Cap every
  thread pool per job (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`
  and torch's own). Uncapped, two jobs oversubscribe the cores and slow each other
  down ~15×. Keep each job under ~7 GB RSS.

## 2. Make it resumable

Use `scripts/step_lib.sh` (bundled with this skill; the codabench variant is
`codabench/scripts/sealed_lib.sh`). A phase script looks like:

```bash
#!/usr/bin/env bash
# <phase>: what it does. Resumable: finished steps are skipped by <step>.done.
LOGDIR="$HOME/codabench/logs/sprint_<date>_<phase>"   # or a Windows-side path in Git Bash
ENV_SETUP='source ~/codabench/env.sh >/dev/null'      # optional; eval'ed by step_lib
THREADS_PER_JOB=10
source "/mnt/c/Users/Ali D/Documents/ML/.claude/skills/project-sprint/scripts/step_lib.sh"

laneA() {
  step a_fast 5  30m python analysis/x.py --study zhou2016 ...
  step a_slow 40 120m python analysis/x.py --study tangermann2012 ...
}
laneB() {
  step b_main 45 150m python analysis/x.py --study scherer2015 ...
}
LANES=${LANES:-AB}
[[ $LANES == *A* ]] && { laneA & }
[[ $LANES == *B* ]] && { laneB & }
wait
finish_if_done a_fast a_slow b_main   # writes ALL DONE only when every step has .done
```

- `step <name> <eta_min> <timeout> <cmd…>` writes `<name>.start` (for the watchdog),
  `<name>.pid`, `<name>.log`, START/END rows in STATUS.md, and `<name>.done` on
  success. Re-running the script skips done steps.
- Inside a step, make the program itself resumable too: append one result row per
  config to a per-lane or per-study file and skip configs already present. Never let
  two processes append to the same file (on `/mnt/c` concurrent appends lose rows).
- `LANES=A` or `LANES=B` lets you start one lane as soon as a CPU slot frees up,
  instead of idling until the other lane's phase ends.
- Make long loops print a **heartbeat** line (every ~10 epochs or items). Without it,
  "the log has not grown" cannot tell a stall from a slow but healthy step.
- Print the **data shape** (subjects × sessions × windows, channels, classes) in the
  first log line of every run, and check it before believing any score.

## 3. Launch so it survives

- **WSL:** run the phase script in the *foreground* of a Bash tool call with
  `run_in_background: true`, which keeps `wsl.exe` attached and WSL's VM alive:
  `wsl.exe bash -lc 'bash ~/codabench/scripts/<phase>.sh'`. Never `nohup … &` inside
  a short-lived `wsl.exe bash -s`; the VM shuts down under it.
- **Windows-native** (thesis venvs): the same idea in Git Bash, running the phase
  script with `run_in_background: true` and calling `.venv/Scripts/python.exe` with
  `PYTHONUTF8=1`.
- **Waiting for one event** (e.g. "start lane B of phase 3 when phase 2 lane B ends"):
  a background Bash call with
  `until grep -q "END b_main" <logdir>/STATUS.md; do sleep 20; done` notifies you
  exactly once.

## 4. Watch it

- `scripts/watch_run.sh <logdir>` prints one line per step: DONE, FAIL, or RUN with
  elapsed vs ETA, seconds since the log last grew, process alive plus CPU/RSS, free
  RAM, the last log line, and a verdict:
  - `OK`: running within expectations;
  - `SLOW`: elapsed is more than 1.5× the ETA;
  - `STALLED`: the log has not grown for 10 min;
  - `DEAD`: no live process, no `.done`, no END row;
  - `FAILED`: an END row with a non-zero rc.
- Arm the Monitor tool on `scripts/monitor_loop.sh <logdir> 300` (30 min is its
  maximum; re-arm on expiry until ALL DONE). It emits new END/ALL DONE rows, new crash
  signatures (`Traceback|Error|Killed|OOM`) and every non-OK verdict. On re-arm it
  replays earlier END rows, which is harmless. Silence means healthy only because the
  filter covers every terminal state.
- Between checks, do not wait passively: write the next phase's code, smoke-test it
  lightly, analyse finished steps, draft the LOG entry and the deliverable.

## 5. When a check is not OK

Investigate before anything else:
1. Read the last 20 lines of the step log.
2. Check the process: `ps -o pid,etime,%cpu,rss,cmd -p <pid>`, and its thread count
   (`ls /proc/<pid>/task | wc -l`) if CPU looks odd.
3. Check memory (`free -g`) and that WSL is up (`wsl.exe -l --running`).
4. Check whether the machine slept.
5. Check the data shape: is the job silently on the wrong config or cohort?

Fix the **root cause** rather than shrinking the workload to dodge it. Kill by PID
(never `pkill -f`, which can match its own command line), then re-run the phase
script: done steps are skipped. Log the incident in the LOG: time, symptom, cause,
fix.

**Late but alive** (past the ETA, log still growing, CPU sane): extend the ETA once
and write the revision into STATUS.md and the LOG. If the job passes 2× the original
estimate, decide whether the finished steps already answer the phase's question. If
they do, move on with them and let the rest finish in the background.

## 6. Scheduling across phases

- Keep both CPU slots busy with the brief's priority order: when a lane frees up,
  start the next ready lane (from this phase or the next) at once.
- Put the long, independent jobs (overnight training, pre-training) where they
  overlap with the analysis and writing of other phases.
- Before each launch, apply the launch cutoff: 1.5 × the ETA must fit before
  `deadline − wrap-up reserve`, unless the job is checkpointed and its partial output
  is useful (then say it is still running).
- Expect nights to run ~1.7× slower (Windows moves idle work to E-cores). Revise
  ETAs; do not panic.
