# Evidence rules: when a number means something

These rules exist because the datasets here are small (4–9 subjects per Track 2 proxy,
~30 participants in the thesis) and noise easily passes for a finding. They are what
made the 2026-09-25 sealed recipe trustworthy: every claim in it is either a
pre-registered comparison, a bootstrap CI over subjects, or stated plainly as noise.

## Before the numbers exist

- **Check that the code does what the docs say** before building on it. Measured
  results in the LOG and LEDGER are evidence. Descriptions of what a script or
  harness does are claims until you have read the code path. Recipes and handoffs
  are written at the end of long sessions, and they drift.
- **Pre-register the decision rule** in the brief: comparison, metric, threshold, and
  the action on each branch. See sprint-brief-template.md.
- **Know the noise floor** of the data: roughly 2 points balanced accuracy on the
  Track 2 proxies. For the thesis cohort, the participant-level interval is much
  wider than the fold-level one (see README § Results). Plan seeds and folds so the
  expected effect can clear it.
- **Fix the split before looking**, and never tune on the test portion: last session
  per subject for cross-session work, held-out participants for cross-subject work.
  Hyper-parameters come from training data only (earlier sessions,
  leave-one-subject-out, chronological halves). A test-tuned number may appear only as
  a clearly labelled **oracle** upper bound.
- **Tune under the deployment condition.** A weight, threshold or epoch count chosen
  under one setting and used under another is a silent bug. On 09-25 a blend weight
  chosen under the router condition cost 3 points when deployed with online
  re-centring.

## While running

- **Smoke test** on the smallest data, then check the **data shape** line of every
  full run: subjects, sessions, windows, channels, classes, feature count. A silent
  fallback to the wrong config or cohort is this repo's most expensive recurring
  bug.
- **Seeds:** at least 3 for anything stochastic (neural nets), reported as
  mean ± SD. Deterministic fits (Riemann/LDA) need one run, but their uncertainty is
  across subjects, so bootstrap over subjects.
- **Deduplicate** results by config key when combining phases, so a config that ran
  twice does not count as two seeds.

## Concluding

- **Compare paired**, same subjects and same splits, and report the per-subject
  differences, not just two means. For the key claims, give a **paired subject-level
  bootstrap 95 % CI** and the count of subjects that improved
  (`codabench/analysis/sealed_bootstrap.py` is the template).
- Below the noise floor, or with a CI that crosses 0, write **"no gain"** or "not
  measurable". That is a result, not a failure. Do not stack small insignificant
  "improvements" into a recipe.
- **Ties go to the simpler variant**, unless the complex one generalises the simple
  one and its extra knob is chosen on training data (e.g. a blend weight that can
  collapse to either end). If so, say why.
- **Deviations from a pre-registered rule** are allowed when the evidence contradicts
  the rule's premise, never quietly. Write the rule, the evidence and the choice into
  the LOG.
- **Label what every number is:** metric (e.g. cell-averaged vs pooled balanced
  accuracy, AUC), split, dataset, class subset, seeds, and whether the model used
  oracle information (true ids, test statistics).
- **Separate rule-dependent results** (e.g. methods using unlabelled test windows)
  from clean ones in every table, and never make them the default without the user.
- **Cross-check the deployable artifact against the harness.** If a solver trained
  through the official pipeline scores differently from the harness, find out why
  before trusting either. On 09-25 this caught the online-mode session-alignment
  bug: 0.743 → 0.792 after the fix.
