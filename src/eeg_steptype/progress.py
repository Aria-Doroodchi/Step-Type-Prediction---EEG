"""Estimated time-to-completion helpers for nested-CV training loops.

The training driver in ``models/train.py`` already knows how many outer
folds it will run and times each one. This module turns those numbers into
a live ``"~Xm:Ys remaining"`` message that gets appended to every per-fold
log line.

Two estimators live here:

* ``FoldETA`` — per-outer-fold rolling window. Used inside one participant's
  CV loop, where each fold's wall-clock cost is roughly comparable.
* ``CohortETA`` — wall-clock rate across participants. Used in
  ``train.run()`` for cohort-level progress, with native support for
  joblib-parallel pools (rate is measured in wall-clock time, so parallelism
  is captured automatically).

Both estimators are intentionally simple: each ``record()`` call is O(1),
state is at most a list of a few hundred floats, and no external library is
needed. The cost over a full run is dwarfed by a single sklearn refit.
"""

from __future__ import annotations

import math
import statistics
import time


# ---------------------------------------------------------------------------
def _human(seconds: float) -> str:
    """Format a seconds value as e.g. ``38s``, ``4m12s``, ``1h23m``."""
    if not math.isfinite(seconds):
        return "?"
    s = max(0, int(round(seconds)))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        m, ss = divmod(s, 60)
        return f"{m}m{ss:02d}s"
    h, rem = divmod(s, 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h{m:02d}m"


# ---------------------------------------------------------------------------
class FoldETA:
    """Rolling-window ETA over a fixed number of outer CV folds.

    Parameters
    ----------
    total_folds : int
        Total number of outer folds we expect to run.
    window : int
        Number of most-recent fold durations to average when projecting the
        next-fold cost. Smaller values adapt faster to fold-cost drift
        (useful when Halving search allocates different ``n_estimators``
        budgets to different candidates across folds). Larger values are
        smoother but slower to react. Default 5 is a good compromise for
        the 25-fold (5×5) typical run.

    Notes
    -----
    The first fold's ETA is reported as ``(estimating…)`` because there is
    no completed-fold sample to project from. Once at least one fold has
    finished, subsequent ``format()`` calls return the projected remaining
    wall time plus the rolling-window mean used to derive it.
    """

    def __init__(self, total_folds: int, *, window: int = 5):
        self.total = max(0, int(total_folds))
        self.window = max(1, int(window))
        self.times: list[float] = []
        self.start = time.perf_counter()

    # ----- recording -----
    def record(self, duration: float) -> None:
        """Append the wall-clock duration (seconds) of one completed fold."""
        self.times.append(float(duration))

    # ----- queries -----
    @property
    def completed(self) -> int:
        return len(self.times)

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.completed)

    def mean_recent(self) -> float | None:
        if not self.times:
            return None
        return statistics.mean(self.times[-self.window:])

    def eta_seconds(self) -> float:
        m = self.mean_recent()
        if m is None or not math.isfinite(m):
            return float("nan")
        return self.remaining * m

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.start

    # ----- presentation -----
    def format(self) -> str:
        """Return ``"~X remaining (avg Y/fold over last N)"`` or estimating banner."""
        if self.completed < 1:
            return "(estimating...)"
        eta = self.eta_seconds()
        if not math.isfinite(eta):
            return "(estimating...)"
        avg = self.mean_recent() or 0.0
        n = min(self.completed, self.window)
        return (
            f"~{_human(eta)} remaining "
            f"(avg {_human(avg)}/fold over last {n})"
        )


# ---------------------------------------------------------------------------
class CohortETA:
    """Wall-clock-rate ETA across participants.

    Tracks how many participants have completed so far and projects the
    remaining wall time from the observed completion rate. Because the rate
    is wall-clock, it automatically captures joblib parallelism: with K
    workers and a saturated pool, the rate is ``≈K / per_participant_time``,
    which is exactly what we want for "when will the whole cohort finish".

    Parameters
    ----------
    total_participants : int
        Total number of participants this run will process.
    n_workers : int, optional
        Informational only; included in :py:meth:`format` output so log
        readers can sanity-check the rate. Does not affect the math.
    """

    def __init__(self, total_participants: int, *, n_workers: int = 1):
        self.total = max(0, int(total_participants))
        self.n_workers = max(1, int(n_workers))
        self.completed = 0
        self.start = time.perf_counter()

    def record_completion(self, n: int = 1) -> None:
        self.completed += int(n)

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.completed)

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.start

    def rate_per_second(self) -> float:
        elapsed = self.elapsed_seconds()
        if self.completed < 1 or elapsed <= 0.0:
            return float("nan")
        return self.completed / elapsed

    def eta_seconds(self) -> float:
        rate = self.rate_per_second()
        if not math.isfinite(rate) or rate <= 0.0:
            return float("nan")
        return self.remaining / rate

    def format(self) -> str:
        if self.completed < 1:
            return "(estimating...)"
        eta = self.eta_seconds()
        elapsed = self.elapsed_seconds()
        if not math.isfinite(eta):
            return "(estimating...)"
        per = elapsed / self.completed if self.completed else float("nan")
        suffix = f" with {self.n_workers} parallel worker(s)" if self.n_workers > 1 else ""
        return (
            f"{self.completed}/{self.total} participants done, "
            f"~{_human(eta)} remaining "
            f"(avg {_human(per)}/participant{suffix})"
        )
