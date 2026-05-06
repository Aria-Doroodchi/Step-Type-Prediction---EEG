"""Runtime resource helpers.

The project config treats negative ``n_jobs`` as "reserve this many logical
CPUs" instead of relying on each library's slightly different interpretation.
For example, ``n_jobs: -8`` means "use all available logical CPUs except 8".
"""

from __future__ import annotations

import os


def resolve_n_jobs(cfg: dict, value: int | None = None, *, default: int | None = None) -> int:
    """Return a positive worker count from project resource settings."""
    resources = cfg.get("resources", {})
    requested = value
    if requested is None:
        requested = resources.get("n_jobs", default)
    if requested is None:
        return 1
    requested = int(requested)
    if requested == 0:
        return 1
    if requested < 0:
        cpus = os.cpu_count() or 1
        return max(1, cpus - abs(requested))
    return requested
