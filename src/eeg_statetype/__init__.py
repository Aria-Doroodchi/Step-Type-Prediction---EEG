"""3-state motor-state EEG classification (standing / straight / diagonal).

Parallel sibling of ``eeg_steptype`` (binary CNV step-type). Mirrors that
pipeline's architecture; only the dependent variable (2 -> 3 classes) and the
independent variables (CNV-window blocks + a new per-epoch foot-SEP block)
change. Reuses CNV/stim machinery by import; the CNV package is never modified.

Source of truth for results: ``outputs/state_module/LEDGER.md``.
"""

__all__ = ["config", "io"]
