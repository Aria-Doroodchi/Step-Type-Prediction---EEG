"""State-task preprocessing (Phase 1).

Dual-branch per condition across two source files (Stim + Standing):
  (a) CNV automated chain -> CSD analysis epochs (window features);
  (b) non-CSD avg-ref branch -> source epochs + per-e-stim SEP epochs.
Reuses eeg_steptype.preprocessing primitives; forks only events + the
orchestrator (two files, 3 conditions, e-stim offset + attribution).
"""
