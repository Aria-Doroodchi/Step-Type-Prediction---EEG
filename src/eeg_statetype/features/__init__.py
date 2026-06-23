"""State-task features (Phase 2).

Reuses eeg_steptype.features amplitude/slopes/psd/src blocks and adds the new
per-epoch ``sep`` block (vertex foot-SEP P50/N90/P2P/RMS from in-epoch e-stims).
``assemble`` registers ``sep`` in the block dispatch.
"""
