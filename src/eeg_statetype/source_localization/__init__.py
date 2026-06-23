"""State-task source localization.

Thin wrapper that reuses the CNV forward/inverse/labels primitives (the
leadfield is montage-only and identical across tasks) on the state module's
non-CSD avg-ref source epochs, writing per-epoch label time-courses to the
state ``src_state/`` namespace. ``src`` is a default feature block.
"""
