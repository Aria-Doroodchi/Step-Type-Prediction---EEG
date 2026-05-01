"""Source localization (eLORETA).

Each module owns one concern; ``pipeline.py`` ties them together.

Key efficiency note: the original ``SRC_writer.py`` rebuilt the forward
solution and inverse operator inside the per-epoch loop. Here both are
computed *once per participant* and reused for every epoch — typically an
order-of-magnitude speedup.
"""
