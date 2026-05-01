"""Step-Type Prediction from EEG (MSc thesis pipeline).

Top-level package. Submodules:

    config              YAML config loading and merging
    io                  filesystem helpers (paths, read/write CSV/parquet)
    logging_utils       run-stamping logger setup
    preprocessing       raw .bdf → cleaned epochs .fif
    source_localization epochs → per-participant eLORETA CSV
    features            epochs + src CSV → wide feature matrix
    models              feature selection + classifier training
    viz                 plots and reports
"""

__version__ = "0.1.0"
