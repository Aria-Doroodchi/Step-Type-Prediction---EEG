# R-side analyses

This folder holds the **active R code** for the project. It sits alongside
the Python pipeline (`src/eeg_steptype/`, `scripts/`) rather than inside it,
because the two are independent: R consumes artifacts produced by the Python
pipeline (mainly under `data/` and `outputs/`), but doesn't depend on the
Python package itself.

## Layout

```
analysis_R/
  models/
    CNV_model.R           # main R modeling script
    CNV_XGB_ch.Rmd        # XGBoost-by-channel notebook
  visualization/
    ML_vis_1.R
    ML_vis_1.1_electrode_src.R   # was ML_vis_1.1_(elctd+src).R; renamed to drop parens
    ML_vis_gen.R
    ML_LSTM_vis_1.R
```

## Notes on the rename

`ML_vis_1.1_(elctd+src).R` was renamed to `ML_vis_1.1_electrode_src.R`.
Parentheses and `+` in filenames cause problems with PowerShell quoting,
`Rscript` invocation, `here::here()` resolution, and some `renv` operations.
If anything in R or in your shell history still refers to the old name,
update it.

## Inputs these scripts expect

Most of the original scripts were written against the legacy master CSV
(`CNV_epochs_df.csv`, ~4.9 GB) produced by the old
`01_preprocessing/CNV_epoch_extraction.py`. That file is gitignored and may
no longer be present.

The new Python pipeline writes:

- Per-participant source CSVs: `data/src/{Pxx}_{One|Two}_src.csv`
- Per-participant feature parquets: `data/features/{Pxx}_{One|Two}_features_t{tmin}-{tmax}.parquet`
- Per-run metrics CSVs: `outputs/runs/{run_id}/metrics.csv` and `rollup.csv`

If you want these R scripts to consume the new layout, expect to do some
adapter work (the parquet files in particular need `arrow::read_parquet`).

## RStudio project file

If you use RStudio, consider adding an `analysis_R.Rproj` here so the R
working directory is scoped to this folder. Keep it out of the Python project
root to avoid the two contexts colliding.
