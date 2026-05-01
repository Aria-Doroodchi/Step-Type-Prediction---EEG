# Data

The data files used by these scripts are **not** stored in the Git
repository — they are too large for GitHub. This document explains what's
expected, where it comes from, and how to regenerate it.

## Expected files

These files live at the repo root (or in `src/`) on the working copy.
They are listed in `.gitignore` and are not committed.

| File | Approx. size | Produced by |
|---|---:|---|
| `CNV_epochs_df.csv`        | 4.9 GB | `01_preprocessing/CNV_epoch_extraction.py` |
| `CNV_ch_df_full_disk.rds`  | 1.9 GB | R-side wrangling of the master CSV |
| `CNV_ch_df_sub_disk.rds`   |   3 MB | Subset of the above for prototyping |
| `CNV_binned.rds`           |  30 MB | Time-binned version used by `02_models/R/CNV_model.R` |
| `src/Pxx_{One,Two}_src.csv` | ~2 MB × 60 files | `01_preprocessing/SRC_writer.py` |

## Original source

The raw EEG (.fif epoch files per participant per condition) lives outside
this repo, under the lab's Participants folder
(`Thesis/Data/Participants/bad_interpolated/Epochs/CNV/`). The
preprocessing scripts read from that location.

## Regenerating the data

1. Make sure the `bad_interpolated/Epochs/CNV/*.fif` files are in place
   under the Participants folder.
2. Run `01_preprocessing/CNV_epoch_extraction.py` to produce
   `CNV_epochs_df.csv`.
3. Run `01_preprocessing/SRC_writer.py` to produce the per-participant
   source-localized CSVs in `src/`.
4. The R `.rds` files are produced by the R modeling scripts on first
   run; see `02_models/R/CNV_model.R`.

## Known portability issue

The preprocessing and modeling scripts contain hard-coded paths of the
form `C:/Users/Aria/OneDrive - The University of Western Ontario/...`.
These will need to be edited (or replaced with a config / environment
variable) before the scripts will run on a different machine.
