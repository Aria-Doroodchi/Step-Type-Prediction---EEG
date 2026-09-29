"""Summarise benchopt result parquets into a markdown table.

    python summarize_runs.py tracks/bci_decoding/outputs/<prefix>_*.parquet

Groups runs by solver + parameters (ignoring `seed`), and reports mean, SD
and n of test balanced accuracy, plus mean run time. Sorted best first.
The header names the dataset(s) the parquets hold (the fixed Dreyer 2023
sentence only for dreyer2023 runs); with more than one dataset the table
gets a dataset column.
"""

import re
import sys

import pandas as pd

frames = [pd.read_parquet(p).assign(file=p.split("/")[-1]) for p in sys.argv[1:]]
if not frames:
    sys.exit("no parquet files given")
df = pd.concat(frames, ignore_index=True)
df["solver"] = df.solver_name.str.split("[").str[0]
df["config"] = (df.solver_name.str.extract(r"\[(.*)\]")[0].fillna("")
                .map(lambda s: re.sub(r"(^|,)seed=[^,]*", "", s).strip(",")))
# the dataset: benchopt's dataset_name (e.g. BCI[...,study=zhou2016_xsess,...],
# MockSealed[...]); the study from p_dataset_study, else from that name
names = (df["dataset_name"].astype(str) if "dataset_name" in df
         else pd.Series(["?"] * len(df), index=df.index))
study = (df["p_dataset_study"].astype(str) if "p_dataset_study" in df
         else names.str.extract(r"study=([^,\]]+)")[0])
df["dataset"] = names
datasets = sorted(names.unique())
keys = ["dataset", "solver", "config"] if len(datasets) > 1 else ["solver", "config"]
g = (df.groupby(keys)
       .agg(bal_acc_mean=("objective_balanced_accuracy", "mean"),
            bal_acc_sd=("objective_balanced_accuracy", "std"),
            n=("objective_balanced_accuracy", "size"),
            time_s=("time", "mean"))
       .reset_index().sort_values("bal_acc_mean", ascending=False))

print(f"# Results ({len(df)} runs from {len(frames)} files)\n")
if set(study.fillna("?").unique()) == {"dreyer2023"}:
    print("Dreyer 2023 test split (participants 61-81), balanced accuracy, chance 0.50.\n")
else:
    ncls = (sorted(df["objective_n_classes"].dropna().astype(int).unique())
            if "objective_n_classes" in df else [])
    chance = (f", chance {1 / ncls[0]:.2f} ({ncls[0]} classes)" if len(ncls) == 1 and ncls[0]
              else "")
    whose = "its" if len(datasets) == 1 else "each dataset's"
    print(f"Dataset{'s' if len(datasets) > 1 else ''}: {'; '.join(datasets)}. "
          f"Balanced accuracy on {whose} test split{chance}.\n")
head = "| dataset " if len(keys) == 3 else ""
print(f"{head}| solver | config | bal. acc. mean | SD | n | time (s) |")
print("|---" * (len(keys) + 4) + "|")
for r in g.itertuples():
    sd = "" if pd.isna(r.bal_acc_sd) else f"{r.bal_acc_sd:.3f}"
    lead = f"| {r.dataset} " if len(keys) == 3 else ""
    print(f"{lead}| {r.solver} | {r.config or '-'} | {r.bal_acc_mean:.3f} | {sd} | {r.n} "
          f"| {r.time_s:.0f} |")
