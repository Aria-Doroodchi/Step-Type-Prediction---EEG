"""Summarise benchopt result parquets into a markdown table.

    python summarize_runs.py tracks/bci_decoding/outputs/<prefix>_*.parquet

Groups runs by solver + parameters (ignoring `seed`), and reports mean, SD
and n of test balanced accuracy, plus mean run time. Sorted best first.
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
g = (df.groupby(["solver", "config"])
       .agg(bal_acc_mean=("objective_balanced_accuracy", "mean"),
            bal_acc_sd=("objective_balanced_accuracy", "std"),
            n=("objective_balanced_accuracy", "size"),
            time_s=("time", "mean"))
       .reset_index().sort_values("bal_acc_mean", ascending=False))

print(f"# Results ({len(df)} runs from {len(frames)} files)\n")
print("Dreyer 2023 test split (participants 61-81), balanced accuracy, chance 0.50.\n")
print("| solver | config | bal. acc. mean | SD | n | time (s) |")
print("|---|---|---|---|---|---|")
for r in g.itertuples():
    sd = "" if pd.isna(r.bal_acc_sd) else f"{r.bal_acc_sd:.3f}"
    print(f"| {r.solver} | {r.config or '-'} | {r.bal_acc_mean:.3f} | {sd} | {r.n} | {r.time_s:.0f} |")
