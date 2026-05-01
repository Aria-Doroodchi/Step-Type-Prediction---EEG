"""Quick visual summary of a training run's metrics CSV."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..io import ensure_dir
from ..logging_utils import get_logger


log = get_logger(__name__)


def plot_per_participant_accuracy(metrics_csv: Path, out_png: Path) -> Path:
    df = pd.read_csv(metrics_csv)
    df = df.sort_values("overall_accuracy", ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(3, 0.25 * len(df))))
    ax.barh(df["participant_id"], df["overall_accuracy"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Overall accuracy")
    ax.set_title(f"Per-participant accuracy ({metrics_csv.parent.name})")
    fig.tight_layout()
    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", out_png)
    return out_png
