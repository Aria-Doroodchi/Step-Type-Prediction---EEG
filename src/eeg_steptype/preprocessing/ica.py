"""ICA fit + automated artifact-component classification.

Conservative auto-exclusion: a component is excluded only if ``mne-icalabel``
classifies it into one of the artifact labels (eye, muscle, heart, line,
channel) with probability > ``iclabel_artifact_prob_threshold``. The
threshold defaults to 0.9 — components below the threshold are kept.

Per-participant overrides:

    ica:
      n_components: 20
      train_window_seconds: [50, 100]
      manual_exclude: [3, 12]   # add to whatever ICLabel returns
      manual_keep:    [7]       # remove from the auto-exclude set

These mirror the original hand-tuned exclude lists in the per-participant
scripts; ICLabel runs alongside them so the final list is the union, minus
``manual_keep``.
"""

from __future__ import annotations

import mne

from ..logging_utils import get_logger


log = get_logger(__name__)


# ---------------------------------------------------------------------------
def fit_ica(raw: mne.io.BaseRaw, cfg: dict) -> mne.preprocessing.ICA:
    ica_cfg = cfg["preprocessing"]["ica"]
    tmin, tmax = ica_cfg["train_window_seconds"]
    hp = cfg["preprocessing"]["filter"]["ica_highpass"]

    log.info(
        "Fitting ICA: n_components=%d, train_window=[%s, %s], hp=%.2f",
        ica_cfg["n_components"], tmin, tmax, hp,
    )

    train = raw.copy().crop(tmin=tmin, tmax=tmax).filter(l_freq=hp, h_freq=None)
    ica = mne.preprocessing.ICA(
        n_components=ica_cfg["n_components"],
        random_state=ica_cfg["random_state"],
    )
    ica.fit(train)
    return ica


def auto_exclude(
    ica: mne.preprocessing.ICA,
    raw: mne.io.BaseRaw,
    cfg: dict,
) -> list[int]:
    """Return component indices to exclude using ICLabel + override hooks."""
    ica_cfg = cfg["preprocessing"]["ica"]
    threshold = float(ica_cfg.get("iclabel_artifact_prob_threshold", 0.9))
    artifact_labels = set(ica_cfg.get("artifact_labels", [
        "eye blink", "muscle artifact", "heart beat",
        "line noise", "channel noise",
    ]))

    auto: list[int] = _iclabel_predict(ica, raw, threshold, artifact_labels)

    override = cfg.get("ica") or {}
    manual_exclude = set(override.get("manual_exclude") or [])
    manual_keep    = set(override.get("manual_keep")    or [])

    final = (set(auto) | manual_exclude) - manual_keep
    log.info(
        "ICA exclude — auto=%s, manual_add=%s, manual_keep=%s, final=%s",
        sorted(auto), sorted(manual_exclude), sorted(manual_keep), sorted(final),
    )
    return sorted(final)


def apply_ica(
    ica: mne.preprocessing.ICA,
    raw: mne.io.BaseRaw,
    exclude: list[int],
) -> mne.io.BaseRaw:
    ica.exclude = list(exclude)
    ica.apply(raw)
    return raw


# ---------------------------------------------------------------------------
def _iclabel_predict(
    ica: mne.preprocessing.ICA,
    raw: mne.io.BaseRaw,
    threshold: float,
    artifact_labels: set[str],
) -> list[int]:
    """Run mne-icalabel and return indices flagged as artifacts above threshold."""
    try:
        from mne_icalabel import label_components
    except Exception as exc:                            # noqa: BLE001
        log.warning(
            "mne-icalabel unavailable (%s); ICA auto-exclusion will be empty. "
            "Use ica.manual_exclude in the participant override.",
            exc,
        )
        return []

    try:
        result = label_components(raw, ica, method="iclabel")
        labels = result["labels"]
        probs = result["y_pred_proba"]
        excluded = [
            i for i, (lbl, p) in enumerate(zip(labels, probs))
            if lbl in artifact_labels and float(p) >= threshold
        ]
        return excluded
    except Exception as exc:                            # noqa: BLE001
        log.warning("ICLabel failed: %s; falling back to empty auto-exclude.", exc)
        return []
