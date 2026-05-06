"""ICA fit + automated artifact-component classification.

Conservative auto-exclusion: a component is excluded only if ``mne-icalabel``
classifies it into one of the artifact labels (eye, muscle, heart, line,
channel) with probability > ``iclabel_artifact_prob_threshold``. The
threshold defaults to 0.8 — components below the threshold are kept.

Per-participant overrides in full override mode:

    ica:
      method: picard
      extended: true
      n_components: rank_minus_one
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
    """Fit ICA on the full-duration high-pass training copy."""
    ica_cfg = cfg["preprocessing"]["ica"]
    n_components = _resolve_n_components(raw, cfg)
    method = ica_cfg.get("method", "picard")
    extended = bool(ica_cfg.get("extended", True))
    fit_params = {}
    if method == "picard":
        fit_params["extended"] = extended

    log.info(
        "Fitting ICA on full data: method=%s, extended=%s, n_components=%s",
        method, extended, n_components,
    )

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=ica_cfg["random_state"],
        method=method,
        fit_params=fit_params or None,
        max_iter=ica_cfg.get("max_iter", "auto"),
    )
    ica.fit(raw)
    return ica


def auto_exclude(
    ica: mne.preprocessing.ICA,
    raw: mne.io.BaseRaw,
    cfg: dict,
) -> list[int]:
    """Return component indices to exclude using ICLabel + override hooks."""
    ica_cfg = cfg["preprocessing"]["ica"]
    threshold = float(ica_cfg.get("iclabel_artifact_prob_threshold", 0.8))
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
    log.info("Applying ICA unmixing to 0.1-40 Hz analysis copy")
    ica.apply(raw)
    return raw


def _resolve_n_components(raw: mne.io.BaseRaw, cfg: dict) -> int | float | None:
    requested = cfg["preprocessing"]["ica"].get("n_components", "rank_minus_one")
    if requested in (None, "none"):
        return None
    if requested == "rank_minus_one":
        rank = _eeg_rank(raw)
        n_components = max(1, rank - 1)
        log.info("ICA EEG rank=%d; using n_components=rank-1=%d", rank, n_components)
        return n_components
    return requested


def _eeg_rank(raw: mne.io.BaseRaw) -> int:
    ranks = mne.compute_rank(raw, rank=None, proj=True, verbose=False)
    if "eeg" in ranks:
        return int(ranks["eeg"])
    return len(mne.pick_types(raw.info, eeg=True, exclude=[]))


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
