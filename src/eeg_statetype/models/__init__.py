"""State-task models (Phase 3).

Generalizes the CNV nested-CV driver from binary to 3-class multiclass:
label map {standing:0, straight:1, diagonal:2}, per-class sample weights,
macro one-vs-rest AUC scoring, 3x3 confusion + per-class recall metrics, and a
``multi:softprob`` XGB factory. Reuses the CNV feature-selection funnel and CV.
"""
