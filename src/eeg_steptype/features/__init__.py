"""Feature blocks (amplitude, slopes, PSD) + assemble into a wide matrix.

These were inline in CNV_XGB_4.3.py / CNV_LSTM_3.py / CNV_ML_SVM_1.py.
Extracted here so all three model lines share the same code path and so the
result can be cached to parquet (avoiding re-computation on every model run).
"""
