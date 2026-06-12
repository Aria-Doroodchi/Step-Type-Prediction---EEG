#!/usr/bin/env bash
# Round 1 — XGB candidate screens (8-subject express, 2.3k set), run sequentially.
# Each is ONE lever vs the base_xgb_iter baseline (cohort AUC 0.5974, gap +0.141).
set -u
cd "C:/Users/Ali D/Documents/ML"
echo "######## ROUND 1 XGB SCREENS ########"
bash outputs/perf_loop/screen.sh r1_xgb_scoring  xgb screen configs/_cand_scoring_auc.yaml
bash outputs/perf_loop/screen.sh r1_xgb_gridtight xgb screen configs/_cand_grid_tight.yaml
bash outputs/perf_loop/screen.sh r1_xgb_funnel   xgb screen configs/_cand_funnel_tight.yaml
bash outputs/perf_loop/screen.sh r1_xgb_stabthr  xgb screen configs/_cand_stab_thresh.yaml
bash outputs/perf_loop/screen.sh r1_xgb_pool     xgb screen configs/_cand_pool_partial.yaml
echo "######## ROUND 1 XGB SCREENS DONE ########"
echo "=== baseline for reference ==="
./.venv/Scripts/python.exe outputs/perf_loop/aggregate.py outputs/runs/base_xgb_iter
