# Build the remaining 12 participants for the 3-state (SEP) model, then retrain
# the model + ablation on all 32. Runs in a visible window for live monitoring.
# Resumable: skips already-built participants and reloads trained ones.
# ASCII-only on purpose (Windows PowerShell 5.1 mis-parses non-ASCII glyphs).

$ErrorActionPreference = "Continue"
$ROOT = "C:\Users\Ali D\Documents\ML"
$PY   = Join-Path $ROOT ".venv\Scripts\python.exe"
Set-Location $ROOT
$env:PYTHONUTF8 = "1"
$env:PYTHONWARNINGS = "ignore::FutureWarning"

# Keep the machine awake for the duration (idle-sleep was killing past runs).
powercfg /change standby-timeout-ac 0 | Out-Null
powercfg /change standby-timeout-dc 0 | Out-Null
powercfg /change hibernate-timeout-ac 0 | Out-Null
powercfg /change hibernate-timeout-dc 0 | Out-Null

$REMAIN = @("P21","P22","P23","P24","P25","P27","P28","P31","P33","P35","P37","P39")

function Banner($t) {
  Write-Host ""
  Write-Host ("=" * 78) -ForegroundColor Cyan
  Write-Host ("  " + $t + "   [" + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + "]") -ForegroundColor Cyan
  Write-Host ("=" * 78) -ForegroundColor Cyan
}

Banner "STEP 1 of 2  -  Building the final 12 participants (preprocess - src - features)"
Write-Host ("Participants: " + ($REMAIN -join ", ")) -ForegroundColor Yellow
Write-Host "Per-participant heartbeats below (each step is its own timeout-guarded subprocess)."
& $PY "scripts\state_module\run_cohort.py" --config "configs\state\screen32.yaml" --participants $REMAIN --stages preprocess src features

Banner "STEP 2 of 2  -  Retraining 3-class model + 4-arm ablation on all 32 participants"
Write-Host "Reuses the 20 already-trained participants from checkpoints; computes the 12 new ones."
Write-Host "Arms: combined / window / electrode / sep. The comparison table prints at the end."
& $PY "scripts\state_module\screen_ablation.py" xgb "configs\state\screen32.yaml"

Banner "DONE  -  Full 32-participant build + retrain complete"
Write-Host "Results: outputs\state_module\runs\state_screen_xgb_*  (rollup.csv per arm)" -ForegroundColor Green
Write-Host "Figures: outputs\state_module\figs\" -ForegroundColor Green
Write-Host "This window will stay open; you can close it now." -ForegroundColor Green
