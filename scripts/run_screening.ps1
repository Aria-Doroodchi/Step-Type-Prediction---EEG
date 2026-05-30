<#
.SYNOPSIS
    Screening sweep for the model-comparison diagnostic.

.DESCRIPTION
    Warms the upstream cache for the 8 selected participants (preprocess +
    source localization + features), trains the candidate models at their
    primary tiers, runs a Lightning pass on the 3 classical models for the
    tier-response slope diagnostic, and finally aggregates everything into
    outputs/screening/SCREENING_RESULTS.md.

    The script continues past per-run failures so a partial sweep still
    produces a report.

.NOTES
    Run from any working directory; this script CDs into the repo root
    before doing anything. Activate the .venv first if you haven't already,
    or this script will try to source .venv\Scripts\Activate.ps1 itself.

    Expected wall time:
      - Cache warmup (cold):        2-6 hours (one-time, depends on participants)
      - Cache warmup (warm):        < 1 minute
      - Training all model x tiers: neural runs may add substantial time
      - Analysis:                   seconds
#>

$ErrorActionPreference = "Continue"

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot  = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

if (-not $env:VIRTUAL_ENV) {
    $ActivateScript = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
    if (Test-Path $ActivateScript) {
        Write-Host "Activating .venv..."
        & $ActivateScript
    } else {
        Write-Warning ".venv not found at $ActivateScript. Continuing with system python."
    }
}

$Python = "python"
$Participants = @("P08", "P11", "P19", "P23", "P24", "P25", "P30", "P39")

function Write-Banner {
    param([string]$Text)
    Write-Host ""
    Write-Host ("=" * 78)
    Write-Host "  $Text"
    Write-Host ("=" * 78)
}

function Invoke-Step {
    param(
        [string]$Label,
        [scriptblock]$Action
    )
    Write-Banner $Label
    & $Action
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "$Label exited with code $LASTEXITCODE -- continuing."
    }
}

$ScreenStart = Get-Date

# ----------------------------------------------------------------------------
# Step 1: Warm the upstream cache for every participant.
# Stages preprocess + src + features are model-agnostic, so we only pay this
# cost once even though we'll be training 4 models. Participants that are
# already cached are short-circuited inside run.py.
# ----------------------------------------------------------------------------
Invoke-Step "Step 1/11: Cache warmup (preprocess + src + features)" {
    & $Python run.py `
        --speed-tier express `
        --participants $Participants `
        --model xgb `
        --stages preprocess src features
}

# ----------------------------------------------------------------------------
# Step 2-7: Train each model at its primary tier.
# The participants list is the same for every step, so we exploit the cohort
# parallelism inside run.py (parallel.participants defaults to -8).
# ----------------------------------------------------------------------------
Invoke-Step "Step 2/11: Train XGB @ Express" {
    & $Python run.py `
        --speed-tier express `
        --participants $Participants `
        --model xgb `
        --run-id screen_xgb_express `
        --stages train
}

Invoke-Step "Step 3/11: Train SVM @ Express" {
    & $Python run.py `
        --speed-tier express `
        --participants $Participants `
        --model svm `
        --run-id screen_svm_express `
        --stages train
}

Invoke-Step "Step 4/11: Train Logistic Regression @ Express" {
    & $Python run.py `
        --speed-tier express `
        --participants $Participants `
        --model logistic `
        --run-id screen_logistic_express `
        --stages train
}

Invoke-Step "Step 5/11: Train Riemannian (builds tensor cache lazily)" {
    & $Python run.py `
        --speed-tier riemannian `
        --participants $Participants `
        --run-id screen_riemannian `
        --stages train
}

Invoke-Step "Step 6/11: Train CNN starter (builds tensor cache lazily)" {
    & $Python run.py `
        --speed-tier cnn `
        --participants $Participants `
        --run-id screen_cnn `
        --stages train
}

Invoke-Step "Step 7/11: Train EEGNet (builds tensor cache lazily)" {
    & $Python run.py `
        --speed-tier eegnet `
        --participants $Participants `
        --run-id screen_eegnet `
        --stages train
}

# ----------------------------------------------------------------------------
# Step 6-8: Lightning tier for the classical models -- needed for the
# tier-response slope (Diagnostic 2). Riemannian has only one tier, so no
# Lightning pass for it.
# ----------------------------------------------------------------------------
Invoke-Step "Step 8/11: Train XGB @ Lightning (slope)" {
    & $Python run.py `
        --speed-tier lightning `
        --participants $Participants `
        --model xgb `
        --run-id screen_xgb_lightning `
        --stages train
}

Invoke-Step "Step 9/11: Train SVM @ Lightning (slope)" {
    & $Python run.py `
        --speed-tier lightning `
        --participants $Participants `
        --model svm `
        --run-id screen_svm_lightning `
        --stages train
}

Invoke-Step "Step 10/11: Train Logistic @ Lightning (slope)" {
    & $Python run.py `
        --speed-tier lightning `
        --participants $Participants `
        --model logistic `
        --run-id screen_logistic_lightning `
        --stages train
}

# ----------------------------------------------------------------------------
# Step 9: Aggregate everything into the SCREENING_RESULTS.md file.
# Even if some upstream steps failed, this still produces a partial report
# from whatever runs DID complete.
# ----------------------------------------------------------------------------
$RunDirs = @(
    "outputs/runs/screen_xgb_express",
    "outputs/runs/screen_svm_express",
    "outputs/runs/screen_logistic_express",
    "outputs/runs/screen_riemannian",
    "outputs/runs/screen_cnn",
    "outputs/runs/screen_eegnet",
    "outputs/runs/screen_xgb_lightning",
    "outputs/runs/screen_svm_lightning",
    "outputs/runs/screen_logistic_lightning"
)

Invoke-Step "Step 11/11: Aggregate diagnostics into SCREENING_RESULTS.md" {
    & $Python scripts/06_compare_runs.py `
        --runs $RunDirs `
        --output outputs/screening/SCREENING_RESULTS.md
}

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
$Elapsed = (Get-Date) - $ScreenStart
Write-Host ""
Write-Host ("=" * 78)
Write-Host "Screening sweep complete in $($Elapsed.ToString())."
Write-Host ""
Write-Host "Results written to:"
Write-Host "  outputs/screening/SCREENING_RESULTS.md"
Write-Host "  SCREENING_RESULTS.md (repo-root copy)"
Write-Host ""
Write-Host "Per-run training artifacts under outputs/runs/screen_*"
Write-Host ("=" * 78)
