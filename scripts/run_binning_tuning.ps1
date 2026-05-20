<#
.SYNOPSIS
    Long-running binning-variant sweep for rich EEG features.

.DESCRIPTION
    Builds rich feature parquets for amplitude-binning variants, trains the
    four screened model families on a 20-participant subset, and writes one
    comparison report per (prediction window, binning variant).

    The three tabular models (xgb, svm, logistic) use:
      configs/features_rich.yaml + configs/express.yaml + configs/binning/*.yaml

    Riemannian does not consume the tabular binned features, so it is run once
    per prediction window as a comparator and reused in each report.

.PARAMETER Resume
    Skip forced feature rebuilds. Training checkpoints are always reused.
#>

param(
    [switch]$Resume
)

$ErrorActionPreference = "Continue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot = Split-Path -Parent $ScriptDir
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
$Participants = @(
    "P01", "P02", "P03", "P05", "P06",
    "P07", "P08", "P10", "P11", "P12",
    "P13", "P14", "P15", "P19", "P23",
    "P24", "P25", "P30", "P35", "P39"
)
$TabularModels = @("xgb", "svm", "logistic")
$Windows = @("late_cnv", "full_cnv")
$ResourceArgs = @("--n-jobs", "-3", "--parallel-participants", "-3")

$Variants = @(
    [pscustomobject]@{
        Name = "rich_mean_0125"
        Configs = @("configs/features_rich.yaml", "configs/express.yaml", "configs/binning/rich_mean_0125.yaml")
        Windows = @("late_cnv", "full_cnv")
    },
    [pscustomobject]@{
        Name = "stats_0125"
        Configs = @("configs/features_rich.yaml", "configs/express.yaml", "configs/binning/stats_0125.yaml")
        Windows = @("late_cnv")
    },
    [pscustomobject]@{
        Name = "pyramid_mean_core"
        Configs = @("configs/features_rich.yaml", "configs/express.yaml", "configs/binning/pyramid_mean_core.yaml")
        Windows = @("late_cnv")
    },
    [pscustomobject]@{
        Name = "pyramid_mean_fine"
        Configs = @("configs/features_rich.yaml", "configs/express.yaml", "configs/binning/pyramid_mean_fine.yaml")
        Windows = @("late_cnv")
    },
    [pscustomobject]@{
        Name = "stats_pyramid_core"
        Configs = @("configs/features_rich.yaml", "configs/express.yaml", "configs/binning/stats_pyramid_core.yaml")
        Windows = @("late_cnv", "full_cnv")
    }
)

function Write-Banner {
    param([string]$Text)
    Write-Host ""
    Write-Host ("=" * 90)
    Write-Host "  $Text"
    Write-Host ("=" * 90)
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

function Invoke-BinningReport {
    param(
        [string]$Window,
        [string]$VariantName
    )
    $RunDirs = @()
    foreach ($Model in $TabularModels) {
        $RunDirs += "outputs/runs/bin_${Window}_${VariantName}_${Model}"
    }
    $RunDirs += "outputs/runs/bin_${Window}_riemannian"
    $Output = "outputs/screening/binning_${Window}_${VariantName}.md"

    Invoke-Step "Report: $Window / $VariantName" {
        & $Python scripts/06_compare_runs.py `
            --runs $RunDirs `
            --output $Output `
            --no-root-copy `
            --default-tier express
    }
}

$Started = Get-Date
$LogDir = Join-Path $RepoRoot "outputs\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Transcript = Join-Path $LogDir ("binning_tuning_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
Start-Transcript -Path $Transcript | Out-Null

Write-Banner "Binning sweep setup"
Write-Host "Participants: $($Participants -join ', ')"
Write-Host "Resource posture: --n-jobs -3 --parallel-participants -3"
Write-Host "Resume mode: $Resume"
Write-Host "Transcript: $Transcript"

# Riemannian comparator. It uses raw epoch tensors, not tabular binning
# variants, so one run per window is enough.
foreach ($Window in $Windows) {
    $RunId = "bin_${Window}_riemannian"
    Invoke-Step "Riemannian comparator / $Window" {
        & $Python run.py `
            --speed-tier riemannian `
            --prediction-window $Window `
            --participants $Participants `
            --model riemannian `
            --run-id $RunId `
            @ResourceArgs `
            --stages train
    }
}

foreach ($Variant in $Variants) {
    foreach ($Window in $Variant.Windows) {
        $FeatureArgs = @(
            "run.py",
            "--config"
        ) + $Variant.Configs + @(
            "--prediction-window", $Window,
            "--participants"
        ) + $Participants + $ResourceArgs + @(
            "--model", "xgb",
            "--stages", "features"
        )
        if (-not $Resume) {
            $FeatureArgs += "--force"
        }

        Invoke-Step "Feature build / $Window / $($Variant.Name)" {
            & $Python @FeatureArgs
        }

        foreach ($Model in $TabularModels) {
            $RunId = "bin_${Window}_$($Variant.Name)_${Model}"
            Invoke-Step "Train $Model / $Window / $($Variant.Name)" {
                & $Python run.py `
                    --config $($Variant.Configs) `
                    --prediction-window $Window `
                    --participants $Participants `
                    --model $Model `
                    --run-id $RunId `
                    @ResourceArgs `
                    --stages train
            }
        }

        Invoke-BinningReport -Window $Window -VariantName $Variant.Name
    }
}

$Elapsed = (Get-Date) - $Started
Write-Banner "Binning sweep complete"
Write-Host "Elapsed: $($Elapsed.ToString())"
Write-Host "Reports: outputs/screening/binning_*.md"
Write-Host "Transcript: $Transcript"

Stop-Transcript | Out-Null
