param(
    [string]$OneDriveRoot = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "[onedrive-artifacts] $Message"
}

function Get-RepoRoot {
    $root = git rev-parse --show-toplevel 2>$null
    if (-not $root) {
        throw "Run this script from inside the ML_V2 Git repository."
    }
    return (Resolve-Path $root).Path
}

function Get-DefaultOneDriveRoot {
    if ($env:ML_V2_ONEDRIVE_ROOT) {
        return $env:ML_V2_ONEDRIVE_ROOT
    }

    $candidates = @(
        (Join-Path $env:USERPROFILE "OneDrive - The University of Western Ontario\MSc\Thesis\Data\ML_V2"),
        (Join-Path $env:USERPROFILE "OneDrive\MSc\Thesis\Data\ML_V2")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return $candidates[0]
}

function Test-IsJunction {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }

    $item = Get-Item -LiteralPath $Path -Force
    return (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0)
}

function Invoke-Robocopy {
    param(
        [string]$Source,
        [string]$Destination
    )

    if ($DryRun) {
        Write-Step "DRY RUN: would copy '$Source' -> '$Destination'"
        return
    }

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    robocopy $Source $Destination /E /XO /FFT /Z /R:3 /W:5 | Out-Host
    $code = $LASTEXITCODE
    if ($code -gt 7) {
        throw "Robocopy failed with exit code $code while copying '$Source' to '$Destination'."
    }
}

function New-Junction {
    param(
        [string]$LinkPath,
        [string]$TargetPath
    )

    if ($DryRun) {
        Write-Step "DRY RUN: would create junction '$LinkPath' -> '$TargetPath'"
        return
    }

    New-Item -ItemType Directory -Force -Path $TargetPath | Out-Null
    New-Item -ItemType Junction -Path $LinkPath -Target $TargetPath | Out-Null
}

function Enable-ArtifactJunction {
    param(
        [string]$RepoRoot,
        [string]$OneDriveRoot,
        [string]$RelativePath
    )

    $linkPath = Join-Path $RepoRoot $RelativePath
    $targetPath = Join-Path $OneDriveRoot $RelativePath
    $backupRoot = Join-Path $RepoRoot ".artifact-sync-backups"
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $backupPath = Join-Path $backupRoot ($RelativePath.Replace("\", "__").Replace("/", "__") + ".$timestamp")

    if (Test-IsJunction $linkPath) {
        Write-Step "$RelativePath is already a junction."
        return
    }

    if (Test-Path -LiteralPath $linkPath) {
        Write-Step "Copying existing $RelativePath into OneDrive."
        Invoke-Robocopy -Source $linkPath -Destination $targetPath

        if ($DryRun) {
            Write-Step "DRY RUN: would move local '$linkPath' backup to '$backupPath'"
        }
        else {
            New-Item -ItemType Directory -Force -Path $backupRoot | Out-Null
            Move-Item -LiteralPath $linkPath -Destination $backupPath
        }
    }

    Write-Step "Linking $RelativePath to OneDrive."
    New-Junction -LinkPath $linkPath -TargetPath $targetPath
}

$repoRoot = Get-RepoRoot
if (-not $OneDriveRoot) {
    $OneDriveRoot = Get-DefaultOneDriveRoot
}

$OneDriveRoot = [IO.Path]::GetFullPath($OneDriveRoot)
$artifactPaths = @(
    "data\interim",
    "data\features",
    "data\src",
    "outputs\runs",
    "outputs\qc",
    "outputs\stim_module\evokeds"
)

Write-Step "Repository: $repoRoot"
Write-Step "OneDrive artifact root: $OneDriveRoot"

foreach ($relativePath in $artifactPaths) {
    Enable-ArtifactJunction -RepoRoot $repoRoot -OneDriveRoot $OneDriveRoot -RelativePath $relativePath
}

Write-Step "Done. OneDrive will now sync the linked artifact folders."
Write-Step "Local pre-link backups, if any, are under .artifact-sync-backups and can be removed after you confirm OneDrive has the files."
