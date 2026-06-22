# OneDrive Artifact Sync

How large, Git-ignored pipeline artifacts are shared across machines using
OneDrive and Windows directory junctions.

## Why this exists

The pipeline produces large generated artifacts — feature parquets, interim
data, and run outputs — that should **not** live in Git (they are big, binary,
and regenerable). They are already excluded by `.gitignore`:

| Artifact folder   | Contents                                      |
| ----------------- | --------------------------------------------- |
| `data/interim`    | Intermediate preprocessing data               |
| `data/features`   | Extracted feature parquets                    |
| `data/src`        | Source-localization inputs / intermediates    |
| `outputs/runs`    | Per-run model outputs                         |
| `outputs/qc`      | Quality-control artifacts                     |
| `outputs/stim_module/evokeds` | Stim-module cached per-participant SEP evokeds (.npz) |

Keeping them out of Git solves the "don't bloat the repo" problem but creates a
new one: **how do you get the same artifacts onto a second machine** without
re-running the whole pipeline? This module is the answer — it lets OneDrive
carry the big files while Git carries only the code.

## The core idea: junctions

After setup, each artifact folder in the repo is no longer a real directory —
it is a **directory junction** (a Windows reparse point, similar to a symlink)
that transparently redirects to a folder inside your OneDrive.

```
Repo (Git)                              OneDrive (synced across machines)
-----------                             ---------------------------------
ML/data/features  ──[junction]────────▶ .../ML_V2/data/features
ML/outputs/runs   ──[junction]────────▶ .../ML_V2/outputs/runs
        ...                                       ...
```

Because the redirect happens at the filesystem level, **the pipeline is
completely unaware of it**. Scripts read and write `data/features` exactly as
before; the bytes simply land in the OneDrive-backed location, and OneDrive
syncs them to your other machine in the background.

Net effect:

- **Code** travels through **Git** (`git push` / `git pull`).
- **Large artifacts** travel through **OneDrive** (automatic background sync).
- Both appear as one normal folder tree on every machine.

## Usage

### First-time setup on a machine

1. Clone the repo and let OneDrive finish syncing the `ML_V2` artifact folder.
2. Run the setup once:

   ```
   make sync-setup
   ```

   which is shorthand for:

   ```
   powershell -ExecutionPolicy Bypass -File scripts/setup_onedrive_artifacts.ps1
   ```

3. Run the pipeline as usual — the artifact folders now resolve to OneDrive.

### Overriding the OneDrive location

The script auto-detects the OneDrive root (see below). To point it elsewhere,
either set an environment variable:

```
$env:ML_V2_ONEDRIVE_ROOT = "C:\path\to\ML_V2"
make sync-setup
```

or pass it explicitly:

```
powershell -ExecutionPolicy Bypass -File scripts/setup_onedrive_artifacts.ps1 -OneDriveRoot "C:\path\to\ML_V2"
```

### Previewing without making changes

```
powershell -ExecutionPolicy Bypass -File scripts/setup_onedrive_artifacts.ps1 -DryRun
```

`-DryRun` logs every copy / move / junction it *would* perform, but changes
nothing on disk.

## How the script works (step by step)

Source: `scripts/setup_onedrive_artifacts.ps1`

1. **Locate the repo** — `git rev-parse --show-toplevel`; aborts if not run from
   inside the Git repository.

2. **Resolve the OneDrive root**, in priority order:
   1. `-OneDriveRoot` parameter, if given.
   2. `$ML_V2_ONEDRIVE_ROOT` environment variable, if set.
   3. First existing path among the defaults:
      - `%USERPROFILE%\OneDrive - The University of Western Ontario\MSc\Thesis\Data\ML_V2`
      - `%USERPROFILE%\OneDrive\MSc\Thesis\Data\ML_V2`
   4. Falls back to the first default path if none exist yet.

3. **For each artifact folder**, run `Enable-ArtifactJunction`, which is
   **idempotent** and **non-destructive**:
   - If the path is **already a junction** → skip it.
   - If a **real folder exists** there → `robocopy` its contents into the
     OneDrive target first (`/E /XO /FFT /Z /R:3 /W:5` — recursive, skip older
     files, FAT-time tolerance, restartable, with retries), then **move** the
     original local folder into a timestamped backup under
     `.artifact-sync-backups/` rather than deleting it.
   - Finally, **create the junction** from the repo path to the OneDrive target.

4. **Report** completion and remind you that pre-link backups live under
   `.artifact-sync-backups/` and can be deleted once you confirm OneDrive holds
   the files.

### Safety properties

- **No data loss** — existing local artifacts are copied to OneDrive *and*
  backed up locally before the original is replaced by a junction.
- **Idempotent** — re-running skips already-linked folders, so it is safe to run
  after pulling new artifact paths.
- **Robocopy hardening** — exit codes `> 7` (genuine failures) throw; codes
  `0–7` (success / files-copied / minor) are treated as success.
- **`.artifact-sync-backups/`** is itself Git-ignored, so the backups never get
  committed.

## Files involved

| File                                      | Role                                              |
| ----------------------------------------- | ------------------------------------------------- |
| `scripts/setup_onedrive_artifacts.ps1`    | The setup script (copy → backup → junction)       |
| `Makefile` (`sync-setup` target)          | Convenience entry point: `make sync-setup`        |
| `README.md`                               | Short usage note in the quick-start section       |
| `.gitignore`                              | Ignores the artifact folders + `.artifact-sync-backups/` |

## Operational notes & gotchas

- **OneDrive must finish syncing first.** On a fresh machine, wait for the
  `ML_V2` folder to download before running `make sync-setup`, or the junction
  will point at an empty/partial folder.
- **Files On-Demand.** If OneDrive Files On-Demand is on, artifacts may be
  cloud-only placeholders until first access; large pipeline reads will trigger
  downloads. Mark the artifact folder "Always keep on this device" if you want
  them hydrated up front.
- **Junctions are local.** The junction itself is never committed (the folders
  are Git-ignored). Each machine creates its own junctions via `make sync-setup`.
- **Cleanup.** Once OneDrive confirms it holds everything, the timestamped
  folders under `.artifact-sync-backups/` can be removed.
- **Reverting.** To undo, delete the junction (`rmdir <path>` — this removes only
  the link, not the OneDrive target) and copy the data back from OneDrive into a
  real folder.
