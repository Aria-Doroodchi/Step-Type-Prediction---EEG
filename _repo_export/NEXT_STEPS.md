# Next steps — pushing to GitHub

## TL;DR

The repo has been organized, committed, and tagged (`v1.0-organized`). The
`.git` folder lives outside OneDrive because OneDrive's Files-On-Demand
sync corrupts Git internals. To push to GitHub, **clone the bundle in this
folder to a non-OneDrive location, then push from there.**

## What's in this `_repo_export/` folder

- `step-type-prediction-eeg.bundle` — single-file Git bundle of the entire
  repo history (the `main` branch and the `v1.0-organized` tag).
- `step-type-prediction-eeg.tar.gz` — the full working tree plus the
  `.git/` folder, ready to extract anywhere.
- `NEXT_STEPS.md` — this file.

> **Important:** Do NOT do Git work directly inside this OneDrive folder.
> OneDrive will corrupt `.git/` on every sync. Use a non-synced location
> like `C:\Users\Ali D\Documents\repos\` or similar.

## Recommended workflow

### Option 1 — clone from the bundle (cleanest)

From a Windows shell, in a non-OneDrive folder:

```bash
cd C:\Users\Ali D\Documents\repos
git clone "C:\Users\Ali D\OneDrive - The University of Western Ontario\MSc\Thesis\Data\ML\_repo_export\step-type-prediction-eeg.bundle" Step-Type-Prediction---EEG
cd Step-Type-Prediction---EEG
git remote remove origin
git remote add origin https://github.com/Aria-Doroodchi/Step-Type-Prediction---EEG.git
git push -u origin main
git push origin v1.0-organized
```

GitHub will prompt for credentials. If you have a personal access token
(PAT), use that as the password.

### Option 2 — extract the tarball

If you'd rather have the whole working tree pre-extracted:

```bash
cd C:\Users\Ali D\Documents\repos
tar -xzf "C:\Users\Ali D\OneDrive - The University of Western Ontario\MSc\Thesis\Data\ML\_repo_export\step-type-prediction-eeg.tar.gz"
cd step-type-prediction-eeg
git remote -v   # already points at your GitHub repo
git push -u origin main
git push origin v1.0-organized
```

## Cleanup tasks

A few small things to do once the push succeeds:

1. **Delete the leftover `.git/` folder at the root of this OneDrive ML
   folder.** It's empty/broken and was created during a failed init
   attempt before we switched to `/tmp`. Delete it manually from Windows
   Explorer.

2. **Optionally delete `_repo_export/` from OneDrive** once you've pushed
   to GitHub. After that point GitHub is the source of truth.

3. **Optionally delete the original code files from the OneDrive ML
   folder.** Keep the data files (`CNV_epochs_df.csv`, the `.rds` files,
   `src/`) since those are gitignored and not stored on GitHub. The
   organized code lives in your non-OneDrive clone.

## About the missing figures

The 86 PNG files under `outputs/figs/` were not included in the initial
commit because OneDrive had them as cloud-only stubs that the Linux
build environment couldn't read. To add them in a follow-up commit:

1. In Windows Explorer, navigate to the original `figs/` location, right
   click on it, and choose **Always keep on this device**. Wait for
   OneDrive to fully download.
2. Copy the contents into your non-OneDrive clone at
   `Step-Type-Prediction---EEG/outputs/figs/`.
3. Then:

   ```bash
   git add outputs/figs/
   git commit -m "Add LORETA and topomap figures"
   git push
   ```

## Repo summary

```
40 files committed, 13,887 insertions
Tag: v1.0-organized
Initial commit: ee3df3a
Remote: https://github.com/Aria-Doroodchi/Step-Type-Prediction---EEG
```
