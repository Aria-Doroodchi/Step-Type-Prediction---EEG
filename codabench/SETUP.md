# Setup — the CPU-only environment on this machine

How the competition environment is built, where everything lives, and how to
rebuild it. Built 2026-09-23.

## The machine

- Windows 11 desktop, i7-12700K (20 logical cores), 64 GB RAM,
  **no NVIDIA GPU** (Intel UHD 770 only), so training is CPU-only. That's
  allowed: training compute is uncapped, and scoring happens on the
  organisers' GPU.
- Power: AC sleep and hibernate are "never" (checked 2026-09-23), so overnight
  jobs aren't killed by idle sleep.

## Why WSL (Ubuntu) and not native Windows

`neuralbench/main.py` does `import resource`, a **Unix-only** module, so
NeuralBench training can't run on native Windows. The competition's
scoring image is Linux too. Everything runs inside **WSL2 Ubuntu 26.04**
(user `ali_d`).

## Where things live

| What | Where | Why there |
|---|---|---|
| This folder (docs, our solvers, scripts, config) | `C:\Users\Ali D\Documents\ML\codabench\` = WSL `~/codabench` (symlink) | tracked in git; edit from Windows or WSL |
| Official competition code | `codabench/2026-competition/` (git clone, not tracked by our repo) | benchopt runs from here |
| Python env | WSL `~/neuralbench/.venv` (Python 3.12) | a venv must sit on the Linux disk (fast, correct permissions) |
| Competition data (benchopt) | WSL `~/neuralbench/benchopt_data/neural_compet/` | large; kept off OneDrive and out of git |
| NeuralBench data / cache / results | WSL `~/neuralbench/{data,cache,results}` | same |
| NeuralBench config | `codabench/config/neuralbench_config.json`, symlinked from WSL `~/.neuralbench/config.json` | one tracked copy |
| Logs | `codabench/logs/` (gitignored) | |

From Windows Explorer, the WSL side is at `\\wsl.localhost\Ubuntu\home\ali_d\`.

**Don't** move the data or the venv into the OneDrive-synced `Documents` folder.
The datasets run to tens of GB (Stieger 2021 is large), OneDrive would try to sync
them, and file access across the Windows/WSL boundary is several times slower.

## Daily use

Open **Ubuntu** (Start menu) and run:

```bash
source ~/codabench/env.sh
```

That activates the venv, sets `NEURALBENCH_CONFIG` and `BENCHOPT_DATA_HOME`,
and moves into `~/codabench`. Then work from `2026-competition/`, e.g.:

```bash
cd 2026-competition
benchopt run tracks/bci_decoding -d Simulated -o "BCI-decoding[training=True]"
```

## What's installed (venv `~/neuralbench/.venv`)

| Package | Version | Note |
|---|---|---|
| Python | 3.12.14 | via `uv` (Ubuntu's system Python is 3.14) |
| torch / torchvision / torchaudio | 2.14.0+cpu / 0.29.0+cpu / 2.11.0+cpu | **CPU wheels** from the PyTorch CPU index |
| neuralbench / neuralset / neuralfetch | 0.3.1 | = the competition's pinned versions |
| exca | 0.5.29 | = pinned |
| benchopt | 1.10.0 | the competition runner |
| moabb | 1.7.2 | EEG datasets (pulls in `pyriemann`) |
| braindecode | 1.8.1 | |
| mne | 1.13.2 | |
| scikit-learn | 1.9.1 | |

## Rebuild from scratch

```bash
# 1. uv + Python 3.12 venv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
mkdir -p ~/neuralbench && cd ~/neuralbench && uv venv --python 3.12 .venv
source .venv/bin/activate

# 2. CPU torch FIRST, so neuralbench doesn't pull the multi-GB CUDA build
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. the competition stack (pins from 2026-competition/requirements.txt)
uv pip install neuralbench==0.3.1 neuralset==0.3.1 neuralfetch==0.3.1 exca==0.5.29 \
               "benchopt>=1.10" moabb mne braindecode pandas scikit-learn

# 4. config + links
mkdir -p ~/.neuralbench ~/neuralbench/{data,cache,results}
ln -sfn "/mnt/c/Users/Ali D/Documents/ML/codabench" ~/codabench
ln -sfn ~/codabench/config/neuralbench_config.json ~/.neuralbench/config.json

# 5. competition code (see "symlink fix" below)
cd ~/codabench && git clone https://github.com/neural-interfaces26/2026-competition.git
```

Check: `python -c "import torch; print(torch.__version__)"` should end in `+cpu`.

## Gotchas hit during setup

1. **Symlinks in the competition repo (important).** The repo stores
   `tracks/*/benchmark_utils` and `codabench/phases/warmup/*/benchmark` as git
   symlinks. Cloned with *Windows* git (`core.symlinks=false`), they become
   one-line text files, and benchopt fails with
   `ModuleNotFoundError: No module named 'benchmark_utils'`. Fix, run in WSL:
   ```bash
   cd ~/codabench/2026-competition
   for t in tracks/*/; do rm -f "$t/benchmark_utils"; ln -s ../../benchmark_utils "$t/benchmark_utils"; done
   for f in codabench/phases/warmup/*/benchmark; do [ -L "$f" ] || { t=$(cat "$f"); rm "$f"; ln -s "$t" "$f"; }; done
   ```
   Re-run this after any `git pull` that touches those paths, or pull from WSL
   git, which keeps them as real symlinks.
2. **Stale weights across datasets.** Every local run of a solver shares
   `tracks/bci_decoding/outputs/<SolverName>/`. The upstream baselines load any
   `weights.pt` found there, even in a *training* run, so switching datasets
   (different channel count) crashes with `Error(s) in loading state_dict`.
   Our solvers skip saved weights when training. For the upstream ones,
   delete the folder first.
3. **HTTP/3 warning on downloads** (`MustDowngradeError ... HttpVersion.h3`).
   Harmless: urllib3 retries over HTTP/2 and the download proceeds.
4. **`benchopt install`** would try to create a conda env. Don't use it here;
   the venv already has everything (install extras with `uv pip install`).
5. Don't pass `--config` or `NEURALBENCH_CONFIG` pointing at a missing file:
   check the first log lines of any run.

## Updating the competition code

```bash
cd ~/codabench/2026-competition && git pull   # WSL git keeps symlinks intact
```

Our solvers live outside the clone (`codabench/solvers/`), so a pull never
touches them. Re-check `requirements.txt` after a pull: if the pinned versions
change, match them in the venv.
