# codabench/ — EEG/EMG Foundation Challenge 2026

Everything for the NeurIPS 2026 **EEG/EMG Foundation Challenge** (Codabench),
**Track 2: BCI decoding (cross-session)**. Separate from the thesis
pipeline but reusing its models.

## Start here tomorrow

1. Open **Ubuntu** (Start menu) and run `source ~/codabench/env.sh`.
2. Read [LOG.md](LOG.md) to see what ran overnight and what came out.
3. Follow [TRACK2_BCI.md § Next steps](TRACK2_BCI.md#next-steps-suggested).

## Documents

| File | What's in it |
|---|---|
| [COMPETITION.md](COMPETITION.md) | Dates, tracks, rules, limits, published baselines, links |
| [TRACK2_BCI.md](TRACK2_BCI.md) | Our track: task spec, data on disk, commands, test results, next steps |
| [SETUP.md](SETUP.md) | The CPU-only WSL environment: where everything lives, rebuild steps, gotchas |
| [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md) | The `submission.py` contract, allowed imports, build → test → zip → upload |
| [REUSING_MY_MODELS.md](REUSING_MY_MODELS.md) | Which thesis models are compatible, what was ported, what wasn't and why |
| [SUBMISSIONS.md](SUBMISSIONS.md) | Ledger of uploads and reference runs |
| [LOG.md](LOG.md) | Dated log of setup steps, runs and timings |

## Layout

```text
codabench/
├── README.md, *.md          docs (above)
├── env.sh                   `source` in WSL: venv + env vars + cd
├── config/
│   └── neuralbench_config.json   NeuralBench config (~/.neuralbench/config.json links here)
├── solvers/
│   └── bci_decoding/        OUR Track 2 solvers (each file = one submission.py)
│       ├── eegnet_steptype.py    thesis EEGNet, ported
│       └── riemann_steptype.py   thesis Riemannian pipeline, ported
├── scripts/
│   ├── prepare_track2_data.sh    download/prepare Track 2 datasets (resumable)
│   └── test_track2_solvers.sh    quick real-data train+score test
├── submissions/             ZIPs ready to upload (gitignored)
├── logs/                    run logs (gitignored)
└── 2026-competition/        official competition repo (git clone; gitignored)
```

Large things live **inside WSL**, not here: the venv (`~/neuralbench/.venv`)
and all data (`~/neuralbench/benchopt_data`, `~/neuralbench/data`). See
[SETUP.md](SETUP.md#where-things-live) for why.
