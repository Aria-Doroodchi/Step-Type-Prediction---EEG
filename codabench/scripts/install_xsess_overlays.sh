#!/usr/bin/env bash
# Install the cross-session dataset overlays (codabench/config/nb_overlays/)
# into the NeuralBench package and register them in the (untracked) upstream
# clone's bci_studies.py _OVERLAYS. Idempotent; re-run after re-creating the
# venv or re-cloning 2026-competition.
#   bash ~/codabench/scripts/install_xsess_overlays.sh
set -eu
source "$HOME/codabench/env.sh" >/dev/null
NB=$(python -c 'import neuralbench, os; print(os.path.dirname(neuralbench.__file__))')
for f in "$HOME"/codabench/config/nb_overlays/motor_imagery/*.yaml; do
  ln -sf "$f" "$NB/tasks/eeg/motor_imagery/datasets/$(basename "$f")"
  echo "linked $(basename "$f")"
done
BS="$HOME/codabench/2026-competition/tracks/bci_decoding/datasets/bci_studies.py"
python - "$BS" <<'PY'
import sys, re
p = sys.argv[1]; s = open(p).read()
new = ["tangermann2012_xsess", "zhou2016_xsess", "scherer2015_xsess", "scherer2015_xsess3"]
missing = [n for n in new if f'"{n}"' not in s]
if missing:
    add = "".join(f'    "{n}": "{n}",\n' for n in missing)
    s = s.replace('    "dreyer2023": "dreyer2023",\n', '    "dreyer2023": "dreyer2023",\n' + add)
    open(p, "w").write(s)
print("bci_studies _OVERLAYS now has:", [n for n in new if f'"{n}"' in open(p).read()])
PY
