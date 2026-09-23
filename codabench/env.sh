# Source this inside WSL (Ubuntu) before any competition work:
#
#     source ~/codabench/env.sh
#
# Activates the CPU-only Python 3.12 venv, points benchopt's data folder at
# the Linux filesystem (not OneDrive), and moves into this folder.
# See SETUP.md for what each piece is and why it lives where it does.

export PATH="$HOME/.local/bin:$PATH"          # uv
source "$HOME/neuralbench/.venv/bin/activate"   # Python 3.12, torch+cpu, neuralbench 0.3.1

# neuralbench reads ~/.neuralbench/config.json, which is a symlink to
# config/neuralbench_config.json in this folder. Set explicitly as well.
export NEURALBENCH_CONFIG="$HOME/codabench/config/neuralbench_config.json"

# Large downloads stay on the WSL ext4 disk: fast, and out of OneDrive/git.
export BENCHOPT_DATA_HOME="$HOME/neuralbench/benchopt_data"
mkdir -p "$BENCHOPT_DATA_HOME"

# Same noise suppression as the Codabench scoring image (tools/Dockerfile).
export PYTHONWARNINGS=ignore::FutureWarning
export MNE_LOGGING_LEVEL=ERROR

cd "$HOME/codabench"
echo "codabench env ready: $(python --version), torch $(python -c 'import torch; print(torch.__version__)'), cwd $(pwd)"
