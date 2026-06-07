#!/usr/bin/env bash
# run.sh — run any repo script using the active conda environment.
#
# Usage:
#   ./run.sh classification/train.py [args...]
#   ./run.sh generative/train.py [args...]
#   ./run.sh generative/eval.py [args...]
#
# The script must be run from the repo root (or the data root when data files
# such as banking77/ are expected relative to the working directory).
#
# Optionally set CONDA_ENV to point to a specific conda environment:
#   CONDA_ENV=/path/to/envs/myenv ./run.sh classification/train.py [args...]

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <script.py> [args...]"
    exit 1
fi

if [[ -n "$CONDA_ENV" ]]; then
    PYTHON="$CONDA_ENV/bin/python3"
else
    PYTHON="$(which python3)"
fi

exec "$PYTHON" "$@"
