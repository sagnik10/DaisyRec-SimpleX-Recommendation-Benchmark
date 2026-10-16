#!/usr/bin/env bash
set -euo pipefail
DAISY_VENV="${DAISY_VENV:-$HOME/.venvs/daisyrec}"
python3 -m venv --without-pip "$DAISY_VENV"
if ! "$DAISY_VENV/bin/python" -m pip --version >/dev/null 2>&1; then
    DAISY_BOOTSTRAP=$(mktemp)
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$DAISY_BOOTSTRAP"
    "$DAISY_VENV/bin/python" "$DAISY_BOOTSTRAP"
    rm "$DAISY_BOOTSTRAP"
fi
"$DAISY_VENV/bin/python" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
"$DAISY_VENV/bin/python" -m pip install -r requirements.txt
printf 'Activate with: source "%s/bin/activate"\n' "$DAISY_VENV"
