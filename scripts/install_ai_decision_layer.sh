#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ_FILE="$ROOT_DIR/requirements-ai-decision.txt"

echo "AI decision layer optional dependency installer"
echo "Project: $ROOT_DIR"

if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "WARNING: no active virtualenv detected. Activate your project venv first."
  echo "Example: source .venv/bin/activate"
  exit 0
fi

echo "Using virtualenv: $VIRTUAL_ENV"
python -m pip install -r "$REQ_FILE" || {
  echo "WARNING: some optional requirements failed to install. The pipeline will use fallbacks where needed."
  echo "         pymixconsole source: git+https://github.com/csteinmetz1/pymixconsole"
}

python - <<'PY'
import importlib.util

checks = [
    ("nevergrad", "Nevergrad optimizer"),
    ("optuna", "Optuna optimizer"),
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
    ("soundfile", "SoundFile"),
    ("pyloudnorm", "pyloudnorm"),
    ("pandas", "pandas"),
    ("yaml", "PyYAML"),
    ("pymixconsole", "pymixconsole"),
    ("dasp_pytorch", "dasp-pytorch"),
]
for module, label in checks:
    if importlib.util.find_spec(module):
        print(f"OK: {label} installed")
    else:
        print(f"WARNING: {label} not installed")
print("")
print("pymixconsole: installed from upstream GitHub when available; fallback_virtual_mixer remains active if installation fails.")
print("dasp-pytorch: optional trainable DSP is disabled unless installed manually and compatible with your Python/PyTorch stack.")
PY

echo "Done. Missing optional packages do not disable the fallback offline correction pipeline."
