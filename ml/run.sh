#!/usr/bin/env bash
# Convenience runner for the neural segmentation pipeline.
# Usage: ./run.sh <prepare|train|train-quick|export|all|all-quick> [extra train.py args...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
SELF="$ROOT/run.sh"

PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing ml/.venv — create it once:"
  echo "  cd ml && python3 -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
  echo "  pip install -r requirements.txt"
  exit 1
fi

cmd="${1:-}"
shift || true

case "$cmd" in
  prepare)
    exec "$PY" prepare_dataset.py "$@"
    ;;
  train)
    # Documented default: LOO + full fit
    exec "$PY" train.py --mode both --epochs 80 --repeats 48 "$@"
    ;;
  train-quick)
    # Faster smoke (~15–20 min on an RTX 5080 with n=4)
    exec "$PY" train.py --mode both --epochs 40 --repeats 32 "$@"
    ;;
  export)
    "$PY" export_onnx.py "$@"
    mkdir -p ../public/models
    cp -f export/toolseg.onnx export/toolseg.json ../public/models/
    echo "Copied export/toolseg.{onnx,json} → public/models/"
    ;;
  all)
    "$SELF" prepare
    "$SELF" train "$@"
    "$SELF" export
    ;;
  all-quick)
    "$SELF" prepare
    "$SELF" train-quick "$@"
    "$SELF" export
    ;;
  *)
    echo "Usage: $SELF <prepare|train|train-quick|export|all|all-quick> [args...]"
    echo
    echo "  prepare      rebuild ml/dataset from training/*.jpg(+.json)"
    echo "  train        LOO+full, epochs=80 repeats=48 (~45–60 min @ n=4)"
    echo "  train-quick  LOO+full, epochs=40 repeats=32 (~20 min @ n=4)"
    echo "  export       ONNX → ml/export/ and public/models/"
    echo "  all          prepare + train + export"
    echo "  all-quick    prepare + train-quick + export"
    exit 1
    ;;
esac
