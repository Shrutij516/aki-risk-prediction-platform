#!/usr/bin/env bash
# One-command local demo: synthetic data -> PostgreSQL -> models -> FastAPI + React at http://127.0.0.1:8000
#
#   scripts/demo.sh                      # model source auto: real-trained bundles if present locally, else synthetic-trained
#   AKI_MODEL_SOURCE=synthetic scripts/demo.sh
#   PORT=8001 scripts/demo.sh
#
# Uses ONLY synthetic patients (database aki_demo). Real data, if present, is not read.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8000}"

echo "==> Python environment"
if [ ! -x .venv/bin/python ]; then
  command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/"; exit 1; }
  uv venv .venv --python 3.11
fi
uv pip install --quiet --python .venv/bin/python -e ".[dev]"

echo "==> Local PostgreSQL (embedded, Unix socket only)"
.venv/bin/python -m aki.db.local_server start >/dev/null

echo "==> Synthetic demo database + model scoring"
MLFLOW_DISABLE_AGENT_HINT=1 .venv/bin/python -m aki.demo.build --reset --model-source "${AKI_MODEL_SOURCE:-auto}"

echo "==> Frontend build"
command -v npm >/dev/null || { echo "Node.js/npm is required"; exit 1; }
(cd web && { [ -d node_modules ] || npm ci --no-audit --no-fund; } && npm run build)

echo "==> Serving http://127.0.0.1:${PORT}  (Ctrl-C to stop)"
exec .venv/bin/uvicorn aki.api.main:app --host 127.0.0.1 --port "${PORT}"
