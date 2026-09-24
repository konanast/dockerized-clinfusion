#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
  if [[ -t 0 ]]; then
    read -r -p "Existing ClinFusion cache directory (the folder containing models/) [./cache]: " CACHE_INPUT
    if [[ -n "${CACHE_INPUT}" ]]; then
      python - "${CACHE_INPUT}" <<'PYENV'
from pathlib import Path
import sys
p = Path('.env')
value = sys.argv[1]
lines = p.read_text().splitlines()
lines = [f"HOST_CACHE_DIR={value}" if x.startswith("HOST_CACHE_DIR=") else x for x in lines]
p.write_text("\n".join(lines) + "\n")
PYENV
    fi
  else
    echo "If model downloads are elsewhere, set HOST_CACHE_DIR in .env before rerunning."
  fi
  echo
fi

# v2 .env files did not have MODEL_VARIANT. Adding the 8B default is deliberately non-disruptive.
if ! grep -q '^MODEL_VARIANT=' .env; then
  printf '\n# Added by v3; preserves v2 behavior\nMODEL_VARIANT=8B\n' >> .env
fi

set -a
# shellcheck disable=SC1091
source ./.env
set +a
case "${MODEL_VARIANT^^}" in
  8|8B) MODEL_VARIANT=8B ;;
  32|32B) MODEL_VARIANT=32B ;;
  *) echo "Invalid MODEL_VARIANT=${MODEL_VARIANT}; use 8B or 32B" >&2; exit 2 ;;
esac
export MODEL_VARIANT
mkdir -p "${HOST_CACHE_DIR:-./cache}" "${HOST_DATA_DIR:-./data}"

echo "Building ClinFusion ROCm image (selected model: ${MODEL_VARIANT})..."
docker compose build api setup

echo
echo "Checking ${MODEL_VARIANT} model files. Existing/partial Hugging Face downloads are reused."
docker compose --profile tools run --rm setup --variant "$MODEL_VARIANT" check --prompt || true

echo
echo "Starting API with ClinFusion ${MODEL_VARIANT}..."
docker compose up -d api

echo
printf 'Model:    ClinFusion %s\n' "$MODEL_VARIANT"
printf 'API:      http://localhost:%s\n' "${API_PORT:-8008}"
printf 'Docs:     http://localhost:%s/docs\n' "${API_PORT:-8008}"
printf 'Status:   http://localhost:%s/v1/model/status\n' "${API_PORT:-8008}"
echo "Logs:     ./logs.sh"
echo "GPU test: ./gpu-test.sh"
echo "Models:   ./models.sh [8B|32B]"
echo "Switch:   ./select-model.sh {8B|32B}"
