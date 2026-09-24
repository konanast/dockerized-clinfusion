#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
docker compose build api >/dev/null
docker compose run --rm api python /app/scripts/smoke_gpu.py
