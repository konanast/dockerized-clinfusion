#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

[[ -f .env ]] && set -a && source ./.env && set +a
VARIANT="${1:-${MODEL_VARIANT:-8B}}"
case "${VARIANT^^}" in
  8|8B) VARIANT=8B ;;
  32|32B) VARIANT=32B ;;
  *) echo "Usage: $0 [8B|32B]" >&2; exit 2 ;;
esac

docker compose build setup >/dev/null
docker compose --profile tools run --rm setup --variant "$VARIANT" check --prompt
