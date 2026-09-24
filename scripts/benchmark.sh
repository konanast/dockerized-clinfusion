#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f .env ]] && set -a && source ./.env && set +a
PORT="${API_PORT:-8008}"
BASE="http://localhost:${PORT}"
AUTH=()
[[ -n "${API_KEY:-}" ]] && AUTH=(-H "X-API-Key: ${API_KEY}")
MODE="${1:-text}"

pretty() { if command -v jq >/dev/null; then jq .; else cat; fi; }

case "$MODE" in
  text)
    curl -sS "${AUTH[@]}" -H 'Content-Type: application/json' \
      -d '{"prompt":"In one short sentence, what is pneumonia?","max_new_tokens":64}' \
      "$BASE/v1/infer" | pretty
    ;;
  image)
    FILE="${2:?Usage: $0 image /path/to/image.jpg}"
    curl -sS "${AUTH[@]}" \
      -F 'prompt=Describe this medical image briefly.' \
      -F 'max_new_tokens=64' \
      -F "images=@${FILE}" \
      "$BASE/v1/infer/files" | pretty
    ;;
  nifti)
    FILE="${2:?Usage: $0 nifti /path/to/scan.nii.gz}"
    curl -sS "${AUTH[@]}" \
      -F 'prompt=Briefly describe the important findings in this volume.' \
      -F 'max_new_tokens=64' \
      -F "niftis=@${FILE}" \
      "$BASE/v1/infer/files" | pretty
    ;;
  *) echo "Usage: $0 {text|image FILE|nifti FILE}"; exit 2;;
esac
