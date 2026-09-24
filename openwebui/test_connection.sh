#!/usr/bin/env bash
set -euo pipefail
BASE="${1:-http://localhost:8008}"
KEY="${2:-}"
AUTH=()
[[ -n "$KEY" ]] && AUTH=(-H "Authorization: Bearer $KEY")

echo "== health =="
curl -fsS "$BASE/healthz"; echo

echo "== models =="
curl -fsS "${AUTH[@]}" "$BASE/v1/models"; echo

echo "== chat =="
curl -fsS "${AUTH[@]}" -H 'Content-Type: application/json' \
  "$BASE/v1/chat/completions" \
  -d '{"model":"clinfusion","messages":[{"role":"user","content":"Reply with exactly: ClinFusion connection OK"}],"max_tokens":32}'
echo
