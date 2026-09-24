#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {8B|32B}" >&2
  exit 2
fi
case "${1^^}" in
  8|8B) VARIANT=8B ;;
  32|32B) VARIANT=32B ;;
  *) echo "Usage: $0 {8B|32B}" >&2; exit 2 ;;
esac

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

python - "$VARIANT" <<'PY'
from pathlib import Path
import sys
variant=sys.argv[1]
p=Path('.env')
lines=p.read_text().splitlines()
out=[]
seen=False
for line in lines:
    if line.startswith('MODEL_VARIANT='):
        out.append(f'MODEL_VARIANT={variant}')
        seen=True
    else:
        out.append(line)
if not seen:
    out.append(f'MODEL_VARIANT={variant}')
p.write_text('\n'.join(out)+'\n')
PY

echo "Selected ClinFusion ${VARIANT} for the next container start."
if [[ "$VARIANT" == "32B" ]]; then
  cat <<'EOF'

32B note for 128-GB Strix Halo:
  - Keep only one ClinFusion model loaded at a time.
  - Aim for roughly 100 GiB GPU-visible TTM/GTT allocation before loading 32B.
  - Check/download without changing selection: ./models.sh 32B
  - 8B remains the recommended everyday/default model.
EOF
fi
cat <<EOF

No running container was changed by this command.
Run ./start.sh to apply ${VARIANT}, or ./select-model.sh 8B to switch back.
EOF
