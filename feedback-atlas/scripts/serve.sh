#!/usr/bin/env bash
# Foreground TLS service; systemd owns persistence and restarts.
set -euo pipefail
cd "$(dirname "$0")/.."
umask 077
set -a
. ./.env
set +a
cert="${ATLAS_TLS_CERT:-.run/tls/live/feedback-atlas-ip/fullchain.pem}"
key="${ATLAS_TLS_KEY:-.run/tls/live/feedback-atlas-ip/privkey.pem}"
test -r "$cert" && test -r "$key" || {
  echo 'A readable trusted TLS certificate and private key are required.' >&2
  exit 1
}
export ATLAS_ALLOW_INSECURE_HTTP=0
exec env -u HF_TOKEN HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m uvicorn --factory src.server:create_app \
  --host 0.0.0.0 --port "${ATLAS_PORT:-8888}" --no-proxy-headers \
  --ssl-certfile "$cert" --ssl-keyfile "$key" \
  --ws-max-size 8192 --ws-max-queue 8 --timeout-keep-alive 5 \
  --limit-concurrency 160 --backlog 128
