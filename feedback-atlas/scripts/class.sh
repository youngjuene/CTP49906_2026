#!/usr/bin/env bash
# Classroom controls use the managed HTTPS service; no public tunnel is created.
set -euo pipefail
cd "$(dirname "$0")/.."
umask 077
set -a
. ./.env
set +a
URL="https://${ATLAS_PUBLIC_IP:-143.248.249.7}:${ATLAS_PORT:-8888}"

health_ok() { curl --noproxy '*' -sf --max-time 3 "$URL/healthz" >/dev/null; }

url() {
  health_ok || { echo 'HTTPS server is not ready. Run scripts/class.sh start.' >&2; return 1; }
  .venv/bin/python scripts/make_qr.py "$URL" .run/qr.png
  echo "학생 공유 주소: $URL"
  echo "관리자: $URL/admin"
  echo "슬라이드용 QR: $PWD/.run/qr.png"
  echo '각 참가자에게 본인의 고정 개인 접근 코드만 따로 전달하세요.'
}

start() {
  systemctl --user start feedback-atlas.service feedback-atlas-renew.timer
  for _ in $(seq 1 60); do
    if health_ok; then url; return; fi
    sleep 2
  done
  echo 'HTTPS startup failed; inspect journalctl --user -u feedback-atlas.service.' >&2
  return 1
}

case "${1:-start}" in
  start) start ;;
  url) url ;;
  status)
    systemctl --user --no-pager status feedback-atlas.service
    curl --noproxy '*' -fsS --max-time 3 "$URL/healthz"
    ;;
  stop) systemctl --user stop feedback-atlas.service ;;
  *) echo 'Usage: scripts/class.sh {start|url|status|stop}' >&2; exit 2 ;;
esac
