#!/usr/bin/env bash
# 수업용 기동 스크립트.
#
# 두 프로세스를 띄웁니다. 서버(uvicorn)와 공개 터널(cloudflared)입니다. 둘 다
# setsid로 SSH 세션에서 분리해 띄우는 것이 이 스크립트의 핵심입니다 -- 평범한
# 셸에서 cloudflared를 실행하면 노트북이 절전에 들거나 와이파이가 끊기는 순간
# 터널이 죽고, 학생들이 보던 주소가 수업 도중에 사라집니다.
#
#   scripts/class.sh start    서버 + 터널 기동, 공유할 주소 출력
#   scripts/class.sh url      현재 주소 다시 보기 (슬라이드에 붙일 때)
#   scripts/class.sh status   무엇이 돌고 있는지
#   scripts/class.sh backup   비공개 SQLite 복구 백업
#   scripts/class.sh stop     접수/대기 확인 + 복구 백업 후 정지
#
# 수업 시작 5분 전에 실행하세요. 첫 기동은 모델을 올리고 UMAP JIT를 예열합니다.
set -uo pipefail
cd "$(dirname "$0")/.."

PORT="${ATLAS_PORT:-8100}"
RUN=".run"
mkdir -p "$RUN"
SRV_LOG="$RUN/server.log"; CF_LOG="$RUN/tunnel.log"; URL_FILE="$RUN/url.txt"

SEMANTIC_OVERRIDE_FROM_ENV="${ATLAS_ALLOW_UNREVIEWED_SEMANTICS:-}"
[ -f .env ] && set -a && . ./.env && set +a
if [ -n "$SEMANTIC_OVERRIDE_FROM_ENV" ]; then
  export ATLAS_ALLOW_UNREVIEWED_SEMANTICS="$SEMANTIC_OVERRIDE_FROM_ENV"
fi
unset SEMANTIC_OVERRIDE_FROM_ENV

srv_pid() { pgrep -f "uvicorn --factory src.server:create_app .*--port $PORT" | head -1; }
cf_pid()  { pgrep -f "cloudflared tunnel --url http://localhost:$PORT" | head -1; }
health_ok() { curl -sf --max-time 2 "http://127.0.0.1:$PORT/healthz" >/dev/null; }

start() {
  [ -x .venv/bin/python ] || { echo "먼저 venv를 만드세요: uv venv --python 3.12 --seed .venv"; exit 1; }
  [ -n "${ATLAS_ADMIN_CODE:-}" ] || { echo "ATLAS_ADMIN_CODE가 없습니다 (.env에 넣으세요)"; exit 1; }
  [ -f "${ATLAS_ROSTER:-roster.csv}" ] || { echo "명단 파일이 없습니다: ${ATLAS_ROSTER:-roster.csv}"; exit 1; }

  if [ -z "$(srv_pid)" ]; then
    echo "서버 기동 중… (첫 실행은 모델 적재와 JIT 예열로 1분쯤 걸립니다)"
    # HF_TOKEN은 의도적으로 제거합니다. 잘못된 값이 저장된 로그인을 가리고,
    # 허브는 그것을 gated 오류로 보고합니다. README의 설정 항목을 보세요.
    setsid env -u HF_TOKEN .venv/bin/python -m uvicorn --factory src.server:create_app \
      --host 127.0.0.1 --port "$PORT" --proxy-headers >"$SRV_LOG" 2>&1 </dev/null &
    for _ in $(seq 1 60); do
      health_ok && break
      sleep 2
      [ -z "$(srv_pid)" ] && { echo "서버 기동 실패:"; tail -20 "$SRV_LOG"; exit 1; }
    done
    health_ok || { echo "서버가 /healthz에 응답하지 않아 터널을 열지 않습니다:"; tail -20 "$SRV_LOG"; exit 1; }
  else
    echo "서버는 이미 돌고 있습니다 (pid $(srv_pid))"
    health_ok || { echo "이미 실행 중인 서버가 /healthz에 응답하지 않아 터널을 열지 않습니다."; exit 1; }
  fi

  local review_args=()
  [ "${ATLAS_ALLOW_UNREVIEWED_SEMANTICS:-0}" = "1" ] && review_args+=(--allow-unreviewed)
  curl -sf --max-time 2 "http://127.0.0.1:$PORT/healthz" |     .venv/bin/python scripts/class_readiness.py "${review_args[@]}" || exit 1

  if [ -z "$(cf_pid)" ]; then
    echo "터널 여는 중…"
    : > "$CF_LOG"
    setsid cloudflared tunnel --url "http://localhost:$PORT" >"$CF_LOG" 2>&1 </dev/null &
    for _ in $(seq 1 40); do
      grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$CF_LOG" | head -1 >"$URL_FILE"
      [ -s "$URL_FILE" ] && break
      sleep 2
    done
    [ -s "$URL_FILE" ] || { echo "터널이 주소를 주지 않았습니다:"; tail -15 "$CF_LOG"; exit 1; }
  else
    echo "터널은 이미 열려 있습니다 (pid $(cf_pid))"
  fi
  url
}

url() {
  [ -s "$URL_FILE" ] || { echo "열린 터널이 없습니다. scripts/class.sh start"; exit 1; }
  local u; u=$(cat "$URL_FILE")
  # 주소를 눈으로 읽어 옮겨 적게 하지 않습니다. 무작위 네 단어짜리 HTTPS 주소는
  # 안전하지만 강의실 뒤에서 읽을 물건이 아니고, 서른 명이 오타를 내는 것이 실제
  # 실패 지점입니다. QR 하나면 사라집니다.
  .venv/bin/python scripts/make_qr.py "$u" "$RUN/qr.png"
  echo
  echo "  학생 공유 주소 :  $u"
  echo "  관리자          :  $u/#admin"
  echo "  슬라이드용 QR  :  $(pwd)/$RUN/qr.png"
  echo
  echo "  임시 주소가 바뀌면 이전 주소의 브라우저 초안을 자동 복구할 수 없습니다. 수업에는 고정 주소를 권장합니다."
}

status() {
  local s c
  s=$(srv_pid); c=$(cf_pid)
  echo "  서버   : ${s:-정지}"
  if [ -n "$s" ]; then
    curl -sf --max-time 2 "http://127.0.0.1:$PORT/healthz" | \
      .venv/bin/python scripts/class_readiness.py --status-only || echo "서버 상태를 확인하지 못했습니다."
  fi
  echo "  터널   : ${c:-정지} $( [ -s "$URL_FILE" ] && cat "$URL_FILE")"
}

backup() {
  local destination="${1:-$RUN/backups/atlas-$(date -u +%Y%m%dT%H%M%SZ)-$$.db}"
  .venv/bin/python scripts/backup_database.py --db "${ATLAS_DB:-atlas.db}" --out "$destination"
}

stop() {
  if [ -n "$(srv_pid)" ]; then
    if [ "${1:-}" != "--force" ]; then
      curl -sf --max-time 2 "http://127.0.0.1:$PORT/healthz" | \
        .venv/bin/python scripts/class_readiness.py --check-stop || return 1
    else
      echo "강제 종료: 처리 중인 작업은 다음 시작 때 복구합니다. 비공개 백업을 먼저 만듭니다."
    fi
    backup || { echo "복구 백업 실패: 서버를 종료하지 않습니다."; return 1; }
  fi
  for p in $(cf_pid) $(srv_pid); do kill "$p" 2>/dev/null && echo "  pid $p 정지"; done
  rm -f "$URL_FILE"
}

case "${1:-start}" in
  start) start ;; url) url ;; status) status ;; backup) backup "${2:-}" ;; stop) stop "${2:-}" ;;
  *) echo "사용법: scripts/class.sh {start|url|status|backup [파일]|stop [--force]}"; exit 2 ;;
esac
