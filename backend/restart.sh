#!/usr/bin/env bash
set -euo pipefail            # 예상치 못한 오류 즉시 종료

# 절대경로 사용
BASE_DIR="/home/test_code/test01/rag-chatbot"
BACKEND_DIR="$BASE_DIR/backend"
LOG_FILE="$BACKEND_DIR/chatbot.log"
PID_FILE="$BACKEND_DIR/server.pid"

echo "===== $(date) - 챗봇 서버 재시작 시작 ====="

# 1. 기존 서버 종료  (reload 모드라 PGID 전체 kill)
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    echo "기존 서버 PGID 종료 시도 (PID: $OLD_PID)…"
    # 부모·자식 전체 종료
    kill -TERM -"$OLD_PID" 2>/dev/null || true
    sleep 5
    if ps -p "$OLD_PID" &>/dev/null; then
        echo "정상 종료 실패, SIGKILL…"
        kill -KILL -"$OLD_PID" 2>/dev/null || true
        sleep 2
    fi
    rm -f "$PID_FILE"
else
    echo "PID 파일 없음 → 새로 시작"
fi

# 2. 가상환경 활성화
echo "가상 환경 활성화…"
source "$BASE_DIR/.venv/bin/activate"

# 3. 캐시 드롭 (root만 가능)
if [[ $EUID -eq 0 ]]; then
    echo "메모리 캐시 드롭…"
    sync && echo 3 > /proc/sys/vm/drop_caches
fi

# 4. 작업 디렉터리 이동
cd "$BACKEND_DIR"

# 5. 서버 시작 (parents + child 둘 다 같은 PGID로)
echo "새 서버 프로세스 시작…"
setsid -f bash -c '
  exec python -m uvicorn app.main:app \
    --reload \
    --host 0.0.0.0 --port 8000 \
    --log-level debug \
    --reload-dir app \
    --reload-exclude ../logs
' >> "$LOG_FILE" 2>&1 &
NEW_PID=$!           # = PGID 리더
echo "$NEW_PID" > "$PID_FILE"

echo "$NEW_PID" > "$PID_FILE"
echo "새 서버 시작됨 (PGID: $NEW_PID)"

echo "===== $(date) - 챗봇 서버 재시작 완료 ====="
sleep 2
tail -n 20 "$LOG_FILE"
