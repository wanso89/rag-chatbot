#!/bin/bash

# RAG Chatbot 서비스 중지 스크립트
# 실행 중인 모든 서비스를 안전하게 중지합니다.

echo "===== RAG Chatbot 서비스 중지 중... ====="
echo "$(date '+%Y-%m-%d %H:%M:%S')"

# 작업 디렉토리 설정
WORKSPACE_DIR="/home/test_code/test01/rag-chatbot"
cd $WORKSPACE_DIR

# 저장된 PID로 프로세스 종료
SERVICES_STOPPED=0

# 프론트엔드 중지
if [ -f "$WORKSPACE_DIR/frontend.pid" ]; then
  FRONTEND_PID=$(cat $WORKSPACE_DIR/frontend.pid)
  if ps -p $FRONTEND_PID > /dev/null; then
    echo "프론트엔드 서버 중지 중... (PID: $FRONTEND_PID)"
    kill -15 $FRONTEND_PID
    sleep 2
    # 강제 종료 확인
    if ps -p $FRONTEND_PID > /dev/null; then
      echo "프론트엔드 서버가 정상 종료되지 않아 강제 종료합니다."
      kill -9 $FRONTEND_PID
    fi
    echo "프론트엔드 서버 중지 완료"
    SERVICES_STOPPED=$((SERVICES_STOPPED+1))
  else
    echo "프론트엔드 서버가 이미 중지되었습니다."
  fi
  rm -f $WORKSPACE_DIR/frontend.pid
else
  echo "프론트엔드 PID 파일이 존재하지 않습니다."
  # PID 파일이 없어도 프론트엔드 프로세스 검색 및 종료 시도
  FRONTEND_PROCESSES=$(ps aux | grep "vite --host" | grep -v grep | awk '{print $2}')
  if [ -n "$FRONTEND_PROCESSES" ]; then
    echo "프론트엔드 프로세스 발견, 종료 중... (PID: $FRONTEND_PROCESSES)"
    for PID in $FRONTEND_PROCESSES; do
      kill -15 $PID 2>/dev/null || kill -9 $PID 2>/dev/null
    done
    echo "프론트엔드 프로세스 종료 완료"
    SERVICES_STOPPED=$((SERVICES_STOPPED+1))
  fi
fi

# 백엔드 중지
if [ -f "$WORKSPACE_DIR/backend.pid" ]; then
  BACKEND_PID=$(cat $WORKSPACE_DIR/backend.pid)
  if ps -p $BACKEND_PID > /dev/null; then
    echo "백엔드 서버 중지 중... (PID: $BACKEND_PID)"
    kill -15 $BACKEND_PID
    sleep 3
    # 강제 종료 확인
    if ps -p $BACKEND_PID > /dev/null; then
      echo "백엔드 서버가 정상 종료되지 않아 강제 종료합니다."
      kill -9 $BACKEND_PID
    fi
    echo "백엔드 서버 중지 완료"
    SERVICES_STOPPED=$((SERVICES_STOPPED+1))
  else
    echo "백엔드 서버가 이미 중지되었습니다."
  fi
  rm -f $WORKSPACE_DIR/backend.pid
else
  echo "백엔드 PID 파일이 존재하지 않습니다."
  # PID 파일이 없어도 백엔드 프로세스 검색 및 종료 시도
  BACKEND_PROCESSES=$(ps aux | grep "uvicorn app.main:app" | grep -v grep | awk '{print $2}')
  if [ -n "$BACKEND_PROCESSES" ]; then
    echo "백엔드 프로세스 발견, 종료 중... (PID: $BACKEND_PROCESSES)"
    for PID in $BACKEND_PROCESSES; do
      kill -15 $PID 2>/dev/null || kill -9 $PID 2>/dev/null
    done
    echo "백엔드 프로세스 종료 완료"
    SERVICES_STOPPED=$((SERVICES_STOPPED+1))
  fi
fi

# Redis 서버 중지
if redis-cli ping > /dev/null 2>&1; then
  echo "Redis 서버 중지 중..."
  redis-cli shutdown
  sleep 1
  
  # Redis 종료 확인
  if ! redis-cli ping > /dev/null 2>&1; then
    echo "Redis 서버 중지 완료"
    SERVICES_STOPPED=$((SERVICES_STOPPED+1))
  else
    echo "Redis 서버 중지 실패! 강제 종료를 시도합니다."
    REDIS_PID=$(pgrep -f "redis-server")
    if [ -n "$REDIS_PID" ]; then
      kill -9 $REDIS_PID
      echo "Redis 서버 강제 종료 완료"
      SERVICES_STOPPED=$((SERVICES_STOPPED+1))
    else
      echo "Redis 프로세스를 찾을 수 없습니다."
    fi
  fi
else
  echo "Redis 서버가 이미 중지되었습니다."
fi

# 서비스 상태 요약
echo ""
echo "===== 서비스 상태 요약 ====="
echo "중지된 서비스 수: $SERVICES_STOPPED"

# 추가 정리 작업
echo "임시 파일 정리 중..."
rm -f $WORKSPACE_DIR/backend.pid $WORKSPACE_DIR/frontend.pid

echo ""
echo "모든 서비스가 중지되었습니다."
echo "서비스 시작 방법: ./start-service.sh"
echo "$(date '+%Y-%m-%d %H:%M:%S')"
echo "===== 서비스 중지 완료 =====" 
