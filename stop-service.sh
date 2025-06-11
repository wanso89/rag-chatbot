#!/bin/bash

# RAG Chatbot 서비스 중지 스크립트
# 실행 중인 모든 서비스를 안전하게 중지합니다.

echo "===== RAG Chatbot 서비스 중지 중... ====="
echo "$(date '+%Y-%m-%d %H:%M:%S')"

# 작업 디렉토리 설정
cd /home/test_code/test01/rag-chatbot

# 저장된 PID로 프로세스 종료
SERVICES_STOPPED=0

# 프론트엔드 중지
if [ -f "./frontend.pid" ]; then
  FRONTEND_PID=$(cat ./frontend.pid)
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
  rm -f ./frontend.pid
else
  echo "프론트엔드 PID 파일이 존재하지 않습니다."
fi

# 백엔드 중지
if [ -f "./backend.pid" ]; then
  BACKEND_PID=$(cat ./backend.pid)
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
  rm -f ./backend.pid
else
  echo "백엔드 PID 파일이 존재하지 않습니다."
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
    echo "Redis 서버 중지 실패!"
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
rm -f ./backend.pid ./frontend.pid

echo ""
echo "모든 서비스가 중지되었습니다."
echo "서비스 시작 방법: ./start-services.sh"
echo "$(date '+%Y-%m-%d %H:%M:%S')"
echo "===== 서비스 중지 완료 =====" 
