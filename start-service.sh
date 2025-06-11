#!/bin/bash

# RAG Chatbot 서비스 시작 스크립트
# 백엔드 서비스가 완전히 준비된 후에 프론트엔드를 시작합니다.

echo "===== RAG Chatbot 서비스 시작 중... ====="
echo "$(date '+%Y-%m-%d %H:%M:%S')"

# 작업 디렉토리 설정
cd /home/test_code/test01/rag-chatbot
source .venv/bin/activate

# 로그 디렉토리 생성
mkdir -p logs

# Redis 데이터 디렉토리 확인 및 생성
REDIS_DATA_DIR="/var/lib/redis"
if [ ! -d "$REDIS_DATA_DIR" ]; then
  echo "Redis 데이터 디렉토리 생성 중..."
  sudo mkdir -p $REDIS_DATA_DIR
  sudo chown -R redis:redis $REDIS_DATA_DIR
fi

# Redis 서버 상태 확인 
echo "Redis 서버 상태 확인 중..."
if redis-cli ping | grep -q "PONG"; then
  echo "Redis 서버가 이미 실행 중입니다."
else
  echo "Redis 서버 시작 중..."
  redis-server --daemonize yes
fi

# Redis 서버 시작 확인
echo "Redis 서버 시작 확인 중..."
REDIS_RETRY=0
REDIS_MAX_RETRY=30
REDIS_READY=false

while [ $REDIS_RETRY -lt $REDIS_MAX_RETRY ]; do
  if redis-cli ping | grep -q "PONG"; then
    echo "Redis 서버 시작 완료!"
    REDIS_READY=true
    break
  fi
  echo "Redis 서버 대기 중... (${REDIS_RETRY}/${REDIS_MAX_RETRY})"
  REDIS_RETRY=$((REDIS_RETRY+1))
  sleep 1
done

if [ "$REDIS_READY" = false ]; then
  echo "Redis 서버 시작 실패! 계속 진행합니다. (백엔드에서 파일 기반 저장소로 폴백됩니다)"
else
  # Redis 데이터베이스 초기화 (기존 데이터 유지)
  echo "Redis 데이터베이스 상태:"
  redis-cli info keyspace
fi

# 백엔드 서버 시작 (백그라운드로 실행)
echo "백엔드 서버 시작 중..."
cd backend
nohup python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > ../backend.pid
cd ..

# 백엔드 서버 시작 확인
echo "백엔드 서버 시작 확인 중..."
BACKEND_RETRY=0
BACKEND_MAX_RETRY=60
BACKEND_READY=false

while [ $BACKEND_RETRY -lt $BACKEND_MAX_RETRY ]; do
  if curl -s http://localhost:8000/api/health-check | grep -q "status"; then
    echo "백엔드 서버 시작 완료!"
    BACKEND_READY=true
    break
  fi
  echo "백엔드 서버 대기 중... (${BACKEND_RETRY}/${BACKEND_MAX_RETRY})"
  BACKEND_RETRY=$((BACKEND_RETRY+1))
  sleep 1
done

if [ "$BACKEND_READY" = false ]; then
  echo "백엔드 서버 시작 실패! 다시 확인해 주세요."
  exit 1
fi

# Redis와 백엔드 연결 확인
echo "Redis와 백엔드 연결 확인 중..."
if curl -s http://localhost:8000/api/debug/redis-status | grep -q "connected"; then
  echo "Redis와 백엔드 연결 성공!"
else
  echo "경고: Redis와 백엔드 연결 확인 실패. 파일 기반 저장소로 폴백됩니다."
fi

# 프론트엔드 서버 시작 (백그라운드로 실행)
echo "프론트엔드 서버 시작 중..."
cd frontend
nohup npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > ../frontend.pid
cd ..

echo "===== 모든 서비스 시작 완료! ====="
echo "백엔드 서버: http://localhost:8000"
echo "프론트엔드 서버: http://localhost:5173"
echo "백엔드 PID: $BACKEND_PID"
echo "프론트엔드 PID: $FRONTEND_PID"
echo "로그 파일: logs/backend.log, logs/frontend.log" 
