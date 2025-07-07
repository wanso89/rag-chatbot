#!/bin/bash

# Kill processes running on ports 8000 and 5173
fuser -k 8000/tcp
fuser -k 5173/tcp

# Start backend service
cd /home/test_code/test01/rag-chatbot/backend
nohup ./start.sh >>chatbot.log 2>&1 &
# Navigate to frontend directory and start frontend service
cd /home/test_code/test01/rag-chatbot/frontend
nohup npm run dev >>frontend.log 2>&1 &

echo "tail -f backend/chatbot.log  로 백엔드 로그 확인"
