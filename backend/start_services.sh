#!/bin/bash

# Kill processes running on ports 8000 and 5173
fuser -k 8000/tcp
fuser -k 5173/tcp

# Start backend service
./start.sh &
cd backend 
nohup ./start.sh >>chatbot.log 2>&1 &
# Navigate to frontend directory and start frontend service
cd ../frontend
npm run dev &

echo "tail -f backend/chatbot.log  로 백엔드 로그 확인"