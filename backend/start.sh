#!/bin/bash
#cd /home/test_code/test01/rag-chatbot/backend
#source /home/test_code/test01/rag-chatbot/.venv/bin/activate
nohup uvicorn app.main:app \
       --reload \
       --host 0.0.0.0 \
       --port 8000 \
       --log-level debug \
       --access-log \
       >> ./chatbot.log 2>&1 &