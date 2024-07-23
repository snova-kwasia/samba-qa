#!/bin/bash

if [ "$RUN_SLACKAPP" = "true" ]; then
    echo "Starting Slack App"
    python backend/slackapp/app.py
else
    echo "Starting Backend Server"
    prisma db push --schema ./backend/database/schema.prisma
    uvicorn --host 0.0.0.0 --port 8000 backend.server.app:app --reload
fi