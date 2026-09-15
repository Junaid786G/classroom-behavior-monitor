#!/bin/bash
cd ~/classroom_monitor
source venv/bin/activate
# --ws-ping-timeout far above uvicorn's 20s default: the Live Monitor upload
# path is browser-pumped, and websocket-client only pongs from inside recv(),
# which stops while the instructor is on another page. See docker-compose.yml.
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload \
        --ws-ping-interval 20 --ws-ping-timeout 3600
