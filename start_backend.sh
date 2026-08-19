#!/bin/bash
cd ~/classroom_monitor
source venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
