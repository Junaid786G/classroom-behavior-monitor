#!/bin/bash
cd ~/classroom_monitor
source venv/bin/activate
streamlit run frontend/app.py --server.port 8501
