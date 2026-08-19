# Classroom Behavior & Attendance Monitor

AI-powered classroom attendance and behavior analysis system built for CAE Avionics coursework. Uses CCTV/recorded video to automatically mark attendance via face recognition and classify student engagement (ATTENTIVE / DISTRACTED / SLEEPING) from head pose and eye state.

## What it does

- **Attendance**: detects and recognizes enrolled students from classroom video, marks PRESENT/LATE/ABSENT per session
- **Behavior analysis**: tracks each student's head pose and eye state to classify engagement, with dwell-timer smoothing so brief blinks or head turns don't trigger false labels
- **Dashboards**: per-student and per-classroom views of attendance history and behavior trends
- **Admin panel**: tunable detection/behavior thresholds, student enrollment, gallery management

## Tech stack

- **Backend**: FastAPI, PostgreSQL
- **Frontend**: Streamlit
- **Face detection/recognition**: InsightFace (RetinaFace + ArcFace)
- **Tracking**: ByteTrack
- **Pose/landmarks**: MediaPipe FaceLandmarker
- **Eye-state detection**: EAR (Eye Aspect Ratio, default) or a pretrained ONNX CNN (`open-closed-eye-0001`), switchable
- **GPU**: CUDA-accelerated inference (tested on RTX A4000)

## Architecture

## Setup

**Requirements**: Python 3.12, PostgreSQL (via Docker), NVIDIA GPU + CUDA recommended, `ffmpeg`.

```bash
# 1. Clone and enter the project
git clone https://github.com/Junaid786G/classroom-behavior-monitor.git
cd classroom-behavior-monitor

# 2. Create venv and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# edit .env: database credentials, thresholds, EYE_STATE_METHOD, etc.

# 4. Start PostgreSQL (see docker-compose.yml)
docker compose up -d db

# 5. Set up the database schema
python scripts/01_setup_db.py

# 6. Register students (enrolls faces into the recognition gallery)
python scripts/02_register_students.py

# 7. Start the backend
./start_backend.sh

# 8. Start the frontend (separate terminal)
./start_frontend.sh
```

Frontend runs at `http://localhost:8501`, backend API at `http://localhost:8000`.

## Testing

```bash
pytest backend/tests/ -v
```

34 tests covering behavior classification, eye-state gate selection, the pose baseline, and video-timestamp handling.

## Development history

- **Initial build**: FastAPI + Streamlit scaffold, InsightFace/ArcFace recognition, ByteTrack, MediaPipe-based behavior detection using EAR for eye state
- **Eye-state CNN**: EAR was found unreliable for narrow-eyed/glasses-wearing students at small face crop sizes; a pretrained ONNX eye-state CNN was added as an alternative, made switchable via `EYE_STATE_METHOD`
- **2026-08-19 — major validation & bugfix session**:
  - Fixed dwell timers to use video-relative time instead of wall-clock time (affected recorded-video processing accuracy)
  - Replaced the single global head-yaw/pitch threshold with a per-student adaptive baseline, so side-seated students aren't unfairly flagged as distracted
  - Found and fixed a 25x clock error specific to the live WebSocket monitoring path that made DISTRACTED/SLEEPING effectively unreachable in live mode
  - Ran frame-level validation of the CNN eye-state gate against ground truth: measured a 58% false-positive rate on 30-70px faces. Switched the default back to EAR, backed by a measured 0% false-positive rate on the same footage
  - Full results and reasoning are documented in code comments (`.env`, `backend/config.py`) and `backend/tests/test_eye_gate_selection.py`

## Known limitations

- **Short eye-closures may be undercounted** in live-monitor mode: the client samples at ~1fps, and the sleeping dwell timer requires ~2 seconds of sustained closure to fire, so closures shorter than that (or split across the sampling gap) may not register. Documented and tested in `backend/tests/test_eye_gate_selection.py`; fixing it requires increasing the live client's sampling rate, which is a deliberate, deferred scope decision
- **Single classroom/course** in the current data model; multi-course support (multiple subjects, instructor/HOD roles, per-course rosters) is designed but not yet implemented
- **Privacy tier incomplete**: raw video auto-deletion, student photo removal, and gallery encryption are planned but not yet built
- Gaze direction and posture detection (mentioned in the original project brief) were evaluated for feasibility but not built, pending a resolution check at this camera's face-crop sizes

## Project structure
