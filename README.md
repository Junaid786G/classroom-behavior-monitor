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
- **Eye-state detection**: EAR (Eye Aspect Ratio, default) or a pretrained ONNX CNN (open-closed-eye-0001), switchable
- **GPU**: CUDA-accelerated inference (tested on RTX A4000)

## Architecture

Video Input (recorded file or live stream)
|
Face Detection (RetinaFace)
|
Face Recognition (ArcFace + FAISS gallery)
|
Multi-object Tracking (ByteTrack)
|
Behavior Analysis (MediaPipe pose + eye-state gate)
|
Attendance + Behavior Records -> PostgreSQL
|
Streamlit Dashboards


## Setup

Requirements: Python 3.12, PostgreSQL (via Docker), NVIDIA GPU + CUDA recommended, ffmpeg.

    git clone https://github.com/Junaid786G/classroom-behavior-monitor.git
    cd classroom-behavior-monitor

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

    cp .env.example .env
    # edit .env: database credentials, thresholds, EYE_STATE_METHOD, etc.

    docker compose up -d db
    python scripts/01_setup_db.py
    python scripts/02_register_students.py

    ./start_backend.sh
    ./start_frontend.sh

Frontend runs at http://localhost:8501, backend API at http://localhost:8000.

## Testing

    pytest backend/tests/ -v

34 tests covering behavior classification, eye-state gate selection, the pose baseline, and video-timestamp handling.

## Development history

- Initial build: FastAPI + Streamlit scaffold, InsightFace/ArcFace recognition, ByteTrack, MediaPipe-based behavior detection using EAR for eye state
- Eye-state CNN: EAR was found unreliable for narrow-eyed/glasses-wearing students at small face crop sizes; a pretrained ONNX eye-state CNN was added as an alternative, made switchable via EYE_STATE_METHOD
- 2026-08-19, major validation and bugfix session:
    - Fixed dwell timers to use video-relative time instead of wall-clock time (affected recorded-video processing accuracy)
    - Replaced the single global head-yaw/pitch threshold with a per-student adaptive baseline, so side-seated students aren't unfairly flagged as distracted
    - Found and fixed a 25x clock error specific to the live WebSocket monitoring path that made DISTRACTED/SLEEPING effectively unreachable in live mode
    - Ran frame-level validation of the CNN eye-state gate against ground truth: measured a 58 percent false-positive rate on 30-70px faces. Switched the default back to EAR, backed by a measured 0 percent false-positive rate on the same footage
    - Full results and reasoning are documented in code comments (.env, backend/config.py) and backend/tests/test_eye_gate_selection.py

## Known limitations

- Ground truth was labeled by a single annotator (the author). A full inter-rater reliability study was outside project scope, and a planned test-retest self-consistency check was not completed due to project timeline constraints. Every accuracy figure in this project rests on that single annotator's judgement, and no measurement of how repeatable that judgement is has been made. One further disclosure: the aggregate predicted-class distribution for the 50-window pilot subset (counts only, no per-clip answers) was seen before those 50 were labeled. Because the pilot was subsequently folded into the full 387-window sample rather than discarded, that constitutes a minor information leak into the headline accuracy figure
- Short eye-closures may be undercounted in live-monitor mode: the client samples at roughly 1fps, and the sleeping dwell timer requires about 2 seconds of sustained closure to fire, so closures shorter than that, or split across the sampling gap, may not register. Documented and tested in backend/tests/test_eye_gate_selection.py. Fixing it requires increasing the live client's sampling rate, which is a deliberate, deferred scope decision
- Concurrent sessions silently lose data to connection-pool exhaustion. `database_pool_size=10` + `database_max_overflow=20` gives 30 connections, and the live WebSocket path fires one background `_persist_frame_results` task per frame, each taking a connection for the duration of its writes. Two sessions streaming at once at ~1fps over ~14 faces per frame drain that pool: measured on 2026-09-03, two overlapping 8-minute sessions produced 115 `QueuePool limit of size 10 overflow 20 reached, connection timed out` failures in four minutes, and one of those sessions persisted only 418 of the 460 frames it streamed. The failure is invisible to everyone who matters: it is logged at WARNING inside `_persist_frame_results`, the affected frames are dropped rather than retried, the WebSocket client is never told, and the session still finalises as COMPLETED. Attendance and behaviour rows for BOTH sessions are silently incomplete. Not triggered by single-session use, which is how the system is demonstrated and how it has been measured throughout, so every accuracy figure in this project stands; but it is a correctness bug, not a capacity limit, and it must be fixed before any deployment where two instructors could record at the same time. Fixing it properly means bounding in-flight persistence per session (a semaphore) and failing the session loudly rather than dropping frames — raising the pool size alone only moves the threshold. **Higher priority than the re-enrolment item below.**
- Two students are enrolled with embeddings too weak for reliable recognition. Measured over 60 sampled frames of the 8-minute clip: uzair (CS-015) peaks at 0.433 cosine similarity against a class where seven students exceed 0.60, with the roster's weakest top-1-vs-top-2 margin (median 0.143), and is top-1 for more faces than there are frames — so at the 0.35 threshold he recovers correct attendance but may absorb a few frames belonging to a neighbour, misattributing their behaviour events. samaira (CS-012) peaks at 0.246 and is never the top-1 candidate for any face in the clip, so no threshold recovers her; she is recorded ABSENT from footage she may well appear in. Re-enrolling both from cleaner, more frontal footage is the fix; lowering the threshold further is not, and would start trading false positives for it
- Single classroom/course in the current data model. Multi-course support, multiple subjects, instructor and HOD roles, per-course rosters, is designed but not yet implemented
- Privacy tier incomplete: raw video auto-deletion, student photo removal, and gallery encryption are planned but not yet built
- Gaze direction and posture detection, mentioned in the original project brief, were evaluated for feasibility but not built, pending a resolution check at this camera's face-crop sizes

## Project structure

    backend/
      pipeline/       detection, recognition, tracking, behavior analysis
      routers/        FastAPI endpoints: stream, attendance, students, analytics
      gallery/        FAISS face-embedding gallery management
      tests/          pytest suite
    frontend/
      pages/          Streamlit pages: live monitor, attendance, dashboards, admin
    scripts/           DB setup, student enrollment, diagnostics
