# Deployment Guide — Classroom CCTV Monitor

Two supported targets:

- **[Path A — Linux / Ubuntu](#path-a--linux--ubuntu)** (the current dev machine)
- **[Path B — Windows + Docker Desktop + WSL2 + NVIDIA](#path-b--windows--docker-desktop--wsl2--nvidia)** (the Radar Lab PC)

Both paths share [§2 Prerequisites](#2-prerequisites) and [§3 Model files](#3-model-files--do-this-before-the-first-build).

---

## 1. What gets deployed

| Service | Image | Port | Purpose |
|---|---|---|---|
| `postgres` | `pgvector/pgvector:pg16` | 5432 | PostgreSQL 16 + pgvector |
| `redis` | `redis:7-alpine` | 6379 | Celery broker + result backend |
| `backend` | `cm_backend:latest` / `cm_backend:gpu` | 8000 | FastAPI + Uvicorn |
| `celery_worker` | same image as backend | — | Async video processing — **not started by default**, see below |
| `flower` | same image as backend | 5555 | Celery UI (`monitoring` profile) |
| `frontend` | `cm_frontend:latest` | 8501 | Streamlit dashboard |

Compose files:

| File | Role |
|---|---|
| `docker-compose.yml` | Base stack, CPU inference |
| `docker-compose.gpu.yml` | **Overlay** — swaps backend/celery to the CUDA image and reserves the GPU |

The overlay never replaces the base; always pass both `-f` flags in that order.

> **Celery is not wired up yet.** `celery_worker` and `flower` run
> `celery -A backend.tasks`, but `backend/tasks.py` does not exist and nothing in
> the codebase imports Celery. Started, the worker crash-loops with
> `The module backend.tasks was not found`. Both services therefore sit behind
> opt-in profiles (`workers` and `monitoring`) so `docker compose up -d` brings up
> a clean stack. They are left in the compose file as the intended shape for when
> async processing is implemented. A default `up` starts exactly four services:
> `postgres`, `redis`, `backend`, `frontend`.

---

## 2. Prerequisites

### Common

| Requirement | Minimum | Notes |
|---|---|---|
| Docker Engine / Desktop | 24.0 / 4.30 | |
| **Docker Compose v2** | 2.20 | **Required for GPU.** See the warning below. |
| Disk | ~20 GB (GPU) / ~8 GB (CPU) | Measured images: `cm_backend:gpu` 9.5 GB, `cm_backend:latest` 3.4 GB, `cm_frontend:latest` 1.5 GB, plus ~600 MB InsightFace cache and the database |
| RAM | 8 GB | 16 GB recommended with Celery running |

> **⚠ Compose v1 will not do GPU.**
> The legacy `docker-compose` (1.29.x, the Python one) *parses* these files without
> complaint but **silently ignores** `deploy.resources.reservations.devices`.
> Containers then start with no GPU and quietly fall back to CPU inference.
> Check which one you have:
>
> ```bash
> docker compose version     # want: Docker Compose version v2.x  ← space, not hyphen
> docker-compose version     # legacy 1.29.x — do not use for GPU
> ```
>
> Every command in this guide uses `docker compose` (v2, with a space).

### GPU only

| Requirement | Minimum |
|---|---|
| NVIDIA driver | 535 (Linux) / 550 (Windows) |
| `nvidia-container-toolkit` | 1.14 (Linux only — Docker Desktop bundles it on Windows) |
| GPU | Any CUDA compute capability ≥ 6.1, ≥ 6 GB VRAM |

The GPU image is built on `nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04` with
`onnxruntime-gpu==1.18.1`. The host driver only needs to be **newer** than the
container's CUDA runtime — a CUDA 13 driver runs a CUDA 12.2 container fine.

> ### ⚠ VERIFY THIS FIRST, BEFORE ANYTHING ELSE ON A NEW MACHINE
>
> **`cm_backend:gpu` does not fall back to CPU. Without GPU access it crashes.**
>
> Measured 2026-09-03: the GPU image started without `--gpus` loads Python and
> imports the whole pipeline fine, then **segfaults (exit 139) the moment the
> detector touches its model**. There is no error message and no CPU fallback —
> the container simply dies. Isolated against a control: the crash happens with
> and without a network, and does *not* happen when the GPU is present, so it is
> GPU access alone.
>
> This contradicts the "silently ignores … and quietly fall back to CPU
> inference" wording in the Compose-v1 warning above. That is true of the
> *config* being dropped, but the practical result for **this** image is a crash,
> not slow inference. Treat `gpu_available: false` as a hard stop.
>
> One command settles it, before you deploy anything else:
>
> ```bash
> docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi
> ```
>
> If that does not print your GPU, **stop** and fix the driver /
> `nvidia-container-toolkit` first. Nothing downstream will work.
>
> If the target machine genuinely has no GPU, deploy the CPU image
> (`cm_backend:latest`, built from `Dockerfile.backend`) instead — that one is
> built against CPU `onnxruntime` and runs correctly without a GPU.

---

## 3. Model files — do this before the first build

Three model artifacts are needed. **None of them are baked into the images.**

| Artifact | Size | How it arrives | Used by |
|---|---|---|---|
| `models/face_landmarker.task` | 3.6 MB | `scripts/00_fetch_models.py` downloads it | MediaPipe FaceLandmarker (behaviour) |
| `models/public/open-closed-eye-0001/open-closed-eye.onnx` | 46 KB | **Vendored in git** — already in your clone | Eye-state CNN gate |
| InsightFace `buffalo_l` | ~600 MB | Auto-downloads on first use into the `cm_insightface_cache` volume | Face detection + ArcFace embeddings |

> **Why the eye model is committed to git rather than downloaded:** its Open Model
> Zoo mirror (`download.01.org`) now returns 404, and `storage.openvinotoolkit.org`
> does not serve it at any documented path. At 46 KB, vendoring it is the only way
> a fresh clone reliably has it.

Run once, from the project root:

```bash
python3 scripts/00_fetch_models.py
```

Expected output:

```
Models directory: /path/to/classroom_monitor/models

Downloadable artifacts
  ✓ MediaPipe FaceLandmarker  (ok)

Vendored artifacts (shipped in git)
  ✓ OMZ open-closed-eye-0001  (ok)

All model artifacts present and verified.
```

The script is stdlib-only (no `pip install` needed first), verifies SHA-256 on
every artifact, and writes atomically so an interrupted run never leaves a
half-written model behind. To verify without downloading — useful in CI or after
copying models to an offline machine:

```bash
python3 scripts/00_fetch_models.py --check   # exit 1 if anything is missing
```

**If you skip this step**, the stack still starts and `/health` still returns
healthy — but behaviour analysis throws on the *first processed frame*, not at
startup. That is exactly the silent failure this script exists to prevent, so
run `--check` as part of any fresh deployment.

`models/` is bind-mounted **read-only** into the backend and Celery containers at
`/app/models`, and the path is set by the `MODELS_PATH` env var (already wired in
both compose files).

---

## Path A — Linux / Ubuntu

### A.1 Clone and configure

```bash
git clone <repo-url> classroom_monitor
cd classroom_monitor

cp .env.example .env
```

Edit `.env` and set a real secret — the default is a placeholder that the app
will happily boot with:

```bash
python3 -c "import secrets; print('SECRET_KEY=' + secrets.token_hex(32))"
```

Paste that over the `SECRET_KEY=` line.

> **The containerised database is a separate database from your native one.**
> `.env` carries a `DATABASE_URL` for local development, but compose does not read
> it — it builds its own from `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB`,
> which `.env.example` does not define, so they fall back to
> `cm_user` / `cm_password` / `classroom_monitor` and land in the `cm_postgres_data`
> volume. That is self-consistent and fine for a fresh deployment. To have the
> stack use the *same* credentials as your native setup, add all three to `.env`
> explicitly.
>
> **Compose can adopt a container you did not start with it.** Any container
> carrying `com.docker.compose.project=classroom_monitor` labels — e.g. one left by
> an older compose file — is treated as this project's and will be recreated or
> removed by `docker compose up`. Check before the first run on a machine with
> existing containers:
>
> ```bash
> docker inspect <container> --format '{{json .Config.Labels}}'
> ```
>
> Named volumes are never deleted this way, so data survives; the container object
> does not. Start standalone databases with plain `docker run` (no compose labels)
> if you want them left alone.

### A.2 Install Docker Compose v2

```bash
sudo apt-get update
sudo apt-get install -y docker-compose-plugin
docker compose version          # expect v2.x
```

### A.3 Install the NVIDIA container toolkit *(GPU only)*

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

> Restarting the Docker daemon stops every running container. Check
> `docker ps` first if anything else on this machine matters.

Confirm the toolkit works *before* touching the app:

```bash
docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi
```

You should see your GPU's table. If this fails, nothing below will have GPU
access either — fix it here.

### A.4 Fetch models

```bash
python3 scripts/00_fetch_models.py
```

### A.5 Build

CPU:

```bash
docker compose build backend frontend
```

GPU:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build backend
docker compose build frontend
```

Both backend images are **two-stage**: `insightface==0.7.3` has no manylinux
wheel and compiles a Cython C++ extension from source, so a toolchain is
mandatory at install time. The `builder` stage carries `g++`; the runtime stage
receives only the finished virtualenv, so the compiler never ships.

The GPU build takes noticeably longer (CUDA base layers plus a Python 3.11
install on top of Ubuntu 22.04). It contains a guard that **fails the build** if
the CPU `onnxruntime` ends up installed alongside `onnxruntime-gpu` — having both
makes ONNX Runtime silently pick CPU.

### A.6 Initialise the database (first deployment only)

```bash
docker compose up -d postgres redis
docker compose run --rm backend python scripts/01_setup_db.py
docker compose run --rm backend python scripts/05_add_user.py    # create the first login
```

If you are enrolling students from video, also:

```bash
docker compose run --rm backend python scripts/02_register_students.py
```

### A.7 Start

CPU:

```bash
docker compose up -d
```

GPU:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

Then jump to [§6 Verification](#6-verification).

---

## Path B — Windows + Docker Desktop + WSL2 + NVIDIA

This is the Radar Lab scenario. Do the steps in order — the GPU checks build on
each other, and diagnosing a failure at step 5 is much harder if step 2 was
skipped.

### B.1 NVIDIA driver — Windows side only

Install the standard **Windows** NVIDIA driver (Game Ready or Studio, ≥ 550)
from nvidia.com, or via GeForce Experience.

> **Do not install an NVIDIA driver inside WSL.** This is the single most common
> way to break GPU passthrough. WSL2 receives the GPU through the Windows driver
> via `/usr/lib/wsl/lib`; installing a Linux driver in the distro overwrites that
> shim and GPU access stops working. There is no separate driver to install on
> the Linux side — only the container toolkit, and Docker Desktop provides that.

Verify from PowerShell:

```powershell
nvidia-smi
```

### B.2 WSL2

In an **Administrator** PowerShell:

```powershell
wsl --install -d Ubuntu-22.04
wsl --update
wsl --set-default-version 2
wsl --status
```

Reboot if prompted. Confirm the distro is version 2 — WSL**1** cannot do GPU:

```powershell
wsl -l -v          # VERSION column must read 2
```

### B.3 Docker Desktop

Install Docker Desktop for Windows (≥ 4.30), then in **Settings**:

- **General** → enable *Use the WSL 2 based engine*
- **Resources → WSL Integration** → enable integration for `Ubuntu-22.04`
- **Resources → Advanced** → give it ≥ 8 GB RAM and ≥ 40 GB disk

Apply & Restart. Docker Desktop ships the NVIDIA container runtime itself — do
**not** run the `nvidia-container-toolkit` apt steps from Path A here.

### B.4 Confirm GPU passthrough *before* the app

From the Ubuntu WSL shell:

```bash
nvidia-smi                      # the WSL shim; should list your GPU
docker compose version          # expect v2.x — Desktop bundles it

docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi
```

All three must succeed. If the third fails, stop and fix Docker Desktop's WSL
integration — the app cannot work around it.

### B.5 Clone into the WSL filesystem, not `/mnt/c`

```bash
cd ~                            # e.g. /home/<you>, NOT /mnt/c/Users/...
git clone <repo-url> classroom_monitor
cd classroom_monitor
```

> **This matters more than it looks.** Bind-mounting from `/mnt/c` crosses the
> 9P filesystem bridge: video I/O runs roughly an order of magnitude slower, and
> file-watching is unreliable. `~/` inside WSL is a native ext4 volume.

If Git for Windows was used to clone onto `/mnt/c` at any point, line endings can
turn shell scripts into CRLF files that fail with `bad interpreter`. Cloning from
inside WSL as above avoids this. If you hit it anyway:

```bash
git config --global core.autocrlf input
git rm --cached -r . && git reset --hard
```

### B.6 Configure, fetch models, build, start

```bash
cp .env.example .env
python3 -c "import secrets; print('SECRET_KEY=' + secrets.token_hex(32))"
# paste into .env

python3 scripts/00_fetch_models.py

docker compose -f docker-compose.yml -f docker-compose.gpu.yml build backend
docker compose build frontend

docker compose up -d postgres redis
docker compose run --rm backend python scripts/01_setup_db.py
docker compose run --rm backend python scripts/05_add_user.py

docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

### B.7 Access

From the **Windows** browser:

- Dashboard — <http://localhost:8501>
- API docs — <http://localhost:8000/docs>

Docker Desktop forwards published ports to the Windows host automatically; no
`netsh portproxy` is needed.

---

## 6. Verification

Run these in order on either platform.

### 6.1 Build context is small

```bash
DOCKER_BUILDKIT=0 docker build -f Dockerfile.backend -t cm_backend:latest . 2>&1 | grep "build context"
```

Expect roughly **`Sending build context to Docker daemon  831.5kB`**. If you see
hundreds of MB or more, `.dockerignore` is not being applied — confirm it sits in
the same directory as `docker-compose.yml` and that you are building with
`context: .` from the project root.

### 6.2 Everything is up and healthy

```bash
docker compose ps
```

All four services should read `healthy`:

```
cm_backend    Up (healthy)   0.0.0.0:8000->8000/tcp
cm_frontend   Up (healthy)   0.0.0.0:8501->8501/tcp
cm_postgres   Up (healthy)   0.0.0.0:5432->5432/tcp
cm_redis      Up (healthy)   0.0.0.0:6379->6379/tcp
```

The backend has a 90–120 s `start_period` because loading InsightFace on first
boot is slow — do not judge it before then.

### 6.3 GPU is live *inside* the container

This is the check that matters. `nvidia-smi` on the host proves nothing about the
container.

```bash
# a) the driver is visible inside the backend container
docker compose exec backend nvidia-smi
```

This is the check that actually proves passthrough. If it prints your GPU table,
the toolkit is working; if it errors, nothing else here will help.

> **`get_available_providers()` is NOT a GPU test.** It lists providers *compiled
> into* the build, not ones that can run. `cm_backend:gpu` reports
> `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`
> even with no GPU attached and no toolkit installed — verified. Treating that
> list as proof is how a CPU-bound deployment gets shipped as a GPU one. Use
> `nvidia-smi` and `cuda_available` below instead.

Then confirm the app itself agrees — `/health` reports a `gpu_available` flag:

```bash
curl -s localhost:8000/health | python3 -m json.tool
```

```json
{
  "status": "ok",
  "db": true,
  "gallery_size": 42,
  "gpu_available": true,
  "version": "..."
}
```

For the full device breakdown (`/health` deliberately keeps its payload small),
call the helper the startup log uses. `cuda_available` here is meaningful because
`gpu_devices` is populated by shelling out to `nvidia-smi` — it reflects a GPU
that genuinely responded, not one that merely compiled in:

```bash
docker compose exec backend python -c \
  "from backend.utils.gpu import get_device_info; import json; print(json.dumps(get_device_info(), indent=2))"
```

```json
{
  "cuda_available": true,
  "onnx_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
  "onnxruntime_version": "1.18.1",
  "gpu_devices": [
    { "index": 0, "name": "NVIDIA RTX A4000", "memory_total_mb": 16376, "memory_free_mb": 15904, "utilization_pct": 0 }
  ]
}
```

`"gpu_available": false` while running the GPU overlay means one of: the toolkit
is missing, Compose v1 was used (it drops the `deploy` block), or the CPU image is
still tagged `cm_backend:latest` and being reused. Check
`docker compose images backend` — the GPU stack must show `cm_backend:gpu`.

### 6.4 Models resolve inside the container

```bash
docker compose exec backend python scripts/00_fetch_models.py --check
```

### 6.5 Application smoke test

```bash
curl -s localhost:8000/health | python3 -m json.tool
```

`"db": true` confirms the backend reached Postgres through the compose network;
`gallery_size` is the number of enrolled face embeddings loaded from
`data/embeddings_cache`:

```json
{
  "status": "ok",
  "db": true,
  "gallery_size": 15,
  "gpu_available": false,
  "version": "0.1.0"
}
```

Then open <http://localhost:8501>, log in with the account created by
`05_add_user.py`, and confirm the sidebar renders.

> **First login:** seeded student accounts have `must_change_password` set and are
> redirected to a password-change screen before any other page will load. That is
> expected behaviour, not a deployment fault.

---

## 7. Moving to an offline machine

The Radar Lab PC may have no internet. Build on a connected machine, then
transfer.

> This section moves the **images and models**. It does not move the roster —
> students, face embeddings and history live in Postgres and are covered
> separately in [§7a](#7a-moving-the-roster--students-embeddings-and-history).
> A stack deployed without that step starts healthy and recognises nobody.

**On the build machine:**

```bash
docker save cm_backend:gpu cm_frontend:latest pgvector/pgvector:pg16 redis:7-alpine \
  | gzip > classroom_monitor_images.tar.gz          # ~6–7 GB gzipped

tar czf classroom_monitor_models.tar.gz models/     # ~3.7 MB

# buffalo_l lives in a named volume, not in models/ — extract it too
docker run --rm -v cm_insightface_cache:/src -v "$PWD":/out alpine \
  tar czf /out/insightface_cache.tar.gz -C /src .   # ~600 MB
```

Copy those three archives plus a `git clone` (or a source tarball) of the repo.

**On the target machine:**

```bash
gunzip -c classroom_monitor_images.tar.gz | docker load
tar xzf classroom_monitor_models.tar.gz             # restores models/

docker volume create cm_insightface_cache
docker run --rm -v cm_insightface_cache:/dst -v "$PWD":/in alpine \
  tar xzf /in/insightface_cache.tar.gz -C /dst

python3 scripts/00_fetch_models.py --check          # must pass with no network
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

Because the images are loaded rather than built, no `docker compose build` runs
and nothing is pulled from the network.

---

## 7a. Moving the roster — students, embeddings and history

§7 moves the *images and models*. It does not move the **roster**, and a stack
brought up without one starts cleanly, reports `healthy`, and recognises nobody.
`/health` shows `gallery_size: 0` and every face in the live feed is an unknown.

### What actually has to move — less than it looks

| Artifact | Size | Needed? | Why |
|---|---|---|---|
| **Postgres database** | small | **YES — this is the whole roster** | `students.face_embedding` is a real column. The embeddings live *in the database*, not in `data/` |
| `data/embeddings_cache/` | 256 KB | Optional | The FAISS index — a **derived cache**, rebuilt automatically from Postgres |
| `data/student_photos/` | 583 MB | Optional | Only serves `GET /students/{id}/photo` (the Admin panel thumbnail) and re-enrolment. Recognition never reads it |
| `data/videos/` | 1.4 GB | **No** | Test clips |

The FAISS index rebuilds itself. `backend/main.py` on startup:

```python
if settings.gallery_rebuild_on_startup or gallery.size == 0:
    # rebuild from PostgreSQL
```

So a restored database with **no** `data/embeddings_cache` produces a correct
gallery on the first backend boot, with no manual step. Copying the cache only
saves that one rebuild. `data/` is a **bind mount**, not a named volume, so it
does not travel with `docker save` either way.

### Option 1 — transfer the database *(recommended: keeps history)*

Carries the roster, the embeddings, past sessions, attendance and behaviour.

**On the source machine:**

```bash
docker compose exec -T postgres pg_dump -U cm_user -d classroom_monitor \
  --clean --if-exists > classroom_roster.sql
```

**On the target**, after §B.6 has created the database:

```bash
docker compose up -d postgres
docker compose exec -T postgres psql -U cm_user -d classroom_monitor < classroom_roster.sql
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d backend
docker compose logs backend | grep -i gallery      # expect "Rebuilding FAISS gallery"
```

`cm_user` and `classroom_monitor` are the compose defaults; if `.env` sets
`POSTGRES_USER` or `POSTGRES_DB`, substitute those in both commands.

Run `01_setup_db.py` **before** restoring, not after — it creates the schema the
dump expects. Do not run `02_register_students.py` or `06_seed_student_logins.py`
afterwards: the dump already contains those rows, and re-running them on top is
how duplicate students appear.

### Option 2 — re-enrol on site *(no history)*

Only if the database cannot leave the source machine. Needs the enrolment videos
or photos physically present on the target.

```bash
docker compose run --rm backend python scripts/02_register_students.py
docker compose run --rm backend python scripts/06_seed_student_logins.py
```

This produces **new student ids**, so any attendance history from the old machine
can never be reattached. It also re-runs face detection over every enrolment
video, which is the slow part — budget minutes per student, on the GPU.

> **Radar Lab note.** That PC holds enrolment videos for the whole Avionics
> department. `02_register_students.py --video-dir` defaults to
> `data/student_photos` and enrols *every* per-student folder it finds there, so
> pointing it at the departmental directory enrols hundreds of students instead
> of the 15-student 99B roster — and puts all of them into one FAISS gallery that
> `best_match` then searches for every face. If the intent is the existing 99B
> roster, use **Option 1** and do not run the enrolment scripts at all.

### Verifying the roster landed

Per §6.5, `gallery_size` in `/health` is the check:

```bash
curl -s localhost:8000/health | python3 -m json.tool
```

```json
{ "status": "ok", "db": true, "gallery_size": 15, "gpu_available": true }
```

| `gallery_size` | Meaning |
|---|---|
| **15** | The 99B roster is loaded. Correct |
| **0** | No embeddings reached the database. The dump did not restore, or restored into a different database than the backend is pointed at |
| Some other number | You enrolled something other than the intended roster — check `02_register_students.py`'s source directory |

Confirm the count independently, since `gallery_size` counts FAISS vectors rather
than roster rows:

```bash
docker compose exec -T postgres psql -U cm_user -d classroom_monitor -c \
  "SELECT count(*) AS students, count(face_embedding) AS with_embedding FROM students;"
```

Both numbers should equal the roster size. A student row with no embedding is
enrolled on paper and invisible to recognition.

---

## 7b. Live RTSP capture

The backend opens the camera itself and runs the pipeline on a dedicated thread
(`backend/pipeline/live_worker.py`). The browser never touches the stream, so the
URL only has to resolve **from the backend container** — a camera on the
classroom LAN works even when the operator's laptop is on a different network.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/sessions/{id}/live/start` | Open the URL and begin. Returns `202` as soon as the thread is armed |
| `POST` | `/api/v1/sessions/{id}/live/stop` | Stop, flush the final batch, close the session. Idempotent |
| `GET` | `/api/v1/sessions/{id}/live/status` | State, counters, detections, latest annotated JPEG |
| `GET` | `/api/v1/live/active` | Which session currently holds the pipeline |

All four require an instructor token. In the UI: **Live Monitor → Step 4 →
📡 Live stream URL**, which prefills from the room's `camera_url` if the Admin
panel has one recorded.

### One session at a time — by design

`register()` in `live_worker.py` refuses a second concurrent live session with a
`409`. `FaceRecognizer`, `ByteTracker` and `BehaviorAnalyzer` are process-wide
singletons holding mutable per-track state (track ids, EAR/yaw history, dwell
timers); two feeds would interleave into the same dicts and corrupt both.
Lifting the restriction means giving each worker its own tracker and analyzer.

### Supported sources

`rtsp://` · `rtsps://` · `http(s)://` · `udp://` · `tcp://` · or a bare camera
index (`0`). Anything else is rejected with `422` before a thread is started.

### Tuning (`.env`)

| Setting | Default | Effect |
|---|---|---|
| `LIVE_FRAME_SKIP` | `24` | Process every 25th frame (~1 fps from a 25 fps camera) |
| `LIVE_JPEG_QUALITY` | `70` | Quality of the annotated preview |
| `LIVE_FLUSH_EVERY` | `20` | Frames buffered before a DB write |
| `LIVE_RECONNECT_DELAY` | `2.0` | First backoff step, in seconds |
| `LIVE_MAX_RECONNECTS` | `20` | Retries before the session is marked `FAILED` |

Reconnects back off exponentially (capped at 30 s), so a camera reboot recovers
without burning the whole budget in the first minute.

### Verifying a camera from the backend container

The container is what has to reach the camera, so test from inside it:

```bash
docker compose exec backend python - <<'EOF'
from backend.pipeline.capture import VideoCapture
cap = VideoCapture("rtsp://user:pass@192.168.1.50:554/Streaming/Channels/101")
cap.open()
print("resolution:", cap.resolution, "fps:", cap.fps)
for n, ts, frame in cap.frames():
    print("first frame", n, ts, frame.shape); break
cap.close()
EOF
```

A dead host fails in ~5 s rather than hanging: `capture.py` passes
`timeout;5000000` (microseconds) to FFMPEG, and forces `rtsp_transport;tcp`
because RTSP-over-UDP tears frames on a congested LAN.

> **Do not "fix" that option back to `stimeout`.** Every older RTSP snippet uses
> that spelling, but FFmpeg renamed it, and the build behind
> `opencv-python-headless==4.10` (avformat 59.x) silently ignores the old name —
> measured against an unroutable host, `stimeout` opened in 29.9 s, identical to
> passing no options at all, while `timeout` gave up in 5.7 s.

### Timestamps

Live sources stamp frames from a wall clock, not `frame_number / fps`. Behaviour
dwell timers (`_DISTRACTED_SECONDS`, `_SLEEPING_SECONDS`) measure elapsed time
from those stamps; on a feed that stalls, reconnects, or runs off-nominal fps, a
derived clock drifts from reality and silently rescales every threshold.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `gpu_available: false` under the GPU overlay | Compose v1 dropped the `deploy` block | Use `docker compose` (v2, space), not `docker-compose` |
| Same, with v2 | `nvidia-container-toolkit` missing/unconfigured | Re-run §A.3, then `sudo systemctl restart docker` |
| Same, on Windows | WSL1, or a driver installed inside WSL | `wsl -l -v` must show 2; never install a Linux NVIDIA driver in the distro |
| Backend container exits immediately with code **139**, no error logged | `cm_backend:gpu` running without GPU access — it segfaults rather than falling back to CPU | Verify `docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi` works; if the machine has no GPU, deploy `cm_backend:latest` (the CPU image) instead |
| `Error response from daemon: could not select device driver "nvidia"` | Toolkit not registered with Docker | `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| `bind: address already in use` on 5432 | Another Postgres already holds the port — on the dev machine a container named `classroom_pg` does | `docker stop classroom_pg`, or set `POSTGRES_PORT=5433` in `.env` |
| Behaviour analysis throws on the first frame; startup was clean | Models missing (§3 skipped) | `docker compose exec backend python scripts/00_fetch_models.py --check` |
| Backend flapping / marked unhealthy on first boot | InsightFace downloading `buffalo_l` (~600 MB) | Wait out the `start_period`; `docker compose logs -f backend` to watch |
| `buffalo_l` download fails behind a proxy | No egress to the InsightFace CDN | Transfer the `cm_insightface_cache` volume per §7 |
| `Failed building wheel for insightface`, `No such file or directory: 'g++'` | A single-stage backend image with no C++ toolchain | Both backend Dockerfiles are two-stage for this reason — build deps live in the `builder` stage. Do not collapse them into one stage. |
| Build context still huge | `.dockerignore` not at the context root | It must sit beside `docker-compose.yml` |
| `__pycache__` still shipping into the image | A pattern was written without `**/` | `.dockerignore` patterns do **not** recurse — see the note at the top of that file |
| `422 Not a live stream URL` | A file path or bare hostname was submitted | Use a full `rtsp://` / `http://` URL, or a camera index |
| `409 A live session is already running` | The single-session policy (above) | Stop the other session; `GET /api/v1/live/active` names it |
| Live status stays `starting`, then `error` | The **container** cannot reach the camera | Test from inside the container as shown above; check VLAN/firewall, not the laptop's network |
| Live capture opens but no faces are recognised | Roster is course-scoped and empty, or the gallery is unbuilt | Confirm the session's subject maps to a course with enrolled students |
| Live session ends `FAILED` after a camera blip | Reconnect budget exhausted | Raise `LIVE_MAX_RECONNECTS` / `LIVE_RECONNECT_DELAY` in `.env` |
| Live endpoints return `404` after an upgrade | Image predates the live work — sources are baked in, not mounted | `docker compose build backend && docker compose up -d backend` |
| Student user is stuck on the password screen | `must_change_password` is set (by design) | Complete the change, or clear the flag via `scripts/07_force_student_password_change.py` |

> **Known wart:** `docker-compose.yml` sets `STORAGE_PATH`, but `Settings` has no
> `storage_path` field (it uses `video_storage_path`, `student_photos_path`, …)
> and `extra="ignore"` swallows it. The variable is a no-op. Nothing breaks —
> the relative defaults resolve correctly under `WORKDIR /app` — but do not
> expect editing `STORAGE_PATH` to relocate anything.

---

## 9. Day-to-day operations

```bash
docker compose logs -f backend            # follow logs
docker compose restart backend            # restart one service
docker compose --profile monitoring up -d # add the Flower UI on :5555

docker compose down                       # stop, KEEP database + model cache
docker compose down -v                    # stop and DELETE all volumes ⚠
```

`docker compose down -v` destroys `cm_postgres_data` (all attendance history) and
`cm_insightface_cache` (forcing a 600 MB re-download). Named volumes survive a
plain `down`, image rebuilds, and host reboots.

To upgrade after pulling new code:

```bash
git pull
python3 scripts/00_fetch_models.py --check
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build backend
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```
