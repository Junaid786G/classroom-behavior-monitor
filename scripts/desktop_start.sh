#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Classroom CCTV Monitor — desktop start
#
#  SAFE BY DESIGN. This script only ever runs `docker start`, which resumes
#  containers that already exist. It NEVER runs `docker compose up`, `down`,
#  `rm`, or anything that can recreate or delete a container or a volume —
#  those stay manual, typed commands on purpose, because cm_postgres holds the
#  only copy of the real data.
#
#  Usage: ./scripts/desktop_start.sh     (or via the desktop launcher)
# ─────────────────────────────────────────────────────────────────────────────

BOLD=$'\e[1m'; RED=$'\e[31m'; GRN=$'\e[32m'; YEL=$'\e[33m'; CYA=$'\e[36m'; RS=$'\e[0m'

# Always pause, even on failure — the whole point is that the window stays open.
pause_and_exit() {
    echo
    echo "${CYA}────────────────────────────────────────────────────────${RS}"
    echo "${BOLD}Press any key to close this window...${RS}"
    read -r -n 1 -s
    exit "${1:-0}"
}
trap 'pause_and_exit 130' INT TERM

hr() { echo "${CYA}────────────────────────────────────────────────────────${RS}"; }

echo
echo "${BOLD}  Classroom CCTV Monitor — starting up${RS}"
hr

# ── 0. Docker daemon reachable? ──────────────────────────────────────────────
if ! docker info >/dev/null 2>&1; then
    echo "${RED}✗ Docker is not running (or this user cannot reach it).${RS}"
    echo "  Start Docker Desktop / the docker service, then run this again."
    pause_and_exit 1
fi
echo "${GRN}✓${RS} Docker daemon reachable"

# ── helpers ──────────────────────────────────────────────────────────────────
start_one() {
    local name="$1"
    if ! docker inspect "$name" >/dev/null 2>&1; then
        echo "  ${RED}✗ $name does not exist${RS} — it must be created manually"
        echo "    (deliberately not automated: creating containers is a typed command)"
        return 1
    fi
    if [ "$(docker inspect -f '{{.State.Running}}' "$name")" = "true" ]; then
        echo "  ${GRN}✓${RS} $name already running"
        return 0
    fi
    if docker start "$name" >/dev/null 2>&1; then
        echo "  ${GRN}✓${RS} $name started"
        return 0
    fi
    echo "  ${RED}✗ failed to start $name${RS}"
    return 1
}

# classroom_pg belongs to the AI Lab PC only — the Radar Lab PC has never had
# it. Absent means "not applicable on this machine", not a failure. If it IS
# present it is started and reported exactly like any required container, and
# a genuine failure to start still counts. Only classroom_pg is optional; every
# cm_* container stays required.
start_optional_one() {
    local name="$1"
    if ! docker inspect "$name" >/dev/null 2>&1; then
        echo "  ${CYA}–${RS} $name not present on this machine — skipping (optional)"
        return 0
    fi
    start_one "$name"
}

# Poll until healthy rather than sleeping blind. Containers without a
# healthcheck (classroom_pg) only need to be running.
wait_healthy() {
    local name="$1" limit="${2:-90}" waited=0 state health
    while [ "$waited" -lt "$limit" ]; do
        state=$(docker inspect -f '{{.State.Status}}' "$name" 2>/dev/null)
        health=$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$name" 2>/dev/null)
        [ "$state" != "running" ] && { sleep 2; waited=$((waited+2)); continue; }
        case "$health" in
            healthy|none) echo "  ${GRN}✓${RS} $name ready (${waited}s)"; return 0 ;;
        esac
        sleep 2; waited=$((waited+2))
        [ $((waited % 10)) -eq 0 ] && echo "     …waiting on $name (${waited}s, health=$health)"
    done
    echo "  ${YEL}!${RS} $name not healthy after ${limit}s (health=$health)"
    return 1
}

# WSL2 cold-boot quirk: on this machine the bind-mounted host models folder
# has twice come up EMPTY inside cm_backend even though the host copy is fine.
# /health still answers "ok" in that state, so the healthcheck alone cannot be
# trusted — look at the actual file. A single `docker restart` has always
# cleared it (it is a mount-staleness problem, not a missing-file problem).
MODEL_FILE=/app/models/face_landmarker.task

models_visible() { docker exec cm_backend test -f "$MODEL_FILE" >/dev/null 2>&1; }

verify_backend_models() {
    if models_visible; then
        echo "  ${GRN}✓${RS} models visible inside cm_backend ($MODEL_FILE)"
        return 0
    fi

    echo "  ${YEL}!${RS} $MODEL_FILE is MISSING inside cm_backend"
    echo "     (known WSL2 bind-mount staleness — /health can still say ok)"
    echo "     restarting cm_backend once to re-mount…"
    if ! docker restart cm_backend >/dev/null 2>&1; then
        echo "  ${RED}✗ docker restart cm_backend failed${RS}"
        return 1
    fi
    wait_healthy cm_backend 150 || return 1

    if models_visible; then
        echo "  ${GRN}✓${RS} models visible after restart — backend genuinely ready"
        return 0
    fi

    echo
    echo "${RED}✗ $MODEL_FILE is STILL missing after one restart.${RS}"
    echo "${RED}  Stopping here — do NOT use the app in this state; face and${RS}"
    echo "${RED}  behaviour analysis would run against a model that isn't there.${RS}"
    echo
    echo "  Investigate manually:"
    echo "    docker exec cm_backend ls -l /app/models"
    echo "    ls -l ./models          # the host side of the bind mount"
    echo "    docker inspect -f '{{json .Mounts}}' cm_backend"
    echo
    echo "  What the container currently sees in /app/models:"
    docker exec cm_backend ls -la /app/models 2>&1 | sed 's/^/    /'
    return 1
}

failed=0

# ── 1. Data layer first ──────────────────────────────────────────────────────
echo
echo "${BOLD}[1/3] Databases and cache${RS}"
start_optional_one classroom_pg || failed=1
for c in cm_postgres cm_redis; do start_one "$c" || failed=1; done
wait_healthy cm_postgres 90 || failed=1
wait_healthy cm_redis    60 || failed=1

# ── 2. Backend (slow: InsightFace model load) ────────────────────────────────
echo
echo "${BOLD}[2/3] Backend${RS}"
start_one cm_backend || failed=1
echo "  (model loading can take up to ~90s on first start)"
wait_healthy cm_backend 150 || failed=1
# Healthy is not the same as ready — confirm the model file is really there.
verify_backend_models || pause_and_exit 1

# ── 3. Frontend ──────────────────────────────────────────────────────────────
echo
echo "${BOLD}[3/3] Frontend${RS}"
start_one cm_frontend || failed=1
wait_healthy cm_frontend 90 || failed=1

# ── Health report ────────────────────────────────────────────────────────────
echo
hr
echo "${BOLD}  Backend /health${RS}"
hr
body=$(curl -s --max-time 15 http://localhost:8000/health 2>/dev/null)
if [ -z "$body" ]; then
    echo "${RED}✗ No response from http://localhost:8000/health${RS}"
    failed=1
else
    if command -v python3 >/dev/null 2>&1; then
        echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body"
    else
        echo "$body"
    fi
    case "$body" in
        *'"status":"ok"'*) echo; echo "${GRN}✓ backend reports OK${RS}" ;;
        *)                 echo; echo "${YEL}! backend responded but not status=ok${RS}"; failed=1 ;;
    esac
    case "$body" in
        *'"gpu_available":true'*)  echo "${GRN}✓ GPU available${RS}" ;;
        *'"gpu_available":false'*) echo "${YEL}! GPU NOT available — running on CPU${RS}" ;;
    esac
fi

echo
echo "${BOLD}  Container status${RS}"
hr
docker ps --filter name=classroom_pg --filter name=cm_ \
          --format 'table {{.Names}}\t{{.Status}}' 2>/dev/null

# ── Browser ──────────────────────────────────────────────────────────────────
echo
if [ "$failed" -eq 0 ]; then
    echo "${GRN}All services up.${RS} Opening http://localhost:8501 ..."
    (xdg-open http://localhost:8501 >/dev/null 2>&1 &)
else
    echo "${YEL}Something above needs attention — not opening the browser.${RS}"
    echo "Open http://localhost:8501 yourself once it looks right."
fi

pause_and_exit "$failed"
