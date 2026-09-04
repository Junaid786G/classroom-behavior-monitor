#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Classroom CCTV Monitor — DEV MODE OFF  (back to the certified baseline)
#
#  Runs exactly one command, the one verified on 2026-09-03:
#
#    docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
#                   up -d --no-deps backend frontend
#
#  Switching back is simply omitting docker-compose.dev.yml from the -f list.
#  Compose recreates the two containers WITHOUT the source mounts, so they
#  serve the code baked into their images again.
#
#  SAFE BY DESIGN: `up -d --no-deps` only. Never `down`, `rm`, `prune`, and it
#  never names a volume or a database container.
# ─────────────────────────────────────────────────────────────────────────────

# Derive the project directory from this script's OWN location, so a clone at
# any path under any username works. readlink -f resolves symlinks, so a
# symlinked launcher still lands on the real scripts/ directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "$(readlink -f -- "$0")")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
BOLD=$'\e[1m'; RED=$'\e[31m'; GRN=$'\e[32m'; YEL=$'\e[33m'; CYA=$'\e[36m'
INV=$'\e[7m'; RS=$'\e[0m'

pause_and_exit() {
    echo
    echo "${CYA}────────────────────────────────────────────────────────────────${RS}"
    echo "${BOLD}Press any key to close this window...${RS}"
    read -r -n 1 -s
    exit "${1:-0}"
}
trap 'pause_and_exit 130' INT TERM
hr() { echo "${CYA}────────────────────────────────────────────────────────────────${RS}"; }

echo
echo "${BOLD}  Classroom CCTV Monitor — returning to CERTIFIED BASELINE${RS}"
hr

cd "$PROJECT_DIR" 2>/dev/null || {
    echo "${RED}✗ Cannot enter $PROJECT_DIR${RS}"
    echo "  (derived from this script at $SCRIPT_DIR)"; pause_and_exit 1; }

# A derived path is only as good as what it points at. Fail loudly here rather
# than letting `docker compose` fail with a confusing error further down.
if [ ! -f "$PROJECT_DIR/docker-compose.yml" ]; then
    echo "${RED}✗ $PROJECT_DIR does not look like the project${RS}"
    echo "  (no docker-compose.yml; this script must stay in the repo's scripts/ dir)"
    pause_and_exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "${RED}✗ Docker is not running (or this user cannot reach it).${RS}"
    echo "  Start Docker, then run this again."
    pause_and_exit 1
fi
echo "${GRN}✓${RS} Docker daemon reachable"

for f in docker-compose.yml docker-compose.gpu.yml; do
    [ -f "$f" ] || { echo "${RED}✗ Missing $f${RS}"; pause_and_exit 1; }
done
echo "${GRN}✓${RS} compose files present"

echo
echo "Recreating backend and frontend from their baked images..."
if ! docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
                    up -d --no-deps backend frontend 2>&1 | sed 's/^/   /'; then
    echo "${RED}✗ compose command failed — see the output above.${RS}"
    pause_and_exit 1
fi

wait_healthy() {
    local name="$1" limit="${2:-150}" waited=0 health
    while [ "$waited" -lt "$limit" ]; do
        health=$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$name" 2>/dev/null)
        case "$health" in healthy|none) echo "  ${GRN}✓${RS} $name ready (${waited}s)"; return 0 ;; esac
        sleep 2; waited=$((waited+2))
        [ $((waited % 10)) -eq 0 ] && echo "     …waiting on $name (${waited}s, health=$health)"
    done
    echo "  ${YEL}!${RS} $name not healthy after ${limit}s (health=$health)"; return 1
}

echo
failed=0
wait_healthy cm_backend 150 || failed=1
wait_healthy cm_frontend 90 || failed=1

# Prove the mounts are GONE - the whole point of this script.
echo
echo "${BOLD}  Verifying the source mounts are gone${RS}"
hr
be_mnt=$(docker inspect -f '{{range .Mounts}}{{.Destination}} {{end}}' cm_backend 2>/dev/null)
fe_mnt=$(docker inspect -f '{{range .Mounts}}{{.Destination}} {{end}}' cm_frontend 2>/dev/null)
case "$be_mnt" in *"/app/backend"*)  echo "  ${RED}✗ /app/backend is STILL mounted - dev mode did not clear${RS}"; failed=1 ;;
                  *) echo "  ${GRN}✓${RS} /app/backend  not mounted - serving the image" ;; esac
case "$fe_mnt" in *"/app/frontend"*) echo "  ${RED}✗ /app/frontend is STILL mounted - dev mode did not clear${RS}"; failed=1 ;;
                  *) echo "  ${GRN}✓${RS} /app/frontend not mounted - serving the image" ;; esac
case "$be_mnt" in *"/app/data"*) echo "  ${GRN}✓${RS} base mounts intact (/app/data present)" ;;
                  *) echo "  ${RED}✗ base mounts missing - stop and investigate${RS}"; failed=1 ;; esac

# The certified thresholds are the other half of "baseline".
echo
echo "${BOLD}  Backend /health and thresholds${RS}"
hr
body=$(curl -s --max-time 15 http://localhost:8000/health 2>/dev/null)
if [ -z "$body" ]; then
    echo "${RED}✗ No response from http://localhost:8000/health${RS}"; failed=1
else
    command -v python3 >/dev/null 2>&1 && { echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body"; } || echo "$body"
    case "$body" in *'"status":"ok"'*) echo; echo "${GRN}✓ backend reports OK${RS}" ;;
                    *) echo; echo "${YEL}! backend responded but not status=ok${RS}"; failed=1 ;; esac
    case "$body" in *'"gpu_available":true'*)  echo "${GRN}✓ GPU available${RS}" ;;
                    *'"gpu_available":false'*) echo "${YEL}! GPU NOT available — running on CPU${RS}" ;; esac
fi
thr=$(docker exec cm_backend sh -c 'env | grep -E "^(FACE_RECOGNITION|ATTENDANCE_CONFIDENCE)_THRESHOLD"' 2>/dev/null | sort | tr '\n' ' ')
echo "  thresholds: ${thr:-<none found>}"
case "$thr" in *"FACE_RECOGNITION_THRESHOLD=0.35"*) echo "  ${GRN}✓${RS} recognition threshold is the certified 0.35" ;;
               *) echo "  ${YEL}!${RS} recognition threshold is NOT 0.35 - check docker-compose.yml" ;; esac

echo
if [ "$failed" -eq 0 ]; then
    echo "${INV}${BOLD}                                                                ${RS}"
    echo "${INV}${BOLD}   ####   CERTIFIED BASELINE RESTORED   ####                    ${RS}"
    echo "${INV}${BOLD}                                                                ${RS}"
    echo
    echo "${GRN}  Dev mode is OFF. The containers are serving the code baked${RS}"
    echo "${GRN}  into their images again - the configuration the certified${RS}"
    echo "${GRN}  numbers were measured under.${RS}"
    echo
    echo "  Note: any edit still sitting in your working tree is NOT live"
    echo "  now. It is safe on disk, it simply is not what is running."
    echo
    echo "  Before a run whose numbers you intend to quote, use the"
    echo "  'Restart Backend (clean state)' icon for a fresh process."
else
    echo "${RED}${BOLD}  DID NOT RETURN TO BASELINE CLEANLY - see the errors above.${RS}"
    echo "${RED}  Do NOT quote numbers from this state until it is resolved.${RS}"
fi

pause_and_exit "$failed"
