#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Classroom CCTV Monitor — DEV MODE ON  (live-mounted source)
#
#  Runs exactly one command, the one verified on 2026-09-03:
#
#    docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
#                   -f docker-compose.dev.yml up -d --no-deps backend frontend
#
#  SAFE BY DESIGN: `up -d --no-deps` only. It recreates the backend and
#  frontend containers with ./backend and ./frontend bind-mounted over the
#  baked-in copies. It NEVER runs `down`, `rm`, `prune`, or anything that
#  touches a volume — the databases are not even named on the command line.
#
#  This does NOT rebuild an image. Switching back is the companion script,
#  desktop_dev_mode_off.sh.
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
echo "${BOLD}  Classroom CCTV Monitor — switching to DEV MODE${RS}"
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

for f in docker-compose.yml docker-compose.gpu.yml docker-compose.dev.yml; do
    [ -f "$f" ] || { echo "${RED}✗ Missing $f${RS}"; pause_and_exit 1; }
done
echo "${GRN}✓${RS} all three compose files present"

echo
echo "Recreating backend and frontend with live-mounted source..."
if ! docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
                    -f docker-compose.dev.yml \
                    up -d --no-deps backend frontend 2>&1 | sed 's/^/   /'; then
    echo "${RED}✗ compose command failed — see the output above.${RS}"
    pause_and_exit 1
fi

# Poll rather than sleep blind. The backend reloads InsightFace on every
# recreate, which is the slow part (~40s, allow up to 150s).
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

# "It started" is not "it worked". Prove the mounts actually landed.
echo
echo "${BOLD}  Verifying the mounts actually landed${RS}"
hr
be_mnt=$(docker inspect -f '{{range .Mounts}}{{.Destination}} {{end}}' cm_backend 2>/dev/null)
fe_mnt=$(docker inspect -f '{{range .Mounts}}{{.Destination}} {{end}}' cm_frontend 2>/dev/null)
case "$be_mnt" in *"/app/backend"*)  echo "  ${GRN}✓${RS} ./backend  -> /app/backend  (live)" ;;
                  *) echo "  ${RED}✗ /app/backend NOT mounted — dev mode did not take${RS}"; failed=1 ;; esac
case "$fe_mnt" in *"/app/frontend"*) echo "  ${GRN}✓${RS} ./frontend -> /app/frontend (live)" ;;
                  *) echo "  ${RED}✗ /app/frontend NOT mounted — dev mode did not take${RS}"; failed=1 ;; esac
# The base file's data/models mounts must survive the merge.
case "$be_mnt" in *"/app/data"*) echo "  ${GRN}✓${RS} base mounts preserved (/app/data present)" ;;
                  *) echo "  ${RED}✗ base mounts LOST — stop and investigate${RS}"; failed=1 ;; esac

echo
echo "${BOLD}  Backend /health${RS}"
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

echo
if [ "$failed" -eq 0 ]; then
    echo "${INV}${BOLD}                                                                ${RS}"
    echo "${INV}${BOLD}   ####   DEV MODE IS ON   ####                                 ${RS}"
    echo "${INV}${BOLD}                                                                ${RS}"
    echo "${INV}${BOLD}   THIS IS NOT THE CERTIFIED BASELINE CONFIGURATION.            ${RS}"
    echo "${INV}${BOLD}                                                                ${RS}"
    echo
    echo "${YEL}  The containers are now serving source from your working tree,${RS}"
    echo "${YEL}  NOT the code baked into the certified images.${RS}"
    echo
    echo "  What this means:"
    echo "    * frontend edits apply on save - the browser reruns by itself"
    echo "    * backend edits need:  docker restart cm_backend   (~40s)"
    echo "      or the 'Restart Backend (clean state)' desktop icon"
    echo "    * an uncommitted or half-finished edit IS what runs"
    echo
    echo "${BOLD}  Before demonstrating or quoting any numbers, switch back with${RS}"
    echo "${BOLD}  the 'Dev Mode OFF' desktop icon.${RS}"
else
    echo "${RED}${BOLD}  DEV MODE DID NOT COME UP CLEANLY - see the errors above.${RS}"
    echo "${RED}  Do not assume live editing is working. Switch back with the${RS}"
    echo "${RED}  'Dev Mode OFF' icon and investigate before relying on it.${RS}"
fi

pause_and_exit "$failed"
