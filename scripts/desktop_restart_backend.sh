#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Classroom CCTV Monitor — restart the backend only
#
#  WHY THIS EXISTS
#  The BehaviorAnalyzer is a process-wide singleton whose per-track state
#  (EAR baselines, dwell timers) is keyed by track_id, and ByteTracker restarts
#  ids at 1 every session. Leaked state measurably changed results on the 8-min
#  clip: SLEEPING 177 -> 44. A fresh process guarantees a clean slate, so this
#  is the "restart before a run whose numbers you intend to quote" habit.
#
#  SAFE BY DESIGN: `docker restart` only. Never compose up/down, never rm.
#  It touches nothing but the backend container — the databases keep running.
# ─────────────────────────────────────────────────────────────────────────────

BOLD=$'\e[1m'; RED=$'\e[31m'; GRN=$'\e[32m'; YEL=$'\e[33m'; CYA=$'\e[36m'; RS=$'\e[0m'

pause_and_exit() {
    echo
    echo "${CYA}────────────────────────────────────────────────────────${RS}"
    echo "${BOLD}Press any key to close this window...${RS}"
    read -r -n 1 -s
    exit "${1:-0}"
}
trap 'pause_and_exit 130' INT TERM

echo
echo "${BOLD}  Restarting cm_backend (clean in-memory state)${RS}"
echo "${CYA}────────────────────────────────────────────────────────${RS}"

if ! docker info >/dev/null 2>&1; then
    echo "${RED}✗ Docker is not running.${RS}"
    pause_and_exit 1
fi

if ! docker inspect cm_backend >/dev/null 2>&1; then
    echo "${RED}✗ cm_backend does not exist.${RS}"
    echo "  Creating it is a manual, typed command — not automated here."
    pause_and_exit 1
fi

echo "Restarting ..."
if ! docker restart cm_backend >/dev/null 2>&1; then
    echo "${RED}✗ docker restart cm_backend failed${RS}"
    pause_and_exit 1
fi
echo "${GRN}✓${RS} restart issued"

# ~40s is the usual model-load time; poll rather than sleep blind, and give it
# headroom up to 150s before calling it a problem.
echo
echo "Waiting for the backend to come back (InsightFace load is ~40s)..."
waited=0; limit=150; health=starting
while [ "$waited" -lt "$limit" ]; do
    health=$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' cm_backend 2>/dev/null)
    [ "$health" = "healthy" ] && break
    sleep 2; waited=$((waited+2))
    [ $((waited % 10)) -eq 0 ] && echo "   …${waited}s (health=$health)"
done

if [ "$health" = "healthy" ]; then
    echo "${GRN}✓${RS} healthy after ${waited}s"
else
    echo "${YEL}!${RS} still '$health' after ${waited}s"
fi

echo
echo "${BOLD}  Backend /health${RS}"
echo "${CYA}────────────────────────────────────────────────────────${RS}"
body=$(curl -s --max-time 15 http://localhost:8000/health 2>/dev/null)
rc=0
if [ -z "$body" ]; then
    echo "${RED}✗ No response from http://localhost:8000/health${RS}"
    rc=1
else
    if command -v python3 >/dev/null 2>&1; then
        echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body"
    else
        echo "$body"
    fi
    case "$body" in
        *'"status":"ok"'*) echo; echo "${GRN}✓ Backend is up with clean state — safe to start a measured run.${RS}" ;;
        *)                 echo; echo "${YEL}! Responded, but not status=ok${RS}"; rc=1 ;;
    esac
fi

pause_and_exit "$rc"
