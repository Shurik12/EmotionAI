#!/usr/bin/env bash
# Bring the stack up on localhost:
#   Dragonfly (Redis) -> backend :80 (tmux) -> Vite dev server :3000 (tmux).
# Both servers survive this shell exiting because they run in detached tmux.
set -uo pipefail

REPO=${EMOTIONAI_REPO:-/home/rinn/projects/EmotionAI}
FRONT_PORT=${EMOTIONAI_FRONT_PORT:-3000}
API_TARGET=${EMOTIONAI_API_URL:-http://localhost:80}
START_BACKEND=${EMOTIONAI_START_BACKEND:-/home/rinn/start_emotionai.sh}
LOG_DIR=${EMOTIONAI_HOME:-$HOME/emotionai-run}
mkdir -p "$LOG_DIR"

# 1. Dependency: Dragonfly (Redis) on 127.0.0.1:6379
if command -v docker >/dev/null 2>&1; then
  docker start dragonfly >/dev/null 2>&1 || true
fi

# 2. Backend (http://localhost -> :80)
tmux kill-session -t emotionai 2>/dev/null || true
tmux new-session -d -s emotionai "$START_BACKEND"

# 3. Frontend dev server (proxies /api to the backend)
tmux kill-session -t emotionai-front 2>/dev/null || true
tmux new-session -d -s emotionai-front \
  "cd $REPO/frontend && API_URL=$API_TARGET npm run dev -- --host 0.0.0.0 --no-open"

# 4. Wait for both to answer
b=""; f=""
for _ in $(seq 1 25); do
  b=$(curl -s -o /dev/null -w '%{http_code}' http://localhost/api/health || true)
  f=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$FRONT_PORT/" || true)
  [ "$b" = "200" ] && [ "$f" = "200" ] && break
  sleep 1
done

echo "backend  :80   -> $b"
echo "frontend :$FRONT_PORT -> $f"
echo "open http://localhost:$FRONT_PORT/"
