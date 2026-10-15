#!/usr/bin/env bash
# Serve the built SPA from the backend (http://localhost/, port 80) without the
# "Failed to load module script ... MIME type text/html" breakage:
#   1. point paths.frontend at frontend/dist
#   2. rebuild the SPA
#   3. restart the backend session
set -uo pipefail

REPO=${EMOTIONAI_REPO:-/home/rinn/projects/EmotionAI}
CFG="$REPO/config.yaml"
START_BACKEND=${EMOTIONAI_START_BACKEND:-/home/rinn/start_emotionai.sh}

# 1. Static root must be frontend/dist (Vite outDir), not frontend/.
if [ -f "$CFG" ] && grep -q 'frontend: "./frontend"' "$CFG"; then
  sed -i 's|frontend: "./frontend"|frontend: "./frontend/dist"|' "$CFG"
  echo "config: paths.frontend -> ./frontend/dist"
fi

# 2. Build the SPA
( cd "$REPO/frontend" && npm run build ) || { echo "frontend build failed"; exit 1; }

# 3. Restart the backend so it re-reads config.yaml
tmux kill-session -t emotionai 2>/dev/null || true
tmux new-session -d -s emotionai "$START_BACKEND"
sleep 7

idx=$(curl -s http://localhost/)
js=$(printf '%s' "$idx" | grep -o '/static/main-[^"]*\.js' | head -1)
echo "health: $(curl -s -o /dev/null -w '%{http_code}' http://localhost/api/health)"
if [ -n "$js" ]; then
  echo "js: $js -> $(curl -s -o /dev/null -w '%{http_code} %{content_type}' "http://localhost$js")"
fi
