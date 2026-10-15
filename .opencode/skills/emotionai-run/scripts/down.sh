#!/usr/bin/env bash
# Stop the backend and frontend tmux sessions.
set -uo pipefail
tmux kill-session -t emotionai 2>/dev/null || true
tmux kill-session -t emotionai-front 2>/dev/null || true
echo "stopped: emotionai, emotionai-front"
