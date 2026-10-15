#!/usr/bin/env bash
# Rebuild the backend target only.
# - persistent log in ~/emotionai-run/build.log (NOT /tmp: it does not survive a
#   WSL restart)
# - never deletes build/.ninja_log (that forces a full 58-file rebuild)
# - retries once if ninja reports a corrupt log ("premature end of file")
set -uo pipefail

REPO=${EMOTIONAI_REPO:-/home/rinn/projects/EmotionAI}
JOBS=${JOBS:-4}
LOG_DIR=${EMOTIONAI_HOME:-$HOME/emotionai-run}
LOG="$LOG_DIR/build.log"
mkdir -p "$LOG_DIR"

cd "$REPO" || { echo "no repo at $REPO"; exit 2; }

: >"$LOG"
cmake --build build --target emotionai -j"$JOBS" >>"$LOG" 2>&1
rc=$?

# A corrupted ninja log aborts the run early; a second pass usually completes.
if [ "$rc" -ne 0 ] && grep -q 'premature end of file' "$LOG"; then
  echo "ninja log looks corrupt, retrying once..." | tee -a "$LOG"
  cmake --build build --target emotionai -j"$JOBS" >>"$LOG" 2>&1
  rc=$?
fi

echo "EXIT=$rc" >>"$LOG"
tail -n 25 "$LOG"
exit "$rc"
