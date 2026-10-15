#!/usr/bin/env bash
# Sync edited sources from the Windows tree into the WSL build tree.
# Only the directories we actually edit are synced (src/, frontend/src/).
# Touches files whose content changed so ninja/cmake rebuilds them.
set -euo pipefail

WIN=${EMOTIONAI_WIN:-/mnt/d/projects/EmotionAI}
REPO=${EMOTIONAI_REPO:-/home/rinn/projects/EmotionAI}

for d in src frontend/src tests; do
  mkdir -p "$REPO/$d"
  echo "-- $d"
  if command -v rsync >/dev/null 2>&1; then
    list=$(mktemp)
    rsync -rc --itemize-changes --no-perms --no-owner --no-group \
      "$WIN/$d/" "$REPO/$d/" | tee "$list"
    # touch transferred files: Windows mtimes are older than build objects,
    # so without this the build would consider them up to date.
    awk '/^[<>]/{print $2}' "$list" | while read -r rel; do
      [ -n "$rel" ] && [ -f "$REPO/$d/$rel" ] && touch "$REPO/$d/$rel"
    done
    rm -f "$list"
  else
    cp -a "$WIN/$d/." "$REPO/$d/"
    find "$REPO/$d" -type f -exec touch {} +
  fi
done

echo "synced src/ frontend/src/ -> $REPO"
