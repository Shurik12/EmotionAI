#!/usr/bin/env bash
# Sync the Windows working tree into the WSL build tree.
# Only what the build actually reads is synced: src/, frontend/src/, tests/,
# plus a few root-level build files. Files deleted in the Windows tree are also
# deleted in WSL (--delete), otherwise stale sources break the CMake glob.
# Touches files whose content changed so ninja/cmake rebuilds them.
set -euo pipefail

WIN=${EMOTIONAI_WIN:-/mnt/d/projects/EmotionAI}
REPO=${EMOTIONAI_REPO:-/home/rinn/projects/EmotionAI}

for d in src frontend/src tests; do
  mkdir -p "$REPO/$d"
  echo "-- $d"
  if command -v rsync >/dev/null 2>&1; then
    list=$(mktemp)
    rsync -rc --delete --itemize-changes --no-perms --no-owner --no-group \
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

# Root-level files the build reads (changed by main: new src/analytics, no cluster).
for f in CMakeLists.txt Makefile config_template.yaml frontend/package.json frontend/vite.config.js; do
  if [ -f "$WIN/$f" ]; then
    cp -p "$WIN/$f" "$REPO/$f"
    touch "$REPO/$f"
  fi
done

echo "synced src/ frontend/src/ tests/ + build files -> $REPO"
