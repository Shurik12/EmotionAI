#!/usr/bin/env bash
# Install the helper scripts into ~/emotionai-run/scripts (strip CRLF, chmod +x).
set -euo pipefail
SRC=/mnt/d/projects/EmotionAI/.opencode/skills/emotionai-run/scripts
DST=${EMOTIONAI_HOME:-$HOME/emotionai-run}/scripts
mkdir -p "$DST"
for f in sync rebuild up down prod; do
  sed 's/\r$//' "$SRC/$f.sh" >"$DST/$f.sh"
done
chmod +x "$DST"/*.sh
echo "installed -> $DST"
ls -l "$DST"
