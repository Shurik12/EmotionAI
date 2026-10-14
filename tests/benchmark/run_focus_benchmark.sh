#!/usr/bin/env bash
# Run the Focus camera session benchmarks and save results to measurements/.
#
# Usage:
#   ./run_focus_benchmark.sh [server_host] [server_port]
#
# Defaults:
#   server_host = localhost
#   server_port = 8080
#
# The benchmark binary must be built first (make benchmark).
# A running EmotionAI server is required.

set -euo pipefail

SERVER_HOST="${1:-localhost}"
SERVER_PORT="${2:-8080}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEASUREMENTS_DIR="${SCRIPT_DIR}/measurements"
BINARY="${SCRIPT_DIR}/../../build/tests/benchmark/emotionai_benchmark"

mkdir -p "${MEASUREMENTS_DIR}"

# --- Check prerequisites -----------------------------------------------------
if [[ ! -x "${BINARY}" ]]; then
    echo "error: benchmark binary not found at ${BINARY}" >&2
    echo "hint: run 'make benchmark' from the repo root first" >&2
    exit 1
fi

# Quick reachability check
if ! curl -fsS "http://${SERVER_HOST}:${SERVER_PORT}/api/health" > /dev/null 2>&1; then
    echo "error: server not reachable at http://${SERVER_HOST}:${SERVER_PORT}/api/health" >&2
    echo "hint: start the server with 'make up' or point to a running instance" >&2
    exit 1
fi

# --- Run benchmarks ----------------------------------------------------------
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT_FILE="${MEASUREMENTS_DIR}/focus-${TIMESTAMP}.json"

echo "Running Focus benchmarks against ${SERVER_HOST}:${SERVER_PORT}..."
echo "Saving results to ${OUTPUT_FILE}"
echo

# The binary resolves test assets relative to CWD, so run from the benchmark dir.
cd "${SCRIPT_DIR}"
"${BINARY}" \
    --benchmark_filter="BM_Focus" \
    --benchmark_format=json \
    --benchmark_repetitions=1 \
    > "${OUTPUT_FILE}" 2>/tmp/focus_benchmark_stderr.log

# --- Print human-readable summary -------------------------------------------
echo
echo "=== Focus Benchmark Results (${TIMESTAMP}) ==="
echo
python3 - "${OUTPUT_FILE}" <<'PYEOF'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)

rows = []
for bm in data.get("benchmarks", []):
    name = bm.get("name", "").replace("/iterations:1/manual_time", "")
    rps = bm.get("RPS_frames") or bm.get("RPS_polls") or 0
    clients = bm.get("ConcurrentClients", 0)
    frames = bm.get("Frames", bm.get("Polls", 0))
    rows.append((name, clients, frames, rps))

rows.sort(key=lambda r: r[0])

print(f"{'Benchmark':<50} {'Clients':>7} {'Total':>10} {'RPS':>10}")
print("-" * 80)
for name, clients, frames, rps in rows:
    print(f"{name:<50} {clients:>7.0f} {frames:>10.0f} {rps:>10.1f}")
PYEOF

echo
echo "Results saved to: ${OUTPUT_FILE}"
echo
echo "Remember to commit the measurement file:"
echo "  git add ${MEASUREMENTS_DIR}/focus-${TIMESTAMP}.json"
