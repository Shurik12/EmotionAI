# Benchmark measurements

Benchmark results are stored here as JSON files, one per run.

## File naming

`focus-YYYYMMDD-HHMMSS.json` — Focus camera session benchmarks (EMO-17)

Example: `focus-20261014-203900.json`

## How to generate

```bash
# From the repo root (requires a running EmotionAI server on :8080)
make benchmark_focus
```

This builds the benchmark binary and runs all `BM_Focus*` benchmarks,
saving the JSON to `measurements/focus-<timestamp>.json` via Google
Benchmark's native `--benchmark_out` flag. The console also prints a
human-readable summary table.

## Committing results

Measurement files are committed to the repository so that performance
regressions can be spotted in code review. When a PR changes Focus server
code, run `make benchmark_focus` and commit the new measurement file.

## Reading the JSON

Google Benchmark JSON format. The key fields for the Focus benchmarks:

- `benchmarks[].name` — e.g. `BM_FocusConcurrentSessions/16/3000`
- `benchmarks[].real_time` — wall-clock seconds for the run
- `benchmarks[].counters.RPS_frames` — frames per second (the RPS metric)
- `benchmarks[].counters.Frames` — total frames uploaded
- `benchmarks[].counters.ConcurrentClients` — number of concurrent clients
