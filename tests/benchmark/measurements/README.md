# Benchmark measurements

Benchmark results are stored here as JSON files, one per run.

## File naming

`focus-YYYYMMDD-HHMMSS.json` — Focus camera session benchmarks (EMO-17)

Example: `focus-20261014-203900.json`

## How to generate

```bash
# Run the Focus benchmarks and save results to measurements/
cd tests/benchmark
./run_focus_benchmark.sh
```

The script:
1. Builds the benchmark binary (if needed)
2. Runs all `BM_Focus*` benchmarks in JSON format
3. Saves the output to `measurements/focus-<timestamp>.json`
4. Prints a human-readable summary

## Committing results

Measurement files are committed to the repository so that performance
regressions can be spotted in code review. When a PR changes Focus server
code, run the script and commit the new measurement file.

## Reading the JSON

Google Benchmark JSON format. The key fields for the Focus benchmarks:

- `benchmarks[].name` — e.g. `BM_FocusConcurrentSessions/16/3000`
- `benchmarks[].real_time` — wall-clock seconds for the run
- `benchmarks[].counters.RPS_frames` — frames per second (the RPS metric)
- `benchmarks[].counters.Frames` — total frames uploaded
- `benchmarks[].counters.ConcurrentClients` — number of concurrent clients
