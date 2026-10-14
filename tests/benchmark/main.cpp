#include <benchmark/benchmark.h>

// Existing benchmarks
void BM_ImageUploadAndProcessing(benchmark::State &state);
void BM_VideoUploadAndProcessing(benchmark::State &state);
void BM_ConcurrentImageUploads(benchmark::State &state);
void BM_ConcurrentVideoUploads(benchmark::State &state);
void BM_MixedWorkload(benchmark::State &state);
void BM_TaskStatusPolling(benchmark::State &state);
void BM_BatchProgressChecking(benchmark::State &state);

// Focus camera session benchmarks (EMO-17)
void BM_FocusSingleSession(benchmark::State &state);
void BM_FocusConcurrentSessions(benchmark::State &state);
void BM_FocusPolling(benchmark::State &state);
void BM_FocusFrameUpload(benchmark::State &state);

// ---------------------------------------------------------------------------
// Register existing benchmarks
// ---------------------------------------------------------------------------
BENCHMARK(BM_ImageUploadAndProcessing)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10)
    ->UseRealTime();

BENCHMARK(BM_VideoUploadAndProcessing)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(5)
    ->UseRealTime();

BENCHMARK(BM_ConcurrentImageUploads)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1)
    ->Arg(5)
    ->Arg(10)
    ->Arg(20)
    ->Arg(50)
    ->UseRealTime();

BENCHMARK(BM_ConcurrentVideoUploads)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1)
    ->Arg(3)
    ->Arg(5)
    ->Arg(10)
    ->UseRealTime();

BENCHMARK(BM_MixedWorkload)
    ->Unit(benchmark::kMillisecond)
    ->Arg(10)
    ->Arg(25)
    ->Arg(50)
    ->Arg(100)
    ->UseRealTime();

BENCHMARK(BM_TaskStatusPolling)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1000);

BENCHMARK(BM_BatchProgressChecking)
    ->Unit(benchmark::kMicrosecond)
    ->Arg(5)
    ->Arg(10)
    ->Arg(25)
    ->Arg(50);

// ---------------------------------------------------------------------------
// Focus benchmarks
// ---------------------------------------------------------------------------

// Single camera client: full session lifecycle at ~1 fps.
BENCHMARK(BM_FocusSingleSession)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->UseManualTime();

// Sustained load with N concurrent camera clients. The second arg is the
// run duration in milliseconds — long enough to get a stable RPS number,
// short enough to keep CI tractable.
BENCHMARK(BM_FocusConcurrentSessions)
    ->Unit(benchmark::kMillisecond)
    ->Args({1, 3000})
    ->Args({4, 3000})
    ->Args({8, 3000})
    ->Args({16, 3000})
    ->Args({32, 3000})
    ->Iterations(1)
    ->UseManualTime();

// Poll-only load: the Focus page's periodic GET while the camera runs.
BENCHMARK(BM_FocusPolling)
    ->Unit(benchmark::kMillisecond)
    ->Args({1})
    ->Args({4})
    ->Args({8})
    ->Args({16})
    ->Iterations(1)
    ->UseManualTime();

// Frame upload throughput on a single shared session.
BENCHMARK(BM_FocusFrameUpload)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Iterations(1)
    ->UseManualTime();

BENCHMARK_MAIN();
