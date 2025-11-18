#include <benchmark/benchmark.h>

// Declare benchmark functions
void BM_ImageUploadAndProcessing(benchmark::State &state);
void BM_VideoUploadAndProcessing(benchmark::State &state);
void BM_ConcurrentImageUploads(benchmark::State &state);
void BM_ConcurrentVideoUploads(benchmark::State &state);
void BM_MixedWorkload(benchmark::State &state);
void BM_TaskStatusPolling(benchmark::State &state);
void BM_BatchProgressChecking(benchmark::State &state);

// Register benchmarks
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

BENCHMARK_MAIN();