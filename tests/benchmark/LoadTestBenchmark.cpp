#include "EmotionServerBenchmark.h"
#include <atomic>
#include <thread>
#include <vector>
#include <future>

// Benchmark for single image upload and processing
void BM_ImageUploadAndProcessing(benchmark::State &state)
{
	EmotionServerBenchmark::BenchmarkConfig config;
	config.test_image_path = "assets/test_image.jpg";
	config.test_video_path = "assets/test_video.mp4";

	EmotionServerBenchmark benchmark(config);

	for (auto _ : state)
	{
		state.PauseTiming(); // Don't count setup in timing

		std::string task_id;
		bool upload_success = benchmark.benchmarkImageUpload(state, task_id);

		if (!upload_success)
		{
			state.ResumeTiming();
			continue;
		}

		state.ResumeTiming();

		// Wait for processing to complete (this is what we're timing)
		bool processing_success = benchmark.waitForTaskCompletion(task_id);

		if (!processing_success)
		{
			state.SkipWithError("Image processing failed or timed out");
		}
	}

	state.counters["Throughput"] = benchmark::Counter(state.iterations(),
													  benchmark::Counter::kIsRate, benchmark::Counter::OneK::kIs1000);
}

// Benchmark for single video upload and processing
void BM_VideoUploadAndProcessing(benchmark::State &state)
{
	EmotionServerBenchmark::BenchmarkConfig config;
	config.test_image_path = "assets/test_image.jpg";
	config.test_video_path = "assets/test_video.mp4";

	EmotionServerBenchmark benchmark(config);

	for (auto _ : state)
	{
		state.PauseTiming();

		std::string task_id;
		bool upload_success = benchmark.benchmarkVideoUpload(state, task_id);

		if (!upload_success)
		{
			state.ResumeTiming();
			continue;
		}

		state.ResumeTiming();

		bool processing_success = benchmark.waitForTaskCompletion(task_id);

		if (!processing_success)
		{
			state.SkipWithError("Video processing failed or timed out");
		}
	}

	state.counters["Throughput"] = benchmark::Counter(state.iterations(),
													  benchmark::Counter::kIsRate, benchmark::Counter::OneK::kIs1000);
}

// Benchmark for concurrent image uploads
void BM_ConcurrentImageUploads(benchmark::State &state)
{
	const int num_concurrent = state.range(0);

	EmotionServerBenchmark::BenchmarkConfig config;
	config.test_image_path = "assets/test_image.jpg";
	config.test_video_path = "assets/test_video.mp4";

	std::vector<std::unique_ptr<EmotionServerBenchmark>> benchmarks;
	for (int i = 0; i < num_concurrent; ++i)
	{
		benchmarks.push_back(std::make_unique<EmotionServerBenchmark>(config));
	}

	std::atomic<int> completed_tasks{0};
	std::atomic<int> failed_tasks{0};

	for (auto _ : state)
	{
		completed_tasks = 0;
		failed_tasks = 0;

		state.PauseTiming();

		std::vector<std::future<bool>> futures;
		std::vector<std::string> task_ids(num_concurrent);

		// Start all uploads
		for (int i = 0; i < num_concurrent; ++i)
		{
			futures.push_back(std::async(std::launch::async,
										 [&, i]() -> bool
										 {
											 return benchmarks[i]->benchmarkImageUpload(state, task_ids[i]);
										 }));
		}

		// Wait for all uploads to complete
		bool all_uploads_success = true;
		for (auto &future : futures)
		{
			if (!future.get())
			{
				all_uploads_success = false;
				failed_tasks++;
			}
		}

		if (!all_uploads_success)
		{
			state.ResumeTiming();
			state.SkipWithError("Some image uploads failed");
			continue;
		}

		state.ResumeTiming();

		// Wait for all processing to complete concurrently
		std::vector<std::future<bool>> processing_futures;
		for (int i = 0; i < num_concurrent; ++i)
		{
			processing_futures.push_back(std::async(std::launch::async,
													[&, i]() -> bool
													{
														bool success = benchmarks[i]->waitForTaskCompletion(task_ids[i]);
														if (success)
															completed_tasks++;
														return success;
													}));
		}

		// Wait for all processing to complete
		for (auto &future : processing_futures)
		{
			future.get();
		}

		if (completed_tasks < num_concurrent)
		{
			state.SkipWithError(("Only " + std::to_string(completed_tasks.load()) +
								 "/" + std::to_string(num_concurrent) + " tasks completed")
									.c_str());
		}
	}

	state.counters["ConcurrentTasks"] = num_concurrent;
	state.counters["SuccessRate"] = benchmark::Counter(completed_tasks.load(),
													   benchmark::Counter::kAvgThreads, benchmark::Counter::OneK::kIs1000);
}

// Benchmark for concurrent video uploads
void BM_ConcurrentVideoUploads(benchmark::State &state)
{
	const int num_concurrent = state.range(0);

	EmotionServerBenchmark::BenchmarkConfig config;
	config.test_image_path = "assets/test_image.jpg";
	config.test_video_path = "assets/test_video.mp4";

	std::vector<std::unique_ptr<EmotionServerBenchmark>> benchmarks;
	for (int i = 0; i < num_concurrent; ++i)
	{
		benchmarks.push_back(std::make_unique<EmotionServerBenchmark>(config));
	}

	std::atomic<int> completed_tasks{0};
	std::atomic<int> failed_tasks{0};

	for (auto _ : state)
	{
		completed_tasks = 0;
		failed_tasks = 0;

		state.PauseTiming();

		std::vector<std::future<bool>> futures;
		std::vector<std::string> task_ids(num_concurrent);

		// Start all uploads
		for (int i = 0; i < num_concurrent; ++i)
		{
			futures.push_back(std::async(std::launch::async,
										 [&, i]() -> bool
										 {
											 return benchmarks[i]->benchmarkVideoUpload(state, task_ids[i]);
										 }));
		}

		// Wait for all uploads to complete
		bool all_uploads_success = true;
		for (auto &future : futures)
		{
			if (!future.get())
			{
				all_uploads_success = false;
				failed_tasks++;
			}
		}

		if (!all_uploads_success)
		{
			state.ResumeTiming();
			state.SkipWithError("Some video uploads failed");
			continue;
		}

		state.ResumeTiming();

		// Wait for all processing to complete concurrently
		std::vector<std::future<bool>> processing_futures;
		for (int i = 0; i < num_concurrent; ++i)
		{
			processing_futures.push_back(std::async(std::launch::async,
													[&, i]() -> bool
													{
														bool success = benchmarks[i]->waitForTaskCompletion(task_ids[i]);
														if (success)
															completed_tasks++;
														return success;
													}));
		}

		// Wait for all processing to complete
		for (auto &future : processing_futures)
		{
			future.get();
		}

		if (completed_tasks < num_concurrent)
		{
			state.SkipWithError(("Only " + std::to_string(completed_tasks.load()) +
								 "/" + std::to_string(num_concurrent) + " tasks completed")
									.c_str());
		}
	}

	state.counters["ConcurrentTasks"] = num_concurrent;
	state.counters["SuccessRate"] = benchmark::Counter(completed_tasks.load(),
													   benchmark::Counter::kAvgThreads, benchmark::Counter::OneK::kIs1000);
}