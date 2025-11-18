#include "EmotionServerBenchmark.h"
#include <atomic>
#include <thread>
#include <vector>
#include <future>
#include <random>

// Benchmark for mixed workload (images and videos)
void BM_MixedWorkload(benchmark::State &state)
{
	const int total_requests = state.range(0);
	const int image_ratio = 70; // 70% images, 30% videos

	EmotionServerBenchmark::BenchmarkConfig config;
	config.test_image_path = "assets/test_image.jpg";
	config.test_video_path = "assets/test_video.mp4";

	std::vector<std::unique_ptr<EmotionServerBenchmark>> benchmarks;
	for (int i = 0; i < total_requests; ++i)
	{
		benchmarks.push_back(std::make_unique<EmotionServerBenchmark>(config));
	}

	std::atomic<int> completed_tasks{0};
	std::atomic<int> image_tasks{0};
	std::atomic<int> video_tasks{0};

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(1, 100);

	for (auto _ : state)
	{
		completed_tasks = 0;
		image_tasks = 0;
		video_tasks = 0;

		state.PauseTiming();

		std::vector<std::future<bool>> upload_futures;
		std::vector<std::string> task_ids(total_requests);
		std::vector<bool> is_image_task(total_requests, false);

		// Start mixed uploads
		for (int i = 0; i < total_requests; ++i)
		{
			bool is_image = (dis(gen) <= image_ratio);
			is_image_task[i] = is_image;

			upload_futures.push_back(std::async(std::launch::async,
												[&, i, is_image]() -> bool
												{
													if (is_image)
													{
														image_tasks++;
														return benchmarks[i]->benchmarkImageUpload(state, task_ids[i]);
													}
													else
													{
														video_tasks++;
														return benchmarks[i]->benchmarkVideoUpload(state, task_ids[i]);
													}
												}));
		}

		// Wait for all uploads to complete
		bool all_uploads_success = true;
		for (auto &future : upload_futures)
		{
			if (!future.get())
			{
				all_uploads_success = false;
			}
		}

		if (!all_uploads_success)
		{
			state.ResumeTiming();
			state.SkipWithError("Some uploads failed in mixed workload");
			continue;
		}

		state.ResumeTiming();

		// Wait for all processing to complete concurrently
		std::vector<std::future<bool>> processing_futures;
		for (int i = 0; i < total_requests; ++i)
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

		if (completed_tasks < total_requests)
		{
			state.SkipWithError(("Only " + std::to_string(completed_tasks.load()) +
								 "/" + std::to_string(total_requests) + " tasks completed")
									.c_str());
		}
	}

	state.counters["TotalRequests"] = total_requests;
	state.counters["ImageTasks"] = benchmark::Counter(image_tasks.load(),
													  benchmark::Counter::kAvgThreads);
	state.counters["VideoTasks"] = benchmark::Counter(video_tasks.load(),
													  benchmark::Counter::kAvgThreads);
	state.counters["SuccessRate"] = benchmark::Counter(completed_tasks.load(),
													   benchmark::Counter::kAvgThreads, benchmark::Counter::OneK::kIs1000);
}

// Benchmark for task status polling performance
void BM_TaskStatusPolling(benchmark::State &state)
{
	EmotionServerBenchmark::BenchmarkConfig config;
	config.test_image_path = "assets/test_image.jpg";
	config.test_video_path = "assets/test_video.mp4";

	EmotionServerBenchmark benchmark(config);

	std::string task_id;
	if (!benchmark.benchmarkImageUpload(state, task_id))
	{
		state.SkipWithError("Failed to upload test image for polling benchmark");
		return;
	}

	// Give it a moment to start processing
	std::this_thread::sleep_for(std::chrono::milliseconds(100));

	for (auto _ : state)
	{
		auto response = benchmark.checkTaskProgress(task_id);

		if (!response.success())
		{
			state.SkipWithError("Task status polling failed");
		}

		// Parse response to ensure it's valid JSON
		try
		{
			auto json_data = nlohmann::json::parse(response.body);
			benchmark::DoNotOptimize(json_data);
		}
		catch (...)
		{
			state.SkipWithError("Invalid JSON in polling response");
		}
	}

	// Clean up - wait for task to complete
	benchmark.waitForTaskCompletion(task_id);
}

// Benchmark for batch progress checking
void BM_BatchProgressChecking(benchmark::State &state)
{
	const int batch_size = state.range(0);

	EmotionServerBenchmark::BenchmarkConfig config;
	config.test_image_path = "assets/test_image.jpg";
	config.test_video_path = "assets/test_video.mp4";

	EmotionServerBenchmark benchmark(config);

	// Create batch of tasks
	std::vector<std::string> task_ids;
	for (int i = 0; i < batch_size; ++i)
	{
		std::string task_id;
		if (benchmark.benchmarkImageUpload(state, task_id))
		{
			task_ids.push_back(task_id);
		}
		else
		{
			state.SkipWithError("Failed to create task for batch testing");
			return;
		}
	}

	if (task_ids.size() != batch_size)
	{
		state.SkipWithError("Failed to create required number of tasks for batch testing");
		return;
	}

	// Give tasks more time to start processing
	std::this_thread::sleep_for(std::chrono::milliseconds(500));

	for (auto _ : state)
	{
		auto response = benchmark.batchCheckProgress(task_ids);

		if (!response.success())
		{
			state.SkipWithError("Batch progress check failed");
			continue;
		}

		// Parse and validate response
		try
		{
			auto json_data = nlohmann::json::parse(response.body);

			// The batch endpoint might not return all tasks immediately
			// Just verify we got a valid JSON response with expected structure
			if (!json_data.is_object())
			{
				state.SkipWithError("Batch response is not a JSON object");
				continue;
			}

			// Check if we got at least some task statuses
			if (json_data.empty())
			{
				state.SkipWithError("Batch response is empty");
				continue;
			}

			benchmark::DoNotOptimize(json_data);
		}
		catch (const std::exception &e)
		{
			state.SkipWithError(("Invalid JSON in batch response: " + std::string(e.what())).c_str());
		}
	}

	// Clean up - wait for all tasks to complete
	for (const auto &task_id : task_ids)
	{
		benchmark.waitForTaskCompletion(task_id);
	}

	state.counters["BatchSize"] = batch_size;
}