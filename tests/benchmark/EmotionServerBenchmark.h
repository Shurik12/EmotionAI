#pragma once

#include <benchmark/benchmark.h>
#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <client/Client.h>

class EmotionServerBenchmark
{
public:
	struct BenchmarkConfig
	{
		std::string server_host = "localhost";
		int server_port = 8080;
		std::string test_image_path;
		std::string test_video_path;
		int timeout_seconds = 120;
		int max_retries = 60;
		int retry_delay_ms = 1000;
	};

	EmotionServerBenchmark(const BenchmarkConfig &config);
	~EmotionServerBenchmark();

	// Single request benchmarks
	bool benchmarkImageUpload(benchmark::State &state, std::string &task_id);
	bool benchmarkVideoUpload(benchmark::State &state, std::string &task_id);
	bool benchmarkRealtimeVideoUpload(benchmark::State &state, std::string &task_id);

	// Utility methods
	bool waitForTaskCompletion(const std::string &task_id);
	HttpClient::Response checkTaskProgress(const std::string &task_id);
	HttpClient::Response batchCheckProgress(const std::vector<std::string> &task_ids);

	// File reading
	std::string readTestImage() const;
	std::string readTestVideo() const;

	const BenchmarkConfig &getConfig() const { return config_; }

private:
	BenchmarkConfig config_;
	std::unique_ptr<HttpClient> client_;
	std::string test_image_data_;
	std::string test_video_data_;
	bool files_loaded_ = false;

	void loadTestFiles();
	bool isTaskComplete(const HttpClient::Response &response);
	std::string extractTaskId(const HttpClient::Response &response);
};