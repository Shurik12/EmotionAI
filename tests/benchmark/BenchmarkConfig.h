#pragma once

#include <string>
#include <vector>

struct BenchmarkConfig
{
	struct
	{
		std::string host = "localhost";
		int port = 8081;
		int timeout_seconds = 30;
	} server;

	struct
	{
		std::string upload_dir = "test_uploads";
		std::string results_dir = "test_results";
		std::string static_dir = "test_static";
	} paths;

	struct
	{
		int warmup_iterations = 3;
		int measurement_iterations = 10;
		int max_duration_seconds = 300;
	} benchmark;

	struct
	{
		std::vector<int> client_counts = {10, 50, 100, 200};
		std::vector<int> requests_per_client = {50, 100, 200};
		std::vector<int> concurrent_connections = {100, 500, 1000, 2000};
	} load_levels;

	static BenchmarkConfig &instance()
	{
		static BenchmarkConfig config;
		return config;
	}
};