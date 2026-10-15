#pragma once

#include <atomic>
#include <map>
#include <string>
#include <mutex>
#include <chrono>
#include <functional>

class MetricsCollector
{
public:
	static MetricsCollector &instance();

	// Request metrics. `endpoint` should already be normalized (parameterized
	// segments replaced with ":param") to avoid unbounded label cardinality.
	void recordRequest(const std::string &method, const std::string &endpoint,
					   int status_code, double duration_seconds);

	// Replace path segments that look like IDs (UUIDs, long hex, filenames)
	// with a generic ":id" so parameterized routes don't create a new metric
	// entry per unique ID.
	static std::string normalizePath(const std::string &path);

	// Connection metrics
	void incrementActiveConnections();
	void decrementActiveConnections();

	// Task metrics
	void incrementTaskCount(const std::string &task_type);
	void decrementTaskCount(const std::string &task_type);
	void recordTaskDuration(const std::string &task_type, double duration_seconds);

	// Dragonfly metrics
	void setDragonflyConnected(bool connected);
	void recordDragonflyRequest(const std::string &operation, double duration_seconds);

	// AI (LLM) metrics
	void recordAiCall(int prompt_tokens, int completion_tokens, int total_tokens);

	// System metrics
	void updateSystemMetrics();

	// Getters for metrics data
	size_t getTotalRequests() const { return total_requests_.load(); }
	size_t getActiveConnections() const { return active_connections_.load(); }

	// Export metrics in Prometheus format
	std::string collectMetrics();

private:
	MetricsCollector() = default;
	~MetricsCollector() = default;

	// Request metrics
	std::atomic<size_t> total_requests_{0};
	std::atomic<size_t> active_connections_{0};
	std::map<std::string, std::atomic<size_t>> endpoint_requests_;
	std::map<int, std::atomic<size_t>> status_codes_;
	// Duration: sum and count per endpoint so the export can report an
	// average instead of a last-write-wins gauge.
	struct DurationStats {
		std::atomic<double> sum{0.0};
		std::atomic<size_t> count{0};
	};
	std::map<std::string, DurationStats> request_durations_;

	// Task metrics
	std::map<std::string, std::atomic<size_t>> active_tasks_;
	std::map<std::string, std::atomic<size_t>> completed_tasks_;
	std::map<std::string, std::atomic<double>> task_durations_;

	// System metrics
	std::atomic<double> process_cpu_seconds_{0.0};
	std::atomic<size_t> process_memory_bytes_{0};

	// Dragonfly metrics
	std::atomic<bool> dragonfly_connected_{false};
	std::map<std::string, std::atomic<size_t>> dragonfly_operations_;
	std::map<std::string, std::atomic<double>> dragonfly_durations_;

	// AI (LLM) metrics
	std::atomic<size_t> ai_calls_total_{0};
	std::atomic<size_t> ai_prompt_tokens_total_{0};
	std::atomic<size_t> ai_completion_tokens_total_{0};
	std::atomic<size_t> ai_total_tokens_total_{0};

	mutable std::mutex metrics_mutex_;
};