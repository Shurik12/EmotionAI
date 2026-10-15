#include "MetricsCollector.h"
#include <sys/resource.h>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <logging/Logger.h>

MetricsCollector &MetricsCollector::instance()
{
	static MetricsCollector instance;
	return instance;
}

namespace
{
	// A path segment is "dynamic" (an id/filename) when it looks like one:
	// a UUID/hex token, a filename with an extension, or a very long opaque
	// string. Static route names such as "upload_burnout" stay intact — the old
	// hyphen/underscore heuristic mangled them into ":id".
	bool looksLikeDynamicId(const std::string &segment)
	{
		if (segment.empty())
			return false;
		if (segment.find('.') != std::string::npos)
			return true; // filename
		if (segment.size() >= 16)
		{
			bool hexish = true;
			for (char c : segment)
			{
				if (!std::isxdigit(static_cast<unsigned char>(c)) && c != '-')
				{
					hexish = false;
					break;
				}
			}
			if (hexish)
				return true; // uuid / hex id
		}
		return segment.size() > 24;
	}
}

std::string MetricsCollector::normalizePath(const std::string &path)
{
	// Replace segments that look like dynamic IDs with ":id" so that
	// /api/results/abc-123-xyz and /api/results/def-456-uvw share one metric.
	std::string result;
	result.reserve(path.size());

	size_t start = (!path.empty() && path[0] == '/') ? 1 : 0;
	while (start <= path.size())
	{
		size_t slash = path.find('/', start);
		std::string segment = (slash == std::string::npos)
								  ? path.substr(start)
								  : path.substr(start, slash - start);

		result += '/';
		result += looksLikeDynamicId(segment) ? ":id" : segment;

		if (slash == std::string::npos)
			break;
		start = slash + 1;
	}

	return result.empty() ? "/" : result;
}

void MetricsCollector::recordRequest(const std::string &method, const std::string &endpoint,
									 int status_code, double duration_seconds)
{
	total_requests_++;

	std::string key = method + ":" + endpoint;

	{
		std::lock_guard lock(metrics_mutex_);
		endpoint_requests_[key]++;
		status_codes_[status_code]++;
		request_durations_[key].sum += duration_seconds;
		request_durations_[key].count++;
	}
}

void MetricsCollector::incrementActiveConnections()
{
	active_connections_++;
}

void MetricsCollector::decrementActiveConnections()
{
	active_connections_--;
}

void MetricsCollector::incrementTaskCount(const std::string &task_type)
{
	std::lock_guard lock(metrics_mutex_);
	active_tasks_[task_type]++;
}

void MetricsCollector::decrementTaskCount(const std::string &task_type)
{
	std::lock_guard lock(metrics_mutex_);
	if (active_tasks_[task_type] > 0)
	{
		active_tasks_[task_type]--;
	}
	completed_tasks_[task_type]++;
}

void MetricsCollector::recordTaskDuration(const std::string &task_type, double duration_seconds)
{
	std::lock_guard lock(metrics_mutex_);
	task_durations_[task_type] = duration_seconds;
}

void MetricsCollector::setDragonflyConnected(bool connected)
{
	dragonfly_connected_ = connected;
}

void MetricsCollector::recordDragonflyRequest(const std::string &operation, double duration_seconds)
{
	std::lock_guard lock(metrics_mutex_);
	dragonfly_operations_[operation]++;
	dragonfly_durations_[operation] = duration_seconds;
}

void MetricsCollector::recordAiCall(int prompt_tokens, int completion_tokens, int total_tokens)
{
	ai_calls_total_++;
	ai_prompt_tokens_total_ += prompt_tokens;
	ai_completion_tokens_total_ += completion_tokens;
	ai_total_tokens_total_ += total_tokens;
}

void MetricsCollector::updateSystemMetrics()
{
	struct rusage usage;
	if (getrusage(RUSAGE_SELF, &usage) == 0)
	{
		process_cpu_seconds_.store(
			usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1000000.0 +
			usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1000000.0);
		process_memory_bytes_.store(usage.ru_maxrss * 1024);
	}
}

std::string MetricsCollector::collectMetrics()
{
	updateSystemMetrics();

	std::stringstream metrics;

	// Process metrics
	metrics << "process_cpu_seconds_total " << process_cpu_seconds_.load() << "\n";
	metrics << "process_resident_memory_bytes " << process_memory_bytes_.load() << "\n";

	// Request metrics
	metrics << "http_requests_total " << total_requests_.load() << "\n";
	metrics << "http_active_connections " << active_connections_.load() << "\n";

	// Endpoint-specific metrics
	{
		std::lock_guard lock(metrics_mutex_);
		for (const auto &[endpoint, count] : endpoint_requests_)
		{
			metrics << "http_endpoint_requests_total{endpoint=\"" << endpoint << "\"} " << count.load() << "\n";
		}

		// Status code metrics
		for (const auto &[code, count] : status_codes_)
		{
			metrics << "http_response_status_total{code=\"" << code << "\"} " << count.load() << "\n";
		}

		// Request duration metrics (average, not last-write-wins)
		for (const auto &[endpoint, stats] : request_durations_)
		{
			size_t count = stats.count.load();
			if (count == 0)
				continue;
			double avg = stats.sum.load() / static_cast<double>(count);
			metrics << "http_request_duration_seconds_avg{endpoint=\"" << endpoint << "\"} " << avg << "\n";
			metrics << "http_request_duration_seconds_count{endpoint=\"" << endpoint << "\"} " << count << "\n";
		}

		// Task metrics
		for (const auto &[task_type, count] : active_tasks_)
		{
			metrics << "active_tasks{type=\"" << task_type << "\"} " << count.load() << "\n";
		}

		for (const auto &[task_type, count] : completed_tasks_)
		{
			metrics << "completed_tasks_total{type=\"" << task_type << "\"} " << count.load() << "\n";
		}

		for (const auto &[task_type, duration] : task_durations_)
		{
			metrics << "task_duration_seconds{type=\"" << task_type << "\"} " << duration.load() << "\n";
		}

		// Dragonfly metrics
		metrics << "dragonfly_connected " << (dragonfly_connected_.load() ? 1 : 0) << "\n";

		for (const auto &[operation, count] : dragonfly_operations_)
		{
			metrics << "dragonfly_operations_total{operation=\"" << operation << "\"} " << count.load() << "\n";
		}

		// AI (LLM) metrics
		metrics << "ai_calls_total " << ai_calls_total_.load() << "\n";
		metrics << "ai_prompt_tokens_total " << ai_prompt_tokens_total_.load() << "\n";
		metrics << "ai_completion_tokens_total " << ai_completion_tokens_total_.load() << "\n";
		metrics << "ai_total_tokens_total " << ai_total_tokens_total_.load() << "\n";
	}

	return metrics.str();
}