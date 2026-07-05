#include "MetricsCollector.h"
#include <sys/resource.h>
#include <sstream>
#include <algorithm>
#include <logging/Logger.h>

MetricsCollector &MetricsCollector::instance()
{
	static MetricsCollector instance;
	return instance;
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
		request_durations_[key] = duration_seconds;
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

		// Request duration metrics
		for (const auto &[endpoint, duration] : request_durations_)
		{
			metrics << "http_request_duration_seconds{endpoint=\"" << endpoint << "\"} " << duration.load() << "\n";
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
	}

	return metrics.str();
}