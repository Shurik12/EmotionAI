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
	metrics << "# HELP process_cpu_seconds_total Total user and system CPU time spent in seconds.\n";
	metrics << "# TYPE process_cpu_seconds_total counter\n";
	metrics << "process_cpu_seconds_total " << process_cpu_seconds_.load() << "\n";

	metrics << "# HELP process_resident_memory_bytes Resident memory size in bytes.\n";
	metrics << "# TYPE process_resident_memory_bytes gauge\n";
	metrics << "process_resident_memory_bytes " << process_memory_bytes_.load() << "\n";

	// Request metrics
	metrics << "# HELP http_requests_total Total number of HTTP requests.\n";
	metrics << "# TYPE http_requests_total counter\n";
	metrics << "http_requests_total " << total_requests_.load() << "\n";

	metrics << "# HELP http_active_connections Current number of active connections.\n";
	metrics << "# TYPE http_active_connections gauge\n";
	metrics << "http_active_connections " << active_connections_.load() << "\n";

	// Endpoint-specific metrics
	{
		std::lock_guard lock(metrics_mutex_);
		for (const auto &[endpoint, count] : endpoint_requests_)
		{
			metrics << "# HELP http_endpoint_requests_total Total requests per endpoint.\n";
			metrics << "# TYPE http_endpoint_requests_total counter\n";
			metrics << "http_endpoint_requests_total{endpoint=\"" << endpoint << "\"} " << count.load() << "\n";
		}

		// Status code metrics
		for (const auto &[code, count] : status_codes_)
		{
			metrics << "# HELP http_response_status_total Total responses by status code.\n";
			metrics << "# TYPE http_response_status_total counter\n";
			metrics << "http_response_status_total{code=\"" << code << "\"} " << count.load() << "\n";
		}

		// Request duration metrics
		for (const auto &[endpoint, duration] : request_durations_)
		{
			metrics << "# HELP http_request_duration_seconds Request duration in seconds.\n";
			metrics << "# TYPE http_request_duration_seconds gauge\n";
			metrics << "http_request_duration_seconds{endpoint=\"" << endpoint << "\"} " << duration.load() << "\n";
		}

		// Task metrics
		for (const auto &[task_type, count] : active_tasks_)
		{
			metrics << "# HELP active_tasks Current number of active tasks by type.\n";
			metrics << "# TYPE active_tasks gauge\n";
			metrics << "active_tasks{type=\"" << task_type << "\"} " << count.load() << "\n";
		}

		for (const auto &[task_type, count] : completed_tasks_)
		{
			metrics << "# HELP completed_tasks_total Total completed tasks by type.\n";
			metrics << "# TYPE completed_tasks_total counter\n";
			metrics << "completed_tasks_total{type=\"" << task_type << "\"} " << count.load() << "\n";
		}

		for (const auto &[task_type, duration] : task_durations_)
		{
			metrics << "# HELP task_duration_seconds Task duration in seconds.\n";
			metrics << "# TYPE task_duration_seconds gauge\n";
			metrics << "task_duration_seconds{type=\"" << task_type << "\"} " << duration.load() << "\n";
		}

		// Dragonfly metrics
		metrics << "# HELP dragonfly_connected Dragonfly connection status.\n";
		metrics << "# TYPE dragonfly_connected gauge\n";
		metrics << "dragonfly_connected " << (dragonfly_connected_.load() ? 1 : 0) << "\n";

		for (const auto &[operation, count] : dragonfly_operations_)
		{
			metrics << "# HELP dragonfly_operations_total Total Dragonfly operations by type.\n";
			metrics << "# TYPE dragonfly_operations_total counter\n";
			metrics << "dragonfly_operations_total{operation=\"" << operation << "\"} " << count.load() << "\n";
		}
	}

	return metrics.str();
}