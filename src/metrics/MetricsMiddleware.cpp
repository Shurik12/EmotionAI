#include "MetricsMiddleware.h"
#include "MetricsCollector.h"

std::function<void(const std::string &, const std::string &, int, double)>
MetricsMiddleware::createRequestMetrics()
{
	return [](const std::string &method, const std::string &endpoint,
			  int status_code, double duration_seconds)
	{
		MetricsCollector::instance().recordRequest(method, endpoint, status_code, duration_seconds);
	};
}

std::function<void()> MetricsMiddleware::createConnectionMetricsIncrement()
{
	return []()
	{
		MetricsCollector::instance().incrementActiveConnections();
	};
}

std::function<void()> MetricsMiddleware::createConnectionMetricsDecrement()
{
	return []()
	{
		MetricsCollector::instance().decrementActiveConnections();
	};
}

std::function<void(const std::string &)> MetricsMiddleware::createTaskMetricsIncrement(const std::string &task_type)
{
	return [task_type](const std::string &)
	{
		MetricsCollector::instance().incrementTaskCount(task_type);
	};
}

std::function<void(const std::string &)> MetricsMiddleware::createTaskMetricsDecrement(const std::string &task_type)
{
	return [task_type](const std::string &)
	{
		MetricsCollector::instance().decrementTaskCount(task_type);
	};
}

std::function<void(const std::string &, double)> MetricsMiddleware::createTaskDurationMetrics(const std::string &task_type)
{
	return [task_type](const std::string &, double duration_seconds)
	{
		MetricsCollector::instance().recordTaskDuration(task_type, duration_seconds);
	};
}