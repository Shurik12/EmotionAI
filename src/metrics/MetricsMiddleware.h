#pragma once

#include <string>
#include <chrono>
#include <functional>

class MetricsMiddleware
{
public:
	static std::function<void(const std::string &, const std::string &, int, double)>
	createRequestMetrics();

	static std::function<void()> createConnectionMetricsIncrement();
	static std::function<void()> createConnectionMetricsDecrement();

	static std::function<void(const std::string &)> createTaskMetricsIncrement(const std::string &task_type);
	static std::function<void(const std::string &)> createTaskMetricsDecrement(const std::string &task_type);
	static std::function<void(const std::string &, double)> createTaskDurationMetrics(const std::string &task_type);
};