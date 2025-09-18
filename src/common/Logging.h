#pragma once
#include <string>
#include <common/httplib.h>

namespace Common
{
	// Initialize the unified logging system
	void multi_sink_example(const std::string &log_file);

	// HTTP request/response logger
	void log_request_response(const httplib::Request &req, const httplib::Response &res);
}