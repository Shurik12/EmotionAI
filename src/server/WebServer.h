#pragma once

#include <common/httplib.h>
#include <server/BaseServer.h>

class WebServer : public BaseServer
{
public:
	explicit WebServer();
	~WebServer() override;

	void initialize() override;
	void start() override;
	void stop() noexcept override;

private:
	httplib::Server svr_;

	void setupRoutes();

	// Route handlers (thin wrappers around common implementations)
	void handleUpload(const httplib::Request &req, httplib::Response &res);
	void handleUploadRealtime(const httplib::Request &req, httplib::Response &res);
	void handleProgress(const httplib::Request &req, httplib::Response &res, const std::string &task_id);
	void handleBatchProgress(const httplib::Request &req, httplib::Response &res);
	void handleSubmitApplication(const httplib::Request &req, httplib::Response &res);
	void handleServeResult(const httplib::Request &req, httplib::Response &res, const std::string &filename);
	void handleHealthCheck(const httplib::Request &req, httplib::Response &res);
	void handleServeStatic(const httplib::Request &req, httplib::Response &res, const std::string &filename);
	void handleServeReactFile(const httplib::Request &req, httplib::Response &res, const std::string &filename);
	void handleRoot(const httplib::Request &req, httplib::Response &res);
};