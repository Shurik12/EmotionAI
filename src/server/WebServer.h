#pragma once

#include <map>
#include <set>
#include <string>
#include <filesystem>
#include <functional>
#include <memory>
#include <thread>
#include <mutex>

#include <common/httplib.h>
#include <nlohmann/json.hpp>
#include <server/IServer.h>

// Forward declarations
namespace db
{
	class RedisManager;
}

namespace EmotionAI
{
	class FileProcessor;
}

class WebServer : public IServer
{
public:
	explicit WebServer();
	~WebServer() override;

	void initialize() override;
	void start() override;
	void stop() noexcept override;

private:
	httplib::Server svr_;
	std::shared_ptr<db::RedisManager> redis_manager_;
	std::unique_ptr<EmotionAI::FileProcessor> file_processor_;
	std::filesystem::path static_files_root_;
	std::filesystem::path upload_folder_;
	std::filesystem::path results_folder_;
	std::filesystem::path log_folder_;
	std::mutex task_mutex_;
	std::map<std::string, std::thread> background_threads_;
	std::condition_variable completion_cv_;
    
    void waitForCompletion();

	void loadConfiguration();
	void setupRoutes();
	void ensureDirectoriesExist();
	void initializeComponents();

	// Route handlers
	void handleUpload(const httplib::Request &req, httplib::Response &res);
	void removeBackgroundThread(const std::string& task_id);
	void handleProgress(const httplib::Request &req, httplib::Response &res, const std::string &task_id);
	void handleSubmitApplication(const httplib::Request &req, httplib::Response &res);
	void handleServeResult(const httplib::Request &req, httplib::Response &res, const std::string &filename);
	void handleHealthCheck(const httplib::Request &req, httplib::Response &res);
	void handleServeStatic(const httplib::Request &req, httplib::Response &res, const std::string &filename);
	void handleServeReactFile(const httplib::Request &req, httplib::Response &res, const std::string &filename);
	void handleRoot(const httplib::Request &req, httplib::Response &res);

	// Helper functions
	static bool isApiEndpoint(const std::string &path);
	void cleanupFinishedThreads();

	// JSON validation
	static void validateJsonDocument(const nlohmann::json &json);
};