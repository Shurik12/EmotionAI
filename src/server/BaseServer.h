#pragma once

#include <map>
#include <set>
#include <string>
#include <filesystem>
#include <functional>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>

#include <nlohmann/json.hpp>
#include <server/IServer.h>
#include <db/DragonflyManager.h>
#include <emotionai/FileProcessor.h>
#include <server/ThreadPool.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

// Forward declarations for cluster components
#ifdef WITH_CLUSTER
class ClusterManager;
class DistributedTaskManager;
#endif

class BaseServer : public IServer
{
public:
	virtual ~BaseServer() override = default;

protected:
	explicit BaseServer();

	// Common initialization methods
	void loadConfiguration();
	void ensureDirectoriesExist();
	void initializeComponents();

	// Cluster management
	void initializeCluster();
	void startClusterServices();
	void stopClusterServices();
	void registerInstance();
	void unregisterInstance();

	// Common route handler implementations - now returns task_id
	std::string handleUploadCommon(const std::string &file_content, const std::string &filename, bool realtime = false);
	std::string handleSubmitApplicationCommon(const std::string &body);
	void validateJsonDocument(const nlohmann::json &json);

	// Distributed task processing
	void startDistributedTaskWorkers();
	void stopDistributedTaskWorkers();
	void processDistributedTask(const nlohmann::json &task);

	// Common file serving
	std::string getMimeType(const std::string &filename) const;
	bool isApiEndpoint(const std::string &path) const;

	// Multipart form data parsing
	std::map<std::string, std::string> parseMultipartFormData(const std::string &body,
															  const std::string &boundary);
	std::string extractBoundary(const std::string &content_type);

	// Common components
	std::shared_ptr<DragonflyManager> dragonfly_manager_;
	std::unique_ptr<FileProcessor> file_processor_;
	std::unique_ptr<ThreadPool> thread_pool_;

	// Cluster components
	std::string instance_id_;

#ifdef WITH_CLUSTER
	std::unique_ptr<ClusterManager> cluster_manager_;
	std::unique_ptr<DistributedTaskManager> distributed_task_manager_;
#endif

	// Task worker threads
	std::vector<std::thread> task_worker_threads_;
	std::atomic<bool> workers_running_{false};

	// Common paths
	fs::path static_files_root_;
	fs::path upload_folder_;
	fs::path results_folder_;
	fs::path log_folder_;

	// Metrics
	std::string collectMetrics();
	void updateRequestMetrics(
		const std::string &method, const std::string &endpoint,
		int status_code, double duration_seconds);

	// Utility methods
	std::string generateInstanceId();

private:
	// Metrics counters
	std::atomic<size_t> total_requests_{0};
	std::atomic<size_t> active_connections_{0};
	std::map<std::string, std::atomic<size_t>> endpoint_requests_;
	std::map<int, std::atomic<size_t>> status_codes_;
	std::mutex metrics_mutex_;
};