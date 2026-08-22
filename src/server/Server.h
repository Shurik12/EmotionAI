#pragma once

#include <sys/epoll.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
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
#include <common/httplib.h>
#include <db/DragonflyManager.h>
#include <emotionai/FileProcessor.h>
#include <server/ThreadPool.h>
#include <storage/FileStorage.h>
#include <storage/FileStorageFactory.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

// Forward declarations for cluster components
#ifdef WITH_CLUSTER
class ClusterManager;
class DistributedTaskManager;
#endif

class Server
{
public:
    explicit Server();
    ~Server();

    void initialize();
    void start();
    void stop() noexcept;

    // Non-copyable, non-movable
    Server(const Server &) = delete;
    Server &operator=(const Server &) = delete;
    Server(Server &&) = delete;
    Server &operator=(Server &&) = delete;

private:
    struct ClientContext
    {
        int fd;
        std::string buffer;
        std::string method;
        std::string path;
        std::map<std::string, std::string> headers;
        std::map<std::string, std::string> params;
        bool headers_complete;
        size_t content_length;

        ClientContext(int socket_fd) : fd(socket_fd), headers_complete(false), content_length(0) {}
    };

    int server_fd_{-1};
    int epoll_fd_{-1};
    bool running_{false};
    std::map<int, std::shared_ptr<ClientContext>> clients_;

    // Route handlers
    std::map<std::string, std::function<void(const std::shared_ptr<ClientContext> &, const std::string &)>> post_routes_;
    std::map<std::string, std::function<void(const std::shared_ptr<ClientContext> &)>> get_routes_;
    std::map<std::string, std::function<void(const std::shared_ptr<ClientContext> &)>> options_routes_;

    // Common components
    std::shared_ptr<DragonflyManager> dragonfly_manager_;
    std::unique_ptr<FileProcessor> file_processor_;
    std::unique_ptr<ThreadPool> thread_pool_;

    // Cluster components
    std::string instance_id_;

    // Storage components
    std::shared_ptr<FileStorage> file_storage_;

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

    // Core server methods
    void setupRoutes();
    void createSocket();
    void setupEpoll();
    void handleEvents();
    void acceptNewConnection();
    void handleClientData(int client_fd);
    void closeClient(int client_fd);

    // HTTP processing
    void processRequest(const std::shared_ptr<ClientContext> &context);
    void parseHttpRequest(const std::shared_ptr<ClientContext> &context);
    void sendHttpResponse(int client_fd, int status_code, const std::string &content_type, const std::string &body);
    void sendFileResponse(int client_fd, const fs::path &file_path);

    // Route handlers
    void handleUpload(const std::shared_ptr<ClientContext> &context, const std::string &body);
    void handleUploadRealtime(const std::shared_ptr<ClientContext> &context, const std::string &body);
    void handleUploadBurnout(const std::shared_ptr<ClientContext> &context, const std::string &body);
    void handleMetrics(const std::shared_ptr<ClientContext> &context);
    void handleProgress(const std::shared_ptr<ClientContext> &context);
    void handleBatchProgress(const std::shared_ptr<ClientContext> &context, const std::string &body);
    void handleSubmitApplication(const std::shared_ptr<ClientContext> &context, const std::string &body);
    void handleServeResult(const std::shared_ptr<ClientContext> &context);
    void handleStorageInfo(const std::shared_ptr<ClientContext> &context);
    void handleHealthCheck(const std::shared_ptr<ClientContext> &context);
    void handleServeStatic(const std::shared_ptr<ClientContext> &context);
    void handleServeReactFile(const std::shared_ptr<ClientContext> &context);
    void handleRoot(const std::shared_ptr<ClientContext> &context);
    void handleOptions(const std::shared_ptr<ClientContext> &context);

    // Burnout Route Handlers
    void handleBurnoutAnalyze(const std::shared_ptr<ClientContext> &context, const std::string &body);
    void handleBurnoutBaseline(const std::shared_ptr<ClientContext> &context, const std::string &body);
    void handleBurnoutBaselineGet(const std::shared_ptr<ClientContext> &context);

    // Common handlers
    std::string handleUploadCommon(const std::string &file_content, const std::string &filename, bool realtime = false);
    std::string handleUploadBurnoutCommon(const std::string &file_content, const std::string &filename);
    std::string handleSubmitApplicationCommon(const std::string &body);
    void validateJsonDocument(const nlohmann::json &json);

    // Distributed task processing
    void startDistributedTaskWorkers();
    void stopDistributedTaskWorkers();
    void processDistributedTask(const nlohmann::json &task);

    // Common file serving
    std::string getMimeType(const std::string &filename) const;
    bool isApiEndpoint(const std::string &path) const;
    bool isStaticAsset(const std::string &path) const;

    // Multipart form data parsing
    std::map<std::string, std::string> parseMultipartFormData(const std::string &body,
                                                              const std::string &boundary);
    std::string extractBoundary(const std::string &content_type);

    // Initialization methods
    void loadConfiguration();
    void ensureDirectoriesExist();
    void initializeComponents();
    void initializeStorage();

    // Cluster management
    void initializeCluster();
    void startClusterServices();
    void stopClusterServices();
    void registerInstance();
    void unregisterInstance();

    // Helper methods
    std::string generateInstanceId();
    void sendErrorResponse(int client_fd, int status_code, const std::string &message);
    void cleanupResources();

    // Metrics
    std::string collectMetrics();
    void updateRequestMetrics(const std::string &method, const std::string &endpoint,
                              int status_code, double duration_seconds);

    // Metrics counters
    std::atomic<size_t> total_requests_{0};
    std::atomic<size_t> active_connections_{0};
    std::map<std::string, std::atomic<size_t>> endpoint_requests_;
    std::map<int, std::atomic<size_t>> status_codes_;
    std::mutex metrics_mutex_;
};