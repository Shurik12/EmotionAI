#pragma once

#include <map>
#include <set>
#include <string>
#include <filesystem>
#include <functional>
#include <memory>
#include <thread>
#include <mutex>
#include <sys/epoll.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <nlohmann/json.hpp>
#include <server/IServer.h>
#include <db/DragonflyManager.h>
#include <db/TaskBatcher.h>
#include <emotionai/FileProcessor.h>

class MultiplexingServer : public IServer
{
public:
	explicit MultiplexingServer();
	~MultiplexingServer() override;

	void initialize() override;
	void start() override;
	void stop() noexcept override;

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

	int server_fd_;
	int epoll_fd_;
	bool running_;
	std::shared_ptr<DragonflyManager> dragonfly_manager_;
	std::unique_ptr<FileProcessor> file_processor_;
	std::unique_ptr<TaskBatcher> task_batcher_;
	std::filesystem::path static_files_root_;
	std::filesystem::path upload_folder_;
	std::filesystem::path results_folder_;
	std::filesystem::path log_folder_;
	std::mutex task_mutex_;
	std::map<std::string, std::thread> background_threads_;
	std::map<int, std::shared_ptr<ClientContext>> clients_;

	// Route handlers
	std::map<std::string, std::function<void(const std::shared_ptr<ClientContext> &, const std::string &)>> post_routes_;
	std::map<std::string, std::function<void(const std::shared_ptr<ClientContext> &)>> get_routes_;
	std::map<std::string, std::function<void(const std::shared_ptr<ClientContext> &)>> options_routes_;

	void loadConfiguration();
	void setupRoutes();
	void ensureDirectoriesExist();
	void initializeComponents();
	void initializeTaskBatcher();

	// Server core methods
	void createSocket();
	void setupEpoll();
	void handleEvents();
	void acceptNewConnection();
	void handleClientData(int client_fd);
	void closeClient(int client_fd);
	void processRequest(const std::shared_ptr<ClientContext> &context);

	// HTTP processing
	void parseHttpRequest(const std::shared_ptr<ClientContext> &context);
	void sendResponse(int client_fd, int status_code, const std::string &content_type, const std::string &body);
	void sendFileResponse(int client_fd, const std::filesystem::path &file_path);
	std::string getMimeType(const std::string &filename);

	// Route handler implementations
	void handleUpload(const std::shared_ptr<ClientContext> &context, const std::string &body);
	void handleUploadRealtime(const std::shared_ptr<ClientContext> &context, const std::string &body);
	void handleProgress(const std::shared_ptr<ClientContext> &context);
	void handleBatchProgress(const std::shared_ptr<ClientContext> &context, const std::string &body);
	void handleSubmitApplication(const std::shared_ptr<ClientContext> &context, const std::string &body);
	void handleServeResult(const std::shared_ptr<ClientContext> &context);
	void handleHealthCheck(const std::shared_ptr<ClientContext> &context);
	void handleServeStatic(const std::shared_ptr<ClientContext> &context);
	void handleServeReactFile(const std::shared_ptr<ClientContext> &context);
	void handleRoot(const std::shared_ptr<ClientContext> &context);
	void handleOptions(const std::shared_ptr<ClientContext> &context);

	// Helper functions
	static bool isApiEndpoint(const std::string &path);
	void cleanupFinishedThreads();
	void removeBackgroundThread(const std::string &task_id);
	std::map<std::string, std::string> parseMultipartFormData(const std::string &body, const std::string &boundary);
	std::string extractBoundary(const std::string &content_type);

	// JSON validation
	static void validateJsonDocument(const nlohmann::json &json);
};