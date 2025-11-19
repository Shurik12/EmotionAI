#pragma once

#include <sys/epoll.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <server/BaseServer.h>

class MultiplexingServer : public BaseServer
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

	int server_fd_{-1};
	int epoll_fd_{-1};
	bool running_{false};
	std::map<int, std::shared_ptr<ClientContext>> clients_;

	// Route handlers
	std::map<std::string, std::function<void(const std::shared_ptr<ClientContext> &, const std::string &)>> post_routes_;
	std::map<std::string, std::function<void(const std::shared_ptr<ClientContext> &)>> get_routes_;
	std::map<std::string, std::function<void(const std::shared_ptr<ClientContext> &)>> options_routes_;

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
	void handleProgress(const std::shared_ptr<ClientContext> &context);
	void handleBatchProgress(const std::shared_ptr<ClientContext> &context, const std::string &body);
	void handleSubmitApplication(const std::shared_ptr<ClientContext> &context, const std::string &body);
	void handleServeResult(const std::shared_ptr<ClientContext> &context);
	void handleHealthCheck(const std::shared_ptr<ClientContext> &context);
	void handleServeStatic(const std::shared_ptr<ClientContext> &context);
	void handleServeReactFile(const std::shared_ptr<ClientContext> &context);
	void handleRoot(const std::shared_ptr<ClientContext> &context);
	void handleOptions(const std::shared_ptr<ClientContext> &context);

	// Helper methods
	bool isStaticAsset(const std::string &path) const;
	bool isApiEndpoint(const std::string &path) const;
	void sendErrorResponse(int client_fd, int status_code, const std::string &message);
	void cleanupResources();
};