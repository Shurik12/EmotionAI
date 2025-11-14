// File: MultiplexingServer.cpp (refactored)
#include <filesystem>
#include <fstream>
#include <vector>
#include <utility>
#include <thread>
#include <mutex>
#include <sstream>
#include <algorithm>
#include <cstring>

#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <server/MultiplexingServer.h>
#include <config/Config.h>
#include <logging/Logger.h>
#include <common/uuid.h>
#include <db/TaskManager.h>
#include <emotionai/FileProcessor.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

MultiplexingServer::MultiplexingServer()
	: server_fd_(-1), epoll_fd_(-1), running_(false)
{
}

MultiplexingServer::~MultiplexingServer()
{
	try
	{
		stop();

		if (epoll_fd_ != -1)
		{
			close(epoll_fd_);
		}
		if (server_fd_ != -1)
		{
			close(server_fd_);
		}
	}
	catch (...)
	{
		LOG_ERROR("Exception during server shutdown");
	}
}

void MultiplexingServer::initialize()
{
	LOG_INFO("MultiplexingServer::initialize");
	loadConfiguration();
	ensureDirectoriesExist();
	initializeComponents();
	setupRoutes();
	createSocket();
	setupEpoll();
}

void MultiplexingServer::start()
{
	auto &config = Config::instance();
	std::string host = config.server().host;
	int port = config.server().port;

	LOG_INFO("Starting multiplexing server on {}:{}", host, port);

	running_ = true;
	handleEvents();
}

void MultiplexingServer::stop() noexcept
{
	try
	{
		running_ = false;
	}
	catch (const std::exception &e)
	{
		spdlog::error("Error stopping server: {}", e.what());
	}
}

void MultiplexingServer::setupRoutes()
{
	// POST routes
	post_routes_["/api/upload"] = [this](const std::shared_ptr<ClientContext> &context, const std::string &body)
	{
		handleUpload(context, body);
	};

	post_routes_["/api/upload_realtime"] = [this](const std::shared_ptr<ClientContext> &context, const std::string &body)
	{
		handleUploadRealtime(context, body);
	};

	post_routes_["/api/submit_application"] = [this](const std::shared_ptr<ClientContext> &context, const std::string &body)
	{
		handleSubmitApplication(context, body);
	};

	post_routes_["/api/batch_progress"] = [this](const std::shared_ptr<ClientContext> &context, const std::string &body)
	{
		handleBatchProgress(context, body);
	};

	// GET routes
	get_routes_["/api/progress"] = [this](const std::shared_ptr<ClientContext> &context)
	{
		handleProgress(context);
	};

	get_routes_["/api/results"] = [this](const std::shared_ptr<ClientContext> &context)
	{
		handleServeResult(context);
	};

	get_routes_["/api/health"] = [this](const std::shared_ptr<ClientContext> &context)
	{
		handleHealthCheck(context);
	};

	get_routes_["/static"] = [this](const std::shared_ptr<ClientContext> &context)
	{
		handleServeStatic(context);
	};

	get_routes_["/"] = [this](const std::shared_ptr<ClientContext> &context)
	{
		if (context->path == "/")
		{
			handleRoot(context);
		}
		else
		{
			handleServeReactFile(context);
		}
	};

	// OPTIONS routes (for CORS)
	options_routes_["/api/upload"] = [this](const std::shared_ptr<ClientContext> &context)
	{
		handleOptions(context);
	};
	options_routes_["/api/upload_realtime"] = [this](const std::shared_ptr<ClientContext> &context)
	{
		handleOptions(context);
	};
	options_routes_["/api/submit_application"] = [this](const std::shared_ptr<ClientContext> &context)
	{
		handleOptions(context);
	};
	options_routes_["/api/progress"] = [this](const std::shared_ptr<ClientContext> &context)
	{
		handleOptions(context);
	};
	options_routes_["/api/results"] = [this](const std::shared_ptr<ClientContext> &context)
	{
		handleOptions(context);
	};
	options_routes_["/api/health"] = [this](const std::shared_ptr<ClientContext> &context)
	{
		handleOptions(context);
	};
	options_routes_["/static"] = [this](const std::shared_ptr<ClientContext> &context)
	{
		handleOptions(context);
	};
}

void MultiplexingServer::createSocket()
{
	server_fd_ = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
	if (server_fd_ == -1)
	{
		throw std::runtime_error("Failed to create socket");
	}

	int opt = 1;
	if (setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0)
	{
		close(server_fd_);
		throw std::runtime_error("Failed to set socket options");
	}

	auto &config = Config::instance();
	std::string host = config.server().host;
	int port = config.server().port;

	struct sockaddr_in address;
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = inet_addr(host.c_str());
	address.sin_port = htons(port);

	if (bind(server_fd_, (struct sockaddr *)&address, sizeof(address)) < 0)
	{
		close(server_fd_);
		throw std::runtime_error("Failed to bind socket");
	}

	if (listen(server_fd_, SOMAXCONN) < 0)
	{
		close(server_fd_);
		throw std::runtime_error("Failed to listen on socket");
	}
}

void MultiplexingServer::setupEpoll()
{
	epoll_fd_ = epoll_create1(0);
	if (epoll_fd_ == -1)
	{
		close(server_fd_);
		throw std::runtime_error("Failed to create epoll instance");
	}

	struct epoll_event event;
	event.events = EPOLLIN;
	event.data.fd = server_fd_;

	if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, server_fd_, &event) == -1)
	{
		close(epoll_fd_);
		close(server_fd_);
		throw std::runtime_error("Failed to add server socket to epoll");
	}
}

void MultiplexingServer::handleEvents()
{
	const int MAX_EVENTS = 64;
	struct epoll_event events[MAX_EVENTS];

	while (running_)
	{
		int num_events = epoll_wait(epoll_fd_, events, MAX_EVENTS, 100);

		if (num_events == -1)
		{
			if (errno == EINTR)
				continue;
			spdlog::error("epoll_wait error: {}", strerror(errno));
			break;
		}

		for (int i = 0; i < num_events; ++i)
		{
			if (events[i].data.fd == server_fd_)
			{
				acceptNewConnection();
			}
			else
			{
				handleClientData(events[i].data.fd);
			}
		}
	}
}

void MultiplexingServer::acceptNewConnection()
{
	struct sockaddr_in client_addr;
	socklen_t client_len = sizeof(client_addr);

	int client_fd = accept4(server_fd_, (struct sockaddr *)&client_addr, &client_len, SOCK_NONBLOCK);
	if (client_fd == -1)
	{
		spdlog::error("Failed to accept connection: {}", strerror(errno));
		return;
	}

	auto context = std::make_shared<ClientContext>(client_fd);
	clients_[client_fd] = context;

	struct epoll_event event;
	event.events = EPOLLIN | EPOLLET;
	event.data.fd = client_fd;

	if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, client_fd, &event) == -1)
	{
		spdlog::error("Failed to add client to epoll: {}", strerror(errno));
		closeClient(client_fd);
		return;
	}

	LOG_DEBUG("New client connected: fd={}", client_fd);
}

void MultiplexingServer::handleClientData(int client_fd)
{
	auto it = clients_.find(client_fd);
	if (it == clients_.end())
	{
		return;
	}

	auto context = it->second;
	char buffer[4096];

	while (true)
	{
		ssize_t bytes_read = recv(client_fd, buffer, sizeof(buffer) - 1, 0);

		if (bytes_read == -1)
		{
			if (errno == EAGAIN || errno == EWOULDBLOCK)
			{
				break;
			}
			else
			{
				spdlog::error("Error reading from client {}: {}", client_fd, strerror(errno));
				closeClient(client_fd);
				return;
			}
		}
		else if (bytes_read == 0)
		{
			closeClient(client_fd);
			return;
		}
		else
		{
			buffer[bytes_read] = '\0';
			context->buffer.append(buffer, bytes_read);

			if (!context->headers_complete)
			{
				parseHttpRequest(context);
			}

			if (context->headers_complete && context->buffer.length() >= context->content_length)
			{
				processRequest(context);
				break;
			}
		}
	}
}

void MultiplexingServer::parseHttpRequest(const std::shared_ptr<ClientContext> &context)
{
	size_t header_end = context->buffer.find("\r\n\r\n");
	if (header_end == std::string::npos)
	{
		return;
	}

	std::string headers_str = context->buffer.substr(0, header_end);
	std::istringstream headers_stream(headers_str);
	std::string line;

	if (std::getline(headers_stream, line))
	{
		std::istringstream request_line(line);
		request_line >> context->method >> context->path;
		std::transform(context->method.begin(), context->method.end(), context->method.begin(), ::toupper);
	}

	while (std::getline(headers_stream, line))
	{
		if (line.back() == '\r')
			line.pop_back();
		if (line.empty())
			continue;

		size_t colon_pos = line.find(':');
		if (colon_pos != std::string::npos)
		{
			std::string key = line.substr(0, colon_pos);
			std::string value = line.substr(colon_pos + 1);
			value.erase(0, value.find_first_not_of(" \t"));
			value.erase(value.find_last_not_of(" \t") + 1);
			context->headers[key] = value;
		}
	}

	auto content_length_it = context->headers.find("Content-Length");
	if (content_length_it != context->headers.end())
	{
		context->content_length = std::stoul(content_length_it->second);
	}

	size_t query_pos = context->path.find('?');
	if (query_pos != std::string::npos)
	{
		std::string query_str = context->path.substr(query_pos + 1);
		context->path = context->path.substr(0, query_pos);

		std::istringstream query_stream(query_str);
		std::string param;
		while (std::getline(query_stream, param, '&'))
		{
			size_t equal_pos = param.find('=');
			if (equal_pos != std::string::npos)
			{
				std::string key = param.substr(0, equal_pos);
				std::string value = param.substr(equal_pos + 1);
				context->params[key] = value;
			}
		}
	}

	context->headers_complete = true;
	context->buffer = context->buffer.substr(header_end + 4);
}

void MultiplexingServer::processRequest(const std::shared_ptr<ClientContext> &context)
{
	LOG_DEBUG("Processing: {} {}", context->method, context->path);

	try
	{
		if (context->method == "OPTIONS")
		{
			auto it = options_routes_.find(context->path);
			if (it != options_routes_.end())
			{
				it->second(context);
			}
			else
			{
				sendResponse(context->fd, 404, "application/json", R"({"error": "Not found"})");
			}
		}
		else if (context->method == "POST")
		{
			std::string body = context->buffer.substr(0, context->content_length);

			auto it = post_routes_.find(context->path);
			if (it != post_routes_.end())
			{
				it->second(context, body);
			}
			else
			{
				sendResponse(context->fd, 404, "application/json", R"({"error": "Not found"})");
			}
		}
		else if (context->method == "GET")
		{
			if (context->path.find("/static/") == 0)
			{
				context->params["filename"] = context->path.substr(8);
				handleServeStatic(context);
				return;
			}
			else if (context->path.find("/api/progress/") == 0)
			{
				context->params["task_id"] = context->path.substr(14);
				handleProgress(context);
				return;
			}
			else if (context->path.find("/api/results/") == 0)
			{
				context->params["filename"] = context->path.substr(13);
				handleServeResult(context);
				return;
			}
			else if (context->path.find("/api/health") == 0)
			{
				handleHealthCheck(context);
				return;
			}

			auto it = get_routes_.find(context->path);
			if (it != get_routes_.end())
			{
				it->second(context);
				return;
			}

			if (!isApiEndpoint(context->path))
			{
				handleServeReactFile(context);
				return;
			}

			sendResponse(context->fd, 404, "application/json", R"({"error": "Not found"})");
		}
		else
		{
			sendResponse(context->fd, 405, "application/json", R"({"error": "Method not allowed"})");
		}
	}
	catch (const std::exception &e)
	{
		spdlog::error("Error processing request: {}", e.what());
		sendResponse(context->fd, 500, "application/json", R"({"error": "Internal server error"})");
	}
}

void MultiplexingServer::sendResponse(int client_fd, int status_code, const std::string &content_type, const std::string &body)
{
	std::string status_text;
	switch (status_code)
	{
	case 200:
		status_text = "OK";
		break;
	case 201:
		status_text = "Created";
		break;
	case 202:
		status_text = "Accepted";
		break;
	case 400:
		status_text = "Bad Request";
		break;
	case 404:
		status_text = "Not Found";
		break;
	case 405:
		status_text = "Method Not Allowed";
		break;
	case 500:
		status_text = "Internal Server Error";
		break;
	default:
		status_text = "Unknown";
		break;
	}

	std::string response = fmt::format(
		"HTTP/1.1 {} {}\r\n"
		"Content-Type: {}\r\n"
		"Access-Control-Allow-Origin: *\r\n"
		"Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS\r\n"
		"Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
		"Content-Length: {}\r\n"
		"Connection: close\r\n"
		"\r\n"
		"{}",
		status_code, status_text, content_type, body.length(), body);

	ssize_t bytes_sent = send(client_fd, response.c_str(), response.length(), 0);
	if (bytes_sent == -1)
	{
		spdlog::error("Failed to send response to client {}: {}", client_fd, strerror(errno));
	}
	else
	{
		spdlog::debug("Sent {} bytes to client {}", bytes_sent, client_fd);
	}
}

void MultiplexingServer::sendFileResponse(int client_fd, const fs::path &file_path)
{
	std::ifstream file(file_path, std::ios::binary | std::ios::ate);
	if (!file)
	{
		sendResponse(client_fd, 404, "text/plain", "File not found");
		return;
	}

	std::streamsize size = file.tellg();
	file.seekg(0, std::ios::beg);

	std::vector<char> buffer(size);
	if (!file.read(buffer.data(), size))
	{
		sendResponse(client_fd, 500, "text/plain", "Error reading file");
		return;
	}

	std::string mime_type = getMimeType(file_path.string());
	std::string headers = fmt::format(
		"HTTP/1.1 200 OK\r\n"
		"Content-Type: {}\r\n"
		"Access-Control-Allow-Origin: *\r\n"
		"Content-Length: {}\r\n"
		"Connection: close\r\n"
		"\r\n",
		mime_type, size);

	send(client_fd, headers.c_str(), headers.length(), 0);
	send(client_fd, buffer.data(), size, 0);
}

void MultiplexingServer::closeClient(int client_fd)
{
	auto it = clients_.find(client_fd);
	if (it != clients_.end())
	{
		epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, client_fd, nullptr);
		close(client_fd);
		clients_.erase(it);
		LOG_DEBUG("Client disconnected: fd={}", client_fd);
	}
}

void MultiplexingServer::handleUpload(const std::shared_ptr<ClientContext> &context, const std::string &body)
{
	try
	{
		auto content_type_it = context->headers.find("Content-Type");
		if (content_type_it == context->headers.end())
		{
			sendResponse(context->fd, 400, "application/json", R"({"error": "No Content-Type header"})");
			closeClient(context->fd);
			return;
		}

		if (content_type_it->second.find("multipart/form-data") == std::string::npos)
		{
			sendResponse(context->fd, 400, "application/json", R"({"error": "Expected multipart form data"})");
			closeClient(context->fd);
			return;
		}

		std::string boundary = extractBoundary(content_type_it->second);
		if (boundary.empty())
		{
			sendResponse(context->fd, 400, "application/json", R"({"error": "Invalid multipart data"})");
			closeClient(context->fd);
			return;
		}

		auto form_data = parseMultipartFormData(body, boundary);
		auto file_it = form_data.find("file");
		if (file_it == form_data.end())
		{
			sendResponse(context->fd, 400, "application/json", R"({"error": "No file provided"})");
			closeClient(context->fd);
			return;
		}

		const std::string &file_content = file_it->second;
		if (file_content.empty())
		{
			sendResponse(context->fd, 400, "application/json", R"({"error": "No file selected or empty file"})");
			closeClient(context->fd);
			return;
		}

		std::string filename;
		auto filename_it = form_data.find("filename");
		if (filename_it != form_data.end() && !filename_it->second.empty())
		{
			filename = filename_it->second;
		}
		else
		{
			size_t filename_pos = body.find("filename=\"");
			if (filename_pos != std::string::npos)
			{
				filename_pos += 10;
				size_t filename_end = body.find("\"", filename_pos);
				if (filename_end != std::string::npos)
				{
					filename = body.substr(filename_pos, filename_end - filename_pos);
				}
			}
		}

		if (filename.empty())
		{
			filename = "uploaded_file";
		}

		if (!file_processor_ || !file_processor_->allowed_file(filename))
		{
			sendResponse(context->fd, 400, "application/json", R"({"error": "Invalid file type"})");
			closeClient(context->fd);
			return;
		}

		// Use the common upload handler and get the task_id
		std::string task_id = handleUploadCommon(file_content, filename, false);

		LOG_INFO("Upload accepted from client {}, task_id: {}", context->fd, task_id);
		sendResponse(context->fd, 202, "application/json",
					 fmt::format(R"({{"task_id": "{}"}})", task_id));

		closeClient(context->fd);
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Exception in handleUpload from client {}: {}", context->fd, e.what());
		sendResponse(context->fd, 500, "application/json", R"({"error": "Internal server error"})");
		closeClient(context->fd);
	}
}

void MultiplexingServer::handleUploadRealtime(const std::shared_ptr<ClientContext> &context, const std::string &body)
{
	try
	{
		auto content_type_it = context->headers.find("Content-Type");
		if (content_type_it == context->headers.end())
		{
			sendResponse(context->fd, 400, "application/json", R"({"error": "No Content-Type header"})");
			closeClient(context->fd);
			return;
		}

		if (content_type_it->second.find("multipart/form-data") == std::string::npos)
		{
			sendResponse(context->fd, 400, "application/json", R"({"error": "Expected multipart form data"})");
			closeClient(context->fd);
			return;
		}

		std::string boundary = extractBoundary(content_type_it->second);
		if (boundary.empty())
		{
			sendResponse(context->fd, 400, "application/json", R"({"error": "Invalid multipart data"})");
			closeClient(context->fd);
			return;
		}

		auto form_data = parseMultipartFormData(body, boundary);
		auto file_it = form_data.find("file");
		if (file_it == form_data.end())
		{
			sendResponse(context->fd, 400, "application/json", R"({"error": "No file provided"})");
			closeClient(context->fd);
			return;
		}

		const std::string &file_content = file_it->second;
		if (file_content.empty())
		{
			sendResponse(context->fd, 400, "application/json", R"({"error": "No file selected or empty file"})");
			closeClient(context->fd);
			return;
		}

		std::string filename;
		auto filename_it = form_data.find("filename");
		if (filename_it != form_data.end() && !filename_it->second.empty())
		{
			filename = filename_it->second;
		}
		else
		{
			size_t filename_pos = body.find("filename=\"");
			if (filename_pos != std::string::npos)
			{
				filename_pos += 10;
				size_t filename_end = body.find("\"", filename_pos);
				if (filename_end != std::string::npos)
				{
					filename = body.substr(filename_pos, filename_end - filename_pos);
				}
			}
		}

		if (filename.empty())
		{
			filename = "uploaded_file";
		}

		if (!file_processor_ || !file_processor_->allowed_file(filename))
		{
			sendResponse(context->fd, 400, "application/json", R"({"error": "Invalid file type"})");
			closeClient(context->fd);
			return;
		}

		// Use the common upload handler with realtime flag and get the task_id
		std::string task_id = handleUploadCommon(file_content, filename, true);

		LOG_INFO("Real-time upload accepted from client {}, task_id: {}", context->fd, task_id);
		sendResponse(context->fd, 202, "application/json",
					 fmt::format(R"({{"task_id": "{}", "mode": "realtime"}})", task_id));

		closeClient(context->fd);
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Exception in real-time upload from client {}: {}", context->fd, e.what());
		sendResponse(context->fd, 500, "application/json", R"({"error": "Internal server error"})");
		closeClient(context->fd);
	}
}

void MultiplexingServer::handleProgress(const std::shared_ptr<ClientContext> &context)
{
	try
	{
		std::string task_id;

		auto task_id_it = context->params.find("task_id");
		if (task_id_it != context->params.end())
		{
			task_id = task_id_it->second;
		}
		else
		{
			auto query_it = context->params.find("task_id");
			if (query_it != context->params.end())
			{
				task_id = query_it->second;
			}
		}

		if (task_id.empty())
		{
			sendResponse(context->fd, 400, "application/json", R"({"error": "Task ID required"})");
			return;
		}

		auto &task_manager = TaskManager::instance();
		auto status = task_manager.get_task_status(task_id);

		if (!status)
		{
			sendResponse(context->fd, 404, "application/json", R"({"error": "Task not found"})");
			return;
		}

		sendResponse(context->fd, 200, "application/json", status->dump());
	}
	catch (const std::exception &e)
	{
		spdlog::error("Exception in handleProgress: {}", e.what());
		sendResponse(context->fd, 500, "application/json", R"({"error": "Internal server error"})");
	}
}

void MultiplexingServer::handleBatchProgress(const std::shared_ptr<ClientContext> &context, const std::string &body)
{
	try
	{
		auto task_ids_json = nlohmann::json::parse(body);
		if (!task_ids_json.is_array())
		{
			sendResponse(context->fd, 400, "application/json", R"({"error": "Expected array of task IDs"})");
			return;
		}

		std::vector<std::string> task_ids;
		for (const auto &id : task_ids_json)
		{
			if (id.is_string())
			{
				task_ids.push_back(id.get<std::string>());
			}
		}

		if (task_ids.empty())
		{
			sendResponse(context->fd, 400, "application/json", R"({"error": "No task IDs provided"})");
			return;
		}

		auto &task_manager = TaskManager::instance();
		auto results = task_manager.batch_get_status(task_ids);

		nlohmann::json response;
		for (const auto &[task_id, status] : results)
		{
			response[task_id] = status;
		}

		sendResponse(context->fd, 200, "application/json", response.dump());
	}
	catch (const nlohmann::json::parse_error &e)
	{
		spdlog::error("JSON parse error in batch progress: {}", e.what());
		sendResponse(context->fd, 400, "application/json", R"({"error": "Invalid JSON"})");
	}
	catch (const std::exception &e)
	{
		spdlog::error("Exception in handleBatchProgress: {}", e.what());
		sendResponse(context->fd, 500, "application/json", R"({"error": "Internal server error"})");
	}
}

void MultiplexingServer::handleSubmitApplication(const std::shared_ptr<ClientContext> &context, const std::string &body)
{
	try
	{
		std::string application_id = handleSubmitApplicationCommon(body);

		sendResponse(context->fd, 201, "application/json",
					 fmt::format(R"({{"application_id": "{}"}})", application_id));
	}
	catch (const json::parse_error &e)
	{
		sendResponse(context->fd, 400, "application/json", R"({"error": "Invalid JSON"})");
	}
	catch (const std::exception &e)
	{
		spdlog::error("Error submitting application: {}", e.what());
		sendResponse(context->fd, 500, "application/json", R"({"error": "Internal server error"})");
	}
}

void MultiplexingServer::handleServeResult(const std::shared_ptr<ClientContext> &context)
{
	try
	{
		auto filename_it = context->params.find("filename");
		if (filename_it == context->params.end())
		{
			sendResponse(context->fd, 400, "text/plain", "Filename required");
			return;
		}

		fs::path file_path = results_folder_ / filename_it->second;
		if (!fs::exists(file_path) || !fs::is_regular_file(file_path))
		{
			sendResponse(context->fd, 404, "text/plain", "File not found");
			return;
		}

		sendFileResponse(context->fd, file_path);
	}
	catch (const std::exception &e)
	{
		spdlog::error("Exception serving result file: {}", e.what());
		sendResponse(context->fd, 500, "text/plain", "Internal server error");
	}
}

void MultiplexingServer::handleHealthCheck(const std::shared_ptr<ClientContext> &context)
{
	try
	{
		sendResponse(context->fd, 200, "application/json", R"({"status": "healthy"})");
	}
	catch (const std::exception &e)
	{
		spdlog::error("Health check failed: {}", e.what());
		sendResponse(context->fd, 500, "application/json",
					 fmt::format(R"({{"status": "unhealthy", "error": "{}"}})", e.what()));
	}
}

void MultiplexingServer::handleServeStatic(const std::shared_ptr<ClientContext> &context)
{
	try
	{
		auto filename_it = context->params.find("filename");
		if (filename_it == context->params.end())
		{
			sendResponse(context->fd, 404, "application/json", R"({"error": "File not found"})");
			return;
		}

		fs::path static_path = static_files_root_ / "static" / filename_it->second;
		if (fs::exists(static_path) && fs::is_regular_file(static_path))
		{
			sendFileResponse(context->fd, static_path);
			return;
		}

		sendResponse(context->fd, 404, "application/json", R"({"error": "File not found"})");
	}
	catch (const std::exception &e)
	{
		spdlog::error("Exception serving static file: {}", e.what());
		sendResponse(context->fd, 500, "text/plain", "Internal server error");
	}
}

void MultiplexingServer::handleServeReactFile(const std::shared_ptr<ClientContext> &context)
{
	try
	{
		if (isApiEndpoint(context->path))
		{
			sendResponse(context->fd, 404, "application/json", R"({"error": "Not found"})");
			return;
		}

		fs::path file_path = static_files_root_ / context->path.substr(1);
		if (fs::exists(file_path) && fs::is_regular_file(file_path))
		{
			sendFileResponse(context->fd, file_path);
			return;
		}

		handleRoot(context);
	}
	catch (const std::exception &e)
	{
		spdlog::error("Exception serving React file: {}", e.what());
		sendResponse(context->fd, 500, "text/plain", "Internal server error");
	}
}

void MultiplexingServer::handleRoot(const std::shared_ptr<ClientContext> &context)
{
	try
	{
		fs::path index_path = static_files_root_ / "index.html";
		if (!fs::exists(index_path))
		{
			spdlog::error("Index file not found: {}", index_path.string());
			sendResponse(context->fd, 404, "text/plain", "Page not found");
			return;
		}

		sendFileResponse(context->fd, index_path);
	}
	catch (const std::exception &e)
	{
		spdlog::error("Exception serving root: {}", e.what());
		sendResponse(context->fd, 500, "text/plain", "Internal server error");
	}
}

void MultiplexingServer::handleOptions(const std::shared_ptr<ClientContext> &context)
{
	sendResponse(context->fd, 200, "application/json", R"({"status": "ok"})");
}