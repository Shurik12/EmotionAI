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
#include <db/RedisManager.h>
#include <emotionai/FileProcessor.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

MultiplexingServer::MultiplexingServer()
	: server_fd_(-1), epoll_fd_(-1), running_(false),
	  redis_manager_(nullptr), file_processor_(nullptr)
{
}

MultiplexingServer::~MultiplexingServer()
{
	try
	{
		stop();

		// Wait for all background threads to finish
		for (auto &[task_id, thread] : background_threads_)
		{
			if (thread.joinable())
			{
				thread.join();
			}
		}

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
		spdlog::error("Exception during server shutdown");
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

void MultiplexingServer::initializeComponents()
{
	LOG_INFO("Initializing RedisManager and FileProcessor...");
	try
	{
		// Create RedisManager as shared_ptr
		redis_manager_ = std::make_shared<db::RedisManager>();
		redis_manager_->initialize();
		LOG_INFO("RedisManager initialized successfully");

		// FileProcessor now expects shared_ptr
		file_processor_ = std::make_unique<EmotionAI::FileProcessor>(redis_manager_);
		LOG_INFO("FileProcessor initialized successfully");
	}
	catch (const std::exception &e)
	{
		spdlog::error("Failed to initialize components: {}", e.what());
		throw;
	}
}

void MultiplexingServer::start()
{
	auto &config = Common::Config::instance();
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

// duplication
void MultiplexingServer::loadConfiguration()
{
	auto &config = Common::Config::instance();
	log_folder_ = config.paths().logs;
	static_files_root_ = config.paths().frontend;
	upload_folder_ = config.paths().upload;
	results_folder_ = config.paths().results;
}

void MultiplexingServer::ensureDirectoriesExist()
{
	try
	{
		fs::create_directories(upload_folder_);
		fs::create_directories(results_folder_);
		fs::create_directories(static_files_root_);
		LOG_INFO("Directories ensured: upload={}, results={}, static={}",
				 upload_folder_.string(), results_folder_.string(), static_files_root_.string());
	}
	catch (const std::exception &e)
	{
		spdlog::error("Failed to create directories: {}", e.what());
		throw;
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

	// GET routes - only exact matches now
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

	auto &config = Common::Config::instance();
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
		int num_events = epoll_wait(epoll_fd_, events, MAX_EVENTS, 100); // 100ms timeout

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

		cleanupFinishedThreads();
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
	event.events = EPOLLIN | EPOLLET; // Edge-triggered
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
				// No more data to read
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
			// Client disconnected
			closeClient(client_fd);
			return;
		}
		else
		{
			buffer[bytes_read] = '\0';
			context->buffer.append(buffer, bytes_read);

			// Parse HTTP request if headers are not complete
			if (!context->headers_complete)
			{
				parseHttpRequest(context);
			}

			// If headers are complete and we have the full body, process the request
			if (context->headers_complete && context->buffer.length() >= context->content_length)
			{
				processRequest(context);
				// DON'T close client here - let the response be sent first
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
		return; // Headers not complete yet
	}

	std::string headers_str = context->buffer.substr(0, header_end);
	std::istringstream headers_stream(headers_str);
	std::string line;

	// Parse request line
	if (std::getline(headers_stream, line))
	{
		std::istringstream request_line(line);
		request_line >> context->method >> context->path;

		// Convert method to uppercase
		std::transform(context->method.begin(), context->method.end(), context->method.begin(), ::toupper);
	}

	// Parse headers
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
			// Trim whitespace
			value.erase(0, value.find_first_not_of(" \t"));
			value.erase(value.find_last_not_of(" \t") + 1);
			context->headers[key] = value;
		}
	}

	// Extract content length
	auto content_length_it = context->headers.find("Content-Length");
	if (content_length_it != context->headers.end())
	{
		context->content_length = std::stoul(content_length_it->second);
	}

	// Extract query parameters
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
	context->buffer = context->buffer.substr(header_end + 4); // Remove headers from buffer
}

void MultiplexingServer::processRequest(const std::shared_ptr<ClientContext> &context)
{
	LOG_DEBUG("Processing: {} {}", context->method, context->path);

	try
	{
		if (context->method == "OPTIONS")
		{
			// Handle preflight requests
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
			// Handle parameterized routes FIRST (before exact matches)
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

			// THEN try exact route matches from setupRoutes()
			auto it = get_routes_.find(context->path);
			if (it != get_routes_.end())
			{
				it->second(context);
				return;
			}

			// Finally, fall back to React routing for non-API paths
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

	// For now, we'll always close connections after response
	// In the future, you could implement keep-alive based on headers
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

void MultiplexingServer::sendFileResponse(int client_fd, const std::filesystem::path &file_path)
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

std::string MultiplexingServer::getMimeType(const std::string &filename)
{
	size_t dot_pos = filename.rfind('.');
	if (dot_pos == std::string::npos)
		return "application/octet-stream";

	std::string ext = filename.substr(dot_pos + 1);
	std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

	static std::map<std::string, std::string> mime_types = {
		{"html", "text/html"},
		{"css", "text/css"},
		{"js", "application/javascript"},
		{"json", "application/json"},
		{"png", "image/png"},
		{"jpg", "image/jpeg"},
		{"jpeg", "image/jpeg"},
		{"gif", "image/gif"},
		{"svg", "image/svg+xml"},
		{"pdf", "application/pdf"},
		{"txt", "text/plain"}};

	auto it = mime_types.find(ext);
	return it != mime_types.end() ? it->second : "application/octet-stream";
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

// Route handler implementations (adapted from original)

void MultiplexingServer::handleUpload(const std::shared_ptr<ClientContext> &context, const std::string &body)
{
	LOG_INFO("Handling file upload, body size: {}", body.size());

	try
	{
		auto content_type_it = context->headers.find("Content-Type");
		if (content_type_it == context->headers.end())
		{
			spdlog::error("No Content-Type header");
			sendResponse(context->fd, 400, "application/json", R"({"error": "No Content-Type header"})");
			closeClient(context->fd);
			return;
		}

		LOG_INFO("Content-Type: {}", content_type_it->second);

		if (content_type_it->second.find("multipart/form-data") == std::string::npos)
		{
			spdlog::error("Expected multipart/form-data, got: {}", content_type_it->second);
			sendResponse(context->fd, 400, "application/json", R"({"error": "Expected multipart form data"})");
			closeClient(context->fd);
			return;
		}

		std::string boundary = extractBoundary(content_type_it->second);
		if (boundary.empty())
		{
			spdlog::error("Could not extract boundary from: {}", content_type_it->second);
			sendResponse(context->fd, 400, "application/json", R"({"error": "Invalid multipart data"})");
			closeClient(context->fd);
			return;
		}

		auto form_data = parseMultipartFormData(body, boundary);

		// Debug: log all form fields
		for (const auto &[key, value] : form_data)
		{
			LOG_INFO("Form field: '{}', size: {}", key, value.size());
		}

		auto file_it = form_data.find("file");
		if (file_it == form_data.end())
		{
			spdlog::error("No 'file' field found in multipart data. Available fields:");
			for (const auto &[key, value] : form_data)
			{
				spdlog::error("  Field: '{}'", key);
			}
			sendResponse(context->fd, 400, "application/json", R"({"error": "No file provided"})");
			closeClient(context->fd);
			return;
		}

		const std::string &file_content = file_it->second;
		if (file_content.empty())
		{
			spdlog::error("File content is empty");
			sendResponse(context->fd, 400, "application/json", R"({"error": "No file selected or empty file"})");
			closeClient(context->fd);
			return;
		}

		// Get filename from multipart data or use a default
		std::string filename;
		auto filename_it = form_data.find("filename");
		if (filename_it != form_data.end() && !filename_it->second.empty())
		{
			filename = filename_it->second;
			LOG_INFO("Using filename from multipart: {}", filename);
		}
		else
		{
			// Try to extract from Content-Disposition in the body as fallback
			size_t filename_pos = body.find("filename=\"");
			if (filename_pos != std::string::npos)
			{
				filename_pos += 10;
				size_t filename_end = body.find("\"", filename_pos);
				if (filename_end != std::string::npos)
				{
					filename = body.substr(filename_pos, filename_end - filename_pos);
					LOG_INFO("Extracted filename from body: {}", filename);
				}
			}
		}

		if (filename.empty())
		{
			filename = "uploaded_file";
			spdlog::warn("No filename found, using default: {}", filename);
		}

		LOG_INFO("Processing upload: filename='{}', size={} bytes", filename, file_content.size());

		if (!file_processor_ || !file_processor_->allowed_file(filename))
		{
			spdlog::error("File type not allowed: {}", filename);
			sendResponse(context->fd, 400, "application/json", R"({"error": "Invalid file type"})");
			closeClient(context->fd);
			return;
		}

		std::string task_id = db::RedisManager::generate_uuid();
		fs::path filepath = upload_folder_ / (task_id + "_" + filename);

		// Save the file with better error handling
		try
		{
			LOG_INFO("Saving file to: {}", filepath.string());
			std::ofstream out_file(filepath, std::ios::binary);
			if (!out_file)
			{
				throw std::runtime_error("Failed to create file: " + filepath.string());
			}
			out_file.write(file_content.data(), file_content.size());
			out_file.close();

			// Verify file was written
			if (!fs::exists(filepath))
			{
				throw std::runtime_error("File was not created");
			}

			auto file_size = fs::file_size(filepath);
			if (file_size == 0)
			{
				throw std::runtime_error("File is empty");
			}

			LOG_INFO("File saved successfully: {}, size: {} bytes", filepath.string(), file_size);
		}
		catch (const std::exception &e)
		{
			spdlog::error("Failed to save file {}: {}", filepath.string(), e.what());
			sendResponse(context->fd, 500, "application/json", R"({"error": "Failed to save uploaded file"})");
			closeClient(context->fd);
			return;
		}

		// Process file in background thread
		std::lock_guard<std::mutex> lock(task_mutex_);
		background_threads_[task_id] = std::thread(
			[this, task_id, filepath, filename]()
			{
				try
				{
					LOG_INFO("Starting background processing for task: {}", task_id);
					file_processor_->process_file(task_id, filepath.string(), filename);
					LOG_INFO("Background processing completed for task: {}", task_id);
				}
				catch (const std::exception &e)
				{
					spdlog::error("Background processing failed for task {}: {}", task_id, e.what());
				}
				// Remove thread from map when done
				std::lock_guard<std::mutex> lock(task_mutex_);
				background_threads_.erase(task_id);
			});
		background_threads_[task_id].detach();

		LOG_INFO("Upload accepted, task_id: {}", task_id);
		sendResponse(context->fd, 202, "application/json",
					 fmt::format(R"({{"task_id": "{}"}})", task_id));

		// Close connection after sending response
		closeClient(context->fd);
	}
	catch (const std::exception &e)
	{
		spdlog::error("Exception in handleUpload: {}", e.what());
		sendResponse(context->fd, 500, "application/json", R"({"error": "Internal server error"})");
		closeClient(context->fd);
	}
}

void MultiplexingServer::handleUploadRealtime(const std::shared_ptr<ClientContext> &context, const std::string &body)
{
	// Similar to handleUpload but calls process_video_realtime
	// Implementation would be very similar to handleUpload
	handleUpload(context, body); // Simplified for example
}

void MultiplexingServer::handleProgress(const std::shared_ptr<ClientContext> &context)
{
	try
	{
		std::string task_id;

		// Check if task_id is in params (from parameterized route) or query string
		auto task_id_it = context->params.find("task_id");
		if (task_id_it != context->params.end())
		{
			task_id = task_id_it->second;
		}
		else
		{
			// Check query parameters
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

		if (!redis_manager_)
		{
			sendResponse(context->fd, 500, "application/json", R"({"error": "Server not properly initialized"})");
			return;
		}

		auto status = redis_manager_->get_task_status(task_id);
		if (!status)
		{
			sendResponse(context->fd, 404, "application/json", R"({"error": "Task not found"})");
			return;
		}

		sendResponse(context->fd, 200, "application/json", status.value());
	}
	catch (const std::exception &e)
	{
		spdlog::error("Exception in handleProgress: {}", e.what());
		sendResponse(context->fd, 500, "application/json", R"({"error": "Internal server error"})");
	}
}

void MultiplexingServer::handleSubmitApplication(const std::shared_ptr<ClientContext> &context, const std::string &body)
{
	try
	{
		if (!redis_manager_)
		{
			sendResponse(context->fd, 500, "application/json", R"({"error": "Server not properly initialized"})");
			return;
		}

		json application_data;
		try
		{
			application_data = json::parse(body);
			validateJsonDocument(application_data);
		}
		catch (const json::parse_error &e)
		{
			sendResponse(context->fd, 400, "application/json", R"({"error": "Invalid JSON"})");
			return;
		}
		catch (const std::exception &e)
		{
			sendResponse(context->fd, 400, "application/json",
						 fmt::format(R"({{"error": "{}"}})", e.what()));
			return;
		}

		std::string application_id = redis_manager_->save_application(application_data.dump());
		sendResponse(context->fd, 201, "application/json",
					 fmt::format(R"({{"application_id": "{}"}})", application_id));
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

// Helper functions

bool MultiplexingServer::isApiEndpoint(const std::string &path)
{
	static const std::set<std::string> apiPrefixes = {"/api/", "/static/"};
	for (const auto &prefix : apiPrefixes)
	{
		if (path.find(prefix) == 0)
			return true;
	}
	return false;
}

void MultiplexingServer::cleanupFinishedThreads()
{
	std::lock_guard<std::mutex> lock(task_mutex_);
	for (auto it = background_threads_.begin(); it != background_threads_.end();)
	{
		if (!it->second.joinable())
		{
			it = background_threads_.erase(it);
		}
		else
		{
			++it;
		}
	}
}

std::string MultiplexingServer::extractBoundary(const std::string &content_type)
{
	size_t boundary_pos = content_type.find("boundary=");
	if (boundary_pos == std::string::npos)
	{
		spdlog::error("No boundary found in Content-Type: {}", content_type);
		return "";
	}

	boundary_pos += 9; // Length of "boundary="

	// Extract the boundary value
	std::string boundary;
	if (boundary_pos < content_type.length())
	{
		if (content_type[boundary_pos] == '"')
		{
			// Quoted boundary
			boundary_pos++;
			size_t end_quote = content_type.find('"', boundary_pos);
			if (end_quote != std::string::npos)
			{
				boundary = content_type.substr(boundary_pos, end_quote - boundary_pos);
			}
		}
		else
		{
			// Unquoted boundary
			size_t end_pos = content_type.find(';', boundary_pos);
			if (end_pos == std::string::npos)
			{
				boundary = content_type.substr(boundary_pos);
			}
			else
			{
				boundary = content_type.substr(boundary_pos, end_pos - boundary_pos);
			}
		}
	}

	// Trim whitespace
	boundary.erase(0, boundary.find_first_not_of(" \t"));
	boundary.erase(boundary.find_last_not_of(" \t") + 1);

	spdlog::debug("Extracted boundary: '{}'", boundary);
	return "--" + boundary;
}

std::map<std::string, std::string> MultiplexingServer::parseMultipartFormData(const std::string &body, const std::string &boundary)
{
	std::map<std::string, std::string> result;

	if (body.empty() || boundary.empty())
	{
		spdlog::error("Empty body or boundary in multipart data");
		return result;
	}

	spdlog::debug("Parsing multipart data, body size: {}, boundary: '{}'", body.size(), boundary);

	size_t pos = 0;

	// Find first boundary
	size_t boundary_pos = body.find(boundary);
	if (boundary_pos == std::string::npos)
	{
		spdlog::error("First boundary not found");
		return result;
	}

	pos = boundary_pos + boundary.length();

	while (pos < body.length())
	{
		// Skip CRLF after boundary
		if (pos + 2 <= body.length() && body.substr(pos, 2) == "\r\n")
		{
			pos += 2;
		}
		else if (pos + 2 <= body.length() && body.substr(pos, 2) == "--")
		{
			// End of multipart data
			break;
		}

		// Parse headers
		size_t headers_end = body.find("\r\n\r\n", pos);
		if (headers_end == std::string::npos)
		{
			spdlog::error("Headers end not found");
			break;
		}

		std::string headers_str = body.substr(pos, headers_end - pos);
		pos = headers_end + 4; // Skip \r\n\r\n

		// Parse headers to get field name and filename
		std::string field_name;
		std::string filename;

		std::istringstream headers_stream(headers_str);
		std::string header_line;
		while (std::getline(headers_stream, header_line))
		{
			if (header_line.back() == '\r')
				header_line.pop_back();

			if (header_line.find("Content-Disposition:") == 0)
			{
				// Parse Content-Disposition header
				size_t name_pos = header_line.find("name=\"");
				if (name_pos != std::string::npos)
				{
					name_pos += 6;
					size_t name_end = header_line.find("\"", name_pos);
					if (name_end != std::string::npos)
					{
						field_name = header_line.substr(name_pos, name_end - name_pos);
					}
				}

				size_t filename_pos = header_line.find("filename=\"");
				if (filename_pos != std::string::npos)
				{
					filename_pos += 10;
					size_t filename_end = header_line.find("\"", filename_pos);
					if (filename_end != std::string::npos)
					{
						filename = header_line.substr(filename_pos, filename_end - filename_pos);
					}
				}
			}
		}

		// Find next boundary
		size_t next_boundary = body.find(boundary, pos);
		if (next_boundary == std::string::npos)
		{
			// Last part - read until end (but remove trailing -- if present)
			size_t data_end = body.length();
			if (data_end >= 2 && body.substr(data_end - 2) == "--")
			{
				data_end -= 2;
			}
			// Also remove trailing CRLF if present
			if (data_end >= 2 && body.substr(data_end - 2, 2) == "\r\n")
			{
				data_end -= 2;
			}

			if (!field_name.empty() && data_end > pos)
			{
				std::string field_data = body.substr(pos, data_end - pos);
				result[field_name] = field_data;
				if (!filename.empty())
				{
					result["filename"] = filename;
				}
				spdlog::debug("Found field '{}' with data size: {}", field_name, field_data.size());
			}
			break;
		}

		// Extract data between current position and next boundary
		// Remove trailing CRLF before the boundary
		size_t data_end = next_boundary;
		if (data_end >= 2 && body.substr(data_end - 2, 2) == "\r\n")
		{
			data_end -= 2;
		}

		if (!field_name.empty() && data_end > pos)
		{
			std::string field_data = body.substr(pos, data_end - pos);
			result[field_name] = field_data;
			if (!filename.empty())
			{
				result["filename"] = filename;
			}
			spdlog::debug("Found field '{}' with data size: {}", field_name, field_data.size());
		}

		pos = next_boundary + boundary.length();

		// Check for final boundary
		if (pos + 2 <= body.length() && body.substr(pos, 2) == "--")
		{
			break;
		}
	}

	LOG_INFO("Parsed {} multipart fields", result.size());
	for (const auto &[key, value] : result)
	{
		LOG_INFO("  Field: '{}', data size: {}", key, value.size());
	}

	return result;
}

void MultiplexingServer::validateJsonDocument(const nlohmann::json &json)
{
	if (!json.is_object())
	{
		throw std::runtime_error("Expected JSON object");
	}
	if (json.empty())
	{
		throw std::runtime_error("Empty JSON object");
	}
}