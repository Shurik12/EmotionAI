#include <filesystem>
#include <fstream>
#include <vector>
#include <thread>
#include <sstream>
#include <algorithm>
#include <cstring>

#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <metrics/MetricsCollector.h>
#include <metrics/MetricsMiddleware.h>
#include <config/Config.h>
#include <logging/Logger.h>
#include <common/uuid.h>
#include <db/TaskManager.h>
#include <emotionai/FileProcessor.h>
#include "MultiplexingServer.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

// Constants
namespace
{
    constexpr int MAX_EVENTS = 64;
    constexpr size_t CHUNK_SIZE = 8192;
    constexpr size_t MAX_SEND_ATTEMPTS = 1000;
    constexpr int EPOLL_TIMEOUT_MS = 100;

    const std::vector<std::string> STATIC_EXTENSIONS = {
        ".js", ".css", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
        ".woff", ".woff2", ".ttf", ".eot", ".map", ".json", ".txt"};
}

MultiplexingServer::MultiplexingServer() = default;

MultiplexingServer::~MultiplexingServer()
{
    try
    {
        stop();
        cleanupResources();
    }
    catch (...)
    {
        LOG_ERROR("Exception during server shutdown");
    }
}

void MultiplexingServer::cleanupResources()
{
    if (epoll_fd_ != -1)
    {
        close(epoll_fd_);
        epoll_fd_ = -1;
    }
    if (server_fd_ != -1)
    {
        close(server_fd_);
        server_fd_ = -1;
    }
}

void MultiplexingServer::initialize()
{
    LOG_INFO("Initializing multiplexing server");
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
    LOG_INFO("Starting server on {}:{}", config.server().host, config.server().port);

    running_ = true;
    handleEvents();
}

void MultiplexingServer::stop() noexcept
{
    running_ = false;
}

void MultiplexingServer::setupRoutes()
{
    // POST routes
    post_routes_ = {
        {"/api/upload", [this](auto &&ctx, auto &&body)
         { handleUpload(ctx, body); }},
        {"/api/upload_realtime", [this](auto &&ctx, auto &&body)
         { handleUploadRealtime(ctx, body); }},
        {"/api/submit_application", [this](auto &&ctx, auto &&body)
         { handleSubmitApplication(ctx, body); }},
        {"/api/batch_progress", [this](auto &&ctx, auto &&body)
         { handleBatchProgress(ctx, body); }}};

    // GET routes
    get_routes_ = {
        {"/api/metrics", [this](auto &&ctx)
         { handleMetrics(ctx); }},
        {"/api/progress", [this](auto &&ctx)
         { handleProgress(ctx); }},
        {"/api/results", [this](auto &&ctx)
         { handleServeResult(ctx); }},
        {"/api/starage/info", [this](auto &&ctx)
         { handleStorageInfo(ctx); }},
        {"/api/health", [this](auto &&ctx)
         { handleHealthCheck(ctx); }},
        {"/static", [this](auto &&ctx)
         { handleServeStatic(ctx); }},
        {"/", [this](auto &&ctx)
         {
             (ctx->path == "/") ? handleRoot(ctx) : handleServeReactFile(ctx);
         }}};

    // OPTIONS routes
    for (const auto &route : {"/api/upload", "/api/upload_realtime", "/api/submit_application",
                              "/api/progress", "/api/results", "/api/health", "/static"})
    {
        options_routes_[route] = [this](auto &&ctx)
        { handleOptions(ctx); };
    }
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
        throw std::runtime_error("Failed to set socket options");
    }

    auto &config = Config::instance();
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = inet_addr(config.server().host.c_str());
    address.sin_port = htons(config.server().port);

    if (bind(server_fd_, reinterpret_cast<sockaddr *>(&address), sizeof(address)) < 0)
    {
        throw std::runtime_error("Failed to bind socket");
    }

    if (listen(server_fd_, SOMAXCONN) < 0)
    {
        throw std::runtime_error("Failed to listen on socket");
    }
}

void MultiplexingServer::setupEpoll()
{
    epoll_fd_ = epoll_create1(0);
    if (epoll_fd_ == -1)
    {
        throw std::runtime_error("Failed to create epoll instance");
    }

    epoll_event event{};
    event.events = EPOLLIN;
    event.data.fd = server_fd_;

    if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, server_fd_, &event) == -1)
    {
        throw std::runtime_error("Failed to add server socket to epoll");
    }
}

void MultiplexingServer::handleEvents()
{
    epoll_event events[MAX_EVENTS];

    while (running_)
    {
        int num_events = epoll_wait(epoll_fd_, events, MAX_EVENTS, EPOLL_TIMEOUT_MS);

        if (num_events == -1)
        {
            if (errno == EINTR)
                continue;
            LOG_ERROR("epoll_wait error: {}", strerror(errno));
            break;
        }

        for (int i = 0; i < num_events; ++i)
        {
            (events[i].data.fd == server_fd_) ? acceptNewConnection() : handleClientData(events[i].data.fd);
        }
    }
}

void MultiplexingServer::acceptNewConnection()
{
    sockaddr_in client_addr{};
    socklen_t client_len = sizeof(client_addr);

    int client_fd = accept4(server_fd_, reinterpret_cast<sockaddr *>(&client_addr), &client_len, SOCK_NONBLOCK);
    if (client_fd == -1)
    {
        LOG_ERROR("Failed to accept connection: {}", strerror(errno));
        return;
    }

    auto context = std::make_shared<ClientContext>(client_fd);
    clients_[client_fd] = context;

    // Metrics: increment active connections
    MetricsCollector::instance().incrementActiveConnections();

    epoll_event event{};
    event.events = EPOLLIN;
    event.data.fd = client_fd;

    if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, client_fd, &event) == -1)
    {
        LOG_ERROR("Failed to add client to epoll: {}", strerror(errno));
        closeClient(client_fd);
    }
    else
    {
        LOG_DEBUG("New client connected: fd={}", client_fd);
    }
}

void MultiplexingServer::handleClientData(int client_fd)
{
    auto it = clients_.find(client_fd);
    if (it == clients_.end())
        return;

    auto context = it->second;
    char buffer[4096];

    while (true)
    {
        ssize_t bytes_read = recv(client_fd, buffer, sizeof(buffer) - 1, 0);

        if (bytes_read == -1)
        {
            if (errno == EAGAIN || errno == EWOULDBLOCK)
                break;
            LOG_ERROR("Error reading from client {}: {}", client_fd, strerror(errno));
            closeClient(client_fd);
            return;
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

            if (context->headers_complete &&
                (context->content_length == 0 || context->buffer.length() >= context->content_length))
            {
                processRequest(context);
                return;
            }
        }
    }
}

void MultiplexingServer::parseHttpRequest(const std::shared_ptr<ClientContext> &context)
{
    size_t header_end = context->buffer.find("\r\n\r\n");
    if (header_end == std::string::npos)
        return;

    std::string headers_str = context->buffer.substr(0, header_end);
    std::istringstream headers_stream(headers_str);
    std::string line;

    // Parse request line
    if (std::getline(headers_stream, line))
    {
        std::istringstream request_line(line);
        request_line >> context->method >> context->path;
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

            // Trim and convert to lowercase
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);

            std::string lower_key = key;
            std::transform(lower_key.begin(), lower_key.end(), lower_key.begin(), ::tolower);
            context->headers[lower_key] = value;
        }
    }

    // Parse content length
    auto content_length_it = context->headers.find("content-length");
    if (content_length_it != context->headers.end())
    {
        try
        {
            context->content_length = std::stoul(content_length_it->second);
        }
        catch (const std::exception &)
        {
            LOG_WARN("Invalid Content-Length: {}", content_length_it->second);
        }
    }

    // Parse query parameters
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
                context->params[param.substr(0, equal_pos)] = param.substr(equal_pos + 1);
            }
        }
    }

    context->headers_complete = true;
    context->buffer = context->buffer.substr(header_end + 4);

    LOG_DEBUG("Request parsed: {} {} (content-length: {})",
              context->method, context->path, context->content_length);
}

void MultiplexingServer::processRequest(const std::shared_ptr<ClientContext> &context)
{
    LOG_DEBUG("Processing: {} {}", context->method, context->path);
    auto start_time = std::chrono::steady_clock::now();

    try
    {

        if (context->method == "OPTIONS")
        {
            auto it = options_routes_.find(context->path);
            (it != options_routes_.end()) ? it->second(context) : handleOptions(context);
        }
        else if (context->method == "POST")
        {
            std::string body = context->buffer.substr(0, context->content_length);
            auto it = post_routes_.find(context->path);
            (it != post_routes_.end()) ? it->second(context, body) : sendErrorResponse(context->fd, 404, "Not found");
        }
        else if (context->method == "GET")
        {
            // Exact API route matches
            auto exact_it = get_routes_.find(context->path);
            if (exact_it != get_routes_.end())
            {
                exact_it->second(context);
                return;
            }

            // Parameterized API routes
            if (context->path.find("/api/progress/") == 0)
            {
                context->params["task_id"] = context->path.substr(14);
                handleProgress(context);
            }
            else if (context->path.find("/api/results/") == 0)
            {
                context->params["filename"] = context->path.substr(13);
                handleServeResult(context);
            }
            else if (context->path.find("/api/health") == 0)
            {
                handleHealthCheck(context);
            }
            // Static files
            else if (context->path.find("/static/") == 0)
            {
                handleServeStatic(context);
            }
            // Other static assets
            else if (isStaticAsset(context->path))
            {
                fs::path file_path = static_files_root_ / context->path.substr(1);
                (fs::exists(file_path) && fs::is_regular_file(file_path))
                    ? sendFileResponse(context->fd, file_path)
                    : sendErrorResponse(context->fd, 404, "File not found");
            }
            // React routes
            else
            {
                handleServeReactFile(context);
            }
        }
        else
        {
            sendErrorResponse(context->fd, 405, "Method not allowed");
        }

        // Record metrics
        auto end_time = std::chrono::steady_clock::now();
        double duration = std::chrono::duration<double>(end_time - start_time).count();

        MetricsCollector::instance().recordRequest(
            context->method, context->path, 200, duration);
    }
    catch (const std::exception &e)
    {
        auto end_time = std::chrono::steady_clock::now();
        double duration = std::chrono::duration<double>(end_time - start_time).count();

        MetricsCollector::instance().recordRequest(
            context->method, context->path, 500, duration);

        LOG_ERROR("Error processing request: {}", e.what());
        sendErrorResponse(context->fd, 500, "Internal server error");
    }
}

void MultiplexingServer::sendHttpResponse(int client_fd, int status_code, const std::string &content_type, const std::string &body)
{
    static const std::map<int, std::string> STATUS_TEXTS = {
        {200, "OK"}, {201, "Created"}, {202, "Accepted"}, {400, "Bad Request"}, {404, "Not Found"}, {405, "Method Not Allowed"}, {500, "Internal Server Error"}};

    std::string status_text = STATUS_TEXTS.count(status_code) ? STATUS_TEXTS.at(status_code) : "Unknown";

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

    // Send with retry logic for partial sends
    const char *data = response.c_str();
    size_t remaining = response.length();
    size_t attempts = 0;

    while (remaining > 0 && attempts < MAX_SEND_ATTEMPTS)
    {
        ssize_t bytes_sent = send(client_fd, data, remaining, MSG_NOSIGNAL);

        if (bytes_sent == -1)
        {
            if (errno == EAGAIN || errno == EWOULDBLOCK)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                attempts++;
                continue;
            }
            LOG_ERROR("Failed to send response to client {}: {}", client_fd, strerror(errno));
            break;
        }

        data += bytes_sent;
        remaining -= bytes_sent;
        attempts = 0;
    }

    if (remaining == 0)
    {
        LOG_DEBUG("Sent {} bytes to client {}", response.length(), client_fd);
    }

    closeClient(client_fd);
}

void MultiplexingServer::sendFileResponse(int client_fd, const fs::path &file_path)
{
    LOG_DEBUG("Sending file: {} to client {}", file_path.string(), client_fd);

    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        sendErrorResponse(client_fd, 404, "File not found");
        return;
    }

    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::string mime_type = getMimeType(file_path.string());
    std::string headers = fmt::format(
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: {}\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Content-Length: {}\r\n"
        "Connection: close\r\n"
        "\r\n",
        mime_type, file_size);

    // Send headers
    if (send(client_fd, headers.c_str(), headers.length(), MSG_NOSIGNAL) == -1)
    {
        LOG_ERROR("Failed to send headers to client {}: {}", client_fd, strerror(errno));
        closeClient(client_fd);
        return;
    }

    // Send file in chunks
    std::vector<char> buffer(CHUNK_SIZE);
    size_t total_sent = 0;
    size_t attempts = 0;

    while (total_sent < static_cast<size_t>(file_size) && attempts < MAX_SEND_ATTEMPTS)
    {
        size_t remaining = static_cast<size_t>(file_size) - total_sent;
        size_t chunk_size = std::min(CHUNK_SIZE, remaining);

        file.read(buffer.data(), chunk_size);
        std::streamsize bytes_read = file.gcount();

        if (bytes_read <= 0)
            break;

        ssize_t bytes_sent = send(client_fd, buffer.data(), bytes_read, MSG_NOSIGNAL);

        if (bytes_sent == -1)
        {
            if (errno == EAGAIN || errno == EWOULDBLOCK)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                attempts++;
                continue;
            }
            LOG_ERROR("Failed to send file chunk: {}", strerror(errno));
            break;
        }

        total_sent += bytes_sent;
        attempts = 0;
    }

    LOG_DEBUG("File transfer {}: {}/{} bytes",
              (total_sent == static_cast<size_t>(file_size)) ? "complete" : "incomplete",
              total_sent, file_size);

    closeClient(client_fd);
}

void MultiplexingServer::closeClient(int client_fd)
{
    if (auto it = clients_.find(client_fd); it != clients_.end())
    {
        MetricsCollector::instance().decrementActiveConnections();

        epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, client_fd, nullptr);
        close(client_fd);

        // Clear the buffer to free memory
        it->second->buffer.clear();
        it->second->buffer.shrink_to_fit();

        clients_.erase(it);
        LOG_DEBUG("Client disconnected: fd={}", client_fd);
    }
}

// Helper methods
bool MultiplexingServer::isStaticAsset(const std::string &path) const
{
    return std::any_of(STATIC_EXTENSIONS.begin(), STATIC_EXTENSIONS.end(),
                       [&path](const std::string &ext)
                       {
                           return path.length() >= ext.length() &&
                                  path.compare(path.length() - ext.length(), ext.length(), ext) == 0;
                       });
}

bool MultiplexingServer::isApiEndpoint(const std::string &path) const
{
    return path.find("/api/") == 0;
}

void MultiplexingServer::sendErrorResponse(int client_fd, int status_code, const std::string &message)
{
    sendHttpResponse(client_fd, status_code, "application/json",
                     fmt::format(R"({{"error": "{}"}})", message));
}

// Route handlers
void MultiplexingServer::handleUpload(const std::shared_ptr<ClientContext> &context, const std::string &body)
{
    try
    {
        LOG_DEBUG("Handling upload request");

        auto content_type_it = context->headers.find("content-type");
        if (content_type_it == context->headers.end() ||
            content_type_it->second.find("multipart/form-data") == std::string::npos)
        {
            sendErrorResponse(context->fd, 400, "Expected multipart form data");
            return;
        }

        std::string boundary = extractBoundary(content_type_it->second);
        if (boundary.empty())
        {
            sendErrorResponse(context->fd, 400, "Invalid multipart data");
            return;
        }

        auto form_data = parseMultipartFormData(body, boundary);
        auto file_it = form_data.find("file");
        if (file_it == form_data.end() || file_it->second.empty())
        {
            sendErrorResponse(context->fd, 400, "No file provided");
            return;
        }

        std::string filename = form_data.contains("filename") ? form_data["filename"] : "uploaded_file";
        if (filename.empty())
        {
            filename = "uploaded_file";
        }

        if (!file_processor_ || !file_processor_->allowed_file(filename))
        {
            sendErrorResponse(context->fd, 400, "Invalid file type");
            return;
        }

        std::string task_id = handleUploadCommon(file_it->second, filename, false);
        LOG_INFO("Upload accepted from client {}, task_id: {}", context->fd, task_id);

        sendHttpResponse(context->fd, 202, "application/json",
                         fmt::format(R"({{"task_id": "{}"}})", task_id));
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Exception in handleUpload from client {}: {}", context->fd, e.what());
        sendErrorResponse(context->fd, 500, "Internal server error");
    }
}

void MultiplexingServer::handleUploadRealtime(const std::shared_ptr<ClientContext> &context, const std::string &body)
{
    try
    {
        LOG_DEBUG("Handling real-time upload request");

        auto content_type_it = context->headers.find("content-type");
        if (content_type_it == context->headers.end() ||
            content_type_it->second.find("multipart/form-data") == std::string::npos)
        {
            sendErrorResponse(context->fd, 400, "Expected multipart form data");
            return;
        }

        std::string boundary = extractBoundary(content_type_it->second);
        if (boundary.empty())
        {
            sendErrorResponse(context->fd, 400, "Invalid multipart data");
            return;
        }

        auto form_data = parseMultipartFormData(body, boundary);
        auto file_it = form_data.find("file");
        if (file_it == form_data.end() || file_it->second.empty())
        {
            sendErrorResponse(context->fd, 400, "No file provided");
            return;
        }

        std::string filename = form_data.contains("filename") ? form_data["filename"] : "uploaded_file";
        if (filename.empty())
        {
            filename = "uploaded_file";
        }

        if (!file_processor_ || !file_processor_->allowed_file(filename))
        {
            sendErrorResponse(context->fd, 400, "Invalid file type");
            return;
        }

        std::string task_id = handleUploadCommon(file_it->second, filename, true);
        LOG_INFO("Real-time upload accepted from client {}, task_id: {}", context->fd, task_id);

        sendHttpResponse(context->fd, 202, "application/json",
                         fmt::format(R"({{"task_id": "{}", "mode": "realtime"}})", task_id));
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Exception in real-time upload from client {}: {}", context->fd, e.what());
        sendErrorResponse(context->fd, 500, "Internal server error");
    }
}

void MultiplexingServer::handleMetrics(const std::shared_ptr<ClientContext> &context)
{
    try
    {
        auto metrics = collectMetrics();
        sendHttpResponse(context->fd, 200, "text/plain; version=0.0.4", metrics);
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Metrics error: {}", e.what());
        sendErrorResponse(context->fd, 500, "Internal server error");
    }
}

void MultiplexingServer::handleProgress(const std::shared_ptr<ClientContext> &context)
{
    try
    {
        std::string task_id = context->params["task_id"];
        if (task_id.empty())
        {
            sendErrorResponse(context->fd, 400, "Task ID required");
            return;
        }

        auto &task_manager = TaskManager::instance();
        if (auto status = task_manager.get_task_status(task_id))
        {
            sendHttpResponse(context->fd, 200, "application/json", status->dump());
        }
        else
        {
            sendErrorResponse(context->fd, 404, "Task not found");
        }
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Exception in handleProgress: {}", e.what());
        sendErrorResponse(context->fd, 500, "Internal server error");
    }
}

void MultiplexingServer::handleBatchProgress(const std::shared_ptr<ClientContext> &context, const std::string &body)
{
    try
    {
        auto task_ids_json = nlohmann::json::parse(body);
        if (!task_ids_json.is_array())
        {
            sendErrorResponse(context->fd, 400, "Expected array of task IDs");
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
            sendErrorResponse(context->fd, 400, "No task IDs provided");
            return;
        }

        auto &task_manager = TaskManager::instance();
        auto results = task_manager.batch_get_status(task_ids);

        nlohmann::json response;
        for (const auto &[task_id, status] : results)
        {
            response[task_id] = status;
        }

        sendHttpResponse(context->fd, 200, "application/json", response.dump());
    }
    catch (const nlohmann::json::parse_error &e)
    {
        LOG_ERROR("JSON parse error in batch progress: {}", e.what());
        sendErrorResponse(context->fd, 400, "Invalid JSON");
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Exception in handleBatchProgress: {}", e.what());
        sendErrorResponse(context->fd, 500, "Internal server error");
    }
}

void MultiplexingServer::handleSubmitApplication(const std::shared_ptr<ClientContext> &context, const std::string &body)
{
    try
    {
        std::string application_id = handleSubmitApplicationCommon(body);
        sendHttpResponse(context->fd, 201, "application/json",
                         fmt::format(R"({{"application_id": "{}"}})", application_id));
    }
    catch (const json::parse_error &e)
    {
        sendErrorResponse(context->fd, 400, "Invalid JSON");
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Error submitting application: {}", e.what());
        sendErrorResponse(context->fd, 500, "Internal server error");
    }
}

void MultiplexingServer::handleServeResult(const std::shared_ptr<ClientContext> &context)
{
    try
    {
        auto filename_it = context->params.find("filename");
        if (filename_it == context->params.end())
        {
            sendErrorResponse(context->fd, 400, "Filename required");
            return;
        }

        // Try to serve from shared storage first
        std::string storage_path = "results/" + filename_it->second;
        if (file_storage_->fileExists(storage_path))
        {
            // Serve from storage
            std::vector<uint8_t> file_content = file_storage_->readFileBinary(storage_path);
            if (!file_content.empty())
            {
                std::string content_str(file_content.begin(), file_content.end());
                std::string mime_type = getMimeType(filename_it->second);
                sendHttpResponse(context->fd, 200, mime_type, content_str);
                return;
            }
        }

        // Fallback to local filesystem for backward compatibility
        fs::path file_path = results_folder_ / filename_it->second;
        if (fs::exists(file_path) && fs::is_regular_file(file_path))
        {
            sendFileResponse(context->fd, file_path);
        }
        else
        {
            sendErrorResponse(context->fd, 404, "File not found");
        }
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Exception serving result file: {}", e.what());
        sendErrorResponse(context->fd, 500, "Internal server error");
    }
}

void MultiplexingServer::handleStorageInfo(const std::shared_ptr<ClientContext> &context)
{
    try
    {
        auto storage_info = file_storage_->getStorageInfo();
        sendHttpResponse(context->fd, 200, "application/json", storage_info.dump());
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Error getting storage info: {}", e.what());
        sendErrorResponse(context->fd, 500, "Internal server error");
    }
}

void MultiplexingServer::handleHealthCheck(const std::shared_ptr<ClientContext> &context)
{
    sendHttpResponse(context->fd, 200, "application/json", R"({"status": "healthy"})");
}

void MultiplexingServer::handleServeStatic(const std::shared_ptr<ClientContext> &context)
{
    try
    {
        std::string filename = context->path.substr(8); // Remove "/static/"
        fs::path static_path = static_files_root_ / "static" / filename;

        if (fs::exists(static_path) && fs::is_regular_file(static_path))
        {
            sendFileResponse(context->fd, static_path);
        }
        else
        {
            sendErrorResponse(context->fd, 404, "File not found");
        }
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Exception serving static file: {}", e.what());
        sendErrorResponse(context->fd, 500, "Internal server error");
    }
}

void MultiplexingServer::handleServeReactFile(const std::shared_ptr<ClientContext> &context)
{
    fs::path index_path = static_files_root_ / "index.html";
    if (fs::exists(index_path))
    {
        sendFileResponse(context->fd, index_path);
    }
    else
    {
        sendErrorResponse(context->fd, 404, "Page not found");
    }
}

void MultiplexingServer::handleRoot(const std::shared_ptr<ClientContext> &context)
{
    handleServeReactFile(context);
}

void MultiplexingServer::handleOptions(const std::shared_ptr<ClientContext> &context)
{
    sendHttpResponse(context->fd, 200, "application/json", R"({"status": "ok"})");
}