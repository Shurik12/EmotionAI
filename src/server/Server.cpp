#include <filesystem>
#include <fstream>
#include <vector>
#include <thread>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <sys/resource.h>

#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <config/Config.h>
#include <logging/Logger.h>
#include <common/uuid.h>
#include <db/TaskManager.h>
#include <emotionai/FileProcessor.h>
#include <metrics/MetricsCollector.h>
#include <metrics/MetricsMiddleware.h>
#include <storage/FileStorageFactory.h>
#include <audio/BurnoutModels.h>
#include "Server.h"

#ifdef WITH_CLUSTER
#include <cluster/ClusterManager.h>
#include <cluster/DistributedTaskManager.h>
#endif

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

Server::Server() = default;

Server::~Server()
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

void Server::cleanupResources()
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

void Server::loadConfiguration()
{
    auto &config = Config::instance();
    log_folder_ = config.paths().logs;
    static_files_root_ = config.paths().frontend;
    upload_folder_ = config.paths().uploads;
    results_folder_ = config.paths().results;
}

void Server::ensureDirectoriesExist()
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
        LOG_ERROR("Failed to create directories: {}", e.what());
        throw;
    }
}

void Server::initializeComponents()
{
    LOG_INFO("Initializing components...");
    try
    {
        // Generate instance ID
        instance_id_ = generateInstanceId();
        LOG_INFO("Instance ID: {}", instance_id_);

        // Create thread pool
        unsigned int num_threads = std::max(2u, std::thread::hardware_concurrency());
        thread_pool_ = std::make_unique<ThreadPool>(num_threads);
        LOG_INFO("ThreadPool initialized with {} threads", num_threads);

        // Create DragonflyManager
        dragonfly_manager_ = std::make_shared<DragonflyManager>();
        dragonfly_manager_->initialize();
        LOG_INFO("DragonflyManager initialized successfully");

        // Initialize storage first
        initializeStorage();

        // Initialize cluster components if enabled
        auto &config = Config::instance();
        if (config.cluster().enabled)
        {
            initializeCluster();
        }

        // Initialize TaskManager
        auto &task_manager = TaskManager::instance();
        task_manager.set_dragonfly_manager(dragonfly_manager_);
        LOG_INFO("TaskManager initialized successfully");

        // Create FileProcessor with storage
        file_processor_ = std::make_unique<FileProcessor>(dragonfly_manager_, file_storage_);
        LOG_INFO("FileProcessor initialized successfully with {} storage", file_storage_->getStorageType());

        if (config.gigachat().enabled && !config.gigachat().auth_key.empty())
        {
            LOG_INFO("Initializing GigaChat client...");

            emotionai::gigachat::GigaChatConfig gigaConfig;
            gigaConfig.enabled = true;
            gigaConfig.authKey = config.gigachat().auth_key;
            gigaConfig.model = config.gigachat().model;
            gigaConfig.apiUrl = config.gigachat().api_url;
            gigaConfig.authUrl = config.gigachat().auth_url;
            gigaConfig.verifySsl = config.gigachat().verify_ssl;

            auto gigachat_client = std::make_unique<emotionai::gigachat::GigaChatClient>(gigaConfig);
            file_processor_->setGigaChatClient(std::move(gigachat_client));

            LOG_INFO("GigaChat client initialized successfully");
        }

        LOG_INFO("FileProcessor initialized successfully");
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Failed to initialize components: {}", e.what());
        throw;
    }
}

void Server::initializeStorage()
{
    try
    {
        auto &config = Config::instance();
        auto storage_config = config.storage();

        LOG_INFO("Initializing storage system: type={}", storage_config.type);

        file_storage_ = FileStorageFactory::createStorageFromConfig();

        LOG_INFO("Storage system initialized: {}", file_storage_->getStorageType());
        auto storage_info = file_storage_->getStorageInfo();
        LOG_INFO("Storage info: {}", storage_info.dump());
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Failed to initialize storage system: {}", e.what());
        throw;
    }
}

void Server::initializeCluster()
{
#ifdef WITH_CLUSTER
    auto &config = Config::instance();
    LOG_INFO("Initializing cluster components...");

    try
    {
        // Create cluster manager
        cluster_manager_ = std::make_unique<ClusterManager>(dragonfly_manager_);
        cluster_manager_->initialize();
        LOG_INFO("ClusterManager initialized successfully");

        // Create distributed task manager
        distributed_task_manager_ = std::make_unique<DistributedTaskManager>(
            dragonfly_manager_, instance_id_);
        LOG_INFO("DistributedTaskManager initialized successfully");

        // Register this instance
        registerInstance();
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Failed to initialize cluster components: {}", e.what());
        throw;
    }
#else
    LOG_WARN("Cluster support not compiled in, but cluster.enabled=true in config");
#endif
}

void Server::startClusterServices()
{
#ifdef WITH_CLUSTER
    auto &config = Config::instance();
    if (config.cluster().enabled && cluster_manager_)
    {
        LOG_INFO("Starting cluster services...");
        cluster_manager_->start();
        startDistributedTaskWorkers();
        LOG_INFO("Cluster services started successfully");
    }
#endif
}

void Server::stopClusterServices()
{
#ifdef WITH_CLUSTER
    auto &config = Config::instance();
    if (config.cluster().enabled)
    {
        LOG_INFO("Stopping cluster services...");
        stopDistributedTaskWorkers();
        if (cluster_manager_)
        {
            cluster_manager_->stop();
        }
        unregisterInstance();
        LOG_INFO("Cluster services stopped successfully");
    }
#endif
}

void Server::registerInstance()
{
#ifdef WITH_CLUSTER
    if (!cluster_manager_)
        return;

    try
    {
        // Instance info will be registered by ClusterManager
        LOG_INFO("Instance registered with cluster: {}", instance_id_);
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Failed to register instance: {}", e.what());
    }
#endif
}

void Server::unregisterInstance()
{
#ifdef WITH_CLUSTER
    if (!cluster_manager_)
        return;

    try
    {
        LOG_INFO("Unregistering instance from cluster: {}", instance_id_);
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Failed to unregister instance: {}", e.what());
    }
#endif
}

void Server::initialize()
{
    LOG_INFO("Initializing server");
    loadConfiguration();
    ensureDirectoriesExist();
    initializeComponents();
    setupRoutes();
    createSocket();
    setupEpoll();
}

void Server::start()
{
    auto &config = Config::instance();
    LOG_INFO("Starting server on {}:{}", config.server().host, config.server().port);

    running_ = true;
    startClusterServices();
    handleEvents();
}

void Server::stop() noexcept
{
    running_ = false;
    stopClusterServices();
}

void Server::setupRoutes()
{
    // POST routes
    post_routes_ = {
        {"/api/upload", [this](auto &&ctx, auto &&body)
         { handleUpload(ctx, body); }},
        {"/api/upload_realtime", [this](auto &&ctx, auto &&body)
         { handleUploadRealtime(ctx, body); }},
        {"/api/upload_burnout", [this](auto &&ctx, auto &&body)
         { handleUploadBurnout(ctx, body); }},
        {"/api/submit_application", [this](auto &&ctx, auto &&body)
         { handleSubmitApplication(ctx, body); }},
        {"/api/batch_progress", [this](auto &&ctx, auto &&body)
         { handleBatchProgress(ctx, body); }},
        {"/api/burnout/analyze", [this](auto &&ctx, auto &&body)
         { handleBurnoutAnalyze(ctx, body); }},
        {"/api/burnout/baseline", [this](auto &&ctx, auto &&body)
         { handleBurnoutBaseline(ctx, body); }}
    };

    // GET routes
    get_routes_ = {
        {"/api/metrics", [this](auto &&ctx)
         { handleMetrics(ctx); }},
        {"/api/progress", [this](auto &&ctx)
         { handleProgress(ctx); }},
        {"/api/results", [this](auto &&ctx)
         { handleServeResult(ctx); }},
        {"/api/storage/info", [this](auto &&ctx)
         { handleStorageInfo(ctx); }},
        {"/api/health", [this](auto &&ctx)
         { handleHealthCheck(ctx); }},
        {"/static", [this](auto &&ctx)
         { handleServeStatic(ctx); }},
        {"/api/burnout/baseline", [this](auto &&ctx)
         { handleBurnoutBaselineGet(ctx); }}
    };

    // OPTIONS routes
    for (const auto &route : {"/api/upload", "/api/upload_realtime", "/api/upload_burnout",
                              "/api/submit_application", "/api/progress", "/api/results", 
                              "/api/health", "/api/burnout/analyze", "/api/burnout/baseline", "/static"})
    {
        options_routes_[route] = [this](auto &&ctx)
        { handleOptions(ctx); };
    }
}

void Server::createSocket()
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

void Server::setupEpoll()
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

void Server::handleEvents()
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

void Server::acceptNewConnection()
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

void Server::handleClientData(int client_fd)
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

void Server::parseHttpRequest(const std::shared_ptr<ClientContext> &context)
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

void Server::processRequest(const std::shared_ptr<ClientContext> &context)
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
            else if (context->path.find("/api/burnout/baseline/") == 0)
            {
                context->params["user_id"] = context->path.substr(22);
                handleBurnoutBaselineGet(context);
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

void Server::sendHttpResponse(int client_fd, int status_code, const std::string &content_type, const std::string &body)
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

void Server::sendFileResponse(int client_fd, const fs::path &file_path)
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

void Server::closeClient(int client_fd)
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
bool Server::isStaticAsset(const std::string &path) const
{
    return std::any_of(STATIC_EXTENSIONS.begin(), STATIC_EXTENSIONS.end(),
                       [&path](const std::string &ext)
                       {
                           return path.length() >= ext.length() &&
                                  path.compare(path.length() - ext.length(), ext.length(), ext) == 0;
                       });
}

bool Server::isApiEndpoint(const std::string &path) const
{
    return path.find("/api/") == 0;
}

void Server::sendErrorResponse(int client_fd, int status_code, const std::string &message)
{
    sendHttpResponse(client_fd, status_code, "application/json",
                     fmt::format(R"({{"error": "{}"}})", message));
}

// Route handlers
void Server::handleUpload(const std::shared_ptr<ClientContext> &context, const std::string &body)
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

void Server::handleUploadRealtime(const std::shared_ptr<ClientContext> &context, const std::string &body)
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

void Server::handleMetrics(const std::shared_ptr<ClientContext> &context)
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

void Server::handleProgress(const std::shared_ptr<ClientContext> &context)
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

void Server::handleBatchProgress(const std::shared_ptr<ClientContext> &context, const std::string &body)
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

void Server::handleSubmitApplication(const std::shared_ptr<ClientContext> &context, const std::string &body)
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

void Server::handleServeResult(const std::shared_ptr<ClientContext> &context)
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

void Server::handleStorageInfo(const std::shared_ptr<ClientContext> &context)
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

void Server::handleHealthCheck(const std::shared_ptr<ClientContext> &context)
{
    sendHttpResponse(context->fd, 200, "application/json", R"({"status": "healthy"})");
}

void Server::handleServeStatic(const std::shared_ptr<ClientContext> &context)
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

void Server::handleServeReactFile(const std::shared_ptr<ClientContext> &context)
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

void Server::handleRoot(const std::shared_ptr<ClientContext> &context)
{
    handleServeReactFile(context);
}

void Server::handleOptions(const std::shared_ptr<ClientContext> &context)
{
    sendHttpResponse(context->fd, 200, "application/json", R"({"status": "ok"})");
}

void Server::handleUploadBurnout(
    const std::shared_ptr<ClientContext> &context,
    const std::string &body)
{
    try {
        LOG_DEBUG("Handling burnout upload request");

        auto content_type_it = context->headers.find("content-type");
        if (content_type_it == context->headers.end() ||
            content_type_it->second.find("multipart/form-data") == std::string::npos) {
            sendErrorResponse(context->fd, 400, "Expected multipart form data");
            return;
        }

        std::string boundary = extractBoundary(content_type_it->second);
        if (boundary.empty()) {
            sendErrorResponse(context->fd, 400, "Invalid multipart data");
            return;
        }

        auto form_data = parseMultipartFormData(body, boundary);
        auto file_it = form_data.find("file");
        if (file_it == form_data.end() || file_it->second.empty()) {
            sendErrorResponse(context->fd, 400, "No file provided");
            return;
        }

        std::string filename = form_data.contains("filename") ? form_data["filename"] : "uploaded_file";
        if (filename.empty()) {
            filename = "uploaded_file";
        }

        if (!file_processor_ || !file_processor_->allowed_file(filename)) {
            sendErrorResponse(context->fd, 400, "Invalid file type");
            return;
        }

        // Use the burnout upload handler
        std::string task_id = handleUploadBurnoutCommon(file_it->second, filename);
        LOG_INFO("Burnout upload accepted from client {}, task_id: {}", context->fd, task_id);

        sendHttpResponse(context->fd, 202, "application/json",
                         fmt::format(R"({{"task_id": "{}", "mode": "burnout"}})", task_id));
    } catch (const std::exception &e) {
        LOG_ERROR("Exception in burnout upload from client {}: {}", context->fd, e.what());
        sendErrorResponse(context->fd, 500, "Internal server error");
    }
}

void Server::handleBurnoutAnalyze(
    const std::shared_ptr<ClientContext> &context,
    const std::string &body)
{
    try {
        LOG_DEBUG("Handling burnout analysis request");
        
        auto json_body = nlohmann::json::parse(body);
        
        std::string task_id = json_body.value("task_id", "");
        if (task_id.empty()) {
            sendErrorResponse(context->fd, 400, "task_id required");
            return;
        }
        
        // Get emotion result
        auto &task_manager = TaskManager::instance();
        auto emotion_result = task_manager.get_task_status(task_id);
        
        if (!emotion_result) {
            sendErrorResponse(context->fd, 404, "Task not found");
            return;
        }
        
        // Get baseline if user_id provided
        nlohmann::json baseline;
        if (json_body.contains("user_id")) {
            std::string user_id = json_body["user_id"];
            baseline = file_processor_->get_user_baseline(user_id);
        }
        
        // Run burnout analysis
        auto burnout_result = file_processor_->analyze_burnout_from_result(*emotion_result, baseline);
        
        sendHttpResponse(context->fd, 200, "application/json", burnout_result.toJson().dump());
        
    } catch (const nlohmann::json::parse_error &e) {
        LOG_ERROR("JSON parse error in burnout analyze: {}", e.what());
        sendErrorResponse(context->fd, 400, "Invalid JSON");
    } catch (const std::exception &e) {
        LOG_ERROR("Exception in burnout analyze: {}", e.what());
        sendErrorResponse(context->fd, 500, fmt::format("Internal server error: {}", e.what()));
    }
}

void Server::handleBurnoutBaseline(
    const std::shared_ptr<ClientContext> &context,
    const std::string &body)
{
    try {
        LOG_DEBUG("Handling burnout baseline save request");
        
        auto json_body = nlohmann::json::parse(body);
        
        std::string user_id = json_body.value("user_id", "");
        if (user_id.empty()) {
            sendErrorResponse(context->fd, 400, "user_id required");
            return;
        }
        
        if (!json_body.contains("baseline") || !json_body["baseline"].is_object()) {
            sendErrorResponse(context->fd, 400, "baseline object required");
            return;
        }
        
        // Save baseline
        file_processor_->save_user_baseline(user_id, json_body["baseline"]);
        
        sendHttpResponse(context->fd, 200, "application/json", 
                         R"({"status": "baseline_saved", "message": "Baseline saved successfully"})");
        
    } catch (const nlohmann::json::parse_error &e) {
        LOG_ERROR("JSON parse error in burnout baseline: {}", e.what());
        sendErrorResponse(context->fd, 400, "Invalid JSON");
    } catch (const std::exception &e) {
        LOG_ERROR("Exception in burnout baseline: {}", e.what());
        sendErrorResponse(context->fd, 500, fmt::format("Internal server error: {}", e.what()));
    }
}

void Server::handleBurnoutBaselineGet(
    const std::shared_ptr<ClientContext> &context)
{
    try {
        LOG_DEBUG("Handling burnout baseline get request");
        
        // Get user_id from query params
        auto user_id_it = context->params.find("user_id");
        if (user_id_it == context->params.end() || user_id_it->second.empty()) {
            sendErrorResponse(context->fd, 400, "user_id query parameter required");
            return;
        }
        
        std::string user_id = user_id_it->second;
        auto baseline = file_processor_->get_user_baseline(user_id);
        
        if (baseline.empty()) {
            sendErrorResponse(context->fd, 404, "Baseline not found for user");
            return;
        }
        
        nlohmann::json response = {
            {"user_id", user_id},
            {"baseline", baseline}
        };
        
        sendHttpResponse(context->fd, 200, "application/json", response.dump());
        
    } catch (const std::exception &e) {
        LOG_ERROR("Exception in burnout baseline get: {}", e.what());
        sendErrorResponse(context->fd, 500, fmt::format("Internal server error: {}", e.what()));
    }
}

// Common handlers
std::string Server::handleUploadCommon(const std::string &file_content, 
                                          const std::string &filename, 
                                          bool realtime)
{
    auto &config = Config::instance();
    std::string task_id = DragonflyManager::generate_uuid();
    
    // Get extension
    std::string extension;
    size_t dot_pos = filename.find_last_of('.');
    if (dot_pos != std::string::npos) {
        extension = filename.substr(dot_pos);
    } else {
        extension = ".bin";
    }
    
    std::string safe_filename = task_id + extension;
    std::string storage_path = "uploads/" + safe_filename;
    
    LOG_INFO("Saving file to storage: {} (original: {})", storage_path, filename);
    LOG_INFO("Storage base path: ./uploads, full path will be: /emotionai/./uploads/{}", storage_path);
    
    if (!file_storage_->saveFile(file_content, storage_path))
    {
        throw std::runtime_error("Failed to save uploaded file to storage");
    }
    
    // Verify file was saved
    if (!file_storage_->fileExists(storage_path))
    {
        throw std::runtime_error("Failed to verify uploaded file in storage");
    }
    
    LOG_INFO("File saved successfully to storage: {}, size: {} bytes", 
             storage_path, file_content.size());
    
    // For file processor, we still need a local path in the uploads folder
    fs::path local_path = upload_folder_ / safe_filename;
    
    // Copy from storage to local uploads folder for processing
    try {
        std::string file_content_from_storage = file_storage_->readFile(storage_path);
        fs::create_directories(local_path.parent_path());
        std::ofstream local_file(local_path, std::ios::binary);
        local_file.write(file_content_from_storage.data(), file_content_from_storage.size());
        local_file.close();
        LOG_INFO("Copied file to local path for processing: {}", local_path.string());
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to copy file to local path: {}", e.what());
        throw std::runtime_error("Failed to prepare file for processing");
    }
    
    // Check if we should use distributed processing
#ifdef WITH_CLUSTER
    if (config.cluster().enabled && distributed_task_manager_)
    {
        // Use distributed task queue
        std::string task_type = realtime ? "realtime_video" : "batch_processing";

        nlohmann::json task = {
            {"task_id", task_id},
            {"type", task_type},
            {"filename", filename},
            {"file_path", local_path.string()},  // Use local path for processing
            {"storage_path", storage_path},      // Store storage path for reference
            {"instance_id", instance_id_},
            {"created_at", std::chrono::system_clock::now().time_since_epoch().count()},
            {"status", "pending"},
            {"retry_count", 0}};

        std::string queue_name = realtime ? config.queue().realtime_queue_name : config.queue().batch_queue_name;

        if (distributed_task_manager_->submitTask(queue_name, task))
        {
            LOG_INFO("Task {} submitted to distributed queue: {}", task_id, queue_name);
            
            auto &task_manager = TaskManager::instance();
            task_manager.set_task_status(task_id, {{"task_id", task_id},
                                                   {"progress", 0},
                                                   {"message", realtime ? "Queued for real-time processing" : "Queued for batch processing"},
                                                   {"error", nullptr},
                                                   {"complete", false},
                                                   {"mode", realtime ? "realtime" : "batch"},
                                                   {"instance_id", instance_id_},
                                                   {"queued", true},
                                                   {"storage_path", storage_path},
                                                   {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                                                                     std::chrono::system_clock::now().time_since_epoch())
                                                                     .count()}});

            return task_id;
        }
    }
#endif

    // Fallback to local processing
    auto *file_processor = file_processor_.get();

    thread_pool_->enqueue([this, file_processor, task_id, local_path, safe_filename, realtime, storage_path]()
                          {
        try
        {
            LOG_INFO("Starting {} processing for task: {}", realtime ? "real-time" : "background", task_id);
            
            auto& task_manager = TaskManager::instance();
            
            task_manager.set_task_status(task_id, {
                {"task_id", task_id},
                {"progress", 0},
                {"message", realtime ? "Starting real-time video processing" : "Starting file processing"},
                {"error", nullptr},
                {"complete", false},
                {"mode", realtime ? "realtime" : "batch"},
                {"instance_id", instance_id_},
                {"storage_path", storage_path},
                {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count()}
            });
            
            if (realtime)
            {
                file_processor->process_video_realtime(task_id, local_path.string(), safe_filename);
            }
            else
            {
                file_processor->process_file(task_id, local_path.string(), safe_filename);
            }
            
            LOG_INFO("Processing completed for task: {}", task_id);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("Processing failed for task {}: {}", task_id, e.what());
            
            try {
                auto& task_manager = TaskManager::instance();
                task_manager.set_task_status(task_id, {
                    {"task_id", task_id},
                    {"progress", 0},
                    {"message", "Processing failed"},
                    {"error", e.what()},
                    {"complete", true},
                    {"mode", realtime ? "realtime" : "batch"},
                    {"instance_id", instance_id_},
                    {"storage_path", storage_path},
                    {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count()}
                });
            } catch (const std::exception& db_error) {
                LOG_ERROR("Failed to update error status for task {}: {}", task_id, db_error.what());
            }
        } });

    return task_id;
}

std::string Server::handleUploadBurnoutCommon(const std::string &file_content, const std::string &filename)
{
    auto &config = Config::instance();
    std::string task_id = DragonflyManager::generate_uuid();
    
    // Get extension
    std::string extension;
    size_t dot_pos = filename.find_last_of('.');
    if (dot_pos != std::string::npos) {
        extension = filename.substr(dot_pos);
    } else {
        extension = ".bin";
    }
    
    std::string safe_filename = task_id + extension;
    std::string storage_path = "uploads/" + safe_filename;
    
    LOG_INFO("Saving file to storage for burnout analysis: {} (original: {})", storage_path, filename);
    
    if (!file_storage_->saveFile(file_content, storage_path)) {
        throw std::runtime_error("Failed to save uploaded file to storage");
    }
    
    // Verify file was saved
    if (!file_storage_->fileExists(storage_path)) {
        throw std::runtime_error("Failed to verify uploaded file in storage");
    }
    
    LOG_INFO("File saved successfully to storage: {}, size: {} bytes", 
             storage_path, file_content.size());
    
    // For file processor, we still need a local path in the uploads folder
    fs::path local_path = upload_folder_ / safe_filename;
    
    // Copy from storage to local uploads folder for processing
    try {
        std::string file_content_from_storage = file_storage_->readFile(storage_path);
        fs::create_directories(local_path.parent_path());
        std::ofstream local_file(local_path, std::ios::binary);
        local_file.write(file_content_from_storage.data(), file_content_from_storage.size());
        local_file.close();
        LOG_INFO("Copied file to local path for burnout processing: {}", local_path.string());
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to copy file to local path: {}", e.what());
        throw std::runtime_error("Failed to prepare file for processing");
    }
    
    // Process with burnout analysis
    auto *file_processor = file_processor_.get();
    
    thread_pool_->enqueue([this, file_processor, task_id, local_path, safe_filename, storage_path]() {
        try {
            LOG_INFO("Starting burnout analysis for task: {}", task_id);
            
            auto& task_manager = TaskManager::instance();
            
            // Initial status
            task_manager.set_task_status(task_id, {
                {"task_id", task_id},
                {"progress", 10},
                {"message", "Processing audio for burnout analysis"},
                {"error", nullptr},
                {"complete", false},
                {"mode", "burnout"},
                {"instance_id", instance_id_},
                {"storage_path", storage_path},
                {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count()}
            });
            
            // Process with burnout - this calls Audio::process_audio_with_burnout
            auto result = file_processor->process_audio_with_burnout(task_id, local_path.string(), safe_filename);
            
            // ============ CRITICAL FIX: Mark as COMPLETE ============
            task_manager.set_task_status(task_id, {
                {"task_id", task_id},
                {"progress", 100},
                {"message", "Burnout analysis complete"},
                {"error", nullptr},
                {"complete", true},  // ← THIS IS WHAT THE FRONTEND WAITS FOR
                {"mode", "burnout"},
                {"instance_id", instance_id_},
                {"storage_path", storage_path},
                {"type", "audio_burnout"},
                {"result", result},
                {"duration", result.value("duration", 0.0)},
                {"sample_rate", result.value("sample_rate", 0)},
                {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count()}
            });
            // ========================================================
            
            LOG_INFO("Burnout analysis completed for task: {}", task_id);
            
        } catch (const std::exception &e) {
            LOG_ERROR("Burnout analysis failed for task {}: {}", task_id, e.what());
            
            try {
                auto& task_manager = TaskManager::instance();
                task_manager.set_task_status(task_id, {
                    {"task_id", task_id},
                    {"progress", 0},
                    {"message", "Burnout analysis failed"},
                    {"error", e.what()},
                    {"complete", true},
                    {"mode", "burnout"},
                    {"instance_id", instance_id_},
                    {"storage_path", storage_path},
                    {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count()}
                });
            } catch (const std::exception& db_error) {
                LOG_ERROR("Failed to update error status for task {}: {}", task_id, db_error.what());
            }
        }
    });

    return task_id;
}

std::string Server::handleSubmitApplicationCommon(const std::string &body)
{
    if (!dragonfly_manager_)
    {
        throw std::runtime_error("Server not properly initialized");
    }

    json application_data = json::parse(body);
    validateJsonDocument(application_data);

    std::string application_id = dragonfly_manager_->save_application(application_data.dump());
    return application_id; // Return the application_id
}

void Server::validateJsonDocument(const nlohmann::json &json)
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

std::string Server::getMimeType(const std::string &filename) const
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

// Multipart form data parsing implementation
std::map<std::string, std::string> Server::parseMultipartFormData(const std::string &body, const std::string &boundary)
{
    std::map<std::string, std::string> result;

    if (body.empty() || boundary.empty())
    {
        LOG_ERROR("Empty body or boundary in multipart data");
        return result;
    }

    LOG_DEBUG("Parsing multipart data, body size: {}, boundary: '{}'", body.size(), boundary);

    size_t pos = 0;

    // Find first boundary
    size_t boundary_pos = body.find(boundary);
    if (boundary_pos == std::string::npos)
    {
        LOG_ERROR("First boundary not found");
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
            LOG_ERROR("Headers end not found");
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
                LOG_DEBUG("Found field '{}' with data size: {}", field_name, field_data.size());
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
            LOG_DEBUG("Found field '{}' with data size: {}", field_name, field_data.size());
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

std::string Server::extractBoundary(const std::string &content_type)
{
    size_t boundary_pos = content_type.find("boundary=");
    if (boundary_pos == std::string::npos)
    {
        LOG_ERROR("No boundary found in Content-Type: {}", content_type);
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

    LOG_DEBUG("Extracted boundary: '{}'", boundary);
    return "--" + boundary;
}

std::string Server::collectMetrics()
{
    return MetricsCollector::instance().collectMetrics();
}

void Server::updateRequestMetrics(const std::string &method, const std::string &endpoint,
                                      int status_code, double duration_seconds)
{
    MetricsCollector::instance().recordRequest(method, endpoint, status_code, duration_seconds);
}

std::string Server::generateInstanceId()
{
    auto &config = Config::instance();

    if (config.cluster().instance_id != "auto")
    {
        return config.cluster().instance_id;
    }

    // Generate UUID for this instance
    return DragonflyManager::generate_uuid();
}

void Server::startDistributedTaskWorkers()
{
#ifdef WITH_CLUSTER
    auto &config = Config::instance();
    if (!config.cluster().enabled || !distributed_task_manager_)
    {
        return;
    }

    workers_running_.store(true);

    // Start batch task workers
    unsigned int num_workers = std::max(1u, std::thread::hardware_concurrency() / 2);

    for (unsigned int i = 0; i < num_workers; ++i)
    {
        task_worker_threads_.emplace_back([this, i, &config]()
                                          {
            LOG_INFO("Distributed task worker {} started", i);
            
            while (workers_running_.load()) {
                try {
                    // Try to get next batch task
                    auto task = distributed_task_manager_->getNextTask(
                        config.queue().batch_queue_name,
                        config.queue().visibility_timeout
                    );
                    
                    if (task) {
                        processDistributedTask(*task);
                    } else {
                        // No task available, sleep briefly
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(config.queue().poll_interval_ms)
                        );
                    }
                } catch (const std::exception& e) {
                    LOG_ERROR("Error in distributed task worker {}: {}", i, e.what());
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
            }
            
            LOG_INFO("Distributed task worker {} stopped", i); });
    }

    LOG_INFO("Started {} distributed task workers", num_workers);
#endif
}

void Server::stopDistributedTaskWorkers()
{
    workers_running_.store(false);

    for (auto &thread : task_worker_threads_)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }
    task_worker_threads_.clear();

    LOG_INFO("Distributed task workers stopped");
}

void Server::processDistributedTask(const nlohmann::json &task)
{
#ifdef WITH_CLUSTER
    std::string task_id = task["task_id"];
    std::string task_type = task["type"];
    std::string file_path = task["file_path"];
    std::string filename = task["filename"];

    LOG_INFO("Processing distributed task {}: {} ({})", task_id, filename, task_type);

    try
    {
        // Update task status to processing
        auto &task_manager = TaskManager::instance();
        task_manager.set_task_status(task_id, {{"task_id", task_id},
                                               {"progress", 10},
                                               {"message", "Processing started in distributed worker"},
                                               {"error", nullptr},
                                               {"complete", false},
                                               {"mode", task_type == "realtime_video" ? "realtime" : "batch"},
                                               {"instance_id", instance_id_},
                                               {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                                                                 std::chrono::system_clock::now().time_since_epoch())
                                                                 .count()}});

        // Process the file based on task type
        if (task_type == "realtime_video")
        {
            file_processor_->process_video_realtime(task_id, file_path, filename);
        }
        else
        {
            file_processor_->process_file(task_id, file_path, filename);
        }

        // Mark task as complete
        if (distributed_task_manager_)
        {
            distributed_task_manager_->markTaskComplete(task_id);
        }

        LOG_INFO("Distributed task {} completed successfully", task_id);
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Distributed task {} failed: {}", task_id, e.what());

        // Update task status with error
        try
        {
            auto &task_manager = TaskManager::instance();
            task_manager.set_task_status(task_id, {{"task_id", task_id},
                                                   {"progress", 0},
                                                   {"message", "Processing failed in distributed worker"},
                                                   {"error", e.what()},
                                                   {"complete", true},
                                                   {"mode", task_type == "realtime_video" ? "realtime" : "batch"},
                                                   {"instance_id", instance_id_},
                                                   {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                                                                     std::chrono::system_clock::now().time_since_epoch())
                                                                     .count()}});

            // Mark task as failed in distributed queue
            if (distributed_task_manager_)
            {
                distributed_task_manager_->markTaskFailed(task_id, e.what());
            }
        }
        catch (const std::exception &db_error)
        {
            LOG_ERROR("Failed to update error status for task {}: {}", task_id, db_error.what());
        }
    }
#endif
}