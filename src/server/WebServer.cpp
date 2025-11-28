#include <filesystem>
#include <fstream>
#include <vector>
#include <utility>
#include <thread>
#include <mutex>

#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <server/WebServer.h>
#include <config/Config.h>
#include <logging/Logger.h>
#include <common/uuid.h>
#include <db/DragonflyManager.h>
#include <emotionai/FileProcessor.h>
#include <db/TaskManager.h>
#include <storage/FileStorageFactory.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

WebServer::WebServer() = default;

WebServer::~WebServer()
{
    try
    {
        stop();
    }
    catch (...)
    {
        LOG_ERROR("Exception during server shutdown");
    }
}

void WebServer::initialize()
{
    LOG_INFO("WebServer::initialize");
    loadConfiguration();
    ensureDirectoriesExist();
    initializeComponents();
    setupRoutes();
}

void WebServer::start()
{
    auto &config = Config::instance();
    std::string host = config.server().host;
    int port = config.server().port;

    svr_.set_logger([](const auto &req, const auto &res)
                    { LOG_INFO("{} {} -> {} {}", req.method, req.path, res.status, req.remote_addr); });

    LOG_INFO("Starting server on {}:{}", host, port);
    if (!svr_.listen(host.c_str(), port))
    {
        throw std::runtime_error(fmt::format("Failed to start server on {}:{}", host, port));
    }
}

void WebServer::stop() noexcept
{
    try
    {
        svr_.stop();
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Error stopping server: {}", e.what());
    }
}

void WebServer::setupRoutes()
{
    // API Routes
    svr_.Post("/api/upload_realtime", [this](const httplib::Request &req, httplib::Response &res)
              { handleUploadRealtime(req, res); });

    svr_.Post("/api/upload", [this](const httplib::Request &req, httplib::Response &res)
              { handleUpload(req, res); });

    svr_.Post("/api/batch_progress", [this](const httplib::Request &req, httplib::Response &res)
              { handleBatchProgress(req, res); });

    svr_.Get("/api/progress/:task_id", [this](const httplib::Request &req, httplib::Response &res)
             { handleProgress(req, res, req.path_params.at("task_id")); });

    svr_.Post("/api/submit_application", [this](const httplib::Request &req, httplib::Response &res)
              { handleSubmitApplication(req, res); });

    svr_.Get("/api/results/:filename", [this](const httplib::Request &req, httplib::Response &res)
             { handleServeResult(req, res, req.path_params.at("filename")); });

    svr_.Get("/api/health", [this](const httplib::Request &req, httplib::Response &res)
             { handleHealthCheck(req, res); });

    // Storage info endpoint
    svr_.Get("/api/storage/info", [this](const httplib::Request &req, httplib::Response &res)
             { handleStorageInfo(req, res); });

    // Static files
    svr_.Get("/static/:filename", [this](const httplib::Request &req, httplib::Response &res)
             { handleServeStatic(req, res, req.path_params.at("filename")); });

    // React files and client-side routing
    svr_.Get("/:filename", [this](const httplib::Request &req, httplib::Response &res)
             { handleServeReactFile(req, res, req.path_params.at("filename")); });

    // Root route
    svr_.Get("/", [this](const httplib::Request &req, httplib::Response &res)
             { handleRoot(req, res); });

    // Set CORS headers for all responses
    svr_.set_pre_routing_handler([](const auto &req, auto &res)
                                 {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        return httplib::Server::HandlerResponse::Unhandled; });

    // Handle OPTIONS requests for CORS
    svr_.Options(".*", [](const httplib::Request &, httplib::Response &res)
                 { res.status = 200; });
}

void WebServer::handleUpload(const httplib::Request &req, httplib::Response &res)
{
    try
    {
        if (!req.form.has_file("file"))
        {
            res.status = 400;
            res.set_content(R"({"error": "No file provided"})", "application/json");
            return;
        }

        const auto &file = req.form.get_file("file");
        if (file.filename.empty())
        {
            res.status = 400;
            res.set_content(R"({"error": "No file selected"})", "application/json");
            return;
        }

        if (!file_processor_ || !file_processor_->allowed_file(file.filename))
        {
            res.status = 400;
            res.set_content(R"({"error": "Invalid file type"})", "application/json");
            return;
        }

        std::string filename = file.filename;

        // Use the common upload handler and get the task_id
        std::string task_id = handleUploadCommon(file.content, filename, false);

        LOG_INFO("Upload accepted and queued for processing, task_id: {}", task_id);
        res.status = 202;
        res.set_content(fmt::format(R"({{"task_id": "{}"}})", task_id), "application/json");
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Exception in handleUpload: {}", e.what());
        res.status = 500;
        res.set_content(R"({"error": "Internal server error"})", "application/json");
    }
}

void WebServer::handleUploadRealtime(const httplib::Request &req, httplib::Response &res)
{
    try
    {
        if (!req.form.has_file("file"))
        {
            res.status = 400;
            res.set_content(R"({"error": "No file provided"})", "application/json");
            return;
        }

        const auto &file = req.form.get_file("file");
        if (file.filename.empty())
        {
            res.status = 400;
            res.set_content(R"({"error": "No file selected"})", "application/json");
            return;
        }

        if (!file_processor_ || !file_processor_->allowed_file(file.filename))
        {
            res.status = 400;
            res.set_content(R"({"error": "Invalid file type"})", "application/json");
            return;
        }

        std::string filename = file.filename;

        // Use the common upload handler with realtime flag and get the task_id
        std::string task_id = handleUploadCommon(file.content, filename, true);

        res.status = 202;
        res.set_content(fmt::format(R"({{"task_id": "{}", "mode": "realtime"}})", task_id), "application/json");
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Exception in real-time upload: {}", e.what());
        res.status = 500;
        res.set_content(R"({"error": "Internal server error"})", "application/json");
    }
}

void WebServer::handleSubmitApplication(const httplib::Request &req, httplib::Response &res)
{
    try
    {
        std::string application_id = handleSubmitApplicationCommon(req.body);

        res.status = 201;
        res.set_content(fmt::format(R"({{"application_id": "{}"}})", application_id), "application/json");
    }
    catch (const json::parse_error &e)
    {
        res.status = 400;
        res.set_content(R"({"error": "Invalid JSON"})", "application/json");
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Error submitting application: {}", e.what());
        res.status = 500;
        res.set_content(R"({"error": "Internal server error"})", "application/json");
    }
}

void WebServer::handleProgress(const httplib::Request &req, httplib::Response &res, const std::string &task_id)
{
    try
    {
        auto &task_manager = TaskManager::instance();
        auto status = task_manager.get_task_status(task_id);

        if (!status)
        {
            res.status = 404;
            res.set_content(R"({"error": "Task not found"})", "application/json");
            return;
        }

        res.set_content(status->dump(), "application/json");
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Exception in handleProgress for task {}: {}", task_id, e.what());
        res.status = 500;
        res.set_content(R"({"error": "Internal server error"})", "application/json");
    }
}

void WebServer::handleBatchProgress(const httplib::Request &req, httplib::Response &res)
{
    try
    {
        auto task_ids_json = nlohmann::json::parse(req.body);
        if (!task_ids_json.is_array())
        {
            res.status = 400;
            res.set_content(R"({"error": "Expected array of task IDs"})", "application/json");
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

        auto &task_manager = TaskManager::instance();
        auto results = task_manager.batch_get_status(task_ids);

        nlohmann::json response;
        for (const auto &[task_id, status] : results)
        {
            response[task_id] = status;
        }

        res.set_content(response.dump(), "application/json");
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Batch progress error: {}", e.what());
        res.status = 500;
        res.set_content(R"({"error": "Internal server error"})", "application/json");
    }
}

void WebServer::handleServeResult(const httplib::Request &req, httplib::Response &res, const std::string &filename)
{
    try
    {
        // First try to serve from shared storage
        std::string storage_path = "results/" + filename;
        if (file_storage_->fileExists(storage_path))
        {
            std::vector<uint8_t> file_content = file_storage_->readFileBinary(storage_path);
            if (!file_content.empty())
            {
                std::string mime_type = getMimeType(filename);
                res.set_content(reinterpret_cast<const char *>(file_content.data()),
                                file_content.size(), mime_type.c_str());
                LOG_DEBUG("Served result file from storage: {}", storage_path);
                return;
            }
        }

        // Fallback to local filesystem for backward compatibility
        fs::path file_path = results_folder_ / filename;
        if (fs::exists(file_path) && fs::is_regular_file(file_path))
        {
            res.set_file_content(file_path.string());
            return;
        }

        res.status = 404;
        res.set_content("File not found", "text/plain");
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Exception serving result file {}: {}", filename, e.what());
        res.status = 500;
        res.set_content("Internal server error", "text/plain");
    }
}

void WebServer::handleHealthCheck(const httplib::Request &req, httplib::Response &res)
{
    try
    {
        // Add storage health check
        bool storage_healthy = false;
        try
        {
            auto storage_info = file_storage_->getStorageInfo();
            storage_healthy = !storage_info.contains("error");
        }
        catch (...)
        {
            storage_healthy = false;
        }

        nlohmann::json health_status = {
            {"status", "healthy"},
            {"storage", storage_healthy ? "healthy" : "unhealthy"},
            {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::system_clock::now().time_since_epoch())
                              .count()}};

        res.set_content(health_status.dump(), "application/json");
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Health check failed: {}", e.what());
        res.status = 500;
        res.set_content(fmt::format(R"({{"status": "unhealthy", "error": "{}"}})", e.what()), "application/json");
    }
}

void WebServer::handleStorageInfo(const httplib::Request &req, httplib::Response &res)
{
    try
    {
        auto storage_info = file_storage_->getStorageInfo();
        res.set_content(storage_info.dump(), "application/json");
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Error getting storage info: {}", e.what());
        res.status = 500;
        res.set_content(fmt::format(R"({{"error": "{}"}})", e.what()), "application/json");
    }
}

void WebServer::handleServeStatic(const httplib::Request &req, httplib::Response &res, const std::string &filename)
{
    try
    {
        fs::path static_path = static_files_root_ / "static" / filename;

        if (fs::exists(static_path) && fs::is_regular_file(static_path))
        {
            res.set_file_content(static_path.string());
            return;
        }

        res.status = 404;
        res.set_content(R"({"error": "File not found"})", "application/json");
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Exception serving static file {}: {}", filename, e.what());
        res.status = 500;
        res.set_content("Internal server error", "text/plain");
    }
}

void WebServer::handleServeReactFile(const httplib::Request &req, httplib::Response &res, const std::string &filename)
{
    try
    {
        if (isApiEndpoint(req.path))
        {
            res.status = 404;
            res.set_content(R"({"error": "Not found"})", "application/json");
            return;
        }

        fs::path file_path = static_files_root_ / filename;

        if (fs::exists(file_path) && fs::is_regular_file(file_path))
        {
            res.set_file_content(file_path.string());
            return;
        }

        handleRoot(req, res);
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Exception serving React file {}: {}", filename, e.what());
        res.status = 500;
        res.set_content("Internal server error", "text/plain");
    }
}

void WebServer::handleRoot(const httplib::Request &req, httplib::Response &res)
{
    try
    {
        fs::path index_path = static_files_root_ / "index.html";

        if (!fs::exists(index_path))
        {
            LOG_ERROR("Index file not found: {}", index_path.string());
            res.status = 404;
            res.set_content("Page not found", "text/plain");
            return;
        }

        res.set_file_content(index_path.string());
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Exception serving root: {}", e.what());
        res.status = 500;
        res.set_content("Internal server error", "text/plain");
    }
}