#include <iostream>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <fmt/format.h>
#include <common/Logging.h>
#include <common/Config.h>

namespace fs = std::filesystem;

namespace Common
{
    // Initialize the unified logger
    void multi_sink_example(const std::string &log_file)
    {
        try
        {
            // Get the config instance
            auto &config = Common::Config::instance();

            // Check if logging is configured
            if (config.logFolder().empty())
            {
                throw std::runtime_error("Log folder not configured");
            }

            // Setup sinks
            auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console_sink->set_level(spdlog::level::info);

            // Create full log file path using the configured log folder
            // Convert both paths to std::filesystem::path before using /
            fs::path log_folder_path = config.logFolder();
            fs::path log_file_path = log_file;
            fs::path full_log_path = log_folder_path / log_file_path;

            auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(full_log_path.string(), true);
            file_sink->set_level(spdlog::level::debug);

            // Create and register logger
            std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
            auto logger = std::make_shared<spdlog::logger>("main", sinks.begin(), sinks.end());

            spdlog::set_default_logger(logger);
            spdlog::set_level(spdlog::level::debug);

            spdlog::info("Logger initialized. Log file: {}", full_log_path.string());
        }
        catch (const std::exception &ex)
        {
            std::cerr << "Log initialization failed: " << ex.what() << std::endl;
            throw;
        }
    }

    // HTTP request/response logger implementation
    void log_request_response(const httplib::Request &req, const httplib::Response &res)
    {
        // Build log message
        std::string message1;
        message1 += fmt::format("{} {} {}", req.method, req.path, req.version);

        std::string message;
        message += fmt::format("{} {} {}\n", req.method, req.path, req.version);

        // Add query parameters if any
        if (!req.params.empty())
        {
            message += "Query: ";
            for (auto it = req.params.begin(); it != req.params.end(); ++it)
            {
                message += (it == req.params.begin() ? "?" : "&") + it->first + "=" + it->second;
            }
            message += "\n";
        }

        // Add headers
        for (const auto &header : req.headers)
        {
            message += fmt::format("{}: {}\n", header.first, header.second);
        }

        // Add response info
        message += fmt::format("Response: {} {}\n", res.version, res.status);
        for (const auto &header : res.headers)
        {
            message += fmt::format("{}: {}\n", header.first, header.second);
        }

        // Log with spdlog
        spdlog::info("HTTP Request/Response: {}", message1);

        // Optionally log body if needed
        if (!res.body.empty() && res.body.size() < 1024)
        {
            spdlog::debug("Response body: {}", res.body);
        }
    }
}