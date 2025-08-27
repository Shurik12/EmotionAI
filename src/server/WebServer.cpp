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
#include <common/Config.h>
#include <common/Logging.h>
#include <common/uuid.h>
#include <db/RedisManager.h>
#include <emotionai/FileProcessor.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

WebServer::WebServer()
	: redis_manager_(std::make_unique<db::RedisManager>()),
	  file_processor_(std::make_unique<EmotionAI::FileProcessor>(*redis_manager_))
{
	initialize();
}

WebServer::~WebServer()
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
	}
	catch (...)
	{
		spdlog::error("Exception during server shutdown");
	}
}

void WebServer::initialize()
{
	loadConfiguration();
	initializeLogging();
	ensureDirectoriesExist();
	setupRoutes();
}

void WebServer::start()
{
	svr_.set_logger([](const auto &req, const auto &res)
					{ Common::log_request_response(req, res); });

	spdlog::info("Starting server on 0.0.0.0:8080");
	if (!svr_.listen("0.0.0.0", 8080))
	{
		throw std::runtime_error("Failed to start server");
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
		spdlog::error("Error stopping server: {}", e.what());
	}
}

void WebServer::loadConfiguration()
{
	// Get the config instance
	auto &config = Common::Config::instance();

	// Load configuration from file
	if (!config.loadFromFile())
	{
		spdlog::warn("Failed to load configuration file, using defaults");
	}

	// Setup application environment
	if (!config.setupApplicationEnvironment())
	{
		throw std::runtime_error("Failed to setup application environment");
	}

	// Get the configured paths
	log_folder_ = config.logFolder();
	static_files_root_ = config.frontendBuildPath();
	upload_folder_ = config.uploadFolder();
	results_folder_ = config.resultsFolder();
}

void WebServer::initializeLogging()
{
	try
	{
		Common::multi_sink_example((log_folder_ / "multisink.log").string());
	}
	catch (const std::exception &e)
	{
		throw std::runtime_error(fmt::format("Failed to initialize logger: {}", e.what()));
	}
}

void WebServer::ensureDirectoriesExist()
{
	try
	{
		fs::create_directories(upload_folder_);
		fs::create_directories(results_folder_);
		fs::create_directories(static_files_root_);
		spdlog::info("Directories ensured: upload={}, results={}, static={}",
					 upload_folder_.string(), results_folder_.string(), static_files_root_.string());
	}
	catch (const std::exception &e)
	{
		spdlog::error("Failed to create directories: {}", e.what());
		throw;
	}
}

void WebServer::setupRoutes()
{
	// API Routes
	svr_.Post("/api/upload", [this](const httplib::Request &req, httplib::Response &res)
			  { handleUpload(req, res); });

	svr_.Get("/api/progress/:task_id", [this](const httplib::Request &req, httplib::Response &res)
			 { handleProgress(req, res, req.path_params.at("task_id")); });

	svr_.Post("/api/submit_application", [this](const httplib::Request &req, httplib::Response &res)
			  { handleSubmitApplication(req, res); });

	svr_.Get("/api/results/:filename", [this](const httplib::Request &req, httplib::Response &res)
			 { handleServeResult(req, res, req.path_params.at("filename")); });

	svr_.Get("/api/health", [this](const httplib::Request &req, httplib::Response &res)
			 { handleHealthCheck(req, res); });

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

		if (file_processor_->allowed_file(file.filename))
		{
			std::string filename = file.filename;
			std::string task_id = db::RedisManager::generate_uuid();
			fs::path filepath = upload_folder_ / (task_id + "_" + filename);

			// Save the file
			std::ofstream out_file(filepath, std::ios::binary);
			out_file.write(file.content.data(), file.content.size());
			out_file.close();

			// Process file in background thread
			std::lock_guard<std::mutex> lock(task_mutex_);
			background_threads_[task_id] = std::thread(
				[this, task_id, filepath, filename]()
				{
					file_processor_->process_file(task_id, filepath.string(), filename);
					// Remove thread from map when done
					std::lock_guard<std::mutex> lock(task_mutex_);
					background_threads_.erase(task_id);
				});
			background_threads_[task_id].detach();

			res.status = 202;
			res.set_content(fmt::format(R"({{"task_id": "{}"}})", task_id), "application/json");
		}
		else
		{
			res.status = 400;
			res.set_content(R"({"error": "Invalid file type"})", "application/json");
		}
	}
	catch (const std::exception &e)
	{
		spdlog::error("Exception in handleUpload: {}", e.what());
		res.status = 500;
		res.set_content(R"({"error": "Internal server error"})", "application/json");
	}
}

void WebServer::handleProgress(const httplib::Request &req, httplib::Response &res, const std::string &task_id)
{
	try
	{
		auto status = redis_manager_->get_task_status(task_id);
		if (!status)
		{
			res.status = 404;
			res.set_content(R"({"error": "Task not found"})", "application/json");
			return;
		}

		res.set_content(status.value(), "application/json");
	}
	catch (const std::exception &e)
	{
		spdlog::error("Exception in handleProgress for task {}: {}", task_id, e.what());
		res.status = 500;
		res.set_content(R"({"error": "Internal server error"})", "application/json");
	}
}

void WebServer::handleSubmitApplication(const httplib::Request &req, httplib::Response &res)
{
	try
	{
		json application_data;

		try
		{
			application_data = json::parse(req.body);
			validateJsonDocument(application_data);
		}
		catch (const json::parse_error &e)
		{
			res.status = 400;
			res.set_content(R"({"error": "Invalid JSON"})", "application/json");
			return;
		}
		catch (const std::exception &e)
		{
			res.status = 400;
			res.set_content(fmt::format(R"({{"error": "{}"}})", e.what()), "application/json");
			return;
		}

		std::string application_id = redis_manager_->save_application(application_data.dump());

		res.status = 201;
		res.set_content(fmt::format(R"({{"application_id": "{}"}})", application_id), "application/json");
	}
	catch (const std::exception &e)
	{
		spdlog::error("Error submitting application: {}", e.what());
		res.status = 500;
		res.set_content(R"({"error": "Internal server error"})", "application/json");
	}
}

void WebServer::handleServeResult(const httplib::Request &req, httplib::Response &res, const std::string &filename)
{
	try
	{
		fs::path file_path = results_folder_ / filename;

		if (!fs::exists(file_path) || !fs::is_regular_file(file_path))
		{
			res.status = 404;
			res.set_content("File not found", "text/plain");
			return;
		}

		res.set_file_content(file_path.string());
	}
	catch (const std::exception &e)
	{
		spdlog::error("Exception serving result file {}: {}", filename, e.what());
		res.status = 500;
		res.set_content("Internal server error", "text/plain");
	}
}

void WebServer::handleHealthCheck(const httplib::Request &req, httplib::Response &res)
{
	try
	{
		// redis_manager_->connection()->redisCommand("PING");
		res.set_content(R"({"status": "healthy"})", "application/json");
	}
	catch (const std::exception &e)
	{
		spdlog::error("Health check failed: {}", e.what());
		res.status = 500;
		res.set_content(fmt::format(R"({{"status": "unhealthy", "error": "{}"}})", e.what()), "application/json");
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
		spdlog::error("Exception serving static file {}: {}", filename, e.what());
		res.status = 500;
		res.set_content("Internal server error", "text/plain");
	}
}

void WebServer::handleServeReactFile(const httplib::Request &req, httplib::Response &res, const std::string &filename)
{
	try
	{
		// Don't interfere with API routes
		if (isApiEndpoint(req.path))
		{
			res.status = 404;
			res.set_content(R"({"error": "Not found"})", "application/json");
			return;
		}

		fs::path file_path = static_files_root_ / filename;

		// If it's a file that exists, serve it
		if (fs::exists(file_path) && fs::is_regular_file(file_path))
		{
			res.set_file_content(file_path.string());
			return;
		}

		// For React Router - serve index.html for all other routes
		handleRoot(req, res);
	}
	catch (const std::exception &e)
	{
		spdlog::error("Exception serving React file {}: {}", filename, e.what());
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
			spdlog::error("Index file not found: {}", index_path.string());
			res.status = 404;
			res.set_content("Page not found", "text/plain");
			return;
		}

		res.set_file_content(index_path.string());
	}
	catch (const std::exception &e)
	{
		spdlog::error("Exception serving root: {}", e.what());
		res.status = 500;
		res.set_content("Internal server error", "text/plain");
	}
}

bool WebServer::isApiEndpoint(const std::string &path)
{
	static const std::set<std::string> apiPrefixes = {
		"/api/", "/static/"};

	for (const auto &prefix : apiPrefixes)
	{
		if (path.find(prefix) == 0)
		{
			return true;
		}
	}
	return false;
}

void WebServer::cleanupFinishedThreads()
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

void WebServer::validateJsonDocument(const nlohmann::json &json)
{
	if (!json.is_object())
	{
		throw std::runtime_error("Expected JSON object");
	}

	// Add additional validation as needed
	if (json.empty())
	{
		throw std::runtime_error("Empty JSON object");
	}
}