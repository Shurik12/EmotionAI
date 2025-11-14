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

namespace fs = std::filesystem;
using json = nlohmann::json;

WebServer::WebServer()
	: dragonfly_manager_(nullptr),
	  file_processor_(nullptr)
{
}

WebServer::~WebServer()
{
	try
	{
		stop();
		waitForCompletion();

		// Stop task batcher
		if (task_batcher_)
		{
			task_batcher_->stop();
		}
	}
	catch (...)
	{
		LOG_ERROR("Exception during server shutdown");
	}
}

void WebServer::waitForCompletion()
{
	std::unique_lock<std::mutex> lock(task_mutex_);

	// Wait for all background threads to complete with a timeout
	auto timeout = std::chrono::seconds(30); // 30 second timeout
	auto condition = [this]()
	{ return background_threads_.empty(); };

	if (!completion_cv_.wait_for(lock, timeout, condition))
	{
		LOG_WARN("Timeout waiting for background threads to complete");

		// Force cleanup of any remaining threads
		for (auto &[task_id, thread] : background_threads_)
		{
			if (thread.joinable())
			{
				thread.detach(); // Let them finish on their own
			}
		}
		background_threads_.clear();
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

void WebServer::initializeComponents()
{
	LOG_INFO("Initializing DragonflyManager and FileProcessor...");
	try
	{
		// Create DragonflyManager as shared_ptr
		dragonfly_manager_ = std::make_shared<DragonflyManager>();
		dragonfly_manager_->initialize();
		LOG_INFO("DragonflyManager initialized successfully");

		// Initialize TaskManager with DragonflyManager
		auto &task_manager = TaskManager::instance();
		task_manager.set_dragonfly_manager(dragonfly_manager_);
		LOG_INFO("TaskManager initialized successfully");

		// Pass shared_ptr to FileProcessor
		file_processor_ = std::make_unique<FileProcessor>(dragonfly_manager_);
		LOG_INFO("FileProcessor initialized successfully");

		// Initialize task batcher
		initializeTaskBatcher();
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Failed to initialize components: {}", e.what());
		throw;
	}
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

// duplication
void WebServer::loadConfiguration()
{
	auto &config = Config::instance();
	log_folder_ = config.paths().logs;
	static_files_root_ = config.paths().frontend;
	upload_folder_ = config.paths().upload;
	results_folder_ = config.paths().results;
}

void WebServer::ensureDirectoriesExist()
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

void WebServer::setupRoutes()
{
	// API Routes
	svr_.Post("/api/upload_realtime", [this](const httplib::Request &req, httplib::Response &res)
			  {
		// Similar to handleUpload but for real-time processing
		try {
			if (!req.form.has_file("file")) {
				res.status = 400;
				res.set_content(R"({"error": "No file provided"})", "application/json");
				return;
			}

			const auto &file = req.form.get_file("file");
			if (file.filename.empty()) {
				res.status = 400;
				res.set_content(R"({"error": "No file selected"})", "application/json");
				return;
			}

			if (!file_processor_->allowed_file(file.filename)) {
				res.status = 400;
				res.set_content(R"({"error": "Invalid file type"})", "application/json");
				return;
			}

			std::string filename = file.filename;
			std::string task_id = DragonflyManager::generate_uuid();
			fs::path filepath = upload_folder_ / (task_id + "_" + filename);

			// Save the file
			std::ofstream out_file(filepath, std::ios::binary);
			out_file.write(file.content.data(), file.content.size());
			out_file.close();

			// Process file in real-time mode
			std::lock_guard<std::mutex> lock(task_mutex_);
			background_threads_[task_id] = std::thread(
				[this, task_id, filepath, filename]() {
					file_processor_->process_video_realtime(task_id, filepath.string(), filename);
					std::lock_guard<std::mutex> lock(task_mutex_);
					background_threads_.erase(task_id);
				});
			background_threads_[task_id].detach();

			res.status = 202;
			res.set_content(fmt::format(R"({{"task_id": "{}", "mode": "realtime"}})", task_id), "application/json");

		} catch (const std::exception &e) {
			LOG_ERROR("Exception in real-time upload: {}", e.what());
			res.status = 500;
			res.set_content(R"({"error": "Internal server error"})", "application/json");
		} });

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

		if (!file_processor_ || !file_processor_->allowed_file(file.filename))
		{
			res.status = 400;
			res.set_content(R"({"error": "Invalid file type"})", "application/json");
			return;
		}

		std::string filename = file.filename;
		std::string task_id = DragonflyManager::generate_uuid();
		fs::path filepath = upload_folder_ / (task_id + "_" + filename);

		// Save the file
		std::ofstream out_file(filepath, std::ios::binary);
		out_file.write(file.content.data(), file.content.size());
		out_file.close();

		auto dragonfly_manager = dragonfly_manager_;
		auto *file_processor = file_processor_.get();

		// Create the thread first, then add it to the map
		std::thread background_thread = std::thread(
			[this, dragonfly_manager, file_processor, task_id, filepath, filename]()
			{
				try
				{
					LOG_INFO("Starting background processing for task: {}", task_id);

					// Set initial status using the shared_ptr
					nlohmann::json initial_status = {
						{"task_id", task_id},
						{"status", "processing"},
						{"progress", 0},
						{"message", "Starting file processing"}};
					dragonfly_manager->set_task_status(task_id, initial_status);

					// Process the file
					file_processor->process_file(task_id, filepath.string(), filename);

					LOG_INFO("Background processing completed for task: {}", task_id);
				}
				catch (const std::exception &e)
				{
					LOG_ERROR("Background processing failed for task {}: {}", task_id, e.what());

					// Set error status using the shared_ptr
					try
					{
						nlohmann::json error_status = {
							{"task_id", task_id},
							{"status", "error"},
							{"progress", 0},
							{"message", std::string("Processing failed: ") + e.what()}};
						dragonfly_manager->set_task_status(task_id, error_status);
					}
					catch (const std::exception &redis_error)
					{
						LOG_ERROR("Failed to update Redis status for task {}: {}", task_id, redis_error.what());
					}
				}

				// Remove thread from map when done - use a separate method to avoid deadlock
				this->removeBackgroundThread(task_id);
			});

		// Now add the thread to the map with the lock held
		std::lock_guard<std::mutex> lock(task_mutex_);
		background_threads_[task_id] = std::move(background_thread);

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

void WebServer::removeBackgroundThread(const std::string &task_id)
{
	std::lock_guard<std::mutex> lock(task_mutex_);
	auto it = background_threads_.find(task_id);
	if (it != background_threads_.end())
	{
		if (it->second.joinable())
		{
			it->second.detach(); // Let the thread finish on its own
		}
		background_threads_.erase(it);
	}
}

void WebServer::handleProgress(const httplib::Request &req, httplib::Response &res, const std::string &task_id)
{
	try
	{
		// Use TaskManager with caching
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

void WebServer::handleSubmitApplication(const httplib::Request &req, httplib::Response &res)
{
	try
	{
		if (!dragonfly_manager_)
		{
			res.status = 500;
			res.set_content(R"({"error": "Server not properly initialized"})", "application/json");
			return;
		}

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

		std::string application_id = dragonfly_manager_->save_application(application_data.dump());

		res.status = 201;
		res.set_content(fmt::format(R"({{"application_id": "{}"}})", application_id), "application/json");
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Error submitting application: {}", e.what());
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
		LOG_ERROR("Exception serving result file {}: {}", filename, e.what());
		res.status = 500;
		res.set_content("Internal server error", "text/plain");
	}
}

void WebServer::handleHealthCheck(const httplib::Request &req, httplib::Response &res)
{
	try
	{
		res.set_content(R"({"status": "healthy"})", "application/json");
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Health check failed: {}", e.what());
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
		LOG_ERROR("Exception serving static file {}: {}", filename, e.what());
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

void WebServer::initializeTaskBatcher()
{
	task_batcher_ = std::make_unique<TaskBatcher>(50, std::chrono::milliseconds(100));

	task_batcher_->set_batch_callback([this](const auto &batch)
									  {
        auto& task_manager = TaskManager::instance();
        task_manager.batch_update_status(batch); });

	task_batcher_->start();
	LOG_INFO("TaskBatcher initialized with batch_size=50, timeout=100ms");
}