#include <sstream>
#include <algorithm>
#include <fstream>
#include <sys/resource.h>

#include <config/Config.h>
#include <logging/Logger.h>
#include <common/uuid.h>
#include <db/TaskManager.h>
#include <emotionai/FileProcessor.h>
#include <metrics/MetricsCollector.h>
#include "BaseServer.h"

// Include cluster headers - use forward declarations in header, includes in cpp
#ifdef WITH_CLUSTER
#include <cluster/ClusterManager.h>
#include <cluster/DistributedTaskManager.h>
#endif

BaseServer::BaseServer()
	: dragonfly_manager_(nullptr),
	  file_processor_(nullptr),
	  instance_id_("")
{
}

void BaseServer::loadConfiguration()
{
	auto &config = Config::instance();
	log_folder_ = config.paths().logs;
	static_files_root_ = config.paths().frontend;
	upload_folder_ = config.paths().upload;
	results_folder_ = config.paths().results;
}

void BaseServer::ensureDirectoriesExist()
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

void BaseServer::initializeComponents()
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

		// Create FileProcessor
		file_processor_ = std::make_unique<FileProcessor>(dragonfly_manager_);
		LOG_INFO("FileProcessor initialized successfully");
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Failed to initialize components: {}", e.what());
		throw;
	}
}

void BaseServer::initializeCluster()
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

void BaseServer::startClusterServices()
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

void BaseServer::stopClusterServices()
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

void BaseServer::registerInstance()
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

void BaseServer::unregisterInstance()
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

void BaseServer::startDistributedTaskWorkers()
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

void BaseServer::stopDistributedTaskWorkers()
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

void BaseServer::processDistributedTask(const nlohmann::json &task)
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

std::string BaseServer::generateInstanceId()
{
	auto &config = Config::instance();

	if (config.cluster().instance_id != "auto")
	{
		return config.cluster().instance_id;
	}

	// Generate UUID for this instance
	return DragonflyManager::generate_uuid();
}

std::string BaseServer::handleUploadCommon(const std::string &file_content, const std::string &filename, bool realtime)
{
	auto &config = Config::instance();
	std::string task_id = DragonflyManager::generate_uuid();
	fs::path filepath = upload_folder_ / (task_id + "_" + filename);

	// Save the file
	std::ofstream out_file(filepath, std::ios::binary);
	out_file.write(file_content.data(), file_content.size());
	out_file.close();

	if (!fs::exists(filepath) || fs::file_size(filepath) == 0)
	{
		throw std::runtime_error("Failed to save uploaded file");
	}

	LOG_INFO("File saved successfully: {}, size: {} bytes", filepath.string(), fs::file_size(filepath));

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
			{"file_path", filepath.string()},
			{"instance_id", instance_id_}, // Creator instance
			{"created_at", std::chrono::system_clock::now().time_since_epoch().count()},
			{"status", "pending"},
			{"retry_count", 0}};

		std::string queue_name = realtime ? config.queue().realtime_queue_name : config.queue().batch_queue_name;

		if (distributed_task_manager_->submitTask(queue_name, task))
		{
			LOG_INFO("Task {} submitted to distributed queue: {}", task_id, queue_name);

			// Set initial status
			auto &task_manager = TaskManager::instance();
			task_manager.set_task_status(task_id, {{"task_id", task_id},
												   {"progress", 0},
												   {"message", realtime ? "Queued for real-time processing" : "Queued for batch processing"},
												   {"error", nullptr},
												   {"complete", false},
												   {"mode", realtime ? "realtime" : "batch"},
												   {"instance_id", instance_id_},
												   {"queued", true},
												   {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
																	 std::chrono::system_clock::now().time_since_epoch())
																	 .count()}});

			return task_id;
		}
		else
		{
			LOG_WARN("Failed to submit task to distributed queue, falling back to local processing");
			// Fall through to local processing
		}
	}
#endif

	// Fallback to local processing (original behavior)
	auto *file_processor = file_processor_.get();

	thread_pool_->enqueue([this, file_processor, task_id, filepath, filename, realtime]()
						  {
        try
        {
            LOG_INFO("Starting {} processing for task: {}", realtime ? "real-time" : "background", task_id);
            
            auto& task_manager = TaskManager::instance();
            
            // Set initial status
            task_manager.set_task_status(task_id, {
                {"task_id", task_id},
                {"progress", 0},
                {"message", realtime ? "Starting real-time video processing" : "Starting file processing"},
                {"error", nullptr},
                {"complete", false},
                {"mode", realtime ? "realtime" : "batch"},
                {"instance_id", instance_id_},
                {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count()}
            });
            
            // Process file
            if (realtime)
            {
                file_processor->process_video_realtime(task_id, filepath.string(), filename);
            }
            else
            {
                file_processor->process_file(task_id, filepath.string(), filename);
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
                    {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count()}
                });
            } catch (const std::exception& db_error) {
                LOG_ERROR("Failed to update error status for task {}: {}", task_id, db_error.what());
            }
        } });

	return task_id; // Return the task_id so the caller can use it in the response
}

std::string BaseServer::handleSubmitApplicationCommon(const std::string &body)
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

void BaseServer::validateJsonDocument(const nlohmann::json &json)
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

std::string BaseServer::getMimeType(const std::string &filename) const
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

bool BaseServer::isApiEndpoint(const std::string &path) const
{
	static const std::set<std::string> apiPrefixes = {"/api/", "/static/"};
	for (const auto &prefix : apiPrefixes)
	{
		if (path.find(prefix) == 0)
		{
			return true;
		}
	}
	return false;
}

// Multipart form data parsing implementation (moved from MultiplexingServer)
std::map<std::string, std::string> BaseServer::parseMultipartFormData(const std::string &body, const std::string &boundary)
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

std::string BaseServer::extractBoundary(const std::string &content_type)
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

std::string BaseServer::collectMetrics()
{
	return MetricsCollector::instance().collectMetrics();
}

void BaseServer::updateRequestMetrics(const std::string &method, const std::string &endpoint,
									  int status_code, double duration_seconds)
{
	MetricsCollector::instance().recordRequest(method, endpoint, status_code, duration_seconds);
}