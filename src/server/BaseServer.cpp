#include "BaseServer.h"
#include <config/Config.h>
#include <logging/Logger.h>
#include <common/uuid.h>
#include <db/TaskManager.h>
#include <emotionai/FileProcessor.h>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <sys/resource.h>

BaseServer::BaseServer()
	: dragonfly_manager_(nullptr), file_processor_(nullptr)
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
		// Create thread pool
		unsigned int num_threads = std::max(2u, std::thread::hardware_concurrency());
		thread_pool_ = std::make_unique<ThreadPool>(num_threads);
		LOG_INFO("ThreadPool initialized with {} threads", num_threads);

		// Create DragonflyManager
		dragonfly_manager_ = std::make_shared<DragonflyManager>();
		dragonfly_manager_->initialize();
		LOG_INFO("DragonflyManager initialized successfully");

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

std::string BaseServer::handleUploadCommon(const std::string &file_content, const std::string &filename, bool realtime)
{
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

	// Submit to thread pool
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
	std::stringstream metrics;

	// Process metrics
	struct rusage usage;
	getrusage(RUSAGE_SELF, &usage);

	// CPU metrics
	metrics << "# HELP process_cpu_seconds_total Total user and system CPU time spent in seconds.\n";
	metrics << "# TYPE process_cpu_seconds_total counter\n";
	metrics << "process_cpu_seconds_total "
			<< (usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1000000.0 +
				usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1000000.0)
			<< "\n";

	// Memory metrics
	metrics << "# HELP process_resident_memory_bytes Resident memory size in bytes.\n";
	metrics << "# TYPE process_resident_memory_bytes gauge\n";
	metrics << "process_resident_memory_bytes " << usage.ru_maxrss * 1024 << "\n";

	// Thread pool metrics
	if (thread_pool_)
	{
		metrics << "# HELP thread_pool_pending_tasks Number of pending tasks in thread pool.\n";
		metrics << "# TYPE thread_pool_pending_tasks gauge\n";
		metrics << "thread_pool_pending_tasks " << thread_pool_->get_pending_tasks() << "\n";

		metrics << "# HELP thread_pool_active_threads Number of active threads in thread pool.\n";
		metrics << "# TYPE thread_pool_active_threads gauge\n";
		metrics << "thread_pool_active_threads " << thread_pool_->get_active_threads() << "\n";
	}

	// Request metrics
	metrics << "# HELP http_requests_total Total number of HTTP requests.\n";
	metrics << "# TYPE http_requests_total counter\n";
	metrics << "http_requests_total " << total_requests_.load() << "\n";

	metrics << "# HELP http_active_connections Current number of active connections.\n";
	metrics << "# TYPE http_active_connections gauge\n";
	metrics << "http_active_connections " << active_connections_.load() << "\n";

	// Endpoint-specific metrics
	std::lock_guard lock(metrics_mutex_);
	for (const auto &[endpoint, count] : endpoint_requests_)
	{
		metrics << "# HELP http_endpoint_requests_total Total requests per endpoint.\n";
		metrics << "# TYPE http_endpoint_requests_total counter\n";
		metrics << "http_endpoint_requests_total{endpoint=\"" << endpoint << "\"} " << count.load() << "\n";
	}

	// Status code metrics
	for (const auto &[code, count] : status_codes_)
	{
		metrics << "# HELP http_response_status_total Total responses by status code.\n";
		metrics << "# TYPE http_response_status_total counter\n";
		metrics << "http_response_status_total{code=\"" << code << "\"} " << count.load() << "\n";
	}

	// Dragonfly metrics (if available)
	if (dragonfly_manager_)
	{
		// Add Dragonfly connection metrics here
		metrics << "# HELP dragonfly_connections Dragonfly connection status.\n";
		metrics << "# TYPE dragonfly_connections gauge\n";
		metrics << "dragonfly_connections 1\n"; // Simplified - implement actual check
	}

	return metrics.str();
}

void BaseServer::updateRequestMetrics(const std::string &method, const std::string &endpoint, int status_code, double duration_seconds)
{
	total_requests_++;

	std::string key = method + ":" + endpoint;

	{
		std::lock_guard lock(metrics_mutex_);
		endpoint_requests_[key]++;
		status_codes_[status_code]++;
	}
}