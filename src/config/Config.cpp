#include "Config.h"
#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <cstdlib>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;
namespace Common
{

	Config::Config()
	{
		// All initialization is done in ConfigData struct with default values
	}

	Config &Config::instance()
	{
		static Config instance;
		return instance;
	}

	bool Config::loadFromFile(const std::string &config_path)
	{
		std::lock_guard<std::mutex> lock(config_mutex_);

		try
		{
			config_file_path_ = config_path;

			if (!fs::exists(config_path))
			{
				spdlog::warn("Configuration file not found: {}", config_path);
				loaded_.store(false);
				return false;
			}

			YAML::Node root = YAML::LoadFile(config_path);
			ConfigData new_data;

			// Parse server configuration
			if (root["server"])
			{
				const auto &server = root["server"];
				new_data.server.host = server["host"].as<std::string>(new_data.server.host);
				new_data.server.port = server["port"].as<int>(new_data.server.port);
				new_data.server.type = server["type"].as<std::string>(new_data.server.type);
			}

			// Parse paths configuration
			if (root["paths"])
			{
				const auto &paths = root["paths"];
				new_data.paths.upload = paths["upload"].as<std::string>(new_data.paths.upload);
				new_data.paths.results = paths["results"].as<std::string>(new_data.paths.results);
				new_data.paths.logs = paths["logs"].as<std::string>(new_data.paths.logs);
				new_data.paths.frontend = paths["frontend"].as<std::string>(new_data.paths.frontend);
			}

			// Parse application configuration
			if (root["app"])
			{
				const auto &app = root["app"];
				new_data.app.max_content_length = app["max_content_length"].as<int>(new_data.app.max_content_length);
				new_data.app.task_expiration = app["task_expiration"].as<int>(new_data.app.task_expiration);
				new_data.app.application_expiration = app["application_expiration"].as<int>(new_data.app.application_expiration);

				if (app["allowed_extensions"])
				{
					new_data.app.allowed_extensions = app["allowed_extensions"].as<std::vector<std::string>>();
				}
				if (app["emotion_categories"])
				{
					new_data.app.emotion_categories = app["emotion_categories"].as<std::vector<std::string>>();
				}
			}

			// Parse Redis configuration
			if (root["redis"])
			{
				const auto &redis = root["redis"];
				new_data.redis.host = redis["host"].as<std::string>(new_data.redis.host);
				new_data.redis.port = redis["port"].as<int>(new_data.redis.port);
				new_data.redis.db = redis["db"].as<int>(new_data.redis.db);
				new_data.redis.password = redis["password"].as<std::string>(new_data.redis.password);
			}

			// Parse MTCNN configuration
			if (root["mtcnn"])
			{
				const auto &mtcnn = root["mtcnn"];
				new_data.mtcnn.keep_all = mtcnn["keep_all"].as<bool>(new_data.mtcnn.keep_all);
				new_data.mtcnn.post_process = mtcnn["post_process"].as<bool>(new_data.mtcnn.post_process);
				new_data.mtcnn.min_face_size = mtcnn["min_face_size"].as<int>(new_data.mtcnn.min_face_size);
				new_data.mtcnn.device = mtcnn["device"].as<std::string>(new_data.mtcnn.device);
			}

			// Parse model configuration
			if (root["model"])
			{
				const auto &model = root["model"];
				new_data.model.backend = model["backend"].as<std::string>(new_data.model.backend);
				new_data.model.emotion_model_path = model["emotion_model_path"].as<std::string>(new_data.model.emotion_model_path);
				new_data.model.face_detection_models_path = model["face_detection_models_path"].as<std::string>(new_data.model.face_detection_models_path);
			}

			// Atomically update the configuration data
			data_ = std::move(new_data);
			loaded_.store(true);

			spdlog::info("Configuration loaded successfully from: {}", config_path);

			// Log important configuration values
			spdlog::info("Server configuration:");
			spdlog::info("  Host: {}", data_.server.host);
			spdlog::info("  Port: {}", data_.server.port);
			spdlog::info("  Type: {}", data_.server.type);

			spdlog::info("Paths configuration:");
			spdlog::info("  Upload: {}", data_.paths.upload);
			spdlog::info("  Results: {}", data_.paths.results);
			spdlog::info("  Logs: {}", data_.paths.logs);
			spdlog::info("  Frontend: {}", data_.paths.frontend);

			spdlog::info("Redis configuration:");
			spdlog::info("  Host: {}", data_.redis.host);
			spdlog::info("  Port: {}", data_.redis.port);
			spdlog::info("  DB: {}", data_.redis.db);
			spdlog::info("  Password: {}", data_.redis.password.empty() ? "<empty>" : "***");

			spdlog::info("Model configuration:");
			spdlog::info("  Backend: {}", data_.model.backend);
			spdlog::info("  Emotion Model: {}", data_.model.emotion_model_path);
			spdlog::info("  Face Detection Models: {}", data_.model.face_detection_models_path);

			return true;
		}
		catch (const YAML::Exception &e)
		{
			spdlog::error("Failed to parse configuration file {}: {}", config_path, e.what());
			loaded_.store(false);
			return false;
		}
		catch (const std::exception &e)
		{
			spdlog::error("Failed to load configuration file {}: {}", config_path, e.what());
			loaded_.store(false);
			return false;
		}
	}

	bool Config::reload()
	{
		return loadFromFile(config_file_path_);
	}

	bool Config::setupApplicationEnvironment()
	{
		try
		{
			// Create necessary directories
			fs::create_directories(data_.paths.upload);
			fs::create_directories(data_.paths.results);
			fs::create_directories(data_.paths.logs);
			fs::create_directories(data_.paths.frontend);

			spdlog::info("Application environment setup completed");
			spdlog::info("Upload folder: {}", data_.paths.upload);
			spdlog::info("Results folder: {}", data_.paths.results);
			spdlog::info("Log folder: {}", data_.paths.logs);
			spdlog::info("Frontend folder: {}", data_.paths.frontend);

			return true;
		}
		catch (const std::exception &e)
		{
			spdlog::error("Failed to setup application environment: {}", e.what());
			return false;
		}
	}

	bool Config::validate() const
	{
		if (!loaded_.load())
		{
			spdlog::error("Configuration not loaded");
			return false;
		}

		// Validate required paths
		if (data_.model.emotion_model_path.empty())
		{
			spdlog::error("Emotion model path is required");
			return false;
		}

		if (data_.model.face_detection_models_path.empty())
		{
			spdlog::error("Face detection models path is required");
			return false;
		}

		// Validate server configuration
		if (data_.server.port <= 0 || data_.server.port > 65535)
		{
			spdlog::error("Invalid server port: {}", data_.server.port);
			return false;
		}

		// Validate Redis configuration
		if (data_.redis.port <= 0 || data_.redis.port > 65535)
		{
			spdlog::error("Invalid Redis port: {}", data_.redis.port);
			return false;
		}

		spdlog::info("Configuration validation passed");
		return true;
	}

} // namespace Common