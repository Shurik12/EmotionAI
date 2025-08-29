#include "Config.h"
#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <cstdlib>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;
namespace Common
{

	Config::Config() : config_file_path_("config.yaml")
	{
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

			config_ = YAML::LoadFile(config_path);
			loaded_.store(true);

			spdlog::info("Configuration loaded successfully from: {}", config_path);

			// Log important configuration values
			spdlog::info("Server configuration:");
			spdlog::info("  Host: {}", serverHost());
			spdlog::info("  Port: {}", serverPort());

			spdlog::info("Redis configuration:");
			spdlog::info("  Host: {}", redisHost());
			spdlog::info("  Port: {}", redisPort());
			spdlog::info("  DB: {}", redisDb());
			spdlog::info("  Password : {}", redisPassword());

			spdlog::info("Model configuration:");

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
			fs::create_directories(uploadPath());
			fs::create_directories(resultPath());
			fs::create_directories(logPath());

			spdlog::info("Application environment setup completed");
			spdlog::info("Upload folder: {}", uploadPath());
			spdlog::info("Results folder: {}", resultPath());
			spdlog::info("Log folder: {}", logPath());

			return true;
		}
		catch (const std::exception &e)
		{
			spdlog::error("Failed to setup application environment: {}", e.what());
			return false;
		}
	}

	// server
	std::string Config::serverHost() const
	{
		return config_["server"]["host"].as<std::string>();
	}

	int Config::serverPort() const
	{
		return config_["server"]["port"].as<int>();
	}

	// app
	std::string Config::uploadPath() const
	{
		return config_["app"]["uploadPath"].as<std::string>();
	}

	std::string Config::resultPath() const
	{
		return config_["app"]["resultPath"].as<std::string>();
	}

	std::string Config::frontendBuildPath() const
	{
		return config_["app"]["frontendBuildPath"].as<std::string>();
	}

	std::string Config::logPath() const
	{
		return config_["app"]["logPath"].as<std::string>();
	}

	int Config::maxContentLength() const
	{
		return config_["app"]["max_content_length"].as<int>();
	}

	std::vector<std::string> Config::allowedExtensions() const
	{
		return config_["app"]["allowed_extensions"].as<std::vector<std::string>>();
	}

	int Config::taskExpiration() const
	{
		return config_["app"]["task_expiration"].as<int>();
	}

	int Config::applicationExpiration() const
	{
		return config_["app"]["task_expiration"].as<int>();
	}

	std::vector<std::string> Config::emotionCategories() const
	{
		return config_["app"]["emotion_categories"].as<std::vector<std::string>>();
	}

	// redis
	std::string Config::redisHost() const
	{
		return config_["redis"]["host"].as<std::string>();
	}

	int Config::redisPort() const
	{
		return config_["redis"]["port"].as<int>();
	}

	int Config::redisDb() const
	{
		return config_["redis"]["db"].as<int>();
	}

	std::string Config::redisPassword() const
	{
		return config_["redis"]["password"].as<std::string>();
	}

	// mtcnn
	bool Config::mtcnnKeepAll() const
	{
		return config_["mtcnn"]["keep_all"].as<bool>();
	}

	bool Config::mtcnnPostProcess() const
	{
		return config_["mtcnn"]["post_process"].as<bool>();
	}

	int Config::mtcnnMinFaceSize() const
	{
		return config_["mtcnn"]["min_face_size"].as<int>();
	}

	std::string Config::mtcnnDevice() const
	{
		return config_["mtcnn"]["device"].as<std::string>();
	}

	std::string Config::modelBackend() const
	{
		return config_["model"]["backend"].as<std::string>();
	}

	std::string Config::emotionModelPath() const
	{
		return config_["model"]["emotion_model_path"].as<std::string>();
	}

	std::string Config::faceDetectionModelsPath() const
	{
		return config_["mtcnn"]["face_detection_models_path"].as<std::string>();
	}

} // namespace Common