#pragma once

#include <string>
#include <vector>
#include <memory>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <atomic>
#include <mutex>

namespace Common
{

	class Config
	{
	public:
		// Delete copy constructor and assignment operator
		Config(const Config &) = delete;
		Config &operator=(const Config &) = delete;

		// Get singleton instance
		static Config &instance();

		// Load configuration from file (thread-safe)
		bool loadFromFile(const std::string &config_path = "config.yaml");

		// Reload configuration
		bool reload();

		// Setup application environment
		bool setupApplicationEnvironment();

		// Getters for configuration values
		std::string uploadPath() const;
		std::string resultPath() const;
		std::string frontendBuildPath() const;
		std::string logPath() const;
		int maxContentLength() const;
		std::vector<std::string> allowedExtensions() const;
		int taskExpiration() const;
		int applicationExpiration() const;
		std::vector<std::string> emotionCategories() const;

		std::string redisHost() const;
		int redisPort() const;
		int redisDb() const;
		std::string redisPassword() const;

		bool mtcnnKeepAll() const;
		bool mtcnnPostProcess() const;
		int mtcnnMinFaceSize() const;
		std::string mtcnnDevice() const;
		std::string mtcnnModelsPath() const;
		std::string mtcnnfaceModelsPath() const;

		// Server configuration
		std::string serverHost() const;
		int serverPort() const;

		// Check if config is loaded
		bool isLoaded() const { return loaded_.load(); }

		// Validate configuration
		bool validate() const;

	private:
		Config();
		~Config() = default;

		YAML::Node config_;
		mutable std::mutex config_mutex_;
		std::atomic<bool> loaded_{false};
		std::string config_file_path_;
	};

} // namespace Common