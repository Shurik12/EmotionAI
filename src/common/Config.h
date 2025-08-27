#pragma once

#include <string>
#include <vector>
#include <memory>
#include <yaml-cpp/yaml.h>
#include <filesystem>

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

		// Load configuration from file
		bool loadFromFile(const std::string &config_path = "config.yaml");

		// Setup application environment
		bool setupApplicationEnvironment();

		// Getters for configuration values
		std::string uploadFolder() const;
		std::string resultsFolder() const;
		std::string frontendBuildPath() const;
		std::string logFolder() const;
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

		// Get raw YAML node for advanced access
		const YAML::Node &getRawConfig() const;

	private:
		Config();
		~Config() = default;

		// Helper methods
		std::string getString(const std::vector<std::string> &path, const std::string &default_value = "") const;
		int getInt(const std::vector<std::string> &path, int default_value = 0) const;
		bool getBool(const std::vector<std::string> &path, bool default_value = false) const;
		std::vector<std::string> getStringArray(const std::vector<std::string> &path,
												const std::vector<std::string> &default_value = {}) const;

		YAML::Node config_;
		bool loaded_ = false;
	};

} // namespace Common