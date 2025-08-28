#include "Config.h"
#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;
namespace Common
{

	Config::Config()
	{
		// Constructor
	}

	Config &Config::instance()
	{
		static Config instance;
		return instance;
	}

	bool Config::loadFromFile(const std::string &config_path)
	{
		try
		{
			if (!fs::exists(config_path))
			{
				spdlog::warn("Configuration file not found: {}", config_path);
				return false;
			}

			config_ = YAML::LoadFile(config_path);
			loaded_ = true;

			// DEBUG: Print the entire config structure
			spdlog::info("Configuration structure:");
			try
			{
				if (config_["redis"])
				{
					spdlog::info("Redis section found");
					if (config_["redis"]["password"])
					{
						spdlog::info("Redis password key exists");
						spdlog::info("Redis password type: {}", config_["redis"]["password"].Type());
						spdlog::info("Redis password value: '{}'", config_["redis"]["password"].as<std::string>());
					}
					else
					{
						spdlog::info("Redis password key NOT found");
					}
				}
				else
				{
					spdlog::info("Redis section NOT found");
				}
			}
			catch (const std::exception &e)
			{
				spdlog::warn("Error examining config structure: {}", e.what());
			}

			spdlog::info("Configuration loaded successfully from: {}", config_path);
			return true;
		}
		catch (const YAML::Exception &e)
		{
			spdlog::error("Failed to parse configuration file {}: {}", config_path, e.what());
			return false;
		}
		catch (const std::exception &e)
		{
			spdlog::error("Failed to load configuration file {}: {}", config_path, e.what());
			return false;
		}
	}

	bool Config::setupApplicationEnvironment()
	{
		try
		{
			// Create necessary directories
			fs::create_directories(uploadFolder());
			fs::create_directories(resultsFolder());
			fs::create_directories(logFolder());

			spdlog::info("Application environment setup completed");
			spdlog::info("Upload folder: {}", uploadFolder());
			spdlog::info("Results folder: {}", resultsFolder());
			spdlog::info("Log folder: {}", logFolder());

			return true;
		}
		catch (const std::exception &e)
		{
			spdlog::error("Failed to setup application environment: {}", e.what());
			return false;
		}
	}

	// Helper methods
	std::string Config::getString(const std::vector<std::string> &path, const std::string &default_value) const
	{
		if (!loaded_)
		{
			spdlog::warn("Config not loaded, using default: {}", default_value);
			return default_value;
		}

		try
		{
			YAML::Node node = config_;
			for (size_t i = 0; i < path.size(); ++i)
			{
				const auto &key = path[i];
				if (!node[key])
				{
					return default_value;
				}

				node = node[key];
			}

			// Check if the final node is valid and convertible to string
			if (!node.IsDefined() || node.IsNull())
			{
				return default_value;
			}

			try
			{
				std::string value = node.as<std::string>();
				return value;
			}
			catch (const YAML::BadConversion &e)
			{
				return default_value;
			}
		}
		catch (const YAML::Exception &e)
		{
			return default_value;
		}
		catch (const std::exception &e)
		{
			return default_value;
		}
	}

	int Config::getInt(const std::vector<std::string> &path, int default_value) const
	{
		if (!loaded_)
			return default_value;

		try
		{
			YAML::Node node = config_;
			for (const auto &key : path)
			{
				if (node[key])
				{
					node = node[key];
				}
				else
				{
					return default_value;
				}
			}
			return node.as<int>(default_value);
		}
		catch (const YAML::Exception &e)
		{
			spdlog::warn("Error reading config value: {}", e.what());
			return default_value;
		}
	}

	bool Config::getBool(const std::vector<std::string> &path, bool default_value) const
	{
		if (!loaded_)
			return default_value;

		try
		{
			YAML::Node node = config_;
			for (const auto &key : path)
			{
				if (node[key])
				{
					node = node[key];
				}
				else
				{
					return default_value;
				}
			}
			return node.as<bool>(default_value);
		}
		catch (const YAML::Exception &e)
		{
			spdlog::warn("Error reading config value: {}", e.what());
			return default_value;
		}
	}

	std::vector<std::string> Config::getStringArray(const std::vector<std::string> &path,
													const std::vector<std::string> &default_value) const
	{
		if (!loaded_)
			return default_value;

		try
		{
			YAML::Node node = config_;
			for (const auto &key : path)
			{
				if (node[key])
				{
					node = node[key];
				}
				else
				{
					return default_value;
				}
			}

			if (node.IsSequence())
			{
				std::vector<std::string> result;
				for (const auto &item : node)
				{
					result.push_back(item.as<std::string>());
				}
				return result;
			}
			return default_value;
		}
		catch (const YAML::Exception &e)
		{
			spdlog::warn("Error reading config value: {}", e.what());
			return default_value;
		}
	}

	// Configuration getters
	std::string Config::uploadFolder() const
	{
		return getString({"app", "upload_folder"}, "uploads");
	}

	std::string Config::resultsFolder() const
	{
		return getString({"app", "results_folder"}, "results");
	}

	std::string Config::frontendBuildPath() const
	{
		// This might not be in the config, use default
		return getString({"app", "frontend_build_path"}, "../frontend/build");
	}

	std::string Config::logFolder() const
	{
		return getString({"app", "log_folder"}, "logs");
	}

	int Config::maxContentLength() const
	{
		return getInt({"app", "max_content_length"}, 52428800); // 50MB default
	}

	std::vector<std::string> Config::allowedExtensions() const
	{
		return getStringArray({"app", "allowed_extensions"},
							  {"png", "jpg", "jpeg", "mp4", "avi", "webm"});
	}

	int Config::taskExpiration() const
	{
		return getInt({"app", "task_expiration"}, 3600); // 1 hour default
	}

	int Config::applicationExpiration() const
	{
		return getInt({"app", "application_expiration"}, 2592000); // 30 days default
	}

	std::vector<std::string> Config::emotionCategories() const
	{
		return getStringArray({"app", "emotion_categories"},
							  {"anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"});
	}

	std::string Config::redisHost() const
	{
		return getString({"redis", "host"}, "localhost");
	}

	int Config::redisPort() const
	{
		return getInt({"redis", "port"}, 6379);
	}

	int Config::redisDb() const
	{
		return getInt({"redis", "db"}, 0);
	}

	std::string Config::redisPassword() const
	{
		spdlog::debug("Getting Redis password...");
		std::string password = getString({"redis", "password"}, "");

		// If getString fails, try direct access as fallback
		if (password.empty())
		{
			spdlog::warn("getString failed, trying direct access to Redis password");
			try
			{
				if (config_["redis"] && config_["redis"]["password"])
				{
					password = config_["redis"]["password"].as<std::string>();
					spdlog::info("Direct access successful: '{}'", password);
				}
				else
				{
					spdlog::warn("Direct access also failed - redis or password node not found");
				}
			}
			catch (const std::exception &e)
			{
				spdlog::error("Direct access exception: {}", e.what());
			}
		}

		spdlog::debug("Redis password result: '{}'", password);
		spdlog::debug("Redis password length: {}", password.length());
		spdlog::debug("Redis password empty: {}", password.empty() ? "YES" : "NO");

		return password;
	}

	bool Config::mtcnnKeepAll() const
	{
		return getBool({"mtcnn", "keep_all"}, false);
	}

	bool Config::mtcnnPostProcess() const
	{
		return getBool({"mtcnn", "post_process"}, false);
	}

	int Config::mtcnnMinFaceSize() const
	{
		return getInt({"mtcnn", "min_face_size"}, 40);
	}

	std::string Config::mtcnnDevice() const
	{
		return getString({"mtcnn", "device"}, "cpu");
	}

	const YAML::Node &Config::getRawConfig() const
	{
		return config_;
	}

} // namespace Common