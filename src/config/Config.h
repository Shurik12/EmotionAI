#pragma once

#include <string>
#include <vector>
#include <memory>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <atomic>
#include <mutex>
#include <spdlog/spdlog.h>

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

	// Convert log level string to spdlog level
	spdlog::level::level_enum getSpdLogLevel() const;
	spdlog::level::level_enum getSpdLogFlushLevel() const;

	// Direct access to nested structures (optional)
	const auto &server() const { return data_.server; }
	const auto &paths() const { return data_.paths; }
	const auto &app() const { return data_.app; }
	const auto &redis() const { return data_.redis; }
	const auto &dragonfly() const { return data_.dragonfly; }
	const auto &mtcnn() const { return data_.mtcnn; }
	const auto &model() const { return data_.model; }
	const auto &logging() const { return data_.logging; }

	// Check if config is loaded
	bool isLoaded() const { return loaded_.load(); }

	// Validate configuration
	bool validate() const;

private:
	Config();
	~Config() = default;

	struct ServerConfig
	{
		std::string host = "0.0.0.0";
		int port = 8080;
		std::string type = "non-blocking";
	};

	struct PathsConfig
	{
		std::string upload = "./uploads";
		std::string results = "./results";
		std::string frontend = "../frontend/build";
		std::string logs = "./logs";
	};

	struct AppConfig
	{
		int max_content_length = 52428800; // 50MB
		std::vector<std::string> allowed_extensions = {"png", "jpg", "jpeg", "mp4", "avi", "webm", "mp3"};
		int task_expiration = 3600;			  // 1 hour in seconds
		int application_expiration = 2592000; // 30 days in seconds
		std::vector<std::string> emotion_categories = {"anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"};
	};

	struct LoggingConfig
	{
		std::string level = "info";
		int max_file_size = 20971520; // 20MB
		int max_files = 5;
		bool console_enabled = true;
		bool file_enabled = true;
		std::string pattern = "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%n] %v";
		std::string file_pattern = "[%Y-%m-%d %H:%M:%S.%e] [%l] [%n] %v";
		std::string flush_on_level = "info";
	};

	struct RedisConfig
	{
		std::string host = "localhost";
		int port = 6379;
		int db = 0;
		std::string password = "";
	};

	struct DragonflyConfig
	{
		std::string host = "localhost";
		int port = 6379;
		int db = 0;
		std::string password = "";
	};

	struct MtcnnConfig
	{
		bool keep_all = false;
		bool post_process = false;
		int min_face_size = 40;
		std::string device = "cpu";
	};

	struct ModelConfig
	{
		std::string backend = "torch";
		std::string emotion_model_path = "";
		std::string face_detection_models_path = "";
	};

	struct ConfigData
	{
		ServerConfig server;
		PathsConfig paths;
		AppConfig app;
		LoggingConfig logging;
		RedisConfig redis;
		DragonflyConfig dragonfly;
		MtcnnConfig mtcnn;
		ModelConfig model;
	};

	ConfigData data_;
	mutable std::mutex config_mutex_;
	std::atomic<bool> loaded_{false};
	std::string config_file_path_;
};