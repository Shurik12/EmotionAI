#pragma once

#include <string>
#include <vector>
#include <memory>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <atomic>
#include <mutex>
#include <spdlog/spdlog.h>

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
		std::string uploadPath() const { return data_.paths.upload; }
		std::string resultPath() const { return data_.paths.results; }
		std::string frontendBuildPath() const { return data_.paths.frontend; }
		std::string logPath() const { return data_.paths.logs; }
		int maxContentLength() const { return data_.app.max_content_length; }
		std::vector<std::string> allowedExtensions() const { return data_.app.allowed_extensions; }
		int taskExpiration() const { return data_.app.task_expiration; }
		int applicationExpiration() const { return data_.app.application_expiration; }
		std::vector<std::string> emotionCategories() const { return data_.app.emotion_categories; }

		std::string redisHost() const { return data_.redis.host; }
		int redisPort() const { return data_.redis.port; }
		int redisDb() const { return data_.redis.db; }
		std::string redisPassword() const { return data_.redis.password; }

		bool mtcnnKeepAll() const { return data_.mtcnn.keep_all; }
		bool mtcnnPostProcess() const { return data_.mtcnn.post_process; }
		int mtcnnMinFaceSize() const { return data_.mtcnn.min_face_size; }
		std::string mtcnnDevice() const { return data_.mtcnn.device; }

		std::string modelBackend() const { return data_.model.backend; }
		std::string emotionModelPath() const { return data_.model.emotion_model_path; }
		std::string faceDetectionModelsPath() const { return data_.model.face_detection_models_path; }

		// Server configuration
		std::string serverHost() const { return data_.server.host; }
		int serverPort() const { return data_.server.port; }
		std::string serverType() const { return data_.server.type; }

		// Logging configuration
		std::string logLevel() const { return data_.logging.level; }
		int logMaxFileSize() const { return data_.logging.max_file_size; }
		int logMaxFiles() const { return data_.logging.max_files; }
		bool logConsoleEnabled() const { return data_.logging.console_enabled; }
		bool logFileEnabled() const { return data_.logging.file_enabled; }
		std::string logPattern() const { return data_.logging.pattern; }
		std::string logFilePattern() const { return data_.logging.file_pattern; }
		std::string logFlushOnLevel() const { return data_.logging.flush_on_level; }

		// Convert log level string to spdlog level
		spdlog::level::level_enum getSpdLogLevel() const;
		spdlog::level::level_enum getSpdLogFlushLevel() const;

		// Direct access to nested structures (optional)
		const auto &server() const { return data_.server; }
		const auto &paths() const { return data_.paths; }
		const auto &app() const { return data_.app; }
		const auto &redis() const { return data_.redis; }
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
			MtcnnConfig mtcnn;
			ModelConfig model;
		};

		ConfigData data_;
		mutable std::mutex config_mutex_;
		std::atomic<bool> loaded_{false};
		std::string config_file_path_;
	};

} // namespace Common