#pragma once

#include <string>
#include <vector>
#include <memory>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <atomic>
#include <mutex>
#include <spdlog/spdlog.h>
#include <audio/BurnoutModels.h>
#include <audio/ExternalInfluenceModels.h>

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
	const auto &cluster() const { return data_.cluster; }
	const auto &storage() const { return data_.storage; }
	const auto &ai() const { return data_.ai; }
	const auto &externalInfluence() const { return data_.external_influence; }
	const auto &burnout() const { return data_.burnout; }
	const auto &analytics() const { return data_.analytics; }
	const auto &security() const { return data_.security; }

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
	};

	struct PathsConfig
	{
		std::string uploads = "uploads";
		std::string results = "results";
		std::string frontend = "../frontend/build";
		std::string logs = "logs";
	};

	struct AppConfig
	{
		int max_content_length = 314572800; // 300MB
		// Chunk size advertised to clients using the resumable upload API
		// (POST /api/upload/chunk). Capped by max_content_length.
		int upload_chunk_size = 8388608; // 8MB
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
		std::string audio_model_path = "";
		std::string face_detection_models_path = "";
	};

	struct ClusterConfig
	{
		bool enabled{false};
		std::string name{"emotionai-cluster"};
		std::string instance_id{"auto"};  // "auto" means generate UUID automatically
		int heartbeat_interval{30};		  // seconds
		int instance_timeout{120};		  // seconds
		int leader_election_interval{60}; // seconds
		bool auto_cleanup{true};
	};

	struct TaskManagementConfig
	{
		int cache_ttl{300}; // 5 minutes in seconds
		int max_cache_size{10000};
		int batch_size{50};
		int batch_timeout_ms{100};
	};

	struct StorageConfig
	{
		std::string type = "local"; // "local" or "s3"
		std::string base_path = "./storage";
		std::string s3_endpoint = "";
		std::string s3_access_key = "";
		std::string s3_secret_key = "";
		std::string s3_bucket = "";
		std::string s3_region = "us-east-1";
		bool s3_use_ssl = true;
	};

	struct AIConfig
	{
		// Single switch for everything AI-related:
		//   Ai    -> use the LLM provider (Focus task breakdown + image/video
		//            interpretation).
		//   Local -> no provider, API key or network call: the image/video stage
		//            runs the built-in deterministic scorer; Focus breakdown is
		//            unavailable and the SPA falls back to its local templates.
		//   Off   -> no AI and no local interpretation.
		enum class Mode
		{
			Off,
			Local,
			Ai
		};

		Mode mode{Mode::Off};

		bool isAi() const { return mode == Mode::Ai; }
		bool isLocal() const { return mode == Mode::Local; }
		bool isOff() const { return mode == Mode::Off; }
		// Image/video second stage runs in both "ai" and "local" modes.
		bool interpretationEnabled() const { return mode != Mode::Off; }
		// The Focus breakdown has no local engine, so it needs the actual LLM.
		bool focusEnabled() const { return mode == Mode::Ai; }
		const char *modeName() const
		{
			return mode == Mode::Ai ? "ai" : (mode == Mode::Local ? "local" : "false");
		}
		// OpenAI-compatible endpoint. The client posts to
		// {base_url}/chat/completions with "Authorization: Bearer <api_key>".
		std::string base_url{"https://shared1.multitool.works:4000/v1"};
		std::string model{"deepseek-v4-flash"};
		std::string api_key;
		bool verify_ssl{true};
		long timeout_seconds{10};
		long connect_timeout_seconds{5};
		// Larger budget for the image/video interpretation (verbose reasoning
		// model, runs off the epoll thread).
		long media_timeout_seconds{120};
	};

	struct AnalyticsConfig
	{
		// Structured per-event JSONL log (contest analytics requirement).
		bool enabled{true};
		// Empty -> <paths.logs>/analytics.jsonl.
		std::string path;
	};

	struct SecurityConfig
	{
		// When non-empty, every /api/ request except `exempt_paths` must carry
		// `Authorization: Bearer <api_token>` or `X-API-Key: <api_token>`.
		// An empty token keeps the public demo open (auth disabled).
		// NOTE: a token embedded in a web or mobile bundle is extractable — it
		// is an access-control deterrent plus a revocable kill switch, not a
		// substitute for per-user authentication.
		std::string api_token;
		// /api/results is exempt because result images are loaded by <img>,
		// which cannot attach the Authorization header. The file names are
		// random task UUIDs (capability URLs).
		std::vector<std::string> exempt_paths{"/api/health", "/api/results"};
		// Per-client token bucket, applied to every /api/ request (auth or not).
		// 240/min = 4 req/s sustained, which leaves headroom for the Focus
		// stream (~1 frame/s + session polling) while still capping abuse.
		bool rate_limit_enabled{true};
		int rate_limit_per_minute{240};
		int rate_limit_burst{80};
		int rate_limit_max_keys{10000};
		// Behind nginx: identify clients by X-Real-IP / X-Forwarded-For instead
		// of the socket peer. Enable only when a trusted proxy sets them,
		// otherwise a client can spoof the header to bypass the limit.
		bool trust_proxy_headers{false};
		// Device-scoped identity: POST /api/auth/device issues an opaque token;
		// the server derives user_id from it and ignores client-supplied ids.
		// When true, /api/burnout/baseline requires a valid device token.
		bool device_auth_enabled{true};
		// Token lifetime; the client re-bootstraps after expiry.
		int device_token_ttl_seconds{15552000}; // 180 days
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
		ClusterConfig cluster;
		TaskManagementConfig task_management;
		StorageConfig storage;
		AIConfig ai;
		audio::ExternalInfluenceConfig external_influence;
		audio::BurnoutConfig burnout;
		AnalyticsConfig analytics;
		SecurityConfig security;
	};

	ConfigData data_;
	mutable std::mutex config_mutex_;
	std::atomic<bool> loaded_{false};
	std::string config_file_path_;
};