#include "RedisManager.h"
#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <random>
#include <common/uuid.h>
#include <common/Config.h>

namespace fs = std::filesystem;
namespace db
{

	RedisManager::RedisManager() : connection_(nullptr)
	{
		try
		{
			// Get configuration from environment or config
			auto &config = Common::Config::instance();

			// Load upload folder from config
			upload_folder_ = config.uploadFolder();
			task_expiration_ = config.taskExpiration();

			// Create upload directory if it doesn't exist
			fs::create_directories(upload_folder_);

			connection_ = create_connection();
			spdlog::info("Successfully connected to Redis");
		}
		catch (const std::exception &e)
		{
			spdlog::error("Failed to initialize RedisManager: {}", e.what());
			throw;
		}
	}

	RedisManager::~RedisManager()
	{
		if (connection_)
		{
			redisFree(connection_);
			connection_ = nullptr;
		}
	}

	redisContext *RedisManager::create_connection()
	{
		try
		{
			// Get Redis configuration from environment variables or config
			const char *host = std::getenv("REDIS_HOST");
			const char *port = std::getenv("REDIS_PORT");
			const char *db = std::getenv("REDIS_DB");
			const char *password = std::getenv("REDIS_PASSWORD");

			// Use defaults if environment variables are not set
			std::string redis_host = host ? host : "localhost";
			std::string redis_port_str = port ? port : "6379";
			int redis_db = db ? std::stoi(db) : 0;
			std::string redis_password = password ? password : "";

			int redis_port = std::stoi(redis_port_str);

			// Connect to Redis
			redisContext *conn = redisConnect(redis_host.c_str(), redis_port);
			if (conn == nullptr || conn->err)
			{
				std::string error_msg = conn ? conn->errstr : "Cannot allocate redis context";
				if (conn)
					redisFree(conn);
				throw std::runtime_error("Failed to connect to Redis: " + error_msg);
			}

			// Authenticate if password is provided
			if (!redis_password.empty())
			{
				redisReply *reply = (redisReply *)redisCommand(conn, "AUTH %s", redis_password.c_str());
				if (reply == nullptr || reply->type == REDIS_REPLY_ERROR)
				{
					std::string error_msg = reply ? reply->str : "Authentication failed";
					freeReplyObject(reply);
					redisFree(conn);
					throw std::runtime_error("Redis authentication failed: " + error_msg);
				}
				freeReplyObject(reply);
			}

			// Select database if not default
			if (redis_db != 0)
			{
				redisReply *reply = (redisReply *)redisCommand(conn, "SELECT %d", redis_db);
				if (reply == nullptr || reply->type == REDIS_REPLY_ERROR)
				{
					std::string error_msg = reply ? reply->str : "Database selection failed";
					freeReplyObject(reply);
					redisFree(conn);
					throw std::runtime_error("Redis database selection failed: " + error_msg);
				}
				freeReplyObject(reply);
			}

			// Test connection with PING
			redisReply *reply = (redisReply *)redisCommand(conn, "PING");
			if (reply == nullptr || reply->type == REDIS_REPLY_ERROR ||
				(reply->type == REDIS_REPLY_STRING && std::string(reply->str) != "PONG"))
			{
				std::string error_msg = reply ? reply->str : "PING failed";
				freeReplyObject(reply);
				redisFree(conn);
				throw std::runtime_error("Redis connection test failed: " + error_msg);
			}
			freeReplyObject(reply);

			return conn;
		}
		catch (const std::exception &e)
		{
			spdlog::error("Failed to connect to Redis: {}", e.what());
			throw;
		}
	}

	redisContext *RedisManager::connection()
	{
		if (!connection_)
		{
			throw std::runtime_error("Redis connection is not initialized");
		}
		return connection_;
	}

	const redisContext *RedisManager::connection() const
	{
		if (!connection_)
		{
			throw std::runtime_error("Redis connection is not initialized");
		}
		return connection_;
	}

	void RedisManager::free_reply(redisReply *reply)
	{
		if (reply)
		{
			freeReplyObject(reply);
		}
	}

	void RedisManager::set_task_status(const std::string &task_id, const std::string &status_data)
	{
		try
		{
			std::string key = "task:" + task_id;
			redisReply *reply = (redisReply *)redisCommand(connection_, "SETEX %s %d %s",
														   key.c_str(), task_expiration_, status_data.c_str());

			if (reply == nullptr || reply->type == REDIS_REPLY_ERROR)
			{
				std::string error_msg = reply ? reply->str : "SETEX command failed";
				free_reply(reply);
				throw std::runtime_error("Error updating task status: " + error_msg);
			}

			free_reply(reply);
			spdlog::debug("Set task status for task_id: {}", task_id);
		}
		catch (const std::exception &e)
		{
			spdlog::error("Error updating task status for task_id {}: {}", task_id, e.what());
			throw;
		}
	}

	void RedisManager::set_task_status(const std::string &task_id, const nlohmann::json &status_data)
	{
		set_task_status(task_id, status_data.dump());
	}

	std::optional<std::string> RedisManager::get_task_status(const std::string &task_id)
	{
		try
		{
			std::string key = "task:" + task_id;
			redisReply *reply = (redisReply *)redisCommand(connection_, "GET %s", key.c_str());

			if (reply == nullptr)
			{
				throw std::runtime_error("GET command failed: null reply");
			}

			if (reply->type == REDIS_REPLY_ERROR)
			{
				std::string error_msg = reply->str;
				free_reply(reply);
				throw std::runtime_error("GET command failed: " + error_msg);
			}

			if (reply->type == REDIS_REPLY_NIL)
			{
				free_reply(reply);
				spdlog::debug("Task status not found for task_id: {}", task_id);
				return std::nullopt;
			}

			if (reply->type == REDIS_REPLY_STRING)
			{
				std::string result(reply->str, reply->len);
				free_reply(reply);
				spdlog::debug("Retrieved task status for task_id: {}", task_id);
				return result;
			}

			free_reply(reply);
			spdlog::debug("Unexpected reply type for task_id: {}", task_id);
			return std::nullopt;
		}
		catch (const std::exception &e)
		{
			spdlog::error("Error getting task status for task_id {}: {}", task_id, e.what());
			return std::nullopt;
		}
	}

	std::optional<nlohmann::json> RedisManager::get_task_status_json(const std::string &task_id)
	{
		try
		{
			auto status = get_task_status(task_id);
			if (status)
			{
				return nlohmann::json::parse(*status);
			}
			return std::nullopt;
		}
		catch (const nlohmann::json::parse_error &e)
		{
			spdlog::error("JSON parse error for task_id {}: {}", task_id, e.what());
			return std::nullopt;
		}
		catch (const std::exception &e)
		{
			spdlog::error("Error parsing task status JSON for task_id {}: {}", task_id, e.what());
			return std::nullopt;
		}
	}

	std::string RedisManager::generate_uuid()
	{
		try
		{
			// Use the uuid library's random generator
			std::random_device rd;
			std::mt19937 gen(rd());
			uuids::uuid_random_generator generator(gen);

			uuids::uuid id = generator();
			return uuids::to_string(id);
		}
		catch (const std::exception &e)
		{
			spdlog::error("Error generating UUID: {}", e.what());
			// Fallback to simple random generation if uuid library fails
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_int_distribution<> dis(0, 15);
			std::uniform_int_distribution<> dis2(8, 11);

			std::stringstream ss;
			ss << std::hex;

			for (int i = 0; i < 8; i++)
			{
				ss << dis(gen);
			}
			ss << "-";
			for (int i = 0; i < 4; i++)
			{
				ss << dis(gen);
			}
			ss << "-4";
			for (int i = 0; i < 3; i++)
			{
				ss << dis(gen);
			}
			ss << "-";
			ss << dis2(gen);
			for (int i = 0; i < 3; i++)
			{
				ss << dis(gen);
			}
			ss << "-";
			for (int i = 0; i < 12; i++)
			{
				ss << dis(gen);
			}

			return ss.str();
		}
	}

	std::string RedisManager::save_application(const std::string &application_data)
	{
		try
		{
			// Parse the application data to add metadata
			nlohmann::json application_json;
			try
			{
				application_json = nlohmann::json::parse(application_data);
			}
			catch (const nlohmann::json::parse_error &e)
			{
				throw std::runtime_error("Invalid JSON data: " + std::string(e.what()));
			}

			return save_application(application_json);
		}
		catch (const std::exception &e)
		{
			spdlog::error("Error saving application: {}", e.what());
			throw;
		}
	}

	std::string RedisManager::save_application(const nlohmann::json &application_data)
	{
		try
		{
			std::string application_id = generate_uuid();

			// Create a copy and add metadata
			nlohmann::json application_with_meta = application_data;

			// Get current timestamp in ISO format
			auto now = std::chrono::system_clock::now();
			auto in_time_t = std::chrono::system_clock::to_time_t(now);
			std::stringstream ss;
			ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%dT%H:%M:%S");
			std::string timestamp = ss.str();

			// Add metadata
			application_with_meta["id"] = application_id;
			application_with_meta["timestamp"] = timestamp;
			application_with_meta["status"] = "new";

			// Save to text file
			fs::path applications_file = fs::path(upload_folder_) / "applications.txt";

			std::ofstream out_file(applications_file, std::ios::app);
			if (!out_file.is_open())
			{
				throw std::runtime_error("Failed to open applications file: " + applications_file.string());
			}

			out_file << application_with_meta.dump() << std::endl;
			out_file.close();

			spdlog::info("Saved application with ID: {}", application_id);
			return application_id;
		}
		catch (const std::exception &e)
		{
			spdlog::error("Error saving application: {}", e.what());
			throw;
		}
	}

} // namespace db