#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <random>
#include <cstdarg>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <common/uuid.h>
#include <config/Config.h>
#include <logging/Logger.h>
#include "RedisManager.h"

namespace fs = std::filesystem;
namespace db
{

	RedisManager::RedisManager() : connection_(nullptr), redis_port_(6379), redis_db_(0), task_expiration_(3600)
	{
	}

	RedisManager::~RedisManager()
	{
		std::lock_guard<std::mutex> lock(connection_mutex_);
		// No need to manually free connection_ as unique_ptr will handle it
		initialized_.store(false);
	}

	void RedisManager::initialize()
	{
		loadConfiguration();

		fs::create_directories(upload_folder_);
		std::lock_guard<std::mutex> lock(connection_mutex_);
		connection_ = create_connection();
		initialized_.store(true);
		LOG_INFO("Successfully connected to Redis");
	}

	void RedisManager::loadConfiguration()
	{
		auto &config = Common::Config::instance();

		redis_host_ = config.redis().host;
		redis_port_ = config.redis().port;
		redis_db_ = config.redis().db;
		redis_password_ = config.redis().password;
		upload_folder_ = config.paths().upload;
		task_expiration_ = config.app().task_expiration;

		LOG_INFO("Redis configuration loaded: {}:{} (DB: {})", redis_host_, redis_port_, redis_db_);
	}
	bool RedisManager::ensure_connection()
	{
		if (initialized_.load())
		{
			return true;
		}

		std::lock_guard<std::mutex> lock(connection_mutex_);
		if (!connection_)
		{
			try
			{
				connection_ = create_connection();
				initialized_.store(true);
				LOG_INFO("Reconnected to Redis successfully");
				return true;
			}
			catch (const std::exception &e)
			{
				LOG_ERROR("Failed to reconnect to Redis: {}", e.what());
				return false;
			}
		}

		// Test if connection is still alive
		redisReply *reply = (redisReply *)redisCommand(connection_.get(), "PING");
		if (reply && reply->type == REDIS_REPLY_STRING && std::string(reply->str) == "PONG")
		{
			freeReplyObject(reply);
			return true;
		}

		freeReplyObject(reply);

		// Connection is dead, try to reconnect
		try
		{
			connection_.reset(); // Reset the current connection
			connection_ = create_connection();
			initialized_.store(true);
			LOG_INFO("Reconnected to Redis after connection loss");
			return true;
		}
		catch (const std::exception &e)
		{
			LOG_ERROR("Failed to reconnect to Redis: {}", e.what());
			connection_.reset();
			initialized_.store(false);
			return false;
		}
	}

	std::unique_ptr<redisContext, RedisManager::RedisContextDeleter> RedisManager::create_connection()
	{
		try
		{
			LOG_INFO("Connecting to Redis: {}:{} (DB: {})", redis_host_, redis_port_, redis_db_);

			// Set connection timeout
			struct timeval timeout = {1, 500000}; // 1.5 seconds
			redisContext *conn = redisConnectWithTimeout(redis_host_.c_str(), redis_port_, timeout);

			if (conn == nullptr || conn->err)
			{
				std::string error_msg = conn ? conn->errstr : "Cannot allocate redis context";
				if (conn)
					redisFree(conn);
				throw std::runtime_error("Failed to connect to Redis: " + error_msg);
			}

			// Authenticate if password is provided
			if (!redis_password_.empty())
			{
				redisReply *reply = (redisReply *)redisCommand(conn, "AUTH %s", redis_password_.c_str());
				if (reply == nullptr || reply->type == REDIS_REPLY_ERROR)
				{
					std::string error_msg = reply ? reply->str : "Authentication failed";
					freeReplyObject(reply);
					redisFree(conn);
					throw std::runtime_error("Redis authentication failed: " + error_msg);
				}
				freeReplyObject(reply);
				LOG_INFO("Redis authentication successful");
			}

			// Select database if not default
			if (redis_db_ != 0)
			{
				redisReply *reply = (redisReply *)redisCommand(conn, "SELECT %d", redis_db_);
				if (reply == nullptr || reply->type == REDIS_REPLY_ERROR)
				{
					std::string error_msg = reply ? reply->str : "Database selection failed";
					freeReplyObject(reply);
					redisFree(conn);
					throw std::runtime_error("Redis database selection failed: " + error_msg);
				}
				freeReplyObject(reply);
				LOG_INFO("Selected Redis database: {}", redis_db_);
			}

			// Test connection
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

			LOG_INFO("Redis connection test successful");

			// Return as unique_ptr with custom deleter
			return std::unique_ptr<redisContext, RedisContextDeleter>(conn);
		}
		catch (const std::exception &e)
		{
			LOG_ERROR("Failed to connect to Redis: {}", e.what());
			throw;
		}
	}

	redisReply *RedisManager::execute_command(const char *format, ...)
	{
		if (!ensure_connection())
		{
			throw std::runtime_error("Redis connection is not available");
		}

		std::lock_guard<std::mutex> lock(connection_mutex_);

		va_list ap;
		va_start(ap, format);
		redisReply *reply = (redisReply *)redisvCommand(connection_.get(), format, ap);
		va_end(ap);

		if (reply == nullptr)
		{
			// Connection might be broken, mark as uninitialized
			initialized_.store(false);
			throw std::runtime_error("Redis command failed: null reply - connection may be broken");
		}

		if (reply->type == REDIS_REPLY_ERROR)
		{
			std::string error_msg = reply->str;
			freeReplyObject(reply);

			// Check if it's a connection error
			if (error_msg.find("Connection") != std::string::npos ||
				error_msg.find("broken") != std::string::npos)
			{
				initialized_.store(false);
			}

			throw std::runtime_error("Redis command failed: " + error_msg);
		}

		return reply;
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
			redisReply *reply = execute_command("SETEX %s %d %b",
												key.c_str(),
												task_expiration_,
												status_data.c_str(),
												status_data.size());

			free_reply(reply);
			LOG_DEBUG("Set task status for task_id: {}", task_id);
		}
		catch (const std::exception &e)
		{
			LOG_ERROR("Error updating task status for task_id {}: {}", task_id, e.what());
			// Don't throw here to avoid crashing the application
			// The error will be stored in the task status itself
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
			redisReply *reply = execute_command("GET %s", key.c_str());

			if (reply->type == REDIS_REPLY_NIL)
			{
				free_reply(reply);
				LOG_DEBUG("Task status not found for task_id: {}", task_id);
				return std::nullopt;
			}

			if (reply->type == REDIS_REPLY_STRING)
			{
				std::string result(reply->str, reply->len);
				free_reply(reply);
				LOG_DEBUG("Retrieved task status for task_id: {}", task_id);
				return result;
			}

			free_reply(reply);
			LOG_DEBUG("Unexpected reply type for task_id: {}", task_id);
			return std::nullopt;
		}
		catch (const std::exception &e)
		{
			LOG_ERROR("Error getting task status for task_id {}: {}", task_id, e.what());
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
			LOG_ERROR("JSON parse error for task_id {}: {}", task_id, e.what());
			return std::nullopt;
		}
		catch (const std::exception &e)
		{
			LOG_ERROR("Error parsing task status JSON for task_id {}: {}", task_id, e.what());
			return std::nullopt;
		}
	}

	std::string RedisManager::generate_uuid()
	{
		try
		{
			std::random_device rd;
			std::mt19937 gen(rd());
			uuids::uuid_random_generator generator(gen);
			uuids::uuid id = generator();
			return uuids::to_string(id);
		}
		catch (const std::exception &e)
		{
			LOG_ERROR("Error generating UUID: {}", e.what());
			// Fallback implementation
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_int_distribution<> dis(0, 15);
			std::uniform_int_distribution<> dis2(8, 11);

			std::stringstream ss;
			ss << std::hex;
			for (int i = 0; i < 8; i++)
				ss << dis(gen);
			ss << "-";
			for (int i = 0; i < 4; i++)
				ss << dis(gen);
			ss << "-4";
			for (int i = 0; i < 3; i++)
				ss << dis(gen);
			ss << "-";
			ss << dis2(gen);
			for (int i = 0; i < 3; i++)
				ss << dis(gen);
			ss << "-";
			for (int i = 0; i < 12; i++)
				ss << dis(gen);
			return ss.str();
		}
	}

	std::string RedisManager::save_application(const std::string &application_data)
	{
		try
		{
			nlohmann::json application_json = nlohmann::json::parse(application_data);
			return save_application(application_json);
		}
		catch (const nlohmann::json::parse_error &e)
		{
			throw std::runtime_error("Invalid JSON data: " + std::string(e.what()));
		}
		catch (const std::exception &e)
		{
			LOG_ERROR("Error saving application: {}", e.what());
			throw;
		}
	}

	std::string RedisManager::save_application(const nlohmann::json &application_data)
	{
		try
		{
			std::string application_id = generate_uuid();
			nlohmann::json application_with_meta = application_data;

			auto now = std::chrono::system_clock::now();
			auto in_time_t = std::chrono::system_clock::to_time_t(now);
			std::stringstream ss;
			ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%dT%H:%M:%S");

			application_with_meta["id"] = application_id;
			application_with_meta["timestamp"] = ss.str();
			application_with_meta["status"] = "new";

			fs::path applications_file = fs::path(upload_folder_) / "applications.txt";
			std::ofstream out_file(applications_file, std::ios::app);
			if (!out_file.is_open())
			{
				throw std::runtime_error("Failed to open applications file: " + applications_file.string());
			}

			out_file << application_with_meta.dump() << std::endl;
			out_file.close();

			LOG_DEBUG("Saved application with ID: {}", application_id);
			return application_id;
		}
		catch (const std::exception &e)
		{
			LOG_ERROR("Error saving application: {}", e.what());
			throw;
		}
	}

} // namespace db