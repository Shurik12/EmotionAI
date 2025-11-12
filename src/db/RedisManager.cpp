#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <random>
#include <cstdarg>
#include <condition_variable>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <common/uuid.h>
#include <config/Config.h>
#include <logging/Logger.h>
#include "RedisManager.h"

namespace fs = std::filesystem;
namespace db
{

	// Thread-local connection
	thread_local std::unique_ptr<redisContext, RedisManager::RedisContextDeleter> RedisManager::thread_connection_ = nullptr;

	RedisManager::RedisManager()
		: redis_port_(6379),
		  redis_db_(0),
		  task_expiration_(3600),
		  connection_pool_(std::make_unique<ConnectionPool>())
	{
	}

	RedisManager::~RedisManager()
	{
		initialized_.store(false);

		// Cleanup connection pool
		std::lock_guard<std::mutex> lock(connection_pool_->mutex);
		connection_pool_->connections.clear();
	}

	void RedisManager::initialize()
	{
		loadConfiguration();

		// Test connection by creating one
		auto test_conn = create_connection();
		if (test_conn)
		{
			initialized_.store(true);
			// Add to pool
			std::lock_guard<std::mutex> lock(connection_pool_->mutex);
			connection_pool_->connections.push_back(std::move(test_conn));
			LOG_INFO("RedisManager initialized successfully with connection pool");
		}
		else
		{
			LOG_ERROR("Failed to initialize RedisManager - cannot establish connection");
			throw std::runtime_error("Failed to connect to Redis");
		}
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

	redisContext *RedisManager::get_connection()
	{
		// First try thread-local connection
		if (thread_connection_ && test_connection(thread_connection_.get()))
		{
			return thread_connection_.get();
		}

		// Thread-local connection doesn't exist or is bad, try pool
		std::unique_lock<std::mutex> lock(connection_pool_->mutex);

		// Try to get connection from pool
		while (!connection_pool_->connections.empty())
		{
			auto conn = std::move(connection_pool_->connections.back());
			connection_pool_->connections.pop_back();
			connection_pool_->in_use++;
			lock.unlock();

			if (test_connection(conn.get()))
			{
				// Move to thread-local storage
				thread_connection_ = std::move(conn);
				return thread_connection_.get();
			}
			else
			{
				// Connection is bad, try next one
				lock.lock();
				connection_pool_->in_use--;
			}
		}

		// No available connections in pool, create new one if under limit
		if (connection_pool_->in_use < connection_pool_->max_pool_size)
		{
			connection_pool_->in_use++;
			lock.unlock();

			LOG_DEBUG("Creating new Redis connection for thread");
			auto new_conn = create_connection();
			if (new_conn)
			{
				if (test_connection(new_conn.get()))
				{
					thread_connection_ = std::move(new_conn);
					return thread_connection_.get();
				}
				else
				{
					LOG_WARN("Newly created Redis connection test failed");
				}
			}

			// If we get here, connection creation or test failed
			std::lock_guard<std::mutex> lock2(connection_pool_->mutex);
			connection_pool_->in_use--;
			LOG_ERROR("Failed to create or validate new Redis connection");
		}
		else
		{
			LOG_ERROR("Redis connection pool exhausted (max: {})", connection_pool_->max_pool_size);
		}

		return nullptr;
	}

	void RedisManager::return_connection(redisContext *conn)
	{
		if (!conn)
			return;

		// If this is the thread-local connection, just keep it
		if (thread_connection_.get() == conn)
		{
			return; // Thread keeps its connection
		}

		// Return to pool
		std::lock_guard<std::mutex> lock(connection_pool_->mutex);
		if (test_connection(conn))
		{
			if (connection_pool_->connections.size() < connection_pool_->max_pool_size)
			{
				connection_pool_->connections.push_back(
					std::unique_ptr<redisContext, RedisContextDeleter>(conn));
			}
			else
			{
				// Pool is full, close the connection
				redisFree(conn);
			}
		}
		else
		{
			// Connection is bad, close it
			redisFree(conn);
		}
		connection_pool_->in_use--;
	}

	std::unique_ptr<redisContext, RedisManager::RedisContextDeleter> RedisManager::create_connection()
	{
		redisContext *conn = nullptr;

		try
		{
			LOG_DEBUG("Creating new Redis connection: {}:{} (DB: {})", redis_host_, redis_port_, redis_db_);

			struct timeval timeout = {1, 500000}; // 1.5 seconds
			conn = redisConnectWithTimeout(redis_host_.c_str(), redis_port_, timeout);

			if (conn == nullptr || conn->err)
			{
				std::string error_msg = conn ? conn->errstr : "Cannot allocate redis context";
				LOG_ERROR("Failed to connect to Redis: {}", error_msg);
				if (conn)
				{
					redisFree(conn);
				}
				return nullptr;
			}

			// Authenticate if password is provided
			if (!redis_password_.empty())
			{
				redisReply *reply = (redisReply *)redisCommand(conn, "AUTH %s", redis_password_.c_str());
				if (reply == nullptr || reply->type == REDIS_REPLY_ERROR)
				{
					std::string error_msg = reply ? reply->str : "Authentication failed";
					LOG_ERROR("Redis authentication failed: {}", error_msg);
					freeReplyObject(reply);
					redisFree(conn);
					return nullptr;
				}
				freeReplyObject(reply);
				LOG_DEBUG("Redis authentication successful");
			}

			// Select database if not default
			if (redis_db_ != 0)
			{
				redisReply *reply = (redisReply *)redisCommand(conn, "SELECT %d", redis_db_);
				if (reply == nullptr || reply->type == REDIS_REPLY_ERROR)
				{
					std::string error_msg = reply ? reply->str : "Database selection failed";
					LOG_ERROR("Redis database selection failed: {}", error_msg);
					freeReplyObject(reply);
					redisFree(conn);
					return nullptr;
				}
				freeReplyObject(reply);
				LOG_DEBUG("Selected Redis database: {}", redis_db_);
			}

			// Test the connection - handle both STATUS and STRING reply types
			redisReply *reply = (redisReply *)redisCommand(conn, "PING");
			if (reply == nullptr || reply->type == REDIS_REPLY_ERROR)
			{
				std::string error_msg = reply ? reply->str : "PING failed";
				LOG_ERROR("Redis connection test failed: {}", error_msg);
				freeReplyObject(reply);
				redisFree(conn);
				return nullptr;
			}

			// Check if PING returned PONG (can be STATUS or STRING type)
			bool ping_success = false;
			if (reply->type == REDIS_REPLY_STATUS || reply->type == REDIS_REPLY_STRING)
			{
				ping_success = (std::string(reply->str) == "PONG");
			}

			freeReplyObject(reply);

			if (!ping_success)
			{
				LOG_ERROR("Redis PING did not return PONG");
				redisFree(conn);
				return nullptr;
			}

			LOG_DEBUG("Redis connection created and tested successfully");
			return std::unique_ptr<redisContext, RedisContextDeleter>(conn);
		}
		catch (const std::exception &e)
		{
			LOG_ERROR("Exception creating Redis connection: {}", e.what());
			if (conn)
			{
				redisFree(conn);
			}
			return nullptr;
		}
	}

	bool RedisManager::test_connection(redisContext *conn)
	{
		if (!conn || conn->err)
		{
			LOG_DEBUG("Redis connection is null or has error: {}", conn ? conn->errstr : "null");
			return false;
		}

		redisReply *reply = nullptr;
		try
		{
			reply = (redisReply *)redisCommand(conn, "PING");
			if (!reply)
			{
				LOG_DEBUG("Redis PING command returned null");
				return false;
			}

			// PING can return either STATUS or STRING reply type, both with "PONG" content
			bool success = false;
			if (reply->type == REDIS_REPLY_STATUS)
			{
				success = (std::string(reply->str) == "PONG");
			}
			else if (reply->type == REDIS_REPLY_STRING)
			{
				success = (std::string(reply->str) == "PONG");
			}
			else
			{
				LOG_DEBUG("Redis PING returned unexpected type: {}", reply->type);
				success = false;
			}

			freeReplyObject(reply);
			return success;
		}
		catch (...)
		{
			if (reply)
			{
				freeReplyObject(reply);
			}
			return false;
		}
	}

	redisReply *RedisManager::execute_command(redisContext *conn, const char *format, ...)
	{
		if (!conn || conn->err)
		{
			throw std::runtime_error("Redis connection is not available");
		}

		va_list ap;
		va_start(ap, format);
		redisReply *reply = (redisReply *)redisvCommand(conn, format, ap);
		va_end(ap);

		if (reply == nullptr)
		{
			throw std::runtime_error("Redis command failed: null reply - connection may be broken");
		}

		if (reply->type == REDIS_REPLY_ERROR)
		{
			std::string error_msg = reply->str;
			freeReplyObject(reply);
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
		auto conn = get_connection();
		if (!conn)
		{
			LOG_ERROR("No Redis connection available for setting task status for task_id: {}", task_id);
			return;
		}

		try
		{
			std::string key = "task:" + task_id;
			redisReply *reply = execute_command(conn, "SETEX %s %d %b",
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
			// Don't throw to avoid crashing the application
		}
	}

	void RedisManager::set_task_status(const std::string &task_id, const nlohmann::json &status_data)
	{
		set_task_status(task_id, status_data.dump());
	}

	std::optional<std::string> RedisManager::get_task_status(const std::string &task_id)
	{
		auto conn = get_connection();
		if (!conn)
		{
			LOG_ERROR("No Redis connection available for getting task status for task_id: {}", task_id);
			return std::nullopt;
		}

		try
		{
			std::string key = "task:" + task_id;
			redisReply *reply = execute_command(conn, "GET %s", key.c_str());

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