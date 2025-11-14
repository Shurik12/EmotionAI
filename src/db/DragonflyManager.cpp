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
#include "DragonflyManager.h"

namespace fs = std::filesystem;

// Thread-local connection
thread_local std::unique_ptr<redisContext, DragonflyManager::RedisContextDeleter> DragonflyManager::thread_connection_ = nullptr;

DragonflyManager::DragonflyManager()
	: port_(6379),
	  db_(0),
	  task_expiration_(3600),
	  connection_pool_(std::make_unique<ConnectionPool>())
{
}

DragonflyManager::~DragonflyManager()
{
	initialized_.store(false);

	// Cleanup connection pool
	std::lock_guard<std::mutex> lock(connection_pool_->mutex);
	connection_pool_->connections.clear();
}

void DragonflyManager::initialize()
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
		LOG_INFO("DragonflyManager initialized successfully with connection pool");
	}
	else
	{
		LOG_ERROR("Failed to initialize DragonflyManager - cannot establish connection");
		throw std::runtime_error("Failed to connect to DragonflyDB");
	}
}

void DragonflyManager::loadConfiguration()
{
	auto &config = Config::instance();

	host_ = config.dragonfly().host;
	port_ = config.dragonfly().port;
	db_ = config.dragonfly().db;
	password_ = config.dragonfly().password;
	upload_folder_ = config.paths().upload;
	task_expiration_ = config.app().task_expiration;

	LOG_INFO("DragonflyDB configuration loaded: {}:{} (DB: {})", host_, port_, db_);
}

redisContext *DragonflyManager::get_connection()
{
	// First try thread-local connection
	if (thread_connection_ && test_connection(thread_connection_.get()))
	{
		return thread_connection_.get();
	}

	// Thread-local connection doesn't exist or is bad, try pool
	std::unique_lock<std::mutex> lock(connection_pool_->mutex);

	// Try to get connection from pool with timeout
	auto start_time = std::chrono::steady_clock::now();
	auto timeout = std::chrono::seconds(5); // 5 second timeout

	while (std::chrono::steady_clock::now() - start_time < timeout)
	{
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

			LOG_DEBUG("Creating new DragonflyDB connection for thread");
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
					LOG_WARN("Newly created DragonflyDB connection test failed");
				}
			}

			// If we get here, connection creation or test failed
			std::lock_guard<std::mutex> lock2(connection_pool_->mutex);
			connection_pool_->in_use--;
		}

		// Wait a bit before retrying
		lock.lock();
		if (connection_pool_->cv.wait_for(lock, std::chrono::milliseconds(100), [this]()
										  { return !connection_pool_->connections.empty() || connection_pool_->in_use < connection_pool_->max_pool_size; }))
		{
			continue; // Retry if condition met
		}
	}

	LOG_ERROR("DragonflyDB connection pool timeout after 5 seconds");
	return nullptr;
}

void DragonflyManager::return_connection(redisContext *conn)
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

std::unique_ptr<redisContext, DragonflyManager::RedisContextDeleter> DragonflyManager::create_connection()
{
	redisContext *conn = nullptr;

	try
	{
		LOG_DEBUG("Creating new DragonflyDB connection: {}:{} (DB: {})", host_, port_, db_);

		struct timeval timeout = {1, 500000}; // 1.5 seconds
		conn = redisConnectWithTimeout(host_.c_str(), port_, timeout);

		if (conn == nullptr || conn->err)
		{
			std::string error_msg = conn ? conn->errstr : "Cannot allocate redis context";
			LOG_ERROR("Failed to connect to DragonflyDB: {}", error_msg);
			if (conn)
			{
				redisFree(conn);
			}
			return nullptr;
		}

		// Authenticate if password is provided
		if (!password_.empty())
		{
			redisReply *reply = (redisReply *)redisCommand(conn, "AUTH %s", password_.c_str());
			if (reply == nullptr || reply->type == REDIS_REPLY_ERROR)
			{
				std::string error_msg = reply ? reply->str : "Authentication failed";
				LOG_ERROR("DragonflyDB authentication failed: {}", error_msg);
				freeReplyObject(reply);
				redisFree(conn);
				return nullptr;
			}
			freeReplyObject(reply);
			LOG_DEBUG("DragonflyDB authentication successful");
		}

		// Select database if not default
		if (db_ != 0)
		{
			redisReply *reply = (redisReply *)redisCommand(conn, "SELECT %d", db_);
			if (reply == nullptr || reply->type == REDIS_REPLY_ERROR)
			{
				std::string error_msg = reply ? reply->str : "Database selection failed";
				LOG_ERROR("DragonflyDB database selection failed: {}", error_msg);
				freeReplyObject(reply);
				redisFree(conn);
				return nullptr;
			}
			freeReplyObject(reply);
			LOG_DEBUG("Selected DragonflyDB database: {}", db_);
		}

		// Test the connection
		redisReply *reply = (redisReply *)redisCommand(conn, "PING");
		if (reply == nullptr || reply->type == REDIS_REPLY_ERROR)
		{
			std::string error_msg = reply ? reply->str : "PING failed";
			LOG_ERROR("DragonflyDB connection test failed: {}", error_msg);
			freeReplyObject(reply);
			redisFree(conn);
			return nullptr;
		}

		// Check if PING returned PONG
		bool ping_success = false;
		if (reply->type == REDIS_REPLY_STATUS || reply->type == REDIS_REPLY_STRING)
		{
			ping_success = (std::string(reply->str) == "PONG");
		}

		freeReplyObject(reply);

		if (!ping_success)
		{
			LOG_ERROR("DragonflyDB PING did not return PONG");
			redisFree(conn);
			return nullptr;
		}

		LOG_DEBUG("DragonflyDB connection created and tested successfully");
		return std::unique_ptr<redisContext, RedisContextDeleter>(conn);
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Exception creating DragonflyDB connection: {}", e.what());
		if (conn)
		{
			redisFree(conn);
		}
		return nullptr;
	}
}

bool DragonflyManager::test_connection(redisContext *conn)
{
	if (!conn || conn->err)
	{
		LOG_DEBUG("DragonflyDB connection is null or has error: {}", conn ? conn->errstr : "null");
		return false;
	}

	redisReply *reply = nullptr;
	try
	{
		reply = (redisReply *)redisCommand(conn, "PING");
		if (!reply)
		{
			LOG_DEBUG("DragonflyDB PING command returned null");
			return false;
		}

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
			LOG_DEBUG("DragonflyDB PING returned unexpected type: {}", reply->type);
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

redisReply *DragonflyManager::execute_command(redisContext *conn, const char *format, ...)
{
	if (!conn || conn->err)
	{
		throw std::runtime_error("DragonflyDB connection is not available");
	}

	va_list ap;
	va_start(ap, format);
	redisReply *reply = (redisReply *)redisvCommand(conn, format, ap);
	va_end(ap);

	if (reply == nullptr)
	{
		throw std::runtime_error("DragonflyDB command failed: null reply - connection may be broken");
	}

	if (reply->type == REDIS_REPLY_ERROR)
	{
		std::string error_msg = reply->str;
		freeReplyObject(reply);
		throw std::runtime_error("DragonflyDB command failed: " + error_msg);
	}

	return reply;
}

void DragonflyManager::free_reply(redisReply *reply)
{
	if (reply)
	{
		freeReplyObject(reply);
	}
}

void DragonflyManager::set_task_status(const std::string &task_id, const std::string &status_data)
{
	auto conn = get_connection();
	if (!conn)
	{
		LOG_ERROR("No DragonflyDB connection available for setting task status for task_id: {}", task_id);
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
	}
}

void DragonflyManager::set_task_status(const std::string &task_id, const nlohmann::json &status_data)
{
	set_task_status(task_id, status_data.dump());
}

std::optional<std::string> DragonflyManager::get_task_status(const std::string &task_id)
{
	auto conn = get_connection();
	if (!conn)
	{
		LOG_ERROR("No DragonflyDB connection available for getting task status for task_id: {}", task_id);
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

std::optional<nlohmann::json> DragonflyManager::get_task_status_json(const std::string &task_id)
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

std::string DragonflyManager::generate_uuid()
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

std::string DragonflyManager::save_application(const std::string &application_data)
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

std::string DragonflyManager::save_application(const nlohmann::json &application_data)
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

void DragonflyManager::pipeline_set(const std::vector<std::pair<std::string, std::string>> &key_values)
{
	if (key_values.empty())
		return;

	auto conn = get_connection();
	if (!conn)
	{
		throw std::runtime_error("No DragonflyDB connection available for pipeline");
	}

	try
	{
		// Send all commands
		for (const auto &[key, value] : key_values)
		{
			redisAppendCommand(conn, "SETEX %s %d %b",
							   key.c_str(), task_expiration_,
							   value.c_str(), value.size());
		}

		// Read all replies
		for (size_t i = 0; i < key_values.size(); ++i)
		{
			redisReply *reply = nullptr;
			if (redisGetReply(conn, (void **)&reply) == REDIS_OK)
			{
				free_reply(reply);
			}
		}
	}
	catch (const std::exception &e)
	{
		return_connection(conn);
		throw;
	}

	return_connection(conn);
}

std::vector<std::optional<std::string>> DragonflyManager::pipeline_get(const std::vector<std::string> &keys)
{
	std::vector<std::optional<std::string>> results;
	if (keys.empty())
		return results;

	auto conn = get_connection();
	if (!conn)
	{
		throw std::runtime_error("No DragonflyDB connection available for pipeline");
	}

	try
	{
		// Send all GET commands
		for (const auto &key : keys)
		{
			redisAppendCommand(conn, "GET %s", key.c_str());
		}

		// Read all replies
		for (size_t i = 0; i < keys.size(); ++i)
		{
			redisReply *reply = nullptr;
			if (redisGetReply(conn, (void **)&reply) == REDIS_OK)
			{
				if (reply->type == REDIS_REPLY_STRING)
				{
					results.emplace_back(std::string(reply->str, reply->len));
				}
				else
				{
					results.emplace_back(std::nullopt);
				}
				free_reply(reply);
			}
			else
			{
				results.emplace_back(std::nullopt);
			}
		}
	}
	catch (const std::exception &e)
	{
		return_connection(conn);
		throw;
	}

	return_connection(conn);
	return results;
}

DragonflyManager::PoolStats DragonflyManager::get_pool_stats() const
{
	std::lock_guard<std::mutex> lock(connection_pool_->mutex);
	return PoolStats{
		.total_connections = connection_pool_->connections.size() + connection_pool_->in_use,
		.active_connections = connection_pool_->in_use,
		.idle_connections = connection_pool_->connections.size(),
		.wait_queue_size = 0 // This would need a more sophisticated implementation
	};
}