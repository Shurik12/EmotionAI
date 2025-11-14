#pragma once

#include <string>
#include <optional>
#include <memory>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <thread>
#include <vector>
#include <condition_variable>

#include <hiredis/hiredis.h>
#include <hiredis/async.h>
#include <nlohmann/json.hpp>

#include <common/uuid.h>

class DragonflyManager
{
public:
	DragonflyManager();
	~DragonflyManager();

	DragonflyManager(const DragonflyManager &) = delete;
	DragonflyManager &operator=(const DragonflyManager &) = delete;
	DragonflyManager(DragonflyManager &&) = delete;
	DragonflyManager &operator=(DragonflyManager &&) = delete;

	void initialize();
	void loadConfiguration();
	bool is_initialized() const { return initialized_.load(); }

	// Thread-safe operations
	void set_task_status(const std::string &task_id, const std::string &status_data);
	void set_task_status(const std::string &task_id, const nlohmann::json &status_data);

	std::optional<std::string> get_task_status(const std::string &task_id);
	std::optional<nlohmann::json> get_task_status_json(const std::string &task_id);

	std::string save_application(const std::string &application_data);
	std::string save_application(const nlohmann::json &application_data);

	static std::string generate_uuid();

	// Connection management for thread-local usage
	redisContext *get_connection();
	void return_connection(redisContext *conn);

	// Pipeline operations
	void pipeline_set(const std::vector<std::pair<std::string, std::string>> &key_values);
	std::vector<std::optional<std::string>> pipeline_get(const std::vector<std::string> &keys);

	// Connection pool stats
	struct PoolStats
	{
		size_t total_connections;
		size_t active_connections;
		size_t idle_connections;
		size_t wait_queue_size;
	};

	PoolStats get_pool_stats() const;

private:
	struct RedisContextDeleter
	{
		void operator()(redisContext *ctx) const
		{
			if (ctx)
			{
				redisFree(ctx);
			}
		}
	};

	// Connection pool
	struct ConnectionPool
	{
		std::mutex mutex;
		std::vector<std::unique_ptr<redisContext, RedisContextDeleter>> connections;
		std::condition_variable cv;
		size_t in_use = 0;
		const size_t max_pool_size = 100;
	};

	// Enhanced connection pool with pipeline support
	struct PipelineConnection
	{
		std::unique_ptr<redisContext, RedisContextDeleter> conn;
		std::vector<std::string> pending_commands;
		bool in_use = false;
	};

	std::unique_ptr<redisContext, RedisContextDeleter> create_connection();
	void free_reply(redisReply *reply);

	// Thread-safe command execution
	redisReply *execute_command(redisContext *conn, const char *format, ...);
	bool test_connection(redisContext *conn);

	// Configuration
	std::string host_;
	int port_;
	int db_;
	std::string password_;
	int task_expiration_;

	// Connection pool
	std::unique_ptr<ConnectionPool> connection_pool_;
	std::atomic<bool> initialized_{false};
	std::string upload_folder_;

	// Pipeline connections
	std::vector<PipelineConnection> pipeline_connections_;
	mutable std::mutex pipeline_mutex_;
	std::condition_variable pipeline_cv_;

	// Thread-local storage for connections
	static thread_local std::unique_ptr<redisContext, RedisContextDeleter> thread_connection_;
};