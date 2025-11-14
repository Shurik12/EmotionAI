#pragma once

#include <string>
#include <optional>
#include <memory>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <thread>

#include <hiredis/hiredis.h>
#include <hiredis/async.h>
#include <nlohmann/json.hpp>

#include <common/uuid.h>

namespace db
{

	class RedisManager
	{
	public:
		RedisManager();
		~RedisManager();

		RedisManager(const RedisManager &) = delete;
		RedisManager &operator=(const RedisManager &) = delete;
		RedisManager(RedisManager &&) = delete;
		RedisManager &operator=(RedisManager &&) = delete;

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
			const size_t max_pool_size = 20;
		};

		std::unique_ptr<redisContext, RedisContextDeleter> create_connection();
		void free_reply(redisReply *reply);

		// Thread-safe command execution
		redisReply *execute_command(redisContext *conn, const char *format, ...);
		bool test_connection(redisContext *conn);

		// Configuration
		std::string redis_host_;
		int redis_port_;
		int redis_db_;
		std::string redis_password_;
		int task_expiration_;

		// Connection pool
		std::unique_ptr<ConnectionPool> connection_pool_;
		std::atomic<bool> initialized_{false};
		std::string upload_folder_;

		// Thread-local storage for connections
		static thread_local std::unique_ptr<redisContext, RedisContextDeleter> thread_connection_;
	};

} // namespace db