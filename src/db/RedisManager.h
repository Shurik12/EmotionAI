#pragma once

#include <string>
#include <optional>
#include <memory>
#include <mutex>
#include <atomic>

#include <hiredis/hiredis.h>
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

		void set_task_status(const std::string &task_id, const std::string &status_data);
		void set_task_status(const std::string &task_id, const nlohmann::json &status_data);

		std::optional<std::string> get_task_status(const std::string &task_id);
		std::optional<nlohmann::json> get_task_status_json(const std::string &task_id);

		std::string save_application(const std::string &application_data);
		std::string save_application(const nlohmann::json &application_data);

		static std::string generate_uuid();

	private:
		// Custom deleter for redisContext
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

		std::unique_ptr<redisContext, RedisContextDeleter> create_connection();
		void free_reply(redisReply *reply);

		redisReply *execute_command(const char *format, ...);

		bool ensure_connection();

		// Cached configuration
		std::string redis_host_;
		int redis_port_;
		int redis_db_;
		std::string redis_password_;

		std::unique_ptr<redisContext, RedisContextDeleter> connection_;
		std::mutex connection_mutex_;
		std::atomic<bool> initialized_{false};
		std::string upload_folder_;
		int task_expiration_;
	};

} // namespace db