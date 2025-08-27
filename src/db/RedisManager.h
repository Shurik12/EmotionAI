#pragma once

#include <string>
#include <optional>
#include <memory>
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

		redisContext *connection();
		const redisContext *connection() const;

		void set_task_status(const std::string &task_id, const std::string &status_data);
		void set_task_status(const std::string &task_id, const nlohmann::json &status_data);

		std::optional<std::string> get_task_status(const std::string &task_id);
		std::optional<nlohmann::json> get_task_status_json(const std::string &task_id);

		std::string save_application(const std::string &application_data);
		std::string save_application(const nlohmann::json &application_data);

		// Helper method to generate UUID using the provided uuid.h
		static std::string generate_uuid();

	private:
		redisContext *create_connection();
		void free_reply(redisReply *reply);

		redisContext *connection_;
		std::string upload_folder_;
		int task_expiration_;
	};

} // namespace db