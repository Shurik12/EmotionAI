#pragma once

#include <string>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <chrono>
#include <nlohmann/json.hpp>

class TaskManager
{
public:
	static TaskManager &instance();

	// Task status with local caching
	void set_task_status(const std::string &task_id, const nlohmann::json &status);
	std::optional<nlohmann::json> get_task_status(const std::string &task_id);

	// Batch operations
	void batch_update_status(const std::vector<std::pair<std::string, nlohmann::json>> &updates);
	std::unordered_map<std::string, nlohmann::json> batch_get_status(const std::vector<std::string> &task_ids);

	// Cache management
	void cleanup_expired_tasks();
	size_t get_cache_size() const;

private:
	TaskManager() = default;

	struct CachedTask
	{
		nlohmann::json status;
		std::chrono::steady_clock::time_point last_updated;
		std::chrono::steady_clock::time_point last_accessed;
		uint32_t access_count;

		// Default constructor for std::unordered_map
		CachedTask()
			: status(nlohmann::json::object()), last_updated(std::chrono::steady_clock::now()), last_accessed(std::chrono::steady_clock::now()), access_count(0) {}

		// Parameterized constructor
		CachedTask(const nlohmann::json &s,
				   std::chrono::steady_clock::time_point updated,
				   std::chrono::steady_clock::time_point accessed,
				   uint32_t count)
			: status(s), last_updated(updated), last_accessed(accessed), access_count(count) {}
	};

	mutable std::shared_mutex cache_mutex_;
	std::unordered_map<std::string, CachedTask> task_cache_;
	std::atomic<size_t> cache_hits_{0};
	std::atomic<size_t> cache_misses_{0};

	// Cache configuration
	static constexpr size_t MAX_CACHE_SIZE = 10000;
	static constexpr auto CACHE_TTL = std::chrono::minutes(5);
	static constexpr auto ACCESS_TTL = std::chrono::minutes(10);

	// Dragonfly manager instance (passed from outside)
	std::shared_ptr<class DragonflyManager> dragonfly_manager_;

	void initialize_dragonfly_manager();

public:
	// Set the dragonfly manager (call this before using TaskManager)
	void set_dragonfly_manager(std::shared_ptr<class DragonflyManager> manager)
	{
		dragonfly_manager_ = std::move(manager);
	}
};