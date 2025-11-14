#include "TaskManager.h"
#include <db/DragonflyManager.h>
#include <logging/Logger.h>
#include <algorithm>
#include <thread>

TaskManager &TaskManager::instance()
{
	static TaskManager instance;
	return instance;
}

void TaskManager::initialize_dragonfly_manager()
{
	if (!dragonfly_manager_)
	{
		LOG_ERROR("DragonflyManager not set in TaskManager");
		throw std::runtime_error("DragonflyManager not initialized in TaskManager");
	}
}

void TaskManager::set_task_status(const std::string &task_id, const nlohmann::json &status)
{
	initialize_dragonfly_manager();

	// Update cache with write-through
	auto now = std::chrono::steady_clock::now();
	{
		std::unique_lock lock(cache_mutex_);
		task_cache_[task_id] = CachedTask(status, now, now, 0);
	}

	// Async write to DragonflyDB
	std::thread([this, task_id, status]()
				{
        try {
            dragonfly_manager_->set_task_status(task_id, status);
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to update task status in DragonflyDB: {}", e.what());
        } })
		.detach();
}

std::optional<nlohmann::json> TaskManager::get_task_status(const std::string &task_id)
{
	// First try cache
	{
		std::shared_lock lock(cache_mutex_);
		auto it = task_cache_.find(task_id);
		if (it != task_cache_.end())
		{
			auto &cached_task = it->second;

			// Check if cache entry is still valid
			auto now = std::chrono::steady_clock::now();
			if (now - cached_task.last_updated < CACHE_TTL)
			{
				cached_task.last_accessed = now;
				cached_task.access_count++;
				cache_hits_++;
				return cached_task.status;
			}
		}
	}

	cache_misses_++;

	// Cache miss, fetch from DragonflyDB
	initialize_dragonfly_manager();
	auto status = dragonfly_manager_->get_task_status_json(task_id);

	if (status)
	{
		// Update cache
		auto now = std::chrono::steady_clock::now();
		std::unique_lock lock(cache_mutex_);
		task_cache_[task_id] = CachedTask(*status, now, now, 1);
	}

	return status;
}

void TaskManager::batch_update_status(const std::vector<std::pair<std::string, nlohmann::json>> &updates)
{
	if (updates.empty())
		return;

	initialize_dragonfly_manager();

	// Update cache
	auto now = std::chrono::steady_clock::now();
	{
		std::unique_lock lock(cache_mutex_);
		for (const auto &[task_id, status] : updates)
		{
			task_cache_[task_id] = CachedTask(status, now, now, 0);
		}
	}

	// Batch update DragonflyDB
	std::thread([this, updates]()
				{
        try {
            for (const auto& [task_id, status] : updates) {
                dragonfly_manager_->set_task_status(task_id, status);
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Batch update failed: {}", e.what());
        } })
		.detach();
}

std::unordered_map<std::string, nlohmann::json> TaskManager::batch_get_status(const std::vector<std::string> &task_ids)
{
	std::unordered_map<std::string, nlohmann::json> results;
	std::vector<std::string> missing_ids;

	// First pass: check cache
	{
		std::shared_lock lock(cache_mutex_);
		auto now = std::chrono::steady_clock::now();

		for (const auto &task_id : task_ids)
		{
			auto it = task_cache_.find(task_id);
			if (it != task_cache_.end() && (now - it->second.last_updated < CACHE_TTL))
			{
				results[task_id] = it->second.status;
				it->second.last_accessed = now;
				it->second.access_count++;
				cache_hits_++;
			}
			else
			{
				missing_ids.push_back(task_id);
				cache_misses_++;
			}
		}
	}

	// Second pass: fetch missing from DragonflyDB
	if (!missing_ids.empty())
	{
		initialize_dragonfly_manager();
		std::unique_lock lock(cache_mutex_);
		auto now = std::chrono::steady_clock::now();

		for (const auto &task_id : missing_ids)
		{
			auto status = dragonfly_manager_->get_task_status_json(task_id);
			if (status)
			{
				results[task_id] = *status;
				task_cache_[task_id] = CachedTask(*status, now, now, 1);
			}
		}
	}

	return results;
}

void TaskManager::cleanup_expired_tasks()
{
	std::unique_lock lock(cache_mutex_);
	auto now = std::chrono::steady_clock::now();

	for (auto it = task_cache_.begin(); it != task_cache_.end();)
	{
		const auto &cached_task = it->second;
		if (now - cached_task.last_accessed > ACCESS_TTL)
		{
			it = task_cache_.erase(it);
		}
		else
		{
			++it;
		}
	}

	// If cache is still too large, remove least recently used
	if (task_cache_.size() > MAX_CACHE_SIZE)
	{
		std::vector<std::pair<std::string, CachedTask>> tasks(task_cache_.begin(), task_cache_.end());
		std::sort(tasks.begin(), tasks.end(), [](const auto &a, const auto &b)
				  { return a.second.last_accessed < b.second.last_accessed; });

		for (size_t i = 0; i < tasks.size() - MAX_CACHE_SIZE / 2; ++i)
		{
			task_cache_.erase(tasks[i].first);
		}
	}
}

size_t TaskManager::get_cache_size() const
{
	std::shared_lock lock(cache_mutex_);
	return task_cache_.size();
}