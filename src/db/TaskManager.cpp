#include "TaskManager.h"
#include <db/DragonflyManager.h>
#include <logging/Logger.h>
#include <algorithm>
#include <thread>
#include <queue>
#include <condition_variable>

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

// Single background updater for all status updates
class BackgroundUpdater
{
private:
	std::queue<std::pair<std::string, nlohmann::json>> update_queue_;
	std::mutex queue_mutex_;
	std::condition_variable queue_cv_;
	std::atomic<bool> running_{true};
	std::thread worker_thread_;
	std::shared_ptr<DragonflyManager> dragonfly_manager_;

public:
	BackgroundUpdater(std::shared_ptr<DragonflyManager> manager)
		: dragonfly_manager_(std::move(manager))
	{
		worker_thread_ = std::thread(&BackgroundUpdater::process_updates, this);
	}

	~BackgroundUpdater()
	{
		running_ = false;
		queue_cv_.notify_all();
		if (worker_thread_.joinable())
		{
			worker_thread_.join();
		}
	}

	void enqueue_update(const std::string &task_id, const nlohmann::json &status)
	{
		std::lock_guard lock(queue_mutex_);
		update_queue_.emplace(task_id, status);
		queue_cv_.notify_one();
	}

private:
	void process_updates()
	{
		while (running_)
		{
			std::vector<std::pair<std::string, nlohmann::json>> batch;

			// Wait for updates with timeout
			{
				std::unique_lock lock(queue_mutex_);
				queue_cv_.wait_for(lock, std::chrono::milliseconds(100), [this]()
								   { return !update_queue_.empty() || !running_; });

				if (!running_)
					break;

				// Collect batch of updates
				while (!update_queue_.empty() && batch.size() < 50)
				{
					batch.push_back(std::move(update_queue_.front()));
					update_queue_.pop();
				}
			}

			if (batch.empty())
				continue;

			// Process the batch
			try
			{
				std::vector<std::pair<std::string, std::string>> key_values;
				for (const auto &[task_id, status] : batch)
				{
					key_values.emplace_back("task:" + task_id, status.dump());
				}

				// Try pipeline first
				try
				{
					dragonfly_manager_->pipeline_set(key_values);
					LOG_DEBUG("Successfully processed batch of {} status updates via pipeline", batch.size());
				}
				catch (const std::exception &e)
				{
					LOG_WARN("Pipeline set failed, using individual updates: {}", e.what());
					// Fall back to individual updates
					for (const auto &[task_id, status] : batch)
					{
						try
						{
							dragonfly_manager_->set_task_status(task_id, status);
						}
						catch (const std::exception &individual_error)
						{
							LOG_ERROR("Failed to update task {}: {}", task_id, individual_error.what());
						}
					}
				}
			}
			catch (const std::exception &e)
			{
				LOG_ERROR("Failed to process status update batch: {}", e.what());
			}
		}
	}
};

static std::unique_ptr<BackgroundUpdater> background_updater = nullptr;

void TaskManager::set_task_status(const std::string &task_id, const nlohmann::json &status)
{
	initialize_dragonfly_manager();

	// Update cache immediately
	auto now = std::chrono::steady_clock::now();
	{
		std::unique_lock lock(cache_mutex_);
		task_cache_[task_id] = CachedTask(status, now, now, 0);
	}

	// Initialize background updater if not already created
	static std::once_flag init_flag;
	std::call_once(init_flag, [this]()
				   {
        background_updater = std::make_unique<BackgroundUpdater>(dragonfly_manager_);
        LOG_INFO("Background status updater initialized"); });

	// Queue the update for background processing
	background_updater->enqueue_update(task_id, status);
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

	// Use the same background updater
	static std::once_flag init_flag;
	std::call_once(init_flag, [this]()
				   {
        background_updater = std::make_unique<BackgroundUpdater>(dragonfly_manager_);
        LOG_INFO("Background status updater initialized"); });

	// Queue all updates
	for (const auto &[task_id, status] : updates)
	{
		background_updater->enqueue_update(task_id, status);
	}
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

		try
		{
			// Try pipeline get first
			std::vector<std::string> keys;
			for (const auto &task_id : missing_ids)
			{
				keys.push_back("task:" + task_id);
			}

			auto pipeline_results = dragonfly_manager_->pipeline_get(keys);

			auto now = std::chrono::steady_clock::now();
			std::unique_lock lock(cache_mutex_);

			for (size_t i = 0; i < missing_ids.size(); ++i)
			{
				const auto &task_id = missing_ids[i];
				if (i < pipeline_results.size() && pipeline_results[i])
				{
					try
					{
						auto status_json = nlohmann::json::parse(*pipeline_results[i]);
						results[task_id] = status_json;
						task_cache_[task_id] = CachedTask(status_json, now, now, 1);
					}
					catch (const std::exception &e)
					{
						LOG_WARN("Failed to parse status JSON for task {}: {}", task_id, e.what());
					}
				}
			}
		}
		catch (const std::exception &e)
		{
			LOG_WARN("Pipeline get failed, falling back to individual gets: {}", e.what());
			// Fall back to individual gets
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