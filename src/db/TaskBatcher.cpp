#include "TaskBatcher.h"
#include <logging/Logger.h>
#include <chrono>

TaskBatcher::TaskBatcher(size_t batch_size, std::chrono::milliseconds batch_timeout)
	: batch_size_(batch_size), batch_timeout_(batch_timeout)
{
}

TaskBatcher::~TaskBatcher()
{
	stop();
}

void TaskBatcher::enqueue(const std::string &task_id, const nlohmann::json &status)
{
	std::lock_guard lock(queue_mutex_);
	batch_queue_.push({task_id, status});
	queue_cv_.notify_one();
}

void TaskBatcher::set_batch_callback(BatchCallback callback)
{
	batch_callback_ = std::move(callback);
}

void TaskBatcher::start()
{
	if (running_.exchange(true))
		return;

	batch_thread_ = std::thread(&TaskBatcher::process_batches, this);
	LOG_INFO("TaskBatcher started with batch_size={}, timeout={}ms",
			 batch_size_, batch_timeout_.count());
}

void TaskBatcher::stop()
{
	if (!running_.exchange(false))
		return;

	queue_cv_.notify_all();
	if (batch_thread_.joinable())
	{
		batch_thread_.join();
	}
}

void TaskBatcher::process_batches()
{
	std::vector<BatchItem> current_batch;
	current_batch.reserve(batch_size_);

	while (running_)
	{
		std::unique_lock lock(queue_mutex_);

		// Wait for batch to fill or timeout
		if (batch_queue_.empty())
		{
			queue_cv_.wait_for(lock, batch_timeout_, [this]()
							   { return !batch_queue_.empty() || !running_; });
		}

		if (!running_)
			break;

		// Collect batch items
		while (!batch_queue_.empty() && current_batch.size() < batch_size_)
		{
			current_batch.push_back(std::move(batch_queue_.front()));
			batch_queue_.pop();
		}

		lock.unlock();

		// Process batch if we have items
		if (!current_batch.empty() && batch_callback_)
		{
			try
			{
				std::vector<std::pair<std::string, nlohmann::json>> batch_data;
				batch_data.reserve(current_batch.size());

				for (auto &item : current_batch)
				{
					batch_data.emplace_back(std::move(item.task_id), std::move(item.status));
				}

				batch_callback_(batch_data);
				LOG_DEBUG("Processed batch of {} tasks", current_batch.size());
			}
			catch (const std::exception &e)
			{
				LOG_ERROR("Batch processing failed: {}", e.what());
			}

			current_batch.clear();
		}
	}
}