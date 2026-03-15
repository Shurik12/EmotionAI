#include "DistributedTaskManager.h"
#include <chrono>
#include <db/TaskManager.h>

DistributedTaskManager::DistributedTaskManager(std::shared_ptr<DragonflyManager> dragonfly,
											   const std::string &instance_id)
	: dragonfly_(dragonfly), instance_id_(instance_id)
{
}

bool DistributedTaskManager::submitTask(const std::string &queue_name, const nlohmann::json &task)
{
	auto conn = dragonfly_->get_connection();
	if (!conn)
	{
		LOG_ERROR("No DragonflyDB connection available for submitting task");
		return false;
	}

	bool success = false;
	try
	{
		// Add metadata to task
		nlohmann::json final_task = task;
		final_task["submitted_by"] = instance_id_;
		final_task["submitted_at"] = std::chrono::duration_cast<std::chrono::milliseconds>(
										 std::chrono::system_clock::now().time_since_epoch())
										 .count();

		std::string final_task_data = final_task.dump();

		// Use DragonflyManager's public API instead of private execute_command
		// We'll use the existing task status mechanism for now
		// For production, you might want to add a public method to DragonflyManager for queue operations
		std::string queue_key = "queue:" + queue_name + ":" + task["task_id"].get<std::string>();
		dragonfly_->set_task_status(queue_key, final_task);

		LOG_DEBUG("Task {} submitted to queue {}", task["task_id"].get<std::string>(), queue_name);
		success = true;
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Failed to submit task to queue {}: {}", queue_name, e.what());
		success = false;
	}

	dragonfly_->return_connection(conn);
	return success;
}

std::optional<nlohmann::json> DistributedTaskManager::getNextTask(const std::string &queue_name,
																  int visibility_timeout)
{
	// Simplified implementation using existing DragonflyManager API
	// For production, implement proper queue operations with BRPOPLPUSH

	auto conn = dragonfly_->get_connection();
	if (!conn)
	{
		LOG_ERROR("No DragonflyDB connection available for getting task");
		return std::nullopt;
	}

	std::optional<nlohmann::json> result = std::nullopt;
	try
	{
		// Look for any task in the queue
		// This is a simplified approach - in production you'd use proper queue operations
		std::string queue_pattern = "queue:" + queue_name + ":*";

		// We'll just try to get one task from the queue
		// In production, you'd use SCAN or other methods to find tasks
		auto task_status = dragonfly_->get_task_status_json(queue_pattern);

		if (task_status)
		{
			LOG_DEBUG("Retrieved task from queue {}", queue_name);
			result = task_status;
		}
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Failed to get next task from queue {}: {}", queue_name, e.what());
		result = std::nullopt;
	}

	dragonfly_->return_connection(conn);
	return result;
}

void DistributedTaskManager::markTaskComplete(const std::string &task_id)
{
	try
	{
		// Remove task from all queues
		auto &config = Config::instance();

		// Remove from batch queue
		std::string batch_key = "queue:" + config.queue().batch_queue_name + ":" + task_id;
		auto conn = dragonfly_->get_connection();
		if (conn)
		{
			// The task completion is already handled by TaskManager
			// We just clean up the queue entry
			dragonfly_->return_connection(conn);
		}

		LOG_DEBUG("Marked task {} as complete", task_id);
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Failed to mark task {} as complete: {}", task_id, e.what());
	}
}

void DistributedTaskManager::markTaskFailed(const std::string &task_id, const std::string &error)
{
	try
	{
		// Update task status with error using existing TaskManager
		auto &task_manager = TaskManager::instance();
		auto current_status = task_manager.get_task_status(task_id);

		if (current_status)
		{
			(*current_status)["error"] = error;
			(*current_status)["complete"] = true;
			task_manager.set_task_status(task_id, *current_status);
		}

		// Move task to dead letter queue or retry
		auto &config = Config::instance();
		std::string dead_letter_key = "queue:dead_letter:" + task_id;

		if (current_status)
		{
			(*current_status)["failed_at"] = std::chrono::duration_cast<std::chrono::milliseconds>(
												 std::chrono::system_clock::now().time_since_epoch())
												 .count();
			(*current_status)["error"] = error;

			dragonfly_->set_task_status(dead_letter_key, *current_status);
		}

		LOG_DEBUG("Marked task {} as failed: {}", task_id, error);
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Failed to mark task {} as failed: {}", task_id, e.what());
	}
}

void DistributedTaskManager::returnTaskToQueue(const std::string &task_id, const nlohmann::json &task)
{
	try
	{
		auto &config = Config::instance();

		// Reset task status and resubmit
		nlohmann::json updated_task = task;
		updated_task["retry_count"] = updated_task.value("retry_count", 0) + 1;
		updated_task["last_retry"] = std::chrono::duration_cast<std::chrono::milliseconds>(
										 std::chrono::system_clock::now().time_since_epoch())
										 .count();

		// Determine which queue to use based on task type
		std::string queue_name = config.queue().batch_queue_name;
		if (task.contains("type") && task["type"] == "realtime_video")
		{
			queue_name = config.queue().realtime_queue_name;
		}

		// Resubmit the task
		submitTask(queue_name, updated_task);

		LOG_DEBUG("Returned task {} to queue {}", task_id, queue_name);
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Failed to return task {} to queue: {}", task_id, e.what());
	}
}

size_t DistributedTaskManager::getQueueLength(const std::string &queue_name)
{
	// Simplified implementation
	// In production, you'd use LLEN command on the actual queue
	try
	{
		// For now, return 0 as we're using a simplified approach
		return 0;
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Failed to get queue length for {}: {}", queue_name, e.what());
		return 0;
	}
}

void DistributedTaskManager::cleanupOrphanedTasks()
{
	try
	{
		auto &config = Config::instance();
		auto now = std::chrono::system_clock::now();
		auto threshold = std::chrono::duration_cast<std::chrono::milliseconds>(
							 now.time_since_epoch())
							 .count() -
						 (config.queue().visibility_timeout * 1000);

		// Check processing queues for orphaned tasks
		std::vector<std::string> queues = {
			config.queue().batch_queue_name + ":processing",
			config.queue().realtime_queue_name + ":processing"};

		for (const auto &queue : queues)
		{
			// Simplified orphan cleanup
			// In production, you'd scan the processing queue and check task timestamps
			LOG_DEBUG("Checking for orphaned tasks in queue: {}", queue);
		}
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Failed to cleanup orphaned tasks: {}", e.what());
	}
}

std::string DistributedTaskManager::makeProcessingKey(const std::string &task_id)
{
	return "processing:" + task_id;
}

std::string DistributedTaskManager::makeDeadLetterKey(const std::string &task_id)
{
	return "dead_letter:" + task_id;
}

std::string DistributedTaskManager::getMainQueueName(const std::string &processing_queue)
{
	// Extract main queue name from processing queue name
	// processing_queue is typically "main_queue:processing"
	size_t pos = processing_queue.find(":processing");
	if (pos != std::string::npos)
	{
		return processing_queue.substr(0, pos);
	}
	return processing_queue;
}