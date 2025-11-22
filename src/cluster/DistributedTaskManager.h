#pragma once

#include <string>
#include <optional>
#include <nlohmann/json.hpp>
#include <db/DragonflyManager.h>
#include <config/Config.h>
#include <logging/Logger.h>

class DistributedTaskManager
{
public:
	DistributedTaskManager(std::shared_ptr<DragonflyManager> dragonfly,
						   const std::string &instance_id);

	// Task submission
	bool submitTask(const std::string &queue_name, const nlohmann::json &task);

	// Task consumption
	std::optional<nlohmann::json> getNextTask(const std::string &queue_name,
											  int visibility_timeout = 300);

	// Task lifecycle
	void markTaskComplete(const std::string &task_id);
	void markTaskFailed(const std::string &task_id, const std::string &error);
	void returnTaskToQueue(const std::string &task_id, const nlohmann::json &task);

	// Queue management
	size_t getQueueLength(const std::string &queue_name);
	void cleanupOrphanedTasks();

private:
	std::string makeProcessingKey(const std::string &task_id);
	std::string makeDeadLetterKey(const std::string &task_id);
	std::string getMainQueueName(const std::string &processing_queue);

	std::shared_ptr<DragonflyManager> dragonfly_;
	std::string instance_id_;
};