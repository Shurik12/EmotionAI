#pragma once

#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <functional>
#include <nlohmann/json.hpp>

class TaskBatcher
{
public:
	using BatchCallback = std::function<void(const std::vector<std::pair<std::string, nlohmann::json>> &)>;

	TaskBatcher(size_t batch_size = 50, std::chrono::milliseconds batch_timeout = std::chrono::milliseconds(100));
	~TaskBatcher();

	void enqueue(const std::string &task_id, const nlohmann::json &status);
	void set_batch_callback(BatchCallback callback);

	void start();
	void stop();

private:
	void process_batches();

	struct BatchItem
	{
		std::string task_id;
		nlohmann::json status;
	};

	std::queue<BatchItem> batch_queue_;
	mutable std::mutex queue_mutex_;
	std::condition_variable queue_cv_;

	BatchCallback batch_callback_;
	std::atomic<bool> running_{false};
	std::thread batch_thread_;

	size_t batch_size_;
	std::chrono::milliseconds batch_timeout_;
};