#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <memory>

class ThreadPool
{
public:
	explicit ThreadPool(size_t num_threads);
	~ThreadPool();

	template <class F>
	void enqueue(F &&task);

	size_t get_pending_tasks() const;
	size_t get_active_threads() const;

private:
	std::vector<std::thread> workers_;
	std::queue<std::function<void()>> tasks_;
	mutable std::mutex queue_mutex_;
	std::condition_variable condition_;
	std::atomic<bool> stop_{false};
	std::atomic<size_t> active_threads_{0};
};

template <class F>
void ThreadPool::enqueue(F &&task)
{
	{
		std::unique_lock lock(queue_mutex_);
		if (stop_)
		{
			throw std::runtime_error("enqueue on stopped ThreadPool");
		}
		tasks_.emplace(std::forward<F>(task));
	}
	condition_.notify_one();
}