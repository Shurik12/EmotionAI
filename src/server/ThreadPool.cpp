#include "ThreadPool.h"
#include <logging/Logger.h>

ThreadPool::ThreadPool(size_t num_threads)
{
	workers_.reserve(num_threads);
	for (size_t i = 0; i < num_threads; ++i)
	{
		workers_.emplace_back([this]
							  {
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock lock(queue_mutex_);
                    condition_.wait(lock, [this] {
                        return stop_ || !tasks_.empty();
                    });
                    
                    if (stop_ && tasks_.empty()) {
                        return;
                    }
                    
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                
                active_threads_++;
                try {
                    task();
                } catch (const std::exception& e) {
                    LOG_ERROR("Exception in thread pool task: {}", e.what());
                }
                active_threads_--;
            } });
	}
	LOG_INFO("ThreadPool started with {} threads", num_threads);
}

ThreadPool::~ThreadPool()
{
	{
		std::unique_lock lock(queue_mutex_);
		stop_ = true;
	}
	condition_.notify_all();
	for (std::thread &worker : workers_)
	{
		if (worker.joinable())
		{
			worker.join();
		}
	}
}

size_t ThreadPool::get_pending_tasks() const
{
	std::unique_lock lock(queue_mutex_);
	return tasks_.size();
}

size_t ThreadPool::get_active_threads() const
{
	return active_threads_;
}