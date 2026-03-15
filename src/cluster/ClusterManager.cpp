#include "ClusterManager.h"
#include <chrono>
#include <random>

ClusterManager::ClusterManager(std::shared_ptr<DragonflyManager> dragonfly)
	: dragonfly_(dragonfly)
{
	auto &config = Config::instance();
	cluster_name_ = config.cluster().name;

	if (config.cluster().instance_id == "auto")
	{
		instance_id_ = DragonflyManager::generate_uuid();
	}
	else
	{
		instance_id_ = config.cluster().instance_id;
	}
}

ClusterManager::~ClusterManager()
{
	stop();
}

void ClusterManager::initialize()
{
	LOG_INFO("Initializing ClusterManager for instance: {}", instance_id_);

	// Register this instance
	auto &config = Config::instance();
	nlohmann::json instance_info = {
		{"instance_id", instance_id_},
		{"cluster_name", cluster_name_},
		{"startup_time", std::chrono::duration_cast<std::chrono::milliseconds>(
							 std::chrono::system_clock::now().time_since_epoch())
							 .count()},
		{"last_heartbeat", std::chrono::duration_cast<std::chrono::milliseconds>(
							   std::chrono::system_clock::now().time_since_epoch())
							   .count()},
		{"status", "active"}};

	std::string instance_key = "cluster:" + cluster_name_ + ":instances:" + instance_id_;
	dragonfly_->set_task_status(instance_key, instance_info);

	LOG_INFO("ClusterManager initialized for instance: {}", instance_id_);
}

void ClusterManager::start()
{
	if (running_.load())
	{
		return;
	}

	running_.store(true);
	heartbeat_thread_ = std::thread(&ClusterManager::heartbeatLoop, this);

	LOG_INFO("ClusterManager started for instance: {}", instance_id_);
}

void ClusterManager::stop()
{
	if (!running_.load())
	{
		return;
	}

	running_.store(false);
	if (heartbeat_thread_.joinable())
	{
		heartbeat_thread_.join();
	}

	// Clean up instance registration
	if (dragonfly_)
	{
		std::string instance_key = "cluster:" + cluster_name_ + ":instances:" + instance_id_;
		// The key will expire automatically based on task_expiration
	}

	LOG_INFO("ClusterManager stopped for instance: {}", instance_id_);
}

void ClusterManager::heartbeatLoop()
{
	auto &config = Config::instance();
	int heartbeat_interval = config.cluster().heartbeat_interval;

	while (running_.load())
	{
		try
		{
			// Send heartbeat
			std::string instance_key = "cluster:" + cluster_name_ + ":instances:" + instance_id_;
			nlohmann::json instance_info = {
				{"instance_id", instance_id_},
				{"cluster_name", cluster_name_},
				{"last_heartbeat", std::chrono::duration_cast<std::chrono::milliseconds>(
									   std::chrono::system_clock::now().time_since_epoch())
									   .count()},
				{"status", "active"},
				{"is_leader", is_leader_.load()}};

			dragonfly_->set_task_status(instance_key, instance_info);

			// Run leadership election periodically
			static int election_counter = 0;
			if (++election_counter >= (config.cluster().leader_election_interval / heartbeat_interval))
			{
				leadershipElection();
				election_counter = 0;
			}

			// Clean up stale instances
			cleanupStaleInstances();
		}
		catch (const std::exception &e)
		{
			LOG_ERROR("Error in cluster heartbeat: {}", e.what());
		}

		// Sleep until next heartbeat
		std::this_thread::sleep_for(std::chrono::seconds(heartbeat_interval));
	}
}

void ClusterManager::leadershipElection()
{
	try
	{
		auto members = getClusterMembers();
		if (members.empty())
		{
			return;
		}

		// Simple leadership election: instance with lowest ID becomes leader
		std::string potential_leader = members[0]["instance_id"];
		for (const auto &member : members)
		{
			if (member["instance_id"] < potential_leader)
			{
				potential_leader = member["instance_id"];
			}
		}

		bool was_leader = is_leader_.load();
		is_leader_.store(potential_leader == instance_id_);

		if (is_leader_.load() && !was_leader)
		{
			LOG_INFO("Instance {} elected as cluster leader", instance_id_);
		}
		else if (!is_leader_.load() && was_leader)
		{
			LOG_INFO("Instance {} is no longer cluster leader", instance_id_);
		}
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Error in leadership election: {}", e.what());
	}
}

void ClusterManager::cleanupStaleInstances()
{
	try
	{
		auto &config = Config::instance();
		auto members = getClusterMembers();
		auto now = std::chrono::system_clock::now();
		auto timeout_threshold = std::chrono::duration_cast<std::chrono::milliseconds>(
									 now.time_since_epoch())
									 .count() -
								 (config.cluster().instance_timeout * 1000);

		for (const auto &member : members)
		{
			auto last_heartbeat = member["last_heartbeat"].get<long>();
			if (last_heartbeat < timeout_threshold)
			{
				LOG_WARN("Instance {} appears to be stale, last heartbeat: {}",
						 member["instance_id"].get<std::string>(), last_heartbeat);
				// In production, you might want to remove stale instances
			}
		}
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Error cleaning up stale instances: {}", e.what());
	}
}

std::vector<nlohmann::json> ClusterManager::getClusterMembers()
{
	std::vector<nlohmann::json> members;

	try
	{
		// This is a simplified implementation
		// In production, you'd use SCAN or other methods to get all instances
		std::string instance_key = "cluster:" + cluster_name_ + ":instances:" + instance_id_;
		auto instance_info = dragonfly_->get_task_status_json(instance_key);

		if (instance_info)
		{
			members.push_back(*instance_info);
		}

		// For demo purposes, we only return the current instance
		// In a real cluster, you'd return all active instances
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Error getting cluster members: {}", e.what());
	}

	return members;
}

bool ClusterManager::acquireLock(const std::string &lock_name, int timeout_seconds)
{
	try
	{
		std::string lock_key = "cluster:" + cluster_name_ + ":locks:" + lock_name;

		// Simplified lock implementation using task status
		// In production, use proper Redis/Dragonfly distributed locks
		auto current_lock = dragonfly_->get_task_status(lock_key);

		if (!current_lock)
		{
			// Lock is available, acquire it
			nlohmann::json lock_info = {
				{"owner", instance_id_},
				{"acquired_at", std::chrono::duration_cast<std::chrono::milliseconds>(
									std::chrono::system_clock::now().time_since_epoch())
									.count()},
				{"timeout", timeout_seconds}};

			dragonfly_->set_task_status(lock_key, lock_info);
			locks_[lock_name] = std::chrono::steady_clock::now();
			return true;
		}

		return false;
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Error acquiring lock {}: {}", lock_name, e.what());
		return false;
	}
}

void ClusterManager::releaseLock(const std::string &lock_name)
{
	try
	{
		std::string lock_key = "cluster:" + cluster_name_ + ":locks:" + lock_name;

		// Check if we own the lock
		auto current_lock = dragonfly_->get_task_status_json(lock_key);
		if (current_lock && (*current_lock)["owner"] == instance_id_)
		{
			// Release the lock by letting it expire naturally
			locks_.erase(lock_name);
		}
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Error releasing lock {}: {}", lock_name, e.what());
	}
}