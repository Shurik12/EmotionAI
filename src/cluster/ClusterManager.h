#pragma once

#include <string>
#include <atomic>
#include <thread>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include <db/DragonflyManager.h>

class ClusterManager
{
public:
	ClusterManager(std::shared_ptr<DragonflyManager> dragonfly);
	~ClusterManager();

	void initialize();
	void start();
	void stop();

	bool isLeader() const { return is_leader_; }
	std::string getInstanceId() const { return instance_id_; }
	std::vector<nlohmann::json> getClusterMembers();

	// Distributed locking
	bool acquireLock(const std::string &lock_name, int timeout_seconds = 30);
	void releaseLock(const std::string &lock_name);

private:
	void heartbeatLoop();
	void leadershipElection();
	void cleanupStaleInstances();

	std::shared_ptr<DragonflyManager> dragonfly_;
	std::string instance_id_;
	std::string cluster_name_;
	std::atomic<bool> running_{false};
	std::atomic<bool> is_leader_{false};
	std::thread heartbeat_thread_;
	std::unordered_map<std::string, std::chrono::steady_clock::time_point> locks_;
};