#include "NFSFileStorage.h"
#include <sys/statvfs.h>
#include <logging/Logger.h>

NFSFileStorage::NFSFileStorage(const std::string &mount_point)
	: LocalFileStorage(mount_point), mount_point_(mount_point)
{

	if (!testNFSConnection())
	{
		LOG_WARN("NFS mount point may not be accessible: {}", mount_point);
	}

	LOG_INFO("NFSFileStorage initialized with mount point: {}", mount_point);
}

bool NFSFileStorage::testNFSConnection()
{
	try
	{
		struct statvfs stat;
		if (statvfs(mount_point_.c_str(), &stat) == 0)
		{
			LOG_DEBUG("NFS connection test successful for: {}", mount_point_);
			return true;
		}
		else
		{
			LOG_ERROR("NFS connection test failed for: {}", mount_point_);
			return false;
		}
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Error testing NFS connection for {}: {}", mount_point_, e.what());
		return false;
	}
}

nlohmann::json NFSFileStorage::getStorageInfo()
{
	auto info = LocalFileStorage::getStorageInfo();
	info["type"] = "nfs";
	info["mount_point"] = mount_point_;

	try
	{
		struct statvfs stat;
		if (statvfs(mount_point_.c_str(), &stat) == 0)
		{
			uint64_t total_bytes = stat.f_blocks * stat.f_frsize;
			uint64_t free_bytes = stat.f_bfree * stat.f_frsize;
			uint64_t available_bytes = stat.f_bavail * stat.f_frsize;

			info["total_bytes"] = total_bytes;
			info["free_bytes"] = free_bytes;
			info["available_bytes"] = available_bytes;
			info["used_bytes"] = total_bytes - free_bytes;

			info["total_gb"] = total_bytes / (1024.0 * 1024.0 * 1024.0);
			info["free_gb"] = free_bytes / (1024.0 * 1024.0 * 1024.0);
			info["available_gb"] = available_bytes / (1024.0 * 1024.0 * 1024.0);
			info["used_gb"] = (total_bytes - free_bytes) / (1024.0 * 1024.0 * 1024.0);

			info["usage_percentage"] = ((total_bytes - free_bytes) * 100.0) / total_bytes;
		}
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Error getting NFS storage info: {}", e.what());
		info["error"] = e.what();
	}

	return info;
}