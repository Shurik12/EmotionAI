#include "FileStorageFactory.h"
#include <logging/Logger.h>

std::unique_ptr<FileStorage> FileStorageFactory::createStorage(const std::string &type,
															   const nlohmann::json &config)
{
	try
	{
		if (type == "local")
		{
			std::string base_path = config.value("base_path", "");
			return std::make_unique<LocalFileStorage>(base_path);
		}
		else if (type == "s3")
		{
			std::string endpoint = config.value("endpoint", "localhost:9000");
			std::string access_key = config.value("access_key", "minioadmin");
			std::string secret_key = config.value("secret_key", "minioadmin");
			std::string bucket = config.value("bucket", "emotionai");
			bool use_ssl = config.value("use_ssl", false);
			std::string region = config.value("region", "us-east-1");

			return std::make_unique<S3FileStorage>(endpoint, access_key, secret_key, bucket, use_ssl, region);
		}
		else if (type == "nfs")
		{
			std::string mount_point = config.value("mount_point", "");
			return std::make_unique<NFSFileStorage>(mount_point);
		}
		else
		{
			LOG_ERROR("Unknown storage type: {}", type);
			throw std::runtime_error("Unknown storage type: " + type);
		}
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Failed to create storage type {}: {}", type, e.what());
		throw;
	}
}

std::shared_ptr<FileStorage> FileStorageFactory::createStorageFromConfig()
{
	auto &config = Config::instance();
	if (!config.isLoaded())
	{
		LOG_ERROR("Configuration not loaded");
		return nullptr;
	}

	auto storage_config = config.storage();

	if (storage_config.type == "s3")
	{
		LOG_INFO("Creating S3 storage backend");
		return std::make_shared<S3FileStorage>(
			storage_config.s3_endpoint,
			storage_config.s3_access_key,
			storage_config.s3_secret_key,
			storage_config.s3_bucket,
			storage_config.s3_use_ssl,
			storage_config.s3_region);
	}
	else if (storage_config.type == "nfs")
	{
		LOG_INFO("Creating NFS storage backend");
		return std::make_shared<NFSFileStorage>(storage_config.base_path);
	}
	else
	{
		LOG_INFO("Creating local storage backend");
		return std::make_shared<LocalFileStorage>(storage_config.base_path);
	}
}

std::map<std::string, std::string> FileStorageFactory::getAvailableStorageTypes()
{
	return {
		{"local", "Local filesystem storage"},
		{"s3", "Amazon S3 or S3-compatible storage"},
		{"nfs", "Network File System storage"}};
}