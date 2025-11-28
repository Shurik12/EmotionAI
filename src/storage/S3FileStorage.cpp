#include "S3FileStorage.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <ctime>
#include "logging/Logger.h"

S3FileStorage::S3FileStorage(const std::string &endpoint,
							 const std::string &access_key,
							 const std::string &secret_key,
							 const std::string &bucket,
							 bool use_ssl,
							 const std::string &region)
	: bucket_name_(bucket), region_(region)
{
	try
	{
		// Create S3 base URL
		minio::s3::BaseUrl base_url(endpoint);
		base_url.https = use_ssl;

		// Create credential provider
		minio::creds::StaticProvider provider(access_key, secret_key);

		// Create S3 client
		client_ = std::make_unique<minio::s3::Client>(base_url, &provider);

		// Check if bucket exists, create if not
		minio::s3::BucketExistsArgs bucket_args;
		bucket_args.bucket = bucket_name_;
		minio::s3::BucketExistsResponse bucket_resp = client_->BucketExists(bucket_args);

		if (!bucket_resp)
		{
			LOG_ERROR("Failed to check bucket existence: {}", bucket_resp.Error().String());
			throw std::runtime_error("Failed to check bucket existence");
		}

		if (!bucket_resp.exist)
		{
			minio::s3::MakeBucketArgs make_args;
			make_args.bucket = bucket_name_;
			minio::s3::MakeBucketResponse make_resp = client_->MakeBucket(make_args);
			if (!make_resp)
			{
				LOG_ERROR("Failed to create bucket: {}", make_resp.Error().String());
				throw std::runtime_error("Failed to create bucket");
			}
			LOG_INFO("Created S3 bucket: {}", bucket_name_);
		}

		LOG_INFO("S3FileStorage initialized for bucket: {} in region: {}", bucket_name_, region_);
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Failed to initialize S3FileStorage: {}", e.what());
		throw;
	}
}

bool S3FileStorage::saveFile(const std::string &file_content, const std::string &file_path)
{
	return saveFile(std::vector<uint8_t>(file_content.begin(), file_content.end()), file_path);
}

bool S3FileStorage::saveFile(const std::vector<uint8_t> &file_content, const std::string &file_path)
{
	try
	{
		std::string content_str(file_content.begin(), file_content.end());
		std::stringstream ss(content_str);

		minio::s3::PutObjectArgs args(ss, static_cast<long>(file_content.size()), 0);
		args.bucket = bucket_name_;
		args.object = file_path;

		minio::s3::PutObjectResponse response = client_->PutObject(args);
		if (!response)
		{
			LOG_ERROR("Failed to save file to S3: {} - {}", file_path, response.Error().String());
			return false;
		}

		LOG_DEBUG("File saved to S3: {}", file_path);
		return true;
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Error saving file to S3 {}: {}", file_path, e.what());
		return false;
	}
}

std::string S3FileStorage::readFile(const std::string &file_path)
{
	std::vector<uint8_t> binary_content = readFileBinary(file_path);
	return std::string(binary_content.begin(), binary_content.end());
}

std::vector<uint8_t> S3FileStorage::readFileBinary(const std::string &file_path)
{
	std::vector<uint8_t> file_content;
	try
	{
		minio::s3::GetObjectArgs args;
		args.bucket = bucket_name_;
		args.object = file_path;

		std::string content;
		args.datafunc = [&content](minio::http::DataFunctionArgs args) -> bool
		{
			content += args.datachunk;
			return true;
		};

		minio::s3::GetObjectResponse response = client_->GetObject(args);
		if (!response)
		{
			LOG_ERROR("Failed to load file from S3: {} - {}", file_path, response.Error().String());
			return file_content;
		}

		file_content.assign(content.begin(), content.end());
		LOG_DEBUG("File loaded from S3: {}", file_path);
		return file_content;
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Error loading file from S3 {}: {}", file_path, e.what());
		return file_content;
	}
}

bool S3FileStorage::fileExists(const std::string &file_path)
{
	try
	{
		minio::s3::StatObjectArgs args;
		args.bucket = bucket_name_;
		args.object = file_path;

		minio::s3::StatObjectResponse response = client_->StatObject(args);
		if (!response)
		{
			return false;
		}
		return true;
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Error checking file existence in S3 {}: {}", file_path, e.what());
		return false;
	}
}

bool S3FileStorage::deleteFile(const std::string &file_path)
{
	try
	{
		minio::s3::RemoveObjectArgs args;
		args.bucket = bucket_name_;
		args.object = file_path;

		minio::s3::RemoveObjectResponse response = client_->RemoveObject(args);
		if (!response)
		{
			LOG_ERROR("Failed to delete file from S3: {} - {}", file_path, response.Error().String());
			return false;
		}

		LOG_DEBUG("File deleted from S3: {}", file_path);
		return true;
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Error deleting file from S3 {}: {}", file_path, e.what());
		return false;
	}
}

bool S3FileStorage::createDirectory(const std::string &dir_path)
{
	// In S3, directories don't really exist - they're just prefixes
	// We can create a marker object to represent the directory
	try
	{
		std::string dir_key = dir_path;
		if (!dir_key.empty() && dir_key.back() != '/')
		{
			dir_key += '/';
		}

		// Create an empty object with the directory path as key
		std::stringstream ss;
		minio::s3::PutObjectArgs args(ss, 0, 0);
		args.bucket = bucket_name_;
		args.object = dir_key;

		minio::s3::PutObjectResponse response = client_->PutObject(args);
		if (!response)
		{
			LOG_ERROR("Failed to create directory in S3: {} - {}", dir_path, response.Error().String());
			return false;
		}

		LOG_DEBUG("Directory created in S3: {}", dir_path);
		return true;
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Error creating directory in S3 {}: {}", dir_path, e.what());
		return false;
	}
}

bool S3FileStorage::deleteDirectory(const std::string &dir_path)
{
	try
	{
		std::string prefix = dir_path;
		if (!prefix.empty() && prefix.back() != '/')
		{
			prefix += '/';
		}

		// List all objects with the prefix
		minio::s3::ListObjectsArgs list_args;
		list_args.bucket = bucket_name_;
		list_args.prefix = prefix;
		list_args.recursive = true;

		auto list_result = client_->ListObjects(list_args);

		// Delete each object individually
		for (; list_result; list_result++)
		{
			minio::s3::Item item = *list_result;
			if (!item)
			{
				LOG_ERROR("Error listing objects: {}", item.Error().String());
				continue;
			}

			minio::s3::RemoveObjectArgs remove_args;
			remove_args.bucket = bucket_name_;
			remove_args.object = item.name;

			auto remove_response = client_->RemoveObject(remove_args);
			if (!remove_response)
			{
				LOG_ERROR("Failed to delete object: {} - {}", item.name, remove_response.Error().String());
			}
		}

		LOG_DEBUG("Directory deleted from S3: {}", dir_path);
		return true;
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Error deleting directory from S3 {}: {}", dir_path, e.what());
		return false;
	}
}

std::vector<std::string> S3FileStorage::listFiles(const std::string &dir_path)
{
	std::vector<std::string> files;
	try
	{
		minio::s3::ListObjectsArgs args;
		args.bucket = bucket_name_;
		args.prefix = dir_path;
		args.recursive = false;

		auto result = client_->ListObjects(args);

		for (; result; result++)
		{
			minio::s3::Item item = *result;
			if (!item)
			{
				LOG_ERROR("Error listing objects: {}", item.Error().String());
				continue;
			}
			// Skip directory markers (objects ending with /)
			if (!item.name.empty() && item.name.back() != '/')
			{
				files.push_back(item.name);
			}
		}

		return files;
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Error listing files in S3 directory {}: {}", dir_path, e.what());
		return {};
	}
}

size_t S3FileStorage::getFileSize(const std::string &file_path)
{
	try
	{
		minio::s3::StatObjectArgs args;
		args.bucket = bucket_name_;
		args.object = file_path;

		minio::s3::StatObjectResponse response = client_->StatObject(args);
		if (!response)
		{
			LOG_ERROR("Failed to get file size from S3: {} - {}", file_path, response.Error().String());
			return 0;
		}

		return response.size;
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Error getting file size from S3 {}: {}", file_path, e.what());
		return 0;
	}
}

std::string S3FileStorage::getFileHash(const std::string &file_path)
{
	try
	{
		minio::s3::StatObjectArgs args;
		args.bucket = bucket_name_;
		args.object = file_path;

		minio::s3::StatObjectResponse response = client_->StatObject(args);
		if (!response)
		{
			LOG_ERROR("Failed to get file hash from S3: {} - {}", file_path, response.Error().String());
			return "";
		}

		// Use ETag as hash if available, otherwise use size
		if (!response.etag.empty())
		{
			return response.etag;
		}

		return std::to_string(response.size);
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Error getting file hash from S3 {}: {}", file_path, e.what());
		return "";
	}
}

std::string S3FileStorage::getFileUrl(const std::string &file_path)
{
	// For S3 storage, return a direct S3 URL
	// In a real implementation, you might want to generate pre-signed URLs
	return "https://" + bucket_name_ + ".s3." + region_ + ".amazonaws.com/" + file_path;
}

bool S3FileStorage::copyFile(const std::string &source_path, const std::string &destination_path)
{
	try
	{
		minio::s3::CopySource source;
		source.bucket = bucket_name_;
		source.object = source_path;

		minio::s3::CopyObjectArgs args;
		args.bucket = bucket_name_;
		args.object = destination_path;
		args.source = source;

		minio::s3::CopyObjectResponse response = client_->CopyObject(args);
		if (!response)
		{
			LOG_ERROR("Failed to copy file in S3: {} -> {} - {}", source_path, destination_path, response.Error().String());
			return false;
		}

		LOG_DEBUG("File copied in S3: {} -> {}", source_path, destination_path);
		return true;
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Error copying file in S3 {} -> {}: {}", source_path, destination_path, e.what());
		return false;
	}
}

bool S3FileStorage::moveFile(const std::string &source_path, const std::string &destination_path)
{
	try
	{
		// Copy the file first
		if (!copyFile(source_path, destination_path))
		{
			return false;
		}

		// Then delete the original
		if (!deleteFile(source_path))
		{
			LOG_ERROR("Failed to delete source file after copy in S3: {}", source_path);
			// Try to clean up the copied file
			deleteFile(destination_path);
			return false;
		}

		LOG_DEBUG("File moved in S3: {} -> {}", source_path, destination_path);
		return true;
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Error moving file in S3 {} -> {}: {}", source_path, destination_path, e.what());
		return false;
	}
}

nlohmann::json S3FileStorage::getStorageInfo()
{
	nlohmann::json info;
	try
	{
		info["type"] = "s3";
		info["bucket"] = bucket_name_;
		info["region"] = region_;

		// Get bucket stats by listing objects
		minio::s3::ListObjectsArgs args;
		args.bucket = bucket_name_;
		args.recursive = true;

		uint64_t total_size = 0;
		uint64_t file_count = 0;

		auto result = client_->ListObjects(args);

		for (; result; result++)
		{
			minio::s3::Item item = *result;
			if (!item)
			{
				continue;
			}
			if (!item.name.empty() && item.name.back() != '/')
			{
				total_size += item.size;
				file_count++;
			}
		}

		info["total_files"] = file_count;
		info["total_size"] = total_size;
		info["bucket_name"] = bucket_name_;
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Error getting S3 storage info: {}", e.what());
		info["error"] = e.what();
	}

	return info;
}