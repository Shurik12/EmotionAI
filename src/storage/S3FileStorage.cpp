#include "S3FileStorage.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <ctime>
#include <algorithm>
#include "logging/Logger.h"

S3FileStorage::S3FileStorage(const Config &config)
	: state_(std::make_unique<InternalState>())
{
	state_->bucket_name = config.bucket;
	state_->region = config.region;
	state_->endpoint = config.endpoint;
	state_->use_ssl = config.use_ssl;

	LOG_INFO("Initializing S3 storage with endpoint: {}, bucket: {}",
			 config.endpoint, config.bucket);

	try
	{
		// Create base URL first
		minio::s3::BaseUrl base_url(state_->endpoint);
		base_url.https = state_->use_ssl;

		// Create credential provider
		state_->credential_provider = std::make_shared<minio::creds::StaticProvider>(
			config.access_key, config.secret_key);

		// Create S3 client
		state_->client = std::make_unique<minio::s3::Client>(base_url,
															 state_->credential_provider.get());

		// Mark as connected BEFORE checking bucket existence
		state_->connected = true;

		if (config.create_bucket_if_not_exists)
		{
			// Use direct client access for initialization
			ensureBucketExists();
		}

		LOG_INFO("S3FileStorage initialized successfully for bucket: {} in region: {}",
				 state_->bucket_name, state_->region);
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Failed to initialize S3FileStorage: {}", e.what());
		state_->connected = false;
		throw;
	}
}

S3FileStorage::S3FileStorage(const std::string &endpoint,
							 const std::string &access_key,
							 const std::string &secret_key,
							 const std::string &bucket,
							 bool use_ssl,
							 const std::string &region)
	: S3FileStorage(Config{
		  .endpoint = endpoint,
		  .access_key = access_key,
		  .secret_key = secret_key,
		  .bucket = bucket,
		  .use_ssl = use_ssl,
		  .region = region})
{
}

S3FileStorage::~S3FileStorage()
{
	std::lock_guard<std::mutex> lock(mutex_);
	state_->connected = false;
	LOG_DEBUG("S3FileStorage destroyed");
}

// Direct client access for initialization
minio::s3::Client &S3FileStorage::getClientDirect()
{
	if (!state_->client)
	{
		throw std::runtime_error("S3 client not initialized");
	}
	return *state_->client;
}

bool S3FileStorage::ensureBucketExists()
{
	// Don't use withClient here - we're in constructor
	minio::s3::Client &client = getClientDirect();

	minio::s3::BucketExistsArgs bucket_args;
	bucket_args.bucket = state_->bucket_name;

	LOG_DEBUG("Checking if bucket exists: {}", state_->bucket_name);
	minio::s3::BucketExistsResponse bucket_resp = client.BucketExists(bucket_args);

	if (!bucket_resp)
	{
		LOG_ERROR("Failed to check bucket existence: {}", bucket_resp.Error().String());
		throw std::runtime_error("Failed to check bucket existence: " +
								 bucket_resp.Error().String());
	}

	if (!bucket_resp.exist)
	{
		LOG_INFO("Bucket does not exist, creating: {}", state_->bucket_name);
		minio::s3::MakeBucketArgs make_args;
		make_args.bucket = state_->bucket_name;

		if (!state_->region.empty())
		{
			make_args.region = state_->region;
		}

		minio::s3::MakeBucketResponse make_resp = client.MakeBucket(make_args);

		if (!make_resp)
		{
			LOG_ERROR("Failed to create bucket: {}", make_resp.Error().String());
			throw std::runtime_error("Failed to create bucket: " +
									 make_resp.Error().String());
		}
		LOG_INFO("Created S3 bucket: {}", state_->bucket_name);
		return false; // Bucket was created
	}

	LOG_INFO("Bucket already exists: {}", state_->bucket_name);
	return true; // Bucket existed
}

bool S3FileStorage::reconnect()
{
	std::lock_guard<std::mutex> lock(mutex_);
	try
	{
		minio::s3::BaseUrl base_url(state_->endpoint);
		base_url.https = state_->use_ssl;

		// Recreate credential provider (in case credentials changed)
		// Note: You might want to store credentials in the config
		throw std::runtime_error("Reconnect not implemented - requires credentials");

		state_->client = std::make_unique<minio::s3::Client>(base_url,
															 state_->credential_provider.get());
		state_->connected = true;
		LOG_INFO("S3 storage reconnected successfully");
		return true;
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Failed to reconnect S3 storage: {}", e.what());
		state_->connected = false;
		return false;
	}
}

template <typename Func>
auto S3FileStorage::withClient(Func func) -> decltype(func(std::declval<minio::s3::Client &>()))
{
	std::lock_guard<std::mutex> lock(mutex_);

	if (!state_->connected || !state_->client)
	{
		LOG_ERROR("S3 client not initialized or disconnected");
		throw std::runtime_error("S3 client not initialized or disconnected");
	}

	try
	{
		return func(*state_->client);
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("S3 operation failed: {}", e.what());
		throw;
	}
}

template <typename Func>
auto S3FileStorage::withClient(Func func) const -> decltype(func(std::declval<const minio::s3::Client &>()))
{
	std::lock_guard<std::mutex> lock(mutex_);

	if (!state_->connected || !state_->client)
	{
		LOG_ERROR("S3 client not initialized or disconnected");
		throw std::runtime_error("S3 client not initialized or disconnected");
	}

	try
	{
		return func(*state_->client);
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("S3 operation failed: {}", e.what());
		throw;
	}
}

bool S3FileStorage::saveFile(const std::string &file_content, const std::string &file_path)
{
	return saveFile(std::vector<uint8_t>(file_content.begin(), file_content.end()), file_path);
}

bool S3FileStorage::saveFile(const std::vector<uint8_t> &file_content, const std::string &file_path)
{
	return withClient([&](minio::s3::Client &client)
					  {
        std::string content_str(file_content.begin(), file_content.end());
        std::stringstream ss(std::move(content_str));

        minio::s3::PutObjectArgs args(ss, static_cast<long>(file_content.size()), 0);
        args.bucket = state_->bucket_name;
        args.object = file_path;

        minio::s3::PutObjectResponse response = client.PutObject(args);
        if (!response) {
            LOG_ERROR("Failed to save file to S3: {} - {}", file_path, response.Error().String());
            return false;
        }

        LOG_DEBUG("File saved to S3: {} ({} bytes)", file_path, file_content.size());
        return true; });
}

std::string S3FileStorage::readFile(const std::string &file_path)
{
	std::vector<uint8_t> binary_content = readFileBinary(file_path);
	return std::string(binary_content.begin(), binary_content.end());
}

std::vector<uint8_t> S3FileStorage::readFileBinary(const std::string &file_path)
{
	return withClient([&](minio::s3::Client &client)
					  {
        minio::s3::GetObjectArgs args;
        args.bucket = state_->bucket_name;
        args.object = file_path;

        std::vector<uint8_t> content;
        args.datafunc = [&content](minio::http::DataFunctionArgs data_args) -> bool {
            content.insert(content.end(), 
                          data_args.datachunk.begin(), 
                          data_args.datachunk.end());
            return true;
        };

        minio::s3::GetObjectResponse response = client.GetObject(args);
        if (!response) {
            LOG_ERROR("Failed to load file from S3: {} - {}", file_path, response.Error().String());
            throw std::runtime_error("Failed to load file: " + response.Error().String());
        }

        LOG_DEBUG("File loaded from S3: {} ({} bytes)", file_path, content.size());
        return content; });
}

bool S3FileStorage::fileExists(const std::string &file_path)
{
	try
	{
		return withClient([&](minio::s3::Client &client)
						  {
            minio::s3::StatObjectArgs args;
            args.bucket = state_->bucket_name;
            args.object = file_path;

            minio::s3::StatObjectResponse response = client.StatObject(args);
            return response && response.size > 0; });
	}
	catch (const std::exception &e)
	{
		LOG_DEBUG("File does not exist in S3: {} - {}", file_path, e.what());
		return false;
	}
}

bool S3FileStorage::deleteFile(const std::string &file_path)
{
	return withClient([&](minio::s3::Client &client)
					  {
        minio::s3::RemoveObjectArgs args;
        args.bucket = state_->bucket_name;
        args.object = file_path;

        minio::s3::RemoveObjectResponse response = client.RemoveObject(args);
        if (!response) {
            LOG_ERROR("Failed to delete file from S3: {} - {}", file_path, response.Error().String());
            return false;
        }

        LOG_DEBUG("File deleted from S3: {}", file_path);
        return true; });
}

bool S3FileStorage::createDirectory(const std::string &dir_path)
{
	return withClient([&](minio::s3::Client &client)
					  {
        std::string dir_key = dir_path;
        if (!dir_key.empty() && dir_key.back() != '/') {
            dir_key += '/';
        }

        // Create an empty object with the directory path as key
        std::stringstream ss;
        minio::s3::PutObjectArgs args(ss, 0, 0);
        args.bucket = state_->bucket_name;
        args.object = dir_key;

        minio::s3::PutObjectResponse response = client.PutObject(args);
        if (!response) {
            LOG_ERROR("Failed to create directory in S3: {} - {}", dir_path, response.Error().String());
            return false;
        }

        LOG_DEBUG("Directory created in S3: {}", dir_path);
        return true; });
}

bool S3FileStorage::deleteDirectory(const std::string &dir_path)
{
	return withClient([&](minio::s3::Client &client)
					  {
        std::string prefix = dir_path;
        if (!prefix.empty() && prefix.back() != '/') {
            prefix += '/';
        }

        // List all objects with the prefix
        minio::s3::ListObjectsArgs list_args;
        list_args.bucket = state_->bucket_name;
        list_args.prefix = prefix;
        list_args.recursive = true;

        std::vector<std::string> objects_to_delete;
        auto list_result = client.ListObjects(list_args);

        // Collect all objects first
        for (; list_result; list_result++) {
            minio::s3::Item item = *list_result;
            if (item) {
                objects_to_delete.push_back(item.name);
            }
        }

        // Delete objects in batches
        bool all_success = true;
        for (const auto& object_name : objects_to_delete) {
            minio::s3::RemoveObjectArgs remove_args;
            remove_args.bucket = state_->bucket_name;
            remove_args.object = object_name;

            auto remove_response = client.RemoveObject(remove_args);
            if (!remove_response) {
                LOG_ERROR("Failed to delete object: {} - {}", object_name, remove_response.Error().String());
                all_success = false;
            }
        }

        LOG_DEBUG("Directory deleted from S3: {} ({} objects)", dir_path, objects_to_delete.size());
        return all_success; });
}

std::vector<std::string> S3FileStorage::listFiles(const std::string &dir_path)
{
	return withClient([&](minio::s3::Client &client)
					  {
        minio::s3::ListObjectsArgs args;
        args.bucket = state_->bucket_name;
        args.prefix = dir_path;
        args.recursive = false;

        std::vector<std::string> files;
        auto result = client.ListObjects(args);

        for (; result; result++) {
            minio::s3::Item item = *result;
            if (item && !item.name.empty() && item.name.back() != '/') {
                files.push_back(item.name);
            }
        }

        LOG_DEBUG("Listed {} files from directory: {}", files.size(), dir_path);
        return files; });
}

size_t S3FileStorage::getFileSize(const std::string &file_path)
{
	return withClient([&](minio::s3::Client &client) -> size_t
					  {
        minio::s3::StatObjectArgs args;
        args.bucket = state_->bucket_name;
        args.object = file_path;

        minio::s3::StatObjectResponse response = client.StatObject(args);
        if (!response) {
            LOG_ERROR("Failed to get file size from S3: {} - {}", file_path, response.Error().String());
            return 0;
        }

        return response.size; });
}

std::string S3FileStorage::getFileHash(const std::string &file_path)
{
	return withClient([&](minio::s3::Client &client) -> std::string
					  {
        minio::s3::StatObjectArgs args;
        args.bucket = state_->bucket_name;
        args.object = file_path;

        minio::s3::StatObjectResponse response = client.StatObject(args);
        if (!response) {
            LOG_ERROR("Failed to get file hash from S3: {} - {}", file_path, response.Error().String());
            return "";
        }

        // Remove quotes from ETag if present
        std::string etag = response.etag;
        if (!etag.empty() && etag.front() == '"' && etag.back() == '"') {
            etag = etag.substr(1, etag.size() - 2);
        }

        return !etag.empty() ? etag : std::to_string(response.size); });
}

std::string S3FileStorage::getFileUrl(const std::string &file_path)
{
    return "/api/results/" + fs::path(file_path).filename().string();
}

std::string S3FileStorage::generatePresignedUrl(const std::string &file_path,
												std::chrono::seconds expiration)
{
	return withClient([&](minio::s3::Client &client) -> std::string
					  {
        minio::s3::GetPresignedObjectUrlArgs args;
        args.bucket = state_->bucket_name;
        args.object = file_path;
        args.expiry_seconds = static_cast<int>(expiration.count());

        auto url_result = client.GetPresignedObjectUrl(args);
        if (url_result.Error()) {
            LOG_ERROR("Failed to generate presigned URL: {} - {}", file_path, url_result.Error().String());
            return "";
        }

        return url_result.url; });
}

bool S3FileStorage::copyFile(const std::string &source_path, const std::string &destination_path)
{
	return withClient([&](minio::s3::Client &client)
					  {
        minio::s3::CopySource source;
        source.bucket = state_->bucket_name;
        source.object = source_path;

        minio::s3::CopyObjectArgs args;
        args.bucket = state_->bucket_name;
        args.object = destination_path;
        args.source = source;

        minio::s3::CopyObjectResponse response = client.CopyObject(args);
        if (!response) {
            LOG_ERROR("Failed to copy file in S3: {} -> {} - {}", 
                     source_path, destination_path, response.Error().String());
            return false;
        }

        LOG_DEBUG("File copied in S3: {} -> {}", source_path, destination_path);
        return true; });
}

bool S3FileStorage::moveFile(const std::string &source_path, const std::string &destination_path)
{
	if (!copyFile(source_path, destination_path))
	{
		return false;
	}

	if (!deleteFile(source_path))
	{
		LOG_ERROR("Failed to delete source file after copy in S3: {}", source_path);
		// Optional: Attempt to clean up the copied file
		deleteFile(destination_path);
		return false;
	}

	LOG_DEBUG("File moved in S3: {} -> {}", source_path, destination_path);
	return true;
}

bool S3FileStorage::copyFileBetweenBuckets(const std::string &source_path,
										   const std::string &dest_bucket,
										   const std::string &dest_path)
{
	return withClient([&](minio::s3::Client &client)
					  {
        minio::s3::CopySource source;
        source.bucket = state_->bucket_name;
        source.object = source_path;

        minio::s3::CopyObjectArgs args;
        args.bucket = dest_bucket;
        args.object = dest_path;
        args.source = source;

        minio::s3::CopyObjectResponse response = client.CopyObject(args);
        if (!response) {
            LOG_ERROR("Failed to copy file between buckets: {}:{} -> {}:{} - {}", 
                     state_->bucket_name, source_path, 
                     dest_bucket, dest_path, response.Error().String());
            return false;
        }

        LOG_DEBUG("File copied between buckets: {}:{} -> {}:{}", 
                 state_->bucket_name, source_path, dest_bucket, dest_path);
        return true; });
}

bool S3FileStorage::uploadFromStream(std::istream &stream,
									 const std::string &file_path,
									 const std::string &content_type)
{
	return withClient([&](minio::s3::Client &client)
					  {
        // Get stream size
        stream.seekg(0, std::ios::end);
        long size = stream.tellg();
        stream.seekg(0, std::ios::beg);

        minio::s3::PutObjectArgs args(stream, size, 0);
        args.bucket = state_->bucket_name;
        args.object = file_path;
        
        if (!content_type.empty()) {
            args.content_type = content_type;
        }

        minio::s3::PutObjectResponse response = client.PutObject(args);
        if (!response) {
            LOG_ERROR("Failed to upload from stream to S3: {} - {}", 
                     file_path, response.Error().String());
            return false;
        }

        LOG_DEBUG("Uploaded from stream to S3: {} ({} bytes)", file_path, size);
        return true; });
}

bool S3FileStorage::downloadToStream(const std::string &file_path, std::ostream &stream)
{
	return withClient([&](minio::s3::Client &client)
					  {
        minio::s3::GetObjectArgs args;
        args.bucket = state_->bucket_name;
        args.object = file_path;

        args.datafunc = [&stream](minio::http::DataFunctionArgs data_args) -> bool {
            stream.write(data_args.datachunk.data(), data_args.datachunk.size());
            return true;
        };

        minio::s3::GetObjectResponse response = client.GetObject(args);
        if (!response) {
            LOG_ERROR("Failed to download to stream from S3: {} - {}", 
                     file_path, response.Error().String());
            return false;
        }

        LOG_DEBUG("Downloaded to stream from S3: {}", file_path);
        return true; });
}

nlohmann::json S3FileStorage::getStorageInfo()
{
	return withClient([&](minio::s3::Client &client)
					  {
        nlohmann::json info;
        
        info["type"] = "s3";
        info["bucket"] = state_->bucket_name;
        info["region"] = state_->region;
        info["endpoint"] = state_->endpoint;
        info["use_ssl"] = state_->use_ssl;
        info["connected"] = state_->connected.load();
        info["status"] = state_->connected.load() ? "connected" : "disconnected";
        
        // Try to get more detailed info
        try {
            minio::s3::BucketExistsArgs args;
            args.bucket = state_->bucket_name;
            auto response = client.BucketExists(args);
            info["bucket_exists"] = response && response.exist;
            
            if (response && response.exist) {
                info["accessible"] = true;
            } else {
                info["accessible"] = false;
            }
        }
        catch (...) {
            info["bucket_exists"] = "unknown";
            info["accessible"] = false;
        }
        
        LOG_DEBUG("S3 storage info retrieved successfully");
        return info; });
}

std::string S3FileStorage::ensureTrailingSlash(const std::string &path) const
{
	if (path.empty() || path.back() == '/')
	{
		return path;
	}
	return path + '/';
}