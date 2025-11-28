#pragma once

#include "FileStorage.h"
#include <miniocpp/client.h>
#include <memory>
#include <string>

class S3FileStorage : public FileStorage
{
public:
	S3FileStorage(const std::string &endpoint,
				  const std::string &access_key,
				  const std::string &secret_key,
				  const std::string &bucket,
				  bool use_ssl = true,
				  const std::string &region = "us-east-1");
	~S3FileStorage() override = default;

	// FileStorage interface implementation
	bool saveFile(const std::string &file_content, const std::string &file_path) override;
	bool saveFile(const std::vector<uint8_t> &file_content, const std::string &file_path) override;
	std::string readFile(const std::string &file_path) override;
	std::vector<uint8_t> readFileBinary(const std::string &file_path) override;
	bool deleteFile(const std::string &file_path) override;
	bool fileExists(const std::string &file_path) override;

	// Directory operations
	bool createDirectory(const std::string &path) override;
	bool deleteDirectory(const std::string &path) override;
	std::vector<std::string> listFiles(const std::string &path) override;

	// File info
	size_t getFileSize(const std::string &file_path) override;
	std::string getFileHash(const std::string &file_path) override;

	// URL generation for web access
	std::string getFileUrl(const std::string &file_path) override;

	// Storage info
	std::string getStorageType() const override { return "s3"; }
	nlohmann::json getStorageInfo() override;

	// Copy/move operations between storage systems
	bool copyFile(const std::string &source_path, const std::string &dest_path) override;
	bool moveFile(const std::string &source_path, const std::string &dest_path) override;

private:
	std::unique_ptr<minio::s3::Client> client_;
	std::string bucket_name_;
	std::string region_;
};