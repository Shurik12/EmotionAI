#pragma once

#include "FileStorage.h"
#include <mutex>

class LocalFileStorage : public FileStorage
{
public:
	explicit LocalFileStorage(const std::string &base_path = "");
	~LocalFileStorage() override = default;

	// File operations
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
	std::string getStorageType() const override { return "local"; }
	nlohmann::json getStorageInfo() override;

	// Copy/move operations
	bool copyFile(const std::string &source_path, const std::string &dest_path) override;
	bool moveFile(const std::string &source_path, const std::string &dest_path) override;

private:
	fs::path base_path_;
	std::mutex file_mutex_;

	fs::path resolvePath(const std::string &file_path);
	std::string calculateFileHash(const std::string &file_path);
};