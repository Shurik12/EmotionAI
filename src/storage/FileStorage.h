#pragma once

#include <string>
#include <vector>
#include <memory>
#include <nlohmann/json.hpp>
#include <filesystem>

namespace fs = std::filesystem;

class FileStorage
{
public:
	virtual ~FileStorage() = default;

	// File operations
	virtual bool saveFile(const std::string &file_content, const std::string &file_path) = 0;
	virtual bool saveFile(const std::vector<uint8_t> &file_content, const std::string &file_path) = 0;
	virtual std::string readFile(const std::string &file_path) = 0;
	virtual std::vector<uint8_t> readFileBinary(const std::string &file_path) = 0;
	virtual bool deleteFile(const std::string &file_path) = 0;
	virtual bool fileExists(const std::string &file_path) = 0;

	// Directory operations
	virtual bool createDirectory(const std::string &path) = 0;
	virtual bool deleteDirectory(const std::string &path) = 0;
	virtual std::vector<std::string> listFiles(const std::string &path) = 0;

	// File info
	virtual size_t getFileSize(const std::string &file_path) = 0;
	virtual std::string getFileHash(const std::string &file_path) = 0;

	// URL generation for web access
	virtual std::string getFileUrl(const std::string &file_path) = 0;

	// Storage info
	virtual std::string getStorageType() const = 0;
	virtual nlohmann::json getStorageInfo() = 0;

	// Copy/move operations between storage systems
	virtual bool copyFile(const std::string &source_path, const std::string &dest_path) = 0;
	virtual bool moveFile(const std::string &source_path, const std::string &dest_path) = 0;
};