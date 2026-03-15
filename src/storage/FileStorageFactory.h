#pragma once

#include "FileStorage.h"
#include "LocalFileStorage.h"
#include "S3FileStorage.h"
#include "NFSFileStorage.h"
#include <memory>
#include <config/Config.h>

class FileStorageFactory
{
public:
	static std::unique_ptr<FileStorage> createStorage(const std::string &type,
													  const nlohmann::json &config);

	static std::shared_ptr<FileStorage> createStorageFromConfig();

	static std::map<std::string, std::string> getAvailableStorageTypes();
};