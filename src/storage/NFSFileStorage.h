#pragma once

#include "LocalFileStorage.h"

class NFSFileStorage : public LocalFileStorage
{
public:
	explicit NFSFileStorage(const std::string &mount_point);
	~NFSFileStorage() override = default;

	std::string getStorageType() const override { return "nfs"; }
	nlohmann::json getStorageInfo() override;

private:
	std::string mount_point_;

	bool testNFSConnection();
};