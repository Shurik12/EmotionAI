#pragma once

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/null_sink.h>
#include <filesystem>
#include <fstream>

namespace TestUtils
{
	// Create a temporary directory for tests
	std::filesystem::path createTempDir(const std::string &prefix = "emotionai_test");

	// Create test configuration file
	void createTestConfig(const std::filesystem::path &configPath);

	// Create test image file
	void createTestImage(const std::filesystem::path &imagePath, int width = 100, int height = 100);

	// Setup test logging
	void setupTestLogging();

	// Cleanup test resources
	void cleanupTestDir(const std::filesystem::path &path);
}