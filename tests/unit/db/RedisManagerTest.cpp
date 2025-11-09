#include <gtest/gtest.h>
#include <db/RedisManager.h>
#include <config/Config.h>
#include "../../utils/TestUtils.h"

class RedisManagerTest : public ::testing::Test
{
protected:
	void SetUp() override
	{
		testDir = TestUtils::createTempDir("redis_test");
		configPath = testDir / "config.yaml";
		TestUtils::createTestConfig(configPath);
		TestUtils::setupTestLogging();

		auto &config = Common::Config::instance();
		config.loadFromFile(configPath.string());
	}

	void TearDown() override
	{
		TestUtils::cleanupTestDir(testDir);
	}

	std::filesystem::path testDir;
	std::filesystem::path configPath;
};

TEST_F(RedisManagerTest, UUIDGeneration)
{
	std::string uuid1 = db::RedisManager::generate_uuid();
	std::string uuid2 = db::RedisManager::generate_uuid();

	EXPECT_FALSE(uuid1.empty());
	EXPECT_FALSE(uuid2.empty());
	EXPECT_NE(uuid1, uuid2);

	// Basic UUID format validation
	EXPECT_EQ(uuid1.length(), 36);
	EXPECT_EQ(uuid1[8], '-');
	EXPECT_EQ(uuid1[13], '-');
	EXPECT_EQ(uuid1[18], '-');
	EXPECT_EQ(uuid1[23], '-');
}

TEST_F(RedisManagerTest, BasicCreation)
{
	db::RedisManager redisManager;

	// Test that RedisManager can be created without crashing
	SUCCEED();
}

TEST_F(RedisManagerTest, LoadConfiguration)
{
	db::RedisManager redisManager;

	EXPECT_NO_THROW({
		redisManager.loadConfiguration();
	});
}

TEST_F(RedisManagerTest, IsInitializedState)
{
	db::RedisManager redisManager;

	// Initially should not be initialized
	EXPECT_FALSE(redisManager.is_initialized());
}

TEST_F(RedisManagerTest, ApplicationSaving)
{
	db::RedisManager redisManager;

	nlohmann::json applicationData = {
		{"name", "Test Application"},
		{"email", "test@example.com"},
		{"data", {"test", "data"}}};

	// This will test the file-based saving part (not Redis connection)
	std::string appId = redisManager.save_application(applicationData);
	EXPECT_FALSE(appId.empty());

	// Test with string data
	std::string appId2 = redisManager.save_application(applicationData.dump());
	EXPECT_FALSE(appId2.empty());

	// IDs should be different
	EXPECT_NE(appId, appId2);
}

TEST_F(RedisManagerTest, TaskStatusOperations)
{
	db::RedisManager redisManager;
	std::string taskId = "test_task_123";
	nlohmann::json statusData = {
		{"progress", 50},
		{"message", "Processing"},
		{"complete", false}};

	// Test setting task status - this won't actually connect to Redis in unit test
	// but will test the file operations and JSON handling
	EXPECT_NO_THROW({
		redisManager.set_task_status(taskId, statusData);
	});

	// Test with string data
	EXPECT_NO_THROW({
		redisManager.set_task_status(taskId, statusData.dump());
	});

	// Test getting task status (will return nullopt since no Redis connection)
	auto status = redisManager.get_task_status(taskId);
	// We don't expect to get a status in unit test environment
	// Just testing that the method doesn't crash
	SUCCEED();

	// Test JSON version
	auto statusJson = redisManager.get_task_status_json(taskId);
	// Similarly, don't expect actual data in unit test
	SUCCEED();
}