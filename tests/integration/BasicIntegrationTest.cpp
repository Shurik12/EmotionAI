#include <gtest/gtest.h>
#include <server/WebServer.h>
#include <common/Config.h>
#include <db/RedisManager.h>
#include <emotionai/FileProcessor.h>
#include "../utils/TestUtils.h"

class BasicIntegrationTest : public ::testing::Test
{
protected:
	void SetUp() override
	{
		testDir = TestUtils::createTempDir("basic_integration_test");
		configPath = testDir / "config.yaml";
		TestUtils::createTestConfig(configPath);
		TestUtils::setupTestLogging();
	}

	void TearDown() override
	{
		TestUtils::cleanupTestDir(testDir);
	}

	std::filesystem::path testDir;
	std::filesystem::path configPath;
};

TEST_F(BasicIntegrationTest, WebServerInitialization)
{
	WebServer server;

	EXPECT_NO_THROW({
		server.initialize();
	});

	// Test that configuration was loaded
	auto &config = Common::Config::instance();
	EXPECT_TRUE(config.isLoaded());

	// Test that directories were created
	EXPECT_TRUE(std::filesystem::exists(config.uploadPath()));
	EXPECT_TRUE(std::filesystem::exists(config.resultPath()));
	EXPECT_TRUE(std::filesystem::exists(config.logPath()));
}

TEST_F(BasicIntegrationTest, RedisManagerInitialization)
{
	db::RedisManager redisManager;

	EXPECT_NO_THROW({
		redisManager.loadConfiguration();
	});

	// Test UUID generation
	std::string uuid1 = db::RedisManager::generate_uuid();
	std::string uuid2 = db::RedisManager::generate_uuid();

	EXPECT_FALSE(uuid1.empty());
	EXPECT_FALSE(uuid2.empty());
	EXPECT_NE(uuid1, uuid2);
}

TEST_F(BasicIntegrationTest, ConfigurationReload)
{
	auto &config = Common::Config::instance();

	EXPECT_TRUE(config.loadFromFile(configPath.string()));
	EXPECT_TRUE(config.isLoaded());

	// Test reload
	EXPECT_TRUE(config.reload());
	EXPECT_TRUE(config.isLoaded());

	// Test specific configuration values
	EXPECT_EQ(config.serverHost(), "127.0.0.1");
	EXPECT_EQ(config.serverPort(), 8081);
	EXPECT_EQ(config.redisHost(), "127.0.0.1");
}

TEST_F(BasicIntegrationTest, FileProcessorCreation)
{
	db::RedisManager redisManager;

	// Test that FileProcessor can be created
	EXPECT_NO_THROW({
		EmotionAI::FileProcessor processor(redisManager);
	});
}

TEST_F(BasicIntegrationTest, ComponentIntegration)
{
	// Test that all components can work together
	WebServer server;
	EXPECT_NO_THROW({
		server.initialize();
	});

	// Verify components were initialized
	auto &config = Common::Config::instance();
	EXPECT_TRUE(config.isLoaded());

	// Test directory structure
	EXPECT_TRUE(std::filesystem::exists(config.uploadPath()));
	EXPECT_TRUE(std::filesystem::exists(config.resultPath()));
	EXPECT_TRUE(std::filesystem::exists(config.logPath()));

	// Test RedisManager
	db::RedisManager redisManager;
	EXPECT_NO_THROW({
		redisManager.loadConfiguration();
	});

	// Test FileProcessor
	EXPECT_NO_THROW({
		EmotionAI::FileProcessor processor(redisManager);
	});
}