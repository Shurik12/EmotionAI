
#include <gtest/gtest.h>
#include <config/Config.h>
#include "../../utils/TestUtils.h"

class ConfigTest : public ::testing::Test
{
protected:
	void SetUp() override
	{
		testDir = TestUtils::createTempDir("config_test");
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

TEST_F(ConfigTest, LoadFromFileSuccess)
{
	auto &config = Common::Config::instance();
	EXPECT_TRUE(config.loadFromFile(configPath.string()));
	EXPECT_TRUE(config.isLoaded());
}

TEST_F(ConfigTest, LoadFromFileNotFound)
{
	auto &config = Common::Config::instance();
	EXPECT_FALSE(config.loadFromFile("nonexistent.yaml"));
	EXPECT_FALSE(config.isLoaded());
}

TEST_F(ConfigTest, ServerConfiguration)
{
	auto &config = Common::Config::instance();
	config.loadFromFile(configPath.string());

	EXPECT_EQ(config.serverHost(), "127.0.0.1");
	EXPECT_EQ(config.serverPort(), 8081);
}

TEST_F(ConfigTest, RedisConfiguration)
{
	auto &config = Common::Config::instance();
	config.loadFromFile(configPath.string());

	EXPECT_EQ(config.redisHost(), "127.0.0.1");
	EXPECT_EQ(config.redisPort(), 6379);
	EXPECT_EQ(config.redisDb(), 1);
}

TEST_F(ConfigTest, AppConfiguration)
{
	auto &config = Common::Config::instance();
	config.loadFromFile(configPath.string());

	auto extensions = config.allowedExtensions();
	EXPECT_FALSE(extensions.empty());
	EXPECT_NE(std::find(extensions.begin(), extensions.end(), "jpg"), extensions.end());

	auto emotions = config.emotionCategories();
	EXPECT_FALSE(emotions.empty());
	EXPECT_NE(std::find(emotions.begin(), emotions.end(), "happiness"), emotions.end());
}

TEST_F(ConfigTest, SetupApplicationEnvironment)
{
	auto &config = Common::Config::instance();
	config.loadFromFile(configPath.string());

	EXPECT_TRUE(config.setupApplicationEnvironment());

	// Verify directories were created
	EXPECT_TRUE(std::filesystem::exists(config.uploadPath()));
	EXPECT_TRUE(std::filesystem::exists(config.resultPath()));
	EXPECT_TRUE(std::filesystem::exists(config.logPath()));
}