#include <gtest/gtest.h>
#include <db/RedisManager.h>
#include <logging/Logger.h>

class RedisManagerTest : public ::testing::Test
{
protected:
	void SetUp() override
	{
		Logger::instance().initialize(
			"/tmp/test_logs",
			"EmotionAI-Tests",
			spdlog::level::err);
	}

	void TearDown() override
	{
		// Basic cleanup
	}
};

TEST_F(RedisManagerTest, UuidGeneration)
{
	db::RedisManager manager;

	std::string uuid1 = manager.generate_uuid();
	std::string uuid2 = manager.generate_uuid();

	EXPECT_FALSE(uuid1.empty());
	EXPECT_FALSE(uuid2.empty());
	EXPECT_NE(uuid1, uuid2);

	// Basic UUID format check
	EXPECT_EQ(uuid1.length(), 36);
	EXPECT_EQ(uuid1[8], '-');
	EXPECT_EQ(uuid1[13], '-');
	EXPECT_EQ(uuid1[18], '-');
	EXPECT_EQ(uuid1[23], '-');
}

TEST_F(RedisManagerTest, DefaultConfiguration)
{
	db::RedisManager manager;

	// Test that we can call these methods without throwing
	EXPECT_NO_THROW(manager.loadConfiguration());

	// Initially should not be initialized
	EXPECT_FALSE(manager.is_initialized());
}

TEST_F(RedisManagerTest, DISABLED_RedisConnection)
{
	GTEST_SKIP() << "Skipping Redis connection tests - requires Redis server";
}