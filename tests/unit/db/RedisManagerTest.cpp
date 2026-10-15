#include <gtest/gtest.h>
#include <db/DragonflyManager.h>
#include <logging/Logger.h>

class UuidTest : public ::testing::Test
{
protected:
	void SetUp() override
	{
		Logger::instance().initialize(
			"/tmp/test_logs",
			"EmotionAI-Tests",
			spdlog::level::err);
	}
};

TEST_F(UuidTest, UuidGeneration)
{
	std::string uuid1 = DragonflyManager::generate_uuid();
	std::string uuid2 = DragonflyManager::generate_uuid();

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
