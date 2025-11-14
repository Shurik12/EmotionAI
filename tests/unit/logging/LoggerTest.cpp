#include <gtest/gtest.h>
#include <filesystem>
#include <logging/Logger.h>

namespace fs = std::filesystem;

class LoggerTest : public ::testing::Test
{
protected:
	void SetUp() override
	{
		test_log_dir_ = "./test_logs";
		fs::create_directories(test_log_dir_);
	}

	void TearDown() override
	{
		if (fs::exists(test_log_dir_))
		{
			fs::remove_all(test_log_dir_);
		}
	}

	std::string test_log_dir_;
};

TEST_F(LoggerTest, SingletonPattern)
{
	auto &logger1 = Logger::instance();
	auto &logger2 = Logger::instance();
	EXPECT_EQ(&logger1, &logger2);
}

TEST_F(LoggerTest, InitializeNoThrow)
{
	EXPECT_NO_THROW({
		Logger::instance().initialize(test_log_dir_, "TestApp");
	});
}