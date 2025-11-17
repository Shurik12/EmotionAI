#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>
#include <config/Config.h>

namespace fs = std::filesystem;

class ConfigTest : public ::testing::Test
{
protected:
	void SetUp() override
	{
		test_config_path_ = "tests/configs/unit_config.yaml";
	}

	void TearDown() override
	{
	}

	std::string test_config_path_;
};

TEST_F(ConfigTest, SingletonPattern)
{
	auto &config1 = Config::instance();
	auto &config2 = Config::instance();
	EXPECT_EQ(&config1, &config2);
}

TEST_F(ConfigTest, LoadFromFileSuccess)
{
	auto &config = Config::instance();
	EXPECT_TRUE(config.loadFromFile(test_config_path_));
}

TEST_F(ConfigTest, LoadFromFileNotFound)
{
	auto &config = Config::instance();
	EXPECT_FALSE(config.loadFromFile("nonexistent.yaml"));
}