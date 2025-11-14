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
		test_config_path_ = "test_config.yaml";
		createTestConfig();
	}

	void TearDown() override
	{
		if (fs::exists(test_config_path_))
		{
			fs::remove(test_config_path_);
		}
	}

	void createTestConfig()
	{
		std::ofstream config_file(test_config_path_);
		config_file << R"(
server:
  host: "127.0.0.1"
  port: 8081
)";
		config_file.close();
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