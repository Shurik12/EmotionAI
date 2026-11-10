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

// `ai.mode` is the single tri-state switch: ai | local | false. YAML `false`
// arrives as a boolean scalar, which must be accepted too.
TEST_F(ConfigTest, AIModeTriState)
{
	auto &config = Config::instance();
	const fs::path path = fs::temp_directory_path() / "emotionai_ai_mode_test.yaml";

	auto load = [&](const std::string &contents)
	{
		{
			std::ofstream out(path);
			out << contents;
		}
		EXPECT_TRUE(config.loadFromFile(path.string()));
	};

	load("ai:\n  mode: \"local\"\n");
	EXPECT_TRUE(config.ai().isLocal());
	EXPECT_TRUE(config.ai().interpretationEnabled());
	EXPECT_FALSE(config.ai().focusEnabled());

	load("ai:\n  mode: \"ai\"\n");
	EXPECT_TRUE(config.ai().isAi());
	EXPECT_TRUE(config.ai().interpretationEnabled());
	EXPECT_TRUE(config.ai().focusEnabled());

	load("ai:\n  mode: false\n");
	EXPECT_TRUE(config.ai().isOff());
	EXPECT_FALSE(config.ai().interpretationEnabled());
	EXPECT_FALSE(config.ai().focusEnabled());

	// Backwards compatibility with the old keys, including the originally
	// attempted `enabled: local`.
	load("ai:\n  enabled: false\n  depression_mode: \"local\"\n");
	EXPECT_TRUE(config.ai().isLocal());

	load("ai:\n  enabled: local\n");
	EXPECT_TRUE(config.ai().isLocal());

	load("ai:\n  enabled: true\n");
	EXPECT_TRUE(config.ai().isAi());

	fs::remove(path);
}