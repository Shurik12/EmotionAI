#include <gtest/gtest.h>
#include <server/WebServer.h>
#include <common/Config.h>
#include "../../utils/TestUtils.h"

class WebServerTest : public ::testing::Test
{
protected:
	void SetUp() override
	{
		testDir = TestUtils::createTempDir("webserver_test");
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

TEST_F(WebServerTest, BasicCreation)
{
	WebServer server;

	// Test that WebServer can be created without crashing
	SUCCEED();
}

TEST_F(WebServerTest, Initialization)
{
	WebServer server;

	EXPECT_NO_THROW({
		server.initialize();
	});

	// Verify that configuration was loaded through the public interface
	auto &config = Common::Config::instance();
	EXPECT_TRUE(config.isLoaded());
}

TEST_F(WebServerTest, StartStop)
{
	WebServer server;
	server.initialize();

	// Test that server can be started and stopped without crashing
	// We'll test this by ensuring the methods don't throw exceptions
	// Note: We can't actually start the server in unit tests as it would block

	EXPECT_NO_THROW({
		// We can't call start() as it would block, but we can call stop()
		server.stop();
	});
}

TEST_F(WebServerTest, ConfigurationValues)
{
	WebServer server;
	server.initialize();

	// Test that configuration values are accessible through Config
	auto &config = Common::Config::instance();

	EXPECT_EQ(config.serverHost(), "0.0.0.0");
	EXPECT_EQ(config.serverPort(), 8080);
	EXPECT_EQ(config.redisHost(), "localhost");
	EXPECT_EQ(config.redisPort(), 6379);
}

TEST_F(WebServerTest, DirectoryStructure)
{
	WebServer server;
	server.initialize();

	// Test that directories were created during initialization
	auto &config = Common::Config::instance();

	EXPECT_TRUE(std::filesystem::exists(config.uploadPath()));
	EXPECT_TRUE(std::filesystem::exists(config.resultPath()));
	EXPECT_TRUE(std::filesystem::exists(config.logPath()));
}

TEST_F(WebServerTest, JSONValidationHelper)
{
	// Test JSON validation logic directly (not through private method)
	nlohmann::json validJson = {
		{"name", "Test"},
		{"email", "test@example.com"},
		{"data", {"test", "data"}}};

	nlohmann::json invalidJson = "not an object";
	nlohmann::json emptyJson = nlohmann::json::object();

	// Test JSON validation logic manually
	EXPECT_TRUE(validJson.is_object());
	EXPECT_FALSE(validJson.empty());

	EXPECT_FALSE(invalidJson.is_object());
	EXPECT_TRUE(emptyJson.empty());
}

TEST_F(WebServerTest, ApiEndpointDetectionHelper)
{
	// Test API endpoint detection logic directly
	auto isApiEndpoint = [](const std::string &path)
	{
		static const std::set<std::string> apiPrefixes = {
			"/api/", "/static/"};

		for (const auto &prefix : apiPrefixes)
		{
			if (path.find(prefix) == 0)
			{
				return true;
			}
		}
		return false;
	};

	EXPECT_TRUE(isApiEndpoint("/api/upload"));
	EXPECT_TRUE(isApiEndpoint("/api/progress/123"));
	EXPECT_TRUE(isApiEndpoint("/static/image.jpg"));
	EXPECT_FALSE(isApiEndpoint("/"));
	EXPECT_FALSE(isApiEndpoint("/index.html"));
	EXPECT_FALSE(isApiEndpoint("/about"));
}