#include <gtest/gtest.h>
#include <logging/Logger.h>
#include <config/Config.h>
#include "../../utils/TestUtils.h"

class LoggingTest : public ::testing::Test
{
protected:
	void SetUp() override
	{
		testDir = TestUtils::createTempDir("logging_test");
		configPath = testDir / "config.yaml";
		TestUtils::createTestConfig(configPath);

		auto &config = Common::Config::instance();
		config.loadFromFile(configPath.string());
		config.setupApplicationEnvironment();
	}

	void TearDown() override
	{
		TestUtils::cleanupTestDir(testDir);
	}

	std::filesystem::path testDir;
	std::filesystem::path configPath;
};

TEST_F(LoggingTest, MultiSinkInitialization)
{
	EXPECT_NO_THROW({
		Common::multi_sink_example("test.log");
	});
}

TEST_F(LoggingTest, RequestResponseLogging)
{
	httplib::Request req;
	httplib::Response res;

	req.method = "GET";
	req.path = "/api/test";
	req.version = "HTTP/1.1";

	res.status = 200;
	res.body = "Test response";

	EXPECT_NO_THROW({
		Common::log_request_response(req, res);
	});
}