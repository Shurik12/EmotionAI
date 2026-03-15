#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <server/ServerFactory.h>
#include <logging/Logger.h>
#include <config/Config.h>
#include <client/Client.h>

class ServerIntegrationTest : public ::testing::Test
{
protected:
	void SetUp() override
	{
		// Load integration test configuration
		auto &config = Config::instance();
		if (!config.loadFromFile("tests/configs/integration_config.yaml"))
		{
			GTEST_SKIP() << "Integration test configuration not found, skipping test";
		}

		// Setup test directories using config values
		std::filesystem::create_directories(config.paths().uploads);
		std::filesystem::create_directories(config.paths().results);
		std::filesystem::create_directories(config.paths().frontend);
		std::filesystem::create_directories(config.paths().logs);

		Logger::instance().initialize(
			"/tmp/test_logs",
			"EmotionAI-Tests",
			spdlog::level::err);

		// Create a simple index.html for static file serving
		std::ofstream index_file(config.paths().frontend + "/index.html");
		index_file << "<html><body>Test Server</body></html>";
		index_file.close();

		// Start server in background thread
		server_ = ServerFactory::createServer(config.server().type);
		server_->initialize();
		server_thread_ = std::thread([this]()
									 { server_->start(); });

		// Wait for server to start
		std::this_thread::sleep_for(std::chrono::seconds(1));

		// Create HTTP client using config values
		client_ = std::make_unique<HttpClient>(config.server().host, config.server().port);
	}

	void TearDown() override
	{
		if (server_)
		{
			server_->stop();
		}
		if (server_thread_.joinable())
		{
			server_thread_.join();
		}

		// Cleanup test directories using config values
		auto &config = Config::instance();
		if (config.isLoaded())
		{
			std::filesystem::remove_all(config.paths().uploads);
			std::filesystem::remove_all(config.paths().results);
			std::filesystem::remove_all(config.paths().frontend);
			std::filesystem::remove_all(config.paths().logs);
		}
	}

	std::vector<char> readFile(const std::string &filepath)
	{
		std::ifstream file(filepath, std::ios::binary);
		if (!file)
		{
			throw std::runtime_error("Cannot open file: " + filepath);
		}
		return std::vector<char>((std::istreambuf_iterator<char>(file)),
								 std::istreambuf_iterator<char>());
	}

	bool fileExists(const std::string &filepath)
	{
		return std::filesystem::exists(filepath) &&
			   std::filesystem::is_regular_file(filepath);
	}

	std::unique_ptr<IServer> server_;
	std::thread server_thread_;
	std::unique_ptr<HttpClient> client_;
};

TEST_F(ServerIntegrationTest, HealthCheck)
{
	auto response = client_->get("/api/health");

	EXPECT_TRUE(response.success());
	EXPECT_EQ(response.status_code, 200);
	EXPECT_EQ(response.body, R"({"status": "healthy"})");
}

TEST_F(ServerIntegrationTest, ServeStaticFiles)
{
	auto response = client_->get("/");

	EXPECT_TRUE(response.success());
	EXPECT_EQ(response.status_code, 200);
	EXPECT_NE(response.body.find("Test Server"), std::string::npos);
}

TEST_F(ServerIntegrationTest, SubmitApplication)
{
	nlohmann::json application_data = {
		{"name", "John Doe"},
		{"email", "john@example.com"},
		{"position", "Developer"}};

	auto response = client_->post("/api/submit_application", application_data);

	EXPECT_TRUE(response.success());
	EXPECT_EQ(response.status_code, 201);

	auto response_json = response.json();
	EXPECT_TRUE(response_json.contains("application_id"));
	EXPECT_FALSE(response_json["application_id"].get<std::string>().empty());
}

TEST_F(ServerIntegrationTest, SubmitApplication_InvalidJson)
{
	auto response = client_->post("/api/submit_application", "invalid json", "application/json");

	EXPECT_FALSE(response.success());
	EXPECT_EQ(response.status_code, 400);
}

TEST_F(ServerIntegrationTest, SendCustomRequest)
{
	std::string sync_data = R"({"id": 123, "name": "Test User", "phone": "+1234567890", "number": 42})";
	auto sync_response = client_->sendRequest("/api/submit_application", "POST", sync_data);

	EXPECT_TRUE(sync_response.success() || sync_response.status_code == 400);
	// Either successful or rejected due to schema, but should not be 404 or 500
	EXPECT_NE(sync_response.status_code, 404);
	EXPECT_NE(sync_response.status_code, 500);
}