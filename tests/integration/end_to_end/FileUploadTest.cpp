#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <server/ServerFactory.h>
#include <config/Config.h>
#include <client/Client.h>

class FileUploadTest : public ::testing::Test
{
protected:
	void SetUp() override
	{
		// Load E2E test configuration
		auto &config = Config::instance();
		if (!config.loadFromFile("tests/configs/e2e_config.yaml"))
		{
			GTEST_SKIP() << "E2E test configuration not found, skipping test";
		}

		// Setup test directories using config values
		std::filesystem::create_directories(config.uploadPath());
		std::filesystem::create_directories(config.resultPath());
		std::filesystem::create_directories(config.frontendBuildPath());
		std::filesystem::create_directories(config.logPath());

		// Start server
		server_ = ServerFactory::createServer(config.serverType());
		server_->initialize();
		server_thread_ = std::thread([this]()
									 { server_->start(); });

		std::this_thread::sleep_for(std::chrono::seconds(1));

		// Set fixtures path and create client using config values
		fixtures_path_ = std::filesystem::current_path() / "tests" / "fixtures";
		client_ = std::make_unique<HttpClient>(config.serverHost(), config.serverPort());
	}

	void TearDown() override
	{
		// Wait for up to 30 seconds for all background processing to complete
		auto start_time = std::chrono::steady_clock::now();
		auto timeout = std::chrono::seconds(30);

		while (std::chrono::steady_clock::now() - start_time < timeout)
		{
			// Check if any background threads are still active by polling task status
			bool all_completed = true;

			// You could add a method to check active tasks, or just wait
			std::this_thread::sleep_for(std::chrono::milliseconds(100));

			// Simple approach: just wait a bit longer to ensure processing completes
			// In a real implementation, you might poll Redis to check task statuses

			// For now, we'll break after a reasonable wait
			if (std::chrono::steady_clock::now() - start_time > std::chrono::seconds(5))
			{
				break;
			}
		}

		if (server_)
		{
			server_->stop();
		}
		if (server_thread_.joinable())
		{
			server_thread_.join();
		}

		// Additional wait to ensure all background threads are done
		std::this_thread::sleep_for(std::chrono::seconds(2));

		// Cleanup test directories using config values
		auto &config = Config::instance();
		if (config.isLoaded())
		{
			std::filesystem::remove_all(config.uploadPath());
			std::filesystem::remove_all(config.resultPath());
			std::filesystem::remove_all(config.frontendBuildPath());
			std::filesystem::remove_all(config.logPath());
		}
	}

	std::vector<char> readFixtureFile(const std::string &relative_path)
	{
		std::filesystem::path full_path = fixtures_path_ / relative_path;
		if (!std::filesystem::exists(full_path))
		{
			throw std::runtime_error("Fixture file not found: " + full_path.string());
		}

		std::ifstream file(full_path, std::ios::binary);
		if (!file)
		{
			throw std::runtime_error("Cannot open fixture file: " + full_path.string());
		}

		return std::vector<char>((std::istreambuf_iterator<char>(file)),
								 std::istreambuf_iterator<char>());
	}

	bool fixtureExists(const std::string &relative_path)
	{
		std::filesystem::path full_path = fixtures_path_ / relative_path;
		return std::filesystem::exists(full_path) &&
			   std::filesystem::is_regular_file(full_path);
	}

	std::string getUploadPath() const
	{
		auto &config = Config::instance();
		return config.uploadPath();
	}

	std::unique_ptr<IServer> server_;
	std::thread server_thread_;
	std::filesystem::path fixtures_path_;
	std::unique_ptr<HttpClient> client_;
};

TEST_F(FileUploadTest, UploadValidImage)
{
	if (!fixtureExists("images/test_image.jpg"))
	{
		GTEST_SKIP() << "Test image fixture not found, skipping test";
	}

	auto response = client_->uploadFileFromDisk("/api/upload",
												(fixtures_path_ / "images/test_image.jpg").string());

	EXPECT_TRUE(response.success());
	EXPECT_EQ(response.status_code, 202);

	auto response_json = response.json();
	EXPECT_TRUE(response_json.contains("task_id"));

	std::string task_id = response_json["task_id"];
	EXPECT_FALSE(task_id.empty());

	// Verify file was uploaded using config path
	std::filesystem::path uploaded_file = std::filesystem::path(getUploadPath()) / (task_id + "_test_image.jpg");

	// Wait for file to be created (async processing)
	bool file_created = false;
	for (int i = 0; i < 10; ++i)
	{
		if (std::filesystem::exists(uploaded_file))
		{
			file_created = true;
			break;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}

	EXPECT_TRUE(file_created) << "Uploaded file was not created: " << uploaded_file;
}

TEST_F(FileUploadTest, UploadWithCustomClientMethod)
{
	if (!fixtureExists("images/test_image.jpg"))
	{
		GTEST_SKIP() << "Test image fixture not found, skipping test";
	}

	auto image_data = readFixtureFile("images/test_image.jpg");
	std::string image_content(image_data.begin(), image_data.end());

	// Use the multipart upload method
	std::map<std::string, std::string> fields;
	std::map<std::string, std::pair<std::string, std::string>> files = {
		{"file", {"test_image.jpg", image_content}}};

	auto response = client_->uploadMultipart("/api/upload", fields, files);

	EXPECT_TRUE(response.success());
	EXPECT_EQ(response.status_code, 202);

	auto response_json = response.json();
	EXPECT_TRUE(response_json.contains("task_id"));
}

TEST_F(FileUploadTest, UploadInvalidFileType)
{
	// Create a test file with invalid extension
	std::string invalid_file_path = "/tmp/invalid_file.txt";
	std::ofstream invalid_file(invalid_file_path);
	invalid_file << "This is a text file, not an image or video";
	invalid_file.close();

	auto response = client_->uploadFileFromDisk("/api/upload", invalid_file_path);

	// Should be rejected due to invalid file type
	EXPECT_FALSE(response.success());
	EXPECT_EQ(response.status_code, 400);

	// Cleanup
	std::filesystem::remove(invalid_file_path);
}