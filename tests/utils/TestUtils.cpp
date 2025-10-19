#include "TestUtils.h"
#include <random>
#include <opencv2/opencv.hpp>

namespace TestUtils
{
	std::filesystem::path createTempDir(const std::string &prefix)
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<> dis(1000, 9999);

		std::string dirName = prefix + "_" + std::to_string(dis(gen));
		std::filesystem::path tempDir = std::filesystem::temp_directory_path() / dirName;

		std::filesystem::create_directories(tempDir);
		return tempDir;
	}

	void createTestConfig(const std::filesystem::path &configPath)
	{
		std::ofstream configFile(configPath);
		configFile << R"(
server:
  host: "127.0.0.1"
  port: 8081

app:
  uploadPath: "./test_uploads"
  resultPath: "./test_results"
  frontendBuildPath: "./test_static"
  logPath: "./test_logs"
  max_content_length: 10485760
  allowed_extensions: ["jpg", "jpeg", "png", "mp4", "avi", "webm", "mp3"]
  task_expiration: 3600
  emotion_categories: ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

redis:
  host: "127.0.0.1"
  port: 6379
  db: 1
  password: ""

mtcnn:
  keep_all: false
  post_process: true
  min_face_size: 20
  device: "cpu"

model:
  backend: "onnx"
  emotion_model_path: "./test_models/enet_b2_7.onnx"
  face_detection_models_path: "./test_models"
)";
		configFile.close();
	}

	void createTestImage(const std::filesystem::path &imagePath, int width, int height)
	{
		cv::Mat testImage(height, width, CV_8UC3, cv::Scalar(100, 150, 200));
		cv::imwrite(imagePath.string(), testImage);
	}

	void setupTestLogging()
	{
		auto null_sink = std::make_shared<spdlog::sinks::null_sink_mt>();
		auto logger = std::make_shared<spdlog::logger>("test_logger", null_sink);
		spdlog::set_default_logger(logger);
	}

	void cleanupTestDir(const std::filesystem::path &path)
	{
		if (std::filesystem::exists(path))
		{
			std::filesystem::remove_all(path);
		}
	}
}