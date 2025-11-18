#include "EmotionServerBenchmark.h"
#include <fstream>
#include <sstream>
#include <thread>
#include <chrono>
#include <spdlog/spdlog.h>

using namespace std::chrono_literals;

EmotionServerBenchmark::EmotionServerBenchmark(const BenchmarkConfig &config)
	: config_(config)
{

	client_ = std::make_unique<HttpClient>(config.server_host, config.server_port, config.timeout_seconds);
	loadTestFiles();
}

EmotionServerBenchmark::~EmotionServerBenchmark() = default;

void EmotionServerBenchmark::loadTestFiles()
{
	// Load test image
	std::ifstream image_file(config_.test_image_path, std::ios::binary);
	if (!image_file)
	{
		throw std::runtime_error("Cannot open test image file: " + config_.test_image_path);
	}

	std::stringstream image_buffer;
	image_buffer << image_file.rdbuf();
	test_image_data_ = image_buffer.str();
	image_file.close();

	// Load test video
	std::ifstream video_file(config_.test_video_path, std::ios::binary);
	if (!video_file)
	{
		throw std::runtime_error("Cannot open test video file: " + config_.test_video_path);
	}

	std::stringstream video_buffer;
	video_buffer << video_file.rdbuf();
	test_video_data_ = video_buffer.str();
	video_file.close();

	files_loaded_ = true;
}

std::string EmotionServerBenchmark::readTestImage() const
{
	if (!files_loaded_)
	{
		throw std::runtime_error("Test files not loaded");
	}
	return test_image_data_;
}

std::string EmotionServerBenchmark::readTestVideo() const
{
	if (!files_loaded_)
	{
		throw std::runtime_error("Test files not loaded");
	}
	return test_video_data_;
}

bool EmotionServerBenchmark::benchmarkImageUpload(benchmark::State &state, std::string &task_id)
{
	auto response = client_->uploadFile("/api/upload", test_image_data_, "test_image.jpg", "file");

	if (!response.success() || response.status_code != 202)
	{
		state.SkipWithError(("Image upload failed: " + std::to_string(response.status_code) + " - " + response.body).c_str());
		return false;
	}

	// Parse response to get task_id
	try
	{
		auto json_response = nlohmann::json::parse(response.body);
		task_id = json_response["task_id"].get<std::string>();
		return true;
	}
	catch (const std::exception &e)
	{
		state.SkipWithError(("Failed to parse task_id from response: " + std::string(e.what())).c_str());
		return false;
	}
}

bool EmotionServerBenchmark::benchmarkVideoUpload(benchmark::State &state, std::string &task_id)
{
	auto response = client_->uploadFile("/api/upload", test_video_data_, "test_video.mp4", "file");

	if (!response.success() || response.status_code != 202)
	{
		state.SkipWithError(("Video upload failed: " + std::to_string(response.status_code) + " - " + response.body).c_str());
		return false;
	}

	try
	{
		auto json_response = nlohmann::json::parse(response.body);
		task_id = json_response["task_id"].get<std::string>();
		return true;
	}
	catch (const std::exception &e)
	{
		state.SkipWithError(("Failed to parse task_id from response: " + std::string(e.what())).c_str());
		return false;
	}
}

bool EmotionServerBenchmark::benchmarkRealtimeVideoUpload(benchmark::State &state, std::string &task_id)
{
	auto response = client_->uploadFile("/api/upload_realtime", test_video_data_, "test_video.mp4", "file");

	if (!response.success() || response.status_code != 202)
	{
		state.SkipWithError(("Realtime video upload failed: " + std::to_string(response.status_code) + " - " + response.body).c_str());
		return false;
	}

	try
	{
		auto json_response = nlohmann::json::parse(response.body);
		task_id = json_response["task_id"].get<std::string>();
		return true;
	}
	catch (const std::exception &e)
	{
		state.SkipWithError(("Failed to parse task_id from response: " + std::string(e.what())).c_str());
		return false;
	}
}

HttpClient::Response EmotionServerBenchmark::checkTaskProgress(const std::string &task_id)
{
	return client_->get("/api/progress/" + task_id);
}

HttpClient::Response EmotionServerBenchmark::batchCheckProgress(const std::vector<std::string> &task_ids)
{
	nlohmann::json request_body = task_ids;
	return client_->post("/api/batch_progress", request_body.dump(), "application/json");
}

bool EmotionServerBenchmark::waitForTaskCompletion(const std::string &task_id)
{
	for (int attempt = 0; attempt < config_.max_retries; ++attempt)
	{
		auto response = checkTaskProgress(task_id);

		if (response.success() && response.status_code == 200)
		{
			try
			{
				auto progress_data = nlohmann::json::parse(response.body);

				// Check various completion indicators
				if (progress_data.contains("complete") && progress_data["complete"] == true)
				{
					return true;
				}
				if (progress_data.contains("error"))
				{
					spdlog::warn("Task {} failed with error: {}", task_id, progress_data["error"].dump());
					return false; // Task failed
				}
				if (progress_data.contains("progress") && progress_data["progress"] == 100)
				{
					return true;
				}
				// Check if we have result data
				if (progress_data.contains("result") || progress_data.contains("main_prediction"))
				{
					return true;
				}
			}
			catch (const std::exception &e)
			{
				spdlog::warn("Failed to parse progress response for task {}: {}", task_id, e.what());
			}
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(config_.retry_delay_ms));
	}

	spdlog::error("Timeout waiting for task {} after {} attempts", task_id, config_.max_retries);
	return false;
}

bool EmotionServerBenchmark::isTaskComplete(const HttpClient::Response &response)
{
	if (!response.success() || response.status_code != 200)
	{
		return false;
	}

	try
	{
		auto progress_data = nlohmann::json::parse(response.body);
		return progress_data.contains("complete") && progress_data["complete"] == true;
	}
	catch (...)
	{
		return false;
	}
}

std::string EmotionServerBenchmark::extractTaskId(const HttpClient::Response &response)
{
	try
	{
		auto json_response = nlohmann::json::parse(response.body);
		return json_response["task_id"].get<std::string>();
	}
	catch (...)
	{
		return "";
	}
}