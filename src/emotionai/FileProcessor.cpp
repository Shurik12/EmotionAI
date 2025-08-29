#include "FileProcessor.h"
#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <db/RedisManager.h>
#include <emotionai/Image.h>
#include <common/Config.h>

namespace fs = std::filesystem;
namespace EmotionAI
{
	// Emotion categories configuration
	const std::vector<std::string> FileProcessor::EMOTION_CATEGORIES = {
		"Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"};

	// Allowed file extensions
	const std::vector<std::string> FileProcessor::ALLOWED_EXTENSIONS = {
		"png", "jpg", "jpeg", "mp4", "avi", "webm"};

	FileProcessor::FileProcessor(db::RedisManager &redis_manager)
		: redis_manager_(redis_manager), model_loaded_(false)
	{
		try
		{
			initialize_models();
		}
		catch (const std::exception &e)
		{
			spdlog::error("Failed to initialize models: {}", e.what());
		}
	}

	FileProcessor::~FileProcessor()
	{
		// Cleanup resources
	}

	bool FileProcessor::allowed_file(const std::string &filename)
	{
		size_t dot_pos = filename.find_last_of('.');
		if (dot_pos == std::string::npos)
		{
			return false;
		}

		std::string extension = filename.substr(dot_pos + 1);
		std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

		return std::find(ALLOWED_EXTENSIONS.begin(), ALLOWED_EXTENSIONS.end(), extension) != ALLOWED_EXTENSIONS.end();
	}

	void FileProcessor::cleanup_file(const std::string &filepath)
	{
		try
		{
			if (fs::exists(filepath))
			{
				fs::remove(filepath);
				spdlog::info("Cleaned up file: {}", filepath);
			}
		}
		catch (const std::exception &e)
		{
			spdlog::error("Error cleaning up file {}: {}", filepath, e.what());
		}
	}

	void FileProcessor::initialize_models()
	{
		try
		{
			spdlog::info("Initializing emotion recognition models...");

			auto &config = Common::Config::instance();

			// Get model paths from configuration
			std::string model_backend = "onnx";
			std::string emotion_model_path = "/home/alex/git/EmotionAI/contrib/emotiefflib/models/emotieffcpplib_prepared_models/enet_b2_7.onnx";

			// Try to load emotion model
			try
			{
				if (fs::exists(emotion_model_path))
				{
					fer_ = EmotiEffLib::EmotiEffLibRecognizer::createInstance(model_backend, emotion_model_path);
					model_loaded_ = true;
					spdlog::info("Emotion model loaded successfully: {}", emotion_model_path);
				}
				else
				{
					spdlog::warn("Emotion model file not found: {}", emotion_model_path);
					// Try to find the model in different locations
					std::vector<std::string> possible_paths = {
						emotion_model_path,
						"./models/" + fs::path(emotion_model_path).filename().string(),
						"/usr/share/emotionai/models/" + fs::path(emotion_model_path).filename().string()};

					for (const auto &path : possible_paths)
					{
						if (fs::exists(path))
						{
							fer_ = EmotiEffLib::EmotiEffLibRecognizer::createInstance(model_backend, path);
							model_loaded_ = true;
							spdlog::info("Emotion model loaded from alternative path: {}", path);
							break;
						}
					}

					if (!model_loaded_)
					{
						spdlog::error("Could not find emotion model in any known location");
					}
				}
			}
			catch (const std::exception &e)
			{
				spdlog::error("Failed to load emotion model: {}", e.what());
			}
		}
		catch (const std::exception &e)
		{
			spdlog::error("Error initializing models: {}", e.what());
			throw;
		}
	}
	void FileProcessor::process_image_file(const std::string &task_id, const std::string &filepath, const std::string &filename)
	{
		redis_manager_.set_task_status(task_id, nlohmann::json{
													{"progress", 0},
													{"message", "Processing image..."},
													{"error", nullptr},
													{"complete", false}}
													.dump());

		try
		{
			// Use Image class to load the file
			Image input_image(filepath);
			cv::Mat image = input_image.to_cv_mat();

			// Save uploaded image
			std::string result_filename = "result_" + filename;
			std::string result_path = (fs::path("result") / result_filename).string();
			if (cv::imwrite(result_path, image))
				spdlog::info("Image saved successfully to {}", result_path);
			else
				spdlog::error("Error: Could not save image to {}", result_path);

			if (image.empty())
			{
				throw std::runtime_error("Could not read image");
			}

			redis_manager_.set_task_status(task_id, nlohmann::json{
														{"progress", 50},
														{"message", "Processing image..."},
														{"error", nullptr},
														{"complete", false}}
														.dump());

			auto [processed_image, result] = process_image(image);

			// Save result using Image class
			Image result_image(processed_image, ".jpg");

			// Update status
			redis_manager_.set_task_status(task_id, nlohmann::json{
														{"complete", true},
														{"type", "image"},
														{"image_url", "/api/results/" + result_filename},
														{"result", result},
														{"progress", 100}}
														.dump());
		}
		catch (const std::exception &e)
		{
			throw std::runtime_error("Image processing failed: " + std::string(e.what()));
		}
	}

	void FileProcessor::process_video_file(const std::string &task_id, const std::string &filepath, const std::string &filename)
	{
		redis_manager_.set_task_status(task_id, nlohmann::json{
													{"progress", 0},
													{"message", "Processing video..."},
													{"error", nullptr},
													{"complete", false}}
													.dump());

		try
		{
			cv::VideoCapture cap(filepath);
			if (!cap.isOpened())
			{
				throw std::runtime_error("Could not open video");
			}

			int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
			double fps = cap.get(cv::CAP_PROP_FPS);
			double duration = (fps > 0) ? total_frames / fps : 0;

			spdlog::info("Processing video: {}, Frames: {}, FPS: {:.2f}, Duration: {:.2f}s",
						 filename, total_frames, fps, duration);

			int frame_count = 0;
			int processed_count = 0;
			nlohmann::json results = nlohmann::json::array();
			int frame_interval = (total_frames > 5) ? std::max(1, total_frames / 5) : 1;

			while (cap.isOpened())
			{
				cv::Mat frame;
				if (!cap.read(frame))
				{
					break;
				}

				if (frame_count % frame_interval == 0 || frame_count == total_frames - 1)
				{
					redis_manager_.set_task_status(task_id, nlohmann::json{
																{"progress", static_cast<int>((frame_count * 100.0) / total_frames)},
																{"message", fmt::format("Processing frame {} of {}...", frame_count + 1, total_frames)},
																{"error", nullptr},
																{"complete", false}}
																.dump());

					try
					{
						auto [processed_frame, result] = process_image(frame);

						std::string frame_filename = fmt::format("frame_{}_{}.jpg", processed_count,
																 filename.substr(0, filename.find_last_of('.')));
						std::string frame_path = (fs::path("results") / frame_filename).string();
						cv::imwrite(frame_path, processed_frame);

						nlohmann::json frame_result;
						frame_result["frame"] = frame_count;
						frame_result["image_url"] = "/api/results/" + frame_filename;
						frame_result["result"] = result;
						results.push_back(frame_result);

						processed_count++;

						if (processed_count >= 5)
						{
							break;
						}
					}
					catch (const std::exception &e)
					{
						spdlog::warn("Error processing frame {}: {}", frame_count, e.what());
						continue;
					}
				}

				frame_count++;
			}

			cap.release();

			if (results.empty())
			{
				throw std::runtime_error("No frames were processed successfully");
			}

			redis_manager_.set_task_status(task_id, nlohmann::json{
														{"complete", true},
														{"type", "video"},
														{"frames_processed", results.size()},
														{"results", results},
														{"progress", 100}}
														.dump());
		}
		catch (const std::exception &e)
		{
			throw std::runtime_error("Video processing failed: " + std::string(e.what()));
		}
	}

	void FileProcessor::process_file(const std::string &task_id, const std::string &filepath, const std::string &filename)
	{
		try
		{
			nlohmann::json initial_status = {
				{"progress", 0},
				{"message", "Начало обработки..."},
				{"error", nullptr},
				{"complete", false},
				{"model", "emotieff"},
				{"model_name", "EmotiEffLib"}};
			redis_manager_.set_task_status(task_id, initial_status.dump());

			if (!allowed_file(filename))
			{
				throw std::runtime_error("Unsupported file format");
			}

			size_t dot_pos = filename.find_last_of('.');
			if (dot_pos == std::string::npos)
			{
				throw std::runtime_error("Invalid filename");
			}

			std::string file_ext = filename.substr(dot_pos + 1);
			std::transform(file_ext.begin(), file_ext.end(), file_ext.begin(), ::tolower);

			if (file_ext == "png" || file_ext == "jpg" || file_ext == "jpeg")
			{
				process_image_file(task_id, filepath, filename);
			}
			else if (file_ext == "mp4" || file_ext == "avi" || file_ext == "webm")
			{
				process_video_file(task_id, filepath, filename);
			}
		}
		catch (const std::exception &e)
		{
			spdlog::error("Error processing file {}: {}", filename, e.what());

			// Try to update status even if Redis might be having issues
			try
			{
				redis_manager_.set_task_status(task_id, nlohmann::json{
															{"error", e.what()},
															{"complete", true}});
			}
			catch (const std::exception &redis_error)
			{
				spdlog::error("Failed to update Redis status for task {}: {}", task_id, redis_error.what());
			}
		}

		cleanup_file(filepath);
	}

	std::pair<cv::Mat, nlohmann::json> FileProcessor::process_image(const cv::Mat &image)
	{
		// Pass the emotion recognizer to Image for processing
		Image img(image);
		return img.process_image(image, fer_.get());
	}
} // namespace EmotionAI