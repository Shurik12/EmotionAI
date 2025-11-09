#include <filesystem>

#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <db/RedisManager.h>
#include <emotionai/Image.h>
#include <config/Config.h>
#include <logging/Logger.h>
#include "FileProcessor.h"

namespace fs = std::filesystem;
namespace EmotionAI
{
	FileProcessor::FileProcessor(db::RedisManager &redis_manager)
		: redis_manager_(redis_manager), model_loaded_(false)
	{
		try
		{
			initialize_models();
		}
		catch (const std::exception &e)
		{
			LOG_ERROR("Failed to initialize models: {}", e.what());
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

		auto &config = Common::Config::instance();
		std::vector<std::string> allowed_extensions = config.app().allowed_extensions;

		return std::find(allowed_extensions.begin(), allowed_extensions.end(), extension) != allowed_extensions.end();
	}

	void FileProcessor::cleanup_file(const std::string &filepath)
	{
		try
		{
			if (fs::exists(filepath))
			{
				fs::remove(filepath);
				LOG_INFO("Cleaned up file: {}", filepath);
			}
		}
		catch (const std::exception &e)
		{
			LOG_ERROR("Error cleaning up file {}: {}", filepath, e.what());
		}
	}

	void FileProcessor::initialize_models()
	{
		try
		{
			LOG_INFO("Initializing emotion recognition models...");

			// Get model paths from configuration
			auto &config = Common::Config::instance();
			std::string model_backend = config.model().backend;
			std::string emotion_model_path = config.model().emotion_model_path;

			// Try to load emotion model
			try
			{
				if (fs::exists(emotion_model_path))
				{
					fer_ = EmotiEffLib::EmotiEffLibRecognizer::createInstance(model_backend, emotion_model_path);
					model_loaded_ = true;
					LOG_INFO("Emotion model loaded successfully: {}", emotion_model_path);
				}
				else
				{
					LOG_WARN("Emotion model file not found: {}", emotion_model_path);
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
							LOG_INFO("Emotion model loaded from alternative path: {}", path);
							break;
						}
					}

					if (!model_loaded_)
					{
						LOG_ERROR("Could not find emotion model in any known location");
					}
				}
			}
			catch (const std::exception &e)
			{
				LOG_ERROR("Failed to load emotion model: {}", e.what());
			}
		}
		catch (const std::exception &e)
		{
			LOG_ERROR("Error initializing models: {}", e.what());
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
				LOG_INFO("Image saved successfully to {}", result_path);
			else
				LOG_ERROR("Error: Could not save image to {}", result_path);

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

			LOG_INFO("Processing video: {}, Frames: {}, FPS: {:.2f}, Duration: {:.2f}s",
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
						std::string frame_path = (fs::path("result") / frame_filename).string();

						if (cv::imwrite(frame_path, frame))
							LOG_INFO("Image saved successfully to {}", frame_path);
						else
							LOG_ERROR("Error: Could not save image to {}", frame_path);

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
						LOG_WARN("Error processing frame {}: {}", frame_count, e.what());
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

	void FileProcessor::process_video_realtime(const std::string &task_id, const std::string &filepath, const std::string &filename)
	{
		redis_manager_.set_task_status(task_id, nlohmann::json{
													{"progress", 0},
													{"message", "Processing video for real-time analysis..."},
													{"error", nullptr},
													{"complete", false},
													{"mode", "realtime"}}
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

			LOG_INFO("Real-time video analysis: {}, Frames: {}, FPS: {:.2f}",
						 filename, total_frames, fps);

			int frame_count = 0;
			std::vector<nlohmann::json> frame_results;
			std::vector<double> valence_history;
			std::vector<double> arousal_history;
			std::vector<int> frame_numbers;

			// Process frames at a reasonable rate
			int frame_interval = std::max(1, static_cast<int>(fps / 10)); // Process 2 frames per second
			int max_frames = 60;										 // Maximum frames to process for performance

			while (cap.isOpened() && frame_count < max_frames)
			{
				cv::Mat frame;
				if (!cap.read(frame))
				{
					break;
				}

				// Skip frames based on interval
				if (frame_count % frame_interval != 0)
				{
					frame_count++;
					continue;
				}

				// Update progress
				if (frame_results.size() % 5 == 0)
				{
					redis_manager_.set_task_status(task_id, nlohmann::json{
																{"progress", static_cast<int>((frame_count * 100.0) / std::min(total_frames, max_frames))},
																{"message", fmt::format("Analyzing frame {}...", frame_count + 1)},
																{"error", nullptr},
																{"complete", false},
																{"frames_processed", frame_results.size()}}
																.dump());
				}

				try
				{
					auto [processed_frame, result] = process_image(frame);

					// Save the processed frame image
					std::string frame_filename = fmt::format("realtime_frame_{}_{}.jpg", frame_results.size(),
															 filename.substr(0, filename.find_last_of('.')));
					std::string frame_path = (fs::path("result") / frame_filename).string();

					if (cv::imwrite(frame_path, frame))
					{
						LOG_INFO("Real-time frame saved: {}", frame_path);
					}
					else
					{
						LOG_WARN("Failed to save real-time frame: {}", frame_path);
					}

					// Store frame result with timestamp and image URL
					nlohmann::json frame_result;
					frame_result["frame_number"] = frame_count;
					frame_result["timestamp"] = frame_count / fps; // seconds
					frame_result["image_url"] = "/api/results/" + frame_filename;
					frame_result["result"] = result;

					// Extract valence and arousal if available
					if (result.contains("additional_probs"))
					{
						auto &additional = result["additional_probs"];
						if (additional.contains("valence"))
						{
							double valence = std::stod(additional["valence"].get<std::string>());
							valence_history.push_back(valence);
							frame_result["valence"] = valence;
						}
						if (additional.contains("arousal"))
						{
							double arousal = std::stod(additional["arousal"].get<std::string>());
							arousal_history.push_back(arousal);
							frame_result["arousal"] = arousal;
						}

						// Extract emotion probabilities
						std::vector<std::string> emotion_keys = {"anger", "disgust", "fear", "happiness",
																 "neutral", "sadness", "surprise", "contempt"};
						nlohmann::json emotions;
						for (const auto &emotion : emotion_keys)
						{
							if (additional.contains(emotion))
							{
								emotions[emotion] = std::stod(additional[emotion].get<std::string>());
							}
						}
						frame_result["emotions"] = emotions;
					}

					frame_numbers.push_back(frame_count);
					frame_results.push_back(frame_result);
				}
				catch (const std::exception &e)
				{
					LOG_WARN("Error processing frame {}: {}", frame_count, e.what());
				}

				frame_count++;

				// Break if we have enough samples
				if (frame_results.size() >= 30)
					break; // Max 30 data points for clean charts
			}

			cap.release();

			if (frame_results.empty())
			{
				throw std::runtime_error("No frames were processed successfully");
			}

			// Calculate overall statistics
			nlohmann::json stats;
			if (!valence_history.empty())
			{
				stats["valence_avg"] = std::accumulate(valence_history.begin(), valence_history.end(), 0.0) / valence_history.size();
				stats["valence_min"] = *std::min_element(valence_history.begin(), valence_history.end());
				stats["valence_max"] = *std::max_element(valence_history.begin(), valence_history.end());
			}
			if (!arousal_history.empty())
			{
				stats["arousal_avg"] = std::accumulate(arousal_history.begin(), arousal_history.end(), 0.0) / arousal_history.size();
				stats["arousal_min"] = *std::min_element(arousal_history.begin(), arousal_history.end());
				stats["arousal_max"] = *std::max_element(arousal_history.begin(), arousal_history.end());
			}

			// Calculate average emotion distribution
			nlohmann::json avg_emotions;
			std::vector<std::string> emotion_keys = {"anger", "disgust", "fear", "happiness",
													 "neutral", "sadness", "surprise", "contempt"};
			for (const auto &emotion : emotion_keys)
			{
				double sum = 0.0;
				int count = 0;
				for (const auto &frame : frame_results)
				{
					if (frame["emotions"].contains(emotion))
					{
						sum += frame["emotions"][emotion].get<double>();
						count++;
					}
				}
				if (count > 0)
				{
					avg_emotions[emotion] = sum / count;
				}
			}

			redis_manager_.set_task_status(task_id, nlohmann::json{
														{"complete", true},
														{"type", "video_realtime"},
														{"frames_processed", frame_results.size()},
														{"frame_results", frame_results},
														{"valence_history", valence_history},
														{"arousal_history", arousal_history},
														{"frame_numbers", frame_numbers},
														{"fps", fps},
														{"duration", total_frames / fps},
														{"statistics", stats},
														{"average_emotions", avg_emotions},
														{"progress", 100}}
														.dump());
		}
		catch (const std::exception &e)
		{
			throw std::runtime_error("Real-time video analysis failed: " + std::string(e.what()));
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
			LOG_ERROR("Error processing file {}: {}", filename, e.what());

			// Try to update status even if Redis might be having issues
			try
			{
				redis_manager_.set_task_status(task_id, nlohmann::json{
															{"error", e.what()},
															{"complete", true}});
			}
			catch (const std::exception &redis_error)
			{
				LOG_ERROR("Failed to update Redis status for task {}: {}", task_id, redis_error.what());
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