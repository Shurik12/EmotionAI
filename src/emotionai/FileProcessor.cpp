#include <filesystem>

#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <db/DragonflyManager.h>
#include <emotionai/Image.h>
#include <config/Config.h>
#include <logging/Logger.h>
#include <db/TaskManager.h>
#include "FileProcessor.h"

namespace fs = std::filesystem;

FileProcessor::FileProcessor(std::shared_ptr<DragonflyManager> dragonfly_manager)
	: dragonfly_manager_(std::move(dragonfly_manager))
{
	try
	{
		initialize_models();
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Failed to initialize models in FileProcessor constructor: {}", e.what());
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

	auto &config = Config::instance();
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
		auto &config = Config::instance();
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

nlohmann::json FileProcessor::process_image_file(const std::string &task_id, const std::string &filepath, const std::string &filename)
{
	// Use Image class to load the file
	Image input_image(filepath);
	cv::Mat image = input_image.to_cv_mat();

	// Get results path from config
	auto &config = Config::instance();
	std::string results_path = config.paths().results;

	// Ensure results directory exists
	fs::create_directories(results_path);

	// Save uploaded image with result prefix
	std::string result_filename = "result_" + filename;
	std::string result_file_path = (fs::path(results_path) / result_filename).string();

	if (image.empty())
	{
		throw std::runtime_error("Could not read image");
	}

	// Process the image and get both the annotated image and emotion results
	auto [processed_image, emotion_result] = process_image(image);

	// Save the processed image
	if (!cv::imwrite(result_file_path, image))
	{
		LOG_ERROR("Error: Could not save processed image to {}", result_file_path);
		throw std::runtime_error("Failed to save processed image");
	}

	LOG_INFO("Processed image saved successfully to {}", result_file_path);

	// Build the complete result JSON
	nlohmann::json result = {
		{"type", "image"},
		{"image_url", "/api/results/" + result_filename},
		{"result", emotion_result}};

	return result;
}

nlohmann::json FileProcessor::process_video_file(const std::string &task_id, const std::string &filepath, const std::string &filename)
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

	// Get results path from config
	auto &config = Config::instance();
	std::string results_path = config.paths().results;
	fs::create_directories(results_path);

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
			try
			{
				auto [processed_frame, result] = process_image(frame);

				std::string frame_filename = fmt::format("frame_{}_{}.jpg", processed_count,
														 filename.substr(0, filename.find_last_of('.')));
				std::string frame_file_path = (fs::path(results_path) / frame_filename).string();

				if (cv::imwrite(frame_file_path, frame))
				{
					LOG_INFO("Processed frame saved successfully to {}", frame_file_path);
				}
				else
				{
					LOG_ERROR("Error: Could not save processed frame to {}", frame_file_path);
					// Continue processing even if frame save fails
				}

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

	// Build the complete video result JSON
	nlohmann::json result = {
		{"type", "video"},
		{"frames_processed", results.size()},
		{"results", results},
		{"total_frames", total_frames},
		{"fps", fps},
		{"duration", duration}};

	return result;
}

void FileProcessor::process_video_realtime(const std::string &task_id,
										   const std::string &filepath,
										   const std::string &filename,
										   ProgressCallback progress_callback)
{
	LOG_INFO("=== REAL-TIME VIDEO PROCESSING STARTED ===");
	LOG_INFO("Task: {}, File: {}", task_id, filename);

	try
	{
		// Initial progress update
		if (progress_callback)
		{
			progress_callback(5, "Opening video file for real-time analysis...");
		}

		cv::VideoCapture cap(filepath);
		if (!cap.isOpened())
		{
			throw std::runtime_error("Could not open video file: " + filepath);
		}

		int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
		double fps = cap.get(cv::CAP_PROP_FPS);

		if (total_frames <= 0)
		{
			cap.release();
			throw std::runtime_error("Video has no frames or cannot read frame count");
		}

		LOG_INFO("Real-time video analysis: {}, Frames: {}, FPS: {:.2f}",
				 filename, total_frames, fps);

		if (progress_callback)
		{
			progress_callback(10, fmt::format("Video loaded: {} frames at {:.1f} FPS", total_frames, fps));
		}

		// Get results path from config
		auto &config = Config::instance();
		std::string results_path = config.paths().results;
		fs::create_directories(results_path);

		int frame_count = 0;
		int processed_count = 0;
		std::vector<nlohmann::json> frame_results;
		std::vector<double> valence_history;
		std::vector<double> arousal_history;
		std::vector<int> frame_numbers;

		// Process frames at a reasonable rate for real-time feedback
		int frame_interval = std::max(1, static_cast<int>(fps / 5)); // Process 5 frames per second
		int max_frames = std::min(total_frames, 60);				 // Maximum frames to process for performance

		LOG_INFO("Real-time processing parameters: interval={}, max_frames={}", frame_interval, max_frames);

		while (cap.isOpened() && frame_count < max_frames)
		{
			cv::Mat frame;
			if (!cap.read(frame))
			{
				LOG_DEBUG("No more frames to read at frame {}", frame_count);
				break;
			}

			// Skip frames based on interval for real-time performance
			if (frame_count % frame_interval != 0)
			{
				frame_count++;
				continue;
			}

			// Update progress more frequently for real-time feedback
			int progress = 10 + static_cast<int>((80.0 * frame_count) / max_frames);
			if (progress_callback && (frame_count % (frame_interval * 2) == 0))
			{
				progress_callback(progress, fmt::format("Analyzing frame {}/{}...", frame_count + 1, max_frames));
			}

			try
			{
				LOG_DEBUG("Processing frame {} for real-time analysis", frame_count);

				auto [processed_frame, result] = process_image(frame);

				// Save the processed frame image
				std::string frame_filename = fmt::format("realtime_{}_frame_{}.jpg",
														 task_id, processed_count);
				std::string frame_file_path = (fs::path(results_path) / frame_filename).string();

				if (cv::imwrite(frame_file_path, frame))
				{
					LOG_DEBUG("Real-time frame saved: {}", frame_file_path);
				}
				else
				{
					LOG_WARN("Failed to save real-time frame: {}", frame_file_path);
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
						try
						{
							double valence = std::stod(additional["valence"].get<std::string>());
							valence_history.push_back(valence);
							frame_result["valence"] = valence;
						}
						catch (const std::exception &e)
						{
							LOG_WARN("Failed to parse valence value: {}", e.what());
						}
					}
					if (additional.contains("arousal"))
					{
						try
						{
							double arousal = std::stod(additional["arousal"].get<std::string>());
							arousal_history.push_back(arousal);
							frame_result["arousal"] = arousal;
						}
						catch (const std::exception &e)
						{
							LOG_WARN("Failed to parse arousal value: {}", e.what());
						}
					}

					// Extract emotion probabilities
					std::vector<std::string> emotion_keys = {"anger", "disgust", "fear", "happiness",
															 "neutral", "sadness", "surprise", "contempt"};
					nlohmann::json emotions;
					for (const auto &emotion : emotion_keys)
					{
						if (additional.contains(emotion))
						{
							try
							{
								emotions[emotion] = std::stod(additional[emotion].get<std::string>());
							}
							catch (const std::exception &e)
							{
								LOG_WARN("Failed to parse emotion {}: {}", emotion, e.what());
							}
						}
					}
					frame_result["emotions"] = emotions;
				}

				frame_numbers.push_back(frame_count);
				frame_results.push_back(frame_result);
				processed_count++;

				LOG_DEBUG("Successfully processed frame {} for real-time analysis", frame_count);
			}
			catch (const std::exception &e)
			{
				LOG_WARN("Error processing frame {}: {}", frame_count, e.what());
				// Continue with next frame instead of failing completely
			}

			frame_count++;

			// Small delay to simulate real-time processing and prevent overwhelming the system
			std::this_thread::sleep_for(std::chrono::milliseconds(50));
		}

		cap.release();
		LOG_INFO("Real-time video processing completed. Processed {} frames out of {}", processed_count, frame_count);

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
				if (frame.contains("emotions") && frame["emotions"].contains(emotion))
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

		// Final progress update
		if (progress_callback)
		{
			progress_callback(95, "Finalizing real-time analysis results...");
		}

		// Store final results via TaskManager
		auto &task_manager = TaskManager::instance();
		task_manager.set_task_status(task_id, {{"complete", true},
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
											   {"progress", 100},
											   {"message", "Real-time analysis completed successfully"},
											   {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
																 std::chrono::system_clock::now().time_since_epoch())
																 .count()}});

		if (progress_callback)
		{
			progress_callback(100, "Real-time analysis completed successfully");
		}

		LOG_INFO("=== REAL-TIME VIDEO PROCESSING COMPLETED ===");
		LOG_INFO("Task: {}, Processed frames: {}", task_id, frame_results.size());
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("=== REAL-TIME VIDEO PROCESSING FAILED ===");
		LOG_ERROR("Task: {}, Error: {}", task_id, e.what());

		if (progress_callback)
		{
			progress_callback(0, std::string("Real-time processing failed: ") + e.what());
		}

		// Store error status
		auto &task_manager = TaskManager::instance();
		task_manager.set_task_status(task_id, {{"complete", true},
											   {"error", e.what()},
											   {"progress", 0},
											   {"message", "Real-time processing failed"},
											   {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
																 std::chrono::system_clock::now().time_since_epoch())
																 .count()}});

		throw;
	}
}

void FileProcessor::process_file(const std::string &task_id, const std::string &filepath, const std::string &filename)
{
	// Get TaskManager instance
	auto &task_manager = TaskManager::instance();

	try
	{
		// Initial status via TaskManager
		task_manager.set_task_status(task_id, {{"task_id", task_id},
											   {"progress", 0},
											   {"message", "Starting processing..."},
											   {"error", nullptr},
											   {"complete", false},
											   {"model", "emotieff"},
											   {"model_name", "EmotiEffLib"},
											   {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
																 std::chrono::system_clock::now().time_since_epoch())
																 .count()}});

		LOG_INFO("Starting file processing - Task: {}, File: {}", task_id, filename);

		if (!allowed_file(filename))
		{
			throw std::runtime_error("Unsupported file format: " + filename);
		}

		size_t dot_pos = filename.find_last_of('.');
		if (dot_pos == std::string::npos)
		{
			throw std::runtime_error("Invalid filename: " + filename);
		}

		std::string file_ext = filename.substr(dot_pos + 1);
		std::transform(file_ext.begin(), file_ext.end(), file_ext.begin(), ::tolower);

		// Progress updates via TaskManager
		task_manager.set_task_status(task_id, {{"progress", 10},
											   {"message", "Validating file..."},
											   {"error", nullptr},
											   {"complete", false},
											   {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
																 std::chrono::system_clock::now().time_since_epoch())
																 .count()}});

		nlohmann::json processing_result;

		// Process based on file type
		if (file_ext == "png" || file_ext == "jpg" || file_ext == "jpeg")
		{
			task_manager.set_task_status(task_id, {{"progress", 20},
												   {"message", "Processing image file..."},
												   {"file_type", "image"},
												   {"complete", false}});

			processing_result = process_image_file(task_id, filepath, filename);
		}
		else if (file_ext == "mp4" || file_ext == "avi" || file_ext == "webm")
		{
			task_manager.set_task_status(task_id, {{"progress", 20},
												   {"message", "Processing video file..."},
												   {"file_type", "video"},
												   {"complete", false}});

			processing_result = process_video_file(task_id, filepath, filename);
		}
		else
		{
			throw std::runtime_error("Unsupported file type: " + file_ext);
		}

		// Final success status with results
		nlohmann::json success_status = {
			{"progress", 100},
			{"message", "Processing completed successfully"},
			{"complete", true},
			{"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
							  std::chrono::system_clock::now().time_since_epoch())
							  .count()}};

		// Merge processing results
		success_status.insert(processing_result.begin(), processing_result.end());

		task_manager.set_task_status(task_id, success_status);

		LOG_INFO("File processing completed successfully - Task: {}, File: {}", task_id, filename);
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Error processing file {} for task {}: {}", filename, task_id, e.what());

		// Error status via TaskManager
		task_manager.set_task_status(task_id, {{"task_id", task_id},
											   {"progress", 0},
											   {"message", "Processing failed"},
											   {"error", e.what()},
											   {"complete", true},
											   {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
																 std::chrono::system_clock::now().time_since_epoch())
																 .count()}});
	}

	// Clean up the uploaded file
	cleanup_file(filepath);
}

std::pair<cv::Mat, nlohmann::json> FileProcessor::process_image(const cv::Mat &image)
{
	std::lock_guard<std::mutex> lock(model_mutex_); // Add thread safety

	try
	{
		LOG_DEBUG("Starting image processing...");

		// Check if emotion model is loaded and pointer is valid
		if (!fer_ || !model_loaded_)
		{
			LOG_WARN("Emotion model not loaded, returning empty result");
			nlohmann::json empty_result = {
				{"emotion", "unknown"},
				{"confidence", 0.0},
				{"additional_probs", nlohmann::json::object()}};
			return {image.clone(), empty_result};
		}

		// Additional pointer validation
		if (reinterpret_cast<uintptr_t>(fer_.get()) < 0x1000)
		{
			LOG_ERROR("Invalid emotion recognizer pointer: {}", reinterpret_cast<void *>(fer_.get()));
			throw std::runtime_error("Invalid emotion recognizer pointer");
		}

		LOG_DEBUG("Emotion recognizer pointer is valid: {}", reinterpret_cast<void *>(fer_.get()));

		// Pass the emotion recognizer to Image for processing
		Image img(image);
		LOG_DEBUG("Image object created, calling process_image...");

		auto result = img.process_image(image, fer_.get());
		LOG_DEBUG("Image processing completed successfully");

		return result;
	}
	catch (const std::exception &e)
	{
		LOG_ERROR("Error in process_image: {}", e.what());
		// Return a safe default result
		nlohmann::json error_result = {
			{"emotion", "error"},
			{"confidence", 0.0},
			{"error", e.what()},
			{"additional_probs", nlohmann::json::object()}};
		return {image.clone(), error_result};
	}
}