#include "FileProcessor.h"
#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <db/RedisManager.h>
#include <emotionai/Image.h>

namespace fs = std::filesystem;
namespace EmotionAI
{

	// Define the MTCNNImpl struct (placeholder implementation)
	struct MTCNNImpl
	{
		// Placeholder for MTCNN implementation details
		// This would contain the actual MTCNN models and state
		~MTCNNImpl() = default;
	};

	// Emotion categories configuration
	const std::vector<std::string> FileProcessor::EMOTION_CATEGORIES = {
		"Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"};

	// Allowed file extensions
	const std::vector<std::string> FileProcessor::ALLOWED_EXTENSIONS = {
		"png", "jpg", "jpeg", "mp4", "avi", "webm"};

	FileProcessor::FileProcessor(db::RedisManager &redis_manager)
		: redis_manager_(redis_manager), mtcnn_(nullptr), model_loaded_(false)
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
		// Cleanup resources - unique_ptr will automatically delete mtcnn_
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
			// Placeholder for model loading
			// In a real implementation, this would load ONNX models or TorchScript models
			spdlog::info("Initializing emotion recognition models...");

			// Initialize MTCNN (placeholder)
			mtcnn_ = std::make_unique<MTCNNImpl>();

			// Try to load emotion model
			try
			{
				// This would be the path to your trained model
				std::string model_path = "models/emotion_model.pt";
				if (fs::exists(model_path))
				{
					emotion_model_ = torch::jit::load(model_path);
					emotion_model_.eval();
					model_loaded_ = true;
					spdlog::info("Emotion model loaded successfully");
				}
				else
				{
					spdlog::warn("Emotion model file not found: {}", model_path);
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

	cv::Mat FileProcessor::preprocess_face(const cv::Mat &face_image)
	{
		cv::Mat processed;

		// Resize to model input size (assuming 224x224)
		cv::resize(face_image, processed, cv::Size(224, 224));

		// Convert to float and normalize
		processed.convertTo(processed, CV_32F, 1.0 / 255.0);

		// Convert BGR to RGB
		cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);

		return processed;
	}

	std::vector<float> FileProcessor::predict_emotions(const cv::Mat &face_image)
	{
		if (!model_loaded_)
		{
			// Return dummy probabilities if model isn't loaded
			return std::vector<float>(EMOTION_CATEGORIES.size(), 1.0f / EMOTION_CATEGORIES.size());
		}

		try
		{
			// Preprocess the face image
			cv::Mat processed = preprocess_face(face_image);

			// Convert to tensor
			auto input_tensor = torch::from_blob(processed.data, {1, processed.rows, processed.cols, 3});
			input_tensor = input_tensor.permute({0, 3, 1, 2}); // NHWC to NCHW
			input_tensor = input_tensor.to(torch::kFloat32);

			// Run inference
			auto output = emotion_model_.forward({input_tensor}).toTensor();
			auto probabilities = torch::softmax(output, 1);

			// Convert to vector
			std::vector<float> result(probabilities.data_ptr<float>(),
									  probabilities.data_ptr<float>() + probabilities.numel());

			return result;
		}
		catch (const std::exception &e)
		{
			spdlog::error("Error in emotion prediction: {}", e.what());
			return std::vector<float>(EMOTION_CATEGORIES.size(), 1.0f / EMOTION_CATEGORIES.size());
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
			std::string result_filename = "result_" + filename;
			std::string result_path = (fs::path("results") / result_filename).string();

			// Save the processed image
			std::ofstream out_file(result_path, std::ios::binary);
			out_file << result_image.get_buffer();
			out_file.close();

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
			else
			{
				redis_manager_.set_task_status(task_id, nlohmann::json{
															{"error", "Неподдерживаемый формат файла"},
															{"complete", true}}
															.dump());
			}
		}
		catch (const std::exception &e)
		{
			spdlog::error("Error processing file {}: {}", filename, e.what());
			redis_manager_.set_task_status(task_id, nlohmann::json{
														{"error", e.what()},
														{"complete", true}}
														.dump());
		}

		cleanup_file(filepath);
	}

	std::pair<cv::Mat, nlohmann::json> FileProcessor::process_image(const cv::Mat &image)
	{
		// This method should be implemented in Image.cpp
		// For now, create a placeholder implementation
		Image img(image);
		return img.process_image(image);
	}

} // namespace EmotionAI