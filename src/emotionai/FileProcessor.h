#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <emotiefflib/facial_analysis.h>

// Forward declaration to avoid circular dependency
namespace db
{
	class RedisManager;
}

namespace EmotionAI
{

	class Image; // Forward declaration

	// Forward declaration for MTCNN implementation
	struct MTCNNImpl;

	class FileProcessor
	{
	public:
		explicit FileProcessor(db::RedisManager &redis_manager);
		~FileProcessor();

		FileProcessor(const FileProcessor &) = delete;
		FileProcessor &operator=(const FileProcessor &) = delete;
		FileProcessor(FileProcessor &&) = delete;
		FileProcessor &operator=(FileProcessor &&) = delete;

		bool allowed_file(const std::string &filename);
		void process_file(const std::string &task_id, const std::string &filepath, const std::string &filename);

	private:
		db::RedisManager &redis_manager_;

		// MTCNN implementation using PIMPL pattern
		std::unique_ptr<MTCNNImpl> mtcnn_;

		// Emotion recognition model
		std::unique_ptr<EmotiEffLib::EmotiEffLibRecognizer> fer_;
		bool model_loaded_;

		void cleanup_file(const std::string &filepath);
		std::pair<cv::Mat, nlohmann::json> process_image(const cv::Mat &image);
		void process_image_file(const std::string &task_id, const std::string &filepath, const std::string &filename);
		void process_video_file(const std::string &task_id, const std::string &filepath, const std::string &filename);

		// Helper functions
		void initialize_models();
		cv::Mat preprocess_face(const cv::Mat &face_image);

		// Configuration
		static const std::vector<std::string> EMOTION_CATEGORIES;
		static const std::vector<std::string> ALLOWED_EXTENSIONS;
	};

} // namespace EmotionAI