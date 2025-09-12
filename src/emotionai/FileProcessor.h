#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
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

		// Getter for emotion recognizer to be used by Image class
		EmotiEffLib::EmotiEffLibRecognizer *get_emotion_recognizer() const { return fer_.get(); }
		bool is_model_loaded() const { return model_loaded_; }

	private:
		db::RedisManager &redis_manager_;

		// Emotion recognition model (now shared with Image class)
		std::unique_ptr<EmotiEffLib::EmotiEffLibRecognizer> fer_;
		bool model_loaded_;

		void cleanup_file(const std::string &filepath);
		std::pair<cv::Mat, nlohmann::json> process_image(const cv::Mat &image);
		void process_image_file(const std::string &task_id, const std::string &filepath, const std::string &filename);
		void process_video_file(const std::string &task_id, const std::string &filepath, const std::string &filename);

		// Helper functions
		void initialize_models();
	};
} // namespace EmotionAI