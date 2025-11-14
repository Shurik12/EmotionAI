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
	class DragonflyManager;
}

namespace EmotionAI
{
	class Image; // Forward declaration

	// Forward declaration for MTCNN implementation
	struct MTCNNImpl;

	class FileProcessor
	{
	public:
		explicit FileProcessor(std::shared_ptr<DragonflyManager> dragonfly_manager);
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

		void process_video_realtime(const std::string &task_id, const std::string &filepath, const std::string &filename);

	private:
		std::shared_ptr<DragonflyManager> dragonfly_manager_;

		std::unique_ptr<EmotiEffLib::EmotiEffLibRecognizer> fer_;
		bool model_loaded_;
		std::mutex model_mutex_;

		void cleanup_file(const std::string &filepath);
		std::pair<cv::Mat, nlohmann::json> process_image(const cv::Mat &image);
		nlohmann::json process_image_file(const std::string &task_id, const std::string &filepath, const std::string &filename);
		nlohmann::json process_video_file(const std::string &task_id, const std::string &filepath, const std::string &filename);

		// Helper functions
		void initialize_models();
	};
} // namespace EmotionAI