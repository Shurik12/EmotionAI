#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <emotiefflib/facial_analysis.h>
#include <db/DragonflyManager.h>
#include <emotionai/Image.h>
#include <storage/FileStorage.h>

class FileProcessor
{
public:
    using ProgressCallback = std::function<void(int progress, const std::string &message)>;

    explicit FileProcessor(std::shared_ptr<DragonflyManager> dragonfly_manager,
                           std::shared_ptr<FileStorage> file_storage);
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

    // Updated to accept progress callback
    void process_video_realtime(const std::string &task_id,
                                const std::string &filepath,
                                const std::string &filename,
                                ProgressCallback progress_callback = nullptr);

private:
    std::shared_ptr<DragonflyManager> dragonfly_manager_;
    std::shared_ptr<FileStorage> file_storage_;

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