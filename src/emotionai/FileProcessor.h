#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <emotiefflib/facial_analysis.h>
#include <db/DragonflyManager.h>
#include <emotionai/Image.h>
#include <emotionai/Audio.h>
#include <storage/FileStorage.h>
#include <gigachat/GigaChatClient.h>
#include <torch/script.h>
#include <audio/BurnoutModels.h>
#include <audio/BurnoutAnalyzer.h>

class FileProcessor
{
public:
    using ProgressCallback = std::function<void(int progress, const std::string &message)>;

    explicit FileProcessor(std::shared_ptr<DragonflyManager> dragonfly_manager,
                           std::shared_ptr<FileStorage> file_storage);
    ~FileProcessor() = default;

    // Delete copy/move constructors
    FileProcessor(const FileProcessor&) = delete;
    FileProcessor& operator=(const FileProcessor&) = delete;
    FileProcessor(FileProcessor&&) = delete;
    FileProcessor& operator=(FileProcessor&&) = delete;

    // Public API
    bool allowed_file(const std::string& filename);
    void process_file(const std::string& task_id, const std::string& filepath, const std::string& filename);
    void process_video_realtime(const std::string& task_id, const std::string& filepath, 
                                const std::string& filename, ProgressCallback callback = nullptr);

    // ============ NEW: Burnout API ============
    // Process audio with burnout analysis
    nlohmann::json process_audio_with_burnout(
        const std::string& task_id,
        const std::string& filepath,
        const std::string& filename,
        const nlohmann::json& baseline = {}
    );
    
    // Analyze burnout from existing emotion result
    audio::Result analyze_burnout_from_result(
        const nlohmann::json& emotion_result,
        const nlohmann::json& baseline = {}
    );
    
    // Save user baseline
    void save_user_baseline(
        const std::string& user_id,
        const nlohmann::json& baseline
    );
    
    // Get user baseline
    nlohmann::json get_user_baseline(const std::string& user_id);
    // =========================================

    // Getters
    EmotiEffLib::EmotiEffLibRecognizer* get_emotion_recognizer() const { return fer_.get(); }
    torch::jit::Module* get_audio_torch_model() { return audio_torch_model_.get(); }
    bool is_model_loaded() const { return model_loaded_; }
    bool is_audio_model_loaded() const { return audio_model_loaded_; }
    bool isGigaChatEnabled() const { return gigachat_client_ && gigachat_client_->isEnabled(); }

    void setGigaChatClient(std::unique_ptr<emotionai::gigachat::GigaChatClient> client) {
        gigachat_client_ = std::move(client);
    }

private:
    // Constants
    static constexpr int MAX_VIDEO_FRAMES = 60;
    static constexpr int MIN_FRAMES_FOR_VIDEO = 5;
    static constexpr int REALTIME_FPS_TARGET = 5;

    // Dependencies
    std::shared_ptr<DragonflyManager> dragonfly_manager_;
    std::shared_ptr<FileStorage> file_storage_;
    std::unique_ptr<emotionai::gigachat::GigaChatClient> gigachat_client_;

    // Models
    std::unique_ptr<EmotiEffLib::EmotiEffLibRecognizer> fer_;
    std::unique_ptr<torch::jit::Module> audio_torch_model_;
    std::mutex model_mutex_;
    
    bool model_loaded_ = false;
    bool audio_model_loaded_ = false;

    // Initialization
    void initialize_models();
    bool load_image_model(const std::string& model_path, const std::string& backend);
    bool load_audio_model(const std::string& model_path);

    // File type detection
    static bool is_audio_file(const std::string& filename);
    static std::string get_file_extension(const std::string& filename);
    static bool has_extension(const std::string& filename, const std::vector<std::string>& extensions);

    // Processing methods
    nlohmann::json process_audio_file(const std::string& task_id, const std::string& filepath, const std::string& filename);
    nlohmann::json process_image_file(const std::string& task_id, const std::string& filepath, const std::string& filename);
    nlohmann::json process_video_file(const std::string& task_id, const std::string& filepath, const std::string& filename);
    std::pair<cv::Mat, nlohmann::json> process_image(const cv::Mat& image);

    // Frame processing helpers
    struct FrameResult {
        cv::Mat frame;
        nlohmann::json result;
        std::string storage_path;
        int frame_number;
        double timestamp;
    };
    FrameResult process_video_frame(const cv::Mat& frame, int frame_number, double fps, 
                                     const std::string& task_id);

    // Video processing helpers
    nlohmann::json extract_video_metadata(cv::VideoCapture& cap);
    void update_realtime_progress(int progress, const std::string& message, ProgressCallback callback);
    std::string save_frame_to_storage(const cv::Mat& frame, const std::string& task_id, int frame_index);
    void save_json_to_storage(const nlohmann::json& data, const std::string& task_id, const std::string& prefix);
    std::vector<uint8_t> mat_to_vector(const cv::Mat& mat);

    // Statistics helpers
    nlohmann::json calculate_statistics(const std::vector<double>& values);
    nlohmann::json calculate_average_emotions(const std::vector<nlohmann::json>& frame_results);

    // GigaChat integration
    void addGigaChatAnalysis(nlohmann::json& result, const std::string& task_id, int frame_number = -1);
    emotionai::gigachat::EmotionData extract_emotions_from_result(const nlohmann::json& result);

    // Cleanup
    void cleanup_file(const std::string& filepath);
    void update_task_status(const std::string& task_id, const nlohmann::json& status);
};