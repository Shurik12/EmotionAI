#include <filesystem>
#include <chrono>
#include <thread>

#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <db/DragonflyManager.h>
#include <emotionai/Image.h>
#include <emotionai/Audio.h>
#include <config/Config.h>
#include <logging/Logger.h>
#include <db/TaskManager.h>
#include "FileProcessor.h"

namespace fs = std::filesystem;

//=============================================================================
// File type detection helpers
//=============================================================================
namespace {
    const std::vector<std::string> AUDIO_EXTENSIONS = {"mp3", "wav", "flac", "m4a", "aac", "ogg", "opus", "wma"};
    const std::vector<std::string> IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"};
    const std::vector<std::string> VIDEO_EXTENSIONS = {"mp4", "avi", "webm", "mov", "mkv"};
    const std::vector<std::string> ALLOWED_EXTENSIONS = []{
        auto all = AUDIO_EXTENSIONS;
        all.insert(all.end(), IMAGE_EXTENSIONS.begin(), IMAGE_EXTENSIONS.end());
        all.insert(all.end(), VIDEO_EXTENSIONS.begin(), VIDEO_EXTENSIONS.end());
        return all;
    }();
}

//=============================================================================
// Construction
//=============================================================================
FileProcessor::FileProcessor(std::shared_ptr<DragonflyManager> dragonfly_manager,
                             std::shared_ptr<FileStorage> file_storage)
    : dragonfly_manager_(std::move(dragonfly_manager))
    , file_storage_(std::move(file_storage))
{
    try {
        initialize_models();
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to initialize models: {}", e.what());
    }
}

//=============================================================================
// File type detection
//=============================================================================
std::string FileProcessor::get_file_extension(const std::string& filename)
{
    auto pos = filename.find_last_of('.');
    if (pos == std::string::npos) return "";
    std::string ext = filename.substr(pos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext;
}

bool FileProcessor::has_extension(const std::string& filename, const std::vector<std::string>& extensions)
{
    auto ext = get_file_extension(filename);
    return std::find(extensions.begin(), extensions.end(), ext) != extensions.end();
}

bool FileProcessor::allowed_file(const std::string& filename)
{
    return has_extension(filename, ALLOWED_EXTENSIONS);
}

bool FileProcessor::is_audio_file(const std::string& filename)
{
    return has_extension(filename, AUDIO_EXTENSIONS);
}

//=============================================================================
// Utility: Convert cv::Mat to vector<uint8_t>
//=============================================================================
std::vector<uint8_t> FileProcessor::mat_to_vector(const cv::Mat& mat)
{
    std::vector<uchar> buffer;
    cv::imencode(".jpg", mat, buffer);
    return std::vector<uint8_t>(buffer.begin(), buffer.end());
}

//=============================================================================
// Model Initialization
//=============================================================================
bool FileProcessor::load_image_model(const std::string& model_path, const std::string& backend)
{
    std::vector<std::string> search_paths = {
        model_path,
        "./models/" + fs::path(model_path).filename().string(),
        "/usr/share/emotionai/models/" + fs::path(model_path).filename().string()
    };

    for (const auto& path : search_paths) {
        if (fs::exists(path)) {
            try {
                fer_ = EmotiEffLib::EmotiEffLibRecognizer::createInstance(backend, path);
                model_loaded_ = true;
                LOG_INFO("Image emotion model loaded: {}", path);
                return true;
            } catch (const std::exception& e) {
                LOG_WARN("Failed to load model from {}: {}", path, e.what());
            }
        }
    }
    LOG_ERROR("Image emotion model not found in any location");
    return false;
}

bool FileProcessor::load_audio_model(const std::string& model_path)
{
    std::vector<std::string> search_paths = {
        model_path,
        "./models/" + fs::path(model_path).filename().string(),
        "/usr/share/emotionai/models/" + fs::path(model_path).filename().string()
    };

    for (const auto& path : search_paths) {
        if (fs::exists(path)) {
            try {
                LOG_INFO("Loading audio model: {} ({} bytes)", path, fs::file_size(path));
                auto module = torch::jit::load(path);
                audio_torch_model_ = std::make_unique<torch::jit::Module>(module);
                audio_model_loaded_ = true;
                LOG_INFO("Audio model loaded: {}", path);
                return true;
            } catch (const std::exception& e) {
                LOG_WARN("Failed to load audio model from {}: {}", path, e.what());
            }
        }
    }
    LOG_ERROR("Audio model not found in any location");
    return false;
}

void FileProcessor::initialize_models()
{
    LOG_INFO("Initializing models...");
    auto& config = Config::instance();

    // Load image model
    load_image_model(config.model().emotion_model_path, config.model().backend);
    
    // Load audio model
    load_audio_model(config.model().audio_model_path);

    LOG_INFO("Models initialized. Image: {}, Audio: {}", 
             model_loaded_ ? "OK" : "FAIL", 
             audio_model_loaded_ ? "OK" : "FAIL");
}

//=============================================================================
// Cleanup
//=============================================================================
void FileProcessor::cleanup_file(const std::string& filepath)
{
    try {
        if (filepath.find("uploads/") == 0 || filepath.find("results/") == 0) {
            if (file_storage_->fileExists(filepath)) {
                file_storage_->deleteFile(filepath);
                LOG_INFO("Cleaned up: {}", filepath);
            }
        } else if (fs::exists(filepath)) {
            fs::remove(filepath);
            LOG_INFO("Cleaned up local: {}", filepath);
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Cleanup error for {}: {}", filepath, e.what());
    }
}

void FileProcessor::update_task_status(const std::string& task_id, const nlohmann::json& status)
{
    TaskManager::instance().set_task_status(task_id, status);
}

void FileProcessor::save_json_to_storage(const nlohmann::json& data, const std::string& task_id, const std::string& prefix)
{
    std::string storage_path = fmt::format("results/{}_{}.json", prefix, task_id);
    std::string json_str = data.dump(2);
    std::vector<uint8_t> content(json_str.begin(), json_str.end());
    file_storage_->saveFile(content, storage_path);
}

//=============================================================================
// Main processing entry point
//=============================================================================
void FileProcessor::process_file(const std::string& task_id, const std::string& filepath, const std::string& filename)
{
    auto& task_manager = TaskManager::instance();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    try {
        LOG_INFO("Processing: Task={}, File={}", task_id, filename);
        update_task_status(task_id, {
            {"task_id", task_id}, {"progress", 0}, {"message", "Starting..."},
            {"error", nullptr}, {"complete", false}, {"timestamp", timestamp}
        });

        if (!allowed_file(filename)) {
            throw std::runtime_error("Unsupported file format: " + filename);
        }

        auto ext = get_file_extension(filename);
        nlohmann::json result;

        update_task_status(task_id, {{"progress", 10}, {"message", "Validating file..."}});

        if (has_extension(filename, IMAGE_EXTENSIONS)) {
            update_task_status(task_id, {{"file_type", "image"}, {"progress", 20}, {"message", "Processing image..."}});
            result = process_image_file(task_id, filepath, filename);
        } else if (has_extension(filename, VIDEO_EXTENSIONS)) {
            update_task_status(task_id, {{"file_type", "video"}, {"progress", 20}, {"message", "Processing video..."}});
            result = process_video_file(task_id, filepath, filename);
        } else if (is_audio_file(filename)) {
            update_task_status(task_id, {{"file_type", "audio"}, {"progress", 20}, {"message", "Processing audio..."}});
            result = process_audio_file(task_id, filepath, filename);
        } else {
            throw std::runtime_error("Unsupported file type: " + ext);
        }

        nlohmann::json final_status = {
            {"progress", 100}, {"message", "Completed"}, {"complete", true}, {"timestamp", timestamp}
        };
        final_status.insert(result.begin(), result.end());
        update_task_status(task_id, final_status);

        LOG_INFO("Processing completed: Task={}", task_id);

    } catch (const std::exception& e) {
        LOG_ERROR("Processing failed: Task={}, Error={}", task_id, e.what());
        update_task_status(task_id, {
            {"progress", 0}, {"message", "Failed"}, {"error", e.what()}, 
            {"complete", true}, {"timestamp", timestamp}
        });
    }

    cleanup_file(filepath);
}

//=============================================================================
// Audio Processing
//=============================================================================
nlohmann::json FileProcessor::process_audio_file(const std::string& task_id, 
                                                  const std::string& filepath, 
                                                  const std::string& filename)
{
    LOG_INFO("Processing audio: Task={}, File={}", task_id, filename);

    if (!audio_model_loaded_ || !audio_torch_model_) {
        throw std::runtime_error("Audio model not loaded");
    }

    Audio audio(filepath);
    if (!audio.is_loaded()) {
        throw std::runtime_error("Failed to load audio: " + audio.get_error());
    }

    LOG_INFO("Audio loaded: {:.2f}s, {}Hz, {} channels", 
             audio.get_duration(), audio.get_sample_rate(), audio.get_channels());

    auto result = audio.process_audio(audio_torch_model_.get());
    addGigaChatAnalysis(result, task_id);

    // Save result
    save_json_to_storage(result, task_id, "audio");

    LOG_INFO("Audio processing complete: Task={}", task_id);
    return {
        {"type", "audio"},
        {"filename", filename},
        {"duration", audio.get_duration()},
        {"sample_rate", audio.get_sample_rate()},
        {"storage_path", fmt::format("results/audio_{}.json", task_id)},
        {"result", result}
    };
}

//=============================================================================
// Image Processing
//=============================================================================
nlohmann::json FileProcessor::process_image_file(const std::string& task_id, 
                                                  const std::string& filepath, 
                                                  const std::string& filename)
{
    auto content = file_storage_->readFileBinary(filepath);
    if (content.empty()) {
        throw std::runtime_error("Could not read image from storage");
    }

    // Convert vector<uint8_t> to cv::Mat
    cv::Mat image = cv::imdecode(cv::Mat(content), cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Could not decode image");
    }

    // Save processed image
    auto image_data = mat_to_vector(image);
    std::string storage_path = "results/" + filename;
    file_storage_->saveFile(image_data, storage_path);

    auto [processed_image, emotion_result] = process_image(image);
    addGigaChatAnalysis(emotion_result, task_id);

    return {
        {"type", "image"},
        {"image_url", file_storage_->getFileUrl(storage_path)},
        {"storage_path", storage_path},
        {"result", emotion_result}
    };
}

//=============================================================================
// Video Processing
//=============================================================================
nlohmann::json FileProcessor::extract_video_metadata(cv::VideoCapture& cap)
{
    return {
        {"total_frames", static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT))},
        {"fps", cap.get(cv::CAP_PROP_FPS)},
        {"width", static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH))},
        {"height", static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))}
    };
}

std::string FileProcessor::save_frame_to_storage(const cv::Mat& frame, const std::string& task_id, int frame_index)
{
    auto image_data = mat_to_vector(frame);
    std::string storage_path = fmt::format("results/frame_{}_{}.jpg", task_id, frame_index);
    file_storage_->saveFile(image_data, storage_path);
    return storage_path;
}

nlohmann::json FileProcessor::process_video_file(const std::string& task_id, 
                                                  const std::string& filepath, 
                                                  const std::string& filename)
{
    cv::VideoCapture cap(filepath);
    if (!cap.isOpened()) {
        throw std::runtime_error("Could not open video");
    }

    auto metadata = extract_video_metadata(cap);
    int total_frames = metadata["total_frames"];
    double fps = metadata["fps"];

    LOG_INFO("Processing video: {}, Frames={}, FPS={:.2f}", filename, total_frames, fps);

    int frame_interval = std::max(1, total_frames / 5);
    int processed_count = 0;
    nlohmann::json results = nlohmann::json::array();

    for (int frame_num = 0; frame_num < total_frames; ++frame_num) {
        cv::Mat frame;
        if (!cap.read(frame)) break;

        if (frame_num % frame_interval != 0 && frame_num != total_frames - 1) continue;

        try {
            auto frame_result = process_video_frame(frame, frame_num, fps, task_id);
            results.push_back({
                {"frame", frame_num},
                {"image_url", file_storage_->getFileUrl(frame_result.storage_path)},
                {"result", frame_result.result}
            });
            processed_count++;
        } catch (const std::exception& e) {
            LOG_WARN("Frame {} failed: {}", frame_num, e.what());
        }
    }

    cap.release();

    if (results.empty()) {
        throw std::runtime_error("No frames processed");
    }

    return {
        {"type", "video"},
        {"frames_processed", processed_count},
        {"results", results},
        {"total_frames", total_frames},
        {"fps", fps},
        {"duration", total_frames / fps}
    };
}

FileProcessor::FrameResult FileProcessor::process_video_frame(const cv::Mat& frame, 
                                                               int frame_number, 
                                                               double fps,
                                                               const std::string& task_id)
{
    // Save frame
    std::string storage_path = save_frame_to_storage(frame, task_id, frame_number);

    // Process image
    auto [processed_frame, result] = process_image(frame);
    addGigaChatAnalysis(result, task_id, frame_number);

    return {
        .frame = processed_frame,
        .result = result,
        .storage_path = storage_path,
        .frame_number = frame_number,
        .timestamp = frame_number / fps
    };
}

//=============================================================================
// Real-time Video Processing
//=============================================================================
void FileProcessor::update_realtime_progress(int progress, const std::string& message, 
                                              ProgressCallback callback)
{
    if (callback) {
        callback(progress, message);
    }
}

void FileProcessor::process_video_realtime(const std::string& task_id,
                                            const std::string& filepath,
                                            const std::string& filename,
                                            ProgressCallback callback)
{
    LOG_INFO("Real-time video: Task={}, File={}", task_id, filename);
    update_realtime_progress(5, "Opening video...", callback);

    cv::VideoCapture cap(filepath);
    if (!cap.isOpened()) {
        throw std::runtime_error("Could not open video: " + filepath);
    }

    auto metadata = extract_video_metadata(cap);
    int total_frames = metadata["total_frames"];
    double fps = metadata["fps"];

    if (total_frames <= 0) {
        cap.release();
        throw std::runtime_error("Invalid video: no frames");
    }

    LOG_INFO("Video: {} frames at {:.1f} FPS", total_frames, fps);
    update_realtime_progress(10, fmt::format("Loaded: {} frames", total_frames), callback);

    int frame_interval = std::max(1, static_cast<int>(fps / REALTIME_FPS_TARGET));
    int max_frames = std::min(total_frames, MAX_VIDEO_FRAMES);
    
    LOG_INFO("Processing: interval={}, max_frames={}", frame_interval, max_frames);

    std::vector<nlohmann::json> frame_results;
    std::vector<double> valence_history, arousal_history;

    for (int frame_num = 0; frame_num < max_frames; ++frame_num) {
        cv::Mat frame;
        if (!cap.read(frame)) break;

        if (frame_num % frame_interval != 0) continue;

        int progress = 10 + static_cast<int>((80.0 * frame_num) / max_frames);
        if (callback && frame_num % (frame_interval * 2) == 0) {
            update_realtime_progress(progress, fmt::format("Frame {}/{}", frame_num + 1, max_frames), callback);
        }

        try {
            auto frame_result = process_video_frame(frame, frame_num, fps, task_id);
            
            // Extract emotions
            if (frame_result.result.contains("additional_probs")) {
                auto& probs = frame_result.result["additional_probs"];
                auto get_val = [&](const std::string& key) -> std::optional<double> {
                    if (!probs.contains(key)) return std::nullopt;
                    try { return std::stod(probs[key].get<std::string>()); }
                    catch (...) { return std::nullopt; }
                };

                if (auto v = get_val("valence")) valence_history.push_back(*v);
                if (auto a = get_val("arousal")) arousal_history.push_back(*a);
            }

            frame_results.push_back({
                {"frame_number", frame_num},
                {"timestamp", frame_num / fps},
                {"image_url", file_storage_->getFileUrl(frame_result.storage_path)},
                {"result", frame_result.result}
            });

        } catch (const std::exception& e) {
            LOG_WARN("Frame {} failed: {}", frame_num, e.what());
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    cap.release();
    LOG_INFO("Processed {} frames", frame_results.size());

    if (frame_results.empty()) {
        throw std::runtime_error("No frames processed");
    }

    update_realtime_progress(95, "Finalizing...", callback);

    // Calculate statistics
    nlohmann::json final_result = {
        {"complete", true},
        {"type", "video_realtime"},
        {"frames_processed", frame_results.size()},
        {"frame_results", frame_results},
        {"fps", fps},
        {"duration", total_frames / fps},
        {"statistics", {
            {"valence", calculate_statistics(valence_history)},
            {"arousal", calculate_statistics(arousal_history)}
        }},
        {"average_emotions", calculate_average_emotions(frame_results)},
        {"progress", 100},
        {"message", "Real-time analysis complete"}
    };

    update_task_status(task_id, final_result);
    update_realtime_progress(100, "Complete", callback);

    LOG_INFO("Real-time video complete: Task={}, Frames={}", task_id, frame_results.size());
}

//=============================================================================
// Statistics Helpers
//=============================================================================
nlohmann::json FileProcessor::calculate_statistics(const std::vector<double>& values)
{
    if (values.empty()) return nlohmann::json::object();
    
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return {
        {"avg", sum / values.size()},
        {"min", *std::min_element(values.begin(), values.end())},
        {"max", *std::max_element(values.begin(), values.end())}
    };
}

nlohmann::json FileProcessor::calculate_average_emotions(const std::vector<nlohmann::json>& frame_results)
{
    std::vector<std::string> emotion_keys = {"anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"};
    nlohmann::json avg_emotions;
    std::unordered_map<std::string, double> sums;
    std::unordered_map<std::string, int> counts;

    for (const auto& frame : frame_results) {
        if (!frame.contains("result") || !frame["result"].contains("additional_probs")) continue;
        auto& probs = frame["result"]["additional_probs"];
        
        for (const auto& key : emotion_keys) {
            if (probs.contains(key)) {
                try {
                    sums[key] += std::stod(probs[key].get<std::string>());
                    counts[key]++;
                } catch (...) {}
            }
        }
    }

    for (const auto& key : emotion_keys) {
        if (counts[key] > 0) {
            avg_emotions[key] = sums[key] / counts[key];
        }
    }

    return avg_emotions;
}

//=============================================================================
// Image Processing (Core)
//=============================================================================
std::pair<cv::Mat, nlohmann::json> FileProcessor::process_image(const cv::Mat& image)
{
    std::lock_guard<std::mutex> lock(model_mutex_);

    if (!fer_ || !model_loaded_) {
        LOG_WARN("Emotion model not loaded");
        return {image.clone(), {
            {"emotion", "unknown"},
            {"confidence", 0.0},
            {"additional_probs", nlohmann::json::object()}
        }};
    }

    try {
        Image img(image);
        return img.process_image(image, fer_.get());
    } catch (const std::exception& e) {
        LOG_ERROR("Image processing error: {}", e.what());
        return {image.clone(), {
            {"emotion", "error"},
            {"confidence", 0.0},
            {"error", e.what()},
            {"additional_probs", nlohmann::json::object()}
        }};
    }
}

//=============================================================================
// GigaChat Integration
//=============================================================================
emotionai::gigachat::EmotionData FileProcessor::extract_emotions_from_result(const nlohmann::json& result)
{
    emotionai::gigachat::EmotionData emotions = {0};
    
    if (!result.contains("additional_probs")) return emotions;

    auto& probs = result["additional_probs"];
    auto get_float = [&](const std::string& key) -> float {
        if (!probs.contains(key)) return 0.0f;
        auto& val = probs[key];
        if (val.is_string()) {
            try { return std::stof(val.get<std::string>()); }
            catch (...) { return 0.0f; }
        }
        if (val.is_number()) return val.get<float>();
        return 0.0f;
    };

    emotions.anger = get_float("anger");
    emotions.disgust = get_float("disgust");
    emotions.fear = get_float("fear");
    emotions.happiness = get_float("happiness");
    emotions.neutral = get_float("neutral");
    emotions.sadness = get_float("sadness");
    emotions.surprise = get_float("surprise");

    return emotions;
}

void FileProcessor::addGigaChatAnalysis(nlohmann::json& result, const std::string& task_id, int frame_number)
{
    if (!gigachat_client_ || !gigachat_client_->isEnabled()) {
        return;
    }

    try {
        auto emotions = extract_emotions_from_result(result);
        std::string session_id = task_id + (frame_number >= 0 ? "_frame_" + std::to_string(frame_number) : "");

        LOG_INFO("Calling GigaChat: Task={}", task_id);
        std::string analysis = gigachat_client_->analyzeEmotions(emotions, session_id);

        if (!analysis.empty()) {
            try {
                auto json_response = nlohmann::json::parse(analysis);
                if (json_response.contains("verdict") && 
                    json_response.contains("probability") && 
                    json_response.contains("reasoning")) {
                    result["gigachat"] = json_response;
                    LOG_INFO("GigaChat analysis added: Task={}", task_id);
                } else {
                    LOG_WARN("GigaChat response missing fields");
                    result["gigachat"] = analysis;
                }
            } catch (const std::exception& e) {
                LOG_WARN("GigaChat response not JSON: {}", e.what());
                result["gigachat"] = analysis;
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR("GigaChat failed: {}", e.what());
    }
}