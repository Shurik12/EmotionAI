#include <filesystem>
#include <chrono>
#include <thread>
#include <numeric>
#include <atomic>

#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <db/DragonflyManager.h>
#include <emotionai/Image.h>
#include <emotionai/Audio.h>
#include <config/Config.h>
#include <logging/Logger.h>
#include <db/TaskManager.h>
#include <server/ThreadPool.h>
#include <audio/LibrosaFeatureExtractor.h>
#include <audio/BurnoutAnalyzer.h>
#include <audio/ExternalInfluenceAnalyzer.h>
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
                             std::shared_ptr<FileStorage> file_storage,
                             ThreadPool* thread_pool)
    : dragonfly_manager_(std::move(dragonfly_manager))
    , file_storage_(std::move(file_storage))
    , thread_pool_(thread_pool)
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
                int pool_size = Config::instance().video().pool_size;
                if (pool_size < 1) pool_size = 1;
                
                model_pool_.reserve(pool_size);
                for (int i = 0; i < pool_size; ++i) {
                    auto instance = EmotiEffLib::EmotiEffLibRecognizer::createInstance(backend, path);
                    model_pool_.push_back(std::move(instance));
                }
                model_loaded_ = true;
                LOG_INFO("Image emotion model loaded: {} (pool of {})", path, pool_size);
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
        // A configured path may be empty or point at a directory (e.g. if the
        // key is missing from config.yaml the fallback becomes "./models/"),
        // which torch::jit::load cannot read. Only try regular files.
        if (path.empty() || !fs::is_regular_file(path)) {
            continue;
        }
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
// Audio Processing with Burnout
//=============================================================================
nlohmann::json FileProcessor::process_audio_with_burnout(
    const std::string& task_id,
    const std::string& filepath,
    const std::string& filename,
    const nlohmann::json& baseline)
{
    LOG_INFO("Processing audio with burnout: Task={}, File={}", task_id, filename);
    
    if (!audio_model_loaded_ || !audio_torch_model_) {
        throw std::runtime_error("Audio model not loaded");
    }
    
    Audio audio(filepath);
    if (!audio.is_loaded()) {
        throw std::runtime_error("Failed to load audio: " + audio.get_error());
    }
    
    LOG_INFO("Audio loaded: {:.2f}s, {}Hz, {} channels", 
             audio.get_duration(), audio.get_sample_rate(), audio.get_channels());
    
    // Process with burnout (reuses process_audio internally)
    auto result = audio.process_audio_with_burnout(audio_torch_model_.get(), baseline);
    
    // Save result
    save_json_to_storage(result, task_id, "audio_burnout");
    
    LOG_INFO("Audio burnout processing complete: Task={}", task_id);
    
    return {
        {"type", "audio_burnout"},
        {"filename", filename},
        {"duration", audio.get_duration()},
        {"sample_rate", audio.get_sample_rate()},
        {"storage_path", fmt::format("results/audio_burnout_{}.json", task_id)},
        {"result", result}
    };
}

audio::Result FileProcessor::analyze_burnout_from_result(
    const nlohmann::json& emotion_result,
    const nlohmann::json& baseline)
{
    // Use the static method from Audio class
    auto result_with_burnout = Audio::add_burnout_analysis(emotion_result, baseline);
    
    // Extract burnout result
    audio::Result burnout_result;
    
    if (result_with_burnout.contains("burnout_analysis")) {
        const auto& data = result_with_burnout["burnout_analysis"];
        
        burnout_result.state = audio::stringToState(data.value("state", "INSUFFICIENT_DATA"));
        burnout_result.level = audio::stringToLevel(data.value("level", "low"));
        burnout_result.risk = data.value("risk", 0.0);
        burnout_result.score = data.value("score", 0.0);
        burnout_result.confidence = data.value("confidence", 0.0);
        burnout_result.top_factor = data.value("top_factor", "");
        burnout_result.comment = data.value("comment", "");
        
        if (data.contains("components") && data["components"].is_object()) {
            for (auto& [key, value] : data["components"].items()) {
                if (value.is_number()) {
                    burnout_result.components[key] = value.get<double>();
                }
            }
        }
        
        if (data.contains("recommendations") && data["recommendations"].is_array()) {
            for (const auto& rec : data["recommendations"]) {
                if (rec.is_string()) {
                    burnout_result.recommendations.push_back(rec.get<std::string>());
                }
            }
        }
        
        if (data.contains("error") && data["error"].is_string()) {
            burnout_result.error = data["error"].get<std::string>();
        }
        
        return burnout_result;
    }
    
    if (result_with_burnout.contains("burnout_error")) {
        burnout_result.error = result_with_burnout["burnout_error"].get<std::string>();
    }
    
    return burnout_result;
}

void FileProcessor::save_user_baseline(
    const std::string& user_id,
    const nlohmann::json& baseline)
{
    if (!dragonfly_manager_) {
        LOG_ERROR("DragonflyManager not available for saving baseline");
        return;
    }
    
    std::string key = "baseline:user:" + user_id;
    dragonfly_manager_->set_task_status(key, baseline);
    LOG_INFO("Saved baseline for user: {}", user_id);
}

nlohmann::json FileProcessor::get_user_baseline(const std::string& user_id)
{
    if (!dragonfly_manager_) {
        LOG_ERROR("DragonflyManager not available for getting baseline");
        return nlohmann::json::object();
    }
    
    std::string key = "baseline:user:" + user_id;
    auto baseline = dragonfly_manager_->get_task_status_json(key);
    
    if (baseline) {
        LOG_INFO("Retrieved baseline for user: {}", user_id);
        return *baseline;
    }
    
    LOG_WARN("No baseline found for user: {}", user_id);
    return nlohmann::json::object();
}

//=============================================================================
// External influence (scam signal) processing
//
// The WavLM model sees at most 10 s at once, so a call recording is split
// into consecutive windows (config fragment_seconds) in the model domain
// (16 kHz). Each window gets its own emotion inference and acoustic feature
// set; the ExternalInfluenceAnalyzer then scores the fragments against the
// user baseline (or the built-in default) and produces the pilot status.
//=============================================================================
namespace {

// WavLM input domain (matches Audio::TARGET_SR, which is private)
constexpr int EI_MODEL_SAMPLE_RATE = 16000;

// Linear resample (mirrors Audio::resample_audio) - features and the model
// are calibrated on 16 kHz audio, uploaded WAVs may carry another rate.
std::vector<float> resampleToModelRate(const std::vector<float>& audio, int sample_rate)
{
    if (sample_rate <= 0 || sample_rate == EI_MODEL_SAMPLE_RATE || audio.empty()) {
        return audio;
    }

    float ratio = static_cast<float>(EI_MODEL_SAMPLE_RATE) / sample_rate;
    std::vector<float> resampled(static_cast<size_t>(audio.size() * ratio));

    for (size_t i = 0; i < resampled.size(); ++i) {
        float pos = i / ratio;
        size_t idx = static_cast<size_t>(pos);
        float frac = pos - idx;

        if (idx < audio.size() - 1) {
            resampled[i] = audio[idx] * (1 - frac) + audio[idx + 1] * frac;
        } else if (idx < audio.size()) {
            resampled[i] = audio[idx];
        }
    }

    return resampled;
}

struct AudioFragment {
    std::vector<float> samples;  // 16 kHz PCM
    double start_time = 0.0;     // seconds from the call start
};

// Split 16 kHz PCM into consecutive fragment_seconds windows. A trailing
// window shorter than min_tail_ratio * fragment_seconds is dropped (it does
// not carry enough speech for a meaningful window); at most max_fragments
// windows are produced.
std::vector<AudioFragment> splitCallFragments(
    const std::vector<float>& audio,
    double fragment_seconds,
    double min_tail_ratio,
    int max_fragments)
{
    std::vector<AudioFragment> fragments;
    if (audio.empty() || fragment_seconds <= 0.0 || max_fragments <= 0) {
        return fragments;
    }

    const size_t window_samples = static_cast<size_t>(
        std::llround(fragment_seconds * EI_MODEL_SAMPLE_RATE));
    if (window_samples == 0) {
        return fragments;
    }

    size_t offset = 0;
    while (offset < audio.size() && fragments.size() < static_cast<size_t>(max_fragments)) {
        size_t take = std::min(window_samples, audio.size() - offset);

        double tail_ratio = static_cast<double>(take) / window_samples;
        if (take < window_samples && tail_ratio < min_tail_ratio) {
            break;  // trailing stub is too short to be a fragment
        }

        AudioFragment fragment;
        fragment.samples.assign(audio.begin() + static_cast<long>(offset),
                                audio.begin() + static_cast<long>(offset + take));
        fragment.start_time = static_cast<double>(offset) / EI_MODEL_SAMPLE_RATE;
        fragments.push_back(std::move(fragment));
        offset += take;
    }

    return fragments;
}

constexpr int EI_INFERENCE_BATCH_SIZE = 4;  // windows per WavLM forward

// One batched WavLM forward over full-length windows. Returns, per window,
// the same emotion JSON shape Audio::process_audio produces. The rest of the
// codebase only exercises the scripted model with batch 1, so callers must
// fall back to per-window inference on any exception.
std::vector<nlohmann::json> runWavlmBatch(
    torch::jit::Module* model,
    const std::vector<const AudioFragment*>& windows,
    size_t full_window_samples)
{
    std::vector<nlohmann::json> results(windows.size());
    if (windows.empty()) {
        return results;
    }

    const int64_t batch = static_cast<int64_t>(windows.size());
    torch::Tensor input = torch::zeros({batch, static_cast<int64_t>(full_window_samples)});
    {
        auto acc = input.accessor<float, 2>();
        for (int64_t b = 0; b < batch; ++b) {
            const auto& samples = windows[static_cast<size_t>(b)]->samples;
            for (size_t j = 0; j < samples.size(); ++j) {
                acc[b][static_cast<int64_t>(j)] = samples[j];
            }
        }
    }

    model->eval();
    torch::NoGradGuard no_grad;
    torch::Tensor output = model->forward({input}).toTensor().contiguous();
    if (output.dim() != 2 || output.size(0) != batch) {
        throw std::runtime_error(fmt::format(
            "unexpected batched output shape: {} x {}", output.size(0), output.size(1)));
    }

    const int64_t n_classes = output.size(1);
    const auto& labels = Audio::emotion_labels();
    const torch::Tensor scores = torch::softmax(output, 1);
    const torch::Tensor argmax = output.argmax(1);
    const auto s_acc = scores.accessor<float, 2>();
    const auto a_acc = argmax.accessor<int64_t, 1>();

    for (int64_t b = 0; b < batch; ++b) {
        const int64_t predicted = a_acc[b];
        nlohmann::json probs_json;
        for (int64_t c = 0; c < n_classes && c < static_cast<int64_t>(labels.size()); ++c) {
            probs_json[labels[static_cast<size_t>(c)]] = fmt::format("{:.4f}", s_acc[b][c]);
        }
        const std::string label = (predicted >= 0 && predicted < static_cast<int64_t>(labels.size()))
            ? labels[static_cast<size_t>(predicted)] : "unknown";
        results[static_cast<size_t>(b)] = {
            {"main_prediction", {
                {"index", predicted},
                {"label", label},
                {"probability", (predicted >= 0 && predicted < n_classes) ? s_acc[b][predicted] : 0.0}
            }},
            {"additional_probs", std::move(probs_json)},
            {"model", "wavlm-emotion-russian-resd"},
            {"duration_seconds", static_cast<double>(full_window_samples) / EI_MODEL_SAMPLE_RATE},
            {"sample_rate", EI_MODEL_SAMPLE_RATE}
        };
    }
    return results;
}

} // namespace

nlohmann::json FileProcessor::process_external_influence(
    const std::string& task_id,
    const std::string& filepath,
    const std::string& filename,
    const nlohmann::json& baseline)
{
    LOG_INFO("Processing external influence: Task={}, File={}", task_id, filename);

    if (!audio_model_loaded_ || !audio_torch_model_) {
        throw std::runtime_error("Audio model not loaded");
    }

    const audio::ExternalInfluenceConfig ei_cfg = Config::instance().externalInfluence();

    Audio audio(filepath);
    if (!audio.is_loaded()) {
        throw std::runtime_error("Failed to load audio: " + audio.get_error());
    }

    LOG_INFO("Audio loaded: {:.2f}s, {}Hz, {} channels",
             audio.get_duration(), audio.get_sample_rate(), audio.get_channels());

    // Live progress: the Server posts the initial status, this task patches
    // it after each inference batch so long recordings do not look stuck.
    const auto report_progress = [&task_id](int percent) {
        try {
            auto& task_manager = TaskManager::instance();
            auto current = task_manager.get_task_status(task_id);
            if (!current) {
                return;
            }
            (*current)["progress"] = percent;
            task_manager.set_task_status(task_id, *current);
        } catch (const std::exception& e) {
            LOG_WARN("External influence: progress update failed: {}", e.what());
        }
    };

    // Only the first max_fragments x fragment_seconds of the call are ever
    // analyzed, so slice to that span BEFORE resampling: work and memory stay
    // bounded no matter how long the recording is. Fragmenting, per-window
    // features and the quality estimate all share the 16 kHz model domain.
    const size_t full_window_samples = static_cast<size_t>(
        std::llround(ei_cfg.fragment_seconds * EI_MODEL_SAMPLE_RATE));
    const double span_seconds = ei_cfg.fragment_seconds * ei_cfg.max_fragments;
    const auto& decoded = audio.get_audio_data();
    const int native_sr = audio.get_sample_rate();
    const size_t span_native = native_sr > 0
        ? static_cast<size_t>(std::llround(span_seconds * native_sr))
        : decoded.size();
    const size_t take_native = std::min(decoded.size(), span_native);
    const std::vector<float> pcm = resampleToModelRate(
        std::vector<float>(decoded.begin(), decoded.begin() + static_cast<long>(take_native)),
        native_sr);
    if (pcm.empty()) {
        throw std::runtime_error("Audio decoded to empty data");
    }

    audio::LibrosaFeatureExtractor::Config feature_config;
    feature_config.sample_rate = EI_MODEL_SAMPLE_RATE;

    // Quality gate FIRST: garbage/silent input used to pay for every WavLM
    // run before being rejected. The estimator now only scans the analyzed
    // span, so a long silent tail cannot sink an otherwise valid call.
    const double audio_quality =
        audio::LibrosaFeatureExtractor::estimateAudioQuality(pcm, feature_config);

    auto fragments = splitCallFragments(pcm, ei_cfg.fragment_seconds,
                                        ei_cfg.min_tail_ratio, ei_cfg.max_fragments);
    LOG_INFO("External influence: call split into {} fragment(s) of {:.1f}s, "
             "audio quality {:.2f}",
             fragments.size(), ei_cfg.fragment_seconds, audio_quality);

    nlohmann::json fragment_records = nlohmann::json::array();
    if (audio_quality < ei_cfg.audio_quality_min ||
        fragments.size() < static_cast<size_t>(ei_cfg.min_valid_fragments)) {
        LOG_INFO("External influence: skipping inference (quality {:.2f} < {:.2f} or {} "
                 "fragments < {}); result will be INSUFFICIENT_DATA",
                 audio_quality, ei_cfg.audio_quality_min, fragments.size(),
                 ei_cfg.min_valid_fragments);
        report_progress(95);
    } else {
        const size_t n_full = static_cast<size_t>(std::count_if(
            fragments.begin(), fragments.end(),
            [full_window_samples](const AudioFragment& f) {
                return f.samples.size() == full_window_samples;
            }));

        // Per-window inference + features; windows are recorded raw so the
        // status can be re-derived later with any baseline/context flags.
        // Full 10 s windows (all but a possible trailing window) run in
        // batches: one WavLM forward is ~B x cheaper than B sequential ones.
        // The scripted model is only proven on batch 1, so any failure here
        // falls back to the sequential per-window path for the whole file.
        const auto inference_start = std::chrono::steady_clock::now();
        int batched_windows = 0;
        int sequential_windows = 0;
        std::vector<nlohmann::json> emotions(fragments.size());
        if (n_full >= 2) {
            try {
                std::lock_guard<std::mutex> lock(audio_mutex_);
                for (size_t start = 0; start < n_full; start += EI_INFERENCE_BATCH_SIZE) {
                    const size_t end = std::min(n_full, start + EI_INFERENCE_BATCH_SIZE);
                    std::vector<const AudioFragment*> batch_windows;
                    batch_windows.reserve(end - start);
                    for (size_t i = start; i < end; ++i) {
                        batch_windows.push_back(&fragments[i]);
                    }
                    const auto batch_results =
                        runWavlmBatch(audio_torch_model_.get(), batch_windows,
                                      full_window_samples);
                    for (size_t r = 0; r < batch_results.size(); ++r) {
                        emotions[start + r] = batch_results[r];
                    }
                    batched_windows += static_cast<int>(batch_results.size());
                    report_progress(15 + static_cast<int>(80.0 * end / fragments.size()));
                }
            } catch (const std::exception& e) {
                LOG_WARN("External influence: batched inference failed ({}); "
                         "falling back to per-window inference", e.what());
                std::fill(emotions.begin(), emotions.end(), nlohmann::json());
            }
        }

        for (size_t i = 0; i < fragments.size(); ++i) {
            const auto& fragment = fragments[i];

            nlohmann::json fragment_record = {
                {"index", static_cast<int>(i)},
                {"start_time", fragment.start_time},
                {"duration_seconds", static_cast<double>(fragment.samples.size()) / EI_MODEL_SAMPLE_RATE}
            };

            nlohmann::json emotion = emotions[i];
            if (emotion.empty()) {
                ++sequential_windows;
                std::lock_guard<std::mutex> lock(audio_mutex_);
                Audio fragment_audio(fragment.samples, EI_MODEL_SAMPLE_RATE);
                emotion = fragment_audio.process_audio(audio_torch_model_.get());
            }

            if (emotion.contains("error")) {
                LOG_WARN("External influence: fragment {} inference failed: {}", i, emotion["error"]);
                fragment_record["error"] = emotion["error"];
                fragment_records.push_back(std::move(fragment_record));
                continue;
            }

            audio::AcousticFeatures features =
                audio::LibrosaFeatureExtractor::extractAcousticFeaturesOnly(fragment.samples, feature_config);
            nlohmann::json features_json = features.toJson();
            features_json.erase("mfcc");
            features_json.erase("mel_spectrogram");

            const auto& main_prediction = emotion.value("main_prediction", nlohmann::json::object());

            fragment_record["model_probability"] = main_prediction.value("probability", 0.0);
            fragment_record["additional_probs"] = emotion["additional_probs"];
            fragment_record["acoustic_features"] = std::move(features_json);
            fragment_records.push_back(std::move(fragment_record));

            report_progress(15 + static_cast<int>(
                80.0 * (i + 1) / fragments.size()));
        }

        const double inference_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - inference_start).count();
        LOG_INFO("External influence: {} window(s) analyzed via {} batched + {} sequential "
                 "inference call(s), inference took {:.0f} ms",
                 fragments.size(), batched_windows, sequential_windows, inference_ms);
    }

    audio::ExternalInfluenceAnalyzer analyzer(ei_cfg);
    const audio::ExternalInfluenceResult result =
        analyzer.analyze(fragment_records, baseline, audio_quality, {});

    nlohmann::json envelope = {
        {"type", "audio_external_influence"},
        {"filename", filename},
        {"duration", audio.get_duration()},
        {"sample_rate", audio.get_sample_rate()},
        {"storage_path", fmt::format("results/external_influence_{}.json", task_id)},
        {"fragment_seconds", ei_cfg.fragment_seconds},
        {"fragments", fragment_records},
        {"audio_quality", audio_quality},
        {"result", result.toJson()},
        {"diagnostics", result.diagnosticsToJson()}
    };

    save_json_to_storage(envelope, task_id, "external_influence");

    LOG_INFO("External influence processing complete: Task={}, status={}, score={:.2f}",
             task_id, audio::statusToString(result.status), result.score);

    return envelope;
}

nlohmann::json FileProcessor::analyze_external_influence(
    const nlohmann::json& analysis_data,
    const nlohmann::json& baseline,
    const std::vector<std::string>& context_flags)
{
    if (!analysis_data.contains("fragments") || !analysis_data["fragments"].is_array()) {
        throw std::runtime_error("Analysis data has no fragments array");
    }

    const double audio_quality = analysis_data.value("audio_quality", 0.0);

    const audio::ExternalInfluenceConfig ei_cfg = Config::instance().externalInfluence();
    audio::ExternalInfluenceAnalyzer analyzer(ei_cfg);
    const audio::ExternalInfluenceResult result =
        analyzer.analyze(analysis_data["fragments"], baseline, audio_quality, context_flags);

    nlohmann::json response = result.toJson();
    response["diagnostics"] = result.diagnosticsToJson();
    return response;
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

    return {
        {"type", "image"},
        {"image_url", file_storage_->getFileUrl(storage_path)},
        {"storage_path", storage_path},
        {"result", emotion_result}
    };
}

// In-memory single-frame inference for the Focus camera session (EMO-17):
// no storage write, no task status — just decode + score.
nlohmann::json FileProcessor::process_image_frame(const std::vector<uint8_t>& bytes)
{
    if (bytes.empty()) {
        throw std::runtime_error("Empty image frame");
    }

    cv::Mat image = cv::imdecode(cv::Mat(bytes), cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Could not decode image frame");
    }

    auto [processed_image, emotion_result] = process_image(image);
    (void)processed_image;
    return emotion_result;
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

    // Extract one frame every N seconds (configurable via video.frame_interval_seconds)
    double interval_sec = Config::instance().video().frame_interval_seconds;
    int frame_interval = std::max(1, static_cast<int>(fps * interval_sec));

    LOG_INFO("Frame extraction: interval={} frames ({:.1f}s), total_frames={}", 
             frame_interval, interval_sec, total_frames);

    // Collect target frame numbers (seek directly, don't iterate all frames)
    std::vector<int> target_frames;
    for (int frame_num = 0; 
         frame_num < total_frames && target_frames.size() < MAX_VIDEO_FRAMES; 
         frame_num += frame_interval) {
        target_frames.push_back(frame_num);
    }
    // Always include the last frame if not already selected
    if (!target_frames.empty() && target_frames.back() != total_frames - 1) {
        target_frames.push_back(total_frames - 1);
    }

    LOG_INFO("Target frames to process: {} (interval={} frames)", 
             target_frames.size(), frame_interval);

    // Process frames with pipelining: read sequentially, infer in parallel via ThreadPool
    struct AsyncResult {
        int frame_number;
        double timestamp;
        std::string storage_path;
        nlohmann::json result;
    };

    std::vector<AsyncResult> async_results(target_frames.size());
    std::atomic<int> tasks_pending{(int)target_frames.size()};
    std::mutex collect_mutex;
    std::condition_variable collect_cv;

    for (size_t i = 0; i < target_frames.size(); ++i) {
        int target = target_frames[i];
        cap.set(cv::CAP_PROP_POS_FRAMES, target);
        cv::Mat frame;
        if (!cap.read(frame)) break;

        auto* result_ptr = &async_results[i];

        thread_pool_->enqueue([this, frame = std::move(frame), fn = target, 
                               ts = target / fps, &task_id, result_ptr,
                               &tasks_pending, &collect_mutex, &collect_cv]() {
            // Save frame to storage
            auto image_data = mat_to_vector(frame);
            std::string storage_path = fmt::format("results/frame_{}_{}.jpg", task_id, fn);
            file_storage_->saveFile(image_data, storage_path);

            // Process image with model from pool (no mutex needed)
            auto* model = acquire_model();
            auto [processed_frame, emotion_result] = process_image(frame, model);
            (void)processed_frame;

            result_ptr->frame_number = fn;
            result_ptr->timestamp = ts;
            result_ptr->storage_path = std::move(storage_path);
            result_ptr->result = std::move(emotion_result);

            if (--tasks_pending == 0) {
                std::lock_guard<std::mutex> lock(collect_mutex);
                collect_cv.notify_one();
            }
        });
    }

    cap.release();

    // Wait for all tasks to complete
    {
        std::unique_lock<std::mutex> lock(collect_mutex);
        collect_cv.wait(lock, [&tasks_pending]() { return tasks_pending == 0; });
    }

    // Build results in frame order
    int processed_count = 0;
    nlohmann::json results = nlohmann::json::array();

    for (auto& ar : async_results) {
        results.push_back({
            {"frame", ar.frame_number},
            {"timestamp", ar.timestamp},
            {"image_url", file_storage_->getFileUrl(ar.storage_path)},
            {"result", ar.result}
        });
        processed_count++;
    }

    LOG_INFO("Processed {} frames (parallel, pool={})", processed_count, model_pool_.size());

    if (results.empty()) {
        throw std::runtime_error("No frames processed");
    }

    // Compute average emotions across all processed frames
    nlohmann::json avg_emotions = calculate_average_emotions(results);
    
    // Find the emotion with the highest average probability -> average main emotion
    std::string avg_main_label = "neutral";
    double avg_main_prob = 0.0;
    for (auto& [key, val] : avg_emotions.items()) {
        double prob = val.get<double>();
        if (prob > avg_main_prob) {
            avg_main_prob = prob;
            avg_main_label = key;
        }
    }

    return {
        {"type", "video"},
        {"frames_processed", processed_count},
        {"results", results},
        {"average_emotions", avg_emotions},
        {"average_main_emotion", {
            {"label", avg_main_label},
            {"probability", avg_main_prob}
        }},
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

    return {
        .frame = processed_frame,
        .result = result,
        .storage_path = storage_path,
        .frame_number = frame_number,
        .timestamp = frame_number / fps
    };
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
    std::vector<std::string> emotion_keys = {"anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"};
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
EmotiEffLib::EmotiEffLibRecognizer* FileProcessor::acquire_model()
{
    if (model_pool_.empty()) return nullptr;
    size_t idx = model_pool_idx_.fetch_add(1, std::memory_order_relaxed) % model_pool_.size();
    return model_pool_[idx].get();
}

std::pair<cv::Mat, nlohmann::json> FileProcessor::process_image(const cv::Mat& image, EmotiEffLib::EmotiEffLibRecognizer* model)
{
    // Use provided model or acquire one from the pool
    EmotiEffLib::EmotiEffLibRecognizer* active_model = model ? model : acquire_model();
    
    if (!active_model || !model_loaded_) {
        LOG_WARN("Emotion model not loaded");
        return {image.clone(), {
            {"emotion", "unknown"},
            {"confidence", 0.0},
            {"additional_probs", nlohmann::json::object()}
        }};
    }

    try {
        Image img(image);
        return img.process_image(image, active_model);
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
