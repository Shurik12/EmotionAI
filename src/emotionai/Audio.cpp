#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>

#include <torch/script.h>
#include <common/base64.h>
#include <config/Config.h>
#include <logging/Logger.h>
#include "Audio.h"

#ifdef HAVE_FFMPEG
extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libavutil/avutil.h>
    #include <libswresample/swresample.h>
}
#endif

namespace fs = std::filesystem;

//=============================================================================
// WAV Header
//=============================================================================
struct WavHeader {
    char chunk_id[4];
    uint32_t chunk_size;
    char format[4];
    char subchunk1_id[4];
    uint32_t subchunk1_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char subchunk2_id[4];
    uint32_t subchunk2_size;
};

//=============================================================================
// Static Member Initialization
//=============================================================================
const std::vector<std::string> Audio::EMOTION_LABELS = {
    "anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"
};

//=============================================================================
// Construction / Destruction
//=============================================================================
Audio::Audio(const std::string &filename)
{
    #ifdef HAVE_FFMPEG
    avformat_network_init();
    #endif
    
    LOG_INFO("Loading audio file: {}", filename);
    loaded_ = load_audio_file(filename);
    
    if (loaded_) {
        LOG_INFO("Audio loaded: {} samples, {} Hz, {} channels, {:.2f}s", 
                 audio_data_.size(), sample_rate_, channels_, get_duration());
    } else {
        LOG_ERROR("Failed to load audio: {}", error_);
    }
}

Audio::Audio(const std::vector<float> &audio_data, int sample_rate)
    : audio_data_(audio_data), sample_rate_(sample_rate), channels_(1), loaded_(true)
{
    #ifdef HAVE_FFMPEG
    avformat_network_init();
    #endif
    LOG_INFO("Audio from data: {} samples, {} Hz", audio_data_.size(), sample_rate_);
}

Audio::~Audio() 
{ 
    cleanup_ffmpeg_resources(); 
}

void Audio::cleanup_ffmpeg_resources()
{
    #ifdef HAVE_FFMPEG
    if (swr_ctx_) { 
        swr_free(&swr_ctx_); 
        swr_ctx_ = nullptr; 
    }
    if (codec_ctx_) { 
        avcodec_free_context(&codec_ctx_); 
        codec_ctx_ = nullptr; 
    }
    if (format_ctx_) { 
        avformat_close_input(&format_ctx_); 
        format_ctx_ = nullptr; 
    }
    #endif
}

double Audio::get_duration() const
{
    return (sample_rate_ > 0 && !audio_data_.empty()) 
        ? static_cast<double>(audio_data_.size()) / sample_rate_ 
        : 0.0;
}

//=============================================================================
// File Loading
//=============================================================================
bool Audio::load_audio_file(const std::string &filename)
{
    try {
        if (!fs::exists(filename)) {
            error_ = "File not found: " + filename;
            LOG_ERROR("{}", error_);
            return false;
        }

        std::string ext = fs::path(filename).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".wav") {
            return load_wav_file(filename);
        }

#ifdef HAVE_FFMPEG
        // For non-WAV files, use FFmpeg
        return decode_audio_stream();
#else
        error_ = "Only WAV files supported (FFmpeg not available)";
        LOG_ERROR("{}", error_);
        return false;
#endif
    } catch (const std::exception& e) {
        error_ = std::string("Error loading audio: ") + e.what();
        LOG_ERROR("{}", error_);
        return false;
    }
}

bool Audio::load_wav_file(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        error_ = "Failed to open file";
        LOG_ERROR("{}", error_);
        return false;
    }

    // Get file size for chunk searching
    file.seekg(0, std::ios::end);
    long long file_size = static_cast<long long>(file.tellg());
    file.seekg(0, std::ios::beg);

    WavHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    // Validate WAV
    if (strncmp(header.chunk_id, "RIFF", 4) != 0 || 
        strncmp(header.format, "WAVE", 4) != 0) {
        error_ = "Invalid WAV file";
        LOG_ERROR("{}", error_);
        return false;
    }

    sample_rate_ = header.sample_rate;
    channels_ = header.num_channels;
    int bytes_per_sample = header.bits_per_sample / 8;

    // Find data chunk
    auto [found, data_pos, data_size] = find_data_chunk(file, header);
    if (!found) {
        error_ = "Could not find audio data chunk";
        LOG_ERROR("{}", error_);
        return false;
    }

    // Read audio data
    file.seekg(data_pos, std::ios::beg);
    int num_samples = data_size / (bytes_per_sample * channels_);
    if (num_samples <= 0) {
        error_ = "Invalid audio data size";
        LOG_ERROR("{}", error_);
        return false;
    }

    std::vector<int16_t> int_data(num_samples * channels_);
    file.read(reinterpret_cast<char*>(int_data.data()), data_size);

    size_t bytes_read = file.gcount();
    if (bytes_read == 0) {
        error_ = "Failed to read audio data";
        LOG_ERROR("{}", error_);
        return false;
    }

    // Convert to float [-1, 1]
    int actual_samples = bytes_read / (bytes_per_sample * channels_);
    audio_data_.resize(actual_samples * channels_);
    for (size_t i = 0; i < audio_data_.size(); ++i) {
        audio_data_[i] = int_data[i] / 32768.0f;
    }

    // Convert stereo to mono if needed
    if (channels_ == 2) {
        std::vector<float> mono(audio_data_.size() / 2);
        for (size_t i = 0; i < mono.size(); ++i) {
            mono[i] = (audio_data_[2*i] + audio_data_[2*i+1]) / 2.0f;
        }
        audio_data_ = std::move(mono);
        channels_ = 1;
    }

    LOG_INFO("Loaded WAV: {} samples, {} Hz, {} channels", 
             audio_data_.size(), sample_rate_, channels_);
    
    return !audio_data_.empty();
}

std::tuple<bool, long long, uint32_t> Audio::find_data_chunk(std::ifstream& file, const WavHeader& header)
{
    // Check if header points to data
    if (strncmp(header.subchunk2_id, "data", 4) == 0) {
        return {true, sizeof(WavHeader), header.subchunk2_size};
    }

    // Search for data chunk
    long long fmt_end = 12 + 8 + header.subchunk1_size;
    if (header.subchunk1_size % 2 != 0) {
        fmt_end += 1;
    }
    
    file.seekg(fmt_end, std::ios::beg);
    
    // Get file size
    file.seekg(0, std::ios::end);
    long long file_size = file.tellg();
    file.seekg(fmt_end, std::ios::beg);
    
    while (file.tellg() < file_size - 8) {
        char id[5] = {0};
        uint32_t size;
        
        if (!file.read(id, 4)) break;
        if (!file.read(reinterpret_cast<char*>(&size), 4)) break;
            
        if (std::string(id, 4) == "data") {
            return {true, static_cast<long long>(file.tellg()), size};
        }
        
        long long skip = size;
        if (size % 2 != 0) skip += 1;
        file.seekg(skip, std::ios::cur);
    }
    
    return {false, 0, 0};
}

#ifdef HAVE_FFMPEG
bool Audio::decode_audio_stream()
{
    if (!format_ctx_ || !codec_ctx_) {
        error_ = "Invalid format or codec context";
        LOG_ERROR("{}", error_);
        return false;
    }

    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    if (!packet || !frame) {
        error_ = "Failed to allocate packet or frame";
        LOG_ERROR("{}", error_);
        av_packet_free(&packet);
        av_frame_free(&frame);
        return false;
    }

    std::vector<float> all_samples;

    while (av_read_frame(format_ctx_, packet) >= 0) {
        int response = avcodec_send_packet(codec_ctx_, packet);
        if (response < 0) {
            av_packet_unref(packet);
            continue;
        }

        while (response >= 0) {
            response = avcodec_receive_frame(codec_ctx_, frame);
            if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) {
                break;
            } else if (response < 0) {
                break;
            }

            uint8_t** out_data = nullptr;
            int out_samples = swr_convert(swr_ctx_, out_data, frame->nb_samples,
                                          (const uint8_t**)frame->data, frame->nb_samples);
            
            if (out_samples > 0 && out_data && out_data[0]) {
                float* float_data = reinterpret_cast<float*>(out_data[0]);
                all_samples.insert(all_samples.end(), float_data, float_data + out_samples);
            }
            av_frame_unref(frame);
        }
        av_packet_unref(packet);
    }

    av_packet_free(&packet);
    av_frame_free(&frame);

    if (all_samples.empty()) {
        error_ = "No audio data extracted";
        LOG_ERROR("{}", error_);
        return false;
    }

    audio_data_ = std::move(all_samples);
    return true;
}
#endif

//=============================================================================
// Audio Preprocessing
//=============================================================================
std::vector<float> Audio::resample_audio(int target_sr) const
{
    if (sample_rate_ == target_sr || audio_data_.empty()) {
        return audio_data_;
    }

    float ratio = static_cast<float>(target_sr) / sample_rate_;
    std::vector<float> resampled(static_cast<size_t>(audio_data_.size() * ratio));

    for (size_t i = 0; i < resampled.size(); ++i) {
        float pos = i / ratio;
        size_t idx = static_cast<size_t>(pos);
        float frac = pos - idx;
        
        if (idx < audio_data_.size() - 1) {
            resampled[i] = audio_data_[idx] * (1 - frac) + audio_data_[idx + 1] * frac;
        } else if (idx < audio_data_.size()) {
            resampled[i] = audio_data_[idx];
        }
    }
    
    return resampled;
}

//=============================================================================
// Main Processing - Wav2Vec2
//=============================================================================
nlohmann::json Audio::process_audio(torch::jit::Module* audio_model)
{
    nlohmann::json result;
    
    try {
        if (!audio_model) {
            throw std::runtime_error("Audio model not provided");
        }
        
        if (!loaded_ || audio_data_.empty()) {
            throw std::runtime_error("Audio not loaded: " + error_);
        }

        LOG_INFO("Processing audio with Wav2Vec2: {} samples, {} Hz, {:.2f}s", 
                 audio_data_.size(), sample_rate_, get_duration());

        // Resample to 16kHz (Wav2Vec2 requirement)
        std::vector<float> audio = resample_audio(TARGET_SR);
        
        // Check duration constraints
        double duration = static_cast<double>(audio.size()) / TARGET_SR;
        if (duration > MAX_DURATION) {
            LOG_WARN("Audio too long ({}s), truncating to {}s", duration, MAX_DURATION);
            audio.resize(TARGET_SR * MAX_DURATION);
        }
        
        if (duration < MIN_DURATION) {
            LOG_WARN("Audio too short ({}s), padding to {}s", duration, MIN_DURATION);
            audio.resize(TARGET_SR * MIN_DURATION, 0.0f);
        }

        // Create tensor [1, sequence_length] - Wav2Vec2 expects raw waveform
        torch::Tensor input_tensor = torch::from_blob(
            audio.data(), 
            {1, static_cast<int64_t>(audio.size())}, 
            torch::kFloat
        ).clone();

        LOG_INFO("Running Wav2Vec2 inference on {} samples ({:.2f}s)", 
                 audio.size(), static_cast<double>(audio.size()) / TARGET_SR);

        // Run inference
        audio_model->eval();
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        torch::Tensor output = audio_model->forward(inputs).toTensor();
        
        // Output shape: [1, 7] for 7 emotions
        auto scores = torch::softmax(output, 1).squeeze().to(torch::kCPU).to(torch::kFloat);
        int predicted_class = output.argmax(1).item<int>();
        
        // Convert to vector
        std::vector<float> probs(scores.data_ptr<float>(), 
                                 scores.data_ptr<float>() + scores.numel());

        // Get emotion label
        std::string emotion = EMOTION_LABELS[predicted_class];
        
        // Build result in the same format as original code
        result["main_prediction"] = {
            {"index", predicted_class},
            {"label", emotion},
            {"probability", probs[predicted_class]}
        };

        // All probabilities
        nlohmann::json probs_json;
        for (size_t i = 0; i < probs.size() && i < EMOTION_LABELS.size(); ++i) {
            probs_json[EMOTION_LABELS[i]] = fmt::format("{:.4f}", probs[i]);
        }
        result["additional_probs"] = probs_json;
        
        // Add metadata
        result["model"] = "wav2vec2-emotion-recognition";
        result["duration_seconds"] = static_cast<double>(audio.size()) / TARGET_SR;
        result["sample_rate"] = TARGET_SR;

        LOG_INFO("Wav2Vec2 predicted: {} ({:.1f}%)", emotion, probs[predicted_class] * 100);

    } catch (const c10::Error& e) {
        LOG_ERROR("LibTorch error in Wav2Vec2: {}", e.what());
        result["error"] = std::string("LibTorch error: ") + e.what();
        result["error_type"] = "libtorch";
    } catch (const std::exception& e) {
        LOG_ERROR("Wav2Vec2 processing error: {}", e.what());
        result["error"] = e.what();
        result["error_type"] = "std_exception";
    }
    
    return result;
}

//=============================================================================
// MIME Bundle
//=============================================================================
nlohmann::json Audio::mime_bundle_repr() const
{
    nlohmann::json bundle;
    try {
        std::string audio_str(reinterpret_cast<const char*>(audio_data_.data()), 
                              audio_data_.size() * sizeof(float));
        bundle["audio/raw"] = base64_encode(audio_str);
        bundle["sample_rate"] = sample_rate_;
        bundle["channels"] = channels_;
        bundle["duration"] = get_duration();
    } catch (const std::exception& e) {
        LOG_ERROR("mime_bundle_repr error: {}", e.what());
    }
    return bundle;
}