#pragma once

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <torch/script.h>
#include <fstream>

// Forward declarations for FFmpeg
struct AVFormatContext;
struct AVCodecContext;
struct SwrContext;

// Forward declaration for WavHeader
struct WavHeader;

class Audio
{
public:
    // Constructor from filename
    explicit Audio(const std::string &filename);
    
    // Constructor from audio data
    Audio(const std::vector<float> &audio_data, int sample_rate);
    
    ~Audio();

    // Getters
    const std::vector<float>& get_audio_data() const { return audio_data_; }
    int get_sample_rate() const { return sample_rate_; }
    int get_channels() const { return channels_; }
    double get_duration() const;
    bool is_loaded() const { return loaded_; }
    const std::string& get_error() const { return error_; }

    // Process audio with PyTorch model
    nlohmann::json process_audio(torch::jit::Module* audio_model);
    
    // Get MIME bundle representation
    nlohmann::json mime_bundle_repr() const;

private:
    // Core data
    std::vector<float> audio_data_;
    int sample_rate_ = 0;
    int channels_ = 0;
    bool loaded_ = false;
    std::string error_;
    
    // FFmpeg resources
    AVFormatContext* format_ctx_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    SwrContext* swr_ctx_ = nullptr;
    
    // Helper methods
    bool load_audio_file(const std::string &filename);
    bool load_wav_file(const std::string &filename);
    bool decode_audio_stream();
    void cleanup_ffmpeg_resources();
    
    // WAV parsing helper
    std::tuple<bool, long long, uint32_t> find_data_chunk(std::ifstream& file, const WavHeader& header);
    
    // Feature extraction
    std::vector<float> resample_audio(int target_sr) const;
    std::vector<float> extract_mel_spectrogram() const;
    
    // Emotion mapping
    std::string class_to_emotion(int class_id) const;
    
    // Constants
    static constexpr int TARGET_SR = 16000;
    static constexpr int MAX_DURATION = 30;
    static constexpr int N_MELS = 128;
    static constexpr int N_FFT = 400;
    static constexpr int HOP_LENGTH = 160;
    static constexpr int TIME_STEPS = 3000;
};