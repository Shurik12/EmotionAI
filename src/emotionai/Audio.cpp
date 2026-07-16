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

#include <fftw3.h>

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

Audio::~Audio() { cleanup_ffmpeg_resources(); }

void Audio::cleanup_ffmpeg_resources()
{
    #ifdef HAVE_FFMPEG
    if (swr_ctx_) { swr_free(&swr_ctx_); swr_ctx_ = nullptr; }
    if (codec_ctx_) { avcodec_free_context(&codec_ctx_); codec_ctx_ = nullptr; }
    if (format_ctx_) { avformat_close_input(&format_ctx_); format_ctx_ = nullptr; }
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
        LOG_ERROR("FFmpeg support not implemented for: {}", ext);
        return false;
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

    // Convert stereo to mono
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
// Audio Processing
//=============================================================================
std::vector<float> Audio::resample_audio(int target_sr) const
{
    if (sample_rate_ == target_sr || audio_data_.empty())
        return audio_data_;

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

std::vector<float> Audio::extract_mel_spectrogram() const
{
    if (audio_data_.empty()) {
        LOG_WARN("Audio data is empty, returning empty mel spectrogram");
        return std::vector<float>(N_MELS * TIME_STEPS, 0.0f);
    }

    try {
        // Resample and pad/truncate
        std::vector<float> audio = resample_audio(TARGET_SR);
        if (audio.size() < static_cast<size_t>(TARGET_SR * MAX_DURATION)) {
            audio.resize(TARGET_SR * MAX_DURATION, 0.0f);
        } else {
            audio.resize(TARGET_SR * MAX_DURATION);
        }

        // Compute STFT
        int num_frames = (audio.size() - N_FFT) / HOP_LENGTH + 1;
        int n_freq_bins = N_FFT / 2 + 1;

        auto* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n_freq_bins);
        auto* in = (double*)fftw_malloc(sizeof(double) * N_FFT);
        auto plan = fftw_plan_dft_r2c_1d(N_FFT, in, out, FFTW_ESTIMATE);

        // Magnitude spectrogram
        std::vector<std::vector<float>> mag_spec(
            std::min(num_frames, TIME_STEPS), 
            std::vector<float>(n_freq_bins, 0.0f)
        );

        for (int frame = 0; frame < (int)mag_spec.size(); ++frame) {
            int start = frame * HOP_LENGTH;
            
            for (int i = 0; i < N_FFT; ++i) {
                double window = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (N_FFT - 1)));
                in[i] = (start + i < (int)audio.size()) ? audio[start + i] * window : 0.0;
            }
            
            fftw_execute(plan);
            
            for (int i = 0; i < n_freq_bins; ++i) {
                mag_spec[frame][i] = std::sqrt(out[i][0] * out[i][0] + out[i][1] * out[i][1]);
            }
        }

        fftw_destroy_plan(plan);
        fftw_free(in);
        fftw_free(out);

        // Mel filterbank
        auto hz_to_mel = [](float hz) { return 2595.0f * std::log10(1.0f + hz / 700.0f); };
        auto mel_to_hz = [](float mel) { return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f); };

        float min_mel = hz_to_mel(0.0f);
        float max_mel = hz_to_mel(TARGET_SR / 2.0f);
        
        std::vector<float> mel_points(N_MELS + 2);
        for (int i = 0; i < N_MELS + 2; ++i) {
            mel_points[i] = mel_to_hz(min_mel + (max_mel - min_mel) * i / (N_MELS + 1));
        }

        std::vector<std::vector<float>> mel_filters(N_MELS, std::vector<float>(n_freq_bins, 0.0f));
        
        for (int m = 0; m < N_MELS; ++m) {
            for (int k = 0; k < n_freq_bins; ++k) {
                float freq = (float)k * TARGET_SR / N_FFT;
                if (freq >= mel_points[m] && freq < mel_points[m + 1]) {
                    mel_filters[m][k] = (freq - mel_points[m]) / (mel_points[m + 1] - mel_points[m]);
                } else if (freq >= mel_points[m + 1] && freq <= mel_points[m + 2]) {
                    mel_filters[m][k] = (mel_points[m + 2] - freq) / (mel_points[m + 2] - mel_points[m + 1]);
                }
            }
        }

        // Apply mel filters
        std::vector<float> mel_spec(N_MELS * TIME_STEPS, 0.0f);
        
        for (int frame = 0; frame < (int)mag_spec.size(); ++frame) {
            for (int m = 0; m < N_MELS; ++m) {
                float energy = 0.0f;
                for (int k = 0; k < n_freq_bins; ++k) {
                    energy += mag_spec[frame][k] * mel_filters[m][k];
                }
                mel_spec[m * TIME_STEPS + frame] = std::log(std::max(energy, 1e-10f));
            }
        }

        // Normalize
        float mean = std::accumulate(mel_spec.begin(), mel_spec.end(), 0.0f) / mel_spec.size();
        float sq_sum = 0.0f;
        for (float v : mel_spec) sq_sum += (v - mean) * (v - mean);
        float std_dev = std::sqrt(sq_sum / mel_spec.size());
        
        if (std_dev > 0.0f) {
            for (float& v : mel_spec) v = (v - mean) / std_dev;
        }

        return mel_spec;

    } catch (const std::exception& e) {
        LOG_ERROR("Mel spectrogram error: {}", e.what());
        return std::vector<float>(N_MELS * TIME_STEPS, 0.0f);
    }
}

//=============================================================================
// Emotion Mapping
//=============================================================================
std::string Audio::class_to_emotion(int class_id) const
{
    static const std::unordered_map<int, std::string> mapping = {
        {0, "anger"}, {1, "disgust"}, {2, "fear"}, 
        {3, "happiness"}, {4, "neutral"}, {5, "sadness"}, {6, "surprise"}
    };
    
    auto it = mapping.find(class_id);
    return (it != mapping.end()) ? it->second : "unknown";
}

//=============================================================================
// Main Processing
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

        LOG_INFO("Processing audio: {} samples, {} Hz, {:.2f}s", 
                 audio_data_.size(), sample_rate_, get_duration());

        // Extract mel spectrogram
        std::vector<float> mel_spec = extract_mel_spectrogram();
        if (mel_spec.size() != static_cast<size_t>(N_MELS * TIME_STEPS)) {
            LOG_WARN("Resizing mel spectrogram from {} to {}", 
                     mel_spec.size(), N_MELS * TIME_STEPS);
            mel_spec.resize(N_MELS * TIME_STEPS, 0.0f);
        }

        // Create tensor [1, 128, 3000]
        torch::Tensor input_tensor = torch::from_blob(
            mel_spec.data(), 
            {1, N_MELS, TIME_STEPS}, 
            torch::kFloat
        ).clone();

        // Run inference
        audio_model->eval();
        torch::Tensor output = audio_model->forward({input_tensor}).toTensor();

        // Get predictions
        auto scores = torch::softmax(output, 1).squeeze().to(torch::kCPU).to(torch::kFloat);
        int predicted_class = output.argmax(1).item<int>();
        
        // Convert to vector
        std::vector<float> probs(scores.data_ptr<float>(), 
                                 scores.data_ptr<float>() + scores.numel());

        // Map to emotion
        std::string emotion = class_to_emotion(predicted_class);
        
        // Build result
        std::vector<std::string> emotion_labels = {
            "anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"
        };

        int emotion_index = 0;
        for (size_t i = 0; i < emotion_labels.size(); ++i) {
            if (emotion_labels[i] == emotion) {
                emotion_index = i;
                break;
            }
        }

        result["main_prediction"] = {
            {"index", emotion_index},
            {"label", emotion},
            {"probability", probs[predicted_class]}
        };

        // All probabilities
        nlohmann::json probs_json;
        for (size_t i = 0; i < probs.size() && i < emotion_labels.size(); ++i) {
            probs_json[emotion_labels[i]] = fmt::format("{:.2f}", probs[i]);
        }
        result["additional_probs"] = probs_json;

        LOG_INFO("Predicted: {} ({:.1f}%)", emotion, probs[predicted_class] * 100);

    } catch (const c10::Error& e) {
        LOG_ERROR("LibTorch error: {}", e.what());
        result["error"] = std::string("LibTorch error: ") + e.what();
        result["error_type"] = "libtorch";
    } catch (const std::exception& e) {
        LOG_ERROR("Audio processing error: {}", e.what());
        result["error"] = e.what();
        result["error_type"] = "std_exception";
    }
    
    return result;
}

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