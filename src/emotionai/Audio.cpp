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
#include <audio/LibrosaFeatureExtractor.h>
#include <audio/AudioFeatures.h>
#include <audio/BurnoutAnalyzer.h>
#include "Audio.h"

#ifdef HAVE_FFMPEG
extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libavutil/avutil.h>
    #include <libswresample/swresample.h>
    #include <libavutil/opt.h>
    #include <libavutil/channel_layout.h>
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
    "anger", "disgust", "enthusiasm", "fear", "happiness", "neutral", "sadness"
};

//=============================================================================
// Construction / Destruction
//=============================================================================
Audio::Audio(const std::string &filename)
    : filename_(filename)
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
        if (!ext.empty() && ext[0] == '.') {
            ext = ext.substr(1);
        }

        if (ext == "wav") {
            return load_wav_file(filename);
        }

#ifdef HAVE_FFMPEG
        // Support MP3 and other formats via FFmpeg
        if (ext == "mp3" || ext == "flac" || ext == "m4a" || ext == "aac" || 
            ext == "ogg" || ext == "opus" || ext == "wma") {
            return decode_audio_file(filename);
        }
        // Try FFmpeg for any other format as well
        return decode_audio_file(filename);
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

    file.seekg(0, std::ios::end);
    long long file_size = static_cast<long long>(file.tellg());
    file.seekg(0, std::ios::beg);

    WavHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (strncmp(header.chunk_id, "RIFF", 4) != 0 || 
        strncmp(header.format, "WAVE", 4) != 0) {
        error_ = "Invalid WAV file";
        LOG_ERROR("{}", error_);
        return false;
    }

    sample_rate_ = header.sample_rate;
    channels_ = header.num_channels;
    int bytes_per_sample = header.bits_per_sample / 8;

    auto [found, data_pos, data_size] = find_data_chunk(file, header);
    if (!found) {
        error_ = "Could not find audio data chunk";
        LOG_ERROR("{}", error_);
        return false;
    }

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

    int actual_samples = bytes_read / (bytes_per_sample * channels_);
    audio_data_.resize(actual_samples * channels_);
    for (size_t i = 0; i < audio_data_.size(); ++i) {
        audio_data_[i] = int_data[i] / 32768.0f;
    }

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
    if (strncmp(header.subchunk2_id, "data", 4) == 0) {
        return {true, sizeof(WavHeader), header.subchunk2_size};
    }

    long long fmt_end = 12 + 8 + header.subchunk1_size;
    if (header.subchunk1_size % 2 != 0) {
        fmt_end += 1;
    }
    
    file.seekg(fmt_end, std::ios::beg);
    
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

bool Audio::decode_audio_file(const std::string &filename)
{
    LOG_INFO("Decoding audio file with FFmpeg: {}", filename);
    
    int ret;
    
    // Open input file
    if ((ret = avformat_open_input(&format_ctx_, filename.c_str(), nullptr, nullptr)) < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errbuf, sizeof(errbuf));
        error_ = "Failed to open input file: " + std::string(errbuf);
        LOG_ERROR("{}", error_);
        return false;
    }
    
    // Find stream info
    if ((ret = avformat_find_stream_info(format_ctx_, nullptr)) < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errbuf, sizeof(errbuf));
        error_ = "Failed to find stream info: " + std::string(errbuf);
        LOG_ERROR("{}", error_);
        return false;
    }
    
    // Find audio stream
    int audio_stream_idx = -1;
    for (unsigned int i = 0; i < format_ctx_->nb_streams; i++) {
        if (format_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audio_stream_idx = i;
            break;
        }
    }
    
    if (audio_stream_idx == -1) {
        error_ = "No audio stream found";
        LOG_ERROR("{}", error_);
        return false;
    }
    
    AVCodecParameters* codecpar = format_ctx_->streams[audio_stream_idx]->codecpar;
    
    // Find decoder
    const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
    if (!codec) {
        error_ = "Codec not found";
        LOG_ERROR("{}", error_);
        return false;
    }
    
    // Allocate codec context
    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
        error_ = "Failed to allocate codec context";
        LOG_ERROR("{}", error_);
        return false;
    }
    
    // Copy codec parameters
    if ((ret = avcodec_parameters_to_context(codec_ctx_, codecpar)) < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errbuf, sizeof(errbuf));
        error_ = "Failed to copy codec params: " + std::string(errbuf);
        LOG_ERROR("{}", error_);
        return false;
    }
    
    // Open codec
    if ((ret = avcodec_open2(codec_ctx_, codec, nullptr)) < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errbuf, sizeof(errbuf));
        error_ = "Failed to open codec: " + std::string(errbuf);
        LOG_ERROR("{}", error_);
        return false;
    }
    
    // Get audio info using new API
    int input_sample_rate = codec_ctx_->sample_rate;
    int input_channels = codec_ctx_->channels;
    AVSampleFormat input_sample_fmt = codec_ctx_->sample_fmt;
    
    // Get channel layout using new API
    AVChannelLayout input_ch_layout;
    av_channel_layout_default(&input_ch_layout, input_channels);
    
    // If codec has a specific layout, use it
    #if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(59, 24, 0)
    if (codec_ctx_->ch_layout.nb_channels > 0) {
        av_channel_layout_copy(&input_ch_layout, &codec_ctx_->ch_layout);
    }
    #endif
    
    LOG_INFO("Input audio: {} Hz, {} channels, sample format: {}", 
             input_sample_rate, input_channels, av_get_sample_fmt_name(input_sample_fmt));
    
    // Configure resampler to convert to mono float
    AVChannelLayout target_ch_layout;
    av_channel_layout_default(&target_ch_layout, 1); // Mono
    int target_channels = 1;
    int target_sample_rate = 16000;
    AVSampleFormat target_sample_fmt = AV_SAMPLE_FMT_FLT;
    
    // Create resampler context
    swr_ctx_ = swr_alloc();
    if (!swr_ctx_) {
        error_ = "Failed to allocate resampler context";
        LOG_ERROR("{}", error_);
        return false;
    }
    
    // Set resampler options using new API
    #if LIBSWRESAMPLE_VERSION_INT >= AV_VERSION_INT(4, 0, 0)
    av_opt_set_chlayout(swr_ctx_, "in_chlayout", &input_ch_layout, 0);
    av_opt_set_chlayout(swr_ctx_, "out_chlayout", &target_ch_layout, 0);
    #else
    // Fallback for older FFmpeg versions
    int64_t input_ch_layout_old = input_ch_layout.u.mask;
    int64_t target_ch_layout_old = target_ch_layout.u.mask;
    av_opt_set_int(swr_ctx_, "in_channel_layout", input_ch_layout_old, 0);
    av_opt_set_int(swr_ctx_, "out_channel_layout", target_ch_layout_old, 0);
    #endif
    
    av_opt_set_int(swr_ctx_, "in_sample_rate", input_sample_rate, 0);
    av_opt_set_int(swr_ctx_, "out_sample_rate", target_sample_rate, 0);
    av_opt_set_sample_fmt(swr_ctx_, "in_sample_fmt", input_sample_fmt, 0);
    av_opt_set_sample_fmt(swr_ctx_, "out_sample_fmt", target_sample_fmt, 0);
    
    // Initialize resampler
    if ((ret = swr_init(swr_ctx_)) < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errbuf, sizeof(errbuf));
        error_ = "Failed to initialize resampler: " + std::string(errbuf);
        LOG_ERROR("{}", error_);
        return false;
    }
    
    // Allocate packet and frame
    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    if (!packet || !frame) {
        error_ = "Failed to allocate packet or frame";
        LOG_ERROR("{}", error_);
        av_packet_free(&packet);
        av_frame_free(&frame);
        return false;
    }
    
    // Buffer for converted audio
    std::vector<float> all_samples;
    int total_samples = 0;
    
    // Read and decode frames
    while (av_read_frame(format_ctx_, packet) >= 0) {
        if (packet->stream_index != audio_stream_idx) {
            av_packet_unref(packet);
            continue;
        }
        
        ret = avcodec_send_packet(codec_ctx_, packet);
        if (ret < 0) {
            av_packet_unref(packet);
            continue;
        }
        
        while (ret >= 0) {
            ret = avcodec_receive_frame(codec_ctx_, frame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                break;
            } else if (ret < 0) {
                break;
            }
            
            // Resample frame
            int out_samples = swr_convert(swr_ctx_, nullptr, 0, 
                                          (const uint8_t**)frame->data, frame->nb_samples);
            if (out_samples < 0) {
                av_frame_unref(frame);
                continue;
            }
            
            // Get enough buffer for resampled data
            int max_out_samples = frame->nb_samples * target_sample_rate / input_sample_rate + 1024;
            std::vector<float> buffer(max_out_samples * target_channels);
            
            uint8_t* out_data[1] = { reinterpret_cast<uint8_t*>(buffer.data()) };
            
            out_samples = swr_convert(swr_ctx_, out_data, max_out_samples,
                                      (const uint8_t**)frame->data, frame->nb_samples);
            
            if (out_samples > 0) {
                // Append to all_samples
                all_samples.insert(all_samples.end(), buffer.data(), buffer.data() + out_samples);
                total_samples += out_samples;
            }
            
            av_frame_unref(frame);
        }
        av_packet_unref(packet);
    }
    
    // Flush resampler
    int out_samples = swr_convert(swr_ctx_, nullptr, 0, nullptr, 0);
    if (out_samples > 0) {
        int max_out_samples = out_samples + 1024;
        std::vector<float> buffer(max_out_samples * target_channels);
        uint8_t* out_data[1] = { reinterpret_cast<uint8_t*>(buffer.data()) };
        
        out_samples = swr_convert(swr_ctx_, out_data, max_out_samples, nullptr, 0);
        if (out_samples > 0) {
            all_samples.insert(all_samples.end(), buffer.data(), buffer.data() + out_samples);
            total_samples += out_samples;
        }
    }
    
    // Clean up
    av_packet_free(&packet);
    av_frame_free(&frame);
    
    if (all_samples.empty()) {
        error_ = "No audio data extracted";
        LOG_ERROR("{}", error_);
        return false;
    }
    
    // Store audio data
    audio_data_ = std::move(all_samples);
    sample_rate_ = target_sample_rate;
    channels_ = target_channels;
    
    LOG_INFO("Decoded audio: {} samples, {} Hz, {} channels, {:.2f}s", 
             audio_data_.size(), sample_rate_, channels_, 
             static_cast<double>(audio_data_.size()) / sample_rate_);
    
    return true;
}

// Keep the old method for backward compatibility
bool Audio::decode_audio_stream()
{
    if (filename_.empty()) {
        error_ = "No filename available for decoding";
        LOG_ERROR("{}", error_);
        return false;
    }
    return decode_audio_file(filename_);
}
#endif

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

std::vector<float> Audio::prepare_audio() const
{
    if (!loaded_ || audio_data_.empty()) {
        return {};
    }
    
    // Resample to 16kHz
    std::vector<float> audio = resample_audio(TARGET_SR);
    
    // Normalize
    float max_val = 0.0f;
    for (float sample : audio) {
        if (std::abs(sample) > max_val) max_val = std::abs(sample);
    }
    if (max_val > 1.0f) {
        for (float& sample : audio) {
            sample /= max_val;
        }
    }
    
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
    
    return audio;
}

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

        LOG_INFO("Processing audio with WavLM: {} samples, {} Hz, {:.2f}s", 
                 audio_data_.size(), sample_rate_, get_duration());

        // Prepare audio (resample, normalize, truncate/pad)
        std::vector<float> audio = prepare_audio();
        
        // Create tensor
        torch::Tensor input_tensor = torch::from_blob(
            audio.data(), 
            {1, static_cast<int64_t>(audio.size())}, 
            torch::kFloat
        ).clone();

        LOG_INFO("Running WavLM inference on {} samples ({:.2f}s)", 
                 audio.size(), static_cast<double>(audio.size()) / TARGET_SR);

        // Run inference
        audio_model->eval();
        torch::NoGradGuard no_grad;
        
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        torch::Tensor output = audio_model->forward(inputs).toTensor();
        
        auto scores = torch::softmax(output, 1).squeeze().to(torch::kCPU);
        int predicted_class = output.argmax(1).item<int>();
        
        std::vector<float> probs(scores.data_ptr<float>(), 
                                 scores.data_ptr<float>() + scores.numel());

        std::string emotion = (predicted_class < EMOTION_LABELS.size()) 
            ? EMOTION_LABELS[predicted_class] 
            : "unknown";
        
        // Build result
        result["main_prediction"] = {
            {"index", predicted_class},
            {"label", emotion},
            {"probability", probs[predicted_class]}
        };

        nlohmann::json probs_json;
        for (size_t i = 0; i < probs.size() && i < EMOTION_LABELS.size(); ++i) {
            probs_json[EMOTION_LABELS[i]] = fmt::format("{:.4f}", probs[i]);
        }
        result["additional_probs"] = probs_json;
        
        // Add metadata
        result["model"] = "wavlm-emotion-russian-resd";
        result["duration_seconds"] = static_cast<double>(audio.size()) / TARGET_SR;
        result["sample_rate"] = TARGET_SR;

        LOG_INFO("WavLM predicted: {} ({:.1f}%)", emotion, probs[predicted_class] * 100);

    } catch (const c10::Error& e) {
        LOG_ERROR("LibTorch error in WavLM: {}", e.what());
        result["error"] = std::string("LibTorch error: ") + e.what();
        result["error_type"] = "libtorch";
    } catch (const std::exception& e) {
        LOG_ERROR("WavLM processing error: {}", e.what());
        result["error"] = e.what();
        result["error_type"] = "std_exception";
    }
    
    return result;
}

audio::AcousticFeatures Audio::extract_acoustic_features() const
{
    if (!loaded_ || audio_data_.empty()) {
        LOG_WARN("Cannot extract features: audio not loaded");
        return audio::AcousticFeatures{};
    }
    
    std::vector<float> audio = prepare_audio();
    
    audio::LibrosaFeatureExtractor::Config config;
    config.sample_rate = TARGET_SR;
    // Burnout scoring consumes only the scalar groups (pitch, intensity,
    // pauses, speech rate, voice activity). The Mel spectrogram and MFCCs
    // are not part of the five-component model, so the burnout path does
    // not pay for them; they remain available via extractAllFeatures().
    return audio::LibrosaFeatureExtractor::extractAcousticFeaturesOnly(audio, config);
}

nlohmann::json Audio::add_burnout_analysis(
    const nlohmann::json& emotion_result,
    const nlohmann::json& baseline,
    const audio::AcousticFeatures* acoustic_features)
{
    nlohmann::json result = emotion_result;
    
    try {
        const audio::BurnoutConfig cfg = Config::instance().burnout();

        // Add acoustic features if provided
        if (acoustic_features) {
            result["acoustic_features"] = acoustic_features->toJson();

            // Speech-activity gate: no usable voice means no score. WavLM
            // returns near-uniform probabilities on silence/noise, and
            // scoring that against any baseline yields a confident-looking
            // number that reflects the baseline, not the speaker.
            if (acoustic_features->voice_activity_ratio < cfg.min_voice_activity_ratio) {
                audio::Result insufficient;
                insufficient.state = audio::State::INSUFFICIENT_DATA;
                insufficient.error = "insufficient_speech";
                insufficient.recommendations = {"analysis_failed_retry"};
                result["burnout_analysis"] = insufficient.toJson();
                LOG_WARN("Burnout skipped: voice_activity_ratio {:.3f} < {:.3f}",
                         acoustic_features->voice_activity_ratio,
                         cfg.min_voice_activity_ratio);
                return result;
            }
        }

        // Default baseline is the empirically calibrated population median,
        // used only when the caller did not supply a stored per-user baseline.
        const nlohmann::json default_baseline = cfg.defaultBaselineJson();
        
        // Run burnout analysis
        audio::BurnoutAnalyzer analyzer;
        double audio_quality = result.value("duration_seconds", 1.0) / 5.0;
        audio_quality = std::min(1.0, std::max(0.5, audio_quality));
        double baseline_reliability = baseline.empty() ? 0.3 : 0.7;
        
        const nlohmann::json& baseline_to_use = baseline.empty() ? default_baseline : baseline;
        
        auto burnout_result = analyzer.analyze(
            result,
            baseline_to_use,
            {}, // history
            audio_quality,
            baseline_reliability
        );
        
        result["burnout_analysis"] = burnout_result.toJson();
        
        LOG_INFO("Burnout analysis added: state={}, score={:.1f}", 
                 audio::stateToString(burnout_result.state), burnout_result.score);
                 
    } catch (const std::exception& e) {
        LOG_ERROR("Error adding burnout analysis: {}", e.what());
        result["burnout_error"] = e.what();
    }
    
    return result;
}

nlohmann::json Audio::process_audio_with_burnout(
    torch::jit::Module* audio_model,
    const nlohmann::json& baseline)
{
    // Step 1: Get emotion recognition result
    nlohmann::json emotion_result = process_audio(audio_model);
    
    // Step 2: Check if emotion processing had errors
    if (emotion_result.contains("error")) {
        LOG_ERROR("Emotion processing failed: {}", emotion_result["error"]);
        return emotion_result;
    }
    
    // Step 3: Extract acoustic features
    audio::AcousticFeatures features = extract_acoustic_features();
    
    // Step 4: Add burnout analysis
    nlohmann::json result = add_burnout_analysis(emotion_result, baseline, &features);
    
    LOG_INFO("Audio processing with burnout complete");
    
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