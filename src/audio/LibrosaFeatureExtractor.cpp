#include "LibrosaFeatureExtractor.h"
#include <common/librosa.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <logging/Logger.h>

namespace audio {

namespace {
    constexpr int FRAME_LENGTH_MS = 25;
    constexpr int HOP_LENGTH_MS = 10;
}

//=============================================================================
// extractAllFeatures
//=============================================================================
AcousticFeatures LibrosaFeatureExtractor::extractAllFeatures(
    const std::vector<float>& audio,
    const Config& config)
{
    AcousticFeatures features;
    
    if (audio.empty() || config.sample_rate <= 0) {
        LOG_WARN("Empty audio or invalid sample rate");
        return features;
    }
    
    try {
        LOG_DEBUG("Extracting all features: {} samples, {} Hz", 
                  audio.size(), config.sample_rate);
        
        // FIX: LibrosaCpp expects non-const reference, so make a copy
        std::vector<float> audio_copy = audio;
        
        // 1. Compute Mel Spectrogram using LibrosaCpp
        auto mels = librosa::Feature::melspectrogram(
            audio_copy,  // Now passing non-const
            config.sample_rate,
            config.n_fft,
            config.n_hop,
            config.window,
            config.center,
            config.pad_mode,
            config.power,
            config.n_mel,
            config.fmin,
            config.fmax
        );
        
        // 2. Compute MFCC using LibrosaCpp
        auto mfcc = librosa::Feature::mfcc(
            audio_copy,  // Now passing non-const
            config.sample_rate,
            config.n_fft,
            config.n_hop,
            config.window,
            config.center,
            config.pad_mode,
            config.power,
            config.n_mel,
            config.fmin,
            config.fmax,
            config.n_mfcc,
            config.norm,
            config.dct_type
        );
        
        // 3. Extract traditional acoustic features (these don't use LibrosaCpp)
        auto pitch_feat = extractPitchFeatures(audio, config);
        auto intensity_feat = extractIntensityFeatures(audio, config);
        auto pause_feat = extractPauseFeatures(audio, config);
        auto speech_feat = extractSpeechRateFeatures(audio, config);
        auto voice_feat = extractVoiceActivityFeatures(audio, config);
        
        // 4. Merge scalar features and re-attach the spectral blocks.
        //    mergeFeatures() carries only the scalar groups, so assigning
        //    them here is what makes "all features" actually be all of them
        //    (previously MFCC/mel were computed and silently discarded).
        features = mergeFeatures({pitch_feat, intensity_feat, pause_feat, 
                                   speech_feat, voice_feat});
        features.mfcc = std::move(mfcc);
        features.mel_spectrogram = std::move(mels);
        
        // 5. Set total duration (from voice activity)
        features.total_duration = voice_feat.total_duration;
        
        LOG_DEBUG("Features extracted: duration={:.2f}s, mfcc={}x{}, mels={}x{}",
                  features.total_duration,
                  mfcc.size(), mfcc.empty() ? 0 : mfcc[0].size(),
                  mels.size(), mels.empty() ? 0 : mels[0].size());
                  
    } catch (const std::exception& e) {
        LOG_ERROR("Error extracting features with LibrosaCpp: {}", e.what());
    }
    
    return features;
}

//=============================================================================
// extractAcousticFeaturesOnly
//=============================================================================
AcousticFeatures LibrosaFeatureExtractor::extractAcousticFeaturesOnly(
    const std::vector<float>& audio,
    const Config& config)
{
    AcousticFeatures features;
    
    if (audio.empty() || config.sample_rate <= 0) {
        return features;
    }
    
    try {
        auto pitch_feat = extractPitchFeatures(audio, config);
        auto intensity_feat = extractIntensityFeatures(audio, config);
        auto pause_feat = extractPauseFeatures(audio, config);
        auto speech_feat = extractSpeechRateFeatures(audio, config);
        auto voice_feat = extractVoiceActivityFeatures(audio, config);
        
        features = mergeFeatures({pitch_feat, intensity_feat, pause_feat, 
                                   speech_feat, voice_feat});
        
        features.total_duration = voice_feat.total_duration;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Error extracting acoustic features: {}", e.what());
    }
    
    return features;
}

//=============================================================================
// estimateAudioQuality
//
// Heuristic whole-recording quality in 0..1 used by the external-influence
// gate: rewards recordings with little near-silence, a healthy SPEECH level
// and no heavy clipping. The level is measured on non-silent frames only so
// a long silent tail (typical for call recordings) does not drag it down,
// and up to ~75% silence still leaves partial credit - real two-party calls
// are mostly pauses. Internal thresholds only; the analysis minimum
// (audio_quality_min) lives in config. Expects float PCM in [-1, 1] as
// produced by Audio::decode (int16 / 32768 or FFmpeg float).
//=============================================================================
double LibrosaFeatureExtractor::estimateAudioQuality(
    const std::vector<float>& audio,
    const Config& config)
{
    if (audio.empty() || config.sample_rate <= 0) {
        return 0.0;
    }

    const int frame_length = static_cast<int>(FRAME_LENGTH_MS * config.sample_rate / 1000);
    const int hop_length = static_cast<int>(HOP_LENGTH_MS * config.sample_rate / 1000);
    if (frame_length <= 0 || hop_length <= 0 ||
        audio.size() < static_cast<size_t>(frame_length)) {
        return 0.0;
    }

    double speech_rms_sum = 0.0;
    size_t silence_frames = 0;
    size_t speech_frames = 0;
    size_t clip_frames = 0;
    size_t total_frames = 0;

    for (size_t i = 0; i + frame_length <= audio.size(); i += hop_length) {
        double sum_sq = 0.0;
        float peak = 0.0f;
        for (size_t j = 0; j < static_cast<size_t>(frame_length); ++j) {
            const float sample = audio[i + j];
            sum_sq += static_cast<double>(sample) * sample;
            peak = std::max(peak, std::abs(sample));
        }
        const double rms = std::sqrt(sum_sq / frame_length);
        ++total_frames;
        if (rms < 0.005) {
            ++silence_frames;
        } else {
            ++speech_frames;
            speech_rms_sum += rms;
        }
        if (peak >= 0.98f) ++clip_frames;
    }
    if (total_frames == 0) {
        return 0.0;
    }

    const double silence_ratio = static_cast<double>(silence_frames) / total_frames;
    const double clip_ratio = static_cast<double>(clip_frames) / total_frames;
    const double speech_rms = speech_frames > 0 ? speech_rms_sum / speech_frames : 0.0;

    const double silence_score = 1.0 - std::min(1.0, silence_ratio / 0.75);
    const double level_score = std::min(1.0, speech_rms / 0.03);
    const double clip_score = 1.0 - std::min(1.0, clip_ratio / 0.05);

    return std::max(0.0, std::min(1.0,
        0.60 * silence_score + 0.25 * level_score + 0.15 * clip_score));
}

//=============================================================================
// extractMFCC
//=============================================================================
std::vector<std::vector<float>> LibrosaFeatureExtractor::extractMFCC(
    const std::vector<float>& audio,
    const Config& config)
{
    try {
        // FIX: LibrosaCpp expects non-const reference
        std::vector<float> audio_copy = audio;
        return librosa::Feature::mfcc(
            audio_copy,
            config.sample_rate,
            config.n_fft,
            config.n_hop,
            config.window,
            config.center,
            config.pad_mode,
            config.power,
            config.n_mel,
            config.fmin,
            config.fmax,
            config.n_mfcc,
            config.norm,
            config.dct_type
        );
    } catch (const std::exception& e) {
        LOG_ERROR("Error extracting MFCC: {}", e.what());
        return {};
    }
}

//=============================================================================
// extractMelSpectrogram
//=============================================================================
std::vector<std::vector<float>> LibrosaFeatureExtractor::extractMelSpectrogram(
    const std::vector<float>& audio,
    const Config& config)
{
    try {
        // FIX: LibrosaCpp expects non-const reference
        std::vector<float> audio_copy = audio;
        return librosa::Feature::melspectrogram(
            audio_copy,
            config.sample_rate,
            config.n_fft,
            config.n_hop,
            config.window,
            config.center,
            config.pad_mode,
            config.power,
            config.n_mel,
            config.fmin,
            config.fmax
        );
    } catch (const std::exception& e) {
        LOG_ERROR("Error extracting Mel spectrogram: {}", e.what());
        return {};
    }
}

//=============================================================================
// extractPitchFeatures
//=============================================================================
AcousticFeatures LibrosaFeatureExtractor::extractPitchFeatures(
    const std::vector<float>& audio,
    const Config& config)
{
    AcousticFeatures features;
    
    auto pitch_values = detectPitch(audio, config);
    
    if (!pitch_values.empty()) {
        features.pitch_mean = std::accumulate(pitch_values.begin(), 
                                               pitch_values.end(), 0.0) / pitch_values.size();
        
        double sq_sum = 0.0;
        for (float p : pitch_values) {
            sq_sum += (p - features.pitch_mean) * (p - features.pitch_mean);
        }
        features.pitch_std = std::sqrt(sq_sum / pitch_values.size());
        
        auto [min_it, max_it] = std::minmax_element(pitch_values.begin(), 
                                                     pitch_values.end());
        features.pitch_min = *min_it;
        features.pitch_max = *max_it;
        features.pitch_range = features.pitch_max - features.pitch_min;
        features.pitch_variation = (features.pitch_mean > 0) ? 
            features.pitch_std / features.pitch_mean : 0.0;
    }
    
    return features;
}

//=============================================================================
// extractIntensityFeatures
//=============================================================================
AcousticFeatures LibrosaFeatureExtractor::extractIntensityFeatures(
    const std::vector<float>& audio,
    const Config& config)
{
    AcousticFeatures features;
    
    int frame_length = static_cast<int>(FRAME_LENGTH_MS * config.sample_rate / 1000);
    int hop_length = static_cast<int>(HOP_LENGTH_MS * config.sample_rate / 1000);
    
    auto rms = computeRms(audio, frame_length, hop_length);
    
    if (!rms.empty()) {
        features.intensity_mean = std::accumulate(rms.begin(), rms.end(), 0.0) / rms.size();
        
        double sq_sum = 0.0;
        for (float r : rms) {
            sq_sum += (r - features.intensity_mean) * (r - features.intensity_mean);
        }
        features.intensity_std = std::sqrt(sq_sum / rms.size());
        
        auto [min_it, max_it] = std::minmax_element(rms.begin(), rms.end());
        features.intensity_min = *min_it;
        features.intensity_max = *max_it;
        features.dynamic_range = features.intensity_max - features.intensity_min;
        features.intensity_variation = (features.intensity_mean > 0) ? 
            features.intensity_std / features.intensity_mean : 0.0;
    }
    
    return features;
}

//=============================================================================
// extractPauseFeatures
//=============================================================================
AcousticFeatures LibrosaFeatureExtractor::extractPauseFeatures(
    const std::vector<float>& audio,
    const Config& config)
{
    AcousticFeatures features;
    
    int frame_length = static_cast<int>(FRAME_LENGTH_MS * config.sample_rate / 1000);
    int hop_length = static_cast<int>(HOP_LENGTH_MS * config.sample_rate / 1000);
    
    auto rms = computeRms(audio, frame_length, hop_length);
    float threshold = (rms.empty()) ? 0.0f : 
        std::accumulate(rms.begin(), rms.end(), 0.0f) / rms.size() * config.pause_threshold;
    
    auto voiced_frames = detectVoicedFrames(rms, threshold);
    auto pauses = findPauses(voiced_frames, hop_length, config.sample_rate);
    
    features.total_duration = static_cast<double>(audio.size()) / config.sample_rate;
    
    if (!pauses.empty()) {
        features.pause_count = pauses.size();
        features.pause_total_duration = std::accumulate(pauses.begin(), pauses.end(), 0.0);
        features.pause_mean_duration = features.pause_total_duration / pauses.size();
        
        double sq_sum = 0.0;
        for (float p : pauses) {
            sq_sum += (p - features.pause_mean_duration) * (p - features.pause_mean_duration);
        }
        features.pause_std_duration = std::sqrt(sq_sum / pauses.size());
        
        auto [min_it, max_it] = std::minmax_element(pauses.begin(), pauses.end());
        features.pause_min_duration = *min_it;
        features.pause_max_duration = *max_it;
        features.pause_ratio = (features.total_duration > 0) ? 
            features.pause_total_duration / features.total_duration : 0.0;
    }
    
    return features;
}

//=============================================================================
// extractSpeechRateFeatures
//=============================================================================
AcousticFeatures LibrosaFeatureExtractor::extractSpeechRateFeatures(
    const std::vector<float>& audio,
    const Config& config)
{
    AcousticFeatures features;
    
    int frame_length = static_cast<int>(FRAME_LENGTH_MS * config.sample_rate / 1000);
    int hop_length = static_cast<int>(HOP_LENGTH_MS * config.sample_rate / 1000);
    
    auto rms = computeRms(audio, frame_length, hop_length);
    float max_rms = (rms.empty()) ? 0.0f : *std::max_element(rms.begin(), rms.end());
    
    features.total_duration = static_cast<double>(audio.size()) / config.sample_rate;
    
    if (max_rms > 0 && !rms.empty()) {
        std::vector<float> normalized_rms(rms.size());
        for (size_t i = 0; i < rms.size(); ++i) {
            normalized_rms[i] = rms[i] / max_rms;
        }
        
        // Count peaks (syllables) using simple peak detection
        int peaks = 0;
        for (size_t i = 1; i < normalized_rms.size() - 1; ++i) {
            if (normalized_rms[i] > normalized_rms[i-1] && 
                normalized_rms[i] > normalized_rms[i+1] &&
                normalized_rms[i] > 0.3f) {
                peaks++;
            }
        }
        
        features.syllable_count = peaks;
        
        // Get voice duration from voice activity
        float threshold = std::accumulate(rms.begin(), rms.end(), 0.0f) / rms.size() * 
                          config.voice_activity_threshold;
        auto voiced = detectVoicedFrames(rms, threshold);
        int voiced_count = std::accumulate(voiced.begin(), voiced.end(), 0);
        features.voice_duration = (static_cast<double>(voiced_count) / voiced.size()) * 
                                   features.total_duration;
        
        features.speech_rate = (features.voice_duration > 0) ? 
            peaks / features.voice_duration : 0.0;
    }
    
    return features;
}

//=============================================================================
// extractVoiceActivityFeatures
//=============================================================================
AcousticFeatures LibrosaFeatureExtractor::extractVoiceActivityFeatures(
    const std::vector<float>& audio,
    const Config& config)
{
    AcousticFeatures features;
    
    int frame_length = static_cast<int>(FRAME_LENGTH_MS * config.sample_rate / 1000);
    int hop_length = static_cast<int>(HOP_LENGTH_MS * config.sample_rate / 1000);
    
    auto rms = computeRms(audio, frame_length, hop_length);
    float threshold = (rms.empty()) ? 0.0f :
        std::accumulate(rms.begin(), rms.end(), 0.0f) / rms.size() * 
        config.voice_activity_threshold;
    
    features.total_duration = static_cast<double>(audio.size()) / config.sample_rate;
    
    if (!rms.empty()) {
        auto voiced_frames = detectVoicedFrames(rms, threshold);
        int voiced_count = std::accumulate(voiced_frames.begin(), voiced_frames.end(), 0);
        
        features.voice_activity_ratio = static_cast<double>(voiced_count) / voiced_frames.size();
        features.voice_duration = features.voice_activity_ratio * features.total_duration;
        features.silence_duration = features.total_duration - features.voice_duration;
        
        // Count voiced segments
        int segments = 0;
        for (size_t i = 1; i < voiced_frames.size(); ++i) {
            if (voiced_frames[i] == 1 && voiced_frames[i-1] == 0) {
                segments++;
            }
        }
        features.voiced_segments = segments;
    }
    
    return features;
}

//=============================================================================
// Helper: computeRms
//=============================================================================
std::vector<float> LibrosaFeatureExtractor::computeRms(
    const std::vector<float>& audio,
    int frame_length,
    int hop_length)
{
    std::vector<float> rms;
    if (audio.size() < static_cast<size_t>(frame_length)) {
        return rms;
    }
    
    rms.reserve(audio.size() / hop_length);
    
    for (size_t i = 0; i + frame_length < audio.size(); i += hop_length) {
        float sum_sq = 0.0f;
        for (size_t j = 0; j < static_cast<size_t>(frame_length); ++j) {
            sum_sq += audio[i + j] * audio[i + j];
        }
        rms.push_back(std::sqrt(sum_sq / frame_length));
    }
    
    return rms;
}

//=============================================================================
// Helper: detectVoicedFrames
//=============================================================================
std::vector<int> LibrosaFeatureExtractor::detectVoicedFrames(
    const std::vector<float>& rms,
    float threshold)
{
    std::vector<int> voiced;
    voiced.reserve(rms.size());
    for (float val : rms) {
        voiced.push_back(val > threshold ? 1 : 0);
    }
    return voiced;
}

//=============================================================================
// Helper: findPauses
//=============================================================================
std::vector<float> LibrosaFeatureExtractor::findPauses(
    const std::vector<int>& voiced_frames,
    int hop_length,
    int sample_rate)
{
    std::vector<float> pauses;
    bool in_pause = false;
    float pause_start = 0.0f;
    
    for (size_t i = 0; i < voiced_frames.size(); ++i) {
        float time = static_cast<float>(i * hop_length) / sample_rate;
        
        if (voiced_frames[i] == 0 && !in_pause) {
            in_pause = true;
            pause_start = time;
        } else if (voiced_frames[i] == 1 && in_pause) {
            in_pause = false;
            pauses.push_back(time - pause_start);
        }
    }
    
    if (in_pause) {
        float end_time = static_cast<float>(voiced_frames.size() * hop_length) / sample_rate;
        pauses.push_back(end_time - pause_start);
    }
    
    return pauses;
}

//=============================================================================
// Helper: detectPitch (autocorrelation-based)
//=============================================================================
std::vector<float> LibrosaFeatureExtractor::detectPitch(
    const std::vector<float>& audio,
    const Config& config)
{
    std::vector<float> pitch_values;
    
    int frame_length = static_cast<int>(FRAME_LENGTH_MS * config.sample_rate / 1000);
    int hop_length = static_cast<int>(HOP_LENGTH_MS * config.sample_rate / 1000);
    
    for (size_t i = 0; i + frame_length < audio.size(); i += hop_length) {
        std::vector<float> frame(audio.begin() + i, audio.begin() + i + frame_length);
        
        // Center and normalize
        float mean = std::accumulate(frame.begin(), frame.end(), 0.0f) / frame.size();
        for (auto& sample : frame) sample -= mean;
        
        // Autocorrelation
        float max_corr = 0.0f;
        int max_lag = 0;
        
        for (int lag = config.pitch_min_lag; lag < config.pitch_max_lag; ++lag) {
            float corr = 0.0f;
            for (size_t j = 0; j + lag < frame.size(); ++j) {
                corr += frame[j] * frame[j + lag];
            }
            if (corr > max_corr) {
                max_corr = corr;
                max_lag = lag;
            }
        }
        
        if (max_lag > 0 && max_corr > config.pitch_autocorrelation_threshold) {
            pitch_values.push_back(static_cast<float>(config.sample_rate) / max_lag);
        }
    }
    
    return pitch_values;
}

//=============================================================================
// Helper: mergeFeatures
//=============================================================================
AcousticFeatures LibrosaFeatureExtractor::mergeFeatures(
    const std::vector<AcousticFeatures>& feature_list)
{
    AcousticFeatures merged;
    
    for (const auto& features : feature_list) {
        // Pitch
        if (features.pitch_mean > 0) merged.pitch_mean = features.pitch_mean;
        if (features.pitch_std > 0) merged.pitch_std = features.pitch_std;
        if (features.pitch_range > 0) merged.pitch_range = features.pitch_range;
        if (features.pitch_min > 0) merged.pitch_min = features.pitch_min;
        if (features.pitch_max > 0) merged.pitch_max = features.pitch_max;
        if (features.pitch_variation > 0) merged.pitch_variation = features.pitch_variation;
        
        // Intensity
        if (features.intensity_mean > 0) merged.intensity_mean = features.intensity_mean;
        if (features.intensity_std > 0) merged.intensity_std = features.intensity_std;
        if (features.intensity_min > 0) merged.intensity_min = features.intensity_min;
        if (features.intensity_max > 0) merged.intensity_max = features.intensity_max;
        if (features.intensity_variation > 0) merged.intensity_variation = features.intensity_variation;
        if (features.dynamic_range > 0) merged.dynamic_range = features.dynamic_range;
        
        // Pauses
        if (features.pause_count > 0) merged.pause_count = features.pause_count;
        if (features.pause_total_duration > 0) merged.pause_total_duration = features.pause_total_duration;
        if (features.pause_mean_duration > 0) merged.pause_mean_duration = features.pause_mean_duration;
        if (features.pause_std_duration > 0) merged.pause_std_duration = features.pause_std_duration;
        if (features.pause_ratio > 0) merged.pause_ratio = features.pause_ratio;
        if (features.pause_min_duration > 0) merged.pause_min_duration = features.pause_min_duration;
        if (features.pause_max_duration > 0) merged.pause_max_duration = features.pause_max_duration;
        
        // Speech
        if (features.speech_rate > 0) merged.speech_rate = features.speech_rate;
        if (features.syllable_count > 0) merged.syllable_count = features.syllable_count;
        if (features.voiced_segments > 0) merged.voiced_segments = features.voiced_segments;
        if (features.speech_duration > 0) merged.speech_duration = features.speech_duration;
        
        // Voice activity
        if (features.voice_activity_ratio > 0) merged.voice_activity_ratio = features.voice_activity_ratio;
        if (features.voice_duration > 0) merged.voice_duration = features.voice_duration;
        if (features.silence_duration > 0) merged.silence_duration = features.silence_duration;
        if (features.total_duration > 0) merged.total_duration = features.total_duration;
    }
    
    return merged;
}

} // namespace audio