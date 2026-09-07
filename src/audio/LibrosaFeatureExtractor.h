// src/audio/LibrosaFeatureExtractor.h
#pragma once

#include <vector>
#include <string>
#include "AudioFeatures.h"

namespace audio {

class LibrosaFeatureExtractor {
public:
    struct Config {
        int sample_rate = 16000;
        int n_fft = 400;
        int n_hop = 160;
        std::string window = "hann";
        bool center = false;
        std::string pad_mode = "reflect";
        float power = 2.0f;
        int n_mel = 128;
        int fmin = 80;
        int fmax = 7600;
        int n_mfcc = 13;
        bool norm = true;
        int dct_type = 2;
        
        float voice_activity_threshold = 0.3f;
        float pause_threshold = 0.2f;
        float pitch_autocorrelation_threshold = 0.3f;
        int pitch_min_lag = 40;
        int pitch_max_lag = 200;
    };
    
    // FIX: Use const Config& with default, but define default in implementation
    static AcousticFeatures extractAllFeatures(
        const std::vector<float>& audio,
        const Config& config
    );
    
    static AcousticFeatures extractAcousticFeaturesOnly(
        const std::vector<float>& audio,
        const Config& config
    );

    // Whole-recording quality in 0..1: silence ratio, average level and
    // clipping, computed over short frames (25 ms / 10 ms hop).
    // config.sample_rate must be the actual rate of `audio`.
    static double estimateAudioQuality(
        const std::vector<float>& audio,
        const Config& config
    );
    
    static std::vector<std::vector<float>> extractMFCC(
        const std::vector<float>& audio,
        const Config& config
    );
    
    static std::vector<std::vector<float>> extractMelSpectrogram(
        const std::vector<float>& audio,
        const Config& config
    );
    
private:
    static AcousticFeatures extractPitchFeatures(
        const std::vector<float>& audio,
        const Config& config
    );
    
    static AcousticFeatures extractIntensityFeatures(
        const std::vector<float>& audio,
        const Config& config
    );
    
    static AcousticFeatures extractPauseFeatures(
        const std::vector<float>& audio,
        const Config& config
    );
    
    static AcousticFeatures extractSpeechRateFeatures(
        const std::vector<float>& audio,
        const Config& config
    );
    
    static AcousticFeatures extractVoiceActivityFeatures(
        const std::vector<float>& audio,
        const Config& config
    );
    
    static std::vector<float> computeRms(
        const std::vector<float>& audio,
        int frame_length,
        int hop_length
    );
    
    static std::vector<int> detectVoicedFrames(
        const std::vector<float>& rms,
        float threshold
    );
    
    static std::vector<float> findPauses(
        const std::vector<int>& voiced_frames,
        int hop_length,
        int sample_rate
    );
    
    static std::vector<float> detectPitch(
        const std::vector<float>& audio,
        const Config& config
    );
    
    static AcousticFeatures mergeFeatures(
        const std::vector<AcousticFeatures>& feature_list
    );
};

// FIX: Add inline default config function
inline LibrosaFeatureExtractor::Config getDefaultConfig() {
    return LibrosaFeatureExtractor::Config{};
}

} // namespace audio