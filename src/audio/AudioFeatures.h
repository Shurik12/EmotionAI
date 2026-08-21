#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace audio {

struct AcousticFeatures {
    // Pitch features
    double pitch_mean = 0.0;
    double pitch_std = 0.0;
    double pitch_range = 0.0;
    double pitch_min = 0.0;
    double pitch_max = 0.0;
    double pitch_variation = 0.0;
    
    // Intensity features
    double intensity_mean = 0.0;
    double intensity_std = 0.0;
    double intensity_min = 0.0;
    double intensity_max = 0.0;
    double intensity_variation = 0.0;
    double dynamic_range = 0.0;
    
    // Pause features
    double pause_count = 0.0;
    double pause_total_duration = 0.0;
    double pause_mean_duration = 0.0;
    double pause_std_duration = 0.0;
    double pause_ratio = 0.0;
    double pause_min_duration = 0.0;
    double pause_max_duration = 0.0;
    
    // Speech rate
    double speech_rate = 0.0;
    double syllable_count = 0.0;
    double voiced_segments = 0.0;
    double speech_duration = 0.0;
    
    // Voice activity
    double voice_activity_ratio = 0.0;
    double voice_duration = 0.0;
    double silence_duration = 0.0;
    double total_duration = 0.0;
    
    // MFCC features (from LibrosaCpp)
    std::vector<std::vector<float>> mfcc;
    std::vector<std::vector<float>> mel_spectrogram;
    
    // Convert to JSON
    nlohmann::json toJson() const;
    
    // Parse from JSON
    static AcousticFeatures fromJson(const nlohmann::json& data);
    
    // Check if features are valid
    bool isValid() const { return total_duration > 0.0; }
};

} // namespace audio