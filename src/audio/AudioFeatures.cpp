#include "AudioFeatures.h"
#include <cmath>
#include <logging/Logger.h>

namespace audio {

nlohmann::json AcousticFeatures::toJson() const {
    nlohmann::json j;
    
    // Pitch
    j["pitch_mean"] = pitch_mean;
    j["pitch_std"] = pitch_std;
    j["pitch_range"] = pitch_range;
    j["pitch_min"] = pitch_min;
    j["pitch_max"] = pitch_max;
    j["pitch_variation"] = pitch_variation;
    
    // Intensity
    j["intensity_mean"] = intensity_mean;
    j["intensity_std"] = intensity_std;
    j["intensity_min"] = intensity_min;
    j["intensity_max"] = intensity_max;
    j["intensity_variation"] = intensity_variation;
    j["dynamic_range"] = dynamic_range;
    
    // Pauses
    j["pause_count"] = pause_count;
    j["pause_total_duration"] = pause_total_duration;
    j["pause_mean_duration"] = pause_mean_duration;
    j["pause_std_duration"] = pause_std_duration;
    j["pause_ratio"] = pause_ratio;
    j["pause_min_duration"] = pause_min_duration;
    j["pause_max_duration"] = pause_max_duration;
    
    // Speech
    j["speech_rate"] = speech_rate;
    j["syllable_count"] = syllable_count;
    j["voiced_segments"] = voiced_segments;
    j["speech_duration"] = speech_duration;
    
    // Voice activity
    j["voice_activity_ratio"] = voice_activity_ratio;
    j["voice_duration"] = voice_duration;
    j["silence_duration"] = silence_duration;
    j["total_duration"] = total_duration;
    
    // MFCC (convert to simple JSON array)
    if (!mfcc.empty()) {
        nlohmann::json mfcc_json = nlohmann::json::array();
        for (const auto& coeffs : mfcc) {
            mfcc_json.push_back(coeffs);
        }
        j["mfcc"] = mfcc_json;
    }
    
    // Mel spectrogram
    if (!mel_spectrogram.empty()) {
        nlohmann::json mel_json = nlohmann::json::array();
        for (const auto& frame : mel_spectrogram) {
            mel_json.push_back(frame);
        }
        j["mel_spectrogram"] = mel_json;
    }
    
    return j;
}

AcousticFeatures AcousticFeatures::fromJson(const nlohmann::json& data) {
    AcousticFeatures features;
    
    try {
        // Pitch
        if (data.contains("pitch_mean") && data["pitch_mean"].is_number()) 
            features.pitch_mean = data["pitch_mean"].get<double>();
        if (data.contains("pitch_std") && data["pitch_std"].is_number()) 
            features.pitch_std = data["pitch_std"].get<double>();
        if (data.contains("pitch_range") && data["pitch_range"].is_number()) 
            features.pitch_range = data["pitch_range"].get<double>();
        if (data.contains("pitch_min") && data["pitch_min"].is_number()) 
            features.pitch_min = data["pitch_min"].get<double>();
        if (data.contains("pitch_max") && data["pitch_max"].is_number()) 
            features.pitch_max = data["pitch_max"].get<double>();
        if (data.contains("pitch_variation") && data["pitch_variation"].is_number()) 
            features.pitch_variation = data["pitch_variation"].get<double>();
        
        // Intensity
        if (data.contains("intensity_mean") && data["intensity_mean"].is_number()) 
            features.intensity_mean = data["intensity_mean"].get<double>();
        if (data.contains("intensity_std") && data["intensity_std"].is_number()) 
            features.intensity_std = data["intensity_std"].get<double>();
        if (data.contains("intensity_min") && data["intensity_min"].is_number()) 
            features.intensity_min = data["intensity_min"].get<double>();
        if (data.contains("intensity_max") && data["intensity_max"].is_number()) 
            features.intensity_max = data["intensity_max"].get<double>();
        if (data.contains("intensity_variation") && data["intensity_variation"].is_number()) 
            features.intensity_variation = data["intensity_variation"].get<double>();
        if (data.contains("dynamic_range") && data["dynamic_range"].is_number()) 
            features.dynamic_range = data["dynamic_range"].get<double>();
        
        // Pauses
        if (data.contains("pause_count") && data["pause_count"].is_number()) 
            features.pause_count = data["pause_count"].get<double>();
        if (data.contains("pause_total_duration") && data["pause_total_duration"].is_number()) 
            features.pause_total_duration = data["pause_total_duration"].get<double>();
        if (data.contains("pause_mean_duration") && data["pause_mean_duration"].is_number()) 
            features.pause_mean_duration = data["pause_mean_duration"].get<double>();
        if (data.contains("pause_std_duration") && data["pause_std_duration"].is_number()) 
            features.pause_std_duration = data["pause_std_duration"].get<double>();
        if (data.contains("pause_ratio") && data["pause_ratio"].is_number()) 
            features.pause_ratio = data["pause_ratio"].get<double>();
        if (data.contains("pause_min_duration") && data["pause_min_duration"].is_number()) 
            features.pause_min_duration = data["pause_min_duration"].get<double>();
        if (data.contains("pause_max_duration") && data["pause_max_duration"].is_number()) 
            features.pause_max_duration = data["pause_max_duration"].get<double>();
        
        // Speech
        if (data.contains("speech_rate") && data["speech_rate"].is_number()) 
            features.speech_rate = data["speech_rate"].get<double>();
        if (data.contains("syllable_count") && data["syllable_count"].is_number()) 
            features.syllable_count = data["syllable_count"].get<double>();
        if (data.contains("voiced_segments") && data["voiced_segments"].is_number()) 
            features.voiced_segments = data["voiced_segments"].get<double>();
        if (data.contains("speech_duration") && data["speech_duration"].is_number()) 
            features.speech_duration = data["speech_duration"].get<double>();
        
        // Voice activity
        if (data.contains("voice_activity_ratio") && data["voice_activity_ratio"].is_number()) 
            features.voice_activity_ratio = data["voice_activity_ratio"].get<double>();
        if (data.contains("voice_duration") && data["voice_duration"].is_number()) 
            features.voice_duration = data["voice_duration"].get<double>();
        if (data.contains("silence_duration") && data["silence_duration"].is_number()) 
            features.silence_duration = data["silence_duration"].get<double>();
        if (data.contains("total_duration") && data["total_duration"].is_number()) 
            features.total_duration = data["total_duration"].get<double>();
        
        // MFCC
        if (data.contains("mfcc") && data["mfcc"].is_array()) {
            for (const auto& coeffs : data["mfcc"]) {
                if (coeffs.is_array()) {
                    std::vector<float> row;
                    for (const auto& val : coeffs) {
                        if (val.is_number()) {
                            row.push_back(val.get<float>());
                        }
                    }
                    if (!row.empty()) {
                        features.mfcc.push_back(row);
                    }
                }
            }
        }
        
        // Mel spectrogram
        if (data.contains("mel_spectrogram") && data["mel_spectrogram"].is_array()) {
            for (const auto& frame : data["mel_spectrogram"]) {
                if (frame.is_array()) {
                    std::vector<float> row;
                    for (const auto& val : frame) {
                        if (val.is_number()) {
                            row.push_back(val.get<float>());
                        }
                    }
                    if (!row.empty()) {
                        features.mel_spectrogram.push_back(row);
                    }
                }
            }
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("Error parsing AcousticFeatures from JSON: {}", e.what());
    }
    
    return features;
}

} // namespace audio