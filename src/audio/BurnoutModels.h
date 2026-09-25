#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <nlohmann/json.hpp>

namespace audio {

enum class State {
    NORMAL,
    SHORT_STRESS,
    SUSTAINED_STRESS,
    BURNOUT_LIKE,
    LOW_AFFECT_UNSPECIFIC,
    INSUFFICIENT_DATA
};

enum class Level {
    LOW,
    MODERATE,
    HIGH,
    SEVERE
};

//=============================================================================
// BurnoutConfig
//
// Knobs for the audio burnout path, populated from the optional `burnout:`
// YAML section (Config::instance()). Defaults are the empirically measured
// values from benchmark/control (see tools/burnout_harness.cpp): the previous
// built-in baseline (happy=0.30, pause_ratio=0.10, ...) never matched real
// model output, so positive_affect_loss and pause_tempo saturated for every
// recording and the score became near-constant (~0.5).
//=============================================================================
struct BurnoutConfig {
    // A window whose voiced fraction is below this is rejected as
    // INSUFFICIENT_DATA instead of being scored. The WavLM model emits
    // near-uniform probabilities on non-speech, which the component math
    // would otherwise turn into a confident-looking risk.
    double min_voice_activity_ratio = 0.15;

    // Default baseline (medians over benchmark/control). Used only when the
    // caller does not supply a stored per-user baseline.
    double base_pitch_variation = 0.1189;
    double base_pitch_range = 128.3287;
    double base_intensity_variation = 1.7564;
    double base_pause_ratio = 0.544;
    double base_pause_mean_duration = 0.3619;
    double base_pause_max_duration = 2.935;
    double base_speech_rate = 6.1047;

    double base_neutral = 0.1665;
    double base_happy = 0.1209;
    double base_sad = 0.1124;
    double base_angry = 0.1274;
    double base_fear = 0.1574;
    double base_disgust = 0.1716;

    // Build the baseline JSON consumed by BurnoutAnalyzer::analyze.
    nlohmann::json defaultBaselineJson() const;
};

struct Result {
    State state = State::INSUFFICIENT_DATA;
    Level level = Level::LOW;
    double risk = 0.0;           // 0.0 - 1.0
    double score = 0.0;          // 0 - 100
    double confidence = 0.0;     // 0.0 - 1.0
    std::string top_factor;
    std::unordered_map<std::string, double> components;
    std::vector<std::string> recommendations;  // Stores KEYS for frontend translations
    std::string comment;  // Stores a KEY for frontend translation
    std::string error;
    
    // Convert to JSON
    nlohmann::json toJson() const;
    
    // Parse from JSON
    static Result fromJson(const nlohmann::json& data);
};

std::string stateToString(State state);
std::string levelToString(Level level);
State stringToState(const std::string& str);
Level stringToLevel(const std::string& str);

nlohmann::json getDefaultBaseline();

} // namespace audio