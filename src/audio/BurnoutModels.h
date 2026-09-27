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
// values from benchmark/control (see tests/benchmark/BurnoutBenchmark.cpp):
// the previous built-in baseline (happy=0.30, pause_ratio=0.10, ...) never
// matched real model output, so positive_affect_loss and pause_tempo saturated
// for every recording and the score became near-constant (~0.5).
//
// The acoustic defaults below were re-measured on the current benchmark/control
// set (n=20). The previous acoustic values were 1.3-2.2x higher than the set's
// medians (pitch_range 128.3 vs 77.1, speech_rate 6.10 vs 4.30,
// pause_mean_duration 0.362 vs 0.167, ...), so every recording was scored as a
// large relative drop and `prosodic_flattening` / `pause_tempo` saturated on
// normal speech. The emotion defaults already match control and are unchanged.
//=============================================================================
struct BurnoutConfig {
    // Multi-window analysis: a recording is split into up to max_windows
    // windows of window_seconds, evenly spaced across the WHOLE file (not
    // just the start). Per-window emotion probabilities and acoustic features
    // are aggregated with `aggregation` ("median" or "mean") before scoring.
    double window_seconds = 10.0;   // WavLM input ceiling
    int max_windows = 5;
    int min_valid_windows = 1;      // fewer usable windows -> INSUFFICIENT_DATA
    std::string aggregation = "median";

    // A window whose voiced fraction is below this is rejected instead of
    // being scored. The WavLM model emits near-uniform probabilities on
    // non-speech, which the component math would otherwise turn into a
    // confident-looking risk.
    double min_voice_activity_ratio = 0.15;

    // Default baseline (medians over the non-burnout call-center operators
    // in benchmark/burnout/train — operators 7-11 plus pre-transition
    // recordings of operators 2-6, n≈200). The previous values were from
    // benchmark/control which did not match call-center data. See
    // BurnoutAnalyzer.cpp for the analysis that drove these numbers.
    double base_pitch_variation = 0.1012;
    double base_pitch_range = 66.77;
    double base_intensity_variation = 1.4528;
    double base_pause_ratio = 0.3925;
    double base_pause_mean_duration = 0.1799;
    double base_pause_max_duration = 1.165;
    double base_speech_rate = 4.285;

    // WavLM emotion probabilities on 8 kHz call recordings are compressed
    // into a narrow range (~0.11-0.17) regardless of label, so all emotion
    // baselines are near the uniform midpoint. The emotion delta deadzones
    // in BurnoutAnalyzer have been switched to RELATIVE (ratio) thresholds
    // because absolute deltas never cross the old 0.05 deadzone.
    double base_neutral = 0.1650;
    double base_happy = 0.1209;
    double base_sad = 0.1124;
    double base_angry = 0.1274;
    double base_fear = 0.1550;
    double base_disgust = 0.1712;

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