#include "BurnoutModels.h"
#include <logging/Logger.h>

namespace audio {

nlohmann::json Result::toJson() const {
    nlohmann::json j;
    
    j["state"] = stateToString(state);
    j["level"] = levelToString(level);
    j["risk"] = risk;
    j["score"] = score;
    j["confidence"] = confidence;
    j["top_factor"] = top_factor;
    j["comment"] = comment;
    
    if (!error.empty()) {
        j["error"] = error;
    }
    
    // Components
    nlohmann::json comps = nlohmann::json::object();
    for (const auto& [key, value] : components) {
        comps[key] = value;
    }
    j["components"] = comps;
    
    // Recommendations - these are KEYS, not text
    j["recommendations"] = recommendations;
    
    return j;
}

Result Result::fromJson(const nlohmann::json& data) {
    Result result;
    
    try {
        if (data.contains("state") && data["state"].is_string()) {
            result.state = stringToState(data["state"].get<std::string>());
        }
        
        if (data.contains("level") && data["level"].is_string()) {
            result.level = stringToLevel(data["level"].get<std::string>());
        }
        
        if (data.contains("risk") && data["risk"].is_number()) {
            result.risk = data["risk"].get<double>();
        }
        
        if (data.contains("score") && data["score"].is_number()) {
            result.score = data["score"].get<double>();
        }
        
        if (data.contains("confidence") && data["confidence"].is_number()) {
            result.confidence = data["confidence"].get<double>();
        }
        
        if (data.contains("top_factor") && data["top_factor"].is_string()) {
            result.top_factor = data["top_factor"].get<std::string>();
        }
        
        if (data.contains("comment") && data["comment"].is_string()) {
            result.comment = data["comment"].get<std::string>();
        }
        
        if (data.contains("error") && data["error"].is_string()) {
            result.error = data["error"].get<std::string>();
        }
        
        if (data.contains("components") && data["components"].is_object()) {
            for (auto& [key, value] : data["components"].items()) {
                if (value.is_number()) {
                    result.components[key] = value.get<double>();
                }
            }
        }
        
        if (data.contains("recommendations") && data["recommendations"].is_array()) {
            for (const auto& rec : data["recommendations"]) {
                if (rec.is_string()) {
                    result.recommendations.push_back(rec.get<std::string>());
                }
            }
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("Error parsing Result from JSON: {}", e.what());
        result.error = std::string("Parse error: ") + e.what();
    }
    
    return result;
}

std::string stateToString(State state) {
    switch (state) {
        case State::NORMAL: return "NORMAL";
        case State::SHORT_STRESS: return "SHORT_STRESS";
        case State::SUSTAINED_STRESS: return "SUSTAINED_STRESS";
        case State::BURNOUT_LIKE: return "BURNOUT_LIKE";
        case State::LOW_AFFECT_UNSPECIFIC: return "LOW_AFFECT_UNSPECIFIC";
        case State::INSUFFICIENT_DATA: return "INSUFFICIENT_DATA";
        default: return "UNKNOWN";
    }
}

State stringToState(const std::string& str) {
    if (str == "NORMAL") return State::NORMAL;
    if (str == "SHORT_STRESS") return State::SHORT_STRESS;
    if (str == "SUSTAINED_STRESS") return State::SUSTAINED_STRESS;
    if (str == "BURNOUT_LIKE") return State::BURNOUT_LIKE;
    if (str == "LOW_AFFECT_UNSPECIFIC") return State::LOW_AFFECT_UNSPECIFIC;
    return State::INSUFFICIENT_DATA;
}

std::string levelToString(Level level) {
    switch (level) {
        case Level::LOW: return "low";
        case Level::MODERATE: return "moderate";
        case Level::HIGH: return "high";
        case Level::SEVERE: return "severe";
        default: return "unknown";
    }
}

Level stringToLevel(const std::string& str) {
    if (str == "low") return Level::LOW;
    if (str == "moderate") return Level::MODERATE;
    if (str == "high") return Level::HIGH;
    if (str == "severe") return Level::SEVERE;
    return Level::LOW;
}

nlohmann::json BurnoutConfig::defaultBaselineJson() const {
    return {
        {"acoustic_features", {
            {"pitch_variation", base_pitch_variation},
            {"pitch_range", base_pitch_range},
            {"intensity_variation", base_intensity_variation},
            {"pause_ratio", base_pause_ratio},
            {"pause_mean_duration", base_pause_mean_duration},
            {"pause_max_duration", base_pause_max_duration},
            {"speech_rate", base_speech_rate}
        }},
        {"additional_probs", {
            {"neutral", base_neutral},
            {"happy", base_happy},
            {"sad", base_sad},
            {"angry", base_angry},
            {"fear", base_fear},
            {"disgust", base_disgust}
        }},
        {"detailed_analysis", {
            {"wavlm_emotion_probabilities", {
                {"neutral", base_neutral},
                {"happy", base_happy},
                {"sad", base_sad},
                {"angry", base_angry},
                {"fear", base_fear},
                {"disgust", base_disgust}
            }}
        }}
    };
}

nlohmann::json BurnoutConfig::toJson() const {
    return {
        {"window_seconds", window_seconds},
        {"max_windows", max_windows},
        {"min_valid_windows", min_valid_windows},
        {"aggregation", aggregation},
        {"duration_confidence_threshold_s", duration_confidence_threshold_s},
        {"min_voice_activity_ratio", min_voice_activity_ratio},
        {"base_pitch_variation", base_pitch_variation},
        {"base_pitch_range", base_pitch_range},
        {"base_intensity_variation", base_intensity_variation},
        {"base_pause_ratio", base_pause_ratio},
        {"base_pause_mean_duration", base_pause_mean_duration},
        {"base_pause_max_duration", base_pause_max_duration},
        {"base_speech_rate", base_speech_rate},
        {"base_neutral", base_neutral},
        {"base_happy", base_happy},
        {"base_sad", base_sad},
        {"base_angry", base_angry},
        {"base_fear", base_fear},
        {"base_disgust", base_disgust},
    };
}

void BurnoutConfig::fromJson(const nlohmann::json& data) {
    if (!data.is_object()) return;
    // Each field: read if present and of the right type, else keep the default.
    if (data.contains("window_seconds") && data["window_seconds"].is_number())
        window_seconds = data["window_seconds"].get<double>();
    if (data.contains("max_windows") && data["max_windows"].is_number())
        max_windows = data["max_windows"].get<int>();
    if (data.contains("min_valid_windows") && data["min_valid_windows"].is_number())
        min_valid_windows = data["min_valid_windows"].get<int>();
    if (data.contains("aggregation") && data["aggregation"].is_string())
        aggregation = data["aggregation"].get<std::string>();
    if (data.contains("duration_confidence_threshold_s") && data["duration_confidence_threshold_s"].is_number())
        duration_confidence_threshold_s = data["duration_confidence_threshold_s"].get<double>();
    if (data.contains("min_voice_activity_ratio") && data["min_voice_activity_ratio"].is_number())
        min_voice_activity_ratio = data["min_voice_activity_ratio"].get<double>();

    if (data.contains("base_pitch_variation") && data["base_pitch_variation"].is_number())
        base_pitch_variation = data["base_pitch_variation"].get<double>();
    if (data.contains("base_pitch_range") && data["base_pitch_range"].is_number())
        base_pitch_range = data["base_pitch_range"].get<double>();
    if (data.contains("base_intensity_variation") && data["base_intensity_variation"].is_number())
        base_intensity_variation = data["base_intensity_variation"].get<double>();
    if (data.contains("base_pause_ratio") && data["base_pause_ratio"].is_number())
        base_pause_ratio = data["base_pause_ratio"].get<double>();
    if (data.contains("base_pause_mean_duration") && data["base_pause_mean_duration"].is_number())
        base_pause_mean_duration = data["base_pause_mean_duration"].get<double>();
    if (data.contains("base_pause_max_duration") && data["base_pause_max_duration"].is_number())
        base_pause_max_duration = data["base_pause_max_duration"].get<double>();
    if (data.contains("base_speech_rate") && data["base_speech_rate"].is_number())
        base_speech_rate = data["base_speech_rate"].get<double>();

    if (data.contains("base_neutral") && data["base_neutral"].is_number())
        base_neutral = data["base_neutral"].get<double>();
    if (data.contains("base_happy") && data["base_happy"].is_number())
        base_happy = data["base_happy"].get<double>();
    if (data.contains("base_sad") && data["base_sad"].is_number())
        base_sad = data["base_sad"].get<double>();
    if (data.contains("base_angry") && data["base_angry"].is_number())
        base_angry = data["base_angry"].get<double>();
    if (data.contains("base_fear") && data["base_fear"].is_number())
        base_fear = data["base_fear"].get<double>();
    if (data.contains("base_disgust") && data["base_disgust"].is_number())
        base_disgust = data["base_disgust"].get<double>();
}

} // namespace audio