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
    
    // Recommendations
    j["recommendations"] = recommendations;
    
    return j;
}

//=============================================================================
// Result: fromJson
//=============================================================================
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

nlohmann::json getDefaultBaseline() {
    return {
        {"acoustic_features", {
            {"pitch_variation", 0.19},
            {"pitch_range", 60.0},
            {"intensity_variation", 0.3},
            {"pause_ratio", 0.15},
            {"pause_mean_duration", 0.05},
            {"pause_max_duration", 0.10},
            {"speech_rate", 3.5}
        }},
        {"detailed_analysis", {
            {"wavlm_emotion_probabilities", {
                {"neutral", 0.2},
                {"happy", 0.3},
                {"sad", 0.1},
                {"angry", 0.05},
                {"fear", 0.05},
                {"disgust", 0.03},
                {"surprise", 0.05}
            }}
        }},
        {"model_results", {
            {
                {"all_probabilities", {
                    {"neutral", 0.2},
                    {"happy", 0.3},
                    {"sad", 0.1},
                    {"angry", 0.05},
                    {"fear", 0.05},
                    {"disgust", 0.03},
                    {"surprise", 0.05}
                }}
            }
        }}
    };
}

} // namespace audio