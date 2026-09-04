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