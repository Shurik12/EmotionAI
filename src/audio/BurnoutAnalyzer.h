#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include "BurnoutModels.h"

namespace audio {

//=============================================================================
// BurnoutAnalyzer
// 
// Analyzes audio features and emotion probabilities to detect burnout risk
// Returns KEYS for frontend translations, not raw text
//=============================================================================
class BurnoutAnalyzer {
public:
    BurnoutAnalyzer();
    ~BurnoutAnalyzer() = default;
    
    // Main analysis method - returns Result with translation keys
    Result analyze(
        const nlohmann::json& current,
        const nlohmann::json& baseline,
        const std::vector<nlohmann::json>& history = {},
        double audio_quality = 0.8,
        double baseline_reliability = 0.7
    );
    
    // Component calculation methods (public for testing)
    double calculateEmotionalExhaustion(
        const nlohmann::json& current,
        const nlohmann::json& baseline
    );
    
    double calculateProsodicFlattening(
        const nlohmann::json& current,
        const nlohmann::json& baseline
    );
    
    double calculatePauseTempo(
        const nlohmann::json& current,
        const nlohmann::json& baseline
    );
    
    double calculateNegativeActivation(
        const nlohmann::json& current,
        const nlohmann::json& baseline
    );
    
    double calculatePositiveAffectLoss(
        const nlohmann::json& current,
        const nlohmann::json& baseline
    );
    
private:
    //=========================================================================
    // Component Weights (from Python implementation)
    //=========================================================================
    struct Weights {
        double exhaustion = 0.25;
        double prosodic_flattening = 0.25;
        double pause_tempo = 0.20;
        double negative_activation = 0.15;
        double positive_affect_loss = 0.15;
    } weights_;
    
    //=========================================================================
    // Helper Methods
    //=========================================================================
    
    // Extract emotion probabilities from JSON
    std::unordered_map<std::string, double> getEmotionProbabilities(
        const nlohmann::json& data
    );
    
    // Extract acoustic feature from JSON
    double getAcousticFeature(
        const nlohmann::json& data,
        const std::string& key,
        double default_val = 0.0
    );
    
    //=========================================================================
    // Normalization Functions (from Python implementation)
    //=========================================================================
    
    // Normalize emotion increase: growth <0.05 → 0; 0.05–0.20 → linear; ≥0.20 → 1
    double normalizeEmotionDelta(double current, double baseline);
    
    // Normalize emotion drop: drop <0.05 → 0; 0.05–0.20 → linear; ≥0.20 → 1
    double normalizeEmotionDrop(double baseline, double current);
    
    // Normalize acoustic drop: drop <10% → 0; 10–35% → linear; ≥35% → 1
    double normalizeAcousticDrop(
        double baseline, 
        double current, 
        double deadzone = 0.10, 
        double threshold = 0.35
    );
    
    // Normalize acoustic increase: increase <15% → 0; 15–50% → linear; ≥50% → 1
    double normalizeAcousticIncrease(
        double baseline, 
        double current,
        double deadzone = 0.15, 
        double threshold = 0.50
    );
    
    // Normalize speech rate drop: drop <10% → 0; 10–30% → linear; ≥30% → 1
    double normalizeSpeechRateDrop(double baseline, double current);
    
    //=========================================================================
    // Result Determination
    //=========================================================================
    
    // Determine level from risk score
    std::pair<Level, std::string> determineLevel(double risk);
    
    // Get the dominant factor
    std::string getTopFactor(const std::unordered_map<std::string, double>& components);
    
    // Determine state with history-aware logic
    State determineState(
        double risk,
        const std::unordered_map<std::string, double>& components,
        const std::vector<nlohmann::json>& history
    );
    
    // Generate recommendation KEYS for frontend translations
    std::vector<std::string> generateRecommendationKeys(
        Level level,
        const std::unordered_map<std::string, double>& components
    );
    
    // Generate comment KEY for frontend translations
    std::string getCommentKey(State state, size_t history_size) const;
};

} // namespace audio