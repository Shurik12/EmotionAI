#include "BurnoutAnalyzer.h"
#include <logging/Logger.h>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace audio {

//=============================================================================
// Constructor
//=============================================================================
BurnoutAnalyzer::BurnoutAnalyzer() {
    LOG_INFO("BurnoutAnalyzer initialized");
}

//=============================================================================
// Main analyze method
//=============================================================================
Result BurnoutAnalyzer::analyze(
    const nlohmann::json& current,
    const nlohmann::json& baseline,
    const std::vector<nlohmann::json>& history,
    double audio_quality,
    double baseline_reliability)
{
    Result result;
    
    try {
        LOG_INFO("Starting burnout analysis...");
        
        // Check baseline
        if (baseline.empty() || baseline.is_null()) {
            result.state = State::INSUFFICIENT_DATA;
            result.error = "Baseline not provided";
            result.recommendations = {"⚠️ Baseline required for analysis"};
            LOG_WARN("Baseline not provided");
            return result;
        }
        
        // 1. Calculate components
        std::unordered_map<std::string, double> components;
        components["exhaustion"] = calculateEmotionalExhaustion(current, baseline);
        components["prosodic_flattening"] = calculateProsodicFlattening(current, baseline);
        components["pause_tempo"] = calculatePauseTempo(current, baseline);
        components["negative_activation"] = calculateNegativeActivation(current, baseline);
        components["positive_affect_loss"] = calculatePositiveAffectLoss(current, baseline);
        
        // 2. Calculate raw risk
        double raw_risk = 
            weights_.exhaustion * components["exhaustion"] +
            weights_.prosodic_flattening * components["prosodic_flattening"] +
            weights_.pause_tempo * components["pause_tempo"] +
            weights_.negative_activation * components["negative_activation"] +
            weights_.positive_affect_loss * components["positive_affect_loss"];
        
        raw_risk = std::min(1.0, std::max(0.0, raw_risk));
        
        // 3. Determine level
        auto [level, level_name] = determineLevel(raw_risk);
        result.level = level;
        
        // 4. Determine state
        if (raw_risk < 0.35) {
            result.state = State::NORMAL;
        } else if (raw_risk < 0.50) {
            result.state = State::SHORT_STRESS;
        } else if (raw_risk < 0.65) {
            result.state = State::SUSTAINED_STRESS;
        } else {
            result.state = State::BURNOUT_LIKE;
        }
        
        // 5. Top factor
        result.top_factor = getTopFactor(components);
        
        // 6. Confidence
        result.confidence = std::min(1.0, 
            0.30 * audio_quality +
            0.25 * baseline_reliability +
            0.25 * 0.5  // persistence without history
        );
        
        // 7. Risk and score
        result.risk = raw_risk;
        result.score = raw_risk * 100.0;
        
        // 8. Components
        result.components = components;
        
        // 9. Recommendations
        result.recommendations = generateRecommendations(level, result.top_factor, components);
        
        // 10. Comment
        result.comment = "Без истории нескольких текущих записей нельзя надежно различить short stress vs chronic burnout";
        
        LOG_INFO("Analysis complete: State={}, Score={:.1f}, Confidence={:.2f}",
                 stateToString(result.state), result.score, result.confidence);
                 
    } catch (const std::exception& e) {
        LOG_ERROR("Error in burnout analysis: {}", e.what());
        result.state = State::INSUFFICIENT_DATA;
        result.error = e.what();
        result.recommendations = {"⚠️ Error in analysis. Please try again."};
    }
    
    return result;
}

//=============================================================================
// getEmotionProbabilities
//=============================================================================
std::unordered_map<std::string, double> BurnoutAnalyzer::getEmotionProbabilities(
    const nlohmann::json& data)
{
    std::unordered_map<std::string, double> probs;
    
    // Try model_results
    if (data.contains("model_results") && data["model_results"].is_array() && 
        !data["model_results"].empty()) {
        const auto& first = data["model_results"][0];
        if (first.contains("all_probabilities") && first["all_probabilities"].is_object()) {
            for (auto& [key, value] : first["all_probabilities"].items()) {
                if (value.is_number()) probs[key] = value.get<double>();
            }
            return probs;
        }
    }
    
    // Try additional_probs (EmotionAI format)
    if (data.contains("additional_probs") && data["additional_probs"].is_object()) {
        for (auto& [key, value] : data["additional_probs"].items()) {
            if (value.is_string()) {
                try { probs[key] = std::stod(value.get<std::string>()); }
                catch (...) { probs[key] = 0.0; }
            } else if (value.is_number()) {
                probs[key] = value.get<double>();
            }
        }
        return probs;
    }
    
    // Try detailed_analysis
    if (data.contains("detailed_analysis") && data["detailed_analysis"].is_object()) {
        for (auto& [key, value] : data["detailed_analysis"].items()) {
            if (key.find("probabilities") != std::string::npos && value.is_object()) {
                for (auto& [emotion, prob] : value.items()) {
                    if (prob.is_number()) probs[emotion] = prob.get<double>();
                }
                break;
            }
        }
    }
    
    // Default values
    if (probs.empty()) {
        probs["neutral"] = 0.2;
        probs["happy"] = 0.3;
        probs["sad"] = 0.1;
        probs["angry"] = 0.05;
        probs["fear"] = 0.05;
        probs["disgust"] = 0.03;
        probs["surprise"] = 0.05;
    }
    
    return probs;
}

//=============================================================================
// getAcousticFeature
//=============================================================================
double BurnoutAnalyzer::getAcousticFeature(
    const nlohmann::json& data,
    const std::string& key,
    double default_val)
{
    if (data.contains("acoustic_features") && data["acoustic_features"].is_object()) {
        const auto& features = data["acoustic_features"];
        if (features.contains(key)) {
            const auto& val = features[key];
            if (val.is_number()) return val.get<double>();
            if (val.is_string()) {
                try { return std::stod(val.get<std::string>()); }
                catch (...) { return default_val; }
            }
        }
    }
    return default_val;
}

//=============================================================================
// Normalization Functions
//=============================================================================
double BurnoutAnalyzer::normalizeEmotionDelta(double current, double baseline) {
    double delta = current - baseline;
    if (delta < 0.05) return 0.0;
    if (delta >= 0.20) return 1.0;
    return (delta - 0.05) / 0.15;
}

double BurnoutAnalyzer::normalizeEmotionDrop(double baseline, double current) {
    double delta = baseline - current;
    if (delta < 0.05) return 0.0;
    if (delta >= 0.20) return 1.0;
    return (delta - 0.05) / 0.15;
}

double BurnoutAnalyzer::normalizeAcousticDrop(double baseline, double current, 
                                               double deadzone, double threshold) {
    if (baseline <= 0) return 0.0;
    double relative_drop = (baseline - current) / baseline;
    if (relative_drop < deadzone) return 0.0;
    if (relative_drop >= threshold) return 1.0;
    return (relative_drop - deadzone) / (threshold - deadzone);
}

double BurnoutAnalyzer::normalizeAcousticIncrease(double baseline, double current,
                                                   double deadzone, double threshold) {
    if (baseline <= 0) return (current < 0.05) ? 0.0 : 1.0;
    double relative_increase = (current - baseline) / baseline;
    if (relative_increase < deadzone) return 0.0;
    if (relative_increase >= threshold) return 1.0;
    return (relative_increase - deadzone) / (threshold - deadzone);
}

double BurnoutAnalyzer::normalizeSpeechRateDrop(double baseline, double current) {
    if (baseline <= 0 || current == 0) return 0.0;
    double relative_drop = (baseline - current) / baseline;
    if (relative_drop < 0.10) return 0.0;
    if (relative_drop >= 0.30) return 1.0;
    return (relative_drop - 0.10) / 0.20;
}

//=============================================================================
// Component Calculations
//=============================================================================
double BurnoutAnalyzer::calculateEmotionalExhaustion(
    const nlohmann::json& current,
    const nlohmann::json& baseline)
{
    auto current_probs = getEmotionProbabilities(current);
    auto baseline_probs = getEmotionProbabilities(baseline);
    
    double sad = normalizeEmotionDelta(
        current_probs.count("sad") ? current_probs["sad"] : 0.0,
        baseline_probs.count("sad") ? baseline_probs["sad"] : 0.0
    );
    
    double neutral = normalizeEmotionDelta(
        current_probs.count("neutral") ? current_probs["neutral"] : 0.0,
        baseline_probs.count("neutral") ? baseline_probs["neutral"] : 0.0
    );
    
    return 0.70 * sad + 0.30 * neutral;
}

double BurnoutAnalyzer::calculateProsodicFlattening(
    const nlohmann::json& current,
    const nlohmann::json& baseline)
{
    double pitch_var = normalizeAcousticDrop(
        getAcousticFeature(baseline, "pitch_variation", 0.1),
        getAcousticFeature(current, "pitch_variation", 0.1)
    );
    
    double pitch_range = normalizeAcousticDrop(
        getAcousticFeature(baseline, "pitch_range", 50.0),
        getAcousticFeature(current, "pitch_range", 50.0)
    );
    
    double intensity_var = normalizeAcousticDrop(
        getAcousticFeature(baseline, "intensity_variation", 0.1),
        getAcousticFeature(current, "intensity_variation", 0.1)
    );
    
    return 0.50 * pitch_var + 0.30 * pitch_range + 0.20 * intensity_var;
}

double BurnoutAnalyzer::calculatePauseTempo(
    const nlohmann::json& current,
    const nlohmann::json& baseline)
{
    double pause_ratio = normalizeAcousticIncrease(
        getAcousticFeature(baseline, "pause_ratio", 0.1),
        getAcousticFeature(current, "pause_ratio", 0.1)
    );
    
    double pause_mean = normalizeAcousticIncrease(
        getAcousticFeature(baseline, "pause_mean_duration", 0.05),
        getAcousticFeature(current, "pause_mean_duration", 0.05)
    );
    
    double pause_max = normalizeAcousticIncrease(
        getAcousticFeature(baseline, "pause_max_duration", 0.1),
        getAcousticFeature(current, "pause_max_duration", 0.1)
    );
    
    double speech_rate = getAcousticFeature(current, "speech_rate", 0.0);
    double baseline_speech = getAcousticFeature(baseline, "speech_rate", 3.0);
    
    if (speech_rate > 0 && baseline_speech > 0) {
        double speech_drop = normalizeSpeechRateDrop(baseline_speech, speech_rate);
        return 0.40 * pause_ratio + 0.20 * pause_mean + 0.10 * pause_max + 0.30 * speech_drop;
    } else {
        return (0.40 / 0.70) * pause_ratio + (0.20 / 0.70) * pause_mean + (0.10 / 0.70) * pause_max;
    }
}

double BurnoutAnalyzer::calculateNegativeActivation(
    const nlohmann::json& current,
    const nlohmann::json& baseline)
{
    auto current_probs = getEmotionProbabilities(current);
    auto baseline_probs = getEmotionProbabilities(baseline);
    
    double angry = normalizeEmotionDelta(
        current_probs.count("angry") ? current_probs["angry"] : 0.0,
        baseline_probs.count("angry") ? baseline_probs["angry"] : 0.0
    );
    
    double fear = normalizeEmotionDelta(
        current_probs.count("fear") ? current_probs["fear"] : 0.0,
        baseline_probs.count("fear") ? baseline_probs["fear"] : 0.0
    );
    
    double disgust = normalizeEmotionDelta(
        current_probs.count("disgust") ? current_probs["disgust"] : 0.0,
        baseline_probs.count("disgust") ? baseline_probs["disgust"] : 0.0
    );
    
    return 0.40 * angry + 0.35 * fear + 0.25 * disgust;
}

double BurnoutAnalyzer::calculatePositiveAffectLoss(
    const nlohmann::json& current,
    const nlohmann::json& baseline)
{
    auto current_probs = getEmotionProbabilities(current);
    auto baseline_probs = getEmotionProbabilities(baseline);
    
    return normalizeEmotionDrop(
        baseline_probs.count("happy") ? baseline_probs["happy"] : 0.0,
        current_probs.count("happy") ? current_probs["happy"] : 0.0
    );
}

//=============================================================================
// Result Determination
//=============================================================================
std::pair<Level, std::string> BurnoutAnalyzer::determineLevel(double risk) {
    if (risk < 0.35) return {Level::LOW, "normal"};
    if (risk < 0.50) return {Level::MODERATE, "mild"};
    if (risk < 0.65) return {Level::HIGH, "moderate"};
    return {Level::SEVERE, "severe"};
}

std::string BurnoutAnalyzer::getTopFactor(
    const std::unordered_map<std::string, double>& components)
{
    std::string top = "Unknown";
    double max_val = -1.0;
    
    std::unordered_map<std::string, std::string> names = {
        {"exhaustion", "Emotional Exhaustion"},
        {"prosodic_flattening", "Prosodic Flattening"},
        {"pause_tempo", "Pause/Tempo Changes"},
        {"negative_activation", "Negative Activation"},
        {"positive_affect_loss", "Positive Affect Loss"}
    };
    
    for (const auto& [key, value] : components) {
        if (value > max_val) {
            max_val = value;
            top = names.count(key) ? names.at(key) : key;
        }
    }
    
    return top;
}

std::vector<std::string> BurnoutAnalyzer::generateRecommendations(
    Level level,
    const std::string& top_factor,
    const std::unordered_map<std::string, double>& components)
{
    std::vector<std::string> recs;
    
    switch (level) {
        case Level::SEVERE:
            recs = {
                "🚨 URGENT: Seek professional psychological help immediately",
                "Take medical leave if possible",
                "Contact employee assistance program",
                "Reduce work hours and delegate tasks",
                "Practice self-care and stress management techniques"
            };
            break;
        case Level::HIGH:
            recs = {
                "⚠️ Moderate risk detected - take action to prevent burnout",
                "Schedule regular breaks throughout the day",
                "Consider therapy or counseling sessions",
                "Implement relaxation techniques",
                "Discuss workload with your supervisor"
            };
            break;
        case Level::MODERATE:
            recs = {
                "⚡ Mild risk detected - monitor your state",
                "Take regular breaks and practice mindfulness",
                "Ensure adequate sleep and nutrition",
                "Exercise and physical activity recommended"
            };
            break;
        default:
            recs = {
                "✅ Normal emotional state detected",
                "Continue maintaining healthy habits",
                "Practice preventive self-care",
                "Regular check-ups recommended"
            };
    }
    
    // Add specific recommendations
    auto it = components.find("exhaustion");
    if (it != components.end() && it->second > 0.5) {
        recs.push_back("Emotional exhaustion detected - prioritize mental health");
    }
    it = components.find("prosodic_flattening");
    if (it != components.end() && it->second > 0.5) {
        recs.push_back("Voice monotony detected - speech therapy may help");
    }
    it = components.find("pause_tempo");
    if (it != components.end() && it->second > 0.5) {
        recs.push_back("Speech pattern changes - consider vocal rest");
    }
    it = components.find("negative_activation");
    if (it != components.end() && it->second > 0.5) {
        recs.push_back("High negative activation - consider anger/stress management");
    }
    it = components.find("positive_affect_loss");
    if (it != components.end() && it->second > 0.5) {
        recs.push_back("Reduced positive affect - consider activities that boost mood");
    }
    
    return recs;
}

} // namespace audio