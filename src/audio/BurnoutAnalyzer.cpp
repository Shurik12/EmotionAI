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
// Helper: getNestedData - unwraps nested JSON structures
//=============================================================================
static nlohmann::json getNestedData(const nlohmann::json& data) {
    // If data has a "result" field with the actual data (common in API responses)
    if (data.contains("result") && data["result"].is_object()) {
        return data["result"];
    }
    // If data has a "data" field with the actual data
    if (data.contains("data") && data["data"].is_object()) {
        return data["data"];
    }
    return data;
}

//=============================================================================
// Label canonicalization
//
// The acoustic model (wavlm-emotion-russian-resd) emits seven classes, see
// Audio::EMOTION_LABELS: anger, disgust, enthusiasm, fear, happiness,
// neutral, sadness. Burnout components are defined over the canonical
// vocabulary below, so every probability object is translated through this
// table on read. Keeping the mapping in one explicit place means swapping
// the acoustic model cannot silently rename classes out from under the
// scorer (a missing key used to contribute 0.0 silently).
//
// "enthusiasm" is a positive high-arousal state, not a valence label, so it
// is deliberately NOT folded into "happy": the five-component model was
// calibrated with "happiness" only, and merging the two would change its
// semantics. No component consumes it today; it stays canonicalized so the
// intent is visible instead of it becoming an orphan key.
//=============================================================================
static const std::unordered_map<std::string, std::string>& canonicalEmotionLabels()
{
    static const std::unordered_map<std::string, std::string> kCanonical = {
        {"anger", "angry"},       {"angry", "angry"},
        {"happiness", "happy"},   {"happy", "happy"},
        {"sadness", "sad"},       {"sad", "sad"},
        {"fear", "fear"},         {"fearful", "fear"},
        {"disgust", "disgust"},   {"disgusted", "disgust"},
        {"surprise", "surprise"}, {"surprised", "surprise"},
        {"neutral", "neutral"},
        {"enthusiasm", "enthusiasm"},
    };
    return kCanonical;
}

// Writes canonical_label -> probability into `out` for every numeric entry
// of `probs_json`. First writer wins, so a payload that carries both a
// canonical key and its synonym keeps the canonical value. Returns true if
// at least one probability was extracted.
static bool extractCanonicalProbabilities(
    const nlohmann::json& probs_json,
    std::unordered_map<std::string, double>& out)
{
    if (!probs_json.is_object()) return false;

    const auto& labels = canonicalEmotionLabels();
    size_t extracted = 0;

    for (const auto& [key, value] : probs_json.items()) {
        double prob = 0.0;
        if (value.is_number()) {
            prob = value.get<double>();
        } else if (value.is_string()) {
            try {
                prob = std::stod(value.get<std::string>());
            } catch (...) {
                continue;  // skip non-numeric strings
            }
        } else {
            continue;
        }

        const auto it = labels.find(key);
        const std::string& canonical = (it != labels.end()) ? it->second : key;
        if (out.find(canonical) == out.end()) {
            out[canonical] = prob;
            ++extracted;
        }
    }

    return extracted > 0;
}

//=============================================================================
// Main analyze method - Returns keys for frontend translations
//=============================================================================
Result BurnoutAnalyzer::analyze(
    const nlohmann::json& current_raw,
    const nlohmann::json& baseline_raw,
    const std::vector<nlohmann::json>& history,
    double audio_quality,
    double baseline_reliability)
{
    Result result;
    
    try {
        LOG_INFO("Starting burnout analysis...");
        
        // Unwrap nested data structures
        const nlohmann::json current = getNestedData(current_raw);
        const nlohmann::json baseline = getNestedData(baseline_raw);
        
        // Check baseline
        if (baseline.empty() || baseline.is_null()) {
            result.state = State::INSUFFICIENT_DATA;
            result.error = "analysis_failed";
            result.recommendations = {"analysis_failed_retry"};
            LOG_WARN("Baseline not provided");
            return result;
        }

        // Emotion probabilities are a hard input: three of the five
        // components (55% of the weight) are emotion deltas. Fabricated
        // defaults here used to turn an emotion-model failure into a
        // confident-looking NORMAL. Refuse to score instead.
        const bool has_current_emotions = !getEmotionProbabilities(current).empty();
        const bool has_baseline_emotions = !getEmotionProbabilities(baseline).empty();
        if (!has_current_emotions || !has_baseline_emotions) {
            result.state = State::INSUFFICIENT_DATA;
            result.error = "analysis_failed";
            result.recommendations = {"analysis_failed_retry"};
            LOG_WARN("Emotion probabilities missing (current={}, baseline={}); "
                     "returning INSUFFICIENT_DATA instead of a fabricated score",
                     has_current_emotions, has_baseline_emotions);
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
        result.risk = raw_risk;
        result.score = raw_risk * 100.0;
        
        // 4. Determine state (with history-aware logic)
        result.state = determineState(raw_risk, components, history);
        
        // 5. Top factor
        result.top_factor = getTopFactor(components);
        result.components = components;
        
        // 6. Confidence
        result.confidence = std::min(1.0, 
            0.30 * audio_quality +
            0.25 * baseline_reliability +
            0.25 * std::min(1.0, static_cast<double>(history.size()) / 3.0)
        );
        
        // 7. Generate recommendations as KEYS (not text)
        result.recommendations = generateRecommendationKeys(level, components);
        
        // 8. Generate comment as KEY (not text)
        result.comment = getCommentKey(result.state, history.size());
        
        // 9. For INSUFFICIENT_DATA - don't show risk level, factors, or recommendations
        if (result.state == State::INSUFFICIENT_DATA) {
            result.recommendations = {"analysis_failed_retry"};
        }
        
        LOG_INFO("Analysis complete: State={}, Score={:.1f}, Confidence={:.2f}",
                 stateToString(result.state), result.score, result.confidence);
                 
    } catch (const std::exception& e) {
        LOG_ERROR("Error in burnout analysis: {}", e.what());
        result.state = State::INSUFFICIENT_DATA;
        result.error = "analysis_failed";
        result.recommendations = {"analysis_failed_retry"};
    }
    
    return result;
}

//=============================================================================
// determineState - State determination with history-aware logic
//=============================================================================
State BurnoutAnalyzer::determineState(
    double risk,
    const std::unordered_map<std::string, double>& components,
    const std::vector<nlohmann::json>& history)
{
    // A genuine lack of input is rejected upstream: analyze() refuses when
    // emotions are missing, and Audio::add_burnout_analysis gates on speech
    // activity. With the empirically calibrated baseline a perfectly normal
    // recording can score exactly 0, which is NORMAL, not "insufficient".
    if (risk < 0.01) {
        return State::NORMAL;
    }
    
    // Check for LOW_AFFECT_UNSPECIFIC - specific pattern
    auto it_prosodic = components.find("prosodic_flattening");
    auto it_positive = components.find("positive_affect_loss");
    auto it_exhaustion = components.find("exhaustion");
    auto it_negative = components.find("negative_activation");
    
    double prosodic = (it_prosodic != components.end()) ? it_prosodic->second : 0.0;
    double positive = (it_positive != components.end()) ? it_positive->second : 0.0;
    double exhaustion = (it_exhaustion != components.end()) ? it_exhaustion->second : 0.0;
    double negative = (it_negative != components.end()) ? it_negative->second : 0.0;
    
    if (risk < 0.35) {
        if (prosodic > 0.4 && positive > 0.4 && exhaustion < 0.3 && negative < 0.3) {
            return State::LOW_AFFECT_UNSPECIFIC;
        }
        return State::NORMAL;
    }
    
    // For higher risk, check history to distinguish states
    int sustained_count = 0;
    int burnout_count = 0;
    
    for (const auto& prev : history) {
        if (prev.contains("risk") && prev["risk"].is_number()) {
            double prev_risk = prev["risk"].get<double>();
            if (prev_risk > 0.35) sustained_count++;
            if (prev_risk > 0.50) burnout_count++;
        }
    }
    
    // Determine state based on risk level and history
    if (risk < 0.50) {
        if (history.size() >= 2 && sustained_count >= 2) {
            return State::SUSTAINED_STRESS;
        }
        return State::SHORT_STRESS;
    } else if (risk < 0.65) {
        if (history.size() >= 2 && burnout_count >= 2) {
            return State::BURNOUT_LIKE;
        }
        if (history.size() >= 2 && sustained_count >= 1) {
            return State::SUSTAINED_STRESS;
        }
        return State::SUSTAINED_STRESS;
    } else {
        if (history.size() >= 2 && burnout_count >= 2) {
            return State::BURNOUT_LIKE;
        }
        return State::SUSTAINED_STRESS;
    }
}

//=============================================================================
// getCommentKey - Returns KEY for system comment
//=============================================================================
std::string BurnoutAnalyzer::getCommentKey(State state, size_t history_size) const {
    switch (state) {
        case State::SHORT_STRESS:
            if (history_size < 2) {
                return "need_history_to_distinguish_short_vs_chronic";
            }
            return "";
        case State::SUSTAINED_STRESS:
            if (history_size < 2) {
                return "need_dynamics_for_sustained_stress";
            }
            return "";
        case State::BURNOUT_LIKE:
            return "burnout_compatible_not_diagnosis";
        case State::LOW_AFFECT_UNSPECIFIC:
            return "low_affect_nonspecific_signal";
        default:
            return "";
    }
}

//=============================================================================
// generateRecommendationKeys - Returns KEYS for frontend translations
//=============================================================================
std::vector<std::string> BurnoutAnalyzer::generateRecommendationKeys(
    Level level,
    const std::unordered_map<std::string, double>& components)
{
    std::vector<std::string> keys;
    
    switch (level) {
        case Level::SEVERE:
            keys = {
                "severe_risk_immediate_action",
                "contact_employee_same_day",
                "check_wellbeing_and_workload",
                "remove_non_urgent_tasks",
                "offer_support_resources"
            };
            break;
        case Level::HIGH:
            keys = {
                "high_risk_contact_supervisor",
                "discuss_workload_and_deadlines",
                "check_dynamics_after_actions"
            };
            break;
        case Level::MODERATE:
            keys = {
                "moderate_risk_preventive_measures",
                "discuss_workload_with_supervisor",
                "schedule_regular_breaks",
                "repeat_assessment_7_14_days"
            };
            break;
        default:
            keys = {
                "normal_emotional_state_detected",
                "no_additional_measures",
                "maintain_balanced_schedule",
                "regular_check_ups"
            };
    }
    
    // Add factor-based recommendations (only one per category)
    auto it = components.find("exhaustion");
    if (it != components.end() && it->second > 0.5) {
        keys.push_back("emotional_exhaustion_detected");
    }
    it = components.find("prosodic_flattening");
    if (it != components.end() && it->second > 0.5) {
        keys.push_back("voice_monotony_detected");
    }
    it = components.find("pause_tempo");
    if (it != components.end() && it->second > 0.5) {
        keys.push_back("speech_pattern_changes");
    }
    it = components.find("negative_activation");
    if (it != components.end() && it->second > 0.5) {
        keys.push_back("high_negative_activation");
    }
    it = components.find("positive_affect_loss");
    if (it != components.end() && it->second > 0.5) {
        keys.push_back("reduced_positive_affect");
    }
    
    return keys;
}

//=============================================================================
// getEmotionProbabilities - returns canonical labels, no fabricated defaults
//=============================================================================
std::unordered_map<std::string, double> BurnoutAnalyzer::getEmotionProbabilities(
    const nlohmann::json& data)
{
    std::unordered_map<std::string, double> probs;
    
    // Unwrap nested data if needed
    const nlohmann::json& input = getNestedData(data);
    
    // Try model_results (WavLM format)
    if (input.contains("model_results") && input["model_results"].is_array() && 
        !input["model_results"].empty()) {
        const auto& first = input["model_results"][0];
        if (first.contains("all_probabilities") && first["all_probabilities"].is_object()) {
            if (extractCanonicalProbabilities(first["all_probabilities"], probs)) {
                return probs;
            }
        }
        
        // Try model_results[0] directly (some formats)
        if (first.is_object()) {
            if (extractCanonicalProbabilities(first, probs)) {
                return probs;
            }
        }
    }
    
    // Try additional_probs (EmotionAI format)
    if (input.contains("additional_probs") && input["additional_probs"].is_object()) {
        if (extractCanonicalProbabilities(input["additional_probs"], probs)) {
            return probs;
        }
    }
    
    // Try detailed_analysis
    if (input.contains("detailed_analysis") && input["detailed_analysis"].is_object()) {
        const auto& detailed = input["detailed_analysis"];
        
        // Try wavlm_emotion_probabilities (common in detailed_analysis)
        if (detailed.contains("wavlm_emotion_probabilities") && 
            detailed["wavlm_emotion_probabilities"].is_object()) {
            if (extractCanonicalProbabilities(detailed["wavlm_emotion_probabilities"], probs)) {
                return probs;
            }
        }
        
        // Try any field ending with "probabilities"
        for (const auto& [key, value] : detailed.items()) {
            if (key.find("probabilities") != std::string::npos && value.is_object()) {
                if (extractCanonicalProbabilities(value, probs)) {
                    return probs;
                }
            }
        }
    }
    
    // Try top-level probabilities
    if (input.contains("probabilities") && input["probabilities"].is_object()) {
        if (extractCanonicalProbabilities(input["probabilities"], probs)) {
            return probs;
        }
    }
    
    // Try reading from main_prediction
    if (input.contains("main_prediction") && input["main_prediction"].is_object()) {
        const auto& main = input["main_prediction"];
        if (main.contains("label") && main.contains("probability")) {
            const std::string label = main["label"].get<std::string>();
            const double prob = main["probability"].get<double>();
            
            const auto& labels = canonicalEmotionLabels();
            const auto it = labels.find(label);
            probs[(it != labels.end()) ? it->second : label] = prob;
            return probs;
        }
    }
    
    // No fabricated defaults: callers must treat an empty map as "emotion
    // data unavailable" (analyze() turns this into INSUFFICIENT_DATA).
    LOG_WARN("No emotion probabilities found in analysis payload");
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
    const nlohmann::json& input = getNestedData(data);
    
    if (input.contains("acoustic_features") && input["acoustic_features"].is_object()) {
        const auto& features = input["acoustic_features"];
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

} // namespace audio