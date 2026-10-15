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
    
    double calculateVoiceActivityDrop(
        const nlohmann::json& current,
        const nlohmann::json& baseline
    );
    
private:
    //=========================================================================
    // Component Weights
    //
    // Calibrated from longitudinal effect sizes on the labelled call-center
    // set (benchmark/burnout/train). WavLM emotion probabilities are near-
    // uniform on 8 kHz recordings (range ~0.11-0.17 for all emotions), so
    // emotion-based components get reduced weight. The strongest discriminants
    // are pause structure, speech rate, and prosodic departure (symmetric —
    // some operators flatten, others become more expressive).
    //=========================================================================
    struct Weights {
        double prosodic_flattening = 0.20;  // reduced — saturates at ~0.8 for everyone
        double pause_tempo = 0.35;           // best discriminator (delta +0.14)
        double voice_activity_drop = 0.25;   // good discriminator (delta +0.11)
        double exhaustion = 0.10;            // weak emotion signal
        double negative_activation = 0.05;   // weak emotion signal
        double positive_affect_loss = 0.05;  // weakest emotion signal
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
    
    // Normalize emotion increase using RELATIVE change (ratio).
    // WavLM outputs on 8 kHz call recordings occupy a narrow range
    // (~0.11-0.17) where absolute deltas rarely reach 0.05, so we use
    // (current - baseline) / baseline instead. Deadzone: < 5 % relative
    // increase → 0; 5-25 % → linear; ≥ 25 % → 1.
    double normalizeEmotionDelta(double current, double baseline);
    
    // Normalize emotion drop using RELATIVE change.
    // Drop < 5 % relative → 0; 5-25 % → linear; ≥ 25 % → 1.
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
    
    // Confidence is a data-driven proxy for how reliable THIS recording's
    // score is, not a fixed constant: a long call with several usable windows
    // and a stored per-user baseline is far more trustworthy than a 2 s clip
    // scored against the population default. The three terms are:
    //   duration_confidence_weight      * (recording duration ramped 0..1)
    //   windows_confidence_weight       * (scored windows ramped 0..1 over >=3)
    //   baseline_reliability_confidence * baseline_reliability
    // The weights sum to 1.0 so confidence stays in [0,1]. Public so tests
    // can pin the invariant and the UI can document the model.
public:
    static constexpr double duration_confidence_weight = 0.40;
    static constexpr double windows_confidence_weight = 0.20;
    static constexpr double baseline_reliability_confidence = 0.40;
private:

    // Determine state from a SINGLE recording.
    //
    // NOTE: this intentionally has NO history parameter. The earlier
    // multi-recording logic (distinguishing SUSTAINED_STRESS / BURNOUT_LIKE
    // from SHORT_STRESS using prior recordings) was unreachable dead code:
    // every caller passed an empty history, so those states could never be
    // produced and the 0.50-0.65 band silently fell through to SUSTAINED_STRESS.
    // Until per-user history is persisted and threaded through, a single
    // recording only supports: NORMAL (low risk), LOW_AFFECT_UNSPECIFIC
    // (a specific low-risk pattern), SHORT_STRESS (mild-elevated), and
    // SUSTAINED_STRESS (persistently elevated). BURNOUT_LIKE remains a valid
    // enum value for API stability but is unreachable from a single recording
    // by design — chronic-vs-acute classification needs history, which is not
    // yet wired (see docs/code-review-burnout.md, finding H1).
    State determineState(
        double risk,
        const std::unordered_map<std::string, double>& components
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