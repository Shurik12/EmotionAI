#pragma once

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "ExternalInfluenceModels.h"
#include "BurnoutAnalyzer.h"

namespace audio {

//=============================================================================
// ExternalInfluenceAnalyzer
//
// Detects a sustained combination of emotional tension and two-sided speech
// behavior change (tempo/pauses/prosody deviating in EITHER direction) across
// >=3 consecutive fragments of one call recording, optionally confirmed by
// context flags. Follows the pilot spec:
//
//   fragmentScore = clamp(0.40*negativeActivation + 0.25*pauseTempoDeviation
//                       + 0.30*prosodicDeviation + 0.05*positiveAffectLoss, 0, 1)
//   rollingScore[i]  = median(fragmentScore[i..i+2]); callScore = max(...)
//   emotionalPattern[i] = negativeActivation[i]      >= 0.50
//   speechPattern[i]    = pauseTempoDeviation[i]     >= 0.50
//                         OR prosodicDeviation[i]    >= 0.50
//   combinedPattern[i]  = emotionalPattern[i] AND speechPattern[i]
//   persistentPattern   = combinedPattern in >=2 of the 3 best-window fragments
//   contextConfirmed    = strongContextCount >=1 OR moderateContextCount >=2
//
// Input fragments are JSON records produced by the audio pipeline:
//   { index, start_time, duration_seconds, model_probability,
//     additional_probs: {label: prob}, acoustic_features: {...} }
// All spec weights/boundaries/minimums live in ExternalInfluenceConfig.
//
// NOTE: the WavLM model emits labels "anger"/"happiness" while BurnoutAnalyzer
// reads "angry"/"happy"; fragment probs are canonicalized before delegation.
//=============================================================================
class ExternalInfluenceAnalyzer {
public:
    ExternalInfluenceAnalyzer();
    explicit ExternalInfluenceAnalyzer(const ExternalInfluenceConfig& config);

    ExternalInfluenceResult analyze(
        const nlohmann::json& fragments,      // array of fragment records
        const nlohmann::json& baseline,       // {} -> audio::getDefaultBaseline()
        double audio_quality,                 // whole-call quality, 0..1
        const std::vector<std::string>& context_flags = {}) const;

private:
    ExternalInfluenceConfig cfg_;

    // Fragment component scores
    struct FragmentAnalysis {
        double negativeActivation = 0.0;
        double pauseTempoDeviation = 0.0;
        double prosodicDeviation = 0.0;
        double positiveAffectLoss = 0.0;   // -1.0 signals "absent"
        bool valid = false;
    };

    FragmentAnalysis analyzeFragment(const nlohmann::json& fragment,
                                     const nlohmann::json& canonical_baseline,
                                     BurnoutAnalyzer& burnout) const;

    // |current-baseline| / baseline magnitude mapped to 0..1:
    // below deadzone -> 0, above saturation -> 1, linear between.
    static double deviationMagnitude(double baseline_val, double current_val,
                                     double deadzone, double saturation);

    static double clamp01(double value);

    // Probability containers may use WavLM labels ("anger", "happiness") or
    // burnout labels ("angry", "happy"); expose both spellings.
    static nlohmann::json canonicalizeProbs(const nlohmann::json& data);
    static bool containerHasEmotion(const nlohmann::json& data,
                                    const std::vector<std::string>& emotions);
};

} // namespace audio
