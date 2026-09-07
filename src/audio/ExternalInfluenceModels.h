#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <nlohmann/json.hpp>

namespace audio {

//=============================================================================
// ExternalInfluenceStatus
//
// Pilot severity of a possible external influence (scam pressure) signal.
// The score is an index of signal strength, NOT a fraud probability.
//=============================================================================
enum class ExternalInfluenceStatus {
    INSUFFICIENT_DATA,
    LOW,
    ELEVATED_TENSION,
    POSSIBLE_EXTERNAL_PRESSURE,
    PROBABLE_EXTERNAL_INFLUENCE,
    HIGH_EXTERNAL_INFLUENCE_RISK
};

// Exact spec strings, e.g. "POSSIBLE_EXTERNAL_PRESSURE"
std::string statusToString(ExternalInfluenceStatus status);

// Spec section 8: manager action code for each status
std::string managerActionForStatus(ExternalInfluenceStatus status);

//=============================================================================
// ExternalInfluenceConfig
//
// Every weight, boundary and minimum of the external-influence analysis.
// Populated from the `external_influence:` YAML section (Config::instance()),
// defaults match the pilot spec. First config-driven analyzer in this repo.
//=============================================================================
struct ExternalInfluenceConfig {
    // ---- Fragmenting (used by the audio pipeline) ----
    double fragment_seconds = 10.0;      // max WavLM input window is 10 s
    // Pilot runtime bound: each window costs one model run plus feature
    // extraction, so the analyzed span (max_fragments x fragment_seconds,
    // from the START of the recording) is deliberately small. 5 windows =
    // first 50 s of the call; raise this in config for longer coverage at
    // ~linear time cost. Only the first windows of a call are ever scored.
    int max_fragments = 5;
    double min_tail_ratio = 0.5;         // trailing fragment shorter than this fraction is dropped

    // ---- Sufficiency gates (spec section 5) ----
    // Pilot recalibration (2026-09): real call recordings carry long silent
    // stretches and modest levels, so clean-audio thresholds rejected most
    // real files even when all fragments were valid. Measured spread with
    // the DEFAULT baseline: silence ~0.15, noise ~0.25, real speech ~0.35-0.45
    // (a 40/40-valid conversation measured 0.35-0.40). Floors 0.40 and 0.35
    // both sat INSIDE the legit spread, so 0.30 is the floor: garbage stays
    // below, real calls pass with margin. Loud non-speech may straddle the
    // line - accepted for the pilot (quality gate + LOW cap still guard).
    // Gates are compared with an epsilon and rejections never discard the
    // computed scores (shown with an explicit insufficientReason).
    int min_valid_fragments = 3;
    double confidence_min = 0.30;
    double audio_quality_min = 0.30;

    // ---- Fragment score weights (spec section 3, sum = 1.0) ----
    double weight_negative_activation = 0.40;
    double weight_pause_tempo_deviation = 0.25;
    double weight_prosodic_deviation = 0.30;
    double weight_positive_affect_loss = 0.05;

    // ---- Two-sided deviation normalizers ----
    // |relative change| vs baseline: below deadzone -> 0, above saturation -> 1, linear between.
    double pause_tempo_deadzone = 0.10;
    double pause_tempo_saturation = 0.50;
    double prosodic_deadzone = 0.10;
    double prosodic_saturation = 0.50;

    // Attribute sub-weights for the pause/tempo deviation (renormalized when
    // speech_rate is unusable, mirroring BurnoutAnalyzer).
    double attr_pause_ratio = 0.40;
    double attr_pause_mean_duration = 0.20;
    double attr_pause_max_duration = 0.10;
    double attr_speech_rate = 0.30;

    // Attribute sub-weights for the prosodic deviation
    double attr_pitch_variation = 0.50;
    double attr_pitch_range = 0.30;
    double attr_intensity_variation = 0.20;

    // ---- Pattern persistence (spec section 4) ----
    int window_size = 3;                  // rolling median window
    int persistence_required = 2;         // combined pattern in >=2 of window fragments
    double component_high_threshold = 0.50;

    // ---- Score boundaries (spec section 7) ----
    double score_low_max = 0.30;
    double score_elevated_max = 0.50;
    double score_possible_max = 0.65;
    double score_probable_max = 0.80;

    // ---- Context confirmation (spec section 6) ----
    int strong_context_required = 1;
    int moderate_context_required = 2;
    std::vector<std::string> strong_flags = {
        "THIRD_PARTY_INSTRUCTIONS",
        "COACHED_ANSWERS",
        "COVER_STORY",
        "AUDIBLE_THIRD_PARTY_PROMPT"
    };
    std::vector<std::string> moderate_flags = {
        "URGENCY",
        "SECRECY",
        "SAFE_ACCOUNT",
        "AUTHORITY_IMPERSONATION",
        "PURPOSE_INCONSISTENCY",
        "UNKNOWN_PAYEE",
        "EXTERNAL_CALL_IN_PROGRESS"
    };

    // ---- Confidence composition (sum = 1.0) ----
    // Audio quality kept dominant but capped at 0.40 so a silent-tail-heavy
    // call can no longer zero out a long, valid recording on its own; speech
    // coverage gets the freed weight.
    double confidence_weight_audio_quality = 0.40;
    double confidence_weight_baseline_reliability = 0.25;
    double confidence_weight_model_probability = 0.20;
    double confidence_weight_speech_coverage = 0.15;
    double default_baseline_reliability = 0.30;
    double stored_baseline_reliability = 0.70;
};

//=============================================================================
// ExternalInfluenceResult
//
// toJson() emits exactly the spec section 9 contract keys:
//   status, score, confidence, persistentPattern, contextConfirmed,
//   topFactors[], managerAction
// Diagnostics (components, fragment timeline, evidence) are exported
// separately via diagnosticsToJson() so the UI can render details without
// polluting the contract payload.
//=============================================================================
struct ExternalInfluenceResult {
    ExternalInfluenceStatus status = ExternalInfluenceStatus::INSUFFICIENT_DATA;
    double score = 0.0;               // callScore = max rolling median, 0..1
    double confidence = 0.0;          // reliability of the estimate, 0..1
    bool persistentPattern = false;   // combined pattern in >=2 of best-window fragments
    bool contextConfirmed = false;
    std::vector<std::string> topFactors;  // codes, e.g. "NEGATIVE_ACTIVATION_HIGH"
    std::string managerAction;            // e.g. "ENHANCED_ANTIFRAUD_CHECK"

    // Diagnostics
    std::unordered_map<std::string, double> components;  // means over best-window fragments
    int fragmentCount = 0;         // valid fragments fed into scoring
    int totalFragmentCount = 0;    // fragments produced by the pipeline
    std::vector<double> fragmentScores;   // per valid fragment, original order
    std::vector<double> rollingScores;    // window medians
    int bestWindowIndex = -1;
    int windowSize = 0;
    std::vector<std::string> contextFlags;
    int strongContextCount = 0;
    int moderateContextCount = 0;
    double audioQuality = 0.0;
    double meanModelProbability = 0.0;
    double speechCoverage = 0.0;
    double baselineReliability = 0.0;
    // Why the status is INSUFFICIENT_DATA (empty otherwise):
    // "no_valid_fragments" | "too_few_fragments" | "low_confidence" | "low_audio_quality".
    // Scores/components remain populated on rejection so the UI can still show them.
    std::string insufficientReason;

    nlohmann::json toJson() const;          // spec section 9 keys only
    nlohmann::json diagnosticsToJson() const;
};

} // namespace audio
