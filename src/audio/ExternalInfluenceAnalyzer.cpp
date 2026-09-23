#include "ExternalInfluenceAnalyzer.h"
#include <algorithm>
#include <cmath>
#include <logging/Logger.h>

namespace audio {

namespace {

constexpr const char* COMPONENT_NEGATIVE_ACTIVATION = "negativeActivation";
constexpr const char* COMPONENT_PAUSE_TEMPO_DEVIATION = "pauseTempoDeviation";
constexpr const char* COMPONENT_PROSODIC_DEVIATION = "prosodicDeviation";
constexpr const char* COMPONENT_POSITIVE_AFFECT_LOSS = "positiveAffectLoss";

// Read a numeric (or stringified numeric) field from json.
double jsonNumber(const nlohmann::json& data, const std::string& key, double default_val) {
    if (!data.is_object() || !data.contains(key)) {
        return default_val;
    }
    const auto& value = data[key];
    if (value.is_number()) {
        return value.get<double>();
    }
    if (value.is_string()) {
        try {
            return std::stod(value.get<std::string>());
        } catch (...) {
            return default_val;
        }
    }
    return default_val;
}

bool isFlagIn(const std::string& flag, const std::vector<std::string>& list) {
    return std::find(list.begin(), list.end(), flag) != list.end();
}

} // namespace

ExternalInfluenceAnalyzer::ExternalInfluenceAnalyzer() = default;

ExternalInfluenceAnalyzer::ExternalInfluenceAnalyzer(const ExternalInfluenceConfig& config)
    : cfg_(config) {}

double ExternalInfluenceAnalyzer::clamp01(double value) {
    return std::max(0.0, std::min(1.0, value));
}

double ExternalInfluenceAnalyzer::deviationMagnitude(
    double baseline_val, double current_val, double deadzone, double saturation)
{
    if (baseline_val <= 0.0) {
        return (current_val > 0.05) ? 1.0 : 0.0;
    }
    if (saturation <= deadzone) {
        saturation = deadzone + 0.01;
    }
    double rel = std::abs(current_val - baseline_val) / baseline_val;
    if (rel <= deadzone) return 0.0;
    if (rel >= saturation) return 1.0;
    return (rel - deadzone) / (saturation - deadzone);
}

//=============================================================================
// Probability canonicalization
//
// The WavLM model emits "anger"/"happiness" (Audio.cpp EMOTION_LABELS) while
// BurnoutAnalyzer component math reads "angry"/"happy". Both spellings are
// added to every probability container so the shared BurnoutAnalyzer methods
// see consistent keys regardless of the source model.
//=============================================================================
nlohmann::json ExternalInfluenceAnalyzer::canonicalizeProbs(const nlohmann::json& data)
{
    nlohmann::json copy = data;

    auto addAliases = [](nlohmann::json& probs) {
        if (!probs.is_object()) return;
        auto alias = [&probs](const std::string& from, const std::string& to) {
            if (!probs.contains(to) && probs.contains(from)) {
                probs[to] = probs[from];
            }
        };
        alias("anger", "angry");
        alias("angry", "anger");
        alias("happiness", "happy");
        alias("happy", "happiness");
    };

    if (copy.contains("model_results") && copy["model_results"].is_array() &&
        !copy["model_results"].empty() &&
        copy["model_results"][0].contains("all_probabilities")) {
        addAliases(copy["model_results"][0]["all_probabilities"]);
    }
    if (copy.contains("additional_probs")) {
        addAliases(copy["additional_probs"]);
    }
    if (copy.contains("detailed_analysis") && copy["detailed_analysis"].is_object()) {
        for (auto& [key, value] : copy["detailed_analysis"].items()) {
            if (value.is_object() && key.find("probabilities") != std::string::npos) {
                addAliases(value);
            }
        }
    }
    return copy;
}

// True when any of the given emotions is present in any probability container.
bool ExternalInfluenceAnalyzer::containerHasEmotion(
    const nlohmann::json& data, const std::vector<std::string>& emotions)
{
    auto containerHasAny = [&emotions](const nlohmann::json& probs) {
        if (!probs.is_object()) return false;
        for (const auto& emotion : emotions) {
            if (probs.contains(emotion)) return true;
        }
        return false;
    };

    if (data.contains("model_results") && data["model_results"].is_array() &&
        !data["model_results"].empty() &&
        data["model_results"][0].contains("all_probabilities") &&
        containerHasAny(data["model_results"][0]["all_probabilities"])) {
        return true;
    }
    if (data.contains("additional_probs") && containerHasAny(data["additional_probs"])) {
        return true;
    }
    if (data.contains("detailed_analysis") && data["detailed_analysis"].is_object()) {
        for (auto& [key, value] : data["detailed_analysis"].items()) {
            if (value.is_object() && key.find("probabilities") != std::string::npos &&
                containerHasAny(value)) {
                return true;
            }
        }
    }
    return false;
}

//=============================================================================
// Per-fragment component calculation
//=============================================================================
ExternalInfluenceAnalyzer::FragmentAnalysis ExternalInfluenceAnalyzer::analyzeFragment(
    const nlohmann::json& fragment,
    const nlohmann::json& canonical_baseline,
    BurnoutAnalyzer& burnout) const
{
    FragmentAnalysis result;

    if (fragment.contains("error")) {
        LOG_WARN("External influence: skipping fragment with error: {}",
                 fragment["error"].dump());
        return result;
    }
    if (!fragment.contains("acoustic_features") ||
        !fragment["acoustic_features"].is_object()) {
        LOG_WARN("External influence: skipping fragment without acoustic features");
        return result;
    }

    const nlohmann::json current = canonicalizeProbs(fragment);
    const nlohmann::json& base = canonical_baseline;

    const bool hasNegative = containerHasEmotion(current, {"angry", "fear", "disgust"}) &&
                              containerHasEmotion(base, {"angry", "fear", "disgust"});
    const bool hasAffect = containerHasEmotion(current, {"happy"}) &&
                           containerHasEmotion(base, {"happy"});

    const auto& cur_feat = fragment["acoustic_features"];
    const nlohmann::json* base_feat = nullptr;
    if (base.contains("acoustic_features") && base["acoustic_features"].is_object()) {
        base_feat = &base["acoustic_features"];
    }

    const auto feature = [&cur_feat](const char* key, double default_val) {
        return jsonNumber(cur_feat, key, default_val);
    };
    const auto baseVal = [base_feat](const char* key, double default_val) {
        return base_feat ? jsonNumber(*base_feat, key, default_val) : default_val;
    };

    // ---- Pause/tempo deviation: two-sided magnitude across pause structure
    //      and speech rate (deviation counts in BOTH directions) ----
    const double w_ratio = cfg_.attr_pause_ratio;
    const double w_mean = cfg_.attr_pause_mean_duration;
    const double w_max = cfg_.attr_pause_max_duration;
    const double w_speech = cfg_.attr_speech_rate;

    const double base_speech = baseVal("speech_rate", 0.0);
    const bool speech_usable = feature("speech_rate", 0.0) > 0.0 && base_speech > 0.0;
    // A pause/tempo reference exists when the baseline carries any of the
    // pause attributes (mirrors BurnoutAnalyzer's "no speech_rate" branch).
    const bool pause_ref_usable =
        baseVal("pause_ratio", 0.0) > 0.0 ||
        baseVal("pause_mean_duration", 0.0) > 0.0 ||
        baseVal("pause_max_duration", 0.0) > 0.0 ||
        speech_usable;

    double dev_ratio = 0.0, dev_mean = 0.0, dev_max = 0.0, dev_speech = 0.0;
    double usable_pause_weights = 0.0;
    if (pause_ref_usable) {
        if (baseVal("pause_ratio", 0.0) > 0.0) {
            dev_ratio = deviationMagnitude(
                baseVal("pause_ratio", 0.0), feature("pause_ratio", 0.0),
                cfg_.pause_tempo_deadzone, cfg_.pause_tempo_saturation);
            usable_pause_weights += w_ratio;
        }
        if (baseVal("pause_mean_duration", 0.0) > 0.0) {
            dev_mean = deviationMagnitude(
                baseVal("pause_mean_duration", 0.0), feature("pause_mean_duration", 0.0),
                cfg_.pause_tempo_deadzone, cfg_.pause_tempo_saturation);
            usable_pause_weights += w_mean;
        }
        if (baseVal("pause_max_duration", 0.0) > 0.0) {
            dev_max = deviationMagnitude(
                baseVal("pause_max_duration", 0.0), feature("pause_max_duration", 0.0),
                cfg_.pause_tempo_deadzone, cfg_.pause_tempo_saturation);
            usable_pause_weights += w_max;
        }
        if (speech_usable) {
            dev_speech = deviationMagnitude(
                base_speech, feature("speech_rate", 0.0),
                cfg_.pause_tempo_deadzone, cfg_.pause_tempo_saturation);
            usable_pause_weights += w_speech;
        }
        if (usable_pause_weights > 0.0) {
            result.pauseTempoDeviation = clamp01(
                (w_ratio * dev_ratio + w_mean * dev_mean + w_max * dev_max +
                 w_speech * dev_speech) / usable_pause_weights);
        }
    }

    // ---- Prosodic deviation: two-sided magnitude of pitch/range/intensity ----
    const double p_var = cfg_.attr_pitch_variation;
    const double p_range = cfg_.attr_pitch_range;
    const double p_int = cfg_.attr_intensity_variation;

    const bool prosodic_ref_usable =
        baseVal("pitch_variation", 0.0) > 0.0 ||
        baseVal("pitch_range", 0.0) > 0.0 ||
        baseVal("intensity_variation", 0.0) > 0.0;

    double dev_pitch_var = 0.0, dev_pitch_range = 0.0, dev_intensity_var = 0.0;
    double usable_prosodic_weights = 0.0;
    if (prosodic_ref_usable) {
        if (baseVal("pitch_variation", 0.0) > 0.0) {
            dev_pitch_var = deviationMagnitude(
                baseVal("pitch_variation", 0.0), feature("pitch_variation", 0.0),
                cfg_.prosodic_deadzone, cfg_.prosodic_saturation);
            usable_prosodic_weights += p_var;
        }
        if (baseVal("pitch_range", 0.0) > 0.0) {
            dev_pitch_range = deviationMagnitude(
                baseVal("pitch_range", 0.0), feature("pitch_range", 0.0),
                cfg_.prosodic_deadzone, cfg_.prosodic_saturation);
            usable_prosodic_weights += p_range;
        }
        if (baseVal("intensity_variation", 0.0) > 0.0) {
            dev_intensity_var = deviationMagnitude(
                baseVal("intensity_variation", 0.0), feature("intensity_variation", 0.0),
                cfg_.prosodic_deadzone, cfg_.prosodic_saturation);
            usable_prosodic_weights += p_int;
        }
        if (usable_prosodic_weights > 0.0) {
            result.prosodicDeviation = clamp01(
                (p_var * dev_pitch_var + p_range * dev_pitch_range +
                 p_int * dev_intensity_var) / usable_prosodic_weights);
        }
    }

    // ---- Emotion components delegated to the shared burnout math ----
    result.negativeActivation = hasNegative
        ? clamp01(burnout.calculateNegativeActivation(current, base))
        : 0.0;
    if (hasAffect) {
        result.positiveAffectLoss = clamp01(burnout.calculatePositiveAffectLoss(current, base));
    } else {
        result.positiveAffectLoss = -1.0;  // absent marker
    }

    result.valid = hasNegative && pause_ref_usable && prosodic_ref_usable;
    return result;
}

//=============================================================================
// Main analysis
//=============================================================================
ExternalInfluenceResult ExternalInfluenceAnalyzer::analyze(
    const nlohmann::json& fragments,
    const nlohmann::json& baseline,
    double audio_quality,
    const std::vector<std::string>& context_flags) const
{
    ExternalInfluenceResult result;
    result.audioQuality = clamp01(audio_quality);
    result.contextFlags = context_flags;
    result.totalFragmentCount = fragments.is_array()
        ? static_cast<int>(fragments.size()) : 0;

    const nlohmann::json& baseline_to_use =
        baseline.empty() ? getDefaultBaseline() : baseline;
    result.baselineReliability = baseline.empty()
        ? cfg_.default_baseline_reliability
        : cfg_.stored_baseline_reliability;
    // Canonicalized once: the per-fragment pass compares every window
    // against this same baseline.
    const nlohmann::json canonical_baseline = canonicalizeProbs(baseline_to_use);

    // Context evidence (spec section 6)
    for (const auto& flag : context_flags) {
        if (isFlagIn(flag, cfg_.strong_flags)) {
            ++result.strongContextCount;
        } else if (isFlagIn(flag, cfg_.moderate_flags)) {
            ++result.moderateContextCount;
        } else {
            LOG_WARN("External influence: ignoring unknown context flag '{}'", flag);
        }
    }
    result.contextConfirmed =
        result.strongContextCount >= cfg_.strong_context_required ||
        result.moderateContextCount >= cfg_.moderate_context_required;

    // Per-fragment components (spec sections 2-3). An empty fragment set is
    // not an early return: the sufficiency gates below still evaluate so an
    // INSUFFICIENT_DATA result carries the true reason for the UI.
    BurnoutAnalyzer burnout;
    std::vector<FragmentAnalysis> analyses;
    analyses.reserve(result.totalFragmentCount);

    double model_prob_sum = 0.0;
    double coverage_sum = 0.0;
    for (const auto& fragment : fragments) {
        FragmentAnalysis fa = analyzeFragment(fragment, canonical_baseline, burnout);
        if (!fa.valid) {
            LOG_WARN("External influence: fragment invalid (no usable components)");
            continue;
        }
        analyses.push_back(fa);
        model_prob_sum += jsonNumber(fragment, "model_probability", 0.0);
        coverage_sum += jsonNumber(
            fragment["acoustic_features"], "voice_activity_ratio", 0.0);
    }

    result.fragmentCount = static_cast<int>(analyses.size());
    result.meanModelProbability = analyses.empty()
        ? 0.0 : model_prob_sum / analyses.size();
    result.speechCoverage = analyses.empty()
        ? 0.0 : coverage_sum / analyses.size();

    const double confidence = clamp01(
        cfg_.confidence_weight_audio_quality * result.audioQuality +
        cfg_.confidence_weight_baseline_reliability * result.baselineReliability +
        cfg_.confidence_weight_model_probability * result.meanModelProbability +
        cfg_.confidence_weight_speech_coverage * result.speechCoverage);
    result.confidence = confidence;

    // Fragment score with the config weights; the weighted sum is divided by
    // the weights of the PRESENT components, so a missing positiveAffectLoss
    // normalizes by 0.95 exactly as the spec requires.
    const auto fragmentScore = [this](const FragmentAnalysis& fa) {
        const double w_sum =
            cfg_.weight_negative_activation +
            cfg_.weight_pause_tempo_deviation +
            cfg_.weight_prosodic_deviation +
            (fa.positiveAffectLoss >= 0.0 ? cfg_.weight_positive_affect_loss : 0.0);
        if (w_sum <= 0.0) return 0.0;
        double score =
            cfg_.weight_negative_activation * fa.negativeActivation +
            cfg_.weight_pause_tempo_deviation * fa.pauseTempoDeviation +
            cfg_.weight_prosodic_deviation * fa.prosodicDeviation;
        if (fa.positiveAffectLoss >= 0.0) {
            score += cfg_.weight_positive_affect_loss * fa.positiveAffectLoss;
        }
        return clamp01(score / w_sum);
    };

    result.fragmentScores.reserve(analyses.size());
    for (const auto& fa : analyses) {
        result.fragmentScores.push_back(fragmentScore(fa));
    }

    // Rolling median windows (spec section 4). Scoring runs on whatever was
    // valid even below min_valid_fragments so rejected sets still show their
    // timeline in the UI; the effective window shrinks to the fragment count.
    const int window = std::max(1, std::min(cfg_.window_size,
                                            static_cast<int>(analyses.size())));
    result.windowSize = window;
    if (!analyses.empty()) {
        const int num_windows = static_cast<int>(analyses.size()) - window + 1;
        const auto windowMedian = [&result, window](int start) {
            std::vector<double> vals(result.fragmentScores.begin() + start,
                                     result.fragmentScores.begin() + start + window);
            std::sort(vals.begin(), vals.end());
            return (window % 2 == 1)
                ? vals[window / 2]
                : (vals[window / 2 - 1] + vals[window / 2]) / 2.0;
        };

        result.rollingScores.reserve(num_windows);
        double best = -1.0;
        for (int i = 0; i < num_windows; ++i) {
            const double med = windowMedian(i);
            result.rollingScores.push_back(med);
            if (med > best) {
                best = med;
                result.bestWindowIndex = i;
            }
        }
        result.score = clamp01(best);

        // Pattern stability on the best window (first argmax)
        int combined_count = 0;
        double mean_neg = 0.0, mean_pause = 0.0, mean_prosodic = 0.0, mean_affect = 0.0;
        for (int i = result.bestWindowIndex; i < result.bestWindowIndex + window; ++i) {
            const auto& fa = analyses[i];
            const bool emotional = fa.negativeActivation >= cfg_.component_high_threshold;
            const bool speech = fa.pauseTempoDeviation >= cfg_.component_high_threshold ||
                                fa.prosodicDeviation >= cfg_.component_high_threshold;
            if (emotional && speech) {
                ++combined_count;
            }
            mean_neg += fa.negativeActivation;
            mean_pause += fa.pauseTempoDeviation;
            mean_prosodic += fa.prosodicDeviation;
            mean_affect += (fa.positiveAffectLoss >= 0.0) ? fa.positiveAffectLoss : 0.0;
        }
        result.persistentPattern =
            combined_count >= cfg_.persistence_required;

        result.components[COMPONENT_NEGATIVE_ACTIVATION] = mean_neg / window;
        result.components[COMPONENT_PAUSE_TEMPO_DEVIATION] = mean_pause / window;
        result.components[COMPONENT_PROSODIC_DEVIATION] = mean_prosodic / window;
        result.components[COMPONENT_POSITIVE_AFFECT_LOSS] = mean_affect / window;

        // topFactors: dominant components first, then pattern evidence
        std::vector<std::pair<double, std::string>> by_value = {
            {result.components[COMPONENT_NEGATIVE_ACTIVATION], "NEGATIVE_ACTIVATION_HIGH"},
            {result.components[COMPONENT_PAUSE_TEMPO_DEVIATION], "PAUSE_TEMPO_DEVIATION"},
            {result.components[COMPONENT_PROSODIC_DEVIATION], "PROSODIC_DEVIATION"},
            {result.components[COMPONENT_POSITIVE_AFFECT_LOSS], "POSITIVE_AFFECT_LOSS_HIGH"}
        };
        std::sort(by_value.begin(), by_value.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        for (const auto& [value, code] : by_value) {
            if (value >= cfg_.component_high_threshold) {
                result.topFactors.push_back(code);
            }
        }
        if (result.persistentPattern) {
            result.topFactors.push_back("PERSISTENT_PATTERN");
        }
        if (result.contextConfirmed) {
            result.topFactors.push_back("CONTEXT_CONFIRMED");
        }
    }

    // Sufficiency gates (spec section 5). Checked AFTER scoring on purpose:
    // a rejected set must not lose the computed timeline/components, and the
    // reason code tells the UI which gate actually fired. Numeric gates carry
    // an epsilon so a confidence that lands on the boundary from float sums
    // (e.g. exactly 0.40 on real speech) is not spuriously rejected. Gate
    // order: quality -> valid count -> confidence.
    if (result.audioQuality < cfg_.audio_quality_min - 1e-9) {
        result.insufficientReason = "low_audio_quality";
    } else if (result.fragmentCount < cfg_.min_valid_fragments) {
        result.insufficientReason =
            (analyses.empty() && result.totalFragmentCount > 0)
                ? "no_valid_fragments" : "too_few_fragments";
    } else if (confidence < cfg_.confidence_min - 1e-9) {
        result.insufficientReason = "low_confidence";
    }

    if (!result.insufficientReason.empty()) {
        result.status = ExternalInfluenceStatus::INSUFFICIENT_DATA;
        result.managerAction = managerActionForStatus(result.status);
        LOG_INFO("External influence: INSUFFICIENT_DATA ({}), score={:.2f}, "
                 "confidence={:.2f}, fragments={}/{}",
                 result.insufficientReason, result.score, result.confidence,
                 result.fragmentCount, result.totalFragmentCount);
        return result;
    }

    // Status ladder (spec section 7)
    const double score = result.score;
    if (score < cfg_.score_low_max) {
        result.status = ExternalInfluenceStatus::LOW;
    } else if (score < cfg_.score_elevated_max) {
        result.status = ExternalInfluenceStatus::ELEVATED_TENSION;
    } else if (!result.persistentPattern) {
        result.status = ExternalInfluenceStatus::ELEVATED_TENSION;
    } else if (score < cfg_.score_possible_max) {
        result.status = ExternalInfluenceStatus::POSSIBLE_EXTERNAL_PRESSURE;
    } else if (!result.contextConfirmed) {
        result.status = ExternalInfluenceStatus::POSSIBLE_EXTERNAL_PRESSURE;
    } else if (score < cfg_.score_probable_max) {
        result.status = ExternalInfluenceStatus::PROBABLE_EXTERNAL_INFLUENCE;
    } else {
        result.status = ExternalInfluenceStatus::HIGH_EXTERNAL_INFLUENCE_RISK;
    }
    result.managerAction = managerActionForStatus(result.status);

    LOG_INFO("External influence result: status={}, score={:.2f}, confidence={:.2f}, "
             "fragments={}/{}, persistent={}, context={}",
             statusToString(result.status), result.score, result.confidence,
             result.fragmentCount, result.totalFragmentCount,
             result.persistentPattern, result.contextConfirmed);

    return result;
}

} // namespace audio
