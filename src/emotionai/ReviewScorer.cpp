#include "ReviewScorer.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace emotionai
{

// -------------------------------------------------------------------
// Internal helpers
// -------------------------------------------------------------------

namespace
{

    /// Parse a numeric value that may be stored as a string ("0.65") or
    /// as a raw JSON number in average_emotions.
    double parseNumeric(const nlohmann::json &j)
    {
        if (j.is_number())
            return j.get<double>();
        if (j.is_string())
        {
            try
            {
                return std::stod(j.get<std::string>());
            }
            catch (...)
            {
                return 0.0;
            }
        }
        return 0.0;
    }

    /// Clamp a value to [lo, hi].
    double clamp(double v, double lo, double hi)
    {
        return std::max(lo, std::min(hi, v));
    }

    /// Get a named key from average_emotions, returning fallback if absent.
    double getProb(const nlohmann::json &ae, const std::string &key, double fallback = 0.0)
    {
        auto it = ae.find(key);
        if (it == ae.end())
            return fallback;
        return parseNumeric(*it);
    }

    // -------------------------------------------------------------------
    // Tunable pilot parameters. These are heuristics, not a validated
    // classifier — keep every knob here so tweaks stay consistent.
    // -------------------------------------------------------------------
    constexpr double kArousalBoost = 1.5;    // amplitude of the arousal term
    constexpr double kNeutralWeight = 0.5;   // neutral counts as half-positive
    constexpr double kMinTotalMass = 0.001;  // below this the fallback has no signal
    constexpr double kMatchDelta = 1.5;      // |self - model| <= this -> match
    constexpr double kMinorDelta = 3.0;      // <= this -> minor, above -> major

} // anonymous namespace

// -------------------------------------------------------------------
// isValidSelfScore
// -------------------------------------------------------------------

bool ReviewScorer::isValidSelfScore(double self_score)
{
    return std::isfinite(self_score) &&
           self_score >= kMinScore && self_score <= kMaxScore;
}

// -------------------------------------------------------------------
// computeModelScore
// -------------------------------------------------------------------

std::optional<double> ReviewScorer::computeModelScore(const nlohmann::json &average_emotions)
{
    // --- Try the VA (valence/arousal) path ---
    // The MTL heads (enet_b0_8_va_mtl) are regression outputs on ~[-1, 1] for
    // both valence and arousal, so decide by key presence, not by sign (a
    // negative arousal is a valid value, not a missing channel).
    const bool has_va = average_emotions.contains("valence") &&
                        average_emotions.contains("arousal");

    if (has_va)
    {
        double v = clamp(getProb(average_emotions, "valence", 0.0), -1.0, 1.0);
        double a = clamp(getProb(average_emotions, "arousal", 0.0), -1.0, 1.0);

        // Base score from valence alone: [-1..+1] -> [1..10]
        double base = (v + 1.0) * (kMaxScore - kMinScore) / 2.0 + kMinScore;

        // Arousal boosts in the direction of valence:
        //   positive valence + high arousal -> excitement (higher score)
        //   negative valence + high arousal -> distress   (lower score)
        double boost = (v >= 0.0) ? (a * kArousalBoost) : (-a * kArousalBoost);

        return clamp(base + boost, kMinScore, kMaxScore);
    }

    // --- Fallback: heuristic from probabilities (7-class or missing VA) ---
    double pos = getProb(average_emotions, "happiness", 0.0) +
                 getProb(average_emotions, "surprise", 0.0);

    double neg = getProb(average_emotions, "anger", 0.0) +
                 getProb(average_emotions, "contempt", 0.0) +
                 getProb(average_emotions, "disgust", 0.0) +
                 getProb(average_emotions, "fear", 0.0) +
                 getProb(average_emotions, "sadness", 0.0);

    double neu = getProb(average_emotions, "neutral", 0.0);

    double total = pos + neg + neu;
    if (total < kMinTotalMass)
        return std::nullopt; // no usable signal -> no review at all

    // Score leans positive if pos dominates, negative if neg dominates
    double score = (pos + neu * kNeutralWeight) / total * (kMaxScore - kMinScore) + kMinScore;
    return clamp(score, kMinScore, kMaxScore);
}

// -------------------------------------------------------------------
// evaluate
// -------------------------------------------------------------------

std::optional<ReviewResult> ReviewScorer::evaluate(double self_score,
                                                   const nlohmann::json &average_emotions)
{
    auto model_score_opt = computeModelScore(average_emotions);
    if (!model_score_opt)
        return std::nullopt;

    double model_score = *model_score_opt;
    double delta = std::abs(self_score - model_score);
    double adjusted = (self_score + model_score) / 2.0;

    std::string verdict;
    if (delta <= kMatchDelta)
    {
        verdict = "review.verdict.match";
    }
    else if (delta <= kMinorDelta)
    {
        verdict = "review.verdict.minor";
    }
    else
    {
        verdict = "review.verdict.major";
    }

    return ReviewResult{
        .self_score = self_score,
        .model_score = model_score,
        .delta = delta,
        .verdict = verdict,
        .adjusted_score = adjusted,
    };
}

} // namespace emotionai
