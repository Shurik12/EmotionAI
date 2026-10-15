#pragma once

#include <nlohmann/json.hpp>
#include <optional>
#include <string>

namespace emotionai
{

/// Converts video emotion analysis (average_emotions + valence/arousal)
/// into a product score on a 1-10 scale and compares it with the user's
/// self-assessment.
struct ReviewResult
{
    double self_score;   // user-provided score (1-10)
    double model_score;  // score derived from emotions (1-10)
    double delta;        // |self - model|
    std::string verdict; // translation key, e.g. "review.verdict.match"
    double adjusted_score; // final corrected score
};

class ReviewScorer
{
public:
    /// Score scale shared by self_score and model_score.
    static constexpr double kMinScore = 1.0;
    static constexpr double kMaxScore = 10.0;

    /// True when self_score is a usable review request (finite, in [1, 10]).
    static bool isValidSelfScore(double self_score);

    /// Compute model_score from average_emotions JSON.
    /// Expects "valence" and "arousal" as regression outputs on ~[-1..1]
    /// (from the enet_b0_8_va_mtl MTL heads, stored as string "0.36").
    /// Falls back to an emotion-probability heuristic if either is missing.
    /// Returns nullopt when the input carries no usable signal at all (e.g. no
    /// face was detected), so callers can skip the review instead of
    /// publishing a fabricated score.
    static std::optional<double> computeModelScore(const nlohmann::json &average_emotions);

    /// Compare self_score with model_score and produce a verdict.
    /// Returns nullopt when no model score can be derived (see above).
    static std::optional<ReviewResult> evaluate(double self_score,
                                                const nlohmann::json &average_emotions);
};

} // namespace emotionai
