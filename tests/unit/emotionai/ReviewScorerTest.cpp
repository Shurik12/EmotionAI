#include <cmath>
#include <gtest/gtest.h>

#include <emotionai/ReviewScorer.h>
#include <nlohmann/json.hpp>

using emotionai::ReviewScorer;
using nlohmann::json;

// ---------------------------------------------------------------------------
// "No usable data" must not produce a fabricated score.
// ---------------------------------------------------------------------------

TEST(ReviewScorerTest, EmptyInputYieldsNoScore)
{
    EXPECT_FALSE(ReviewScorer::computeModelScore(json::object()).has_value());
    EXPECT_FALSE(ReviewScorer::evaluate(7.0, json::object()).has_value());
}

// ---------------------------------------------------------------------------
// VA path
// ---------------------------------------------------------------------------

TEST(ReviewScorerTest, VaPathAcceptsStringAndNumber)
{
    json as_string = {{"valence", "0.36"}, {"arousal", "-0.12"}};
    json as_number = {{"valence", 0.36}, {"arousal", -0.12}};

    auto a = ReviewScorer::computeModelScore(as_string);
    auto b = ReviewScorer::computeModelScore(as_number);
    ASSERT_TRUE(a.has_value());
    ASSERT_TRUE(b.has_value());
    EXPECT_NEAR(*a, *b, 1e-9);
    EXPECT_GE(*a, ReviewScorer::kMinScore);
    EXPECT_LE(*a, ReviewScorer::kMaxScore);
}

TEST(ReviewScorerTest, ClampsOutOfRangeVa)
{
    json high = {{"valence", 0.0}, {"arousal", 5.0}};   // arousal clamps to +1
    json low = {{"valence", 0.0}, {"arousal", -5.0}};   // arousal clamps to -1

    auto h = ReviewScorer::computeModelScore(high);
    auto l = ReviewScorer::computeModelScore(low);
    ASSERT_TRUE(h.has_value());
    ASSERT_TRUE(l.has_value());
    EXPECT_NEAR(*h, 7.0, 1e-9); // neutral valence 5.5 + 1.5 boost
    EXPECT_NEAR(*l, 4.0, 1e-9); // neutral valence 5.5 - 1.5 boost
}

// ---------------------------------------------------------------------------
// Fallback heuristic (7-class model without VA heads)
// ---------------------------------------------------------------------------

TEST(ReviewScorerTest, FallbackPrefersPositiveEmotions)
{
    json happy = {{"happiness", 0.9}, {"neutral", 0.1}};
    json sad = {{"sadness", 0.9}, {"neutral", 0.1}};

    auto h = ReviewScorer::computeModelScore(happy);
    auto s = ReviewScorer::computeModelScore(sad);
    ASSERT_TRUE(h.has_value());
    ASSERT_TRUE(s.has_value());
    EXPECT_GT(*h, *s);
}

// ---------------------------------------------------------------------------
// Verdict thresholds (1.5 / 3.0) at model_score == 5.5 (valence 0, arousal 0)
// ---------------------------------------------------------------------------

TEST(ReviewScorerTest, VerdictBoundaries)
{
    json ae = {{"valence", 0.0}, {"arousal", 0.0}}; // model_score == 5.5

    auto match = ReviewScorer::evaluate(6.0, ae);  // delta 0.5
    ASSERT_TRUE(match.has_value());
    EXPECT_EQ(match->verdict, "review.verdict.match");

    auto minor = ReviewScorer::evaluate(7.5, ae);  // delta 2.0
    ASSERT_TRUE(minor.has_value());
    EXPECT_EQ(minor->verdict, "review.verdict.minor");

    auto major = ReviewScorer::evaluate(9.0, ae);  // delta 3.5
    ASSERT_TRUE(major.has_value());
    EXPECT_EQ(major->verdict, "review.verdict.major");
}

// ---------------------------------------------------------------------------
// self_score validation
// ---------------------------------------------------------------------------

TEST(ReviewScorerTest, ValidSelfScoreRange)
{
    EXPECT_TRUE(ReviewScorer::isValidSelfScore(1.0));
    EXPECT_TRUE(ReviewScorer::isValidSelfScore(5.5));
    EXPECT_TRUE(ReviewScorer::isValidSelfScore(10.0));

    EXPECT_FALSE(ReviewScorer::isValidSelfScore(0.9));
    EXPECT_FALSE(ReviewScorer::isValidSelfScore(10.1));
    EXPECT_FALSE(ReviewScorer::isValidSelfScore(-1.0));          // "not a review"
    EXPECT_FALSE(ReviewScorer::isValidSelfScore(std::nan("")));  // NaN
}
