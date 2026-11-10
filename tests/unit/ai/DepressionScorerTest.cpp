#include <gtest/gtest.h>

#include <ai/DepressionScorer.h>
#include <nlohmann/json.hpp>

using emotionai::ai::DepressionScorer;
using nlohmann::json;

namespace
{
    // Convenience: build an emotion object from the standard channels, defaulting
    // the rest to zero.
    json makeEmotions(double happiness = 0.0, double neutral = 0.0, double sadness = 0.0,
                      double fear = 0.0, double disgust = 0.0, double anger = 0.0,
                      double surprise = 0.0, double contempt = 0.0)
    {
        return json{
            {"happiness", happiness},
            {"neutral", neutral},
            {"sadness", sadness},
            {"fear", fear},
            {"disgust", disgust},
            {"anger", anger},
            {"surprise", surprise},
            {"contempt", contempt}};
    }
} // namespace

// ---------------------------------------------------------------------------
// Empty / unusable input must produce a null result (so the stage is skipped),
// mirroring the LLM path's "no usable signal" contract.
// ---------------------------------------------------------------------------
TEST(DepressionScorerTest, EmptyInputYieldsNull)
{
    EXPECT_TRUE(DepressionScorer::assess(json::object()).is_null());
    EXPECT_TRUE(DepressionScorer::assess(json()).is_null());
    // All-zero emotions carry no usable signal.
    EXPECT_TRUE(DepressionScorer::assess(makeEmotions()).is_null());
}

// ---------------------------------------------------------------------------
// A clearly positive face -> "low".
// ---------------------------------------------------------------------------
TEST(DepressionScorerTest, PositiveFaceIsLow)
{
    // High happiness, low neutral -> valence > 0, joy high, NA ~ 0, no
    // flattening (neutral < 0.65).
    auto r = DepressionScorer::assess(makeEmotions(/*happiness*/ 0.7, /*neutral*/ 0.1));
    ASSERT_FALSE(r.is_null());
    EXPECT_EQ(r["verdict"], "low");
    EXPECT_DOUBLE_EQ(r["scores"]["total_score"], 0.0);
    EXPECT_DOUBLE_EQ(r["scores"]["intensity_score"], 0.0);
    EXPECT_GE(r["probability"], 0.1);
    EXPECT_LE(r["probability"], 0.3);
    EXPECT_EQ(r["source"], "local");
}

// ---------------------------------------------------------------------------
// Emotional flattening: a near-blank, high-neutral face with no strong negative
// emotion should still escalate via the intensity row (the prompt's no-op row,
// now repurposed to score `neutral`). A flat face is the classic sign the other
// rows (valence/joy/NA/disgust) all miss.
// ---------------------------------------------------------------------------
TEST(DepressionScorerTest, FlattenedNeutralFaceEscalates)
{
    // neutral = 0.9 -> flattening_score 2. happiness 0.05 -> joy 2.
    // Everything else ~0. Total 4 -> monitor (a flat face is *not* "low").
    auto r = DepressionScorer::assess(makeEmotions(
                         /*happiness*/ 0.05, /*neutral*/ 0.9));
    ASSERT_FALSE(r.is_null());
    EXPECT_EQ(r["verdict"], "monitor");
    EXPECT_DOUBLE_EQ(r["scores"]["intensity_score"], 2.0);
    EXPECT_GE(r["scores"]["total_score"], 3);

    // Mild flattening (neutral 0.7) scores 1, not 2.
    auto mild = DepressionScorer::assess(makeEmotions(
                         /*happiness*/ 0.05, /*neutral*/ 0.7));
    ASSERT_FALSE(mild.is_null());
    EXPECT_DOUBLE_EQ(mild["scores"]["intensity_score"], 1.0);

    // A lively face (low neutral) gets no flattening points.
    auto lively = DepressionScorer::assess(makeEmotions(
                         /*happiness*/ 0.5, /*neutral*/ 0.3));
    ASSERT_FALSE(lively.is_null());
    EXPECT_DOUBLE_EQ(lively["scores"]["intensity_score"], 0.0);
}

// ---------------------------------------------------------------------------
// A strongly negative, flattened face -> "high" (>= 6 points).
// ---------------------------------------------------------------------------
TEST(DepressionScorerTest, NegativeFlattenedFaceIsHigh)
{
    // sadness+fear+disgust+anger = 0.8 (NA>=0.5 -> 2), valence ~ -0.72 (<= -0.2 -> 2),
    // happiness 0 (<= 0.10 -> 2), disgust 0.4 (>= 0.15 -> 1) => total 7 -> high.
    auto r = DepressionScorer::assess(makeEmotions(
                    /*happiness*/ 0.0,
                    /*neutral*/ 0.1,
                    /*sadness*/ 0.4,
                    /*fear*/ 0.2,
                    /*disgust*/ 0.4,
                    /*anger*/ 0.1,
                    /*surprise*/ 0.0));
    ASSERT_FALSE(r.is_null());
    EXPECT_EQ(r["verdict"], "high");
    EXPECT_GE(r["scores"]["total_score"], 6);
    EXPECT_GE(r["probability"], 0.8);
    EXPECT_LE(r["probability"], 0.95);
}

// ---------------------------------------------------------------------------
// A mildly negative profile -> "monitor" (3..5 points).
// ---------------------------------------------------------------------------
TEST(DepressionScorerTest, MildlyNegativeIsMonitor)
{
    // NA = 0.4 (>= 0.35 -> 1), valence = 0.05 - 0.18 = -0.13 (<= 0 -> 1),
    // happiness 0.05 (<= 0.10 -> 2) => total 4 -> monitor.
    auto r = DepressionScorer::assess(makeEmotions(
                    /*happiness*/ 0.05,
                    /*neutral*/ 0.3,
                    /*sadness*/ 0.3,
                    /*fear*/ 0.0,
                    /*disgust*/ 0.0,
                    /*anger*/ 0.1,
                    /*surprise*/ 0.0));
    ASSERT_FALSE(r.is_null());
    EXPECT_EQ(r["verdict"], "monitor");
    EXPECT_GE(r["scores"]["total_score"], 3);
    EXPECT_LE(r["scores"]["total_score"], 5);
    EXPECT_GE(r["probability"], 0.4);
    EXPECT_LE(r["probability"], 0.7);
}

// ---------------------------------------------------------------------------
// String-valued probabilities are accepted (the pipeline may emit either).
// ---------------------------------------------------------------------------
TEST(DepressionScorerTest, AcceptsStringValues)
{
    auto as_string = json{
        {"happiness", "0.7"}, {"neutral", "0.1"}, {"sadness", "0.0"}};
    auto as_number = json{
        {"happiness", 0.7}, {"neutral", 0.1}, {"sadness", 0.0}};

    auto a = DepressionScorer::assess(as_string);
    auto b = DepressionScorer::assess(as_number);
    ASSERT_FALSE(a.is_null());
    ASSERT_FALSE(b.is_null());
    EXPECT_EQ(a["verdict"], b["verdict"]);
    EXPECT_DOUBLE_EQ(a["probability"], b["probability"]);
}

// ---------------------------------------------------------------------------
// The scores object always carries the full fixed key set the frontend renders.
// ---------------------------------------------------------------------------
TEST(DepressionScorerTest, ScoresHaveFixedKeySet)
{
    auto r = DepressionScorer::assess(makeEmotions(/*happiness*/ 0.5, /*neutral*/ 0.2));
    ASSERT_FALSE(r.is_null());
    const auto &s = r["scores"];
    for (const char *key : {"valence_score", "intensity_score", "joy_score", "na_score",
                            "disgust_score", "total_score"})
    {
        ASSERT_TRUE(s.contains(key)) << "missing score key: " << key;
    }
    EXPECT_FALSE(r["reasoning"].get<std::string>().empty());
}

// ---------------------------------------------------------------------------
// Reasoning must be user-facing prose: the internal score keys (valence,
// negative_affect, ...) must not leak into the Russian or English text.
// ---------------------------------------------------------------------------
TEST(DepressionScorerTest, ReasoningUsesLocalizedMarkerLabels)
{
    // Flat-ish face -> flattening and joy markers fire, so the reasoning
    // contains a substituted marker list rather than the fallback phrase.
    const auto ru = DepressionScorer::assess(
        makeEmotions(/*happiness*/ 0.05, /*neutral*/ 0.9), "ru");
    ASSERT_FALSE(ru.is_null());
    ASSERT_EQ(ru["verdict"], "monitor");
    const std::string ru_reasoning = ru["reasoning"].get<std::string>();
    EXPECT_NE(ru_reasoning.find("признаки эмоционального уплощения"), std::string::npos);
    EXPECT_NE(ru_reasoning.find("дефицит положительного аффекта"), std::string::npos);
    for (const char *raw : {"valence", "intensity", "joy", "negative_affect", "disgust"})
    {
        EXPECT_EQ(ru_reasoning.find(raw), std::string::npos) << "raw key leaked: " << raw;
    }

    const auto en = DepressionScorer::assess(
        makeEmotions(/*happiness*/ 0.05, /*neutral*/ 0.9), "en");
    ASSERT_FALSE(en.is_null());
    const std::string en_reasoning = en["reasoning"].get<std::string>();
    EXPECT_NE(en_reasoning.find("signs of emotional flattening"), std::string::npos);
    EXPECT_NE(en_reasoning.find("deficit of positive affect"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Verdict bands are monotonic: more negative input never downgrades the verdict.
// ---------------------------------------------------------------------------
TEST(DepressionScorerTest, VerdictIsConsistentAcrossBands)
{
    auto low = DepressionScorer::assess(makeEmotions(/*happiness*/ 0.8, /*neutral*/ 0.1));
    auto high = DepressionScorer::assess(makeEmotions(
                     /*happiness*/ 0.0,
                     /*neutral*/ 0.1,
                     /*sadness*/ 0.5,
                     /*fear*/ 0.3,
                     /*disgust*/ 0.4,
                     /*anger*/ 0.2));
    ASSERT_FALSE(low.is_null());
    ASSERT_FALSE(high.is_null());
    EXPECT_EQ(low["verdict"], "low");
    EXPECT_EQ(high["verdict"], "high");
    EXPECT_LT(low["scores"]["total_score"], high["scores"]["total_score"]);
}
