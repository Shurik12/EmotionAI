// tests/unit/audio/BurnoutAnalyzerTest.cpp
//
// Guards for the audio burnout scorer. The prosodic component is deliberately
// symmetric (a departure in either direction counts), and the acoustic
// defaults are the medians of the current benchmark/control set.

#include <gtest/gtest.h>

#include <audio/BurnoutAnalyzer.h>
#include <audio/BurnoutModels.h>

using nlohmann::json;

namespace {

json acoustic(double pitch_variation, double pitch_range, double intensity_variation) {
    return {
        {"acoustic_features", {
            {"pitch_variation", pitch_variation},
            {"pitch_range", pitch_range},
            {"intensity_variation", intensity_variation},
        }},
    };
}

}  // namespace

// A drop AND a rise in prosodic variability must both raise the component.
// The component used to reward only drops, which made it anti-correlated with
// the labelled burnout groups (whose pitch variability is *higher*).
TEST(BurnoutAnalyzerProsodic, FlagsDepartureInEitherDirection) {
    audio::BurnoutAnalyzer analyzer;
    const json baseline = acoustic(0.10, 50.0, 1.0);

    const json no_change = acoustic(0.10, 50.0, 1.0);
    const json drop = acoustic(0.04, 20.0, 0.4);   // large relative decrease
    const json rise = acoustic(0.22, 120.0, 2.4);  // large relative increase

    EXPECT_DOUBLE_EQ(analyzer.calculateProsodicFlattening(no_change, baseline), 0.0);
    EXPECT_GT(analyzer.calculateProsodicFlattening(drop, baseline), 0.5);
    EXPECT_GT(analyzer.calculateProsodicFlattening(rise, baseline), 0.5);
}

// A change smaller than the dead zone is not a signal.
TEST(BurnoutAnalyzerProsodic, IgnoresSmallChanges) {
    audio::BurnoutAnalyzer analyzer;
    const json baseline = acoustic(0.10, 50.0, 1.0);
    const json slightly_off = acoustic(0.105, 52.0, 1.02);  // < dead zone

    EXPECT_DOUBLE_EQ(analyzer.calculateProsodicFlattening(slightly_off, baseline), 0.0);
}

// Positive-affect loss is a DROP in the happy class.
TEST(BurnoutAnalyzerAffect, PositiveAffectLossUsesHappyDrop) {
    audio::BurnoutAnalyzer analyzer;
    const json baseline = {{"additional_probs", {{"happy", 0.30}}}};
    const json current = {{"additional_probs", {{"happy", 0.10}}}};  // -0.20 -> saturated

    EXPECT_DOUBLE_EQ(analyzer.calculatePositiveAffectLoss(current, baseline), 1.0);
}

// Missing emotion probabilities must not be turned into a fabricated score.
TEST(BurnoutAnalyzerAnalyze, MissingEmotionsIsInsufficientData) {
    audio::BurnoutAnalyzer analyzer;
    const json baseline = acoustic(0.10, 50.0, 1.0);  // no additional_probs
    const json current = acoustic(0.10, 50.0, 1.0);

    const audio::Result result = analyzer.analyze(current, baseline);
    EXPECT_EQ(result.state, audio::State::INSUFFICIENT_DATA);
    EXPECT_EQ(result.error, "analysis_failed");
}

// The built-in acoustic baseline must be the control-set median, not the old
// inflated values (pitch_range 128.33, speech_rate 6.10, ...) that saturated
// prosodic_flattening on normal speech.
TEST(BurnoutConfig, AcousticBaselineMatchesControlMedians) {
    audio::BurnoutConfig cfg;
    EXPECT_NEAR(cfg.base_pitch_range, 66.77, 0.01);
    EXPECT_NEAR(cfg.base_speech_rate, 4.285, 0.001);
    EXPECT_NEAR(cfg.base_pause_mean_duration, 0.1799, 0.0001);
    EXPECT_LT(cfg.base_pitch_range, 100.0);
    EXPECT_LT(cfg.base_speech_rate, 5.0);

    const json baseline = cfg.defaultBaselineJson();
    EXPECT_DOUBLE_EQ(
        baseline["acoustic_features"]["pitch_range"].get<double>(), cfg.base_pitch_range);
}

// M1: BurnoutConfig toJson/fromJson round-trip must preserve every field so
// the external-influence fallback baseline tracks operator-tuned values.
TEST(BurnoutConfig, JsonRoundTrip) {
    audio::BurnoutConfig cfg;
    // Override a few fields to make the round-trip non-trivial.
    cfg.base_pause_ratio = 0.42;
    cfg.duration_confidence_threshold_s = 45.0;
    cfg.max_windows = 7;
    cfg.aggregation = "mean";

    const json j = cfg.toJson();
    audio::BurnoutConfig restored;
    restored.fromJson(j);

    EXPECT_DOUBLE_EQ(restored.base_pause_ratio, 0.42);
    EXPECT_DOUBLE_EQ(restored.duration_confidence_threshold_s, 45.0);
    EXPECT_EQ(restored.max_windows, 7);
    EXPECT_EQ(restored.aggregation, "mean");
    // Unmodified fields must survive the round-trip too.
    EXPECT_DOUBLE_EQ(restored.base_pitch_range, cfg.base_pitch_range);
    EXPECT_DOUBLE_EQ(restored.base_speech_rate, cfg.base_speech_rate);
}

// M1: an empty/partial burnout_config JSON leaves the built-in defaults
// untouched (fromJson must not zero out unset fields).
TEST(BurnoutConfig, FromJsonPartialLeavesDefaults) {
    audio::BurnoutConfig cfg;
    const double original_pause_ratio = cfg.base_pause_ratio;
    const int original_max_windows = cfg.max_windows;

    // Only one field is present.
    const json partial = {{"base_pitch_range", 99.0}};
    cfg.fromJson(partial);

    EXPECT_DOUBLE_EQ(cfg.base_pitch_range, 99.0);
    EXPECT_DOUBLE_EQ(cfg.base_pause_ratio, original_pause_ratio);
    EXPECT_EQ(cfg.max_windows, original_max_windows);
}

// H1: single-recording state bands. determineState is now single-recording and
// must not over-state a single elevated reading as chronic. Lock in the bands.
// Every payload carries emotion probabilities (analyze() refuses to score
// without them) plus the acoustic features that drive the risk.
namespace {
// Emotion-probability payload: the six WavLM classes at their near-uniform
// midpoints so the emotion components read ~0 and only acoustics drive risk.
json emotions() {
    return {{"additional_probs", {
        {"neutral", 0.165}, {"happy", 0.121}, {"sad", 0.112},
        {"angry", 0.127},   {"fear", 0.155}, {"disgust", 0.171}}}};
}
}  // namespace

TEST(BurnoutAnalyzerState, SingleRecordingBands) {
    audio::BurnoutAnalyzer analyzer;

    // Lowest risk -> NORMAL. No acoustic departure at all.
    {
        json baseline = acoustic(0.10, 50.0, 1.0);
        json current = acoustic(0.10, 50.0, 1.0);
        baseline.merge_patch(emotions());
        current.merge_patch(emotions());
        const audio::Result r = analyzer.analyze(current, baseline);
        EXPECT_EQ(r.state, audio::State::NORMAL);
        EXPECT_LT(r.risk, 0.01);
    }

    // Mildly elevated (0.35-0.65) -> SHORT_STRESS, never SUSTAINED_STRESS from
    // a single recording. A 25% pause-ratio rise + a 30% voice-activity drop
    // lands squarely in the mid band without crossing 0.65.
    {
        json baseline = acoustic(0.10, 50.0, 1.0);
        json current = acoustic(0.10, 50.0, 1.0);
        baseline.merge_patch(emotions());
        current.merge_patch(emotions());
        baseline["acoustic_features"]["pause_ratio"] = 0.2;
        baseline["acoustic_features"]["voice_activity_ratio"] = 0.5;
        current["acoustic_features"]["pause_ratio"] = 0.25;   // +25% relative -> 0.2
        current["acoustic_features"]["voice_activity_ratio"] = 0.32;  // -36% relative -> ~0.9
        const audio::Result r = analyzer.analyze(current, baseline);
        ASSERT_GT(r.risk, 0.35);
        ASSERT_LT(r.risk, 0.65);
        EXPECT_EQ(r.state, audio::State::SHORT_STRESS);
    }

    // Very high risk (>= 0.65) -> SUSTAINED_STRESS (the strongest
    // single-recording state). Acoustic-only risk caps at ~0.58 (the emotion
    // components carry the rest of the weight), so crossing 0.65 needs a
    // strong negative-activation signal too. Saturating angry/fear/disgust
    // AND a happy-class drop (positive_affect_loss -> 1.0) pushes the total
    // past 0.65 with margin, without tripping the LOW_AFFECT_UNSPECIFIC
    // pattern (which requires prosodic > 0.4 AND positive_affect_loss > 0.4
    // while negative < 0.3 — here prosodic is ~0, and negative is ~1.0).
    {
        json baseline = acoustic(0.10, 50.0, 1.0);
        json current = acoustic(0.10, 50.0, 1.0);
        baseline.merge_patch(emotions());
        current.merge_patch(emotions());
        baseline["additional_probs"]["angry"] = 0.05;
        baseline["additional_probs"]["fear"] = 0.05;
        baseline["additional_probs"]["disgust"] = 0.05;
        current["additional_probs"]["angry"] = 0.15;    // +200% -> 1.0
        current["additional_probs"]["fear"] = 0.15;     // +200% -> 1.0
        current["additional_probs"]["disgust"] = 0.15;  // +200% -> 1.0
        baseline["additional_probs"]["happy"] = 0.15;
        current["additional_probs"]["happy"] = 0.03;    // -80% -> 1.0 (positive_affect_loss)
        // Saturated acoustic terms for the remaining risk.
        baseline["acoustic_features"]["pause_ratio"] = 0.2;
        baseline["acoustic_features"]["pause_mean_duration"] = 0.1;
        baseline["acoustic_features"]["voice_activity_ratio"] = 0.6;
        current["acoustic_features"]["pause_ratio"] = 0.8;   // +300% -> 1.0
        current["acoustic_features"]["pause_mean_duration"] = 0.6;  // +500% -> 1.0
        current["acoustic_features"]["voice_activity_ratio"] = 0.1;  // -83% -> 1.0
        const audio::Result r = analyzer.analyze(current, baseline);
        EXPECT_GE(r.risk, 0.65);
        EXPECT_EQ(r.state, audio::State::SUSTAINED_STRESS);
    }
}

// H1: BURNOUT_LIKE is unreachable from a single recording — it needs history,
// which is not yet wired. Guard the honest contract.
TEST(BurnoutAnalyzerState, BurnoutLikeUnreachableFromSingleRecording) {
    audio::BurnoutAnalyzer analyzer;
    // Push risk to the maximum with a single recording; it must top out at
    // SUSTAINED_STRESS, never BURNOUT_LIKE.
    json baseline = acoustic(0.10, 50.0, 1.0);
    json current = acoustic(0.10, 50.0, 1.0);
    baseline.merge_patch(emotions());
    current.merge_patch(emotions());
    baseline["acoustic_features"]["speech_rate"] = 4.0;
    baseline["acoustic_features"]["pause_ratio"] = 0.2;
    baseline["acoustic_features"]["pause_mean_duration"] = 0.1;
    baseline["acoustic_features"]["pause_max_duration"] = 0.1;
    baseline["acoustic_features"]["voice_activity_ratio"] = 0.7;
    current["acoustic_features"]["speech_rate"] = 0.1;
    current["acoustic_features"]["pause_ratio"] = 5.0;
    current["acoustic_features"]["pause_mean_duration"] = 3.0;
    current["acoustic_features"]["pause_max_duration"] = 5.0;
    current["acoustic_features"]["voice_activity_ratio"] = 0.05;
    const audio::Result r = analyzer.analyze(current, baseline);
    EXPECT_NE(r.state, audio::State::BURNOUT_LIKE);
}

// H2: confidence must reflect the actual recording, not be a constant.
// Long duration + several windows + a stored baseline => high confidence.
TEST(BurnoutAnalyzerConfidence, ReflectsDurationAndWindows) {
    audio::BurnoutAnalyzer analyzer;
    json baseline = acoustic(0.10, 50.0, 1.0);
    json current = acoustic(0.10, 50.0, 1.0);
    baseline.merge_patch(emotions());
    current.merge_patch(emotions());

    // Short clip, no stored baseline, single window.
    std::vector<json> no_history;
    const audio::Result short_r =
        analyzer.analyze(current, baseline, no_history, 0.1, 0.3);

    // Long call, stored baseline, five usable windows.
    std::vector<json> five_windows(5, json({}));
    const audio::Result long_r =
        analyzer.analyze(current, baseline, five_windows, 1.0, 0.7);

    // Confidence must be in [0,1] and the rich recording must be strictly
    // more confident than the thin one.
    EXPECT_GE(long_r.confidence, 0.0);
    EXPECT_LE(long_r.confidence, 1.0);
    EXPECT_GT(long_r.confidence, short_r.confidence);
}

// H2: confidence terms are weighted to sum to 1.0, so the maximum achievable
// confidence (full duration, >=3 windows, full baseline reliability) is 1.0.
TEST(BurnoutAnalyzerConfidence, MaxIsOne) {
    EXPECT_DOUBLE_EQ(audio::BurnoutAnalyzer::duration_confidence_weight +
                        audio::BurnoutAnalyzer::windows_confidence_weight +
                        audio::BurnoutAnalyzer::baseline_reliability_confidence,
                     1.0);
}
