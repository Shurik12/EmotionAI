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
