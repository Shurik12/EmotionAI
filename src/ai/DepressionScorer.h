#pragma once

#include <nlohmann/json.hpp>
#include <string>

namespace emotionai
{
    namespace ai
    {

        // Deterministic, dependency-free reimplementation of the depression-risk
        // interpretation that the LLM second stage (AIClient::assessDepression)
        // otherwise produces. It encodes the exact scoring table that lives in
        // that prompt (valence / intensity / joy / negative-affect / disgust),
        // so an operator can run the same verdict with no AI provider, no API
        // key and no outbound network call.
        //
        // `assess` is pure: no I/O, no Config, no clock. Callers attach the
        // result as the same `ai_analysis` block the LLM path emits —
        // {verdict, probability, reasoning, scores, source} — so the frontend
        // renders it identically. `source` is "local" here versus "llm" for the
        // model path, letting the UI tell them apart when it wants to.
        class DepressionScorer
        {
        public:
            // Emotion keys the table understands. `happiness` is the joy
            // channel (the 7-class model emits it; the LLM prompt calls it
            // "Joy (happiness)"), `contempt` exists on the 8-class model but is
            // not scored.
            static constexpr const char *kAnger = "anger";
            static constexpr const char *kContempt = "contempt";
            static constexpr const char *kDisgust = "disgust";
            static constexpr const char *kFear = "fear";
            static constexpr const char *kHappiness = "happiness";
            static constexpr const char *kNeutral = "neutral";
            static constexpr const char *kSadness = "sadness";
            static constexpr const char *kSurprise = "surprise";

            // Compute the verdict from per-emotion probabilities in [0,1].
            // `lang` selects the reasoning language ("ru" default / "en"); any
            // other value falls back to Russian.
            // Returns a null json when the input is empty or carries no usable
            // emotion channel, so the caller can skip the stage exactly like
            // the LLM path does on a failed call.
            static nlohmann::json assess(const nlohmann::json &emotions,
                                         const std::string &lang = "ru");
        };

    } // namespace ai
} // namespace emotionai
