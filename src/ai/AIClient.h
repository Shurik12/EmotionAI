#pragma once

#include <string>
#include <nlohmann/json.hpp>

namespace emotionai
{
    namespace ai
    {

        struct AIConfig
        {
            bool enabled{false};
            std::string baseUrl;
            std::string model;
            std::string apiKey;
            bool verifySsl{true};
            long timeoutSeconds{10};
            long connectTimeoutSeconds{5};
            // The image/video interpretation is more verbose (the model emits a
            // reasoning trace) and runs on a worker thread, so it gets a larger
            // budget than the epoll-thread Focus breakdown.
            long mediaTimeoutSeconds{60};
        };

        struct TokenUsage
        {
            int promptTokens{0};
            int completionTokens{0};
            int totalTokens{0};
        };

        // Thin OpenAI-compatible chat client (POST {baseUrl}/chat/completions).
        //
        // Used by the Razuma Focus task breakdown. The only data ever sent is
        // the task text and the block duration — never camera frames, emotions
        // or any personal data.
        class AIClient
        {
        public:
            explicit AIClient(const AIConfig &config);
            ~AIClient();

            bool isEnabled() const { return m_config.enabled; }

            // Splits a task into ~3 feasible steps.
            // `duration` is a free-form label (e.g. "25 мин") or empty.
            // `lang` is "ru" or "en".
            // Returns {"steps": ["...", ...], "usage": {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}}
            // on success, an empty (null) json on any failure so the caller can fall back.
            nlohmann::json breakdownTask(const std::string &text,
                                         const std::string &duration,
                                         const std::string &lang);

            // Second-stage interpretation of an image/video emotion result.
            //
            // `emotions` is an object of per-emotion probabilities in [0,1]
            // (the "additional_probs" block, or the averaged emotions of a
            // video). Returns
            //   {"verdict": "low"|"monitor"|"high", "probability": 0..1,
            //    "reasoning": "...", "scores": {...},
            //    "usage": {"prompt_tokens": N, ...}}
            // or a null json on any failure, so the caller can simply skip the
            // second stage. Only emotion probabilities are sent — never pixel
            // data, frames or files.
            nlohmann::json assessDepression(const nlohmann::json &emotions);

        private:
            // Returns the assistant message content, or "" on failure.
            // If usage is non-null, fills it with token counts from the response.
            // `timeoutSecondsOverride` > 0 replaces the configured timeout.
            std::string chatCompletion(const std::string &systemPrompt,
                                       const std::string &userPrompt,
                                       TokenUsage *usage = nullptr,
                                       long timeoutSecondsOverride = 0);

            AIConfig m_config;
        };

    } // namespace ai
} // namespace emotionai
