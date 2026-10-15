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

        private:
            // Returns the assistant message content, or "" on failure.
            // If usage is non-null, fills it with token counts from the response.
            std::string chatCompletion(const std::string &systemPrompt,
                                       const std::string &userPrompt,
                                       TokenUsage *usage = nullptr);

            AIConfig m_config;
        };

    } // namespace ai
} // namespace emotionai
