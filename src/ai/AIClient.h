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
            long timeoutSeconds{30};
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
            // Returns {"steps": ["...", ...]} on success, an empty
            // (null) json on any failure so the caller can fall back.
            nlohmann::json breakdownTask(const std::string &text,
                                         const std::string &duration,
                                         const std::string &lang);

        private:
            // Returns the assistant message content, or "" on failure.
            std::string chatCompletion(const std::string &systemPrompt,
                                       const std::string &userPrompt);

            AIConfig m_config;
        };

    } // namespace ai
} // namespace emotionai
