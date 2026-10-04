#include "AIClient.h"

#include <curl/curl.h>

#include <algorithm>
#include <cctype>

namespace emotionai
{
    namespace ai
    {

        namespace
        {
            size_t WriteCallback(void *contents, size_t size, size_t nmemb, std::string *s)
            {
                size_t newLength = size * nmemb;
                s->append(static_cast<char *>(contents), newLength);
                return newLength;
            }

            // The model may wrap the JSON in ```json fences or add prose around
            // it, so extract the outermost object before parsing.
            nlohmann::json extractJson(const std::string &raw)
            {
                const auto start = raw.find('{');
                const auto end = raw.rfind('}');
                if (start == std::string::npos || end == std::string::npos || end < start)
                    return nlohmann::json();

                try
                {
                    return nlohmann::json::parse(raw.substr(start, end - start + 1));
                }
                catch (...)
                {
                    return nlohmann::json();
                }
            }

            bool isSupportedLang(const std::string &lang)
            {
                return lang == "ru" || lang == "en";
            }
        } // namespace

        AIClient::AIClient(const AIConfig &config) : m_config(config)
        {
        }

        AIClient::~AIClient() = default;

        std::string AIClient::chatCompletion(const std::string &systemPrompt,
                                             const std::string &userPrompt)
        {
            if (!m_config.enabled)
                return "";

            CURL *curl = curl_easy_init();
            if (!curl)
                return "";

            std::string response;
            const std::string url = m_config.baseUrl + "/chat/completions";

            nlohmann::json request = {
                {"model", m_config.model},
                {"temperature", 0.4},
                {"messages", nlohmann::json::array(
                                 {{{"role", "system"}, {"content", systemPrompt}},
                                  {{"role", "user"}, {"content", userPrompt}}})}};

            struct curl_slist *headers = nullptr;
            headers = curl_slist_append(headers, "Content-Type: application/json");
            headers = curl_slist_append(headers, "Accept: application/json");
            headers = curl_slist_append(headers, ("Authorization: Bearer " + m_config.apiKey).c_str());

            const std::string json_str = request.dump();

            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_POST, 1L);
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_str.c_str());
            curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, m_config.verifySsl ? 1L : 0L);
            curl_easy_setopt(curl, CURLOPT_TIMEOUT, m_config.timeoutSeconds);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

            CURLcode res = curl_easy_perform(curl);
            long http_code = 0;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);

            if (res != CURLE_OK || http_code < 200 || http_code >= 300)
                return "";

            try
            {
                auto json_response = nlohmann::json::parse(response);
                if (json_response.contains("choices") &&
                    !json_response["choices"].empty() &&
                    json_response["choices"][0].contains("message") &&
                    json_response["choices"][0]["message"].contains("content"))
                {
                    return json_response["choices"][0]["message"]["content"].get<std::string>();
                }
            }
            catch (...)
            {
            }

            return "";
        }

        nlohmann::json AIClient::breakdownTask(const std::string &text,
                                               const std::string &duration,
                                               const std::string &lang)
        {
            if (!m_config.enabled || text.empty())
                return nlohmann::json();

            const std::string language = isSupportedLang(lang) ? lang : "ru";
            const std::string language_name = (language == "en") ? "английском (English)" : "русском (Russian)";

            const std::string systemPrompt =
                "Ты — ассистент Razuma, помогаешь человеку начать сложную задачу. "
                "Разбей задачу на 3 небольших посильных шага. Первый шаг должен быть совсем маленьким, "
                "чтобы снизить порог входа. Шаги должны быть конкретными и выполнимыми за один подход. "
                "Ответь строго в формате JSON без дополнительного текста: "
                "{\"steps\": [\"шаг 1\", \"шаг 2\", \"шаг 3\"]}. "
                "Текст шагов верни на " +
                language_name + ".";

            std::string userPrompt = "Задача: " + text;
            if (!duration.empty())
                userPrompt += "\nДлительность блока: " + duration;

            const std::string content = chatCompletion(systemPrompt, userPrompt);
            if (content.empty())
                return nlohmann::json();

            const nlohmann::json parsed = extractJson(content);
            if (!parsed.is_object() || !parsed.contains("steps") || !parsed["steps"].is_array())
                return nlohmann::json();

            nlohmann::json steps = nlohmann::json::array();
            for (const auto &step : parsed["steps"])
            {
                if (step.is_string())
                {
                    const std::string value = step.get<std::string>();
                    if (!value.empty())
                        steps.push_back(value);
                }
            }

            if (steps.empty())
                return nlohmann::json();

            return nlohmann::json{{"steps", steps}};
        }

    } // namespace ai
} // namespace emotionai
