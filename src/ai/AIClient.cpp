#include "AIClient.h"

#include <curl/curl.h>

#include <algorithm>
#include <cctype>
#include <vector>

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
                                             const std::string &userPrompt,
                                             TokenUsage *usage,
                                             long timeoutSecondsOverride)
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
            const long timeout = timeoutSecondsOverride > 0 ? timeoutSecondsOverride
                                                            : m_config.timeoutSeconds;
            curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout);
            curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, m_config.connectTimeoutSeconds);
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

                if (usage && json_response.contains("usage"))
                {
                    const auto &u = json_response["usage"];
                    usage->promptTokens = u.value("prompt_tokens", 0);
                    usage->completionTokens = u.value("completion_tokens", 0);
                    usage->totalTokens = u.value("total_tokens", 0);
                }

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

            std::string systemPrompt, userPrompt;
            if (language == "en")
            {
                systemPrompt =
                    "You are a Razuma assistant that helps a person start a difficult task. "
                    "Break the task into 3 small, manageable steps. The first step must be very small "
                    "to lower the entry barrier. Steps must be concrete and completable in one sitting. "
                    "Reply strictly in JSON with no extra text: "
                    "{\"steps\": [\"step 1\", \"step 2\", \"step 3\"]}.";
                userPrompt = "Task: " + text;
                if (!duration.empty())
                    userPrompt += "\nBlock duration: " + duration;
            }
            else
            {
                systemPrompt =
                    "Ты — ассистент Razuma, помогаешь человеку начать сложную задачу. "
                    "Разбей задачу на 3 небольших посильных шага. Первый шаг должен быть совсем маленьким, "
                    "чтобы снизить порог входа. Шаги должны быть конкретными и выполнимыми за один подход. "
                    "Ответь строго в формате JSON без дополнительного текста: "
                    "{\"steps\": [\"шаг 1\", \"шаг 2\", \"шаг 3\"]}.";
                userPrompt = "Задача: " + text;
                if (!duration.empty())
                    userPrompt += "\nДлительность блока: " + duration;
            }

            TokenUsage usage;
            const std::string content = chatCompletion(systemPrompt, userPrompt, &usage);
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

            return nlohmann::json{
                {"steps", steps},
                {"usage", nlohmann::json{
                     {"prompt_tokens", usage.promptTokens},
                     {"completion_tokens", usage.completionTokens},
                     {"total_tokens", usage.totalTokens}}}};
        }

        nlohmann::json AIClient::assessDepression(const nlohmann::json &emotions)
        {
            if (!m_config.enabled || !emotions.is_object() || emotions.empty())
                return nlohmann::json();

            // Keep only the emotion probabilities; drop derived features
            // (valence/arousal) so the model always sees a stable schema.
            static const std::vector<std::string> keys = {
                "anger", "contempt", "disgust", "fear",
                "happiness", "neutral", "sadness", "surprise"};

            nlohmann::json filtered = nlohmann::json::object();
            for (const auto &key : keys)
            {
                if (!emotions.contains(key))
                    continue;
                const auto &value = emotions[key];
                if (value.is_number())
                {
                    filtered[key] = value.get<double>();
                }
                else if (value.is_string())
                {
                    try
                    {
                        filtered[key] = std::stod(value.get<std::string>());
                    }
                    catch (...)
                    {
                    }
                }
            }

            if (filtered.empty())
                return nlohmann::json();

            const std::string systemPrompt = R"(Ты - AI ассистент для анализа эмоций сотрудника. На основе вероятностей эмоций оцени риск депрессивного состояния.

Правила анализа:

1. Вычисли показатели:
- Valence = happiness - (anger + disgust + fear + sadness + surprise) / 5
- Intensity = 1 - neutral
- NA (Negative Affect) = sadness + fear + disgust + anger

2. Рассчитай баллы по таблице:

| Признак | Условие | Баллы |
|---------|---------|-------|
| Valence | <= -0.20 | 2 |
| Valence | -0.20 < Valence <= 0 | 1 |
| Intensity | <= -0.20 | 2 |
| Intensity | -0.20 < Intensity <= -0.10 | 1 |
| Joy (happiness) | <= 0.10 | 2 |
| Joy (happiness) | 0.10-0.20 | 1 |
| NA | >= 0.50 | 2 |
| NA | 0.35-0.49 | 1 |
| Disgust | >= 0.15 | 1 |

3. Определи вердикт по сумме баллов:
- 0-2 балла → verdict = "low" (Низкая вероятность депрессии)
- 3-5 баллов → verdict = "monitor" (Требуется наблюдение)
- 6+ баллов → verdict = "high" (Высокая вероятность депрессии)

4. Вероятность (probability):
- Для low: 0.1-0.3
- Для monitor: 0.4-0.7
- Для high: 0.8-0.95

5. Reasoning (на русском, кратко, по шаблону):

Для low:
"На текущем видео не выявлен выраженный депрессивный аффективный паттерн. Эмоциональная экспрессивность сохранена, положительный аффект представлен, выраженного негативного смещения и признаков эмоционального уплощения не наблюдается."

Для monitor:
"Выявлены отдельные визуальные маркеры, ассоциированные с депрессивным аффективным паттерном: [указать конкретные признаки - сниженная доля положительного аффекта, умеренное негативное смещение, признаки эмоционального уплощения, рост паттернов презрения и/или отвращения]. Такое сочетание требует наблюдения в динамике и более частых контактов со специалистом."

Для high:
"Выявлена выраженная комбинация визуальных аффективных признаков, ассоциированных с депрессивной симптоматикой: [указать конкретные признаки - отрицательное смещение валентности, дефицит положительного аффекта, признаки эмоционального уплощения, доминирование негативных паттернов]. Такая комбинация повышает вероятность клинически значимого депрессивного состояния и требует очной оценки профильным специалистом."

Ответь строго в формате JSON без дополнительного текста:
{
    "verdict": "low" или "monitor" или "high",
    "probability": число от 0 до 1,
    "reasoning": "объяснение на русском языке",
    "scores": {
        "valence_score": число,
        "intensity_score": число,
        "joy_score": число,
        "na_score": число,
        "disgust_score": число,
        "total_score": число
    }
})";

            const std::string userPrompt =
                "Эмоции (вероятности от 0 до 1):\n" + filtered.dump(2) +
                "\n\nТвой ответ (только JSON):";

            TokenUsage usage;
            const std::string content = chatCompletion(systemPrompt, userPrompt, &usage,
                                                       m_config.mediaTimeoutSeconds);
            if (content.empty())
                return nlohmann::json();

            const nlohmann::json parsed = extractJson(content);
            if (!parsed.is_object())
                return nlohmann::json();

            const std::string verdict = parsed.value("verdict", "");
            if (verdict != "low" && verdict != "monitor" && verdict != "high")
                return nlohmann::json();

            return nlohmann::json{
                {"verdict", verdict},
                {"probability", parsed.value("probability", 0.0)},
                {"reasoning", parsed.value("reasoning", "")},
                {"scores", parsed.contains("scores") && parsed["scores"].is_object()
                               ? parsed["scores"]
                               : nlohmann::json::object()},
                {"usage", nlohmann::json{
                     {"prompt_tokens", usage.promptTokens},
                     {"completion_tokens", usage.completionTokens},
                     {"total_tokens", usage.totalTokens}}}};
        }

    } // namespace ai
} // namespace emotionai
