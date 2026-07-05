#include "GigaChatClient.h"
#include <curl/curl.h>
#include <chrono>
#include <iostream>

namespace emotionai
{
    namespace gigachat
    {

        static size_t WriteCallback(void *contents, size_t size, size_t nmemb, std::string *s)
        {
            size_t newLength = size * nmemb;
            s->append((char *)contents, newLength);
            return newLength;
        }

        GigaChatClient::GigaChatClient(const GigaChatConfig &config) : m_config(config)
        {
            if (m_config.enabled)
            {
                fetchNewToken();
            }
        }

        GigaChatClient::~GigaChatClient() = default;

        bool GigaChatClient::fetchNewToken()
        {
            CURL *curl = curl_easy_init();
            if (!curl)
                return false;

            std::string response;

            curl_easy_setopt(curl, CURLOPT_URL, m_config.authUrl.c_str());
            curl_easy_setopt(curl, CURLOPT_POST, 1L);

            struct curl_slist *headers = nullptr;
            headers = curl_slist_append(headers, "Content-Type: application/x-www-form-urlencoded");
            headers = curl_slist_append(headers, "Accept: application/json");
            headers = curl_slist_append(headers, ("Authorization: Basic " + m_config.authKey).c_str());
            headers = curl_slist_append(headers, "RqUID: 313eca76-ba14-498e-8501-8e63258672d8");
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, "scope=GIGACHAT_API_PERS");
            curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, m_config.verifySsl ? 1L : 0L);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

            CURLcode res = curl_easy_perform(curl);
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);

            if (res == CURLE_OK)
            {
                try
                {
                    auto json = nlohmann::json::parse(response);
                    m_accessToken = json["access_token"];
                    long expires_at_ms = json["expires_at"];
                    m_expiresAt = expires_at_ms / 1000;
                    return true;
                }
                catch (...)
                {
                }
            }
            return false;
        }

        bool GigaChatClient::ensureValidToken()
        {
            if (m_accessToken.empty())
                return fetchNewToken();

            auto now = std::chrono::system_clock::now();
            auto now_sec = std::chrono::duration_cast<std::chrono::seconds>(
                               now.time_since_epoch())
                               .count();

            if (now_sec >= (m_expiresAt - 120))
            {
                return fetchNewToken();
            }
            return true;
        }

        std::string GigaChatClient::buildPrompt(const EmotionData &emotions) const
        {
                return R"(
            Ты - AI ассистент для анализа эмоций сотрудника. На основе вероятностей эмоций оцени риск депрессивного состояния.

            Эмоции (вероятности от 0 до 1):
            )" + emotions.toJson().dump() + R"(

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
            }

            Твой ответ (только JSON):
            )";
        }

        std::string GigaChatClient::analyzeEmotions(const EmotionData &emotions, const std::string &sessionId)
        {
            if (!m_config.enabled)
                return "";

            if (!ensureValidToken())
                return "";

            CURL *curl = curl_easy_init();
            if (!curl)
                return "";

            std::string response;
            std::string url = m_config.apiUrl + "/chat/completions";

            nlohmann::json request = {
                {"model", m_config.model},
                {"messages", nlohmann::json::array({{{"role", "user"},
                                                     {"content", buildPrompt(emotions)}}})}};

            struct curl_slist *headers = nullptr;
            headers = curl_slist_append(headers, "Content-Type: application/json");
            headers = curl_slist_append(headers, "Accept: application/json");
            headers = curl_slist_append(headers, ("Authorization: Bearer " + m_accessToken).c_str());
            headers = curl_slist_append(headers, ("X-Session-ID: " + sessionId).c_str());

            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_POST, 1L);
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

            std::string json_str = request.dump();
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_str.c_str());

            curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, m_config.verifySsl ? 1L : 0L);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

            CURLcode res = curl_easy_perform(curl);
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);

            if (res != CURLE_OK)
                return "";

            try
            {
                auto json_response = nlohmann::json::parse(response);
                if (json_response.contains("choices") &&
                    !json_response["choices"].empty() &&
                    json_response["choices"][0].contains("message"))
                {

                    return json_response["choices"][0]["message"]["content"];
                }
            }
            catch (...)
            {
            }

            return "";
        }

    } // namespace gigachat
} // namespace emotionai