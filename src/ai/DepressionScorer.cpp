#include "DepressionScorer.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace emotionai
{
    namespace ai
    {

        namespace
        {

            // Read a probability field that may arrive as a number or a string
            // (the emotion pipeline emits numbers, but accept both to be
            // robust). Returns 0.0 for missing/non-numeric entries.
            double emotionValue(const nlohmann::json &emotions, const char *key)
            {
                if (!emotions.is_object() || !emotions.contains(key))
                    return 0.0;
                const auto &value = emotions[key];
                if (value.is_number())
                {
                    return std::clamp(value.get<double>(), 0.0, 1.0);
                }
                if (value.is_string())
                {
                    try
                    {
                        return std::clamp(std::stod(value.get<std::string>()), 0.0, 1.0);
                    }
                    catch (...)
                    {
                        return 0.0;
                    }
                }
                return 0.0;
            }

            struct EmotionValues
            {
                double anger{0.0};
                double contempt{0.0};
                double disgust{0.0};
                double fear{0.0};
                double happiness{0.0};
                double neutral{0.0};
                double sadness{0.0};
                double surprise{0.0};
                bool any{false};
            };

            EmotionValues readEmotions(const nlohmann::json &emotions)
            {
                EmotionValues v;
                v.anger = emotionValue(emotions, DepressionScorer::kAnger);
                v.contempt = emotionValue(emotions, DepressionScorer::kContempt);
                v.disgust = emotionValue(emotions, DepressionScorer::kDisgust);
                v.fear = emotionValue(emotions, DepressionScorer::kFear);
                v.happiness = emotionValue(emotions, DepressionScorer::kHappiness);
                v.neutral = emotionValue(emotions, DepressionScorer::kNeutral);
                v.sadness = emotionValue(emotions, DepressionScorer::kSadness);
                v.surprise = emotionValue(emotions, DepressionScorer::kSurprise);
                // A usable input has at least one non-zero emotion channel.
                v.any = v.anger + v.contempt + v.disgust + v.fear + v.happiness +
                            v.neutral + v.sadness + v.surprise >
                        0.0;
                return v;
            }

            // Mirrors the LLM prompt's metrics.
            struct Derived
            {
                double valence;
                double intensity;
                double negativeAffect;
            };

            Derived derive(const EmotionValues &v)
            {
                Derived d;
                d.valence = v.happiness -
                            (v.anger + v.disgust + v.fear + v.sadness + v.surprise) / 5.0;
                d.intensity = 1.0 - v.neutral;
                d.negativeAffect = v.sadness + v.fear + v.disgust + v.anger;
                return d;
            }

            // The prompt's "Intensity" row is a no-op as written: it is keyed on
            // Intensity = 1 - neutral (always in [0, 1]) with negative
            // thresholds (<= -0.20), so it can never fire. The row's clear intent
            // is *emotional flattening* — a near-blank, high-neutral face — which
            // the other rows (valence/joy/NA/disgust) all miss on a flat face.
            // We therefore score `neutral` directly, keeping the row's 2/1 point
            // structure. This only ever *adds* points relative to the no-op
            // prompt, so it can only escalate a verdict, never de-escalate one.
            int flatteningScore(double neutral)
            {
                if (neutral >= 0.80)
                    return 2;
                if (neutral >= 0.65)
                    return 1;
                return 0;
            }

            // The scoring table from the LLM prompt, transcribed verbatim.
            struct Scores
            {
                int valence_score{0};
                int intensity_score{0};
                int joy_score{0};
                int na_score{0};
                int disgust_score{0};
                int total_score{0};
            };

            Scores score(const EmotionValues &v, const Derived &d)
            {
                Scores s;
                // Valence: <= -0.20 -> 2 ; -0.20 < v <= 0 -> 1
                if (d.valence <= -0.20)
                    s.valence_score = 2;
                else if (d.valence <= 0.0)
                    s.valence_score = 1;

                // Intensity row: scores emotional flattening (high neutral), see
                // flatteningScore() for why this is not the prompt's literal
                // (no-op) `1 - neutral` condition.
                s.intensity_score = flatteningScore(v.neutral);

                // Joy (happiness): <= 0.10 -> 2 ; 0.10 < h <= 0.20 -> 1
                if (v.happiness <= 0.10)
                    s.joy_score = 2;
                else if (v.happiness <= 0.20)
                    s.joy_score = 1;

                // Negative Affect: >= 0.50 -> 2 ; 0.35 <= na < 0.50 -> 1
                if (d.negativeAffect >= 0.50)
                    s.na_score = 2;
                else if (d.negativeAffect >= 0.35)
                    s.na_score = 1;

                // Disgust: >= 0.15 -> 1
                if (v.disgust >= 0.15)
                    s.disgust_score = 1;

                s.total_score = s.valence_score + s.intensity_score + s.joy_score +
                                s.na_score + s.disgust_score;
                return s;
            }

            const char *verdictFor(int total)
            {
                if (total <= 2)
                    return "low";
                if (total <= 5)
                    return "monitor";
                return "high";
            }

            // Representative probability per verdict band, matching the ranges
            // the LLM is instructed to emit (low 0.1-0.3, monitor 0.4-0.7,
            // high 0.8-0.95).
            double probabilityFor(const std::string &verdict, int total)
            {
                if (verdict == "high")
                {
                    // 0.8-0.95; scale up within the band as the score grows.
                    return std::clamp(0.80 + 0.025 * static_cast<double>(total - 6),
                                      0.80, 0.95);
                }
                if (verdict == "monitor")
                {
                    // 0.4-0.7; total is 3..5.
                    return std::clamp(0.40 + 0.15 * static_cast<double>(total - 3),
                                      0.40, 0.70);
                }
                // low: total is 0..2.
                return std::clamp(0.10 + 0.10 * static_cast<double>(total), 0.10, 0.30);
            }

            // Which rows actually fired, so the reasoning can name the concrete
            // markers (the LLM is instructed to do the same).
            std::vector<std::string> firedMarkers(const Scores &s)
            {
                std::vector<std::string> markers;
                if (s.valence_score > 0)
                    markers.push_back("valence");
                if (s.intensity_score > 0)
                    markers.push_back("intensity");
                if (s.joy_score > 0)
                    markers.push_back("joy");
                if (s.na_score > 0)
                    markers.push_back("negative_affect");
                if (s.disgust_score > 0)
                    markers.push_back("disgust");
                return markers;
            }

            // The reasoning text mirrors the LLM prompt's three templates
            // (low / monitor / high), with the concrete markers substituted
            // where the template says "указать конкретные признаки".
            std::string buildReasoning(const std::string &verdict, const std::vector<std::string> &markers,
                                       const std::string &lang)
            {
                const bool en = (lang == "en");
                std::string marker_list;
                if (!markers.empty())
                {
                    marker_list = en ? markers[0] : markers[0];
                    for (size_t i = 1; i < markers.size(); ++i)
                        marker_list += ", " + markers[i];
                }

                if (verdict == "low")
                {
                    return en
                        ? "No pronounced depressive affective pattern was detected on the current video. "
                          "Emotional expressiveness is preserved, positive affect is present, and there is no "
                          "pronounced negative shift or sign of emotional flattening."
                        : "На текущем видео не выявлен выраженный депрессивный аффективный паттерн. Эмоциональная "
                          "экспрессивность сохранена, положительный аффект представлен, выраженного негативного "
                          "смещения и признаков эмоционального уплощения не наблюдается.";
                }
                if (verdict == "monitor")
                {
                    const std::string specific =
                        marker_list.empty()
                            ? (en ? "reduced positive affect and mild negative shift"
                                 : "сниженная доля положительного аффекта и умеренное негативное смещение")
                            : (en ? marker_list : marker_list);
                    return en
                        ? "Isolated visual markers associated with a depressive affective pattern were detected: "
                          + specific +
                          ". This combination requires monitoring over time and more frequent contact with a "
                          "specialist."
                        : "Выявлены отдельные визуальные маркеры, ассоциированные с депрессивным аффективным "
                          "паттерном: " + specific +
                          ". Такое сочетание требует наблюдения в динамике и более частых контактов со "
                          "специалистом.";
                }
                // high
                const std::string specific =
                    marker_list.empty()
                        ? (en ? "negative valence shift and deficit of positive affect"
                             : "отрицательное смещение валентности и дефицит положительного аффекта")
                        : (en ? marker_list : marker_list);
                return en
                    ? "A pronounced combination of visual affective signs associated with depressive symptomatology "
                      "was detected: " +
                      specific +
                      ". This combination increases the likelihood of a clinically significant depressive state and "
                      "requires an in-person assessment by a specialist."
                    : "Выявлена выраженная комбинация визуальных аффективных признаков, ассоциированных с "
                      "депрессивной симптоматикой: " +
                      specific +
                      ". Такая комбинация повышает вероятность клинически значимого депрессивного состояния и "
                      "требует очной оценки профильным специалистом.";
            }

        } // namespace

        nlohmann::json DepressionScorer::assess(const nlohmann::json &emotions,
                                                const std::string &lang)
        {
            if (!emotions.is_object() || emotions.empty())
                return nlohmann::json();

            const auto v = readEmotions(emotions);
            if (!v.any)
                return nlohmann::json();

            const auto d = derive(v);
            const auto s = score(v, d);

            const std::string verdict = verdictFor(s.total_score);
            const double probability = probabilityFor(verdict, s.total_score);
            const auto markers = firedMarkers(s);
            const std::string reasoning = buildReasoning(verdict, markers, lang);

            return nlohmann::json{
                {"verdict", verdict},
                {"probability", std::round(probability * 100.0) / 100.0},
                {"reasoning", reasoning},
                {"scores", nlohmann::json{
                     {"valence_score", s.valence_score},
                     {"intensity_score", s.intensity_score},
                     {"joy_score", s.joy_score},
                     {"na_score", s.na_score},
                     {"disgust_score", s.disgust_score},
                     {"total_score", s.total_score}}},
                {"source", "local"}};
        }

    } // namespace ai
} // namespace emotionai
