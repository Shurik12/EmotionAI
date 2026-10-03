#include <focus/FocusSessionManager.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include <db/DragonflyManager.h>

namespace
{
    long long nowMs()
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
            .count();
    }

    // additional_probs arrives as decimal strings ("0.82"), but accept numbers too.
    double toNumber(const nlohmann::json &value)
    {
        if (value.is_number())
        {
            return value.get<double>();
        }
        if (value.is_string())
        {
            try
            {
                return std::stod(value.get<std::string>());
            }
            catch (const std::exception &)
            {
                return 0.0;
            }
        }
        return 0.0;
    }

    const std::vector<std::string> &negativeKeys()
    {
        static const std::vector<std::string> keys = {"anger", "fear", "sad", "disgust", "contempt"};
        return keys;
    }

    // Reaction thresholds. The MTL heads (enet_b0_8_va_mtl) are regression
    // outputs on roughly [-1, 1], not probabilities, so an absolute
    // "arousal >= 0.6" is meaningless. Provisional values, calibrated on the
    // ~15 still faces available in the repo/host (no labelled FER set is
    // shipped) — re-measure before trusting them on real sessions.
    constexpr double kNegativeValence = -0.20; // valence at or below this = clearly negative affect
    constexpr double kHighActivation = 0.68;   // activation ((arousal+1)/2) for the negative-emotion path
    constexpr double kStrongNegative = 0.60;   // summed anger/fear/sadness/disgust/contempt
}

FocusSessionManager &FocusSessionManager::instance()
{
    static FocusSessionManager manager;
    return manager;
}

std::string FocusSessionManager::createSession()
{
    const std::string session_id = DragonflyManager::generate_uuid();
    const long long now = nowMs();

    std::lock_guard<std::mutex> lock(mutex_);
    Session session;
    session.created_ms = now;
    session.updated_ms = now;
    sessions_[session_id] = std::move(session);
    return session_id;
}

bool FocusSessionManager::hasSession(const std::string &session_id) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return sessions_.find(session_id) != sessions_.end();
}

void FocusSessionManager::closeSession(const std::string &session_id)
{
    std::lock_guard<std::mutex> lock(mutex_);
    sessions_.erase(session_id);
}

nlohmann::json FocusSessionManager::classify(const nlohmann::json &result)
{
    nlohmann::json sample = {
        {"level", "noSignal"},
        {"arousal", 0.0},
        {"valence", 0.0},
        {"activation", 0.0},
        {"probability", 0.0}};

    if (!result.is_object())
    {
        return sample;
    }

    const auto probs_it = result.find("additional_probs");
    if (probs_it == result.end() || !probs_it->is_object() || probs_it->empty())
    {
        return sample;
    }

    const auto &probs = *probs_it;
    double neutral = 0.0;
    double negative = 0.0;
    bool has_emotion = false;

    for (auto it = probs.begin(); it != probs.end(); ++it)
    {
        if (it.key() == "valence" || it.key() == "arousal")
        {
            continue;
        }
        has_emotion = true;
        const double value = toNumber(it.value());
        if (it.key() == "neutral")
        {
            neutral = value;
        }
        const auto &negatives = negativeKeys();
        if (std::find(negatives.begin(), negatives.end(), it.key()) != negatives.end())
        {
            negative += value;
        }
    }

    if (!has_emotion)
    {
        return sample;
    }

    // The MTL heads are regression outputs on ~[-1, 1]; the 7-class model has
    // neither, so fall back to 1 - neutral as a crude activation proxy.
    const bool has_va = probs.contains("arousal") && probs.contains("valence");
    const double arousal = has_va ? std::clamp(toNumber(probs.at("arousal")), -1.0, 1.0) : 0.0;
    const double valence = has_va ? std::clamp(toNumber(probs.at("valence")), -1.0, 1.0) : 0.0;
    const double activation = has_va
                                  ? std::clamp((arousal + 1.0) / 2.0, 0.0, 1.0)
                                  : std::clamp(1.0 - neutral, 0.0, 1.0);

    std::string label;
    double probability = 0.0;
    const auto main_it = result.find("main_prediction");
    if (main_it != result.end() && main_it->is_object())
    {
        if (main_it->contains("label") && (*main_it)["label"].is_string())
        {
            label = (*main_it)["label"].get<std::string>();
        }
        if (main_it->contains("probability"))
        {
            probability = toNumber((*main_it)["probability"]);
        }
    }

    // "Rising" = clearly negative affect, or a strong negative expression that
    // is also activated. Valence gates the arousal/negative path so a smile
    // with a secondary contempt component does not fire.
    const bool negative_affect = valence <= kNegativeValence;
    const bool activated_negative = activation >= kHighActivation && negative >= kStrongNegative;

    sample["level"] = (negative_affect || activated_negative) ? "rising" : "steady";
    sample["arousal"] = arousal;
    sample["valence"] = valence;
    sample["activation"] = activation;
    sample["negative"] = negative;
    sample["probability"] = probability;
    if (!label.empty())
    {
        sample["label"] = label;
    }
    return sample;
}

void FocusSessionManager::addResult(const std::string &session_id, const nlohmann::json &result)
{
    nlohmann::json sample = classify(result);

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sessions_.find(session_id);
    if (it == sessions_.end())
    {
        return;
    }

    sample["index"] = static_cast<int>(it->second.total);
    sample["at_ms"] = nowMs();
    it->second.samples.push_back(std::move(sample));
    it->second.total += 1;

    if (it->second.samples.size() > kMaxSamples)
    {
        const size_t excess = it->second.samples.size() - kMaxSamples;
        it->second.samples.erase(it->second.samples.begin(), it->second.samples.begin() + excess);
    }
    it->second.updated_ms = nowMs();
}

nlohmann::json FocusSessionManager::getSession(const std::string &session_id) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sessions_.find(session_id);
    if (it == sessions_.end())
    {
        return nullptr;
    }

    const auto &session = it->second;

    nlohmann::json out;
    out["session_id"] = session_id;
    out["count"] = session.samples.size();
    out["total"] = session.total;
    out["samples"] = session.samples;
    out["created_ms"] = session.created_ms;
    out["updated_ms"] = session.updated_ms;

    if (session.samples.empty())
    {
        out["latest"] = nullptr;
        out["dynamics"] = {
            {"count", 0},
            {"mean_arousal", 0.0},
            {"trend", "noSignal"}};
        return out;
    }

    double sum = 0.0;
    for (const auto &sample : session.samples)
    {
        sum += sample.value("arousal", 0.0);
    }

    out["latest"] = session.samples.back();
    out["dynamics"] = {
        {"count", session.samples.size()},
        {"mean_arousal", sum / static_cast<double>(session.samples.size())},
        {"trend", session.samples.back().value("level", "noSignal")}};
    return out;
}
