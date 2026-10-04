#include <focus/FocusSessionManager.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>
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

    // Map a core category name to the policy's logical name. The policy only
    // uses fear/anger/disgust/sadness actively, but neutral/joy/surprise/contempt
    // are carried through for display and baseline statistics. Unknown or
    // absent categories are never invented.
    std::string policyEmotionName(const std::string &key)
    {
        if (key == "happiness")
        {
            return "joy";
        }
        static const std::set<std::string> known = {
            "fear", "anger", "disgust", "sadness", "joy", "surprise", "neutral", "interest", "shame", "contempt"};
        return known.count(key) ? key : std::string();
    }
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
        {"valid", false},
        {"valence", 0.0},
        {"arousal", 0.0},
        {"intensity", 0.0},
        {"emotions", nlohmann::json::object()}};

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
    nlohmann::json emotions = nlohmann::json::object();
    // The core has no separate intensity channel, so the adapter uses the
    // strongest category score as the expression strength. This is an adapter
    // choice, not a probability and not an independent sensor.
    double intensity = 0.0;
    bool has_emotion = false;

    for (auto it = probs.begin(); it != probs.end(); ++it)
    {
        if (it.key() == "valence" || it.key() == "arousal")
        {
            continue;
        }
        const std::string name = policyEmotionName(it.key());
        if (name.empty())
        {
            continue;
        }
        // The core rounds category scores to two decimals; clamp defensively.
        const double value = std::clamp(toNumber(it.value()), 0.0, 1.0);
        emotions[name] = value;
        intensity = std::max(intensity, value);
        has_emotion = true;
    }

    // The MTL heads (enet_b0_8_va_mtl) are regression outputs on ~[-1, 1]; the
    // 7-class model has neither, so its frames cannot satisfy the policy and
    // stay invalid rather than being filled with a fabricated 0.
    const bool has_va = probs.contains("valence") && probs.contains("arousal");
    sample["valid"] = has_emotion && has_va;
    sample["intensity"] = intensity;
    sample["emotions"] = std::move(emotions);

    if (has_va)
    {
        sample["valence"] = std::clamp(toNumber(probs.at("valence")), -1.0, 1.0);
        const double arousal = std::clamp(toNumber(probs.at("arousal")), -1.0, 1.0);
        // The policy expects activation on [0, 1].
        sample["arousal"] = std::clamp((arousal + 1.0) / 2.0, 0.0, 1.0);
    }

    const auto main_it = result.find("main_prediction");
    if (main_it != result.end() && main_it->is_object())
    {
        if (main_it->contains("label") && (*main_it)["label"].is_string())
        {
            sample["label"] = (*main_it)["label"].get<std::string>();
        }
        if (main_it->contains("probability"))
        {
            sample["probability"] = toNumber((*main_it)["probability"]);
        }
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
            {"valid_count", 0},
            {"mean_arousal", 0.0},
            {"mean_intensity", 0.0}};
        return out;
    }

    double sum_arousal = 0.0;
    double sum_intensity = 0.0;
    size_t valid_count = 0;
    for (const auto &sample : session.samples)
    {
        sum_arousal += sample.value("arousal", 0.0);
        sum_intensity += sample.value("intensity", 0.0);
        if (sample.value("valid", false))
        {
            valid_count += 1;
        }
    }

    const auto size = static_cast<double>(session.samples.size());
    out["latest"] = session.samples.back();
    out["dynamics"] = {
        {"count", session.samples.size()},
        {"valid_count", valid_count},
        {"mean_arousal", sum_arousal / size},
        {"mean_intensity", sum_intensity / size}};
    return out;
}
