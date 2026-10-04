#pragma once

#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

// In-memory, per-camera-session aggregation for the Razuma Focus page (EMO-17).
//
// The epoll server stays non-blocking: a frame is decoded and scored on the
// thread pool, then the resulting observation is appended here. The client polls
// the session, splits the frames into a confirmed baseline and the recent
// window, and runs the `emotion-pilot-v1` policy (frontend/src/utils/emotionPolicy.js).
// Nothing is persisted, and sessions are dropped when closed.
class FocusSessionManager
{
public:
    static FocusSessionManager &instance();

    FocusSessionManager(const FocusSessionManager &) = delete;
    FocusSessionManager &operator=(const FocusSessionManager &) = delete;

    std::string createSession();
    bool hasSession(const std::string &session_id) const;
    void closeSession(const std::string &session_id);

    // Normalize one core image result and append it. No-op if the session is gone.
    void addResult(const std::string &session_id, const nlohmann::json &result);

    // {session_id, count, total, samples:[...], latest, dynamics} or null.
    nlohmann::json getSession(const std::string &session_id) const;

    // Adapter from the core image result to one normalized EmotionFrame:
    //   {valid, valence, arousal, intensity, emotions:{name:score}, label, probability}
    // valence stays in [-1,1], arousal is rescaled to [0,1], intensity and each
    // category score are in [0,1]. A frame is `valid` only when a face was
    // detected AND both valence/arousal heads exist; the 7-class model has no
    // regression heads, so its frames stay invalid (a missing channel is never
    // replaced with 0). Emits data only, like BurnoutAnalyzer — never text.
    static nlohmann::json classify(const nlohmann::json &result);

private:
    FocusSessionManager() = default;

    struct Session
    {
        std::vector<nlohmann::json> samples;
        size_t total{0}; // samples ever added, even after old ones are dropped
        long long created_ms{0};
        long long updated_ms{0};
    };

    // Bound memory for a long session; oldest samples are dropped first.
    // 1 s sampling for 30 min fits without dropping.
    static constexpr size_t kMaxSamples = 1800;

    mutable std::mutex mutex_;
    std::map<std::string, Session> sessions_;
};
