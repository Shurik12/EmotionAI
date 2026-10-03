#pragma once

#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

// In-memory, per-camera-session aggregation for the Razuma Focus page (EMO-17).
//
// The epoll server stays non-blocking: a frame is decoded and scored on the
// thread pool, then the resulting signal is appended here. The client polls the
// session to read the accumulated emotional dynamics over the whole session.
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

    // Classify one core image result and append it. No-op if the session is gone.
    void addResult(const std::string &session_id, const nlohmann::json &result);

    // {session_id, count, samples:[...], latest:{...}, dynamics:{...}} or null.
    nlohmann::json getSession(const std::string &session_id) const;

    // Deterministic mapping of a core image result to a calm reaction level
    // ("noSignal" | "steady" | "rising") plus the signal values used for it.
    // Emits KEYS/symbols only, like BurnoutAnalyzer — never human-readable text.
    static nlohmann::json classify(const nlohmann::json &result);

private:
    FocusSessionManager() = default;

    struct Session
    {
        std::vector<nlohmann::json> samples;
        long long created_ms{0};
        long long updated_ms{0};
    };

    // Bound memory for a long session; oldest samples are dropped first.
    static constexpr size_t kMaxSamples = 240;

    mutable std::mutex mutex_;
    std::map<std::string, Session> sessions_;
};
