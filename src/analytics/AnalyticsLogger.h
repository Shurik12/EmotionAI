#pragma once

#include <fstream>
#include <mutex>
#include <string>

#include <nlohmann/json.hpp>

// Structured, append-only event log for the contest analytics requirement
// (п. 5.1): one JSON object per line (JSONL), so the file is machine-readable
// and can be attached to a report as-is.
//
// Every request record carries an ANONYMIZED user id (never the raw client id),
// a UTC timestamp, the traffic source, the call type, the HTTP result and an
// error code. LLM invocations additionally carry token usage (the LLM-cost
// metric). Writes are mutex-guarded and flushed per line, so a reader always
// sees complete records — even while the server keeps running.
class AnalyticsLogger
{
public:
    static AnalyticsLogger &instance();

    AnalyticsLogger(const AnalyticsLogger &) = delete;
    AnalyticsLogger &operator=(const AnalyticsLogger &) = delete;

    // Open the sink. `path` empty -> <paths.logs>/analytics.jsonl. Parent
    // directories are created. No-op when disabled. Safe to call once at
    // startup; re-initializing closes the previous sink.
    void initialize(const std::string &path = "", bool enabled = true);

    bool enabled() const { return enabled_; }
    const std::string &path() const { return path_; }

    // Stable, non-reversible user id. Prefers the client-provided anonymous id
    // (X-Client-Id header), otherwise derives from ip + user-agent, so every
    // request has an identifier even without a cooperating client. Always
    // prefixed "u-" and hashed: no raw identifier is ever written to disk.
    static std::string anonymousUserId(const std::string &client_id_header,
                                       const std::string &client_ip,
                                       const std::string &user_agent);

    // Extract utm_source/utm_medium/utm_campaign from the page URL (traffic
    // header) or, failing that, from the referer query string.
    static nlohmann::json parseTrafficSource(const std::string &traffic_header,
                                             const std::string &referer);

    // One HTTP request / tool call.
    void logRequest(const std::string &user_id,
                    const std::string &session_id,
                    const std::string &client_ip,
                    const std::string &user_agent,
                    const std::string &referer,
                    const nlohmann::json &traffic_source,
                    const std::string &method,
                    const std::string &path,
                    const std::string &endpoint,
                    const std::string &call_type,
                    int status_code,
                    double duration_ms);

    // One LLM invocation with token usage (the "LLM-cost" metric).
    void logLlmCall(const std::string &user_id,
                    const std::string &session_id,
                    const std::string &call_type,
                    const std::string &model,
                    int prompt_tokens,
                    int completion_tokens,
                    int total_tokens,
                    int status_code,
                    double duration_ms);

private:
    AnalyticsLogger() = default;
    ~AnalyticsLogger();

    void writeLine(const nlohmann::json &record);
    static std::string nowIso8601Utc();

    mutable std::mutex mutex_;
    std::ofstream out_;
    std::string path_;
    bool enabled_{false};
};
