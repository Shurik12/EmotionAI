#include "AnalyticsLogger.h"

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <chrono>
#include <filesystem>
#include <sstream>

#include <config/Config.h>
#include <logging/Logger.h>
#include <fmt/format.h>

namespace fs = std::filesystem;

namespace
{
    // FNV-1a 64-bit. Dependency-free and stable across runs, which is all an
    // anonymized, per-installation user id needs.
    uint64_t fnv1a(const std::string &input)
    {
        uint64_t hash = 1469598103934665603ULL;
        for (unsigned char c : input)
        {
            hash ^= c;
            hash *= 1099511628211ULL;
        }
        return hash;
    }

    std::string hex64(uint64_t value)
    {
        char buf[17];
        std::snprintf(buf, sizeof(buf), "%016llx",
                      static_cast<unsigned long long>(value));
        return buf;
    }
}

AnalyticsLogger &AnalyticsLogger::instance()
{
    static AnalyticsLogger instance;
    return instance;
}

AnalyticsLogger::~AnalyticsLogger()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (out_.is_open())
        out_.flush();
}

void AnalyticsLogger::initialize(const std::string &path, bool enabled)
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (out_.is_open())
        out_.close();

    enabled_ = enabled;

    if (!enabled)
    {
        path_.clear();
        LOG_INFO("Analytics event log disabled");
        return;
    }

    path_ = path;
    if (path_.empty())
    {
        const std::string logs_dir = Config::instance().paths().logs;
        path_ = (fs::path(logs_dir) / "analytics.jsonl").string();
    }

    std::error_code ec;
    fs::create_directories(fs::path(path_).parent_path(), ec);

    out_.open(path_, std::ios::out | std::ios::app);
    if (!out_.is_open())
    {
        enabled_ = false;
        LOG_ERROR("Failed to open analytics event log: {}", path_);
        return;
    }

    LOG_INFO("Analytics event log: {}", path_);
}

std::string AnalyticsLogger::anonymousUserId(const std::string &client_id_header,
                                             const std::string &client_ip,
                                             const std::string &user_agent)
{
    // Prefer the client's own anonymous id; fall back to a stable fingerprint
    // of the peer so requests without the header are still attributable to a
    // (pseudonymous) user. Either way only the hash is ever logged.
    const std::string material = client_id_header.empty()
                                     ? client_ip + "|" + user_agent
                                     : client_id_header;
    return "u-" + hex64(fnv1a(material));
}

nlohmann::json AnalyticsLogger::parseTrafficSource(const std::string &traffic_header,
                                                   const std::string &referer)
{
    nlohmann::json source = nlohmann::json::object();
    source["referer"] = referer;

    std::string query;
    const size_t tpos = traffic_header.find('?');
    if (tpos != std::string::npos)
        query = traffic_header.substr(tpos + 1);
    else
    {
        const size_t rpos = referer.find('?');
        if (rpos != std::string::npos)
            query = referer.substr(rpos + 1);
    }

    auto param = [&query](const std::string &key) -> std::string
    {
        const std::string token = key + "=";
        size_t p = 0;
        while ((p = query.find(token, p)) != std::string::npos)
        {
            if (p == 0 || query[p - 1] == '&')
            {
                const size_t end = query.find('&', p);
                return query.substr(p + token.size(),
                                    end == std::string::npos ? std::string::npos
                                                             : end - (p + token.size()));
            }
            p += token.size();
        }
        return "";
    };

    source["utm_source"] = param("utm_source");
    source["utm_medium"] = param("utm_medium");
    source["utm_campaign"] = param("utm_campaign");
    return source;
}

std::string AnalyticsLogger::nowIso8601Utc()
{
    using namespace std::chrono;
    const auto now = system_clock::now();
    const std::time_t t = system_clock::to_time_t(now);
    const auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

    std::tm tm{};
    gmtime_r(&t, &tm);

    return fmt::format("{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}.{:03d}Z",
                       tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                       tm.tm_hour, tm.tm_min, tm.tm_sec,
                       static_cast<int>(ms.count()));
}

void AnalyticsLogger::writeLine(const nlohmann::json &record)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_ || !out_.is_open())
        return;

    out_ << record.dump() << '\n';
    out_.flush();
}

void AnalyticsLogger::logRequest(const std::string &user_id,
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
                                 double duration_ms)
{
    if (!enabled_)
        return;

    const bool ok = status_code < 400;

    nlohmann::json record = {
        {"ts", nowIso8601Utc()},
        {"event", "http_request"},
        {"source", "server"},
        {"user_id", user_id},
        {"session_id", session_id.empty() ? nlohmann::json(nullptr) : nlohmann::json(session_id)},
        {"client_ip", client_ip},
        {"traffic_source", traffic_source},
        {"user_agent", user_agent},
        {"method", method},
        {"path", path},
        {"endpoint", endpoint},
        {"call_type", call_type},
        {"status_code", status_code},
        {"result", ok ? "success" : "error"},
        {"error_code", ok ? nlohmann::json(nullptr) : nlohmann::json("http_" + std::to_string(status_code))},
        {"duration_ms", duration_ms},
    };

    writeLine(record);
}

void AnalyticsLogger::logLlmCall(const std::string &user_id,
                                 const std::string &session_id,
                                 const std::string &call_type,
                                 const std::string &model,
                                 int prompt_tokens,
                                 int completion_tokens,
                                 int total_tokens,
                                 int status_code,
                                 double duration_ms)
{
    if (!enabled_)
        return;

    const bool ok = status_code < 400;

    nlohmann::json record = {
        {"ts", nowIso8601Utc()},
        {"event", "llm_call"},
        {"source", "server"},
        {"user_id", user_id},
        {"session_id", session_id.empty() ? nlohmann::json(nullptr) : nlohmann::json(session_id)},
        {"call_type", call_type},
        {"model", model},
        {"prompt_tokens", prompt_tokens},
        {"completion_tokens", completion_tokens},
        {"total_tokens", total_tokens},
        {"status_code", status_code},
        {"result", ok ? "success" : "error"},
        {"error_code", ok ? nlohmann::json(nullptr) : nlohmann::json("http_" + std::to_string(status_code))},
        {"duration_ms", duration_ms},
    };

    writeLine(record);
}
