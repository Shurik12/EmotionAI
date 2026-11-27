#pragma once

#include <chrono>
#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>

// Per-client token-bucket rate limiter.
//
// The epoll server is single-threaded, but a mutex is kept anyway so the
// limiter stays correct if it is ever called from a worker thread. Keys are
// opaque strings (a client IP, optionally taken from a proxy header).
//
// Allowance model:
//   * `capacity` tokens available at once (burst),
//   * refilled continuously at `requests_per_minute / 60` tokens per second.
// A rejected request gets the number of seconds until one token is available,
// which the server returns as `Retry-After`.
class RateLimiter
{
public:
    RateLimiter(int requests_per_minute, int burst, std::size_t max_keys = 10000);

    // Returns 0.0 when the request is allowed, otherwise the seconds to wait
    // before retrying (always > 0).
    double allow(const std::string &key);

    // Deterministic variant for tests: `now_seconds` is a monotonic timestamp
    // in seconds. `allow()` forwards to this with steady_clock time.
    double allowAt(const std::string &key, double now_seconds);

    void reset();
    std::size_t trackedKeys() const;

private:
    struct Bucket
    {
        double tokens{0.0};
        double last_seen{0.0};
    };

    static double nowSeconds();
    void evictIfFullLocked();
    void pruneIdleLocked(double now_seconds);

    mutable std::mutex mutex_;
    std::unordered_map<std::string, Bucket> buckets_;
    double rate_per_second_;
    double capacity_;
    std::size_t max_keys_;
    std::size_t requests_since_prune_{0};
};
