#include "RateLimiter.h"

#include <algorithm>

namespace
{
    // Forget clients that have been idle this long. Backstop for traffic that
    // never reaches `max_keys` but would otherwise grow without bound.
    constexpr double kIdleEvictionSeconds = 600.0;
    // Prune at most once per this many requests.
    constexpr std::size_t kPruneInterval = 512;
}

RateLimiter::RateLimiter(int requests_per_minute, int burst, std::size_t max_keys)
    : rate_per_second_(requests_per_minute > 0 ? requests_per_minute / 60.0 : 0.0),
      capacity_(burst > 0 ? static_cast<double>(burst) : 1.0),
      max_keys_(max_keys > 0 ? max_keys : 1)
{
}

double RateLimiter::nowSeconds()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

double RateLimiter::allow(const std::string &key)
{
    return allowAt(key, nowSeconds());
}

double RateLimiter::allowAt(const std::string &key, double now_seconds)
{
    // A non-positive rate means the limiter is disabled.
    if (rate_per_second_ <= 0.0)
        return 0.0;

    std::lock_guard<std::mutex> lock(mutex_);

    if (++requests_since_prune_ >= kPruneInterval)
    {
        requests_since_prune_ = 0;
        pruneIdleLocked(now_seconds);
    }

    auto it = buckets_.find(key);
    if (it == buckets_.end())
    {
        if (buckets_.size() >= max_keys_)
            evictIfFullLocked();
        Bucket bucket;
        bucket.tokens = capacity_ - 1.0;
        bucket.last_seen = now_seconds;
        buckets_.emplace(key, bucket);
        return 0.0;
    }

    Bucket &bucket = it->second;

    // A clock that went backwards (or an idle gap) must not add tokens.
    const double elapsed = std::max(0.0, now_seconds - bucket.last_seen);
    bucket.tokens = std::min(capacity_, bucket.tokens + elapsed * rate_per_second_);
    bucket.last_seen = now_seconds;

    if (bucket.tokens >= 1.0)
    {
        bucket.tokens -= 1.0;
        return 0.0;
    }

    return (1.0 - bucket.tokens) / rate_per_second_;
}

void RateLimiter::evictIfFullLocked()
{
    // Bounded memory: drop the least-recently-seen bucket. Called only when the
    // map is at capacity, so a full scan is rare.
    auto oldest = buckets_.begin();
    for (auto it = buckets_.begin(); it != buckets_.end(); ++it)
    {
        if (it->second.last_seen < oldest->second.last_seen)
            oldest = it;
    }
    buckets_.erase(oldest);
}

void RateLimiter::pruneIdleLocked(double now_seconds)
{
    for (auto it = buckets_.begin(); it != buckets_.end();)
    {
        if (now_seconds - it->second.last_seen > kIdleEvictionSeconds)
            it = buckets_.erase(it);
        else
            ++it;
    }
}

void RateLimiter::reset()
{
    std::lock_guard<std::mutex> lock(mutex_);
    buckets_.clear();
    requests_since_prune_ = 0;
}

std::size_t RateLimiter::trackedKeys() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return buckets_.size();
}
