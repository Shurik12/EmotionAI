#include <gtest/gtest.h>

#include <server/RateLimiter.h>

// The limiter is a token bucket: `burst` tokens available immediately, refilled
// at rate_limit_per_minute / 60 tokens per second.

TEST(RateLimiterTest, AllowsBurstThenRejects)
{
    RateLimiter limiter(60, 3); // 1 token/s, capacity 3

    EXPECT_DOUBLE_EQ(limiter.allowAt("ip", 1000.0), 0.0);
    EXPECT_DOUBLE_EQ(limiter.allowAt("ip", 1000.0), 0.0);
    EXPECT_DOUBLE_EQ(limiter.allowAt("ip", 1000.0), 0.0);

    const double retry_after = limiter.allowAt("ip", 1000.0);
    EXPECT_GT(retry_after, 0.0);
    EXPECT_NEAR(retry_after, 1.0, 0.01);
}

TEST(RateLimiterTest, RefillsOverTime)
{
    RateLimiter limiter(60, 2); // 1 token/s

    EXPECT_DOUBLE_EQ(limiter.allowAt("ip", 100.0), 0.0);
    EXPECT_DOUBLE_EQ(limiter.allowAt("ip", 100.0), 0.0);
    EXPECT_GT(limiter.allowAt("ip", 100.0), 0.0);

    // One second later exactly one token is available...
    EXPECT_DOUBLE_EQ(limiter.allowAt("ip", 101.0), 0.0);
    EXPECT_GT(limiter.allowAt("ip", 101.0), 0.0);
}

TEST(RateLimiterTest, TracksKeysIndependently)
{
    RateLimiter limiter(60, 1);

    EXPECT_DOUBLE_EQ(limiter.allowAt("a", 50.0), 0.0);
    EXPECT_DOUBLE_EQ(limiter.allowAt("b", 50.0), 0.0);

    EXPECT_GT(limiter.allowAt("a", 50.0), 0.0);
    EXPECT_GT(limiter.allowAt("b", 50.0), 0.0);
}

TEST(RateLimiterTest, DisabledWhenRateIsZero)
{
    RateLimiter limiter(0, 5);
    for (int i = 0; i < 1000; ++i)
        EXPECT_DOUBLE_EQ(limiter.allowAt("ip", 10.0), 0.0);
}

TEST(RateLimiterTest, BoundsTrackedKeys)
{
    RateLimiter limiter(60, 1, /*max_keys=*/2);

    EXPECT_DOUBLE_EQ(limiter.allowAt("a", 10.0), 0.0);
    EXPECT_DOUBLE_EQ(limiter.allowAt("b", 10.0), 0.0);
    EXPECT_DOUBLE_EQ(limiter.allowAt("c", 10.0), 0.0);

    EXPECT_LE(limiter.trackedKeys(), 2u);
}

TEST(RateLimiterTest, RetryAfterShrinksAsTimePasses)
{
    RateLimiter limiter(60, 1); // 1 token/s

    EXPECT_DOUBLE_EQ(limiter.allowAt("ip", 200.0), 0.0);
    const double first = limiter.allowAt("ip", 200.0);
    const double later = limiter.allowAt("ip", 200.5);

    EXPECT_NEAR(first, 1.0, 0.01);
    EXPECT_NEAR(later, 0.5, 0.01);
    EXPECT_LT(later, first);
}
