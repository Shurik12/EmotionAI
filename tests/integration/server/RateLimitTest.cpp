#include "TestServerHarness.h"

// Per-client rate limiting (M4): 429 + Retry-After after the burst, and the
// forwarded client address is honored only from a trusted (loopback) peer.
class RateLimitTest : public ::testing::Test
{
protected:
    static TestServer srv;

    static void SetUpTestSuite()
    {
        srv = startTestServer("tests/configs/rate_limit_config.yaml");
    }

    static void TearDownTestSuite()
    {
        stopTestServer(srv);
    }

    void SetUp() override
    {
        if (!srv.client)
            GTEST_SKIP() << "Server unavailable (need DragonflyDB): " << srv.error;
    }

    static HttpClient &client() { return *srv.client; }
};

TestServer RateLimitTest::srv;

TEST_F(RateLimitTest, ThrottlesPerClientAndHonorsTrustedProxyHeader)
{
    client().clearHeaders();
    client().setHeader("X-Real-IP", "3.3.3.3");

    EXPECT_EQ(client().get("/api/health").status_code, 200);

    auto throttled = client().get("/api/health");
    ASSERT_EQ(throttled.status_code, 429) << throttled.body;
    EXPECT_FALSE(responseHeader(throttled, "Retry-After").empty());

    // A different trusted client gets its own bucket.
    client().setHeader("X-Real-IP", "4.4.4.4");
    EXPECT_EQ(client().get("/api/health").status_code, 200);
}
