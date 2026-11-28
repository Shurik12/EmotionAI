#include <algorithm>
#include <cctype>

#include "TestServerHarness.h"

namespace
{
constexpr const char *kAppKey = "test-app-token";
}

// Device identity + baseline isolation (M4). Shares one server across the suite
// because it loads models and binds a port.
class DeviceAuthTest : public ::testing::Test
{
protected:
    static TestServer srv;

    static void SetUpTestSuite()
    {
        srv = startTestServer("tests/configs/device_auth_config.yaml");
    }

    static void TearDownTestSuite()
    {
        stopTestServer(srv);
    }

    void SetUp() override
    {
        if (!srv.client)
            GTEST_SKIP() << "Server unavailable (need DragonflyDB): " << srv.error;
        useAppKey();
    }

    static HttpClient &client() { return *srv.client; }

    static void useAppKey()
    {
        client().clearHeaders();
        client().setHeader("X-API-Key", kAppKey);
    }

    static void useDevice(const std::string &token)
    {
        useAppKey();
        client().setBearerToken(token);
    }

    static std::string bootstrap()
    {
        auto response = client().post("/api/auth/device", std::string("{}"));
        EXPECT_EQ(response.status_code, 201) << response.body;
        return response.json().value("device_token", "");
    }
};

TestServer DeviceAuthTest::srv;

TEST_F(DeviceAuthTest, HealthAndMetricsAreExempt)
{
    client().clearHeaders();
    EXPECT_EQ(client().get("/api/health").status_code, 200);
    EXPECT_EQ(client().get("/api/metrics").status_code, 200);
}

TEST_F(DeviceAuthTest, SimilarPathIsNotExempt)
{
    client().clearHeaders();
    EXPECT_EQ(client().get("/api/healthz").status_code, 401);
}

TEST_F(DeviceAuthTest, BootstrapRequiresAppKey)
{
    client().clearHeaders();
    auto response = client().post("/api/auth/device", std::string("{}"));
    EXPECT_EQ(response.status_code, 401);
}

TEST_F(DeviceAuthTest, BootstrapIssuesIdentity)
{
    auto response = client().post("/api/auth/device", std::string("{}"));
    ASSERT_EQ(response.status_code, 201) << response.body;

    auto data = response.json();
    EXPECT_FALSE(data.value("device_token", "").empty());
    EXPECT_FALSE(data.value("user_id", "").empty());
    EXPECT_EQ(data.value("expires_in", 0), 15552000);
}

TEST_F(DeviceAuthTest, BaselineRequiresDeviceToken)
{
    auto response = client().post("/api/burnout/baseline",
                                  nlohmann::json{{"baseline", {{"x", 1}}}});
    EXPECT_EQ(response.status_code, 401);
    EXPECT_EQ(response.json().value("error", ""), "device_token_required");
}

TEST_F(DeviceAuthTest, BaselineIsScopedToTheDevice)
{
    const std::string token = bootstrap();
    ASSERT_FALSE(token.empty());
    useDevice(token);

    auto write = client().post(
        "/api/burnout/baseline",
        nlohmann::json{{"user_id", "attacker"},
                       {"baseline", {{"happy", 0.42}, {"note", "device"}}}});
    ASSERT_EQ(write.status_code, 200) << write.body;

    // A client-supplied user_id must not redirect the read.
    auto read = client().get("/api/burnout/baseline?user_id=attacker");
    ASSERT_EQ(read.status_code, 200) << read.body;
    auto data = read.json();
    EXPECT_NE(data.value("user_id", ""), "attacker");
    EXPECT_EQ(data["baseline"].value("note", ""), "device");

    // A second device cannot see the first device's baseline.
    useAppKey();
    const std::string other = bootstrap();
    ASSERT_FALSE(other.empty());
    useDevice(other);
    EXPECT_EQ(client().get("/api/burnout/baseline").status_code, 404);
}

TEST_F(DeviceAuthTest, InvalidTokenIsRejected)
{
    useDevice("deadbeef");
    auto response = client().get("/api/burnout/baseline");
    EXPECT_EQ(response.status_code, 401);
    EXPECT_EQ(response.json().value("error", ""), "invalid_device_token");
}

TEST_F(DeviceAuthTest, RevokeDeletesIdentityAndBaseline)
{
    const std::string token = bootstrap();
    ASSERT_FALSE(token.empty());
    useDevice(token);
    ASSERT_EQ(client().post("/api/burnout/baseline",
                            nlohmann::json{{"baseline", {{"x", 1}}}})
                  .status_code,
              200);

    auto revoke = client().post("/api/auth/revoke", std::string("{}"));
    ASSERT_EQ(revoke.status_code, 200) << revoke.body;
    EXPECT_TRUE(revoke.json().value("revoked", false));

    auto after = client().get("/api/burnout/baseline");
    EXPECT_EQ(after.status_code, 401);
    EXPECT_EQ(after.json().value("error", ""), "invalid_device_token");
}
