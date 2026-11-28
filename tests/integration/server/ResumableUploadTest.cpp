#include "TestServerHarness.h"

// Resumable upload state machine (M4): init -> chunk -> complete, offset
// resync, size guard, abort.
class ResumableUploadTest : public ::testing::Test
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
        client().clearHeaders();
        client().setHeader("X-API-Key", "test-app-token");
    }

    static HttpClient &client() { return *srv.client; }

    static std::string initUpload(const std::string &filename, long long size)
    {
        auto response = client().post(
            "/api/upload/init",
            nlohmann::json{{"filename", filename}, {"size", size}, {"mode", "standard"}});
        EXPECT_EQ(response.status_code, 201) << response.body;
        return response.json().value("upload_id", "");
    }
};

TestServer ResumableUploadTest::srv;

TEST_F(ResumableUploadTest, InitRequiresAppKey)
{
    client().clearHeaders();
    auto response = client().post("/api/upload/init",
                                  nlohmann::json{{"filename", "a.png"}, {"size", 5}});
    EXPECT_EQ(response.status_code, 401);
}

TEST_F(ResumableUploadTest, InitRejectsNonIntegerSize)
{
    auto response = client().post("/api/upload/init",
                                  nlohmann::json{{"filename", "a.png"}, {"size", "5"}});
    EXPECT_EQ(response.status_code, 400);
}

TEST_F(ResumableUploadTest, OffsetConflictThenComplete)
{
    const std::string id = initUpload("tiny.png", 5);
    ASSERT_FALSE(id.empty());
    client().setHeader("X-Upload-Id", id);

    client().setHeader("X-Chunk-Offset", "2");
    auto bad = client().post("/api/upload/chunk", std::string("hello"),
                             "application/octet-stream");
    ASSERT_EQ(bad.status_code, 409) << bad.body;
    EXPECT_EQ(bad.json().value("received", -1), 0);

    client().setHeader("X-Chunk-Offset", "0");
    auto ok = client().post("/api/upload/chunk", std::string("hello"),
                            "application/octet-stream");
    ASSERT_EQ(ok.status_code, 200) << ok.body;
    EXPECT_EQ(ok.json().value("received", -1), 5);

    auto complete = client().post("/api/upload/complete",
                                  nlohmann::json{{"upload_id", id}});
    ASSERT_EQ(complete.status_code, 202) << complete.body;
    const std::string task_id = complete.json().value("task_id", "");
    EXPECT_FALSE(task_id.empty());

    // The handler writes the initial task status synchronously, so progress is
    // available immediately.
    auto progress = client().get("/api/progress/" + task_id);
    EXPECT_EQ(progress.status_code, 200) << progress.body;
}

TEST_F(ResumableUploadTest, ChunkBeyondDeclaredSizeIsRejected)
{
    const std::string id = initUpload("tiny.png", 3);
    ASSERT_FALSE(id.empty());
    client().setHeader("X-Upload-Id", id);
    client().setHeader("X-Chunk-Offset", "0");

    auto response = client().post("/api/upload/chunk", std::string("hello"),
                                  "application/octet-stream");
    EXPECT_EQ(response.status_code, 413) << response.body;
}

TEST_F(ResumableUploadTest, AbortDropsTheSession)
{
    const std::string id = initUpload("tiny.png", 5);
    ASSERT_FALSE(id.empty());

    auto abort = client().post("/api/upload/abort", nlohmann::json{{"upload_id", id}});
    EXPECT_EQ(abort.status_code, 200) << abort.body;

    client().setHeader("X-Upload-Id", id);
    client().setHeader("X-Chunk-Offset", "0");
    auto response = client().post("/api/upload/chunk", std::string("hello"),
                                  "application/octet-stream");
    EXPECT_EQ(response.status_code, 404) << response.body;
}
