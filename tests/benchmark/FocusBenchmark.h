#pragma once

#include <benchmark/benchmark.h>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <client/Client.h>

// Benchmark for the Razuma Focus camera session API (EMO-17).
//
// The Focus API is a lightweight in-memory session flow:
//   POST /api/focus/session                     -> create
//   POST /api/focus/session/<id>/frame          -> upload a camera frame (multipart)
//   GET  /api/focus/session/<id>                -> poll accumulated observations
//   POST /api/focus/session/<id>/close          -> close and drop the session
//
// Frame scoring (face detection + emotion model) runs on the server thread
// pool, so the HTTP endpoint returns 202 immediately after queueing. The
// benchmark measures the request/response path the camera actually hits.
class FocusBenchmark
{
public:
    struct Config
    {
        std::string server_host = "localhost";
        int server_port = 8080;
        std::string test_image_path = "assets/test_face.jpg";
        int timeout_seconds = 30;
        // Frames to accumulate per session before reading it back, so the GET
        // response has a realistic (non-empty) sample list.
        int frames_per_session = 10;
        // Seconds to hold the session open before closing it.
        int session_hold_seconds = 2;
    };

    FocusBenchmark(const Config &config);
    ~FocusBenchmark();

    struct RequestOutcome
    {
        bool ok = false;
        int status_code = 0;
        explicit operator bool() const { return ok; }
    };

    // One full session: create, upload n frames, read back, close.
    // Returns true when every step succeeded.
    bool runSession(int frame_count, const std::string &session_id);

    // Single-shot helpers (used by polling benchmarks).
    RequestOutcome createSession(std::string &session_id);
    RequestOutcome uploadFrame(const std::string &session_id);
    RequestOutcome pollSession(const std::string &session_id);
    RequestOutcome closeSession(const std::string &session_id);

    const Config &getConfig() const { return config_; }

private:
    Config config_;
    std::unique_ptr<HttpClient> client_;
    std::string test_image_data_;

    void loadImage();
};
