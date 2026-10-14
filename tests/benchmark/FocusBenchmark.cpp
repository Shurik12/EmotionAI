#include "FocusBenchmark.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

FocusBenchmark::FocusBenchmark(const Config &config)
    : config_(config)
{
    client_ = std::make_unique<HttpClient>(config_.server_host, config_.server_port, config_.timeout_seconds);
    loadImage();
}

FocusBenchmark::~FocusBenchmark() = default;

void FocusBenchmark::loadImage()
{
    std::ifstream file(config_.test_image_path, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Cannot open Focus benchmark image: " + config_.test_image_path);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    test_image_data_ = buffer.str();
}

FocusBenchmark::RequestOutcome FocusBenchmark::createSession(std::string &session_id)
{
    auto response = client_->sendRequest("/api/focus/session", "POST", "");
    if (response.status_code != 201)
    {
        return {false, response.status_code};
    }
    try
    {
        session_id = response.json()["session_id"].get<std::string>();
        return {true, 201};
    }
    catch (const std::exception &)
    {
        return {false, response.status_code};
    }
}

FocusBenchmark::RequestOutcome FocusBenchmark::uploadFrame(const std::string &session_id)
{
    auto response = client_->uploadFile("/api/focus/session/" + session_id + "/frame",
                                        test_image_data_,
                                        "frame.jpg",
                                        "file");
    return {response.status_code == 202, response.status_code};
}

FocusBenchmark::RequestOutcome FocusBenchmark::pollSession(const std::string &session_id)
{
    auto response = client_->get("/api/focus/session/" + session_id);
    return {response.status_code == 200, response.status_code};
}

FocusBenchmark::RequestOutcome FocusBenchmark::closeSession(const std::string &session_id)
{
    auto response = client_->sendRequest("/api/focus/session/" + session_id + "/close", "POST", "");
    return {response.status_code == 200, response.status_code};
}

bool FocusBenchmark::runSession(int frame_count, const std::string &session_id)
{
    std::string sid = session_id;
    if (sid.empty())
    {
        auto created = createSession(sid);
        if (!created.ok)
        {
            return false;
        }
    }

    for (int i = 0; i < frame_count; ++i)
    {
        auto frame = uploadFrame(sid);
        if (!frame.ok)
        {
            closeSession(sid);
            return false;
        }
    }

    // Let the thread pool score a few frames so the poll returns data.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    auto poll = pollSession(sid);
    if (!poll.ok)
    {
        closeSession(sid);
        return false;
    }

    auto close = closeSession(sid);
    if (!close.ok)
    {
        return false;
    }

    return true;
}
