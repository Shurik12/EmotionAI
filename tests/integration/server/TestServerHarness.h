#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include <client/Client.h>
#include <config/Config.h>
#include <logging/Logger.h>
#include <server/Server.h>

// Starts a real Server on the port/cwd configured in `config_path` and returns
// an HttpClient bound to it. `client == nullptr` means the prerequisites
// (DragonflyDB) were unavailable; suites turn that into a GTEST_SKIP.
struct TestServer
{
    std::unique_ptr<Server> server;
    std::thread thread;
    std::unique_ptr<HttpClient> client;
    std::string error;

    // Move-only: std::thread. Intentionally no destructor so the implicit move
    // constructor exists; suites must call stopTestServer() from
    // TearDownTestSuite.
    TestServer() = default;
    TestServer(TestServer &&) = default;
    TestServer &operator=(TestServer &&) = default;
    TestServer(const TestServer &) = delete;
    TestServer &operator=(const TestServer &) = delete;
};

inline TestServer startTestServer(const std::string &config_path)
{
    TestServer result;
    auto &config = Config::instance();
    if (!config.loadFromFile(config_path))
    {
        result.error = "config not found: " + config_path;
        return result;
    }

    std::error_code ec;
    std::filesystem::create_directories(config.paths().uploads, ec);
    std::filesystem::create_directories(config.paths().results, ec);
    std::filesystem::create_directories(config.paths().frontend, ec);
    std::filesystem::create_directories(config.paths().logs, ec);

    Logger::instance().initialize(config.paths().logs, "EmotionAI-Tests", spdlog::level::err);

    std::ofstream index_file(config.paths().frontend + "/index.html");
    index_file << "<html><body>Test Server</body></html>";
    index_file.close();

    try
    {
        result.server = std::make_unique<Server>();
        result.server->initialize();
    }
    catch (const std::exception &e)
    {
        result.error = e.what();
        result.server.reset();
        return result;
    }

    result.thread = std::thread([server = result.server.get()]
                                { server->start(); });
    std::this_thread::sleep_for(std::chrono::seconds(1));
    result.client = std::make_unique<HttpClient>(config.server().host, config.server().port);
    return result;
}

inline void stopTestServer(TestServer &server)
{
    if (server.server)
        server.server->stop();
    if (server.thread.joinable())
        server.thread.join();
    server.server.reset();
    server.client.reset();

    auto &config = Config::instance();
    if (!config.isLoaded())
        return;
    std::error_code ec;
    std::filesystem::remove_all(config.paths().uploads, ec);
    std::filesystem::remove_all(config.paths().results, ec);
    std::filesystem::remove_all(config.paths().frontend, ec);
    std::filesystem::remove_all(config.paths().logs, ec);
}

// Case-insensitive response-header lookup (httplib preserves the sent case).
inline std::string responseHeader(const HttpClient::Response &response, const std::string &name)
{
    auto lower = [](std::string value)
    {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });
        return value;
    };
    const std::string wanted = lower(name);
    for (const auto &[key, value] : response.headers)
    {
        if (lower(key) == wanted)
            return value;
    }
    return "";
}
