#pragma once

#include <string>
#include <vector>

#include <benchmark/benchmark.h>
#include <nlohmann/json.hpp>
#include <client/Client.h>

// Measures LLM token cost of the Focus task-breakdown endpoint.
//
// For each "iteration" the benchmark sends a realistic task text to
// POST /api/focus/breakdown and records the token usage returned in the
// response.  The result is a set of Google Benchmark user counters
// (AvgTokensPerCall, TotalTokens, etc.) that land in the JSON output
// file written by --benchmark_out.
//
// Prerequisites:
//   - A running EmotionAI server with ai.enabled=true and a valid api_key.
//   - If the AI is not configured the benchmark is reported as skipped.
class AiCostBenchmark
{
public:
    struct Config
    {
        std::string server_host = "localhost";
        int server_port = 8080;
        int timeout_seconds = 60;
    };

    AiCostBenchmark(const Config &config);
    ~AiCostBenchmark();

    // Sample task texts that mimic real user input.
    static const std::vector<std::string> &sampleTasks();

    struct CostResult
    {
        bool ok = false;
        int prompt_tokens = 0;
        int completion_tokens = 0;
        int total_tokens = 0;
        int status_code = 0;
    };

    CostResult callBreakdown(const std::string &task_text,
                             const std::string &duration,
                             const std::string &lang);

    const Config &getConfig() const { return config_; }

private:
    Config config_;
    std::unique_ptr<HttpClient> client_;
};
