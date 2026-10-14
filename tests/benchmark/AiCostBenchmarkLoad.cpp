#include "AiCostBenchmark.h"

#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

namespace
{
    // Fixed set of sample tasks cycled through for the measurement.
    const std::vector<std::string> &tasks()
    {
        return AiCostBenchmark::sampleTasks();
    }
}

// ---------------------------------------------------------------------------
// BM_AiCostBreakdown — one call per task, measures tokens per call.
//
// Runs each sample task once (Args(1) means "1 call"), so the benchmark
// performs exactly tasks().size() LLM calls in total.  User counters report
// average and total token counts.
// ---------------------------------------------------------------------------
void BM_AiCostBreakdown(benchmark::State &state)
{
    AiCostBenchmark::Config config;
    // Read host/port from environment so `make benchmark_ai_cost` can
    // override them without rebuilding.
    if (const char *h = std::getenv("BENCHMARK_HOST"))
        config.server_host = h;
    if (const char *p = std::getenv("BENCHMARK_PORT"))
        config.server_port = std::stoi(p);

    try
    {
        AiCostBenchmark benchmark(config);
        const auto &task_list = tasks();
        size_t task_idx = 0;

        int64_t total_prompt = 0;
        int64_t total_completion = 0;
        int64_t total_tokens = 0;
        int64_t calls_ok = 0;
        int64_t calls_total = 0;
        std::vector<double> latencies_ms;

        for (auto _ : state)
        {
            const std::string &task_text = task_list[task_idx % task_list.size()];
            const std::string duration = "25 мин";
            const std::string lang = (task_idx % 2 == 0) ? "ru" : "en";

            auto t0 = std::chrono::steady_clock::now();
            auto result = benchmark.callBreakdown(task_text, duration, lang);
            auto t1 = std::chrono::steady_clock::now();

            latencies_ms.push_back(
                std::chrono::duration<double, std::milli>(t1 - t0).count());

            calls_total++;
            if (result.ok)
            {
                calls_ok++;
                total_prompt += result.prompt_tokens;
                total_completion += result.completion_tokens;
                total_tokens += result.total_tokens;
            }

            task_idx++;
        }

        // If every call failed the AI is likely not configured.
        if (calls_ok == 0)
        {
            std::cerr << "BM_AiCostBreakdown: no successful AI calls — "
                      << "is ai.enabled=true and api_key set in config.yaml?\n";
            state.SkipWithError("AI not configured or all calls failed");
            return;
        }

        double avg_latency =
            std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) /
            latencies_ms.size();

        state.SetIterationTime(static_cast<double>(calls_total));
        state.counters["CallsOk"] = calls_ok;
        state.counters["CallsTotal"] = calls_total;
        state.counters["AvgPromptTokens"] = static_cast<double>(total_prompt) / calls_ok;
        state.counters["AvgCompletionTokens"] = static_cast<double>(total_completion) / calls_ok;
        state.counters["AvgTotalTokens"] = static_cast<double>(total_tokens) / calls_ok;
        state.counters["TotalTokens"] = static_cast<double>(total_tokens);
        state.counters["AvgLatencyMs"] = avg_latency;
    }
    catch (const std::exception &e)
    {
        std::cerr << "BM_AiCostBreakdown: " << e.what() << "\n";
        state.SkipWithError(e.what());
    }
}

BENCHMARK(BM_AiCostBreakdown)
    ->Unit(benchmark::kMillisecond)
    ->UseManualTime()
    ->Iterations(1);
