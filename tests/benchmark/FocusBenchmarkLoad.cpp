#include "FocusBenchmark.h"
#include <atomic>
#include <vector>

namespace
{
    // Run a single camera-like client that loops through the full Focus
    // session flow (create -> upload frames -> poll -> close) until `stop`.
    // Updates the shared counters.
    void runClientLoop(FocusBenchmark &benchmark,
                       int frames_per_session,
                       std::atomic<int64_t> &frames,
                       std::atomic<int64_t> &polls,
                       std::atomic<int64_t> &creates,
                       std::atomic<int64_t> &closes,
                       std::atomic<bool> &stop)
    {
        std::string sid;
        bool session_open = false;

        while (!stop.load(std::memory_order_relaxed))
        {
            if (!session_open)
            {
                if (!benchmark.createSession(sid))
                {
                    continue;
                }
                creates.fetch_add(1);
                session_open = true;
            }

            bool batch_ok = true;
            int local_frames = 0;
            for (int j = 0; j < frames_per_session; ++j)
            {
                auto result = benchmark.uploadFrame(sid);
                if (!result.ok)
                {
                    batch_ok = false;
                    break;
                }
                local_frames++;
            }
            if (local_frames > 0)
            {
                frames.fetch_add(local_frames);
            }

            if (!batch_ok)
            {
                benchmark.closeSession(sid);
                closes.fetch_add(1);
                session_open = false;
                continue;
            }

            auto poll = benchmark.pollSession(sid);
            if (poll.ok)
            {
                polls.fetch_add(1);
            }

            // Periodically close and reopen to keep the session map bounded,
            // mirroring a real user leaving and returning to the page.
            if (frames.load() % (static_cast<int64_t>(frames_per_session) * 10) == 0)
            {
                benchmark.closeSession(sid);
                closes.fetch_add(1);
                session_open = false;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Single camera client: full Focus session flow at ~1 fps.
// ---------------------------------------------------------------------------
void BM_FocusSingleSession(benchmark::State &state)
{
    FocusBenchmark::Config config;
    FocusBenchmark benchmark(config);

    const int duration_ms = 3000;

    for (auto _ : state)
    {
        std::atomic<int64_t> frames{0};
        std::atomic<int64_t> polls{0};
        std::atomic<int64_t> creates{0};
        std::atomic<int64_t> closes{0};
        std::atomic<bool> stop{false};

        std::thread worker([&]()
                           {
                               runClientLoop(benchmark, config.frames_per_session,
                                             frames, polls, creates, closes, stop);
                           });

        auto start = std::chrono::steady_clock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
        stop.store(true);
        worker.join();
        auto end = std::chrono::steady_clock::now();

        const double elapsed_s = std::chrono::duration<double>(end - start).count();
        state.SetIterationTime(elapsed_s);

        state.counters["ConcurrentClients"] = 1;
        state.counters["DurationMs"] = duration_ms;
        state.counters["Frames"] = static_cast<double>(frames.load());
        state.counters["RPS_frames"] = benchmark::Counter(
            static_cast<double>(frames.load()) / elapsed_s,
            benchmark::Counter::kIsRate, benchmark::Counter::OneK::kIs1000);
        state.counters["Polls"] = static_cast<double>(polls.load());
        state.counters["Sessions"] = static_cast<double>(creates.load());
    }
}

// ---------------------------------------------------------------------------
// Sustained load: N concurrent camera clients for a fixed duration.
// Each client runs the full session flow (create -> upload frames -> poll ->
// close) in a loop. RPS is measured on frame uploads, the hot path the
// camera actually hits.
// ---------------------------------------------------------------------------
void BM_FocusConcurrentSessions(benchmark::State &state)
{
    const int concurrency = state.range(0);
    const int duration_ms = state.range(1);

    FocusBenchmark::Config config;
    FocusBenchmark benchmark(config);

    // Warm up: make sure the server is reachable.
    std::string warmup_id;
    if (!benchmark.createSession(warmup_id))
    {
        state.SkipWithError("Server not reachable or /api/focus/session failed");
        return;
    }
    benchmark.closeSession(warmup_id);

    for (auto _ : state)
    {
        std::atomic<int64_t> frames{0};
        std::atomic<int64_t> polls{0};
        std::atomic<int64_t> creates{0};
        std::atomic<int64_t> closes{0};
        std::atomic<bool> stop{false};

        std::vector<std::thread> threads;
        threads.reserve(concurrency);

        for (int i = 0; i < concurrency; ++i)
        {
            threads.emplace_back([&]()
                                 {
                                     runClientLoop(benchmark, config.frames_per_session,
                                                   frames, polls, creates, closes, stop);
                                 });
        }

        auto start = std::chrono::steady_clock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
        stop.store(true);
        for (auto &t : threads)
        {
            t.join();
        }
        auto end = std::chrono::steady_clock::now();

        const double elapsed_s = std::chrono::duration<double>(end - start).count();
        state.SetIterationTime(elapsed_s);

        state.counters["ConcurrentClients"] = concurrency;
        state.counters["DurationMs"] = duration_ms;
        state.counters["Frames"] = static_cast<double>(frames.load());
        state.counters["RPS_frames"] = benchmark::Counter(
            static_cast<double>(frames.load()) / elapsed_s,
            benchmark::Counter::kIsRate, benchmark::Counter::OneK::kIs1000);
        state.counters["Polls"] = static_cast<double>(polls.load());
        state.counters["Sessions"] = static_cast<double>(creates.load());
    }
}

// ---------------------------------------------------------------------------
// Polling-only load: the Focus page polls GET /api/focus/session/<id>
// periodically while the camera is running. Measure max poll RPS.
// ---------------------------------------------------------------------------
void BM_FocusPolling(benchmark::State &state)
{
    const int concurrency = state.range(0);

    FocusBenchmark::Config config;
    FocusBenchmark benchmark(config);

    std::string session_id;
    if (!benchmark.createSession(session_id))
    {
        state.SkipWithError("Failed to create session for polling benchmark");
        return;
    }
    // Seed a few frames so the response is non-trivial.
    for (int i = 0; i < config.frames_per_session; ++i)
    {
        benchmark.uploadFrame(session_id);
    }

    const int duration_ms = 1000;

    for (auto _ : state)
    {
        std::atomic<int64_t> ok_polls{0};
        std::atomic<bool> stop{false};
        std::vector<std::thread> threads;
        threads.reserve(concurrency);

        for (int i = 0; i < concurrency; ++i)
        {
            threads.emplace_back([&]()
                                 {
                                     int local = 0;
                                     while (!stop.load(std::memory_order_relaxed))
                                     {
                                         auto result = benchmark.pollSession(session_id);
                                         if (result.ok)
                                             local++;
                                     }
                                     ok_polls.fetch_add(local);
                                 });
        }

        auto start = std::chrono::steady_clock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
        stop.store(true);
        for (auto &t : threads)
        {
            t.join();
        }
        auto end = std::chrono::steady_clock::now();

        const double elapsed_s = std::chrono::duration<double>(end - start).count();
        state.SetIterationTime(elapsed_s);

        state.counters["ConcurrentClients"] = concurrency;
        state.counters["Polls"] = static_cast<double>(ok_polls.load());
        state.counters["RPS_polls"] = benchmark::Counter(
            static_cast<double>(ok_polls.load()) / elapsed_s,
            benchmark::Counter::kIsRate, benchmark::Counter::OneK::kIs1000);
    }

    benchmark.closeSession(session_id);
}

// ---------------------------------------------------------------------------
// Frame upload throughput: create a session once, then blast frames through
// N concurrent clients. This isolates the hottest Focus endpoint.
// ---------------------------------------------------------------------------
void BM_FocusFrameUpload(benchmark::State &state)
{
    const int concurrency = state.range(0);

    FocusBenchmark::Config config;
    FocusBenchmark benchmark(config);

    std::string session_id;
    if (!benchmark.createSession(session_id))
    {
        state.SkipWithError("Failed to create session for frame benchmark");
        return;
    }

    const int duration_ms = 1000;

    for (auto _ : state)
    {
        std::atomic<int64_t> total_frames{0};
        std::atomic<bool> stop{false};
        std::vector<std::thread> threads;

        for (int i = 0; i < concurrency; ++i)
        {
            threads.emplace_back([&]()
                                 {
                                     int local_frames = 0;
                                     while (!stop.load(std::memory_order_relaxed))
                                     {
                                         auto result = benchmark.uploadFrame(session_id);
                                         if (!result.ok)
                                         {
                                             return;
                                         }
                                         local_frames++;
                                     }
                                     total_frames.fetch_add(local_frames);
                                 });
        }

        auto start = std::chrono::steady_clock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
        stop.store(true);
        for (auto &t : threads)
        {
            t.join();
        }
        auto end = std::chrono::steady_clock::now();

        const double elapsed_s = std::chrono::duration<double>(end - start).count();
        state.SetIterationTime(elapsed_s);

        state.counters["ConcurrentClients"] = concurrency;
        state.counters["Frames"] = static_cast<double>(total_frames.load());
        state.counters["RPS_frames"] = benchmark::Counter(
            static_cast<double>(total_frames.load()) / elapsed_s,
            benchmark::Counter::kIsRate, benchmark::Counter::OneK::kIs1000);
    }

    benchmark.closeSession(session_id);
}
