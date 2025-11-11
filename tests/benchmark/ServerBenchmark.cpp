// tests/benchmark/ServerBenchmark.cpp
#include <benchmark/benchmark.h>
#include <atomic>
#include <thread>
#include <vector>
#include <memory>
#include <chrono>
#include <random>
#include <common/httplib.h>

class ServerBenchmark : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        server_host_ = "localhost";
        server_port_ = 8080;
        client_.reset(new httplib::Client(server_host_, server_port_));
        client_->set_connection_timeout(10);
        client_->set_read_timeout(30);
    }

    void TearDown(const ::benchmark::State& state) override {
        client_.reset();
    }

protected:
    std::string server_host_;
    int server_port_;
    std::unique_ptr<httplib::Client> client_;

    // Test endpoints that we know work
    bool test_root_endpoint() {
        auto response = client_->Get("/");
        return response && response->status == 200;
    }

    bool test_json_post() {
        std::string json_data = R"({
            "name": "benchmark_test",
            "data": "test_data",
            "timestamp": )" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()) + R"(
        })";
        
        auto response = client_->Post("/api/submit_application", json_data, "application/json");
        return response && (response->status == 200 || response->status == 201);
    }

    bool test_static_endpoint() {
        auto response = client_->Get("/static/");
        return response && response->status == 200;
    }

    double calculate_rps(int64_t requests, double duration_seconds) {
        return requests / duration_seconds;
    }

    double calculate_average_response_time(int64_t total_time_ns, int64_t requests) {
        return requests > 0 ? static_cast<double>(total_time_ns) / requests / 1e6 : 0.0;
    }

    double calculate_error_rate(int64_t total_requests, int64_t failed_requests) {
        return total_requests > 0 ? (static_cast<double>(failed_requests) / total_requests) * 100.0 : 0.0;
    }
};

// Benchmark for root endpoint (highest performance)
BENCHMARK_DEFINE_F(ServerBenchmark, BM_RootEndpoint_Load)(benchmark::State& state) {
    const int num_clients = state.range(0);
    const int requests_per_client = state.range(1);
    
    for (auto _ : state) {
        auto test_start = std::chrono::steady_clock::now();
        
        std::vector<std::thread> clients;
        std::atomic<int64_t> total_requests{0};
        std::atomic<int64_t> successful_requests{0};
        std::atomic<int64_t> failed_requests{0};
        std::atomic<int64_t> total_response_time{0};
        
        for (int client_id = 0; client_id < num_clients; ++client_id) {
            clients.emplace_back([&, client_id]() {
                for (int req = 0; req < requests_per_client; ++req) {
                    auto start_time = std::chrono::steady_clock::now();
                    
                    auto response = client_->Get("/");
                    
                    auto end_time = std::chrono::steady_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
                    
                    total_requests++;
                    total_response_time += duration.count();
                    
                    if (response && response->status == 200) {
                        successful_requests++;
                    } else {
                        failed_requests++;
                    }
                }
            });
        }
        
        for (auto& client : clients) {
            client.join();
        }
        
        auto test_end = std::chrono::steady_clock::now();
        double test_duration = std::chrono::duration<double>(test_end - test_start).count();
        
        auto success_count = successful_requests.load();
        auto fail_count = failed_requests.load();
        auto total_count = total_requests.load();
        auto response_time = total_response_time.load();
        
        state.counters["RPS"] = calculate_rps(success_count, test_duration);
        state.counters["Avg_Response_Time_ms"] = calculate_average_response_time(response_time, total_count);
        state.counters["Error_Rate_%"] = calculate_error_rate(total_count, fail_count);
        state.counters["Successful_Requests"] = success_count;
        state.counters["Failed_Requests"] = fail_count;
    }
}

// Benchmark for JSON POST operations
BENCHMARK_DEFINE_F(ServerBenchmark, BM_JsonPost_Load)(benchmark::State& state) {
    const int num_clients = state.range(0);
    const int requests_per_client = state.range(1);
    
    for (auto _ : state) {
        auto test_start = std::chrono::steady_clock::now();
        
        std::vector<std::thread> clients;
        std::atomic<int64_t> total_requests{0};
        std::atomic<int64_t> successful_requests{0};
        std::atomic<int64_t> failed_requests{0};
        std::atomic<int64_t> total_response_time{0};
        
        for (int client_id = 0; client_id < num_clients; ++client_id) {
            clients.emplace_back([&, client_id]() {
                for (int req = 0; req < requests_per_client; ++req) {
                    auto start_time = std::chrono::steady_clock::now();
                    
                    bool success = test_json_post();
                    
                    auto end_time = std::chrono::steady_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
                    
                    total_requests++;
                    total_response_time += duration.count();
                    
                    if (success) {
                        successful_requests++;
                    } else {
                        failed_requests++;
                    }
                }
            });
        }
        
        for (auto& client : clients) {
            client.join();
        }
        
        auto test_end = std::chrono::steady_clock::now();
        double test_duration = std::chrono::duration<double>(test_end - test_start).count();
        
        auto success_count = successful_requests.load();
        auto fail_count = failed_requests.load();
        auto total_count = total_requests.load();
        auto response_time = total_response_time.load();
        
        state.counters["RPS"] = calculate_rps(success_count, test_duration);
        state.counters["Avg_Response_Time_ms"] = calculate_average_response_time(response_time, total_count);
        state.counters["Error_Rate_%"] = calculate_error_rate(total_count, fail_count);
        state.counters["Successful_Requests"] = success_count;
        state.counters["Failed_Requests"] = fail_count;
    }
}

// Benchmark for mixed workload (proportional to real usage)
BENCHMARK_DEFINE_F(ServerBenchmark, BM_MixedWorkload_Realistic)(benchmark::State& state) {
    const int num_clients = state.range(0);
    const int requests_per_client = state.range(1);
    
    for (auto _ : state) {
        auto test_start = std::chrono::steady_clock::now();
        
        std::vector<std::thread> clients;
        std::atomic<int64_t> total_requests{0};
        std::atomic<int64_t> successful_requests{0};
        std::atomic<int64_t> failed_requests{0};
        std::atomic<int64_t> total_response_time{0};
        
        for (int client_id = 0; client_id < num_clients; ++client_id) {
            clients.emplace_back([&, client_id]() {
                for (int req = 0; req < requests_per_client; ++req) {
                    auto start_time = std::chrono::steady_clock::now();
                    bool success = false;
                    
                    // Realistic workload mix:
                    // 70% GET requests (mostly root/static)
                    // 30% POST requests (application submissions)
                    if ((req + client_id) % 10 < 7) {
                        // GET requests
                        if ((req + client_id) % 3 == 0) {
                            success = test_root_endpoint();
                        } else {
                            success = test_static_endpoint();
                        }
                    } else {
                        // POST requests
                        success = test_json_post();
                    }
                    
                    auto end_time = std::chrono::steady_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
                    
                    total_requests++;
                    total_response_time += duration.count();
                    
                    if (success) {
                        successful_requests++;
                    } else {
                        failed_requests++;
                    }
                }
            });
        }
        
        for (auto& client : clients) {
            client.join();
        }
        
        auto test_end = std::chrono::steady_clock::now();
        double test_duration = std::chrono::duration<double>(test_end - test_start).count();
        
        auto success_count = successful_requests.load();
        auto fail_count = failed_requests.load();
        auto total_count = total_requests.load();
        auto response_time = total_response_time.load();
        
        state.counters["RPS"] = calculate_rps(success_count, test_duration);
        state.counters["Avg_Response_Time_ms"] = calculate_average_response_time(response_time, total_count);
        state.counters["Error_Rate_%"] = calculate_error_rate(total_count, fail_count);
        state.counters["Successful_Requests"] = success_count;
        state.counters["Failed_Requests"] = fail_count;
    }
}

// Concurrency scaling test
BENCHMARK_DEFINE_F(ServerBenchmark, BM_ConcurrencyScaling)(benchmark::State& state) {
    const int concurrent_connections = state.range(0);
    
    for (auto _ : state) {
        std::vector<std::thread> clients;
        std::atomic<int> active_connections{0};
        std::atomic<int> max_concurrent_observed{0};
        std::atomic<int64_t> successful_requests{0};
        std::atomic<int64_t> total_response_time{0};
        
        // Barrier for simultaneous start
        std::atomic<int> ready_count{0};
        std::atomic<bool> start_flag{false};
        
        auto client_func = [&](int client_id) {
            ready_count++;
            while (!start_flag.load()) {
                std::this_thread::yield();
            }
            
            int current_active = ++active_connections;
            
            // Update max concurrent
            int current_max = max_concurrent_observed.load();
            while (current_active > current_max) {
                if (max_concurrent_observed.compare_exchange_weak(current_max, current_active)) {
                    break;
                }
            }
            
            auto start_time = std::chrono::steady_clock::now();
            auto response = client_->Get("/");
            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
            
            if (response && response->status == 200) {
                successful_requests++;
            }
            
            total_response_time += duration.count();
            active_connections--;
        };
        
        auto start_time = std::chrono::steady_clock::now();
        
        // Create all clients
        for (int i = 0; i < concurrent_connections; ++i) {
            clients.emplace_back(client_func, i);
        }
        
        // Wait for all to be ready, then start
        while (ready_count.load() < concurrent_connections) {
            std::this_thread::yield();
        }
        start_flag.store(true);
        
        // Wait for completion
        for (auto& client : clients) {
            client.join();
        }
        
        auto end_time = std::chrono::steady_clock::now();
        double duration = std::chrono::duration<double>(end_time - start_time).count();
        
        state.counters["RPS"] = calculate_rps(successful_requests.load(), duration);
        state.counters["Max_Concurrent"] = max_concurrent_observed.load();
        state.counters["Success_Rate_%"] = (static_cast<double>(successful_requests.load()) / concurrent_connections) * 100.0;
        state.counters["Avg_Response_Time_ms"] = calculate_average_response_time(total_response_time.load(), successful_requests.load());
    }
}

// Register benchmarks with comprehensive parameters
BENCHMARK_REGISTER_F(ServerBenchmark, BM_RootEndpoint_Load)
    ->Args({10, 20})     // 10 clients, 20 requests each
    ->Args({50, 10})     // 50 clients, 10 requests each
    ->Args({100, 5})     // 100 clients, 5 requests each
    ->Args({200, 3})     // 200 clients, 3 requests each
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(ServerBenchmark, BM_JsonPost_Load)
    ->Args({10, 10})     // 10 clients, 10 requests each
    ->Args({50, 5})      // 50 clients, 5 requests each
    ->Args({100, 3})     // 100 clients, 3 requests each
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(ServerBenchmark, BM_MixedWorkload_Realistic)
    ->Args({10, 15})     // 10 clients, 15 requests each
    ->Args({50, 8})      // 50 clients, 8 requests each
    ->Args({100, 4})     // 100 clients, 4 requests each
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_REGISTER_F(ServerBenchmark, BM_ConcurrencyScaling)
    ->Arg(10)            // 10 concurrent connections
    ->Arg(50)            // 50 concurrent connections
    ->Arg(100)           // 100 concurrent connections
    ->Arg(200)           // 200 concurrent connections
    ->Arg(500)           // 500 concurrent connections
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK_MAIN();