cmake_minimum_required(VERSION 3.16)

# ---------------------------------------------------------------------------
# Offline burnout pipeline benchmark.
#
# Runs the real inference path over a directory of recordings and emits
# per-file JSON plus an aggregate report. It has no Google Benchmark
# dependency, so it builds whenever the project builds.
# ---------------------------------------------------------------------------
add_executable(emotionai_burnout_benchmark
    BurnoutBenchmark.cpp
)

target_include_directories(emotionai_burnout_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(emotionai_burnout_benchmark PRIVATE
    emotionai_lib
)

set_target_properties(emotionai_burnout_benchmark PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# ---------------------------------------------------------------------------
# HTTP/server micro-benchmarks (Google Benchmark). Optional: only built when
# the benchmark package is available.
# ---------------------------------------------------------------------------
if(TARGET benchmark::benchmark)
    add_executable(emotionai_benchmark
        main.cpp
        EmotionServerBenchmark.cpp
        LoadTestBenchmark.cpp
        ConcurrentClientBenchmark.cpp
    )

    target_include_directories(emotionai_benchmark PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(emotionai_benchmark
        emotionai_lib
        benchmark::benchmark
        ${OpenCV_LIBS}
        Threads::Threads
    )

    target_compile_definitions(emotionai_benchmark PRIVATE
        BENCHMARK_EMOTIONAI
    )

    set_target_properties(emotionai_benchmark PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
else()
    message(STATUS "Google Benchmark not found - skipping emotionai_benchmark target")
endif()
