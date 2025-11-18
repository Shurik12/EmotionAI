cmake_minimum_required(VERSION 3.16)

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
