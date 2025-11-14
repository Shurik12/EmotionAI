cmake_minimum_required(VERSION 3.16)
project(ServerBenchmarks)

if(NOT benchmark_FOUND)
    find_package(benchmark REQUIRED)
endif()

set(BENCHMARK_SOURCES
    ServerBenchmark.cpp
)

add_executable(server_benchmarks ${BENCHMARK_SOURCES})

target_include_directories(server_benchmarks PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${EMOTIEFFLIB_INCLUDE_DIRS}
    ${HIREDIS_INCLUDE_DIR}
)

target_link_libraries(server_benchmarks PRIVATE
    emotionai_lib # link to main project library
    benchmark::benchmark
    Threads::Threads
)

set_target_properties(server_benchmarks PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
