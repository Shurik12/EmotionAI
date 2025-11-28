cmake_minimum_required(VERSION 3.12)
project(emotionai)

option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Find system dependencies
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(CURL REQUIRED)
find_package(curlpp QUIET)
find_package(OpenSSL REQUIRED)
find_package(ZLIB REQUIRED)
find_package(pugixml REQUIRED)
find_path(HIREDIS_INCLUDE_DIR hiredis/hiredis.h)
find_library(HIREDIS_LIBRARY NAMES hiredis)

# Handle contrib dependencies
add_subdirectory(contrib)

# TORCH + ONNX
set(WITH_TORCH "${CMAKE_CURRENT_SOURCE_DIR}/contrib/libtorch" CACHE PATH "Path to LibTorch")
set(WITH_ONNX "${CMAKE_CURRENT_SOURCE_DIR}/contrib/onnxruntime" CACHE PATH "Path to onnxruntime")
if(EXISTS ${WITH_TORCH})
    find_package(Torch REQUIRED PATHS ${WITH_TORCH})
    message(STATUS "LibTorch found at: ${WITH_TORCH}")
    add_definitions(-DHAVE_TORCH)
endif()
if(EXISTS ${WITH_ONNX})
    find_package(ONNXRuntime REQUIRED PATHS ${WITH_ONNX})
    message(STATUS "ONNX Runtime found at: ${WITH_ONNX}")
    add_definitions(-DHAVE_ONNX)
endif()

# Collect source files
set(SRC_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/client
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cluster
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common
    ${CMAKE_CURRENT_SOURCE_DIR}/src/config
    ${CMAKE_CURRENT_SOURCE_DIR}/src/db
    ${CMAKE_CURRENT_SOURCE_DIR}/src/emotionai
    ${CMAKE_CURRENT_SOURCE_DIR}/src/logging
    ${CMAKE_CURRENT_SOURCE_DIR}/src/metrics
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mtcnn
    ${CMAKE_CURRENT_SOURCE_DIR}/src/server
    ${CMAKE_CURRENT_SOURCE_DIR}/src/storage
    ${CMAKE_CURRENT_SOURCE_DIR}/contrib/minio-cpp/src
)
set(src_headers)
set(src_sources)
foreach(DIR ${SRC_DIRS})
    file(GLOB HEADERS "${DIR}/*.h" "${DIR}/*.hpp")
    file(GLOB SOURCES "${DIR}/*.cc" "${DIR}/*.cpp")
    list(APPEND src_headers ${HEADERS})
    list(APPEND src_sources ${SOURCES})
endforeach()

# Create a library with all common code for reuse in tests/benchmarks
add_library(emotionai_lib STATIC ${src_sources})
target_include_directories(emotionai_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${EMOTIEFFLIB_INCLUDE_DIRS}
    ${HIREDIS_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/contrib/minio-cpp/include
)

target_link_libraries(emotionai_lib PUBLIC
    ${OpenCV_LIBS}
    yaml-cpp
    spdlog::spdlog
    fmt::fmt
    nlohmann_json::nlohmann_json
    emotiefflib
    ${HIREDIS_LIBRARY}

    # Add minio-cpp dependencies
    OpenSSL::SSL
    OpenSSL::Crypto
    CURL::libcurl
    ZLIB::ZLIB
    pugixml::pugixml
    ${INIH_LIBRARY}
    ${INIREADER_LIBRARY}
)

# Handle curlpp conditionally
if(curlpp_FOUND)
    target_link_libraries(emotionai_lib PUBLIC curlpp)
else()
    find_library(CURLPP_LIB curlpp)
    if(CURLPP_LIB)
        target_link_libraries(emotionai_lib PUBLIC ${CURLPP_LIB})
        message(STATUS "Found and linked curlpp via fallback: ${CURLPP_LIB}")
    else()
        message(WARNING "curlpp not found, continuing without it")
    endif()
endif()

target_compile_definitions(emotionai_lib PUBLIC
    SPDLOG_COMPILED_LIB
    FMT_HEADER_ONLY=0
)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE emotionai_lib)

# Testing and benchmarking
# enable_testing()

# if(BUILD_TESTS)
#     add_subdirectory(tests)
# endif()

# if(BUILD_BENCHMARKS)
#     find_package(benchmark)
#     if(benchmark_FOUND)
#         add_subdirectory(tests/benchmark)
#     else()
#         message(WARNING "Google Benchmark not found, benchmarks will not be built")
#     endif()
# endif()