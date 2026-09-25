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

# Find FFTW3 for mel spectrogram computation (without PkgConfig)
find_path(FFTW3_INCLUDE_DIR
    NAMES fftw3.h
    PATHS /usr/include /usr/local/include /usr/include/x86_64-linux-gnu
)

find_library(FFTW3_LIBRARY
    NAMES fftw3 fftw3-3
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
)

find_library(FFTW3F_LIBRARY
    NAMES fftw3f fftw3f-3
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
)

# Check if FFTW3 was found
if(FFTW3_INCLUDE_DIR AND FFTW3_LIBRARY AND FFTW3F_LIBRARY)
    set(FFTW3_FOUND TRUE)
    set(FFTW3_INCLUDE_DIRS ${FFTW3_INCLUDE_DIR})
    set(FFTW3_LIBRARIES ${FFTW3_LIBRARY} ${FFTW3F_LIBRARY})
    message(STATUS "FFTW3 found:")
    message(STATUS "  Include dir: ${FFTW3_INCLUDE_DIRS}")
    message(STATUS "  Libraries: ${FFTW3_LIBRARIES}")
else()
    set(FFTW3_FOUND FALSE)
    message(WARNING "FFTW3 not found - mel spectrogram computation will not work")
endif()

find_path(HIREDIS_INCLUDE_DIR hiredis/hiredis.h)
find_library(HIREDIS_LIBRARY NAMES hiredis)

find_path(AVCODEC_INCLUDE_DIR
    NAMES libavcodec/avcodec.h
    PATHS /usr/include /usr/local/include /usr/include/x86_64-linux-gnu
)

find_path(AVFORMAT_INCLUDE_DIR
    NAMES libavformat/avformat.h
    PATHS /usr/include /usr/local/include /usr/include/x86_64-linux-gnu
)

find_path(AVUTIL_INCLUDE_DIR
    NAMES libavutil/avutil.h
    PATHS /usr/include /usr/local/include /usr/include/x86_64-linux-gnu
)

find_path(SWRESAMPLE_INCLUDE_DIR
    NAMES libswresample/swresample.h
    PATHS /usr/include /usr/local/include /usr/include/x86_64-linux-gnu
)

find_library(AVCODEC_LIBRARY NAMES avcodec)
find_library(AVFORMAT_LIBRARY NAMES avformat)
find_library(AVUTIL_LIBRARY NAMES avutil)
find_library(SWRESAMPLE_LIBRARY NAMES swresample)

# Check if FFmpeg was found and set variables
if(AVCODEC_INCLUDE_DIR AND AVFORMAT_INCLUDE_DIR AND AVUTIL_INCLUDE_DIR AND 
   SWRESAMPLE_INCLUDE_DIR AND AVCODEC_LIBRARY AND AVFORMAT_LIBRARY AND 
   AVUTIL_LIBRARY AND SWRESAMPLE_LIBRARY)
    set(FFMPEG_FOUND TRUE)
    set(FFMPEG_INCLUDE_DIRS 
        ${AVCODEC_INCLUDE_DIR}
        ${AVFORMAT_INCLUDE_DIR}
        ${AVUTIL_INCLUDE_DIR}
        ${SWRESAMPLE_INCLUDE_DIR}
    )
    set(FFMPEG_LIBRARIES 
        ${AVCODEC_LIBRARY}
        ${AVFORMAT_LIBRARY}
        ${AVUTIL_LIBRARY}
        ${SWRESAMPLE_LIBRARY}
    )
    message(STATUS "FFmpeg found:")
    message(STATUS "  Include dirs: ${FFMPEG_INCLUDE_DIRS}")
    message(STATUS "  Libraries: ${FFMPEG_LIBRARIES}")
    add_definitions(-DHAVE_FFMPEG)
else()
    set(FFMPEG_FOUND FALSE)
    message(WARNING "FFmpeg not found - audio processing will be limited to WAV files only")
endif()

set(WITH_TORCH "${CMAKE_CURRENT_SOURCE_DIR}/contrib/libtorch" CACHE PATH "Path to LibTorch")
set(WITH_ONNX "${CMAKE_CURRENT_SOURCE_DIR}/contrib/onnxruntime" CACHE PATH "Path to onnxruntime")

add_subdirectory(contrib)

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
    ${CMAKE_CURRENT_SOURCE_DIR}/src/audio
    ${CMAKE_CURRENT_SOURCE_DIR}/src/client
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cluster
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common
    ${CMAKE_CURRENT_SOURCE_DIR}/src/config
    ${CMAKE_CURRENT_SOURCE_DIR}/src/db
    ${CMAKE_CURRENT_SOURCE_DIR}/src/emotionai
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gigachat
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
    ${FFMPEG_INCLUDE_DIRS}
    ${FFTW3_INCLUDE_DIRS}  # Add FFTW3 include
)

target_link_libraries(emotionai_lib PUBLIC
    ${OpenCV_LIBS}
    yaml-cpp
    spdlog::spdlog
    fmt::fmt
    nlohmann_json::nlohmann_json
    emotiefflib
    ${HIREDIS_LIBRARY}
    ${FFMPEG_LIBRARIES}
    ${FFTW3_LIBRARIES}  # Add FFTW3 libraries

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

# Add FFTW3 definitions if found
if(FFTW3_FOUND)
    target_compile_definitions(emotionai_lib PUBLIC HAVE_FFTW3)
endif()

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE emotionai_lib)

# Offline measurement harness for the burnout pipeline (see tools/).
# Off by default so normal builds are unaffected; enable with -DBUILD_HARNESS=ON.
option(BUILD_HARNESS "Build the offline burnout measurement harness" OFF)
if(BUILD_HARNESS)
    add_executable(burnout_harness tools/burnout_harness.cpp)
    target_link_libraries(burnout_harness PRIVATE emotionai_lib)
endif()

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