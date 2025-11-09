cmake_minimum_required(VERSION 3.12)
project(EmotionAI)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Enable folder organization in IDEs
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(nlohmann_json REQUIRED)

find_path(HIREDIS_INCLUDE_DIR hiredis/hiredis.h)
find_library(HIREDIS_LIBRARY NAMES hiredis)

# Set paths for optional dependencies
set(WITH_TORCH "${CMAKE_CURRENT_SOURCE_DIR}/contrib/libtorch" CACHE PATH "Path to LibTorch")
set(WITH_ONNX "${CMAKE_CURRENT_SOURCE_DIR}/contrib/onnxruntime" CACHE PATH "Path to onnxruntime")

# Check for LibTorch
if(EXISTS ${WITH_TORCH})
    find_package(Torch REQUIRED PATHS ${WITH_TORCH})
    message(STATUS "LibTorch found at: ${WITH_TORCH}")
    add_definitions(-DHAVE_TORCH)
endif()

# Check for ONNX Runtime
if(EXISTS ${WITH_ONNX})
    find_package(ONNXRuntime REQUIRED PATHS ${WITH_ONNX})
    message(STATUS "ONNX Runtime found at: ${WITH_ONNX}")
    add_definitions(-DHAVE_ONNX)
endif()

# Add subdirectories
add_subdirectory("${PROJECT_SOURCE_DIR}/contrib/emotiefflib/emotieffcpplib")

get_target_property(EMOTIEFFLIB_INCLUDE_DIRS emotiefflib INTERFACE_INCLUDE_DIRECTORIES)
if(NOT EMOTIEFFLIB_INCLUDE_DIRS)
    # If the property is not set, manually set the include directory
    set(EMOTIEFFLIB_INCLUDE_DIRS
        "${PROJECT_SOURCE_DIR}/contrib/emotiefflib/emotieffcpplib/include"
    )
    message(STATUS "Manually setting emotiefflib include dir: ${EMOTIEFFLIB_INCLUDE_DIRS}")
endif()

set(SRC_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/client
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common
    ${CMAKE_CURRENT_SOURCE_DIR}/src/config
    ${CMAKE_CURRENT_SOURCE_DIR}/src/db
    ${CMAKE_CURRENT_SOURCE_DIR}/src/emotionai
    ${CMAKE_CURRENT_SOURCE_DIR}/src/logging
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mtcnn
    ${CMAKE_CURRENT_SOURCE_DIR}/src/server
)
set(src_headers)
set(src_sources)
foreach(DIR ${SRC_DIRS})
    file(GLOB HEADERS "${DIR}/*.h" "${DIR}/*.hpp")
    file(GLOB SOURCES "${DIR}/*.cc" "${DIR}/*.cpp")
    list(APPEND src_headers ${HEADERS})
    list(APPEND src_sources ${SOURCES})
endforeach()

# Add main executable
add_executable(${PROJECT_NAME} main.cpp)
target_sources(${PROJECT_NAME} PRIVATE ${src_sources})
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${EMOTIEFFLIB_INCLUDE_DIRS}
    ${HIREDIS_INCLUDE_DIR}
)
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    yaml-cpp
    spdlog::spdlog
    fmt::fmt
    nlohmann_json::nlohmann_json
    emotiefflib
    ${HIREDIS_LIBRARY}
)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
target_compile_definitions(${PROJECT_NAME} PRIVATE
    SPDLOG_COMPILED_LIB
    FMT_HEADER_ONLY=0
)

enable_testing()
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    find_package(GTest REQUIRED)
    add_subdirectory(tests)
endif()