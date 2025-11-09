cmake_minimum_required(VERSION 3.12)
project(EmotionAI_Tests)

# Find packages
find_package(GTest REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(nlohmann_json REQUIRED)

find_path(HIREDIS_INCLUDE_DIR hiredis/hiredis.h)
find_library(HIREDIS_LIBRARY NAMES hiredis)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/mocks
)

# List test source files
set(TEST_SOURCES
    main.cpp
    unit/config/ConfigTest.cpp
    unit/logging/LoggerTest.cpp
    unit/db/RedisManagerTest.cpp
)

# List required source files
set(MAIN_SOURCES
    ${CMAKE_SOURCE_DIR}/src/config/Config.cpp
    ${CMAKE_SOURCE_DIR}/src/logging/Logger.cpp
    ${CMAKE_SOURCE_DIR}/src/common/base64.cpp
    ${CMAKE_SOURCE_DIR}/src/db/RedisManager.cpp
)

# Create test executable
add_executable(emotionai_tests
    ${TEST_SOURCES}
    ${MAIN_SOURCES}
)

# Set output directory
set_target_properties(emotionai_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Link libraries
target_link_libraries(emotionai_tests
    GTest::gtest
    GTest::gtest_main
    GTest::gmock
    yaml-cpp
    spdlog::spdlog
    fmt::fmt
    nlohmann_json::nlohmann_json
    ${HIREDIS_LIBRARY}
    pthread
)

# Add compile definitions
target_compile_definitions(emotionai_tests PRIVATE
    SPDLOG_COMPILED_LIB
    FMT_HEADER_ONLY=0
)

# Add test
add_test(NAME EmotionAI_Unit_Tests COMMAND emotionai_tests)

# Create a test target for easy running
add_custom_target(run_tests
    COMMAND ./emotionai_tests
    DEPENDS emotionai_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running EmotionAI tests"
)

# Organize files in IDE
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${TEST_SOURCES})
source_group("Main Sources" FILES ${MAIN_SOURCES})
source_group("Mocks" FILES ${CMAKE_CURRENT_SOURCE_DIR}/mocks/TestServerFactory.cpp)
