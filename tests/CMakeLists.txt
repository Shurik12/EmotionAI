cmake_minimum_required(VERSION 3.16)
project(EmotionAI_Tests)

# Find packages
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

# Test source files
set(TEST_SOURCES
    main.cpp
    unit/config/ConfigTest.cpp
    unit/logging/LoggerTest.cpp
    unit/db/RedisManagerTest.cpp
)

# Create test executable
add_executable(emotionai_tests ${TEST_SOURCES})

# Include directories - FIXED: Add all required includes
target_include_directories(emotionai_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/mocks
    ${HIREDIS_INCLUDE_DIR}
    ${EMOTIEFFLIB_INCLUDE_DIRS}
)

target_link_libraries(emotionai_tests
    emotionai_lib # link to main project library
    GTest::gtest
    GTest::gtest_main
    GTest::gmock
    Threads::Threads
)

set_target_properties(emotionai_tests PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

add_test(NAME EmotionAI_Unit_Tests COMMAND emotionai_tests)
