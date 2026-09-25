cmake_minimum_required(VERSION 3.16)
project(EmotionAI_Tests)

# GoogleTest is required to build any of the C++ tests. Fail soft so that
# `make build` still works on machines without libgtest-dev.
find_package(GTest QUIET)
find_package(Threads REQUIRED)

if(NOT GTest_FOUND)
    message(WARNING "GoogleTest not found - skipping emotionai_tests. "
                    "Install with: sudo apt-get install -y libgtest-dev libgmock-dev")
    return()
endif()

# Integration and end-to-end suites start a real Server (DragonflyDB + models),
# so they are opt-in. Unit tests always build.
option(EMOTIONAI_BUILD_INTEGRATION_TESTS
       "Build integration/e2e tests (need DragonflyDB and model weights)" OFF)

set(TEST_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/config
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/logging
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/db
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/server
)

if(EMOTIONAI_BUILD_INTEGRATION_TESTS)
    list(APPEND TEST_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/integration/server
        ${CMAKE_CURRENT_SOURCE_DIR}/integration/end_to_end
    )
endif()

set(test_sources)
set(test_headers)
foreach(DIR ${TEST_DIRS})
    file(GLOB HEADERS "${DIR}/*.h" "${DIR}/*.hpp")
    file(GLOB SOURCES "${DIR}/*.cc" "${DIR}/*.cpp")
    list(APPEND test_headers ${HEADERS})
    list(APPEND test_sources ${SOURCES})
endforeach()

# Add main test runner
list(APPEND test_sources ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

# Create test executable
add_executable(emotionai_tests ${test_sources})

# Include directories
target_include_directories(emotionai_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/mocks
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link with the main library and test dependencies
target_link_libraries(emotionai_tests
    emotionai_lib # This brings in all the package dependencies
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
