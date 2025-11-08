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

# Check if MTCNN source files exist and build them
if(EXISTS "${PROJECT_SOURCE_DIR}/contrib/emotiefflib/emotieffcpplib/3rdparty/opencv-mtcnn/src")
    file(GLOB MTCNN_SOURCES
        "${PROJECT_SOURCE_DIR}/contrib/emotiefflib/emotieffcpplib/3rdparty/opencv-mtcnn/src/*.cpp"
        "${PROJECT_SOURCE_DIR}/contrib/emotiefflib/emotieffcpplib/3rdparty/opencv-mtcnn/lib/src/*.cpp"
    )

    if(MTCNN_SOURCES)
        add_library(opencv-mtcnn STATIC ${MTCNN_SOURCES})
        target_include_directories(opencv-mtcnn PUBLIC
            "${PROJECT_SOURCE_DIR}/contrib/emotiefflib/emotieffcpplib/3rdparty/opencv-mtcnn/include"
            "${PROJECT_SOURCE_DIR}/contrib/emotiefflib/emotieffcpplib/3rdparty/opencv-mtcnn/lib/include"
        )
        target_link_libraries(opencv-mtcnn PUBLIC ${OpenCV_LIBS})
        message(STATUS "Built opencv-mtcnn from sources")
    endif()
endif()

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

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${yaml-cpp_INCLUDE_DIRS}
    ${spdlog_INCLUDE_DIRS}
    ${fmt_INCLUDE_DIRS}
    ${nlohmann_json_INCLUDE_DIRS}
    ${hiredis_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/contrib/emotiefflib/emotieffcpplib/include
)

# Source files
set(COMMON_SOURCES
    src/common/Config.cpp
    src/common/Logging.cpp
    src/common/base64.cpp
)

set(DB_SOURCES
    src/db/RedisManager.cpp
)

set(EMOTIONAI_SOURCES
    src/emotionai/Image.cpp
    src/emotionai/FileProcessor.cpp
)

set(SERVER_SOURCES
    src/server/WebServer.cpp
    src/server/MultiplexingServer.cpp
)

set(MTCNN_SOURCES
    src/mtcnn/detector.cc
    src/mtcnn/onet.cc
    src/mtcnn/pnet.cc
    src/mtcnn/rnet.cc
)

# Add main executable
add_executable(${PROJECT_NAME}
    main.cpp
    ${COMMON_SOURCES}
    ${DB_SOURCES}
    ${EMOTIONAI_SOURCES}
    ${SERVER_SOURCES}
    ${MTCNN_SOURCES}
)

# Set target properties
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Organize files in IDE
set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER "Main")
set_property(SOURCE ${COMMON_SOURCES} PROPERTY FOLDER "Common")
set_property(SOURCE ${DB_SOURCES} PROPERTY FOLDER "Database")
set_property(SOURCE ${EMOTIONAI_SOURCES} PROPERTY FOLDER "EmotionAI")
set_property(SOURCE ${SERVER_SOURCES} PROPERTY FOLDER "Server")
set_property(SOURCE ${MTCNN_SOURCES} PROPERTY FOLDER "Mtcnn")

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    yaml-cpp
    spdlog::spdlog
    fmt::fmt
    nlohmann_json::nlohmann_json
    emotiefflib
    ${HIREDIS_LIBRARY}
)

# Add definitions
target_compile_definitions(${PROJECT_NAME} PRIVATE
    SPDLOG_COMPILED_LIB
    FMT_HEADER_ONLY=0
)

# Install target (optional)
install(TARGETS ${PROJECT_NAME} DESTINATION bin)

# Copy config file
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/config.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/config.yaml
    COPYONLY
)

# Create models directory if it doesn't exist
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/models
    COMMENT "Creating models directory"
)

# ============================================================================
# TEST CONFIGURATION
# ============================================================================
option(BUILD_TESTS "Build tests" ON)

if(BUILD_TESTS)
    # Download GoogleTest and GoogleMock
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )
    
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    
    enable_testing()
    
    # Create object library from main project sources for testing
    add_library(EmotionAI_Objects OBJECT
        ${COMMON_SOURCES}
        ${DB_SOURCES}
        ${EMOTIONAI_SOURCES}
        ${SERVER_SOURCES}
        ${MTCNN_SOURCES}
    )
    
    target_include_directories(EmotionAI_Objects PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/contrib/emotiefflib/emotieffcpplib/include
    )
    
    target_link_libraries(EmotionAI_Objects PUBLIC
        ${OpenCV_LIBS}
        yaml-cpp
        spdlog::spdlog
        fmt::fmt
        nlohmann_json::nlohmann_json
        emotiefflib
        ${HIREDIS_LIBRARY}
    )
    
    # Unit Tests
    set(UNIT_TEST_SOURCES
        tests/unit/common/ConfigTest.cpp
        tests/unit/common/LoggingTest.cpp
        tests/unit/db/RedisManagerTest.cpp
        tests/unit/emotionai/ImageTest.cpp
        tests/unit/emotionai/FileProcessorTest.cpp
        tests/unit/server/WebServerTest.cpp
        tests/utils/TestUtils.cpp
        tests/unit/test_main.cpp  # Add this file
    )
    
    add_executable(EmotionAI_UnitTests ${UNIT_TEST_SOURCES})
    
    # Integration Tests
    set(INTEGRATION_TEST_SOURCES
        tests/integration/BasicIntegrationTest.cpp
        tests/utils/TestUtils.cpp
        tests/integration/test_main.cpp  # Add this file
    )
    
    add_executable(EmotionAI_IntegrationTests ${INTEGRATION_TEST_SOURCES})
    
    # Common test properties for both test executables
    foreach(test_target EmotionAI_UnitTests EmotionAI_IntegrationTests)
        target_include_directories(${test_target} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/tests
            ${CMAKE_CURRENT_SOURCE_DIR}/contrib/emotiefflib/emotieffcpplib/include
        )
        
        # Link against the object library and dependencies
        target_link_libraries(${test_target} PRIVATE
            EmotionAI_Objects
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
            ${OpenCV_LIBS}
            yaml-cpp
            spdlog::spdlog
            fmt::fmt
            nlohmann_json::nlohmann_json
            ${HIREDIS_LIBRARY}
            emotiefflib
        )
        
        set_target_properties(${test_target} PROPERTIES
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
        )
        
        # Set output directory for test binaries
        set_target_properties(${test_target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
        )
    endforeach()
    
    # Organize test targets in IDE
    set_target_properties(EmotionAI_UnitTests PROPERTIES FOLDER "Tests/Unit")
    set_target_properties(EmotionAI_IntegrationTests PROPERTIES FOLDER "Tests/Integration")
    
    # Add tests to CTest
    add_test(NAME EmotionAI_UnitTests COMMAND $<TARGET_FILE:EmotionAI_UnitTests>)
    add_test(NAME EmotionAI_IntegrationTests COMMAND $<TARGET_FILE:EmotionAI_IntegrationTests>)
    
    # Set test properties for better output
    set_tests_properties(EmotionAI_UnitTests PROPERTIES
        TIMEOUT 60
        LABELS "unit"
    )
    
    set_tests_properties(EmotionAI_IntegrationTests PROPERTIES
        TIMEOUT 120
        LABELS "integration"
    )
    
    message(STATUS "Tests enabled: Unit tests and Integration tests")
endif()

message(STATUS "EmotionAI project configured successfully")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "LibTorch available: ${TORCH_FOUND}")
message(STATUS "ONNX Runtime available: ${ONNXRuntime_FOUND}")
