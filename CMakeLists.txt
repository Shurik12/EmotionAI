cmake_minimum_required(VERSION 3.12)
project(EmotionAI)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Enable folder organization in IDEs
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(hiredis REQUIRED)

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

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/contrib/emotiefflib/emotieffcpplib/include
    ${OpenCV_INCLUDE_DIRS}
    ${yaml-cpp_INCLUDE_DIRS}
    ${spdlog_INCLUDE_DIRS}
    ${fmt_INCLUDE_DIRS}
    ${nlohmann_json_INCLUDE_DIRS}
    ${hiredis_INCLUDE_DIRS}
)

# Source files
set(COMMON_SOURCES
    src/common/Config.cpp
    src/common/Logging.cpp
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
)

# Add main executable
add_executable(${PROJECT_NAME}
    main.cpp
    ${COMMON_SOURCES}
    ${DB_SOURCES}
    ${EMOTIONAI_SOURCES}
    ${SERVER_SOURCES}
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

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    yaml-cpp
    spdlog::spdlog
    fmt::fmt
    nlohmann_json::nlohmann_json
    emotiefflib
    hiredis
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

message(STATUS "EmotionAI project configured successfully")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "LibTorch available: ${TORCH_FOUND}")
message(STATUS "ONNX Runtime available: ${ONNXRuntime_FOUND}")
