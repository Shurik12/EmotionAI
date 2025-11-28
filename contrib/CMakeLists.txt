# Handle internal contrib dependencies

# Build inih from submodule if available
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/inih")
    message(STATUS "Building inih from submodule")
    
    # Add the inih library
    add_library(inih STATIC
        inih/ini.c
    )
    target_include_directories(inih PUBLIC
        inih
    )
    
    # Add the INIReader C++ wrapper
    add_library(inireader STATIC
        inih/cpp/INIReader.cpp
    )
    target_include_directories(inireader PUBLIC
        inih
        inih/cpp
    )
    target_link_libraries(inireader PUBLIC inih)
    
    set(INIH_LIBRARY inih PARENT_SCOPE)
    set(INIREADER_LIBRARY inireader PARENT_SCOPE)
    set(INIH_INCLUDE_DIRS 
        ${CMAKE_CURRENT_SOURCE_DIR}/inih 
        ${CMAKE_CURRENT_SOURCE_DIR}/inih/cpp 
        PARENT_SCOPE
    )
else()
    message(WARNING "inih submodule not found, minio-cpp AWS config provider may not work")
    set(INIH_LIBRARY "" PARENT_SCOPE)
    set(INIREADER_LIBRARY "" PARENT_SCOPE)
    set(INIH_INCLUDE_DIRS "" PARENT_SCOPE)
endif()

# Add emotiefflib subdirectory
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/emotiefflib/emotieffcpplib")
    add_subdirectory("emotiefflib/emotieffcpplib")
    
    get_target_property(EMOTIEFFLIB_INCLUDE_DIRS emotiefflib INTERFACE_INCLUDE_DIRECTORIES)
    if(NOT EMOTIEFFLIB_INCLUDE_DIRS)
        # If the property is not set, manually set the include directory
        set(EMOTIEFFLIB_INCLUDE_DIRS
            "${CMAKE_CURRENT_SOURCE_DIR}/emotiefflib/emotieffcpplib/include"
        )
        message(STATUS "Manually setting emotiefflib include dir: ${EMOTIEFFLIB_INCLUDE_DIRS}")
    endif()
    
    set(EMOTIEFFLIB_INCLUDE_DIRS ${EMOTIEFFLIB_INCLUDE_DIRS} PARENT_SCOPE)
else()
    message(FATAL_ERROR "emotiefflib not found in contrib directory")
endif()

# Note: libtorch and onnxruntime are handled as external pre-built dependencies in main CMakeLists.txt
# Note: minio-cpp source files are included directly in the main project sources