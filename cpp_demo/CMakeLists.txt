cmake_minimum_required(VERSION 3.10)
project(MultiChannelDemo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Voyager SDK
set(VOYAGER_SDK_PATH "$ENV{HOME}/dvir/voyager-sdk" CACHE PATH "Path to Voyager SDK")

if(NOT EXISTS "${VOYAGER_SDK_PATH}")
    message(FATAL_ERROR "Voyager SDK not found at ${VOYAGER_SDK_PATH}")
endif()

message(STATUS "Using Voyager SDK: ${VOYAGER_SDK_PATH}")

# Include directories
include_directories(
    ${VOYAGER_SDK_PATH}/include
)

# Link directories
link_directories(
    ${VOYAGER_SDK_PATH}/lib
)

# Add executable
add_executable(multichannel_demo
    multichannel_demo.cpp
)

# Link libraries
target_link_libraries(multichannel_demo
    axinferencenet
    axruntime
    pthread
)

# Set RPATH for runtime library search
set_target_properties(multichannel_demo PROPERTIES
    BUILD_RPATH "${VOYAGER_SDK_PATH}/lib"
    INSTALL_RPATH "${VOYAGER_SDK_PATH}/lib"
)

message(STATUS "Build configuration:")
message(STATUS "  Executable: multichannel_demo")
message(STATUS "  SDK path: ${VOYAGER_SDK_PATH}")
message(STATUS "  Libraries: axinferencenet, axruntime")
