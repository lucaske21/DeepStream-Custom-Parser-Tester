cmake_minimum_required(VERSION 3.10)
project(deepstream_parser_tester CXX)

# ---------------------------------------------------------------------------
# Language standard: C++17, no GNU extensions.
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Build type (default: Release)
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ---------------------------------------------------------------------------
# DeepStream NvDsInfer headers
#
# Enable this when parser_tester must load a parser .so built against the real
# DeepStream SDK headers. The tester and parser .so must agree on these struct
# layouts because the parser ABI passes NvDsInfer C++ structs and std::vector.
# Example:
#   cmake .. -DUSE_DEEPSTREAM_NVDSINFER=ON \
#            -DDEEPSTREAM_INCLUDE_DIR=/opt/nvidia/deepstream/deepstream/sources/includes
# ---------------------------------------------------------------------------
option(USE_DEEPSTREAM_NVDSINFER
    "Use DeepStream SDK nvdsinfer.h instead of the local fake header" OFF)
set(DEEPSTREAM_INCLUDE_DIR "" CACHE PATH "Directory containing DeepStream nvdsinfer.h")

if(USE_DEEPSTREAM_NVDSINFER)
    find_path(DEEPSTREAM_NVDSINFER_INCLUDE_DIR
        NAMES nvdsinfer.h
        HINTS ${DEEPSTREAM_INCLUDE_DIR}
              /opt/nvidia/deepstream/deepstream/sources/includes
              /opt/nvidia/deepstream/deepstream/include)

    find_path(DEEPSTREAM_CUSTOM_IMPL_INCLUDE_DIR
        NAMES nvdsinfer_custom_impl.h
        HINTS ${DEEPSTREAM_INCLUDE_DIR}
              ${DEEPSTREAM_NVDSINFER_INCLUDE_DIR}
              /opt/nvidia/deepstream/deepstream/sources/includes
              /opt/nvidia/deepstream/deepstream/include)

    if(NOT DEEPSTREAM_NVDSINFER_INCLUDE_DIR)
        message(FATAL_ERROR
            "DeepStream nvdsinfer.h not found.\n"
            "Set -DDEEPSTREAM_INCLUDE_DIR=/path/to/deepstream/sources/includes "
            "or disable -DUSE_DEEPSTREAM_NVDSINFER=OFF.")
    endif()

    if(NOT DEEPSTREAM_CUSTOM_IMPL_INCLUDE_DIR)
        message(FATAL_ERROR
            "DeepStream nvdsinfer_custom_impl.h not found.\n"
            "Set -DDEEPSTREAM_INCLUDE_DIR=/path/to/deepstream/sources/includes "
            "or disable -DUSE_DEEPSTREAM_NVDSINFER=OFF.")
    endif()

    message(STATUS "Using DeepStream nvdsinfer.h from: ${DEEPSTREAM_NVDSINFER_INCLUDE_DIR}")
    message(STATUS "Using DeepStream nvdsinfer_custom_impl.h from: ${DEEPSTREAM_CUSTOM_IMPL_INCLUDE_DIR}")
else()
    message(STATUS "Using local fake_nvdsinfer.h")
endif()

# ---------------------------------------------------------------------------
# OpenCV
# ---------------------------------------------------------------------------
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs:    ${OpenCV_LIBS}")

# ---------------------------------------------------------------------------
# ONNX Runtime
#
# Set ORT_ROOT to your ONNX Runtime installation directory if the library
# is not installed in a standard location, e.g.:
#   cmake -DORT_ROOT=/path/to/onnxruntime ..
# ---------------------------------------------------------------------------
set(ORT_ROOT "" CACHE PATH "ONNX Runtime installation root (optional)")

if(ORT_ROOT)
    set(_ORT_INCLUDE_HINTS "${ORT_ROOT}/include")
    set(_ORT_LIB_HINTS     "${ORT_ROOT}/lib")
else()
    set(_ORT_INCLUDE_HINTS /usr/local/include /usr/include)
    set(_ORT_LIB_HINTS     /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu)
endif()

find_path(ORT_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS ${_ORT_INCLUDE_HINTS}
          ${_ORT_INCLUDE_HINTS}/onnxruntime
          ${_ORT_INCLUDE_HINTS}/onnxruntime/core/session)

find_library(ORT_LIB
    NAMES onnxruntime
    HINTS ${_ORT_LIB_HINTS})

if(NOT ORT_INCLUDE_DIR)
    message(FATAL_ERROR
        "ONNX Runtime headers not found.\n"
        "Install ONNX Runtime or set -DORT_ROOT=/path/to/onnxruntime.")
endif()
if(NOT ORT_LIB)
    message(FATAL_ERROR
        "ONNX Runtime library (libonnxruntime) not found.\n"
        "Install ONNX Runtime or set -DORT_ROOT=/path/to/onnxruntime.")
endif()

message(STATUS "ORT include: ${ORT_INCLUDE_DIR}")
message(STATUS "ORT lib:     ${ORT_LIB}")

# ---------------------------------------------------------------------------
# nlohmann/json  (fetched automatically via CMake FetchContent)
# ---------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE)

# Disable nlohmann test targets so the build stays clean.
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install    OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(nlohmann_json)

# ---------------------------------------------------------------------------
# Include directories
# ---------------------------------------------------------------------------
include_directories(
    ${DEEPSTREAM_NVDSINFER_INCLUDE_DIR}
    ${DEEPSTREAM_CUSTOM_IMPL_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${ORT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS})

# ---------------------------------------------------------------------------
# Main executable
#
# parser_tester discovers the user-supplied parser at runtime via dlopen/dlsym.
# ---------------------------------------------------------------------------
add_executable(parser_tester
    src/main.cpp
    src/ort_runner.cpp
    src/tensor_adapter.cpp
    src/visualizer.cpp)

if(USE_DEEPSTREAM_NVDSINFER)
    target_compile_definitions(parser_tester PRIVATE USE_DEEPSTREAM_NVDSINFER=1)
endif()

target_link_libraries(parser_tester
    PRIVATE
        ${CMAKE_DL_LIBS}
        ${ORT_LIB}
        ${OpenCV_LIBS}
        nlohmann_json::nlohmann_json)

# ---------------------------------------------------------------------------
# Ensure the output directory exists at build time.
# ---------------------------------------------------------------------------
add_custom_command(TARGET parser_tester POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_BINARY_DIR}/output)

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
install(TARGETS parser_tester DESTINATION bin)


