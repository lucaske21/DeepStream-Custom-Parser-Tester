cmake_minimum_required(VERSION 3.10)
project(deepstream_parser_tester CXX)

# ---------------------------------------------------------------------------
# Language standard: C++11, no GNU extensions.
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 11)
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
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/parser
    ${ORT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS})

# ---------------------------------------------------------------------------
# Parser static library
#
# Compiled with HAVE_DEEPSTREAM undefined → picks up fake_nvdsinfer.h.
# To compile against the real DeepStream SDK, pass:
#   cmake -DHAVE_DEEPSTREAM=ON ..
# ---------------------------------------------------------------------------
option(HAVE_DEEPSTREAM "Link against the real DeepStream SDK headers" OFF)

add_library(yolo26_parser STATIC
    parser/yolo26_parser.cpp)

target_include_directories(yolo26_parser PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/parser)

if(HAVE_DEEPSTREAM)
    target_compile_definitions(yolo26_parser PUBLIC HAVE_DEEPSTREAM)
    message(STATUS "HAVE_DEEPSTREAM=ON – using real nvdsinfer.h")
else()
    message(STATUS "HAVE_DEEPSTREAM=OFF – using fake_nvdsinfer.h (no SDK needed)")
endif()

# ---------------------------------------------------------------------------
# Main executable
# ---------------------------------------------------------------------------
add_executable(parser_tester
    src/main.cpp
    src/ort_runner.cpp
    src/tensor_adapter.cpp
    src/visualizer.cpp)

target_link_libraries(parser_tester
    PRIVATE
        yolo26_parser
        ${ORT_LIB}
        ${OpenCV_LIBS}
        nlohmann_json::nlohmann_json)

if(HAVE_DEEPSTREAM)
    target_compile_definitions(parser_tester PRIVATE HAVE_DEEPSTREAM)
endif()

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

# ---------------------------------------------------------------------------
# Parser unit tests (no ONNX Runtime required)
# ---------------------------------------------------------------------------
enable_testing()

add_executable(test_parser tests/test_parser.cpp parser/yolo26_parser.cpp)
target_include_directories(test_parser PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/parser)
if(HAVE_DEEPSTREAM)
    target_compile_definitions(test_parser PRIVATE HAVE_DEEPSTREAM)
endif()

add_test(NAME test_parser COMMAND test_parser)

