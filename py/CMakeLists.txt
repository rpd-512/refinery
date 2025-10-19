cmake_minimum_required(VERSION 3.14)
project(py_refinery LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# FetchContent version (optional, auto-downloads PyBind11 if not installed)
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.1
)
FetchContent_MakeAvailable(pybind11)

# Output directory for the compiled library
set(LIB_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/lib")
file(MAKE_DIRECTORY ${LIB_OUTPUT_DIR})

# Source file (bindings.cpp is in the same folder as this CMakeLists.txt)
set(SRC_FILES bindings.cpp)

# Build the PyBind11 module
pybind11_add_module(py_refinery ${SRC_FILES})

# Output directory settings
set_target_properties(py_refinery PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${LIB_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${LIB_OUTPUT_DIR} # for Windows
)

message(STATUS "PyBind11 module will be output to: ${LIB_OUTPUT_DIR}")
