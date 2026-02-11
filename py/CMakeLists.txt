cmake_minimum_required(VERSION 3.14)
project(pyrefinery_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.12.1
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(_core bindings.cpp)

target_include_directories(_core
    PRIVATE
        ${CMAKE_SOURCE_DIR}/../include
)
