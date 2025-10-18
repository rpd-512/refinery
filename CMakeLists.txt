cmake_minimum_required(VERSION 3.10)
project(Refinery)

set(CMAKE_CXX_STANDARD 17)

# Look for yaml-cpp
find_package(yaml-cpp REQUIRED)

# Eigen is header-only; just include the path
include_directories(/usr/include/eigen3)

add_executable(refinery_linux main.cpp)

target_link_libraries(refinery_linux yaml-cpp)
