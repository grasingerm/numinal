cmake_minimum_required(VERSION 2.8)

project(numinal)

# compiler flags
set(CMAKE_BUILD_TYPE debug)
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -g -std=c++11 -pedantic -pedantic-errors")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -Wextra -g -std=c++11 -pedantic -pedantic-errors -O3")

# build directories
set(CMAKE_BINARY_DIR build)
set(EXECUTABLE_OUTPUT_PATH bin)

include_directories("../../share")
include_directories("../src")

# dependencies
add_executable(test_lu test_lu.cpp ../src/direct_methods.cpp ../src/direct_methods.hpp)

target_link_libraries(test_lu armadillo)

install(TARGETS test_lu DESTINATION bin)

