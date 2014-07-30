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

# dependencies
add_executable(s91 s91.cpp direct_methods.cpp direct_methods.hpp eigen.cpp eigen.hpp)
target_link_libraries(s91 armadillo)

install(TARGETS s91 DESTINATION bin)

