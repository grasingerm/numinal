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
add_executable(s92 s92.cpp direct_methods.cpp direct_methods.hpp eigen.cpp eigen.hpp)
add_executable(s93 s93.cpp direct_methods.cpp direct_methods.hpp eigen.cpp eigen.hpp)
add_executable(s94 s94.cpp direct_methods.cpp direct_methods.hpp eigen.cpp eigen.hpp)

if(UNIX)
    target_link_libraries(s92 m)
    target_link_libraries(s93 m)
    target_link_libraries(s94 m)
endif(UNIX)

target_link_libraries(s92 armadillo)
target_link_libraries(s93 armadillo)
target_link_libraries(s94 armadillo)

install(TARGETS s92 s93 s94 DESTINATION bin)

