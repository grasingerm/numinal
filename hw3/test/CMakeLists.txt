# boilerplate CMakeLists.txt

cmake_minimum_required(VERSION 2.8)

project(numinal)

# uncomment to use clang
set(CC clang)
set(CXX clang)

# set flags for warnings, errors, and standards
set(CMAKE_BUILD_TYPE debug)
set(CMAKE_C_FLAGS_DEBUG "-Wall -Wextra -g -std=c99 -pedantic -pedantic-errors")
set(CMAKE_C_FLAGS_RELEASE "-Wall -Wextra -std=c99 -pedantic -pedantic-errors -O3")
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -g -std=c++11 -pedantic -pedantic-errors")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -Wextra -std=c++11 -pedantic -pedantic-errors -O3")

# build directories
set(CMAKE_BINARY_DIR build)
set(EXECUTABLE_OUTPUT_PATH bin)

# include and link directories
# 
# i.e. ...
include_directories(../src)
include_directories(../../share)

# dependencies
add_executable(test_jacobi test_jacobi.cpp ../src/iterative.cpp)
add_executable(test_conj_grad_sd test_conj_grad_sd.cpp ../src/iterative.cpp)

# link libraries
target_link_libraries(test_jacobi armadillo)
target_link_libraries(test_conj_grad_sd armadillo)

# uncomment if using the standard c math library
# if (UNIX)
#     target_link_libraries(some_executable m)
# endif (UNIX)

install(TARGETS test_jacobi test_conj_grad_sd DESTINATION bin)
