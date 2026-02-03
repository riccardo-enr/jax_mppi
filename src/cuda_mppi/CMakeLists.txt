cmake_minimum_required(VERSION 3.18)
project(cuda_mppi LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Gather sources - purely for IDEs or explicit checking, but not compiled into interface
file(GLOB_RECURSE SOURCES "src/*.cu" "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.cuh" "include/*.h" "include/*.hpp")

# Define header-only library
add_library(cuda_mppi INTERFACE)

# Interface dependencies
target_link_libraries(cuda_mppi INTERFACE 
    Eigen3::Eigen
    CUDA::curand
)

# Interface include directories
target_include_directories(cuda_mppi INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIR}
)

# Example executable to verify build
add_executable(mppi_test src/main_test.cu)
target_link_libraries(mppi_test PRIVATE cuda_mppi)