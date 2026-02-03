cmake_minimum_required(VERSION 3.18)
project(cuda_mppi LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# Gather sources - purely for IDEs or explicit checking, but not compiled into interface
file(GLOB_RECURSE SOURCES "src/*.cu" "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.cuh" "include/*.h" "include/*.hpp")

# Define header-only library (renamed to avoid conflict with extension)
add_library(cuda_mppi_lib INTERFACE)

# Interface dependencies
target_link_libraries(cuda_mppi_lib INTERFACE 
    Eigen3::Eigen
    CUDA::curand
)

# Interface include directories
target_include_directories(cuda_mppi_lib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIR}
)

# Python Extension (this will be jax_mppi.cuda_mppi)
nanobind_add_module(cuda_mppi bindings/bindings.cpp)
target_link_libraries(cuda_mppi PRIVATE cuda_mppi_lib)

# Install into the python package
install(TARGETS cuda_mppi DESTINATION jax_mppi)

# Example executable to verify build
add_executable(mppi_test src/main_test.cu)
target_link_libraries(mppi_test PRIVATE cuda_mppi_lib)
