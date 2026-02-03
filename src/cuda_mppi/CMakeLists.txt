cmake_minimum_required(VERSION 3.18)
project(cuda_mppi LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# Gather sources
file(GLOB_RECURSE SOURCES "src/*.cu" "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.cuh" "include/*.h" "include/*.hpp")

# JIT compiler sources (these need to be compiled separately)
set(JIT_SOURCES
    src/jit/jit_compiler.cpp
    src/jit/jit_mppi_controller.cpp
)

# Define header-only library (renamed to avoid conflict with extension)
add_library(cuda_mppi_lib INTERFACE)

# Interface dependencies
target_link_libraries(cuda_mppi_lib INTERFACE
    Eigen3::Eigen
    CUDA::curand
    CUDA::nvrtc
    CUDA::cuda_driver
)
# Interface include directories
target_include_directories(cuda_mppi_lib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIR}
)

# Python Extension (this will be jax_mppi.cuda_mppi)
nanobind_add_module(cuda_mppi bindings/bindings.cu ${JIT_SOURCES})
target_link_libraries(cuda_mppi PRIVATE cuda_mppi_lib)

# Install into the python package
install(TARGETS cuda_mppi DESTINATION jax_mppi)
install(DIRECTORY include/mppi DESTINATION jax_mppi/include)

# Example executable to verify build (disabled due to Eigen/CUDA compatibility warnings)
# add_executable(mppi_test src/main_test.cu)
# target_link_libraries(mppi_test PRIVATE cuda_mppi_lib)
