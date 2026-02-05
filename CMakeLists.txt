cmake_minimum_required(VERSION 3.18)
project(jax_mppi_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
  enable_language(CUDA)
  add_subdirectory(third_party/cuda-mppi)
else()
  message(STATUS "CUDA Toolkit not found. Building without CUDA extensions.")
endif()