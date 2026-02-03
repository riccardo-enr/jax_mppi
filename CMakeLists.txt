cmake_minimum_required(VERSION 3.18)
project(jax_mppi_project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  add_subdirectory(src/cuda_mppi)
else()
  message(STATUS "CUDA compiler not found. Building without CUDA support.")
endif()
