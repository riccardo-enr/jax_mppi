#ifndef MPPI_CUDA_UTILS_CUH
#define MPPI_CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include <cstdio>

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

inline void __cudaCheckError(const char* file, const int line)
{
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err)
  {
    fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
    exit(-1);
  }
}

inline const char* curandGetErrorString(curandStatus_t code)
{
  switch (code)
  {
    case CURAND_STATUS_SUCCESS: return "No errors.";
    case CURAND_STATUS_VERSION_MISMATCH: return "Header file and linked library version do not match.";
    case CURAND_STATUS_NOT_INITIALIZED: return "Generator not initialized.";
    case CURAND_STATUS_ALLOCATION_FAILED: return "Memory allocation failed.";
    case CURAND_STATUS_TYPE_ERROR: return "Generator is wrong type.";
    case CURAND_STATUS_OUT_OF_RANGE: return "Argument out of range.";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE: return "Length requested is not a multple of dimension.";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: return "GPU does not have double precision required by MRG32k3a.";
    case CURAND_STATUS_LAUNCH_FAILURE: return "Kernel launch failure.";
    case CURAND_STATUS_PREEXISTING_FAILURE: return "Preexisting failure on library entry.";
    case CURAND_STATUS_INITIALIZATION_FAILED: return "Initialization of CUDA failed.";
    case CURAND_STATUS_ARCH_MISMATCH: return "Architecture mismatch, GPU does not support requested feature.";
    case CURAND_STATUS_INTERNAL_ERROR: return "Internal library error.";
    default: return "Curand Error";
  }
}

inline void curandAssert(curandStatus_t code, const char* file, int line, bool abort = true)
{
  if (code != CURAND_STATUS_SUCCESS)
  {
    fprintf(stderr, "Curandassert: %s %s %d\n", curandGetErrorString(code), file, line);
    if (abort)
    {
      exit(code);
    }
  }
}

#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)
#define HANDLE_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define HANDLE_CURAND_ERROR(ans) { curandAssert((ans), __FILE__, __LINE__); }

#endif // MPPI_CUDA_UTILS_CUH
