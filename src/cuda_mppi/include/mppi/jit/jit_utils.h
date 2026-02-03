#ifndef MPPI_JIT_UTILS_H
#define MPPI_JIT_UTILS_H

#include <cuda.h>
#include <iostream>
#include <nvrtc.h>
#include <string>
#include <vector>

#define NVRTC_SAFE_CALL(x)                                                     \
  do {                                                                         \
    nvrtcResult result = x;                                                    \
    if (result != NVRTC_SUCCESS) {                                             \
      std::cerr << "\nerror: " << nvrtcGetErrorString(result)                  \
                << " failed with error " << result << '\n';                    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CUDA_DRIVER_SAFE_CALL(x)                                               \
  do {                                                                         \
    CUresult result = x;                                                       \
    if (result != CUDA_SUCCESS) {                                              \
      const char *msg;                                                         \
      cuGetErrorName(result, &msg);                                            \
      std::cerr << "\nerror: " << msg << " failed with error " << result       \
                << '\n';                                                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#endif // MPPI_JIT_UTILS_H
