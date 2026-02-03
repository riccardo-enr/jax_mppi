#ifndef MPPI_KERNELS_CUH
#define MPPI_KERNELS_CUH

#include <cuda_runtime.h>
#include "mppi_common.cuh"

namespace mppi {
namespace kernels {

template <typename Dynamics, typename Cost>
__global__ void rollout_kernel(
    Dynamics dynamics,
    Cost cost,
    MPPIConfig config,
    const float* __restrict__ initial_state,
    const float* __restrict__ u_nom,    // Nominal control (T, nu)
    const float* __restrict__ noise,    // Noise (K, T, nu)
    float* __restrict__ costs           // Output costs (K)
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= config.num_samples) return;

    // Use compile-time constants if available, otherwise fallback to local arrays
    // assuming reasonable max dimensions or dynamic indexing if supported
    constexpr int MAX_NX = 32; 
    constexpr int MAX_NU = 12;

    float x[MAX_NX];
    float u[MAX_NU];
    float x_next[MAX_NX];

    // Copy initial state
    for(int i=0; i<config.nx; ++i) {
        x[i] = initial_state[i];
    }
    
    float total_cost = 0.0f;
    
    for (int t = 0; t < config.horizon; ++t) {
        // Compute control u = u_nom[t] + noise[k, t]
        for(int i=0; i<config.nu; ++i) {
            // noise index: k * (T * nu) + t * nu + i
            int noise_idx = k * (config.horizon * config.nu) + t * config.nu + i;
            float n_val = noise[noise_idx];
            
            // u_nom index: t * nu + i
            float u_val = u_nom[t * config.nu + i] + n_val;
            
            // Apply scale
            u_val *= config.u_scale;
            
            // TODO: Apply bounds (u_min, u_max) if passed in config or accessible
            
            u[i] = u_val;
        }
        
        // Step dynamics
        dynamics.step(x, u, x_next, config.dt);
        
        // Compute cost
        total_cost += cost.compute(x, u, t);
        
        // Update state
        for(int i=0; i<config.nx; ++i) {
            x[i] = x_next[i];
        }
    }
    
    // Terminal cost
    total_cost += cost.terminal_cost(x);
    
    costs[k] = total_cost;
}

} // namespace kernels
} // namespace mppi

#endif // MPPI_KERNELS_CUH
