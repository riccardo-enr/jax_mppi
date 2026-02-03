#ifndef MPPI_CONTROLLER_CUH
#define MPPI_CONTROLLER_CUH

#include <vector>
#include <cuda_runtime.h>
#include <curand.h>
#include <Eigen/Dense>
#include <iostream>

#include "mppi/core/mppi_common.cuh"
#include "mppi/core/kernels.cuh"
#include "mppi/utils/cuda_utils.cuh"

namespace mppi {

__global__ void weighted_update_kernel(
    float* u_nom,
    const float* noise,
    const float* weights,
    int K,
    int total_params // T * nu
);

__global__ void shift_kernel(float* u_nom, int T, int nu);

template <typename Dynamics, typename Cost>
class MPPIController {
public:
    MPPIController(const MPPIConfig& config, const Dynamics& dynamics, const Cost& cost)
        : config_(config), dynamics_(dynamics), cost_(cost) {
        
        // Allocate device memory
        HANDLE_ERROR(cudaMalloc(&d_u_nom_, config.horizon * config.nu * sizeof(float)));
        HANDLE_ERROR(cudaMalloc(&d_noise_, config.num_samples * config.horizon * config.nu * sizeof(float)));
        HANDLE_ERROR(cudaMalloc(&d_costs_, config.num_samples * sizeof(float)));
        HANDLE_ERROR(cudaMalloc(&d_initial_state_, config.nx * sizeof(float)));
        HANDLE_ERROR(cudaMalloc(&d_weights_, config.num_samples * sizeof(float)));

        // Initialize u_nom to zero
        HANDLE_ERROR(cudaMemset(d_u_nom_, 0, config.horizon * config.nu * sizeof(float)));

        // Setup CuRAND
        HANDLE_CURAND_ERROR(curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT));
        HANDLE_CURAND_ERROR(curandSetPseudoRandomGeneratorSeed(gen_, 1234ULL));
    }

    ~MPPIController() {
        cudaFree(d_u_nom_);
        cudaFree(d_noise_);
        cudaFree(d_costs_);
        cudaFree(d_initial_state_);
        cudaFree(d_weights_);
        curandDestroyGenerator(gen_);
    }

    void compute(const Eigen::VectorXf& state) {
        // Copy state to device
        HANDLE_ERROR(cudaMemcpy(d_initial_state_, state.data(), config_.nx * sizeof(float), cudaMemcpyHostToDevice));

        // Sample noise
        // Assuming normal distribution with stddev 1.0 (sigma handled in update or pre-scaled?)
        // JAX implementation has noise_sigma.
        // For simplicity, let's assume noise_sigma is Identity * scale, or handled by scaling noise here.
        // Or we pass sigma to kernel.
        // Simple version: Generate N(0, 1) and assume sigma=1 for now, or scaled in kernel.
        // JAX: noise = multivar_normal(0, sigma)
        // Here: Generate Normal
        HANDLE_CURAND_ERROR(curandGenerateNormal(gen_, d_noise_, config_.num_samples * config_.horizon * config_.nu, 0.0f, 1.0f));

        // Launch Rollout Kernel
        dim3 block(256);
        dim3 grid((config_.num_samples + block.x - 1) / block.x);
        
        kernels::rollout_kernel<<<grid, block>>>(
            dynamics_,
            cost_,
            config_,
            d_initial_state_,
            d_u_nom_,
            d_noise_,
            d_costs_
        );
        HANDLE_ERROR(cudaGetLastError());
        HANDLE_ERROR(cudaDeviceSynchronize()); // For debugging/safety

        // Compute Weights (Softmax)
        // This usually requires a reduction to find min cost, then exp, then sum, then normalize.
        // I'll implement a simple host-side weight computation for now to get it running, 
        // as implementing full device reduction is verbose.
        // PROTOTYPE: Copy costs to host, compute weights, update u_nom on host, copy back.
        
        std::vector<float> h_costs(config_.num_samples);
        HANDLE_ERROR(cudaMemcpy(h_costs.data(), d_costs_, config_.num_samples * sizeof(float), cudaMemcpyDeviceToHost));

        float min_cost = h_costs[0];
        for(float c : h_costs) if(c < min_cost) min_cost = c;

        std::vector<float> h_weights(config_.num_samples);
        float sum_weights = 0.0f;
        for(int k=0; k<config_.num_samples; ++k) {
            float w = expf(-(h_costs[k] - min_cost) / config_.lambda);
            h_weights[k] = w;
            sum_weights += w;
        }

        for(int k=0; k<config_.num_samples; ++k) {
            h_weights[k] /= sum_weights;
        }

        // Update U_nom
        // u_nom += sum(w * noise)
        // Again, doing on host for simplicity of implementation in this prototype phase.
        // Copy noise to host is expensive (K*T*NU floats).
        // Better: implement a simple update kernel.
        
        // Let's do a simple update kernel.
        // update_control_kernel<<<1, 1>>>( ... );
        
        HANDLE_ERROR(cudaMemcpy(d_weights_, h_weights.data(), config_.num_samples * sizeof(float), cudaMemcpyHostToDevice));
        
        int num_params = config_.horizon * config_.nu;
        int threads = 256;
        int blocks = (num_params + threads - 1) / threads;
        
        weighted_update_kernel<<<blocks, threads>>>(
            d_u_nom_,
            d_noise_,
            d_weights_,
            config_.num_samples,
            num_params 
        );
        HANDLE_ERROR(cudaGetLastError());
        
        // Shift control (handled by caller or explicit method? JAX does it at end of step)
        // We'll add a shift method.
    }
    
    void shift() {
        // Shift d_u_nom_ left by 1 step (nu floats)
        // Fill last step with zero (or u_init)
        // Simple kernel or memcpy
        // Memcpy device-to-device
        int shift_floats = config_.nu;
        int total_floats = config_.horizon * config_.nu;
        int copy_floats = total_floats - shift_floats;
        
        // Allocate temp buffer or just move? overlap is tricky with memcpy?
        // cudaMemcpy handles overlap if we use memmove equivalent? cudaMemcpy is undefined for overlap?
        // "cudaMemcpy: If dst and src overlap, the behavior is undefined."
        // So we need a temp buffer or a kernel.
        // Kernel is easiest.
        
        int threads = 256;
        int blocks = (copy_floats + threads - 1) / threads;
        shift_kernel<<<blocks, threads>>>(d_u_nom_, config_.horizon, config_.nu);
    }

    Eigen::VectorXf get_action() {
        // Return first action from d_u_nom_
        Eigen::VectorXf action(config_.nu);
        HANDLE_ERROR(cudaMemcpy(action.data(), d_u_nom_, config_.nu * sizeof(float), cudaMemcpyDeviceToHost));
        return action;
    }

private:
    MPPIConfig config_;
    Dynamics dynamics_;
    Cost cost_;

    // Device memory
    float* d_u_nom_;
    float* d_noise_;
    float* d_costs_;
    float* d_initial_state_;
    float* d_weights_;
    
    // CuRAND
    curandGenerator_t gen_;
};

// Helper Kernels

__global__ void weighted_update_kernel(
    float* u_nom,
    const float* noise,
    const float* weights,
    int K,
    int total_params // T * nu
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_params) return;
    
    float sum = 0.0f;
    for(int k=0; k<K; ++k) {
        sum += weights[k] * noise[k * total_params + idx];
    }
    
    u_nom[idx] += sum;
}

__global__ void shift_kernel(float* u_nom, int T, int nu) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (T - 1) * nu) return;
    
    // Shift: u[i] = u[i + nu]
    u_nom[idx] = u_nom[idx + nu];
    
    // Zero out last step? 
    // We can do that in a separate check or kernel
}

} // namespace mppi

#endif // MPPI_CONTROLLER_CUH
