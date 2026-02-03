#include "mppi/controllers/jit_mppi.hpp"
#include "mppi/utils/cuda_utils.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

namespace mppi {

JITMPPIController::JITMPPIController(const MPPIConfig& config,
                                     const std::string& dynamics_code,
                                     const std::string& cost_code,
                                     const std::vector<std::string>& include_paths)
    : config_(config), module_(nullptr), rollout_function_(nullptr) {

    // Initialize CUDA context
    CUDA_DRIVER_SAFE_CALL(cuInit(0));
    CUdevice device;
    CUDA_DRIVER_SAFE_CALL(cuDeviceGet(&device, 0));
    // Use primary context instead of creating a new one
    CUDA_DRIVER_SAFE_CALL(cuDevicePrimaryCtxRetain(&context_, device));

    // Allocate device memory
    initialize_cuda_resources();

    // Compile and load the JIT module
    compile_and_load_module(dynamics_code, cost_code, include_paths);
}

JITMPPIController::~JITMPPIController() {
    // Free device memory
    cudaFree(d_u_nom_);
    cudaFree(d_noise_);
    cudaFree(d_costs_);
    cudaFree(d_initial_state_);
    cudaFree(d_weights_);

    // Destroy CuRAND generator
    curandDestroyGenerator(gen_);

    // Unload module
    if (module_) {
        cuModuleUnload(module_);
    }
    // Release primary context
    if (context_) {
        CUdevice device;
        cuDeviceGet(&device, 0);
        cuDevicePrimaryCtxRelease(device);
    }
}

void JITMPPIController::initialize_cuda_resources() {
    // Allocate device memory
    HANDLE_ERROR(cudaMalloc(&d_u_nom_, config_.horizon * config_.nu * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_noise_, config_.num_samples * config_.horizon * config_.nu * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_costs_, config_.num_samples * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_initial_state_, config_.nx * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_weights_, config_.num_samples * sizeof(float)));

    // Initialize u_nom to zero
    HANDLE_ERROR(cudaMemset(d_u_nom_, 0, config_.horizon * config_.nu * sizeof(float)));

    // Setup CuRAND
    HANDLE_CURAND_ERROR(curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT));
    HANDLE_CURAND_ERROR(curandSetPseudoRandomGeneratorSeed(gen_, 1234ULL));
}

void JITMPPIController::compile_and_load_module(const std::string& dynamics_code,
                                                 const std::string& cost_code,
                                                 const std::vector<std::string>& include_paths) {
    // Add the cuda_mppi include directory to the include paths
    std::vector<std::string> all_include_paths = include_paths;

    // Get the include directory from the installation or build directory
    // This assumes the headers are installed alongside the Python module
    // or we're running from the build directory
    const char* cuda_mppi_include = std::getenv("CUDA_MPPI_INCLUDE_DIR");
    if (cuda_mppi_include) {
        all_include_paths.push_back(cuda_mppi_include);
    } else {
        // Try to find it relative to the Python module
        // For now, we'll rely on the user setting the environment variable
        // or providing the path explicitly
        std::cerr << "Warning: CUDA_MPPI_INCLUDE_DIR not set. "
                  << "JIT compilation may fail if headers are not found.\n";
    }

    // Compile the code
    std::string ptx = jit::JITCompiler::compile(dynamics_code, cost_code, all_include_paths);

    // Load the PTX module
    CUDA_DRIVER_SAFE_CALL(cuModuleLoadData(&module_, ptx.c_str()));

    // Get the function handle
    std::string kernel_name = jit::JITCompiler::get_wrapper_name();
    CUDA_DRIVER_SAFE_CALL(cuModuleGetFunction(&rollout_function_, module_, kernel_name.c_str()));
}

void JITMPPIController::compute(const Eigen::VectorXf& state) {
    // Copy state to device
    HANDLE_ERROR(cudaMemcpy(d_initial_state_, state.data(),
                           config_.nx * sizeof(float), cudaMemcpyHostToDevice));

    // Generate noise
    HANDLE_CURAND_ERROR(curandGenerateNormal(gen_, d_noise_,
                                             config_.num_samples * config_.horizon * config_.nu,
                                             0.0f, 1.0f));

    // Launch the JIT-compiled rollout kernel using Driver API
    // Kernel signature: rollout_wrapper(dynamics, cost, config, initial_state, u_nom, noise, costs)

    // Note: For simplicity, we pass structs by value. The user's dynamics and cost structs
    // are created in the kernel wrapper. Here we just pass the config and pointers.

    // Prepare kernel arguments
    void* args[] = {
        &config_,
        &d_initial_state_,
        &d_u_nom_,
        &d_noise_,
        &d_costs_
    };

    // Calculate grid and block dimensions
    int threads = 256;
    int blocks = (config_.num_samples + threads - 1) / threads;

    // Launch kernel
    CUDA_DRIVER_SAFE_CALL(cuLaunchKernel(
        rollout_function_,
        blocks, 1, 1,       // grid dim
        threads, 1, 1,      // block dim
        0,                  // shared mem bytes
        nullptr,            // stream
        args,               // kernel args
        nullptr             // extra
    ));

    // Synchronize
    CUDA_DRIVER_SAFE_CALL(cuCtxSynchronize());

    // Compute weights and update nominal controls
    std::vector<float> h_costs(config_.num_samples);
    std::vector<float> h_weights(config_.num_samples);

    HANDLE_ERROR(cudaMemcpy(h_costs.data(), d_costs_,
                           config_.num_samples * sizeof(float), cudaMemcpyDeviceToHost));

    compute_weights(h_costs.data(), h_weights.data());
    update_nominal_controls(h_weights.data());
}

void JITMPPIController::compute_weights(float* h_costs, float* h_weights) {
    // Find minimum cost for numerical stability
    float min_cost = h_costs[0];
    for (int k = 1; k < config_.num_samples; ++k) {
        if (h_costs[k] < min_cost) {
            min_cost = h_costs[k];
        }
    }

    // Compute exp weights
    float sum_weights = 0.0f;
    for (int k = 0; k < config_.num_samples; ++k) {
        float w = std::exp(-(h_costs[k] - min_cost) / config_.lambda);
        h_weights[k] = w;
        sum_weights += w;
    }

    // Normalize
    for (int k = 0; k < config_.num_samples; ++k) {
        h_weights[k] /= sum_weights;
    }
}

void JITMPPIController::update_nominal_controls(const float* h_weights) {
    // Copy weights to device
    HANDLE_ERROR(cudaMemcpy(d_weights_, h_weights,
                           config_.num_samples * sizeof(float), cudaMemcpyHostToDevice));

    // Copy current u_nom and noise to host for update
    // In a more optimized version, this would be done entirely on GPU
    int total_params = config_.horizon * config_.nu;
    std::vector<float> h_u_nom(total_params);
    std::vector<float> h_noise(config_.num_samples * total_params);

    HANDLE_ERROR(cudaMemcpy(h_u_nom.data(), d_u_nom_,
                           total_params * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(h_noise.data(), d_noise_,
                           config_.num_samples * total_params * sizeof(float),
                           cudaMemcpyDeviceToHost));

    // Update: u_nom += sum(w * noise)
    for (int i = 0; i < total_params; ++i) {
        float weighted_sum = 0.0f;
        for (int k = 0; k < config_.num_samples; ++k) {
            weighted_sum += h_weights[k] * h_noise[k * total_params + i];
        }
        h_u_nom[i] += weighted_sum;
    }

    // Copy updated u_nom back to device
    HANDLE_ERROR(cudaMemcpy(d_u_nom_, h_u_nom.data(),
                           total_params * sizeof(float), cudaMemcpyHostToDevice));
}

Eigen::VectorXf JITMPPIController::get_action() {
    Eigen::VectorXf action(config_.nu);
    HANDLE_ERROR(cudaMemcpy(action.data(), d_u_nom_,
                           config_.nu * sizeof(float), cudaMemcpyDeviceToHost));
    return action;
}

void JITMPPIController::shift() {
    // Shift u_nom left by one time step
    int shift_floats = config_.nu;
    int total_floats = config_.horizon * config_.nu;
    int copy_floats = total_floats - shift_floats;

    // Use host-side buffer for simplicity
    std::vector<float> h_u_nom(total_floats);
    HANDLE_ERROR(cudaMemcpy(h_u_nom.data(), d_u_nom_,
                           total_floats * sizeof(float), cudaMemcpyDeviceToHost));

    // Shift
    for (int i = 0; i < copy_floats; ++i) {
        h_u_nom[i] = h_u_nom[i + shift_floats];
    }

    // Zero out last step
    for (int i = copy_floats; i < total_floats; ++i) {
        h_u_nom[i] = 0.0f;
    }

    // Copy back
    HANDLE_ERROR(cudaMemcpy(d_u_nom_, h_u_nom.data(),
                           total_floats * sizeof(float), cudaMemcpyHostToDevice));
}

Eigen::MatrixXf JITMPPIController::get_nominal_trajectory() {
    int total_floats = config_.horizon * config_.nu;
    std::vector<float> h_u_nom(total_floats);
    HANDLE_ERROR(cudaMemcpy(h_u_nom.data(), d_u_nom_,
                           total_floats * sizeof(float), cudaMemcpyDeviceToHost));

    Eigen::MatrixXf trajectory(config_.horizon, config_.nu);
    for (int t = 0; t < config_.horizon; ++t) {
        for (int i = 0; i < config_.nu; ++i) {
            trajectory(t, i) = h_u_nom[t * config_.nu + i];
        }
    }

    return trajectory;
}

} // namespace mppi
