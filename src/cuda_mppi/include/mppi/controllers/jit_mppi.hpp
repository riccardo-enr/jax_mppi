#ifndef MPPI_JIT_MPPI_CONTROLLER_HPP
#define MPPI_JIT_MPPI_CONTROLLER_HPP

#include <cuda.h>
#include <curand.h>
#include <Eigen/Dense>
#include <string>
#include <vector>

#include "mppi/core/mppi_common.cuh"
#include "mppi/jit/jit_compiler.hpp"
#include "mppi/jit/jit_utils.h"

namespace mppi {

/**
 * @brief JIT-compiled MPPI Controller
 *
 * This controller compiles user-provided dynamics and cost functions at runtime
 * using NVRTC and executes them via the CUDA Driver API. This allows users to
 * define custom systems without recompiling the shared library.
 */
class JITMPPIController {
public:
    /**
     * @brief Construct a JIT MPPI Controller
     *
     * @param config MPPI configuration
     * @param dynamics_code C++ code defining a struct with a step() method
     * @param cost_code C++ code defining a struct with compute() and terminal_cost() methods
     * @param include_paths Additional include paths for compilation
     */
    JITMPPIController(const MPPIConfig& config,
                      const std::string& dynamics_code,
                      const std::string& cost_code,
                      const std::vector<std::string>& include_paths = {});

    /**
     * @brief Destructor - cleanup CUDA resources
     */
    ~JITMPPIController();

    /**
     * @brief Compute optimal control for given state
     *
     * @param state Current state vector (size nx)
     */
    void compute(const Eigen::VectorXf& state);

    /**
     * @brief Get the first action from the nominal trajectory
     *
     * @return Action vector (size nu)
     */
    Eigen::VectorXf get_action();

    /**
     * @brief Shift the nominal trajectory forward by one time step
     */
    void shift();

    /**
     * @brief Get the nominal control trajectory
     *
     * @return Control trajectory matrix (horizon x nu)
     */
    Eigen::MatrixXf get_nominal_trajectory();

private:
    MPPIConfig config_;

    // CUDA Driver API handles
    CUmodule module_;
    CUfunction rollout_function_;
    CUcontext context_;

    // Device memory
    float* d_u_nom_;
    float* d_noise_;
    float* d_costs_;
    float* d_initial_state_;
    float* d_weights_;

    // CuRAND
    curandGenerator_t gen_;

    // Helper methods
    void initialize_cuda_resources();
    void compile_and_load_module(const std::string& dynamics_code,
                                  const std::string& cost_code,
                                  const std::vector<std::string>& include_paths);
    void compute_weights(float* h_costs, float* h_weights);
    void update_nominal_controls(const float* h_weights);
};

} // namespace mppi

#endif // MPPI_JIT_MPPI_CONTROLLER_HPP
