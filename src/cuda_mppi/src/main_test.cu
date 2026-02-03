#include <iostream>
#include <Eigen/Dense>
#include "mppi/controllers/mppi.cuh"
#include "mppi/controllers/smppi.cuh"
#include "mppi/controllers/kmppi.cuh"

using namespace mppi;

// Dummy Dynamics
struct DoubleIntegrator {
    static constexpr int STATE_DIM = 4;
    static constexpr int CONTROL_DIM = 2;

    __device__ void step(const float* state, const float* u, float* next_state, float dt) const {
        // x_pos += x_vel * dt
        next_state[0] = state[0] + state[2] * dt;
        next_state[1] = state[1] + state[3] * dt;
        // x_vel += u * dt
        next_state[2] = state[2] + u[0] * dt;
        next_state[3] = state[3] + u[1] * dt;
    }
};

// Dummy Cost
struct QuadraticCost {
    __device__ float compute(const float* state, const float* u, int t) const {
        float c = 0.0f;
        // State cost
        for(int i=0; i<4; ++i) c += state[i]*state[i];
        // Control cost
        for(int i=0; i<2; ++i) c += u[i]*u[i];
        return c;
    }

    __device__ float terminal_cost(const float* state) const {
        float c = 0.0f;
        for(int i=0; i<4; ++i) c += state[i]*state[i] * 10.0f;
        return c;
    }
};

int main() {
    MPPIConfig config;
    config.num_samples = 128;
    config.horizon = 30;
    config.nx = 4;
    config.nu = 2;
    config.lambda = 1.0f;
    config.dt = 0.05f;
    config.u_scale = 1.0f;
    config.w_action_seq_cost = 1.0f;
    config.num_support_pts = 10;

    DoubleIntegrator dyn;
    QuadraticCost cost;

    // Test MPPI
    std::cout << "Initializing MPPI..." << std::endl;
    MPPIController<DoubleIntegrator, QuadraticCost> mppi(config, dyn, cost);
    Eigen::VectorXf state = Eigen::VectorXf::Zero(4);
    mppi.compute(state);
    Eigen::VectorXf action = mppi.get_action();
    std::cout << "MPPI Action: " << action.transpose() << std::endl;

    // Test SMPPI
    std::cout << "Initializing SMPPI..." << std::endl;
    SMPPIController<DoubleIntegrator, QuadraticCost> smppi(config, dyn, cost);
    smppi.compute(state);
    std::cout << "SMPPI Computed." << std::endl;

    // Test KMPPI
    std::cout << "Initializing KMPPI..." << std::endl;
    KMPPIController<DoubleIntegrator, QuadraticCost> kmppi(config, dyn, cost);
    kmppi.compute(state);
    std::cout << "KMPPI Computed." << std::endl;

    return 0;
}
