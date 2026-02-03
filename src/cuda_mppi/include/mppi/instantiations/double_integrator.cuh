#ifndef MPPI_INSTANTIATIONS_DOUBLE_INTEGRATOR_CUH
#define MPPI_INSTANTIATIONS_DOUBLE_INTEGRATOR_CUH

#include <cuda_runtime.h>

namespace mppi {
namespace instantiations {

struct DoubleIntegrator {
    static constexpr int STATE_DIM = 4;
    static constexpr int CONTROL_DIM = 2;

    __host__ __device__ void step(const float* state, const float* u, float* next_state, float dt) const {
        // x_pos += x_vel * dt
        next_state[0] = state[0] + state[2] * dt;
        next_state[1] = state[1] + state[3] * dt;
        // x_vel += u * dt
        next_state[2] = state[2] + u[0] * dt;
        next_state[3] = state[3] + u[1] * dt;
    }
};

struct QuadraticCost {
    __host__ __device__ float compute(const float* state, const float* u, int t) const {
        float c = 0.0f;
        // State cost (to origin)
        for(int i=0; i<4; ++i) c += state[i]*state[i];
        // Control cost
        for(int i=0; i<2; ++i) c += u[i]*u[i];
        return c;
    }

    __host__ __device__ float terminal_cost(const float* state) const {
        float c = 0.0f;
        for(int i=0; i<4; ++i) c += state[i]*state[i] * 10.0f;
        return c;
    }
};

} // namespace instantiations
} // namespace mppi

#endif // MPPI_INSTANTIATIONS_DOUBLE_INTEGRATOR_CUH
