#ifndef SMPPI_CONTROLLER_CUH
#define SMPPI_CONTROLLER_CUH

#include "mppi.cuh"

namespace mppi {

__global__ void integrate_actions_kernel(
    const float* action_seq_nom,
    const float* u_vel_nom,
    const float* noise_vel,
    float* perturbed_actions,
    MPPIConfig config
);

__global__ void integrate_single_action_kernel(
    float* action_seq,
    const float* u_vel,
    MPPIConfig config
);

__global__ void smoothness_cost_kernel(
    const float* perturbed_actions,
    float* costs,
    MPPIConfig config
);

template <typename Dynamics, typename Cost>
class SMPPIController {
public:
    SMPPIController(const MPPIConfig& config, const Dynamics& dynamics, const Cost& cost)
        : config_(config), dynamics_(dynamics), cost_(cost) {
        
        HANDLE_ERROR(cudaMalloc(&d_u_vel_, config.horizon * config.nu * sizeof(float)));
        HANDLE_ERROR(cudaMalloc(&d_action_seq_, config.horizon * config.nu * sizeof(float)));
        HANDLE_ERROR(cudaMalloc(&d_noise_vel_, config.num_samples * config.horizon * config.nu * sizeof(float)));
        HANDLE_ERROR(cudaMalloc(&d_perturbed_actions_, config.num_samples * config.horizon * config.nu * sizeof(float)));
        HANDLE_ERROR(cudaMalloc(&d_costs_, config.num_samples * sizeof(float)));
        HANDLE_ERROR(cudaMalloc(&d_initial_state_, config.nx * sizeof(float)));
        HANDLE_ERROR(cudaMalloc(&d_weights_, config.num_samples * sizeof(float)));

        HANDLE_ERROR(cudaMemset(d_u_vel_, 0, config.horizon * config.nu * sizeof(float)));
        HANDLE_ERROR(cudaMemset(d_action_seq_, 0, config.horizon * config.nu * sizeof(float)));

        HANDLE_CURAND_ERROR(curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT));
        HANDLE_CURAND_ERROR(curandSetPseudoRandomGeneratorSeed(gen_, 1234ULL));
    }

    ~SMPPIController() {
        cudaFree(d_u_vel_);
        cudaFree(d_action_seq_);
        cudaFree(d_noise_vel_);
        cudaFree(d_perturbed_actions_);
        cudaFree(d_costs_);
        cudaFree(d_initial_state_);
        cudaFree(d_weights_);
        curandDestroyGenerator(gen_);
    }

    void compute(const Eigen::VectorXf& state) {
        HANDLE_ERROR(cudaMemcpy(d_initial_state_, state.data(), config_.nx * sizeof(float), cudaMemcpyHostToDevice));

        // 1. Sample velocity noise
        HANDLE_CURAND_ERROR(curandGenerateNormal(gen_, d_noise_vel_, config_.num_samples * config_.horizon * config_.nu, 0.0f, 1.0f));

        // 2. Integrate to get perturbed actions
        int num_threads = 256;
        int num_blocks = (config_.num_samples * config_.horizon * config_.nu + num_threads - 1) / num_threads;
        // Need a kernel that does integration: action = action_nom + (u_vel + noise) * dt
        // This is tricky because integration is temporal.
        // Parallelizing over K is easy. Inside K, temporal loop.
        dim3 block(256);
        dim3 grid((config_.num_samples + block.x - 1) / block.x);
        
        integrate_actions_kernel<<<grid, block>>>(
            d_action_seq_,
            d_u_vel_,
            d_noise_vel_,
            d_perturbed_actions_,
            config_
        );
        HANDLE_ERROR(cudaGetLastError());

        // 3. Rollout using perturbed actions
        // We reuse rollout_kernel but treat d_perturbed_actions_ as "noise" added to a zero u_nom, 
        // OR modify rollout_kernel to take explicit actions.
        // Let's assume we pass d_perturbed_actions_ as 'noise' and 0 as 'u_nom'.
        // Wait, rollout_kernel does: u = u_nom + noise.
        // So if we pass u_nom=0 (we need a zero buffer) and noise=d_perturbed_actions_, it works.
        // I need a zero buffer.
        float* d_zeros;
        HANDLE_ERROR(cudaMalloc(&d_zeros, config_.horizon * config_.nu * sizeof(float)));
        HANDLE_ERROR(cudaMemset(d_zeros, 0, config_.horizon * config_.nu * sizeof(float)));

        kernels::rollout_kernel<<<grid, block>>>(
            dynamics_,
            cost_,
            config_,
            d_initial_state_,
            d_zeros,             // u_nom = 0
            d_perturbed_actions_, // acts as full u
            d_costs_
        );
        cudaFree(d_zeros); // Inefficient alloc/free every loop, but okay for prototype.

        // 4. Add smoothness cost
        // Kernel to compute diffs and add to d_costs_
        smoothness_cost_kernel<<<grid, block>>>(
            d_perturbed_actions_,
            d_costs_,
            config_
        );

        // 5. Weights & Update (simplified host side for now)
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
        for(int k=0; k<config_.num_samples; ++k) h_weights[k] /= sum_weights;

        // Update u_vel
        HANDLE_ERROR(cudaMemcpy(d_weights_, h_weights.data(), config_.num_samples * sizeof(float), cudaMemcpyHostToDevice));
        
        int num_params = config_.horizon * config_.nu;
        int blocks_upd = (num_params + 256 - 1) / 256;
        
        weighted_update_kernel<<<blocks_upd, 256>>>(
            d_u_vel_,
            d_noise_vel_,
            d_weights_,
            config_.num_samples,
            num_params 
        );

        // Update action_sequence (integrate new u_vel)
        // Similar to integrate_actions_kernel but for single trajectory (K=1)
        integrate_single_action_kernel<<<1, 1>>>(
            d_action_seq_,
            d_u_vel_,
            config_
        );
    }
    
    // ... shift logic similar to MPPI but for both buffers

private:
    MPPIConfig config_;
    Dynamics dynamics_;
    Cost cost_;

    float* d_u_vel_;
    float* d_action_seq_;
    float* d_noise_vel_;
    float* d_perturbed_actions_;
    
    float* d_costs_;
    float* d_initial_state_;
    float* d_weights_;
    
    curandGenerator_t gen_;
};

// Kernels
__global__ void integrate_actions_kernel(
    const float* action_seq_nom,
    const float* u_vel_nom,
    const float* noise_vel,
    float* perturbed_actions,
    MPPIConfig config
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= config.num_samples) return;

    // Temporal integration for sample k
    // We can't easily parallelize over T because step T depends on T-1.
    // So each thread handles one sample (K).
    
    // Need local copy of action?
    // action[t] = action_nom[t] + (u_vel_nom[t] + noise[k,t])*dt ??
    // No, SMPPI logic:
    // perturbed_action = action_sequence[t-1?] + (u_vel + noise) * dt
    // Ideally it's strictly integration: a[0] = a_init + v[0]dt, a[1] = a[0] + v[1]dt...
    // But JAX SMPPI uses: action_sequence + perturbed_control * dt
    // Wait, JAX code:
    // perturbed_actions = smppi_state.action_sequence[None, :, :] + perturbed_control * config.delta_t
    // This implies action_sequence is already the baseline, and we just add the velocity * dt deviation?
    // Ah, `perturbed_control` is `U + noise`.
    // So `action_new = action_old + (U + noise) * dt`.
    // Yes. It's an Euler step from the nominal action sequence.
    
    for (int t=0; t<config.horizon; ++t) {
        for (int i=0; i<config.nu; ++i) {
            int idx_base = t * config.nu + i;
            int idx_sample = k * (config.horizon * config.nu) + idx_base;
            
            float vel = u_vel_nom[idx_base] + noise_vel[idx_sample];
            // Bound velocity if needed
            
            float act = action_seq_nom[idx_base] + vel * config.dt;
            // Bound action if needed
            
            perturbed_actions[idx_sample] = act;
        }
    }
}

__global__ void integrate_single_action_kernel(
    float* action_seq,
    const float* u_vel,
    MPPIConfig config
) {
    // Single thread
    for (int t=0; t<config.horizon; ++t) {
        for (int i=0; i<config.nu; ++i) {
            int idx = t * config.nu + i;
            action_seq[idx] = action_seq[idx] + u_vel[idx] * config.dt;
        }
    }
}

__global__ void smoothness_cost_kernel(
    const float* perturbed_actions,
    float* costs,
    MPPIConfig config
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= config.num_samples) return;
    
    float cost = 0.0f;
    for (int t=0; t<config.horizon-1; ++t) {
        for (int i=0; i<config.nu; ++i) {
            int idx1 = k * (config.horizon * config.nu) + t * config.nu + i;
            int idx2 = k * (config.horizon * config.nu) + (t+1) * config.nu + i;
            
            float diff = (perturbed_actions[idx2] - perturbed_actions[idx1]) * config.u_scale; // Scale?
            cost += diff * diff;
        }
    }
    costs[k] += cost * config.w_action_seq_cost;
}

} // namespace mppi

#endif // SMPPI_CONTROLLER_CUH
