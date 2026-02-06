
from jax_mppi import cuda_mppi
import numpy as np

def main():
    print("Testing CUDA MPPI Controller Binding...")
    config = cuda_mppi.MPPIConfig(
        num_samples=100,
        horizon=30,
        nx=4,
        nu=2,
        lambda_=1.0,
        dt=0.05,
        u_scale=1.0,
        w_action_seq_cost=0.0,
        num_support_pts=10
    )
    print(f"Config created: {config}")

    print("Instantiating DoubleIntegratorMPPI...")
    # Default dynamics/cost
    mppi = cuda_mppi.DoubleIntegratorMPPI(config)
    print("Success!")
    
    state = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    mppi.compute(state)
    action = mppi.get_action()
    print(f"Action: {action}")

if __name__ == "__main__":
    main()
