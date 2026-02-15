import numpy as np

try:
    from jax_mppi import cuda_mppi  # type: ignore
except ImportError:
    cuda_mppi = None


def main():
    print("Testing CUDA MPPI Bindings...")

    # 1. Create Config
    config = cuda_mppi.MPPIConfig(
        num_samples=100,
        horizon=30,
        nx=4,
        nu=2,
        lambda_=1.0,
        dt=0.05,
        u_scale=1.0,
        w_action_seq_cost=0.0,
        num_support_pts=10,
    )
    print(f"Config created: {config}")

    # 2. Instantiate Controllers
    print("\nInstantiating Controllers...")
    mppi_ctrl = cuda_mppi.DoubleIntegratorMPPI(config)
    smppi_ctrl = cuda_mppi.DoubleIntegratorSMPPI(config)
    kmppi_ctrl = cuda_mppi.DoubleIntegratorKMPPI(config)
    print("Controllers instantiated.")

    # 3. Test Compute loop
    state = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    print(f"\nInitial State: {state}")

    # MPPI
    print("\n--- MPPI ---")
    mppi_ctrl.compute(state)
    action = mppi_ctrl.get_action()
    print(f"Computed Action: {action}")
    mppi_ctrl.shift()
    print("Shifted.")

    # SMPPI
    print("\n--- SMPPI ---")
    smppi_ctrl.compute(state)
    action = smppi_ctrl.get_action()
    print(f"Computed Action: {action}")

    # KMPPI
    print("\n--- KMPPI ---")
    kmppi_ctrl.compute(state)
    action = kmppi_ctrl.get_action()
    print(f"Computed Action: {action}")

    print("\nTest passed!")


if __name__ == "__main__":
    main()
