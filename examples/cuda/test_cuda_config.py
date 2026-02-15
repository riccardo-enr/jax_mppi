try:
    from jax_mppi import cuda_mppi  # type: ignore
except ImportError:
    cuda_mppi = None


def main():
    print("Testing CUDA MPPI Config...")
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
    print("Success!")


if __name__ == "__main__":
    main()
