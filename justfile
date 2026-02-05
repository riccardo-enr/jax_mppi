# JAX-MPPI Examples Runner
# Run quadrotor comparison examples

# Default recipe - show available commands
default:
    @just --list

# Run quadrotor hover comparison
hover:
    uv run python examples/quadrotor_hover_comparison.py --visualize

# Run quadrotor figure-8 comparison
figure8:
    uv run python examples/quadrotor_figure8_comparison.py --visualize

# Run all quadrotor comparisons
all: hover figure8

# Run CUDA JIT pendulum example
jit-pendulum:
    @export CUDA_MPPI_INCLUDE_DIR=$(pwd)/third_party/cuda-mppi/include && \
    uv run python examples/cuda_pendulum_jit.py --visualization

# Run hover comparison with custom parameters
hover-custom steps samples horizon lambda:
    uv run python examples/quadrotor_hover_comparison.py \
        --steps {{ steps }} \
        --samples {{ samples }} \
        --horizon {{ horizon }} \
        --lambda {{ lambda }} \
        --visualize

# Run figure-8 comparison with custom parameters
figure8-custom steps samples horizon lambda:
    uv run python examples/quadrotor_figure8_comparison.py \
        --steps {{ steps }} \
        --samples {{ samples }} \
        --horizon {{ horizon }} \
        --lambda {{ lambda }} \
        --visualize

# Test basic CUDA bindings
test-cuda:
    uv run python examples/test_cuda_mppi.py

# Run tests
test:
    uv run pytest tests/

# Initialize and update git submodules (cuda-mppi)
submodule-update:
    git submodule update --init --recursive

# Clean generated media files
clean:
    rm -f docs/media/quadrotor_*_comparison.png

# Show help for hover comparison
help-hover:
    uv run python examples/quadrotor_hover_comparison.py --help

# Show help for figure8 comparison
help-figure8:
    uv run python examples/quadrotor_figure8_comparison.py --help

# Documentation
quarto-doc:
    uv run quarto preview docs

# Publish documentation to GitHub Pages manually
publish-doc:
    cd docs && quarto publish gh-pages --no-browser --no-prompt
