#include "mppi/jit/jit_compiler.hpp"
#include "mppi/jit/jit_utils.h"
#include <sstream>
#include <iostream>
#include <vector>

namespace mppi {
namespace jit {

std::string JITCompiler::generate_source(const std::string& dynamics_code, const std::string& cost_code) {
    std::stringstream ss;
    // Include the kernel definition. 
    // We assume the include path is set up correctly so "mppi/core/kernels.cuh" is found.
    ss << "#include <mppi/core/kernels.cuh>\n\n";
    
    ss << "// User Dynamics\n";
    ss << dynamics_code << "\n\n";
    
    ss << "// User Cost\n";
    ss << cost_code << "\n\n";
    
    // Wrapper to instantiate the template
    // We expect the user code to define structs named 'UserDynamics' and 'UserCost'
    // The wrapper instantiates them (assuming default constructible) and calls the kernel
    ss << "extern \"C\" __global__ void rollout_wrapper(\n";
    ss << "    mppi::MPPIConfig config,\n";
    ss << "    const float* __restrict__ initial_state,\n";
    ss << "    const float* __restrict__ u_nom,\n";
    ss << "    const float* __restrict__ noise,\n";
    ss << "    float* __restrict__ costs\n";
    ss << ") {\n";
    ss << "    UserDynamics dynamics;\n";
    ss << "    UserCost cost;\n";
    ss << "    mppi::kernels::rollout_kernel(dynamics, cost, config, initial_state, u_nom, noise, costs);\n";
    ss << "}\n";
    
    return ss.str();
}

std::string JITCompiler::compile(const std::string& dynamics_code,
                                 const std::string& cost_code,
                                 const std::vector<std::string>& include_paths) {
    std::string source = generate_source(dynamics_code, cost_code);

    nvrtcProgram prog;
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, source.c_str(), "mppi_jit.cu", 0, NULL, NULL));

    std::vector<const char*> opts;
    // TODO: Detect architecture dynamically or allow passing it
    opts.push_back("--gpu-architecture=compute_75");
    opts.push_back("-std=c++17");
    opts.push_back("-dopt=on"); // Optimization

    // Add include paths
    std::vector<std::string> include_opts;
    include_opts.reserve(include_paths.size() + 2); // Reserve extra for CUDA paths

    // Add user-provided include paths
    for(const auto& path : include_paths) {
        include_opts.push_back("-I" + path);
        opts.push_back(include_opts.back().c_str());
    }

    // Try to add CUDA include path automatically
    const char* cuda_path = std::getenv("CUDA_PATH");
    if (cuda_path) {
        include_opts.push_back(std::string("-I") + cuda_path + "/include");
        opts.push_back(include_opts.back().c_str());
    } else {
        // Try default CUDA locations
        std::vector<std::string> default_cuda_paths = {
            "/usr/local/cuda/include",
            "/usr/local/cuda-12/include",
            "/usr/local/cuda-11/include",
            "/opt/cuda/include"
        };
        for (const auto& path : default_cuda_paths) {
            // Check if directory exists (simple check)
            include_opts.push_back("-I" + path);
            opts.push_back(include_opts.back().c_str());
            break; // Just add the first one
        }
    }
    
    nvrtcResult compileResult = nvrtcCompileProgram(prog, opts.size(), opts.data());
    
    if (compileResult != NVRTC_SUCCESS) {
        size_t logSize;
        NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
        std::vector<char> log(logSize);
        NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log.data()));
        std::cerr << "NVRTC Compilation Failed:\n" << log.data() << std::endl;
        // Print source with line numbers for debugging
        std::cerr << "--- Source Code ---\n";
        std::stringstream ss(source);
        std::string line;
        int lineNum = 1;
        while(std::getline(ss, line)) {
            std::cerr << lineNum++ << ": " << line << "\n";
        }
        std::cerr << "-------------------\n";
        exit(1);
    }
    
    size_t ptxSize;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
    std::vector<char> ptx(ptxSize);
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx.data()));
    
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
    
    return std::string(ptx.begin(), ptx.end());
}

} // namespace jit
} // namespace mppi

