#ifndef MPPI_JIT_COMPILER_HPP
#define MPPI_JIT_COMPILER_HPP

#include <cuda.h>
#include <string>
#include <vector>

namespace mppi {
namespace jit {

class JITCompiler {
public:
  // Compiles the dynamics and cost into a PTX string
  // Returns the PTX string
  static std::string compile(const std::string &dynamics_code,
                             const std::string &cost_code,
                             const std::vector<std::string> &include_paths);

  // Helper to get the mangled name of the kernel for the specific types
  // Since we use a fixed wrapper name in generate_source, this might be
  // constant "rollout_wrapper"
  static std::string get_wrapper_name() { return "rollout_wrapper"; }

private:
  static std::string generate_source(const std::string &dynamics_code,
                                     const std::string &cost_code);
};

} // namespace jit
} // namespace mppi

#endif // MPPI_JIT_COMPILER_HPP
