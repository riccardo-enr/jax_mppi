#define NB_CUDA
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/eigen/dense.h> // Enable Eigen <-> Numpy conversion

#include "mppi/core/mppi_common.cuh"
#include "mppi/controllers/mppi.cuh"
#include "mppi/controllers/smppi.cuh"
#include "mppi/controllers/kmppi.cuh"
#include "mppi/instantiations/double_integrator.cuh"

namespace nb = nanobind;
using namespace mppi;

NB_MODULE(cuda_mppi, m) {
    // 1. MPPIConfig
    nb::class_<MPPIConfig>(m, "MPPIConfig")
        .def(nb::init<int, int, int, int, float, float, float, float, int>(),
             nb::arg("num_samples"),
             nb::arg("horizon"),
             nb::arg("nx"),
             nb::arg("nu"),
             nb::arg("lambda_"),
             nb::arg("dt"),
             nb::arg("u_scale"),
             nb::arg("w_action_seq_cost"),
             nb::arg("num_support_pts"))
        .def_rw("num_samples", &MPPIConfig::num_samples)
        .def_rw("horizon", &MPPIConfig::horizon)
        .def_rw("nx", &MPPIConfig::nx)
        .def_rw("nu", &MPPIConfig::nu)
        .def_rw("lambda", &MPPIConfig::lambda)
        .def_rw("dt", &MPPIConfig::dt)
        .def_rw("u_scale", &MPPIConfig::u_scale)
        .def_rw("w_action_seq_cost", &MPPIConfig::w_action_seq_cost)
        .def_rw("num_support_pts", &MPPIConfig::num_support_pts)
        .def("__repr__", [](const MPPIConfig &c) {
            return "MPPIConfig(num_samples=" + std::to_string(c.num_samples) +
                   ", horizon=" + std::to_string(c.horizon) +
                   ", nx=" + std::to_string(c.nx) +
                   ", nu=" + std::to_string(c.nu) + ")";
        });

    // 2. Concrete Dynamics/Cost Instantiations
    nb::class_<instantiations::DoubleIntegrator>(m, "DoubleIntegrator")
        .def(nb::init<>());
    
    nb::class_<instantiations::QuadraticCost>(m, "QuadraticCost")
        .def(nb::init<>());

    using DIMPPI = MPPIController<instantiations::DoubleIntegrator, instantiations::QuadraticCost>;
    using DISMPPI = SMPPIController<instantiations::DoubleIntegrator, instantiations::QuadraticCost>;
    using DIKMPPI = KMPPIController<instantiations::DoubleIntegrator, instantiations::QuadraticCost>;

    // 3. Bind Controllers
    
    // DoubleIntegratorMPPI
    nb::class_<DIMPPI>(m, "DoubleIntegratorMPPI")
        .def(nb::init<const MPPIConfig&, const instantiations::DoubleIntegrator&, const instantiations::QuadraticCost&>(),
             nb::arg("config"),
             nb::arg("dynamics") = instantiations::DoubleIntegrator(),
             nb::arg("cost") = instantiations::QuadraticCost())
        .def("compute", &DIMPPI::compute, nb::arg("state"), "Compute control update based on current state")
        .def("get_action", &DIMPPI::get_action, "Get the current optimal action")
        .def("shift", &DIMPPI::shift, "Shift the nominal trajectory forward");

    // DoubleIntegratorSMPPI
    nb::class_<DISMPPI>(m, "DoubleIntegratorSMPPI")
        .def(nb::init<const MPPIConfig&, const instantiations::DoubleIntegrator&, const instantiations::QuadraticCost&>(),
             nb::arg("config"),
             nb::arg("dynamics") = instantiations::DoubleIntegrator(),
             nb::arg("cost") = instantiations::QuadraticCost())
        .def("compute", &DISMPPI::compute, nb::arg("state"))
        .def("get_action", &DISMPPI::get_action);

    // DoubleIntegratorKMPPI
    nb::class_<DIKMPPI>(m, "DoubleIntegratorKMPPI")
        .def(nb::init<const MPPIConfig&, const instantiations::DoubleIntegrator&, const instantiations::QuadraticCost&>(),
             nb::arg("config"),
             nb::arg("dynamics") = instantiations::DoubleIntegrator(),
             nb::arg("cost") = instantiations::QuadraticCost())
        .def("compute", &DIKMPPI::compute, nb::arg("state"))
        .def("get_action", &DIKMPPI::get_action);
}