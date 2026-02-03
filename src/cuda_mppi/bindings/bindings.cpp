#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "mppi/core/mppi_common.cuh"

namespace nb = nanobind;
using namespace mppi;

NB_MODULE(cuda_mppi, m) {
    nb::class_<MPPIConfig>(m, "MPPIConfig")
        .def(nb::init<int, int, int, int, float, float, float, float, int>(),
             nb::arg("num_samples"),
             nb::arg("horizon"),
             nb::arg("nx"),
             nb::arg("nu"),
             nb::arg("lambda"),
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
}
