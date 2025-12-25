#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Engine.h"
#include <random>

namespace py = pybind11;

std::pair<std::vector<double>, std::vector<double>> py_generate_heston_path(double S0, double r, double v0, double kappa, double theta,
                                            double xi, double rho, double T, int steps)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    return generate_heston_path(S0, r, v0, kappa, theta, xi, rho, T, steps, gen);
}

PYBIND11_MODULE(market_engine, m)
{
    m.doc() = "Heston monte carlo engine in C++";

    // m.def("python_name", &cpp_function, "description");
    m.def("generate_path", &py_generate_heston_path, "Simulate heston model path", py::arg("S0"), py::arg("r"), py::arg("v0"), py::arg("kappa"),
          py::arg("theta"), py::arg("xi"), py::arg("rho"), py::arg("T"), py::arg("steps"));
    m.def("call_payoff", &call_payoff, "Calculates option payoff");
}