#ifndef ENGINE_H
#define ENGINE_H

#include <vector>
#include <random>

double call_payoff(double S_T, double K);

std::vector<double> generate_heston_path(
    double S0, double r, double v0, double kappa, double theta,
    double xi, double rho, double T, int steps, std::mt19937 &gen);

#endif