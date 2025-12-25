#include "Engine.h"
#include <cmath>
#include <algorithm>

std::pair<std::vector<double>, std::vector<double>> generate_heston_path(double S0, double r, double v0, double kappa, double theta,
                                                                         double xi, double rho, double T, int steps, std::mt19937 &gen)
{
    double dt = T / steps;
    std::vector<double> path;
    std::vector<double> vols;

    path.reserve(steps + 1); // allocate memory once to avoid re-allocations
    path.push_back(S0);

    vols.reserve(steps + 1);
    vols.push_back(std::sqrt(v0));

    double sqrt_dt = std::sqrt(dt);
    double S_t = S0;
    double v_t = v0;

    std::normal_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < steps; ++i)
    {
        double z1 = dist(gen);
        double z2 = dist(gen);

        // correlate them using cholesky decomposition
        // rho - correlation between stock and variance
        double dW_S = z1;
        double dW_v = rho * z1 + std::sqrt(1.0 - rho * rho) * z2;

        double v_pos = std::max(v_t, 0.0); // prevent NaNs

        // heston model
        // https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/Heston.pdf
        double dv = kappa * (theta - v_pos) * dt + xi * std::sqrt(v_pos) * dW_v * sqrt_dt;
        v_t += dv;

        // Log-Euler
        double drift = (r - 0.5 * v_pos) * dt;
        double diffusion = std::sqrt(v_pos) * dW_S * sqrt_dt;

        double S_next = S_t * std::exp(drift + diffusion);

        S_t = S_next;
        path.push_back(S_t);
        vols.push_back(std::sqrt(std::max(v_t, 0.0)));
    }

    return {path, vols};
}

double call_payoff(double S_T, double K)
{
    return std::max(S_T - K, 0.0);
}