# Deep Hedging with Stochastic Volatility

This project implements a Deep Reinforcement Learning agent for hedging European Options under realistic market frictions and stochastic volatility. The model uses a custom C++ Heston engine for high-performance training and has been validated against out-of-sample data from the S&P 500 during the 2020 Covid-19 market crash.

The agent optimises for **95% Expected Shortfall (CVaR)** to explicitly minimise tail risk in compliance with Basel III standards.

**Results:** The model (trained purely on simulation) was backtested on **S&P 500** data during the March 2020 crash. This model reduced total hedging losses by approximately **25%** compared to the standard Black-Scholes model.

**Future work:**

* GANs: Replace Heston with GANs to learn model-free market dynamics directly from data.

* Transformers: Upgrade the neural architecture to Attention-based models to better capture long-range volatility dependencies.

* Rough Volatility: Extend the engine to simulate "rough" fractional Brownian motion.

[Buehler, H., Gonon, L., Teichmann, J. and Wood, B. (n.d.). DEEP HEDGING. [online] Available at: https://arxiv.org/pdf/1802.03042.]

