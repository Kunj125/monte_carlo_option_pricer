import market_engine
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class TradingEnvironment:
    def __init__(self, s0=100, r=0.05, sigma=0.2, T=1.0, steps=100, transaction_cost=0.01):
        self.s0 = s0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.steps = steps
        self.dt = T / steps
        self.cost_pct = transaction_cost  # 1% fee per trade -> high, to force ai to learn

        self.path = []
        self.current_step = 0
        self.current_holdings = 0.0
        self.cash = 0.0

    def reset(self):
        self.path = market_engine.generate_path(
            self.s0, self.r, self.sigma, self.T, self.steps)

        self.current_step = 0
        self.current_holdings = 0.0
        self.cash = 0.0

        return self._get_state()

    def step(self, action):
        """
        action: the new target hedge ratio (0.5 means "I want to own 0.5 shares")
        """
        current_price = self.path[self.current_step]

        trade_amount = action - self.current_holdings
        cost = abs(trade_amount) * current_price * self.cost_pct

        # update portfolio
        self.current_holdings = action
        self.cash -= cost  # self-financing portfolio

        self.current_step += 1
        done = (self.current_step >= self.steps)

        # sparse reward calc
        reward = 0.0
        if done:
            reward = self._calculate_final_pnl()

        return self._get_state(), reward, done

    def _get_state(self):
        if self.current_step >= len(self.path):
            price = self.path[-1]
        else:
            price = self.path[self.current_step]

        # normalise
        log_price = np.log(price / self.s0)
        time_left = self.T - (self.current_step * self.dt)

        return torch.tensor([log_price, time_left, self.current_holdings], dtype=torch.float32)

    def _calculate_final_pnl(self):
        final_price = self.path[-1]
        strike = self.s0

        payoff_liability = max(final_price - strike, 0.0)

        portfolio_value = self.cash + (self.current_holdings * final_price)

        return portfolio_value - payoff_liability


class HedgingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # input: 3 neurons (log_price, time_left, current_holdings)
        # hidden: 32 neurons
        # output: 1 neuron (target delta)
        self.model = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, state):
        return self.model(state)
