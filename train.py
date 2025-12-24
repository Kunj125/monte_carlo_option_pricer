import torch
import torch.optim as optim
import numpy as np
from DeepHedging import HedgingNetwork
import market_engine
import matplotlib.pyplot as plt

S0 = 100.0
K = 100.0
R = 0.05
# SIGMA = 0.2
T = 1.0
STEPS = 50

# heston params
V0 = 0.04
KAPPA = 2.0
THETA = 0.04
XI = 0.3  # volat of volat
RHO = -0.7  # Correlation (stock down -> volat up)

BATCH_SIZE = 64
EPOCHS = 2000
RISK_AVERSION = 1.0


def train():
    model = HedgingNetwork()
    learning_rate = 0.005
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # drop lr by half every 500 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    for i in range(EPOCHS):
        paths = []
        start_prices = []
        for _ in range(BATCH_SIZE):
            rand_S0 = S0 * (0.8 + 0.4 * torch.rand(1).item())

            res = market_engine.generate_path(
                rand_S0, R, V0, KAPPA, THETA, XI, RHO, T, STEPS)
            paths.append(res)
            start_prices.append(rand_S0)

        price_paths = torch.tensor(paths, dtype=torch.float32)
        S0_vec = torch.tensor(start_prices, dtype=torch.float32).unsqueeze(1)

        # bs approximation using sqrt(V0) as a proxy for sigma to get a rough fair value
        approx_sigma = np.sqrt(V0)
        d1 = (torch.log(S0_vec / K) + (R + 0.5 * approx_sigma**2) * T) / \
            (approx_sigma * np.sqrt(T))
        d2 = d1 - approx_sigma * np.sqrt(T)
        bs_price = S0_vec * torch.distributions.Normal(0, 1).cdf(d1) - \
            K * np.exp(-R * T) * torch.distributions.Normal(0, 1).cdf(d2)

        cash = bs_price.squeeze()
        holdings = torch.zeros(BATCH_SIZE)
        dt = T / STEPS
        for t in range(STEPS):
            cur_prices = price_paths[:, t]
            time_left = T - (t * dt)

            log_price = torch.log(cur_prices / K).unsqueeze(1)

            time_left_vec = torch.full((BATCH_SIZE, 1), time_left)
            holdings_vec = holdings.unsqueeze(1)

            state = torch.cat((log_price, time_left_vec, holdings_vec), 1)

            target_holdings = model(state).squeeze()

            trade_amount = target_holdings - holdings

            transaction_cost = torch.abs(trade_amount) * cur_prices * 0.005
            execution_price = trade_amount * cur_prices
            cash = cash - transaction_cost - execution_price
            holdings = target_holdings

        final_prices = price_paths[:, -1]
        portfolio_value = cash + (holdings * final_prices)
        payoff = torch.relu(final_prices - K)

        pnl = portfolio_value - payoff

        loss = torch.log(torch.mean(torch.exp(-RISK_AVERSION * pnl)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {i}: Loss = {loss.item():.4f} | LR = {current_lr:.6f}")

    print("Training finished")
    return model


def plot_strategy(model):
    print("Generating Strategy Plot...")
    model.eval()

    prices = np.linspace(80, 120, 100)

    time_left = 0.5  # halfway to expiry
    current_holding = 0.5  # already own 0.5 shares
    approx_sigma = np.sqrt(V0)
    ai_deltas = []
    bs_deltas = []

    for p in prices:
        log_p = np.log(p / S0)
        inp = torch.tensor(
            [log_p, time_left, current_holding], dtype=torch.float32)
        state = inp.unsqueeze(0)

        ai_action = model(state).item()
        ai_deltas.append(ai_action)

        # black-scholes answer
        d1 = (np.log(p/K) + (R + 0.5*approx_sigma**2)*time_left) / \
            (approx_sigma*np.sqrt(time_left))
        bs_action = torch.distributions.Normal(
            0, 1).cdf(torch.tensor(d1)).item()
        bs_deltas.append(bs_action)

    plt.figure(figsize=(10, 6))
    plt.plot(prices, bs_deltas, label='BS benchmark with constant vol',
             color='black', linestyle='--')
    plt.plot(prices, ai_deltas, label='Heston AI (costs + stochastic vol)',
             color='red', linewidth=2)

    plt.title(f"Heston hedging vs Black_Scholes (Time Left = {time_left:.1f}y)")
    plt.xlabel("Stock Price")
    plt.ylabel("Hedge Ratio (Shares Held)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    trained_model = train()
    plot_strategy(trained_model)
