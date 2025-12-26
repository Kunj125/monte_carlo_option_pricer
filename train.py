import plot
import torch
import torch.optim as optim
import numpy as np
from DeepHedging import HedgingNetwork
import market_engine
import config
import os


def train():
    model = HedgingNetwork(4)
    learning_rate = 0.005
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # drop lr by half every 500 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    for i in range(config.EPOCHS):
        paths = []
        vol_list = []
        start_prices = []
        for _ in range(config.BATCH_SIZE):
            rand_S0 = config.S0 * (0.8 + 0.4 * torch.rand(1).item())

            prices, vols = market_engine.generate_path(
                rand_S0, config.R, config.V0, config.KAPPA, config.THETA, config.XI, config.RHO, config.T, config.STEPS)
            paths.append(prices)
            vol_list.append(vols)
            start_prices.append(rand_S0)

        price_paths = torch.tensor(paths, dtype=torch.float32)
        vol_paths = torch.tensor(vol_list, dtype=torch.float32)
        S0_vec = torch.tensor(start_prices, dtype=torch.float32).unsqueeze(1)

        # bs approximation using sqrt(V0) as a proxy for sigma to get a rough fair value
        approx_sigma = np.sqrt(config.V0)
        d1 = (torch.log(S0_vec / config.K) + (config.R + 0.5 * approx_sigma**2) * config.T) / \
            (approx_sigma * np.sqrt(config.T))
        d2 = d1 - approx_sigma * np.sqrt(config.T)
        bs_price = S0_vec * torch.distributions.Normal(0, 1).cdf(d1) - \
            config.K * np.exp(-config.R * config.T) * \
            torch.distributions.Normal(0, 1).cdf(d2)

        cash = bs_price.squeeze()
        holdings = torch.zeros(config.BATCH_SIZE)
        dt = config.T / config.STEPS
        for t in range(config.STEPS):
            cur_prices = price_paths[:, t]
            cur_vols = vol_paths[:, t]
            time_left = config.T - (t * dt)

            log_price = torch.log(cur_prices / config.K).unsqueeze(1)

            time_left_vec = torch.full((config.BATCH_SIZE, 1), time_left)
            holdings_vec = holdings.unsqueeze(1)
            vol_vec = cur_vols.unsqueeze(1)
            state = torch.cat(
                (log_price, time_left_vec, holdings_vec, vol_vec), 1)

            target_holdings = model(state).squeeze()

            trade_amount = target_holdings - holdings

            transaction_cost = torch.abs(trade_amount) * cur_prices * 0.005
            execution_price = trade_amount * cur_prices
            cash = cash - transaction_cost - execution_price
            holdings = target_holdings

        final_prices = price_paths[:, -1]
        portfolio_value = cash + (holdings * final_prices)
        payoff = torch.relu(final_prices - config.K)

        pnl = portfolio_value - payoff

        loss = torch.log(torch.mean(torch.exp(-config.RISK_AVERSION * pnl)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {i}: Loss = {loss.item():.4f} | LR = {current_lr:.6f}")

    print("Training finished")

    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Model weights saved to {config.MODEL_SAVE_PATH}")
    return model


def load_model(path=None):
    if path is None:
        path = config.MODEL_SAVE_PATH

    print(f"Loading model from {path}...")
    model = HedgingNetwork(input_dim=4)

    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        model.eval()
        print("Model loaded successfully")
        return model
    else:
        raise FileNotFoundError(f"No model found at {path}")


if __name__ == "__main__":
    trained_model = train()
    plot.plot_strategy(model=trained_model)
    plot.plot_pnl_distribution(trained_model)
    plot.plot_time_dynamics(model=trained_model)
