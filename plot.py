
import matplotlib.pyplot as plt
import config
import numpy as np
import torch
import market_engine


def plot_strategy(model):
    print("Generating Strategy Plot...")
    model.eval()

    prices = np.linspace(80, 120, 100)

    time_left = 0.5  # halfway to expiry
    current_holding = 0.5  # already own 0.5 shares
    approx_sigma = np.sqrt(config.V0)
    current_vol = np.sqrt(config.V0)
    ai_deltas = []
    bs_deltas = []

    for p in prices:
        log_p = np.log(p / config.S0)
        inp = torch.tensor(
            [log_p, time_left, current_holding, current_vol], dtype=torch.float32)
        state = inp.unsqueeze(0)

        ai_action = model(state).item()
        ai_deltas.append(ai_action)

        # black-scholes answer
        d1 = (np.log(p/config.K) + (config.R + 0.5*approx_sigma**2)*time_left) / \
            (approx_sigma*np.sqrt(time_left))
        bs_action = torch.distributions.Normal(
            0, 1).cdf(torch.tensor(d1)).item()
        bs_deltas.append(bs_action)

    plt.figure(figsize=(10, 6))
    plt.plot(prices, bs_deltas, label='BS benchmark with constant vol',
             color='black', linestyle='--')
    plt.plot(prices, ai_deltas, label='Heston AI (costs + stochastic vol)',
             color='red', linewidth=2)

    plt.title(
        f"Heston hedging vs Black_Scholes (Time Left = {time_left:.1f}y)")
    plt.xlabel("Stock Price")
    plt.ylabel("Hedge Ratio (Shares Held)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_pnl_distribution(model):
    print("Generating PnL Distribution...")
    paths_list = []
    vols_list = []

    for _ in range(config.BATCH_SIZE):
        p, v = market_engine.generate_path(
            config.S0, config.R, config.V0, config.KAPPA, config.THETA, config.XI, config.RHO, config.T, config.STEPS)
        paths_list.append(p)
        vols_list.append(v)

    price_paths = torch.tensor(paths_list, dtype=torch.float32)
    vol_paths = torch.tensor(vols_list, dtype=torch.float32)
    dt = config.T / config.STEPS

    cash_ai = torch.zeros(config.BATCH_SIZE)
    holdings_ai = torch.zeros(config.BATCH_SIZE)

    cash_bs = torch.zeros(config.BATCH_SIZE)
    holdings_bs = torch.zeros(config.BATCH_SIZE)
    approx_sigma = np.sqrt(config.V0)

    d1 = (np.log(config.S0/config.K) + (config.R + 0.5*approx_sigma**2)*config.T) / \
        (approx_sigma*np.sqrt(config.T))
    d2 = d1 - approx_sigma*np.sqrt(config.T)
    premium = config.S0 * torch.distributions.Normal(0, 1).cdf(torch.tensor(d1)) - \
        config.K * np.exp(-config.R*config.T) * torch.distributions.Normal(0,
                                                                           1).cdf(torch.tensor(d2))

    cash_ai[:] = premium
    cash_bs[:] = premium

    for t in range(config.STEPS):
        cur_prices = price_paths[:, t]
        cur_vols = vol_paths[:, t]
        time_left = config.T - (t * dt)

        log_price = torch.log(cur_prices / config.K).unsqueeze(1)
        time_vec = torch.full((config.BATCH_SIZE, 1), time_left)
        h_vec = holdings_ai.unsqueeze(1)
        vol_vec = cur_vols.unsqueeze(1)

        state = torch.cat((log_price, time_vec, h_vec, vol_vec), 1)

        with torch.no_grad():
            target_ai = model(state).squeeze()

        trade_ai = target_ai - holdings_ai
        cost_ai = torch.abs(trade_ai) * cur_prices * 0.005
        cash_ai = cash_ai - (trade_ai * cur_prices) - cost_ai
        holdings_ai = target_ai

        # bs benchmark
        if time_left > 0.001:
            d1_val = (torch.log(cur_prices/config.K) + (config.R + 0.5*approx_sigma**2)*time_left) / \
                     (approx_sigma*np.sqrt(time_left))
            target_bs = torch.distributions.Normal(0, 1).cdf(d1_val)
        else:
            target_bs = (cur_prices > config.K).float()

        trade_bs = target_bs - holdings_bs
        cost_bs = torch.abs(trade_bs) * cur_prices * 0.005
        cash_bs = cash_bs - (trade_bs * cur_prices) - cost_bs
        holdings_bs = target_bs

    final_prices = price_paths[:, -1]
    payoff = torch.relu(final_prices - config.K)

    pnl_ai = cash_ai + (holdings_ai * final_prices) - payoff
    pnl_bs = cash_bs + (holdings_bs * final_prices) - payoff

    plt.figure(figsize=(10, 6))
    plt.hist(pnl_bs.numpy(), bins=50, alpha=0.5,
             label='Black-Scholes', color='black', range=(-20, 10))
    plt.hist(pnl_ai.numpy(), bins=50, alpha=0.6,
             label='Deep Hedging AI (Vol-Aware)', color='red', range=(-20, 10))
    plt.axvline(pnl_bs.mean(), color='black', linestyle='dashed')
    plt.axvline(pnl_ai.mean(), color='red', linestyle='dashed')
    plt.title("PnL Distribution: AI (Vol-Aware) vs BS")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_time_dynamics(model):
    print("Generating Time Dynamics Plot...")
    prices = np.linspace(80, 120, 100)
    current_holding = 0.5
    avg_vol = np.sqrt(config.V0)

    plt.figure(figsize=(10, 6))

    # T = 0.9
    deltas_long = []
    for p in prices:
        state = torch.tensor(
            [np.log(p/config.K), 0.9, current_holding, avg_vol], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            deltas_long.append(model(state).item())
    plt.plot(prices, deltas_long, label='Time Left = 0.9y',
             color='blue', linestyle='--')

    # T= 0.1
    deltas_short = []
    for p in prices:
        state = torch.tensor(
            [np.log(p/config.K), 0.1, current_holding, avg_vol], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            deltas_short.append(model(state).item())
    plt.plot(prices, deltas_short, label='Time Left = 0.1y',
             color='red', linewidth=2)

    plt.title("AI Strategy Evolution (Vol-Aware)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
