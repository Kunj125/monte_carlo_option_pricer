import torch
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from DeepHedging import HedgingNetwork
import config

def load_model():
    print(f"loading model from {config.MODEL_SAVE_PATH}")
    model = HedgingNetwork(input_dim=4)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    model.eval()
    return model

def run_covid_backtest():    
    try:
        data = yf.download(["SPY", "^VIX"], start="2020-02-01", end="2020-05-01", progress=False)['Close']
        if data.empty:
            raise ValueError("No data downloaded.")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return
    
    data = data.dropna()
    
    prices_real = data["SPY"].values
    vix_real = data["^VIX"].values / 100.0  
    dates = data.index
    
    # normalise to start at 100
    norm_factor = 100.0 / prices_real[0]
    prices_norm = prices_real * norm_factor
    
    steps = len(prices_real)
    # Simulate a 3 month option fading to maturity
    dt = 0.25 / steps 
    
    model = load_model()
    
    cash_ai = 0.0; holdings_ai = 0.0
    cash_bs = 0.0; holdings_bs = 0.0
    
    vol0 = vix_real[0]
    d1 = (np.log(100/100) + (config.R + 0.5*vol0**2)*0.25) / (vol0*np.sqrt(0.25))
    d2 = d1 - vol0*np.sqrt(0.25)
    premium = 100 * torch.distributions.Normal(0,1).cdf(torch.tensor(d1)) - \
              100 * np.exp(-config.R*0.25) * torch.distributions.Normal(0,1).cdf(torch.tensor(d2))
    premium = premium.item()
    
    cash_ai = premium; cash_bs = premium
    
    history_ai = []
    history_bs = []
    
    print("Simulating trading...")
    for t in range(steps):
        p_norm = prices_norm[t]
        v = vix_real[t]
        t_left = 0.25 - (t * dt)
        if t_left < 0: t_left = 0
        
        state = torch.tensor([np.log(p_norm/100), t_left, holdings_ai, v], dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            action_ai = model(state).item()
            
        trade_ai = action_ai - holdings_ai
        cost_ai = abs(trade_ai) * p_norm * 0.005 # transaction costs of 0.5% of traded volume
        cash_ai = cash_ai - (trade_ai * p_norm) - cost_ai
        holdings_ai = action_ai
        
        if t_left > 0.001:
            d1_bs = (np.log(p_norm/100) + (config.R + 0.5*v**2)*t_left) / (v*np.sqrt(t_left))
            action_bs = torch.distributions.Normal(0,1).cdf(torch.tensor(d1_bs)).item()
        else:
            action_bs = 1.0 if p_norm > 100 else 0.0
            
        trade_bs = action_bs - holdings_bs
        cost_bs = abs(trade_bs) * p_norm * 0.005
        cash_bs = cash_bs - (trade_bs * p_norm) - cost_bs
        holdings_bs = action_bs
        
        liability = max(p_norm - 100, 0)
        val_ai = cash_ai + (holdings_ai * p_norm) - liability
        val_bs = cash_bs + (holdings_bs * p_norm) - liability
        
        history_ai.append(val_ai)
        history_bs.append(val_bs)

    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(dates, prices_real, color='blue', label='S&P 500 Price')
    plt.ylabel("Price ($)")
    plt.title("The Scenario: Covid Crash (March 2020)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(dates, history_bs, color='grey', linestyle='--', label='Black-Scholes Benchmark')
    plt.plot(dates, history_ai, color='red', linewidth=2, label='Deep Hedging AI')
    plt.ylabel("Cumulative PnL (Normalised)")
    plt.title("The Result: AI vs BS")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_covid_backtest()