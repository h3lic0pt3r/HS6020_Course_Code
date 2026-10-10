import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INDEX_COL = "NIFTY_INDEX"


def compute_log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()


def rolling_beta(stock_ret, market_ret, window):
    rolling_cov = stock_ret.rolling(window).cov(market_ret)
    rolling_var = market_ret.rolling(window).var()
    return rolling_cov / rolling_var


prices = pd.read_csv("sample_market_stocks.csv", index_col="Date", parse_dates=True)
log_ret = compute_log_returns(prices)
Rm = log_ret[INDEX_COL]
Rd = log_ret["STOCK_D"]

beta_30 = rolling_beta(Rd, Rm, window=30)
beta_90 = rolling_beta(Rd, Rm, window=90)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(beta_30.index, beta_30.values, label="30-day window", color="#D85A30", lw=1.3)
ax.plot(beta_90.index, beta_90.values, label="90-day window", color="#002D5A", lw=1.8)
ax.set_xlabel("Date")
ax.set_ylabel("Rolling beta")
ax.set_title("STOCK_D Rolling Beta: 30-day vs. 90-day Window")
ax.legend()
ax.grid(alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig("stock_d_rolling_beta.png", dpi=150)

print("30-day window std dev of beta:", beta_30.std())
print("90-day window std dev of beta:", beta_90.std())
print("30-day window min/max:", beta_30.min(), beta_30.max())
print("90-day window min/max:", beta_90.min(), beta_90.max())
