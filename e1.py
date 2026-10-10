import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


def compute_return_stats(prices):
    log_returns = np.log(prices / prices.shift(1)).dropna()
    mu = log_returns.mean().values * TRADING_DAYS
    Sigma = log_returns.cov().values * TRADING_DAYS
    tickers = list(prices.columns)
    return mu, Sigma, tickers


def portfolio_stats(w, mu, Sigma):
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ Sigma @ w))
    return ret, vol


def tangency_portfolio(mu, Sigma, rf, long_only=True):
    n = len(mu)
    w0 = np.full(n, 1.0 / n)

    def neg_sharpe(w):
        ret, vol = portfolio_stats(w, mu, Sigma)
        return -(ret - rf) / vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n if long_only else [(-1.0, 1.0)] * n

    result = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x


prices = pd.read_csv("sample_nifty_prices.csv", index_col="Date", parse_dates=True)
mu, Sigma, tickers = compute_return_stats(prices)

rf_values = [0.00, 0.03, 0.06, 0.09]
rows = []
for rf in rf_values:
    w = tangency_portfolio(mu, Sigma, rf)
    ret, vol = portfolio_stats(w, mu, Sigma)
    sharpe = (ret - rf) / vol
    rows.append([rf] + list(np.round(w, 4)) + [round(sharpe, 4)])

columns = ["rf"] + tickers + ["Sharpe"]
table = pd.DataFrame(rows, columns=columns)
print(table.to_string(index=False))
