"""
MA5950 -- Mathematical Finance: Theory & Implementation
Week 3 Lab: Quadratic Programming for Portfolio Optimization
-----------------------------------------------------------------------
STUDENT STARTER FILE

Fill in every function marked TODO. Data loading and the main() driver
are provided so you can run the pipeline after each function you
complete and see immediate feedback.

Read Lectures 7-9 before you start -- every formula you need is stated
there, and repeated as a hint in the relevant docstring below.

Run with:  python portfolio_lab_starter.py
Needs 'sample_nifty_prices.csv' in the same folder as an offline
fallback dataset (used automatically if yfinance / internet access
isn't available).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

SAMPLE_CSV_PATH = "sample_nifty_prices.csv"
TICKERS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "ITC"]
TRADING_DAYS = 252


# =====================================================================
#  PROVIDED -- data loading. You do not need to modify this function.
# =====================================================================
def load_price_data(use_live=True, period="2y"):
    """Return a DataFrame of daily close prices, columns = tickers.

    Tries a live fetch via yfinance (if installed and network is
    available); falls back to the bundled sample CSV otherwise, so the
    lab always runs.
    """
    if use_live:
        try:
            import yfinance as yf
            nse_tickers = [t + ".NS" for t in TICKERS]
            data = yf.download(nse_tickers, period=period, progress=False)["Close"]
            data.columns = TICKERS
            print(f"Loaded LIVE price data: {data.shape[0]} days for {len(TICKERS)} stocks")
            return data.dropna()
        except Exception as exc:
            print(f"Live fetch failed ({exc!r}); falling back to sample CSV.")

    data = pd.read_csv(SAMPLE_CSV_PATH, index_col="Date", parse_dates=True)
    print(f"Loaded SAMPLE (offline) price data: {data.shape[0]} days for {data.shape[1]} stocks")
    return data


# =====================================================================
#  TODO 1 -- compute annualized mean vector and covariance matrix from
#  daily prices.
# =====================================================================
def compute_return_stats(prices):
    """
    Parameters
    ----------
    prices : DataFrame of daily close prices, one column per stock

    Returns
    -------
    mu : np.array, ANNUALIZED mean log return for each stock
         (hint: daily mean * 252)
    Sigma : np.array, ANNUALIZED covariance matrix of log returns
         (hint: daily covariance matrix * 252)
    tickers : list of column names, in the same order as mu/Sigma

    Hint: log return_t = ln(P_t / P_{t-1}). Use prices.shift(1) and
    np.log(), then .dropna() to discard the first (undefined) row.
    """
    # TODO: implement as described above
    raise NotImplementedError("TODO 1: implement compute_return_stats")


# =====================================================================
#  PROVIDED -- simple helper, no need to modify.
# =====================================================================
def portfolio_stats(w, mu, Sigma):
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ Sigma @ w))
    return ret, vol


# =====================================================================
#  TODO 2 -- Minimum Variance Portfolio, as a quadratic program.
#
#  Recall from Lecture 8:
#      min_w  w' Sigma w    s.t.  sum(w) = 1   (and w_i >= 0 if long_only)
#
#  Use scipy.optimize.minimize with method="SLSQP". You need:
#    - an objective function: lambda w: w @ Sigma @ w
#    - an equality constraint: {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
#    - bounds: (0.0, 1.0) per weight if long_only, else something wider
#    - a starting guess w0 (equal weights is a reasonable default)
# =====================================================================
def minimum_variance_portfolio(Sigma, long_only=True):
    """Return the weight vector w (np.array) minimizing w'Sigma w
    subject to sum(w)=1 (and w>=0 if long_only)."""
    n = Sigma.shape[0]
    w0 = np.full(n, 1.0 / n)
    # TODO: set up constraints, bounds, and call scipy.optimize.minimize
    raise NotImplementedError("TODO 2: implement minimum_variance_portfolio")


# =====================================================================
#  TODO 3 -- THE CORE OF THE LAB. Trace the entire efficient frontier
#  by solving the Markowitz QP for a whole range of target returns.
#
#  Recall from Lecture 8:
#      min_w  w' Sigma w   s.t.  sum(w) = 1,  w' mu = target
#  for each target in a grid spanning [min(mu), max(mu)].
# =====================================================================
def efficient_frontier(mu, Sigma, n_points=40, long_only=True):
    """
    Returns
    -------
    targets : np.array of target returns tried (length n_points)
    vols : np.array of the corresponding minimized portfolio volatility
    weights_list : list of weight vectors, one per target return

    Hint: loop over `targets = np.linspace(mu.min(), mu.max(), n_points)`.
    For each target, solve the QP above (same idea as TODO 2, but with
    an EXTRA equality constraint w @ mu == target). Store
    np.sqrt(result.fun) as that target's volatility.
    """
    n = len(mu)
    targets = np.linspace(mu.min(), mu.max(), n_points)
    vols, weights_list = [], []
    # TODO: loop over targets, solve the constrained QP for each, and
    #       append the resulting vol and weight vector
    raise NotImplementedError("TODO 3: implement efficient_frontier")


# =====================================================================
#  TODO 4 -- Tangency portfolio: maximize the Sharpe ratio.
#
#  Recall from Lecture 9:
#      max_w  (w'mu - rf) / sqrt(w'Sigma w)   s.t. sum(w) = 1
#  scipy only MINIMIZES, so minimize the NEGATIVE Sharpe ratio instead.
# =====================================================================
def tangency_portfolio(mu, Sigma, rf, long_only=True):
    """Return the weight vector w maximizing the Sharpe ratio."""
    n = len(mu)
    w0 = np.full(n, 1.0 / n)
    # TODO: define neg_sharpe(w) using portfolio_stats(), set up the
    #       sum(w)=1 constraint and bounds, and call scipy.optimize.minimize
    raise NotImplementedError("TODO 4: implement tangency_portfolio")


# =====================================================================
#  PROVIDED -- driver. Once TODOs 1-4 are done, just run this file.
# =====================================================================
def main():
    rf = 0.06  # 6% annual risk-free rate, illustrative

    prices = load_price_data(use_live=True)
    mu, Sigma, tickers = compute_return_stats(prices)

    print("\nAnnualized mean returns:")
    for t, m in zip(tickers, mu):
        print(f"  {t:<10} {m*100:>7.2f}%")
    print("\nAnnualized volatilities (from Sigma diagonal):")
    for t, v in zip(tickers, np.sqrt(np.diag(Sigma))):
        print(f"  {t:<10} {v*100:>7.2f}%")

    w_mvp = minimum_variance_portfolio(Sigma)
    ret_mvp, vol_mvp = portfolio_stats(w_mvp, mu, Sigma)
    print(f"\nMinimum Variance Portfolio: return={ret_mvp*100:.2f}%, vol={vol_mvp*100:.2f}%")
    print("Weights:", dict(zip(tickers, np.round(w_mvp, 3))))

    targets, vols, weights_list = efficient_frontier(mu, Sigma, n_points=40)

    w_tan = tangency_portfolio(mu, Sigma, rf)
    ret_tan, vol_tan = portfolio_stats(w_tan, mu, Sigma)
    sharpe_tan = (ret_tan - rf) / vol_tan
    print(f"\nTangency Portfolio: return={ret_tan*100:.2f}%, vol={vol_tan*100:.2f}%, Sharpe={sharpe_tan:.3f}")
    print("Weights:", dict(zip(tickers, np.round(w_tan, 3))))

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.plot(vols * 100, targets * 100, "-", color="#002D5A", lw=2, label="Efficient frontier")
    ax.scatter(np.sqrt(np.diag(Sigma)) * 100, mu * 100, color="#5F5E5A", s=40, zorder=5, label="Individual stocks")
    for t, s, m in zip(tickers, np.sqrt(np.diag(Sigma)), mu):
        ax.annotate(t, (s * 100, m * 100), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.scatter([vol_mvp * 100], [ret_mvp * 100], color="#1D9E75", s=90, marker="D", zorder=6, label="Min. variance portfolio")
    ax.scatter([vol_tan * 100], [ret_tan * 100], color="#D85A30", s=90, marker="^", zorder=6, label="Tangency portfolio")

    cal_x = np.linspace(0, vol_tan * 100 * 2.2, 20)
    cal_y = rf * 100 + sharpe_tan * cal_x
    ax.plot(cal_x, cal_y, "--", color="#D85A30", lw=1.5, label="Capital market line")

    ax.set_xlabel("Risk (annualized std. dev., %)")
    ax.set_ylabel("Expected return (annualized, %)")
    ax.set_title("Efficient Frontier -- 6-Stock Universe")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("efficient_frontier.png", dpi=150)
    print("\nSaved efficient_frontier.png")


if __name__ == "__main__":
    main()
