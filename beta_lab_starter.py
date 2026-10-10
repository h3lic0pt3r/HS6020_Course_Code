"""
MA5950 -- Mathematical Finance: Theory & Implementation
Week 4 Lab: Rolling Beta Regression and Systemic Risk Analysis
-----------------------------------------------------------------------
STUDENT STARTER FILE

Fill in every function marked TODO. Data loading and the main() driver
are provided so you can run the pipeline after each function you
complete and see immediate feedback.

Read Lectures 10-12 before you start -- every formula you need is
stated there, and repeated as a hint in the relevant docstring below.

Run with:  python beta_lab_starter.py
Needs 'sample_market_stocks.csv' in the same folder as an offline
fallback dataset (used automatically if yfinance / internet access
isn't available). The sample data spans about 3 years and contains a
deliberate ~2-3 month stress window where every stock's true
sensitivity to the market is temporarily elevated -- your job is to
find it using rolling beta, not to be told where it is in advance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SAMPLE_CSV_PATH = "sample_market_stocks.csv"
INDEX_COL = "NIFTY_INDEX"
STOCK_COLS = ["STOCK_A", "STOCK_B", "STOCK_C", "STOCK_D", "STOCK_E"]
ROLLING_WINDOW = 60  # trading days (~3 months)


# =====================================================================
#  PROVIDED -- data loading. You do not need to modify this function.
# =====================================================================
def load_price_data(use_live=True, index_ticker="^NSEI", stock_tickers=None, period="3y"):
    """Return a DataFrame of daily close prices: first column the market
    index, remaining columns the stocks.

    Tries a live fetch via yfinance first; falls back to the bundled
    sample CSV otherwise, so the lab always runs.
    """
    if use_live:
        try:
            import yfinance as yf
            tickers = [index_ticker] + (stock_tickers or [])
            data = yf.download(tickers, period=period, progress=False)["Close"]
            data = data.rename(columns={index_ticker: INDEX_COL})
            print(f"Loaded LIVE price data: {data.shape[0]} days")
            return data.dropna()
        except Exception as exc:
            print(f"Live fetch failed ({exc!r}); falling back to sample CSV.")

    data = pd.read_csv(SAMPLE_CSV_PATH, index_col="Date", parse_dates=True)
    print(f"Loaded SAMPLE (offline) price data: {data.shape[0]} days, "
          f"{data.shape[1]-1} stocks + 1 index")
    return data


# =====================================================================
#  TODO 1 -- daily log returns from prices. One line, same idea as
#  Weeks 1 and 3.
# =====================================================================
def compute_log_returns(prices):
    """Return ln(P_t / P_{t-1}) for every column, with the first
    (undefined) row dropped."""
    # TODO: implement as described above
    raise NotImplementedError("TODO 1: implement compute_log_returns")


# =====================================================================
#  TODO 2 -- full-sample beta, straight from the Lecture 10 definition.
# =====================================================================
def full_sample_beta(stock_ret, market_ret):
    """beta = Cov(stock_ret, market_ret) / Var(market_ret), computed
    over the ENTIRE sample at once.

    Hint: np.cov(a, b) returns a 2x2 covariance matrix; the [0,1] (or
    [1,0]) entry is Cov(a,b). np.var(x, ddof=1) gives the sample
    variance (ddof=1 for the unbiased estimator).
    """
    # TODO: implement as described above
    raise NotImplementedError("TODO 2: implement full_sample_beta")


# =====================================================================
#  TODO 3 -- THE CORE OF THE LAB. Rolling beta: the SAME formula as
#  TODO 2, but re-computed on every trailing `window`-day slice of
#  data, producing a beta ESTIMATE THAT CHANGES OVER TIME.
# =====================================================================
def rolling_beta(stock_ret, market_ret, window=ROLLING_WINDOW):
    """
    Parameters
    ----------
    stock_ret, market_ret : pandas Series of daily log returns, aligned
                             on the same date index
    window : rolling window length in trading days

    Returns
    -------
    A pandas Series (same index as the inputs) of rolling beta
    estimates. The first `window-1` entries will be NaN (not enough
    data yet to fill a full window).

    Hint: pandas Series have a built-in `.rolling(window)` method that
    supports `.cov(other_series)` and `.var()` directly -- you do NOT
    need to write your own loop. Compute the rolling covariance between
    stock_ret and market_ret, divide by the rolling variance of
    market_ret.
    """
    # TODO: implement as described above
    raise NotImplementedError("TODO 3: implement rolling_beta")


# =====================================================================
#  TODO 4 -- rolling R^2 (fraction of risk that is systematic, over
#  time). Recall from Lecture 12: R^2 = rho^2.
# =====================================================================
def rolling_r_squared(stock_ret, market_ret, window=ROLLING_WINDOW):
    """Hint: pandas Series also support `.rolling(window).corr(other)`
    directly. Square the result."""
    # TODO: implement as described above
    raise NotImplementedError("TODO 4: implement rolling_r_squared")


# =====================================================================
#  PROVIDED -- simple systemic-risk-style indicator, once rolling
#  betas are computed for every stock.
# =====================================================================
def systemic_risk_indicator(rolling_betas_df):
    """The cross-sectional AVERAGE rolling beta across all stocks at
    each point in time. A broad, simultaneous rise across the whole
    universe (not just one name) is the signature of systemic stress."""
    return rolling_betas_df.mean(axis=1)


# =====================================================================
#  PROVIDED -- driver. Once TODOs 1-4 are done, just run this file.
# =====================================================================
def main():
    prices = load_price_data(use_live=True, stock_tickers=None)
    log_ret = compute_log_returns(prices)
    Rm = log_ret[INDEX_COL]

    print("\nFull-sample beta estimates:")
    for s in STOCK_COLS:
        b = full_sample_beta(log_ret[s], Rm)
        print(f"  {s:<10} beta = {b:.3f}")

    rolling_betas = pd.DataFrame({
        s: rolling_beta(log_ret[s], Rm) for s in STOCK_COLS
    })
    rolling_r2 = pd.DataFrame({
        s: rolling_r_squared(log_ret[s], Rm) for s in STOCK_COLS
    })
    systemic = systemic_risk_indicator(rolling_betas)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#1D9E75", "#378ADD", "#5F5E5A", "#D85A30", "#993556"]
    for s, c in zip(STOCK_COLS, colors):
        ax.plot(rolling_betas.index, rolling_betas[s], label=s, color=c, lw=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Rolling beta ({ROLLING_WINDOW}-day window)")
    ax.set_title("Rolling Beta by Stock")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig("rolling_beta.png", dpi=150)
    print("\nSaved rolling_beta.png")

    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    ax2.plot(systemic.index, systemic.values, color="#002D5A", lw=2)
    ax2.axhline(systemic.mean(), color="gray", ls="--", lw=1, label="Full-sample average")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Cross-sectional average rolling beta")
    ax2.set_title("Simple Systemic Risk Indicator")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    fig2.autofmt_xdate()
    fig2.tight_layout()
    fig2.savefig("systemic_risk_indicator.png", dpi=150)
    print("Saved systemic_risk_indicator.png")

    peak_date = systemic.idxmax()
    print(f"\nSystemic indicator peaks on {peak_date.date()} at {systemic.max():.3f}, "
          f"vs. full-sample average {systemic.mean():.3f}")


if __name__ == "__main__":
    main()
