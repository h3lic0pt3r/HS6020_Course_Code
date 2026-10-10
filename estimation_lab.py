"""
MA5950 -- Mathematical Finance: Theory & Implementation
Week 8 Lab: Estimating Drift & Diffusion from Historical Log-Returns
-----------------------------------------------------------------------
COMPLETED -- EE23B031 Kaushal R

Fill in every function marked TODO. Data loading, plotting and the
driver are provided.

Read Lectures 23-24 before you start.

Run with:  python estimation_lab_starter.py
Needs 'sample_gbm_prices.csv' in the same folder: five years of daily
prices for three stocks, generated from GBM with TRUE parameters that
are deliberately NOT revealed to you here. Part of this lab is forming
an honest opinion about how much your estimates can be trusted BEFORE
the true values are disclosed in the debrief.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

SAMPLE_CSV = "sample_gbm_prices.csv"
TRADING_DAYS = 252
NAVY, MAROON, FOREST, AMBER = "#002D5A", "#8C0F19", "#1D9E75", "#C87800"


# =====================================================================
#  PROVIDED -- data loading
# =====================================================================
def load_prices(path=SAMPLE_CSV, use_live=True, tickers=None):
    if use_live and tickers:
        try:
            import yfinance as yf
            data = yf.download(tickers, period="5y", progress=False)["Close"]
            print(f"Loaded LIVE data: {data.shape[0]} days")
            return data.dropna()
        except Exception as exc:
            print(f"Live fetch failed ({exc!r}); using sample CSV.")
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    print(f"Loaded SAMPLE data: {df.shape[0]} days, {df.shape[1]} stocks")
    return df


# =====================================================================
#  TODO 1 -- daily log-returns (one line; same idea as Weeks 3, 4, 7).
# =====================================================================
def log_returns(prices):
    """Return ln(P_t / P_{t-1}) for every column, first row dropped."""
    return np.log(prices / prices.shift(1)).iloc[1:]


# =====================================================================
#  TODO 2 -- THE CORE OF THE LAB: estimate mu and sigma from log-returns.
#
#  From Lecture 23, the log-return over a step dt satisfies
#      X ~ N( (mu - sigma^2/2)*dt ,  sigma^2 * dt )
#  Inverting:
#      sigma_hat = sd(X) / sqrt(dt)
#      mu_hat    = mean(X)/dt + sigma_hat^2 / 2
#
#  NOTE the +sigma_hat^2/2. It undoes the Ito correction: the mean
#  log-return estimates (mu - sigma^2/2), NOT mu. Forgetting this term
#  is the single most common error in this lab, and it biases mu_hat
#  DOWNWARD by sigma^2/2 -- for sigma=30%, that is 4.5% per year.
#
#  Standard errors (you may quote these; deriving them is not required):
#      SE(sigma_hat) = sigma_hat / sqrt(2n)
#      SE(mu_hat)    = sigma_hat / sqrt(YEARS)      where YEARS = n*dt
#
#  Look hard at that second formula before you code it. SE(mu_hat)
#  depends on the CALENDAR SPAN of your sample, not on n. Sampling
#  every minute for one year gives exactly the same drift precision as
#  sampling once a month for one year. Experiment C will confirm this.
# =====================================================================
def estimate_gbm(log_ret, dt=1.0 / TRADING_DAYS):
    """Return (mu_hat, sigma_hat, se_mu, se_sigma)."""
    X = np.asarray(log_ret, dtype=float)
    n = len(X)
    sigma_hat = X.std(ddof=1) / np.sqrt(dt)
    mu_hat = X.mean() / dt + 0.5 * sigma_hat**2      # undo the Ito correction
    years = n * dt
    se_mu = sigma_hat / np.sqrt(years)               # depends on SPAN, not n
    se_sigma = sigma_hat / np.sqrt(2 * n)
    return mu_hat, sigma_hat, se_mu, se_sigma


# =====================================================================
#  TODO 3 -- sampling frequency experiment.
#
#  Hold the calendar span FIXED at `years`. For each sampling frequency
#  m (observations per year), simulate n_reps independent GBM histories,
#  estimate mu and sigma from each, and report the SPREAD (standard
#  deviation across reps) of each estimator.
# =====================================================================
def frequency_experiment(mu, sigma, years, obs_per_year_list, n_reps, rng):
    """Return a list of (m, n, sd_of_mu_hat, sd_of_sigma_hat).

    For each m:
      n = int(years * m);  dt = 1/m
      X = rng.normal((mu - 0.5*sigma**2)*dt, sigma*np.sqrt(dt), (n_reps, n))
      then apply the SAME estimator formulas as TODO 2, row-wise
      (use axis=1), and take .std() across the n_reps estimates.
    """
    out = []
    for m in obs_per_year_list:
        n = int(years * m)
        dt = 1.0 / m
        X = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt),
                       (n_reps, n))
        sig_hat = X.std(axis=1, ddof=1) / np.sqrt(dt)
        mu_hat = X.mean(axis=1) / dt + 0.5 * sig_hat**2
        out.append((m, n, mu_hat.std(), sig_hat.std()))
    return out


# =====================================================================
#  TODO 4 -- normality diagnostics for log-returns.
# =====================================================================
def normality_check(log_ret):
    """Return a dict with 'skew', 'excess_kurtosis', 'jarque_bera_p'.

    Hints: scipy.stats.skew, scipy.stats.kurtosis (which returns EXCESS
    kurtosis by default), and scipy.stats.jarque_bera(X).pvalue.
    """
    X = np.asarray(log_ret, dtype=float)
    return {"skew": stats.skew(X),
            "excess_kurtosis": stats.kurtosis(X),
            "jarque_bera_p": stats.jarque_bera(X).pvalue}


# =====================================================================
#  TODO 5 -- lognormal quantiles for a future price (Lecture 24).
# =====================================================================
def lognormal_quantiles(S0, mu, sigma, T, ps=(0.05, 0.25, 0.5, 0.75, 0.95)):
    """Return {p: quantile} using
         Q_p = S0 * exp[ (mu - sigma^2/2) T + sigma sqrt(T) z_p ].
    Use stats.norm.ppf(p) for z_p.
    """
    m = (mu - 0.5 * sigma**2) * T
    v = sigma * np.sqrt(T)
    return {p: S0 * np.exp(m + v * stats.norm.ppf(p)) for p in ps}


# =====================================================================
#  PROVIDED -- driver
# =====================================================================
def main():
    rng = np.random.default_rng(20260808)
    prices = load_prices()
    lr = log_returns(prices)

    print("\n" + "=" * 78)
    print("A/B. PARAMETER ESTIMATES  (5 years of daily data)")
    print("=" * 78)
    print(f"  {'stock':<8} {'mu_hat':>9} {'SE(mu)':>9} {'sig_hat':>9} {'SE(sig)':>9}")
    for name in prices.columns:
        mu_h, sig_h, se_mu, se_sig = estimate_gbm(lr[name])
        print(f"  {name:<8} {mu_h:>9.4f} {se_mu:>9.4f} {sig_h:>9.4f} {se_sig:>9.4f}")

    print("\n" + "=" * 78)
    print("C. SAMPLING FREQUENCY: does more data within the same span help?")
    print("=" * 78)
    print("   (true mu=0.10, sigma=0.25, span fixed at 5 years, 4000 reps)")
    print(f"  {'obs/year':>10} {'n':>8} {'sd(mu_hat)':>12} {'sd(sig_hat)':>13}")
    for m, n, sdm, sds in frequency_experiment(
            0.10, 0.25, years=5, obs_per_year_list=[12, 52, 252, 252 * 8],
            n_reps=4000, rng=rng):
        print(f"  {m:>10} {n:>8} {sdm:>12.4f} {sds:>13.5f}")

    print("\n" + "=" * 78)
    print("D. NORMALITY OF LOG-RETURNS")
    print("=" * 78)
    print(f"  {'stock':<8} {'skew':>9} {'ex.kurt':>10} {'JB p-value':>12}")
    for name in prices.columns:
        s = normality_check(lr[name])
        print(f"  {name:<8} {s['skew']:>+9.4f} {s['excess_kurtosis']:>+10.4f} "
              f"{s['jarque_bera_p']:>12.4f}")

    print("\n" + "=" * 78)
    print("E. FORWARD DISTRIBUTION using estimated parameters (T = 1 year)")
    print("=" * 78)
    for name in prices.columns:
        mu_h, sig_h, _, _ = estimate_gbm(lr[name])
        S0 = float(prices[name].iloc[-1])
        q = lognormal_quantiles(S0, mu_h, sig_h, 1.0)
        print(f"  {name:<8} S0={S0:>9.2f}  "
              f"5%={q[0.05]:>9.2f}  50%={q[0.5]:>9.2f}  95%={q[0.95]:>9.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for ax, name, c in zip(axes, prices.columns, [NAVY, MAROON, FOREST]):
        ax.hist(lr[name], bins=60, density=True, color=c, alpha=0.65)
        xs = np.linspace(lr[name].min(), lr[name].max(), 200)
        ax.plot(xs, stats.norm.pdf(xs, lr[name].mean(), lr[name].std()),
                "k--", lw=1.3, label="fitted normal")
        ax.set_title(name); ax.set_xlabel("daily log-return")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    axes[0].set_ylabel("density")
    fig.suptitle("Log-return distributions with fitted normal densities")
    fig.tight_layout(); fig.savefig("logreturn_distributions.png", dpi=150)
    print("\n  Saved logreturn_distributions.png")

    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    yrs = [1, 2, 5, 10, 20, 40]
    a1.plot(yrs, [0.25 / np.sqrt(y) for y in yrs], "o-", color=MAROON, lw=2)
    a1.axhline(0.10, color=NAVY, ls="--", lw=1.4, label="a plausible true $\\mu$")
    a1.set_xlabel("years of data"); a1.set_ylabel("SE($\\hat\\mu$)")
    a1.set_title("Drift: standard error vs. calendar span")
    a1.legend(fontsize=9); a1.grid(alpha=0.3)

    rows = frequency_experiment(0.10, 0.25, 5, [12, 52, 252, 252 * 8], 2000, rng)
    a2.loglog([r[1] for r in rows], [r[2] for r in rows], "o-",
              color=MAROON, lw=2, label="sd($\\hat\\mu$)")
    a2.loglog([r[1] for r in rows], [r[3] for r in rows], "s-",
              color=FOREST, lw=2, label="sd($\\hat\\sigma$)")
    a2.set_xlabel("observations in a fixed 5-year span")
    a2.set_ylabel("estimator spread")
    a2.set_title("Sampling more often: helps $\\sigma$, not $\\mu$")
    a2.legend(fontsize=9); a2.grid(alpha=0.3, which="both")
    fig2.tight_layout(); fig2.savefig("estimation_risk.png", dpi=150)
    print("  Saved estimation_risk.png")


if __name__ == "__main__":
    main()
