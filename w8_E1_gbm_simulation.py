"""MA5950 Week 8 Assignment - E1 (EE23B031 Kaushal R)
200,000 GBM terminal prices: S0=100, mu=10%, sigma=35%, T=4.
Compare with theory, then omit the -sigma^2/2 term and measure the inflation.
"""
import numpy as np

S0, MU, SIG, T, N = 100.0, 0.10, 0.35, 4.0, 200_000
rng = np.random.default_rng(20260808)
Z = rng.standard_normal(N)

ST = S0 * np.exp((MU - 0.5 * SIG**2) * T + SIG * np.sqrt(T) * Z)   # correct
ST_bad = S0 * np.exp(MU * T + SIG * np.sqrt(T) * Z)               # Ito term omitted

th_mean = S0 * np.exp(MU * T)
th_med = S0 * np.exp((MU - 0.5 * SIG**2) * T)
th_sd = S0 * np.exp(MU * T) * np.sqrt(np.exp(SIG**2 * T) - 1)

print(f"{'':<8}{'simulated':>12}{'theory':>12}")
print(f"{'mean':<8}{ST.mean():>12.3f}{th_mean:>12.3f}")
print(f"{'median':<8}{np.median(ST):>12.3f}{th_med:>12.3f}")
print(f"{'sd':<8}{ST.std(ddof=1):>12.3f}{th_sd:>12.3f}")
print(f"MC s.e. of mean: {ST.std(ddof=1) / np.sqrt(N):.3f}")

f_sim = ST_bad.mean() / ST.mean()
f_th = np.exp(0.5 * SIG**2 * T)
print(f"\nWithout -sigma^2/2: simulated mean = {ST_bad.mean():.3f}")
print(f"Inflation factor (simulated, same Z) = {f_sim:.5f}")
print(f"exp(sigma^2 T / 2)                    = {f_th:.5f}")
