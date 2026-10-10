"""MA5950 Week 8 Assignment - E2 (EE23B031 Kaushal R)
Estimation risk for sigma = 32%: SE(mu_hat) vs history length, SE(sigma_hat)
for 5 years of daily data, and the history needed for SE(mu_hat) < 2%.
"""
import numpy as np

SIG, DAYS = 0.32, 252
print(f"{'years':>6} {'SE(mu_hat)':>12} {'95% CI half-width':>18}")
for y in (3, 5, 10, 25):
    se = SIG / np.sqrt(y)
    print(f"{y:>6} {se:>12.4f} {1.96 * se:>18.4f}")

n = 5 * DAYS
print(f"\nSE(sigma_hat), 5y daily (n={n}): {SIG / np.sqrt(2 * n):.5f}")

need = (SIG / 0.02) ** 2
print(f"Years needed for SE(mu_hat) < 2%: (0.32/0.02)^2 = {need:.0f} years")
