"""
Implied volatility solver for European options under Black-Scholes.

Method: safeguarded Newton-Raphson. Each iteration maintains a bracket
[lo, hi] that is known to contain the root (price is monotonic increasing
in sigma). Newton steps are used when they stay inside the bracket;
otherwise the solver falls back to bisection. This gives Newton's speed
in normal cases and bisection's guaranteed convergence in bad ones
(tiny vega: deep ITM/OTM options, very short maturities).
"""
import numpy as np
from scipy.stats import norm


def bs_price(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes price of a European option (no dividends)."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_vega_raw(S, K, T, r, sigma):
    """Vega per 1.00 (100 vol points) change in sigma. Same for calls and puts."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def implied_vol(price, S, K, T, r, option_type="call",
                tol=1e-10, max_iter=100, vol_lo=1e-6, vol_hi=5.0):
    """
    Solve for the Black-Scholes implied volatility of a European option.

    :param price: observed market price of the option
    :param option_type: 'call' or 'put'
    :param tol: convergence tolerance on the price error
    :param vol_lo, vol_hi: search bracket for sigma (decimal, annualized)
    :return: implied volatility (decimal, annualized)
    :raises ValueError: bad inputs, or price violates no-arbitrage bounds
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if S <= 0 or K <= 0 or T <= 0:
        raise ValueError("S, K and T must be positive")

    disc_K = K * np.exp(-r * T)
    if option_type == "call":
        lower, upper = max(S - disc_K, 0.0), S
    else:
        lower, upper = max(disc_K - S, 0.0), disc_K

    # No-arbitrage bounds: outside these, no sigma can reproduce the price.
    if price <= lower + 1e-14 or price >= upper:
        raise ValueError(
            f"Price {price:.6f} outside no-arbitrage bounds "
            f"({lower:.6f}, {upper:.6f}); implied vol undefined"
        )

    lo, hi = vol_lo, vol_hi
    if bs_price(S, K, T, r, lo, option_type) > price or \
       bs_price(S, K, T, r, hi, option_type) < price:
        raise ValueError("Implied vol lies outside the search bracket")

    # Brenner-Subrahmanyam starting guess (good near the money), clipped to bracket
    sigma = np.sqrt(2 * np.pi / T) * price / S
    sigma = min(max(sigma, lo), hi)

    for _ in range(max_iter):
        diff = bs_price(S, K, T, r, sigma, option_type) - price
        if abs(diff) < tol:
            return float(sigma)

        # Tighten the bracket (price increases with sigma)
        if diff > 0:
            hi = sigma
        else:
            lo = sigma

        vega = bs_vega_raw(S, K, T, r, sigma)
        new_sigma = sigma - diff / vega if vega > 1e-12 else None

        # Fall back to bisection if Newton is unusable or leaves the bracket
        if new_sigma is None or not (lo < new_sigma < hi):
            new_sigma = 0.5 * (lo + hi)
        sigma = new_sigma

    raise RuntimeError("Implied vol solver did not converge")


if __name__ == "__main__":
    S, r = 100.0, 0.045

    print("--- Round-trip tests: price -> implied vol -> compare ---")
    cases = [
        # (K, T, true_vol, type)
        (100, 1.00, 0.20, "call"),
        (100, 1.00, 0.20, "put"),
        (80,  0.50, 0.35, "call"),   # ITM call
        (130, 0.25, 0.50, "call"),   # far OTM call
        (70,  0.25, 0.40, "put"),    # OTM put
        (100, 0.02, 0.25, "call"),   # ~1 week
        (100, 2.00, 1.20, "call"),   # very high vol
    ]
    for K, T, vol, kind in cases:
        p = bs_price(S, K, T, r, vol, kind)
        iv = implied_vol(p, S, K, T, r, kind)
        status = "OK" if abs(iv - vol) < 1e-6 else "CHECK"
        print(f"{kind:4s} K={K:<4} T={T:<5} price={p:9.5f}  "
              f"true={vol:.4f}  solved={iv:.6f}  {status}")

    print("\n--- Arbitrage-bound check ---")
    try:
        implied_vol(0.01, S, 80, 1.0, r, "call")  # below intrinsic
    except ValueError as e:
        print("Caught expected error:", e)
