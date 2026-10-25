import numpy as np
from scipy.stats import norm

class BlackScholesPricing:
    @staticmethod
    def _d1_d2(S: float, K: float, T: float, r: float, sigma: float):
        """
        Calculate d1 and d2 parameters for the Black-Scholes-Merton formula.
        :param S: Spot price of the underlying asset
        :param K: Strike price of the option
        :param T: Time to maturity (in years)
        :param r: Risk-free interest rate (annulized, decimal)
        :param sigma: Volatility of the underlying asset (annulized, decimal)
        """
        # Edge case protection for zero or negative time/volatility
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    @classmethod
    def call_price(cls, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate the Black-Scholes price for a European Call option
        """
        if S <= 0 or K <= 0:
            raise ValueError("S and K must be positive")
        if T <= 0:
            return max(0.0, S - K)
        if sigma <= 0:
            return max(0.0, S - K * np.exp(-r * T))
        d1, d2 = cls._d1_d2(S, K, T, r, sigma)
        call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return float(call)

    @classmethod
    def put_price(cls, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate the Black-Scholes price for a European Put option
        """
        if S <= 0 or K <= 0:
            raise ValueError("S and K must be positive")
        if T <= 0:
            return max(0.0, K - S)
        if sigma <= 0:
            return max(0.0, K * np.exp(-r * T) - S)
        

        d1, d2 = cls._d1_d2(S, K, T, r, sigma)
        put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return float(put)

    @classmethod
    def options_greeks(cls, S: float, K: float, T: float, r: float, sigma: float, option_type: str= 'call') -> dict:
        """
        Calculate option Greeks (Delta, Gamma, Vega, Theta) for risk sensitivity analysis.

        """
        if T <= 0:
            if option_type.lower() == 'call':
                delta = 1.0 if S > K else 0.0
            else:
                delta = -1.0 if S < K else 0.0
            return {"delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

        d1, d2 = cls._d1_d2(S, K, T, r, sigma)

        #Vega (same for call and put)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100.0 #scaled per 1% change in vol
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
            theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                    - r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0 # per calender year
        else:
            delta = norm.cdf(d1) - 1.0
            theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                     + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0 # per calender year

        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "theta": float(theta)
        }

if __name__ == '__main__':
    # Test block for Black-Scholes validation
    S_test = 100.0 # Spot price
    K_test = 100.0 # Strike price
    T_test = 1 # 1 year to maturity
    r_test = 0.045 # 4.5% risk-free rate
    vol_test = 0.20 # 20% volatility

    c_price = BlackScholesPricing.call_price(S_test, K_test, T_test, r_test, vol_test)
    p_price = BlackScholesPricing.put_price(S_test, K_test, T_test, r_test, vol_test)
    greeks = BlackScholesPricing.options_greeks(S_test, K_test, T_test, r_test, vol_test)
    parity = c_price - p_price - (S_test - K_test * np.exp(-r_test * T_test))
    assert abs(parity) < 1e-10, "Put-call parity violated"

    print("--- Black-Scholes Pricing Test ---")
    print(f"European Call Option Price: ${c_price:.2f}")
    print(f"European Put Option Price:  ${p_price:.2f}")
    print(f"Call Delta:                  {greeks['delta']:.4f}")
    print(f"Call Gamma:                  {greeks['gamma']:.4f}")
    print(f"Call Vega (per 1%):         ${greeks['vega']:.4f}")
    print(f"Call Theta (per day):       ${greeks['theta']:.4f}")