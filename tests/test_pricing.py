import pytest
import numpy as np
from src.pricing_models import BlackScholesPricing

def test_black_scholes_call_benchmark():
    """
    Tests European Call option price against a standard benchmark case.
    Parameters: S=100, K=100, T=1.0, r=0.05, sigma=0.20
    Known BSM Call Price ~ 10.4506
    """
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    call_price = BlackScholesPricing.call_price(S, K, T, r, sigma)
    assert pytest.approx(call_price, rel=1e-3) == 10.4506

def test_black_scholes_put_benchmark():
    """
    Tests European Put option price against a standard benchmark case.
    Parameters: S=100, K=100, T=1.0, r=0.05, sigma=0.20
    Known BSM Put price ~ 5.5735
    """
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    put_price = BlackScholesPricing.put_price(S, K, T, r, sigma)
    assert pytest.approx(put_price, rel=1e-3) == 5.5735

def test_put_call_parity():
    """
    Validates Put-Call Parity relation: C - P = S - K * exp(-r * T)
    """
    S, K, T, r, sigma = 105.0, 100, 0.75, 0.04, 0.25
    c = BlackScholesPricing.call_price(S, K, T, r, sigma)
    p = BlackScholesPricing.put_price(S, K, T, r, sigma)

    lhs = c - p
    rhs = S - K * np.exp(-r * T)

    assert pytest.approx(lhs, rel= 1e-5) == rhs

def test_expiry_edge_case():
    """
    Test option pricing when time to maturity (T) is zero or negative (intrinsic value boundary)
    """
    # In-the-money call at expiry (S=110, K = 100 -> intrinsic 10)
    assert BlackScholesPricing.call_price(110, 100, 0.0, 0.05, 0.20)
    # Out-the-money call at expiry (S=90, K=100 -> intrinsic 0)
    assert BlackScholesPricing.call_price(90.0, 100.0, 0.0, 0.05, 0.20)

def test_greeks_output_structure():
    """
    Ensure the option Greeks dictionary contains all required keys and valid float types.

    """
    greeks = BlackScholesPricing.options_greeks(100.0, 100.0, 1.0, 0.05, 0.20, 'call')
    required_keys = {"delta", "gamma", "vega", "theta"}

    assert required_keys.issubset(greeks.keys())
    for key in required_keys:
        assert isinstance(greeks[key], float)
        assert np.isfinite(greeks[key])