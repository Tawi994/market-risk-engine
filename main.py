import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from src.data_loader import DataAcquisitionLayer
from src.risk_metrics import RiskMetrics
from src.simulation import MonteCarloSimulation
from src.pricing_models import BlackScholesPricing

def main():

    np.random.seed(42)
    # Configure clean logging for debugging financial data pipelines
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=" * 80)
    print(f"{'INITIALIZING MULTI-ASSET MARKET RISK ENGINE':^80}")
    print("=" * 80)

    # 1. Configuration
    tickers = ['AAPL', 'MSFT', '^GSPC']
    weights = np.array([0.5, 0.3, 0.2]) # 50% Apple, 30% Microsoft, 20% S&P 500 Index
    initial_value= 100000.0             # $100,000 Portfolio Initial Value

    # 2. Data Acquisition Layer
    loader = DataAcquisitionLayer(start_date="2023-01-01", end_date="2026-01-01")
    prices, log_returns = loader.build_risk_matrix(tickers)

    # Fetch risk-free rate series (TB3MS) and extract latest annualized rate as decimal
    rf_series = loader.fetch_risk_free_rate("TB3MS")
    risk_free_rate = float(rf_series.dropna().iloc[-1])

    # Compute historical daily portfolio return (weighted sum)
    portfolio_hist_returns = log_returns.dot(weights)
    daily_cov_matrix = log_returns.cov()
    current_spots = prices.iloc[-1].values

    print(f"\nPortfolio Loaded Successfully.")
    print(f"Assets: {tickers}")
    print(f"Risk Free Rate (FRED TB3MS): {risk_free_rate * 100:.2f}%")
    print(f"Latest Spot Prices: {dict(zip(tickers, current_spots))}")

    # 3. Risk Calculations: Historical Simulation
    hist_95 = RiskMetrics.historical_var_es(portfolio_hist_returns, confidence_level=0.95, initial_value=initial_value)
    hist_99 = RiskMetrics.historical_var_es(portfolio_hist_returns, confidence_level=0.99, initial_value=initial_value)

    # 4. Risk Calculations: Parametric (Variance-Covariance)
    param_95 = RiskMetrics.parametric_var_es(weights, daily_cov_matrix, confidence_level= 0.95, initial_value=initial_value)
    param_99 = RiskMetrics.parametric_var_es(weights, daily_cov_matrix, confidence_level= 0.99, initial_value=initial_value)

    # Risk Calculations: Monte Carlo Simulation (with Option Revaluation)
    # Define an options overlay: 10% allocation in an out-of-the-money Apple Call Option
    aapl_spot = current_spots[0]
    strike_price = aapl_spot * 1.05
    time_to_maturity = 0.5 # 6 months
    volatility = np.sqrt(daily_cov_matrix.iloc[0,0] * 252) # annualized of AAPL

    initial_call_price = BlackScholesPricing.call_price(aapl_spot, strike_price, time_to_maturity, risk_free_rate, volatility)

    option_params = [{
        'S_idx': 0,
        'K': strike_price,
        'T': time_to_maturity,
        'type': 'call',
        'weight': 0.1,
        'vol': volatility,
        'initial_opt_price': initial_call_price
    }]

    mc_engine = MonteCarloSimulation(spot_prices=current_spots, weights=weights, cov_matrix=daily_cov_matrix * 252, risk_free_rate=risk_free_rate)
    mc_95 = mc_engine.monte_carlo_portfolio_var(option_params, confidence_level= 0.95, initial_value=initial_value)
    mc_99 = mc_engine.monte_carlo_portfolio_var(option_params, confidence_level= 0.99, initial_value=initial_value)

    # 6. Print Risk Framework Comparison Table
    print("\n" + "=" * 94)
    print(f"{'PORTFOLIO VALUE AT RISK (VaR) & EXPECTED SHORTFALL (ES) COMPARISON REPORT':^94}")
    print(f"{'Initial Portfolio Value: $' + f'{initial_value:,.2f} | Horizon: 1 Trading Day':^94}")
    print("=" * 94)
    print(f"{'Methodology':<32} | {'Conf':<6} | {'VaR ($)':<12} | {'VaR (%)':<8} | {'ES ($)':<12} | {'ES (%)':<8}")
    print("-" * 94)

    results = [hist_95, hist_99, param_95, param_99, mc_95, mc_99]
    for res in results:
        conf_str = f"{int(res['confidence']*100)}%"
        print(f"{res['method']:<32} | {conf_str:<6} | ${res['var_dollar']:<11,.2f} | {res['var_pct']*100:<6.2f}% | ${res['es_dollar']:<11,.2f} | {res['es_pct']*100:<6.2f}%")

    print("=" * 94)

    # 7. Visualization: Matplotlib Risk Distribution Plot
    print("\n Generating Monte Carlo Risk Distribution Plot...")
    simulated_portfolio_values = mc_95['simulated_portfolio_values']
    portfolio_losses = mc_95['base_value'] - simulated_portfolio_values

    plt.figure(figsize=(11, 6))
    n, bins, patches = plt.hist(portfolio_losses, bins=60, density=True, alpha=0.65, color='#2b5c8f', edgecolor='black', linewidth=0.5)

    var_95_dollar = mc_95['var_dollar']
    var_99_dollar = mc_99['var_dollar']

    plt.axvline(var_95_dollar, color= 'orange', linestyle='--', linewidth=2.5, label=f'95% VaR: ${var_95_dollar:,.2f}')
    plt.axvline(var_99_dollar, color='crimson', linestyle='-', linewidth=2.5, label=f'99% VaR: ${var_99_dollar:,.2f}')
    plt.title('Multi-Asset Portfolio Loss Distribution (Monte Carlo GBM with Option Revaluation)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Simulated 1-Day Portfolio Loss ($)', fontsize=11)
    plt.ylabel('Probability Density', fontsize=11)
    plt.legend(loc='upper right', frameon=True, facecolor= 'white', framealpha= 0.9)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()