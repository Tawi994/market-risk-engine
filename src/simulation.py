import numpy as np
import pandas as pd
from src.pricing_models import BlackScholesPricing

class MonteCarloSimulation:
    def __init__(self, spot_prices: np.array, weights: np.array, cov_matrix: pd.DataFrame, risk_free_rate: float):
        """
        Initialize the Monte Carlo Simulation engine.
        """
        self.spots = spot_prices # Current spot price vector for assets
        self.weights = weights # Portfolio weights vector
        self.cov_matrix = cov_matrix
        self.r = risk_free_rate

    def run_gbm_simulation(self, days_horizon: int = 1, num_simulations: int = 10000) -> np.array:
        """
        Generates terminal asset price paths using Geometric Brownian Motion (GBM)
        S_t = S_0 * exp((mu - 0.5 * sigma ** 2) * dt + sigma * sqrt(dt) * Z)
        """
        dt = days_horizon / 252.0 #Trading days scaling
        num_assets = len(self.spots)

        # Extract volatilities from covariance matrix diagonal
        volatilities = np.sqrt(np.diagonal(self.cov_matrix))

        # Compute drift vector (using risk free rate or asset drift approximation)
        drift = (self.r - 0.5 * (volatilities ** 2)) * dt

        # Cholesky decomposition for correlated random normal variables
        cholesky_L = np.linalg.cholesky(self.cov_matrix.values * dt)

        # Generate standard normal random shocks: shape (num_simulations, num_assets)
        random_normals = np.random.normal(0, 1, size=(num_simulations, num_assets))
        correlated_shocks = np.dot(random_normals, cholesky_L.T)

        # Calculate simulated terminal prices
        simulated_prices = self.spots * np.exp(drift + correlated_shocks)
        return simulated_prices
    
    def monte_carlo_portfolio_var(self, option_params: list, confidence_level: float = 0.95, initial_value: float = 100000.0) -> dict:
        """
        Runs Monte Carlo simulation and revalues equities and derivative portfolios dynamically.
        :param option_params: List of dicts describing portfolio options[{'S_idx': 0, 'T': 0.5, 'type': 'call', 'weight': 0.1}, ...]
        """
        num_sims = 10000
        simulated_spots = self.run_gbm_simulation(days_horizon=1, num_simulations=num_sims)

        # Calculate baseline portfolio value
        base_equity_value = initial_value * (np.sum(self.weights) + sum(o['weight'] for o in option_params))

        # Simulate portfolio values at horizon
        portfolio_sim_values = np.zeros(num_sims)

        for i in range(num_sims):
            sim_spot_vector = simulated_spots[i]
            sim_portfolio_val = 0.0

            # Revalue underlying asset components
            for idx, w in enumerate(self.weights):
                sim_portfolio_val += initial_value * w * (sim_spot_vector[idx] / self.spots[idx])

            # Revalue option components dynamically using Black-Scholes
            for opt in option_params:
                spot_idx = opt['S_idx']
                new_S = sim_spot_vector[spot_idx]
                new_T = max(0.001, opt['T'] - (1.0 / 252.0)) # reduce time by 1 day
                vol = opt['vol']

                if opt['type'].lower() == 'call':
                    new_opt_price = BlackScholesPricing.call_price(new_S, opt['K'], new_T, self.r, vol)
                else:
                    new_opt_price = BlackScholesPricing.put_price(new_S, opt['K'], new_T, self.r, vol)

                sim_portfolio_val += initial_value * opt['weight'] * (new_opt_price / opt['initial_opt_price'])

            portfolio_sim_values[i] = sim_portfolio_val

        # Calculate simulated portfolio losses (Delta P)
        portfolio_losses = base_equity_value - portfolio_sim_values
        sorted_losses = np.sort(portfolio_losses)

        index = int(confidence_level * num_sims)
        var_dollar = sorted_losses[index]
        var_pct = var_dollar / base_equity_value

        # Expected Shortfall (Conditional VaR): Average of losss exceeding the VaR threshold
        tail_losses = sorted_losses[index:]
        if len(tail_losses) > 0:
            es_dollar = float(np.mean(tail_losses))
        else:
            es_dollar = var_dollar

        es_pct = float(es_dollar / base_equity_value)

        return {
            "method": "Monte Carlo Simulation",
            "confidence": confidence_level,
            "var_pct": float(var_pct),
            "var_dollar": float(var_dollar),
            "es_pct": float(es_pct),
            "es_dollar": float(es_dollar),
            "base_value": base_equity_value,
            "simulated_portfolio_values": portfolio_sim_values
        }