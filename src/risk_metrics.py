import numpy as np
import pandas as pd
from scipy.stats import norm

class RiskMetrics:
    @staticmethod
    def historical_var_es(portfolio_returns: pd.Series, confidence_level: float = 0.95, initial_value: float = 100000.0) -> dict:
        """
        Calculate Historical VaR and Expected Shortfall (Conditional VaR).
        :param portfolio_returns: Series of historical portfolio percentage returns
        :param confidence_level: e.g, 0.95 or 0.99
        :param initial_value: Total portfolio market value in currency units
        """
        sorted_returns = np.sort(portfolio_returns.values)
        # Find index corresponding to the confindence percentile
        index = int((1.0 - confidence_level) * len(sorted_returns))

        # VaR is the negative of the return at that percentile
        var_pct = -sorted_returns[index]
        var_dollar = var_pct * initial_value

        # Expected Shortfall (Conditional VaR) is the average loss worse than VaR
        tail_losses = sorted_returns[:index]
        es_pct = -np.mean(tail_losses) if len(tail_losses) > 0 else var_pct
        es_dollar = es_pct * initial_value

        return {
            "method": "Historical Simulation",
            "confidence": confidence_level,
            "var_pct": float(var_pct),
            "var_dollar": float(var_dollar),
            "es_pct": float(es_pct),
            "es_dollar": float(es_dollar)
        }
    
    @staticmethod
    def parametric_var_es(weights: np.ndarray, cov_matrix: pd.DataFrame, confidence_level: float = 0.95, initial_value: float = 100000.0) -> dict:
        """
        Calculate Parametric (Variance-Covariance) VaR and Expected Shortfall assuming normal distribution.
        """
        # Portfolio variance: w^T * Sigma * w
        port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        port_volatility = np.sqrt(port_variance)

        # Z-score for confidence level
        z_score = norm.ppf(confidence_level)

        # Parametric VaR (assuming zero mean drift for daily horizon)
        var_pct = z_score * port_volatility
        var_dollar = var_pct * initial_value

        # Parametric Expected Shortfall: ES = sigma * (pdf(Z) / (1 - confidence))
        pdf_z = norm.pdf(z_score)
        es_pct = port_volatility * (pdf_z / (1.0 - confidence_level))
        es_dollar = es_pct * initial_value

        return {
            "method": "Parametric (Variance-Covariance)",
            "confidence": confidence_level,
            "var_pct": float(var_pct),
            "var_dollar": float(var_dollar),
            "es_pct": float(es_pct),
            "es_dollar": float(es_dollar)
        }
    