import os
import logging
from typing import List, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web

class DataAcquisitionLayer:
    """
    Handles robust fetching of historical market data and risk-free rates,
    computing mathematical log returns for market risk simulation engine.
    """
    def __init__(self, start_date: str, end_date: str):
        """
        Initialize with a standard temporal window.
        Format: 'YYYY-MM-DD'
        """
        self.start_date = start_date
        self.end_date = end_date
    
    def fetch_equity_prices(self, tickers: List[str]) -> pd.DataFrame:
        """
        Fetches adjusted daily closing prices for global equities/ETFs from Yahoo Finance.
        """
        logging.info(f"Initiating Yahoo Finance data pull for tickers: {tickers}")
        try:
            #Group download to optimize network calls
            data = yf.download(tickers, start=self.start_date, end=self.end_date, progress=False)

            if data.empty:
                raise ValueError(f"No data returned for: {tickers}")

            # Extract only the Adjusted Close column (handles stock splits and dividends)
            if 'Adj Close' in data.columns:
                adj_close = data['Adj Close']
            else:
                adj_close = data['Close']   #Fallback is Adj Close isn't structural
            
            adj_close = adj_close[tickers]
            
            #Handle single ticker case where yfinance returns a Series instead of a DataFrame
            if isinstance(adj_close, pd.Series):
                adj_close = adj_close.to_frame(name=tickers[0])

            # Drop any entirely empty columns or rows caused by localized market holidays
            adj_close = adj_close.dropna(how='all')

            #Forward fill minor gaps (e.g., mismatched market holidays between exchanges)
            adj_close = adj_close.ffill().bfill()

            logging.info("Equity price acquisition successful.")
            return adj_close
        
        except Exception as e:
            logging.error(f"Critical error pulling Yahoo Finance data: {str(e)}")
            raise e
        

    def fetch_risk_free_rate(self, series_id: str = "TB3MS") -> pd.Series:
        """
        Fetches historical macroeconomic risk-free rates from St. Louis FRED
        Default: 'TB3MS' (3-Month US Treasury Bill Secondary Market Rate).

        """
        logging.info(f"Initiating FRED data pull for series: {series_id}")
        try:
            df = web.DataReader(series_id, 'fred', self.start_date, self.end_date)
            # FRED returns monthly/daily yields as percentage (e.g., 4.5 meaning 4.5%)
            # Convert to a localized daily decimal form for Option pricing equations
            daily_rf_decimal = (df[series_id] / 100)

            # Reindex to forward fill across daily equity intervals downstream
            logging.info("Risk free rate acquisition successful.")
            return daily_rf_decimal
        
        except Exception as e:
            logging.error(f"Critical error pulling macroeconomic rate from FRED: {str(e)}")
            raise e
        
    @staticmethod
    def calculate_log_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw price time series into stationary daily log returns.
        Formula: r_t = ln(P_t / P_{t-1})
        """
        logging.info("Executing mathematical transformation to daily log returns.")
        # np.log(Pt / Pt-1) handles full matrices efficiently
        log_returns = np.log(prices_df / prices_df.shift(1))

        # The first row will naturally contain NaN values due to the shift operation.
        return log_returns.dropna(how='all')
    
    def build_risk_matrix(self, tickers: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Orchestrates the pipeline to deliver clean inputs for downstream risk calculations.
        Returns:
            Tuple[DataFrame of clean closing prices, DataFrame of stationary log returns]
        """
        prices = self.fetch_equity_prices(tickers)
        log_returns = self.calculate_log_returns(prices)
        return prices, log_returns
    
# Local Unit Testing blck to verify pipeline validity independently
if __name__ == "__main__":

    # Configure clean logging for debugging financial data pipelines
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Test with a mock 3-year historical frame
    loader = DataAcquisitionLayer(start_date="2023-01-01", end_date="2026-01-01")

    # Portfolio mix: Tech stocks and a broad market index
    test_portfolio = ["AAPL", "MSFT", "GOOGL"]

    prices, returns = loader.build_risk_matrix(test_portfolio)
    rf = loader.fetch_risk_free_rate()

    # Real Sanity Checks
    assert not prices.isna().any().any(), "Prices contains NaNs"
    assert np.isfinite(returns.to_numpy()).all(), "Returns contain NaN/inf"

    cov = returns.cov()
    assert np.allclose(cov, cov.T), "Covariance matrix not symmetric"
    assert np.linalg.eigvalsh(cov).min() > -1e-12, "Covariance not positive semi-definite"

    print(f"Prices: {prices.shape}, Returns: {returns.shape}")
    print(f"Risk-free rate points: {len(rf)}")
    print("Annualized volatility: \n", returns.std() * np.sqrt(252))
    print("Pipeline validated successfully.")