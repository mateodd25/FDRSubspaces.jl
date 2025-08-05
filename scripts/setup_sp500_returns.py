#!/usr/bin/env python3
"""
setup_sp500_returns.py

Fetches daily adjusted close prices for all S&P 500 constituents,
computes log-returns, centers each series, and saves the resulting
matrix X (T × N) into a .mat file for use in FDR subspace selection experiments.

Similar to setup_single_cell_RNA.py but for financial data.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from scipy.io import savemat
import os
import warnings
warnings.filterwarnings('ignore')

def fetch_sp500_tickers():
    """Scrape the list of S&P 500 tickers from Wikipedia."""
    print("Fetching S&P 500 tickers from Wikipedia...")
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = table[0]
        tickers = df['Symbol'].tolist()
        # Clean up tickers (some may have dots that need to be replaced)
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        return tickers
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        print("Using fallback list of major S&P 500 companies...")
        # Fallback to major companies if Wikipedia scraping fails
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B',
                'JPM', 'JNJ', 'V', 'PG', 'UNH', 'MA', 'HD', 'DIS', 'BAC', 'ADBE',
                'CRM', 'NFLX', 'KO', 'PEP', 'TMO', 'ABBV', 'COST', 'AVGO', 'XOM',
                'WMT', 'ACN', 'DHR', 'VZ', 'MCD', 'TXN', 'NEE', 'CVX', 'LLY',
                'ABT', 'QCOM', 'PM', 'BMY', 'T', 'HON', 'LOW', 'MDT', 'AMGN',
                'UPS', 'IBM', 'BA', 'CAT']

def download_prices(tickers, start_date, end_date):
    """Download adjusted close prices for given tickers."""
    print(f"Downloading price data from {start_date} to {end_date}...")
    
    # Download in batches to avoid API limits
    batch_size = 50
    all_data = {}
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        print(f"  Downloading batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}: {len(batch)} tickers")
        
        try:
            data = yf.download(
                batch,
                start=start_date,
                end=end_date,
                progress=False,
                group_by='ticker',
                threads=True
            )
            
            # Handle single ticker case
            if len(batch) == 1:
                if 'Adj Close' in data.columns:
                    all_data[batch[0]] = data['Adj Close']
            else:
                # Multiple tickers
                for ticker in batch:
                    try:
                        if (ticker, 'Adj Close') in data.columns:
                            all_data[ticker] = data[ticker]['Adj Close']
                        elif ticker in data.columns.levels[0]:
                            all_data[ticker] = data[ticker]['Adj Close']
                    except (KeyError, AttributeError):
                        print(f"    Warning: Could not extract data for {ticker}")
                        continue
                        
        except Exception as e:
            print(f"    Error downloading batch: {e}")
            continue
    
    if not all_data:
        raise ValueError("No data was successfully downloaded")
    
    # Convert to DataFrame
    prices_df = pd.DataFrame(all_data)
    return prices_df

def compute_log_returns(price_df):
    """Compute log returns and center each column."""
    print("Computing log returns...")
    
    # Remove columns with insufficient data
    min_observations = 100  # Require at least 100 trading days
    price_df = price_df.dropna(thresh=min_observations, axis=1)
    
    # Compute log-returns
    R = np.log(price_df / price_df.shift(1))
    
    # Drop rows with all NaNs (typically the first row)
    R = R.dropna(how='all')
    
    # Drop columns (tickers) with any remaining NaNs
    initial_cols = R.shape[1]
    R = R.dropna(axis=1, how='any')
    final_cols = R.shape[1]
    
    if final_cols < initial_cols:
        print(f"  Removed {initial_cols - final_cols} tickers with missing data")
    
    if R.empty or final_cols < 10:
        raise ValueError("Insufficient data after cleaning")
    
    # Center each column (zero mean)
    R_centered = R - R.mean(axis=0)
    
    print(f"  Final return matrix: {R_centered.shape[0]} days × {R_centered.shape[1]} assets")
    print(f"  Date range: {R_centered.index[0].date()} to {R_centered.index[-1].date()}")
    
    return R_centered.values, R_centered.columns.tolist()

def main():
    # User parameters
    start_date = "2020-01-01"
    end_date = "2024-12-31"
    
    # Create data directory if it doesn't exist
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    mat_filename = os.path.join(data_dir, "sp500_returns.mat")
    
    try:
        # 1. Get tickers
        tickers = fetch_sp500_tickers()
        print(f"Found {len(tickers)} S&P 500 tickers.")
        
        # 2. Download prices
        prices = download_prices(tickers, start_date, end_date)
        print(f"Downloaded price data: {prices.shape[0]} days × {prices.shape[1]} assets.")
        
        # 3. Compute return matrix X
        X, final_tickers = compute_log_returns(prices)
        print(f"Final return matrix X: shape = {X.shape} (days × assets)")
        
        # 4. Save to .mat file with additional metadata
        save_data = {
            'X': X,
            'tickers': final_tickers,
            'start_date': start_date,
            'end_date': end_date,
            'description': 'S&P 500 daily log returns (centered), T×N matrix where T=days, N=assets'
        }
        
        savemat(mat_filename, save_data)
        print(f"Saved return matrix to {mat_filename}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"  Time period: {start_date} to {end_date}")
        print(f"  Number of assets: {X.shape[1]}")
        print(f"  Number of trading days: {X.shape[0]}")
        print(f"  Mean daily return: {np.mean(X):.6f}")
        print(f"  Std daily return: {np.std(X):.6f}")
        print(f"  Matrix saved to: {mat_filename}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Failed to download and process S&P 500 data.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())