import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr

# Set date range
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=365*5)
print(f"Start date: {startDate}")
print(f"End date: {endDate}")

# Define stocks
stocks = ["msft", "aapl", "googl", "amzn", "meta"]

# Download data (with auto_adjust=True by default)
df = yf.download(stocks, start=startDate, end=endDate)
print("\nDataFrame head:")
print(df.head())

# Extract close prices
close_prices = df["Close"]

# Method 1: Log returns approach
logReturns = np.log(close_prices / close_prices.shift(1))
print("\nLog returns head:")
print(logReturns.head())

# For log returns, we accumulate by summing them
# and then exponentiating to get the cumulative return
cumulativeLogReturns = logReturns.cumsum()
accumulatedReturns = np.exp(cumulativeLogReturns) - 1  # Convert to percentage gain/loss
print("\nAccumulated returns head (from log returns):")
print(accumulatedReturns.head())

# Method 2: Simple returns approach (alternative)
# This is a more direct way if you want to use simple returns
simpleReturns = close_prices.pct_change()
accumulatedSimpleReturns = (1 + simpleReturns).cumprod() - 1
print("\nAccumulated returns head (from simple returns):")
print(accumulatedSimpleReturns.head())

# Plot the accumulated returns from log returns
plt.figure(figsize=(12, 6))
accumulatedReturns.plot(title="Accumulated Returns (from log returns)")
plt.grid(True)
plt.ylabel("Return (%)")
plt.show()

# Optional: Plot the accumulated returns from simple returns
plt.figure(figsize=(12, 6))
accumulatedSimpleReturns.plot(title="Accumulated Returns (from simple returns)")
plt.grid(True)
plt.ylabel("Return (%)")
plt.show()