import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf


def load_data(tickers, years=5):
    end = datetime.today()
    start = end - timedelta(days=365 * years)
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            continue
        df['MA50'] = df['Close'].rolling(50).mean()
        df['MA150'] = df['Close'].rolling(150).mean()
        df['MA200'] = df['Close'].rolling(200).mean()
        df['52w_high'] = df['Close'].rolling(252).max()
        df['52w_low'] = df['Close'].rolling(252).min()
        df['VOL20'] = df['Volume'].rolling(20).mean()
        df.dropna(inplace=True)
        data[ticker] = df
    return data
