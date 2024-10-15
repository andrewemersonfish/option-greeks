# utils.py
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple
from polygon import RESTClient
import os
import re
from dotenv import load_dotenv

load_dotenv()

def get_polygon_client() -> RESTClient:
    """
    Initialize the Polygon RESTClient with the API key from the environment variable.
    """
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        raise ValueError("Please set the POLYGON_API_KEY environment variable.")
    return RESTClient(api_key)

def parse_option_ticker(option_ticker: str) -> Tuple[str, str, str, float]:
    """
    Parse the option ticker in Polygon format.

    Example ticker: O:TSLA210917C00700000
    """
    # Remove the 'O:' prefix if present
    if option_ticker.startswith('O:'):
        option_ticker = option_ticker[2:]

    # Pattern to match the ticker
    pattern = r'^([A-Z]+)(\d{6})([CP])(\d{8})$'
    match = re.match(pattern, option_ticker)
    if not match:
        raise ValueError("Invalid option ticker format.")

    underlying_symbol = match.group(1)
    date_str = match.group(2)
    option_type = match.group(3)
    strike_price_str = match.group(4)

    # Parse expiration date
    expiration_date = datetime.strptime(date_str, '%y%m%d').strftime('%Y-%m-%d')

    # Parse strike price (divide by 10000 to get the actual strike price)
    strike_price = int(strike_price_str) / 10000

    return underlying_symbol, expiration_date, option_type, strike_price

def fetch_underlying_data(client: RESTClient, underlying_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical data for the underlying asset using Polygon API.
    """
    aggs = client.get_aggs(ticker=underlying_symbol, multiplier=1, timespan="hour", from_=start_date, to=end_date, adjusted=True, limit=5000)
    data = []
    for agg in aggs:
        data.append({
            'timestamp': pd.to_datetime(agg.timestamp, unit='ms'),
            'underlying_price': agg.close
        })
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame()
    df.set_index('timestamp', inplace=True)
    return df

def fetch_option_data(client: RESTClient, option_ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical data for the option contract using Polygon API.
    """
    aggs = client.get_aggs(ticker=option_ticker, multiplier=1, timespan="hour", from_=start_date, to=end_date, adjusted=True, limit=5000)
    data = []
    for agg in aggs:
        data.append({
            'timestamp': pd.to_datetime(agg.timestamp, unit='ms'),
            'option_price': agg.close
        })
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame()
    df.set_index('timestamp', inplace=True)
    return df

def calculate_time_to_expiration(expiration_date: str, timestamps: pd.DatetimeIndex) -> pd.Series:
    """
    Calculate time to expiration in hours for each timestamp.
    """
    expiration_datetime = pd.to_datetime(expiration_date)
    time_to_expiration = (expiration_datetime - timestamps).total_seconds() / 3600
    return time_to_expiration