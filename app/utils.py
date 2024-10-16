# utils.py
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any
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
    Fetch historical data for the underlying asset using Polygon API with 1-minute bars.
    """
    aggs = client.get_aggs(ticker=underlying_symbol, multiplier=1, timespan="minute", from_=start_date, to=end_date, adjusted=True, limit=50000)
    data = []
    for agg in aggs:
        data.append({
            'timestamp': pd.to_datetime(agg.timestamp, unit='ms'),
            'underlying_price': agg.close,
            'underlying_volume': agg.volume
        })
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame()
    df.set_index('timestamp', inplace=True)
    return df

def fetch_option_data(client: RESTClient, option_ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical data for the option contract using Polygon API with 1-minute bars.
    """
    aggs = client.get_aggs(ticker=option_ticker, multiplier=1, timespan="minute", from_=start_date, to=end_date, adjusted=True, limit=50000)
    data = []
    for agg in aggs:
        data.append({
            'timestamp': pd.to_datetime(agg.timestamp, unit='ms'),
            'option_price': agg.close,
            'option_volume': agg.volume
        })
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame()
    df.set_index('timestamp', inplace=True)
    return df

def get_option_snapshot(client: RESTClient, option_ticker: str) -> Dict[str, Any]:
    """
    Get the latest snapshot data for the option contract.
    """
    underlying_symbol, _, _, _ = parse_option_ticker(option_ticker)
    snapshot = client.get_snapshot_option(underlying_symbol, option_ticker)
    if snapshot:
        return {
            'underlying_price': snapshot.underlying_asset.price,
            'option_price': snapshot.last_trade.price if snapshot.last_trade else None,
            'greeks': {
                'delta': snapshot.greeks.delta if snapshot.greeks else None,
                'gamma': snapshot.greeks.gamma if snapshot.greeks else None,
                'theta': snapshot.greeks.theta if snapshot.greeks else None,
                'vega': snapshot.greeks.vega if snapshot.greeks else None
            }
        }
    return {}

def calculate_time_to_expiration(expiration_date: str, timestamps: pd.DatetimeIndex) -> pd.Series:
    """
    Calculate time to expiration in hours for each timestamp.
    """
    expiration_datetime = pd.to_datetime(expiration_date)
    time_to_expiration = (expiration_datetime - timestamps).total_seconds() / 3600
    return time_to_expiration
