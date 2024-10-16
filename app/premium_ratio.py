import pandas as pd
import numpy as np
from scipy import stats
from polygon import RESTClient
from datetime import datetime, timedelta
from utils import parse_option_ticker, get_polygon_client

def analyze_option_premiums(client: RESTClient, underlying_symbol: str, expiration_date: str, option_type: str) -> dict:
    """
    Analyze option premiums and estimate ratios for OTM options using linear trend from ITM options.
    
    :param client: Polygon API client
    :param underlying_symbol: Underlying ticker symbol
    :param expiration_date: Expiration date as string in format 'YYYY-MM-DD'
    :param option_type: 'call' or 'put'
    :return: Dictionary with average premium ratio and estimated premium
    """
    # Convert expiration_date to datetime if it's a string
    if isinstance(expiration_date, str):
        expiration_date = datetime.strptime(expiration_date, '%Y-%m-%d')

    # Fetch underlying price data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Fetch last 30 days of data

    underlying_aggs = client.get_aggs(
        ticker=underlying_symbol,
        multiplier=1,
        timespan="day",
        from_=start_date.strftime("%Y-%m-%d"),
        to=end_date.strftime("%Y-%m-%d")
    )

    underlying_prices = {
        pd.to_datetime(agg.timestamp, unit='ms'): agg.close
        for agg in underlying_aggs
    }

    # Get the most recent (current) price of the underlying asset
    current_price = underlying_aggs[-1].close if underlying_aggs else None

    if current_price is None:
        return {
            'average_premium_ratio': np.nan,
            'average_premium': np.nan,
            'data': pd.DataFrame()
        }

    data = []

    # Define a range of strike prices around the current price
    strike_prices = [current_price * (1 + i * 0.05) for i in range(-5, 6)]  # Adjust range as needed

    # Fetch option data for multiple strike prices
    for strike in strike_prices:
        # Generate option ticker for each strike
        option_ticker = f"O:{underlying_symbol}{expiration_date.strftime('%y%m%d')}{option_type.upper()}{int(strike*1000):08d}"

        # Fetch option price data
        option_aggs = client.get_aggs(
            ticker=option_ticker,
            multiplier=1,
            timespan="day",
            from_=start_date.strftime("%Y-%m-%d"),
            to=end_date.strftime("%Y-%m-%d")
        )

        # Combine data
        for agg in option_aggs:
            date = pd.to_datetime(agg.timestamp, unit='ms')
            if date in underlying_prices:
                data.append({
                    'date': date,
                    'underlying_price': underlying_prices[date],
                    'option_price': agg.close,
                    'strike': strike,
                    'type': option_type.lower()
                })

    options_df = pd.DataFrame(data)

    # Proceed if options_df is not empty
    if options_df.empty:
        return {
            'average_premium_ratio': np.nan,
            'average_premium': np.nan,
            'data': pd.DataFrame()
        }

    # Calculate intrinsic value
    options_df['intrinsic_value'] = np.where(
        options_df['type'] == 'call',
        np.maximum(options_df['underlying_price'] - options_df['strike'], 0),
        np.maximum(options_df['strike'] - options_df['underlying_price'], 0)
    )

    # Calculate extrinsic value and premium ratio
    options_df['extrinsic_value'] = options_df['option_price'] - options_df['intrinsic_value']
    options_df['premium_ratio'] = options_df['option_price'] / options_df['intrinsic_value']

    # Separate ITM and OTM options
    itm_mask = options_df['intrinsic_value'] > 0
    itm_options = options_df[itm_mask]
    otm_options = options_df[~itm_mask]

    # Calculate average premium ratio for ITM options
    avg_premium_ratio_itm = itm_options['premium_ratio'].mean()

    # Estimate premium ratio for OTM options using linear regression
    if not itm_options.empty and not otm_options.empty:
        X = itm_options[['strike']]
        y = itm_options['premium_ratio']
        model = stats.linregress(X['strike'], y)

        estimated_ratios = model.slope * otm_options['strike'] + model.intercept
        estimated_premiums = estimated_ratios * otm_options['intrinsic_value']

        avg_estimated_premium_otm = estimated_premiums.mean()
    else:
        avg_estimated_premium_otm = np.nan

    # Combine ITM and OTM results
    total_options = len(options_df)
    itm_weight = len(itm_options) / total_options if total_options > 0 else 0
    otm_weight = len(otm_options) / total_options if total_options > 0 else 0

    avg_premium_ratio = (avg_premium_ratio_itm * itm_weight +
                         (avg_estimated_premium_otm / otm_options['intrinsic_value'].mean()) * otm_weight)

    avg_premium = (itm_options['option_price'].mean() * itm_weight +
                   avg_estimated_premium_otm * otm_weight)

    return {
        'average_premium_ratio': avg_premium_ratio,
        'average_premium': avg_premium,
        'data': options_df
    }
