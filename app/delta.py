import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any

def calculate_delta(merged_df: pd.DataFrame, option_type: str, strike_price: float) -> Dict[str, Any]:
    """
    Calculate Delta using linear regression, incorporating additional factors.
    """
    # Prepare data
    merged_df['delta_underlying_price'] = merged_df['underlying_price'].diff()
    merged_df['delta_option_price'] = merged_df['option_price'].diff()
    merged_df['moneyness'] = merged_df['underlying_price'] / strike_price
    merged_df['volume_ratio'] = merged_df['option_volume'] / merged_df['underlying_volume']
    
    # Add day of week as a categorical variable
    merged_df['day_of_week'] = merged_df.index.dayofweek
    day_of_week_dummies = pd.get_dummies(merged_df['day_of_week'], prefix='day')
    merged_df = pd.concat([merged_df, day_of_week_dummies], axis=1)
    
    merged_df.dropna(inplace=True)

    X = merged_df[['delta_underlying_price', 'time_to_expiration', 'moneyness', 'volume_ratio'] + list(day_of_week_dummies.columns)]
    y = merged_df['delta_option_price']

    # Fit model
    model = LinearRegression()
    model.fit(X, y)

    # Predictions and error metrics
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r_squared = r2_score(y, y_pred)

    # Calculate average delta
    average_delta = model.coef_[0]

    # Calculate delta for different moneyness levels
    moneyness_levels = [0.9, 1.0, 1.1]
    delta_by_moneyness = {}
    for level in moneyness_levels:
        X_moneyness = X.copy()
        X_moneyness['moneyness'] = level
        delta_by_moneyness[level] = model.predict(X_moneyness)[0]

    results = {
        'average_delta': average_delta,
        'delta_by_moneyness': delta_by_moneyness,
        'time_to_expiration_coefficient': model.coef_[1],
        'moneyness_coefficient': model.coef_[2],
        'volume_ratio_coefficient': model.coef_[3],
        'day_of_week_coefficients': dict(zip(day_of_week_dummies.columns, model.coef_[4:])),
        'intercept': model.intercept_,
        'error_metrics': {
            'mean_absolute_error': mae,
            'mean_squared_error': mse,
            'r_squared': r_squared
        }
    }
    return results
