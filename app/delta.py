# delta.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, Dict, Any

def calculate_delta(merged_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate Delta using linear regression, incorporating time to expiration.
    """
    # Prepare data
    merged_df['delta_underlying_price'] = merged_df['underlying_price'].diff()
    merged_df['delta_option_price'] = merged_df['option_price'].diff()
    merged_df.dropna(inplace=True)

    X = merged_df[['delta_underlying_price', 'time_to_expiration']]
    y = merged_df['delta_option_price']

    # Fit model
    model = LinearRegression()
    model.fit(X, y)

    # Predictions and error metrics
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r_squared = r2_score(y, y_pred)

    results = {
        'delta_coefficient': model.coef_[0],
        'time_to_expiration_coefficient': model.coef_[1],
        'delta_intercept': model.intercept_,
        'delta_error_metrics': {
            'mean_absolute_error': mae,
            'mean_squared_error': mse,
            'r_squared': r_squared
        }
    }
    return results
