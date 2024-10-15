# theta.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Tuple, Dict, Any


def calculate_theta(merged_df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Calculate Theta as the percentage decrease in the option's value over one hour due to time decay,
    using regression analysis while including time to expiration.
    """
    # Prepare data
    merged_df['delta_underlying_price'] = merged_df['underlying_price'].diff()
    merged_df['delta_option_price'] = merged_df['option_price'].diff()
    merged_df['delta_time'] = -merged_df['time_to_expiration'].diff()
    merged_df.dropna(inplace=True)

    X = merged_df[['delta_underlying_price', 'delta_time']]
    y = merged_df['delta_option_price']

    # Fit model
    model = LinearRegression()
    model.fit(X, y)

    # Extract Theta
    theta_coefficient = model.coef_[1]
    average_option_price = merged_df['option_price'].mean()
    theta_percentage = (theta_coefficient / average_option_price) * 100

    # Predictions and error metrics
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r_squared = r2_score(y, y_pred)

    results = {
        'theta_coefficient_per_hour': theta_coefficient,
        'theta_percentage_per_hour': theta_percentage,
        'model_coefficients': dict(zip(X.columns, model.coef_)),
        'intercept': model.intercept_,
        'mean_absolute_error': mae,
        'root_mean_squared_error': rmse,
        'r_squared': r_squared,
    }

    return results, merged_df