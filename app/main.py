from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from schemas import OptionGreeksRequest, OptionGreeksResponse
from utils import (
    fetch_underlying_data,
    fetch_option_data,
    parse_option_ticker,
    calculate_time_to_expiration,
    get_polygon_client
)
from delta import calculate_delta
from theta import calculate_theta

import pandas as pd
from datetime import datetime, timedelta

app = FastAPI()
templates = Jinja2Templates(directory="templates")

client = get_polygon_client()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/calculate_greeks", response_class=HTMLResponse)
async def calculate_greeks(request: Request, option_contract: str = Form(...)):
    request_data = OptionGreeksRequest(option_contract=option_contract)

    # Extract underlying symbol and expiration date
    try:
        underlying_symbol, expiration_date, option_type, strike_price = parse_option_ticker(option_contract)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Define date range (e.g., last 30 days)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=30)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Fetch data
    underlying_df = fetch_underlying_data(client, underlying_symbol, start_date_str, end_date_str)
    option_df = fetch_option_data(client, option_contract, start_date_str, end_date_str)

    if underlying_df.empty or option_df.empty:
        raise HTTPException(status_code=404, detail="Data not found for the given symbols.")

    # Merge data
    merged_df = underlying_df.join(option_df, how='inner')

    # Calculate time to expiration
    merged_df['time_to_expiration'] = calculate_time_to_expiration(expiration_date, merged_df.index)

    # Calculate Delta
    delta_results = calculate_delta(merged_df)

    # Calculate Theta
    theta_results, merged_df_with_theta = calculate_theta(merged_df)

    # Prepare data preview
    data_preview = merged_df_with_theta.head(10).reset_index().to_dict(orient='records')

    # Prepare response
    response = OptionGreeksResponse(
        delta_coefficient=delta_results['delta_coefficient'],
        delta_error_metrics=delta_results['delta_error_metrics'],
        theta_coefficient_per_hour=theta_results['theta_coefficient_per_hour'],
        theta_percentage_per_hour=theta_results['theta_percentage_per_hour'],
        model_coefficients=theta_results['model_coefficients'],
        intercept=theta_results['intercept'],
        mean_absolute_error=theta_results['mean_absolute_error'],
        root_mean_squared_error=theta_results['root_mean_squared_error'],
        r_squared=theta_results['r_squared'],
        data_preview=data_preview
    )

    return templates.TemplateResponse("index.html", {"request": request, "result": response})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ... existing code...
