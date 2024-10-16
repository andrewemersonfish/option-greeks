# schemas.py
from pydantic import BaseModel
from typing import List, Dict, Any

class OptionGreeksRequest(BaseModel):
    option_contract: str

class OptionGreeksResponse(BaseModel):
    delta_coefficient: float
    delta_error_metrics: Dict[str, Any]
    theta_coefficient_per_hour: float
    theta_percentage_per_hour: float
    model_coefficients: Dict[str, float]
    intercept: float
    mean_absolute_error: float
    root_mean_squared_error: float
    r_squared: float
    data_preview: List[Dict[str, Any]]
    premium_ratio: float
    estimated_premium: float
    snapshot_data: Dict[str, Any]
