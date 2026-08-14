import json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Islamabad Housing Price Predictor")

try:
    coeff = np.load("model_coeff.npy")
    x_mean = np.load("x_mean.npy")
    x_std = np.load("x_std.npy")
    with open("feature_columns.json") as f:
        FEATURE_COLUMNS = json.load(f)
except FileNotFoundError as e:
    raise RuntimeError(
        "Model artifacts not found. Run the notebook's 'save artifacts' "
        "cell first, and put the .npy/.json files next to api.py."
    ) from e

NUMERIC_COLUMNS = ["Bedrooms", "Bathrooms", "Area_sqft"]
LOCATION_COLUMNS = [c for c in FEATURE_COLUMNS if c not in NUMERIC_COLUMNS]


def area_to_sqft(value: float, unit: str) -> float:
    unit = unit.strip().lower()
    if unit == "marla":
        return value * 272.251
    elif unit == "kanal":
        return value * 5445
    elif unit in ("sqft", "sq.ft", "sq. ft", "sq ft"):
        return value
    raise HTTPException(status_code=422, detail=f"Unknown area unit: {unit}")


def refined_location(loc: str)-> str:
    loc = str(loc).lower()
    for i in range(5, 13):
        tag = f"f-{i}"
        if tag in loc:
            for segment in loc.split(','):
                if tag in segment:
                    return segment.strip().upper()
    for i in range(6, 15):
        tag = f"g-{i}"
        if tag in loc:
            for segment in loc.split(','):
                if tag in segment:
                    return segment.strip().upper()
    elif_checks = {'dha': 'DHA', 'bahria': 'Bahria', 'gulberg': 'Gulberg',
                   'soan': 'Soan_Garden', 'pwd': 'PWD', 'naval': 'Naval_Anchorage'}
    for key, label in elif_checks.items():
        if key in loc:
            return label
    return 'Other'


class HouseFeatures(BaseModel):
    bedrooms: int = Field(..., ge=0, le=20, example=4)
    bathrooms: int = Field(..., ge=0, le=20, example=3)
    area_value: float = Field(..., gt=0, example=10)
    area_unit: str = Field(..., example="Marla", description="Marla, Kanal, or Sq.Ft")
    location: str = Field(..., example="F-7, Islamabad")


class PredictionResponse(BaseModel):
    predicted_price_pkr: float
    location_category_used: str
    area_sqft: float


@app.get("/")
def root():
    return {"message": "Islamabad Housing Price Predictor is running. See /docs."}


@app.post("/predict", response_model=PredictionResponse)
def predict(house: HouseFeatures):
    area_sqft = area_to_sqft(house.area_value, house.area_unit)
    loc_category = refined_location(house.location)
    if loc_category not in LOCATION_COLUMNS:
        # Anything the model never saw a dummy column for falls back to "Other"
        loc_category = "Other"

    one_hot = {col: 0 for col in LOCATION_COLUMNS}
    one_hot[loc_category] = 1

    x = np.array(
        [house.bedrooms, house.bathrooms, area_sqft]
        + [one_hot[c] for c in LOCATION_COLUMNS],
        dtype=float,
    )

    x_norm = (x - x_mean) / x_std
    x_bias = np.append(x_norm, 1.0)  # same bias trick as training

    predicted_price = float(np.dot(x_bias, coeff))

    return PredictionResponse(
        predicted_price_pkr=round(predicted_price, 2),
        location_category_used=loc_category,
        area_sqft=round(area_sqft, 2),
    )
