from fastapi import FastAPI, HTTPException, Header, Depends
import pandas as pd
from prophet import Prophet
import numpy as np
from typing import List, Dict, Optional
from pydantic import BaseModel
import asyncio
from mangum import Mangum
import os
import logging
from aws_lambda_powertools import Logger
logger = Logger()

app = FastAPI()

# Generate a random token at startup


# Set up logging (add this near the top of your file)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

API_TOKEN = '3b14bf4897ef89cd08b10a0a5dfbc206'
print(f"Use this Auth token for API requests: {API_TOKEN}")

# Initialize Prophet model once (reused across invocations)
# Note: Prophet is not thread-safe, so this is optional but helps with cold starts.
_prophet_model = None

def verify_auth(auth: Optional[str] = Header(None)):
    if auth is None or auth != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing Auth token")

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Successful deploy-v3!"}

@app.get("/add/{num1}/{num2}")
async def add(num1: int, num2: int, auth: str = Depends(verify_auth)):
    logger.info(f"Adding numbers: {num1} + {num2}")
    return {"total": num1 + num2}

class SeriesItem(BaseModel):
    Fecha: str
    value: float

class ForecastParams(BaseModel):
    ts: int
    df: str
    sp: int
    fp: int

class InputData(BaseModel):
    parameters: ForecastParams
    series: List[dict]

async def process_forecast(request: InputData):
    logger.info("Starting forecast processing")
    parameters = request.parameters
    series = request.series
    logger.info(f"Received forecast request with parameters: {parameters}")
    
    if not series:
        logger.error("Empty series data received")
        raise HTTPException(status_code=400, detail="Series data is empty.")
    
    print(series)


    data = pd.DataFrame(series)
    data.rename(columns={"Fecha": "ds", "value": "y"}, inplace=True)

    try:
        data['ds'] = pd.to_datetime(data['ds'])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")

    data = data.groupby('ds', as_index=False).mean()
    data['y'] = np.log1p(data['y'])

    def train_model():
        model = Prophet(
            yearly_seasonality=True,
            daily_seasonality=False,
            weekly_seasonality=False
        )
        model.add_seasonality(name="yearly", period=parameters.ts, fourier_order=5)
        model.fit(data)

        future = model.make_future_dataframe(periods=parameters.fp, freq=parameters.df)
        forecast = model.predict(future)

        forecast[['yhat', 'yhat_lower', 'yhat_upper']] = np.expm1(forecast[['yhat', 'yhat_lower', 'yhat_upper']])
        forecast["yhat"] = forecast["yhat"].clip(lower=0)

        return forecast.rename(columns={'ds': 'Fecha', 'yhat': 'F'})[['Fecha', 'F']].tail(parameters.ts).to_dict(orient="records")

    loop = asyncio.get_event_loop()
    forecast_result = await loop.run_in_executor(None, train_model)
    return {"series": forecast_result}

@app.post("/forecast")
async def make_forecast(request: InputData, auth: str = Depends(verify_auth)):
    return await process_forecast(request)

from appmain import process_forecastJD
@app.post("/forecastjd")
async def process_series(request: InputData, auth: str = Depends(verify_auth)):
    print(request)
    return await process_forecastJD(request)

# Lambda handler (must be at module level)
handler = Mangum(app)