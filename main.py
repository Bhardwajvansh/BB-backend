from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from prophet import Prophet

app = FastAPI()

class TimeSeriesPoint(BaseModel):
    ds: str  # Date as string e.g. '2023-01-01'
    y: float

class ForecastRequest(BaseModel):
    data: List[TimeSeriesPoint]
    periods: int = 30  # number of future days to forecast

@app.post("/forecast")
def forecast(req: ForecastRequest):
    df = pd.DataFrame([d.dict() for d in req.data])
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=req.periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']].tail(req.periods).to_dict(orient='records')
