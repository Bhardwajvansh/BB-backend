from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from prophet import Prophet
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000",         
    "https://budget-buddy-693559507420.asia-south1.run.app/",   
    "https://budget-buddy-909236466645.asia-south1.run.app",
    "https://budget-buddy-weld.vercel.app/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FinanceEntry(BaseModel):
    month: str
    revenue: float
    expenses: float

class ForecastRequest(BaseModel):
    data: List[FinanceEntry]
    periods: int

@app.post("/forecast")
def forecast(req: ForecastRequest):
    month_strs = [d.month for d in req.data]
    start_year = datetime.today().year

    month_to_num = {m: i for i, m in enumerate(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], 1)}
    base_dates = [datetime(start_year, month_to_num[m], 1) for m in month_strs]

    df = pd.DataFrame({
        "ds": base_dates,
        "revenue": [d.revenue for d in req.data],
        "expenses": [d.expenses for d in req.data],
    })

    def forecast_column(col_name):
        m = Prophet()
        m.fit(df[["ds", col_name]].rename(columns={col_name: "y"}))
        future = m.make_future_dataframe(periods=req.periods, freq='MS')
        forecast = m.predict(future)
        return forecast[["ds", "yhat"]].tail(req.periods)

    revenue_forecast = forecast_column("revenue")
    expenses_forecast = forecast_column("expenses")

    result = []
    for i in range(req.periods):
        forecast_date = revenue_forecast.iloc[i]["ds"]
        result.append({
            "month": forecast_date.strftime("%b"),
            "revenue": round(revenue_forecast.iloc[i]["yhat"]),
            "expenses": round(expenses_forecast.iloc[i]["yhat"])
        })

    return result
