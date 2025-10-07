# api.py
from fastapi import FastAPI
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
from datetime import timedelta

app = FastAPI(title="Stock Forecast API")

TICKER = "AAPL"
LSTM_MODEL_FILE = "lstm_model_aapl.h5"

def backtest_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def forecast_lstm(model, scaler, data, days, window=60):
    seq = data[-window:].reshape(1, window, 1)
    preds = []
    for _ in range(days):
        p = model.predict(seq, verbose=0)
        seq = np.concatenate((seq[:,1:,:], p.reshape(1,1,1)), axis=1)
        preds.append(float(scaler.inverse_transform(p)[0][0]))
    return preds

@app.get("/")
def root():
    return {"message": f"Welcome to Stock Forecast API for {TICKER}"}

@app.get("/forecast")
def forecast(days: int = 30):
    results = {}

    # Download data
    df = yf.download(TICKER, start="2015-01-01")[["Close"]].asfreq("D").ffill().reset_index()
    df["Date"] = pd.to_datetime(df["Date"])

    # --- Scale for LSTM ---
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["Close"]])

    # --- LSTM Forecast ---
    try:
        lstm_model = load_model(LSTM_MODEL_FILE)
        lstm_preds = forecast_lstm(lstm_model, scaler, scaled, days)
        
        # Compute RMSE on last 30 days as evaluation
        window = 60
        X, y = [], []
        for i in range(window, len(scaled)):
            X.append(scaled[i-window:i, 0])
            y.append(scaled[i, 0])
        X, y = np.array(X), np.array(y)
        X_eval = X[-30:].reshape(-1, window, 1)
        y_eval = y[-30:]
        y_pred_eval = lstm_model.predict(X_eval, verbose=0)
        rmse_lstm = backtest_rmse(scaler.inverse_transform(y_eval.reshape(-1,1)),
                                   scaler.inverse_transform(y_pred_eval))
        
        results["lstm"] = {"forecast": lstm_preds, "rmse": float(rmse_lstm)}
    except Exception as e:
        results["lstm"] = f"LSTM failed: {e}"

    # --- ARIMA Forecast ---
    try:
        train_size = int(len(df)*0.8)
        train = df["Close"].iloc[:train_size]
        test = df["Close"].iloc[train_size:]

        arima_model = ARIMA(train, order=(5,1,0)).fit()
        arima_pred_test = arima_model.forecast(steps=len(test))
        rmse_arima = backtest_rmse(test, arima_pred_test)

        arima_pred_future = arima_model.forecast(steps=days)
        ci = arima_model.get_forecast(steps=days).conf_int(alpha=0.05)
        results["arima"] = {
            "forecast": list(arima_pred_future),
            "rmse": float(rmse_arima),
            "lower_ci": list(ci.iloc[:,0]),
            "upper_ci": list(ci.iloc[:,1])
        }
    except Exception as e:
        results["arima"] = f"ARIMA failed: {e}"

    # Dates
    future_dates = pd.date_range(df["Date"].iloc[-1]+timedelta(1), periods=days).strftime("%Y-%m-%d").tolist()
    results["dates"] = future_dates
    results["ticker"] = TICKER
    results["days"] = days

    return results
