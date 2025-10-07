# app.py
import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import load_model

# --- Streamlit config ---
st.set_page_config(page_title="📈 Stock Forecast Dashboard", layout="wide")
st.title("📊 Stock Forecast Dashboard")

# --- Sidebar ---
st.sidebar.header(" Settings")
ticker = "AAPL"  # fixed ticker
st.sidebar.text(f"Ticker fixed to: {ticker}")

date_range = st.sidebar.date_input("Date Range", [date(2015,1,1), date.today()])
if len(date_range) != 2:
    st.stop()
start, end = date_range

forecast_days = st.sidebar.slider("Forecast Days", 7, 180, 30)
eval_window = st.sidebar.slider("Backtest Window (days)", 7, 60, 30)
lstm_file = st.sidebar.text_input("LSTM Model Filename", "lstm_model_aapl.h5")
allow_train = st.sidebar.checkbox("Allow Training if Missing", value=False)

if not st.sidebar.button("Run Forecast"):
    st.stop()

# --- Helper Functions ---
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end + timedelta(1))[["Close"]].asfreq("D").ffill().reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def create_sequences(values, window=60):
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i-window:i, 0])
        y.append(values[i,0])
    return np.array(X), np.array(y)

def backtest_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def forecast_lstm(model, scaler, data, days, window=60):
    seq = data[-window:].reshape(1, window, 1)
    preds = []
    for _ in range(days):
        p = model.predict(seq, verbose=0)
        seq = np.concatenate((seq[:,1:,:], p.reshape(1,1,1)), axis=1)
        preds.append(p[0,0])
    return scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()

# --- Load Data ---
data = load_data(ticker, start, end)
st.subheader(f"Historical Prices for {ticker}")
st.line_chart(data.set_index("Date")["Close"])

# --- Scale Data for LSTM ---
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data[["Close"]])

# --- Load LSTM Model ---
model_lstm = None
if os.path.exists(lstm_file):
    try:
        model_lstm = load_model(lstm_file)
        st.success(f"LSTM model loaded: `{lstm_file}`")
    except:
        st.warning("Failed to load LSTM model.")
elif allow_train:
    st.warning("No model found. Training not implemented in this version.")

# --- Forecast ---
future_dates = pd.date_range(data["Date"].iloc[-1]+timedelta(1), periods=forecast_days)
results = pd.DataFrame({"Date": future_dates})
eval_rows = []

# --- LSTM Forecast ---
if model_lstm is not None:
    try:
        preds = forecast_lstm(model_lstm, scaler, scaled, forecast_days)
        
        # Backtest RMSE using last eval_window days
        X_seq, y_seq = create_sequences(scaled)
        X_eval = X_seq[-eval_window:].reshape(-1, 60, 1)
        y_eval = y_seq[-eval_window:]
        preds_eval = model_lstm.predict(X_eval, verbose=0)
        rmse = backtest_rmse(scaler.inverse_transform(y_eval.reshape(-1,1)),
                             scaler.inverse_transform(preds_eval))
        eval_rows.append(["LSTM", float(rmse)])
        
        # Residual std for confidence
        resid_std = np.std(scaler.inverse_transform(y_eval.reshape(-1,1)) - scaler.inverse_transform(preds_eval))
        results["LSTM"] = preds
        results["LSTM_lower"] = preds - 2*resid_std
        results["LSTM_upper"] = preds + 2*resid_std
    except Exception as e:
        st.warning(f"LSTM forecast failed: {e}")

# --- ARIMA Forecast (match Colab) ---
try:
    # Use same train/test split as Colab
    train_size = int(len(data)*0.8)
    train = data.iloc[:train_size]
    test = data.iloc[train_size:]
    
    arima_model = ARIMA(train["Close"], order=(5,1,0)).fit()
    
    # Forecast test for RMSE
    arima_pred_test = arima_model.forecast(steps=len(test))
    rmse_arima = backtest_rmse(test["Close"], arima_pred_test)
    eval_rows.append(["ARIMA", float(rmse_arima)])
    
    # Forecast future for plotting
    arima_pred_future = arima_model.forecast(steps=forecast_days)
    results["ARIMA"] = arima_pred_future.values
    
    # Confidence interval
    ci = arima_model.get_forecast(steps=forecast_days).conf_int(alpha=0.05)
    results["ARIMA_lower"], results["ARIMA_upper"] = ci.iloc[:,0].values, ci.iloc[:,1].values
except Exception as e:
    st.warning(f"ARIMA forecast failed: {e}")

# --- Plot Forecast ---
st.subheader(f"{ticker} Forecast for {forecast_days} Days")
fig, ax = plt.subplots(figsize=(12,6))

ax.plot(data["Date"], data["Close"], color="black", label="Actual", linewidth=1.5)
for model, color in [("LSTM","green"), ("ARIMA","red")]:
    if model in results.columns:
        forecast = results[model]
        if model=="LSTM":
            forecast = pd.Series(forecast).rolling(3, min_periods=1).mean().values
        ax.plot(results["Date"], forecast, label=f"{model} Forecast", color=color, linewidth=1.8)
        lower = results[f"{model}_lower"]
        upper = results[f"{model}_upper"]
        ax.fill_between(results["Date"], lower, upper, color=color, alpha=0.25)

ax.set_title(f"Stock Forecast: {ticker} ({forecast_days}-Day Horizon)", fontsize=16)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Price ($)", fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, linestyle="--", alpha=0.5)
st.pyplot(fig)

# --- Evaluation ---
if eval_rows:
    st.subheader("Model Performance & Metrics")
    eval_df = pd.DataFrame(eval_rows, columns=["Model","RMSE"]).sort_values("RMSE")
    st.table(eval_df)

# --- Download Forecast ---
results["Date"] = results["Date"].dt.strftime("%Y-%m-%d")
csv = io.StringIO()
results.to_csv(csv, index=False)
st.download_button("⬇ Download Forecast CSV", csv.getvalue().encode(),
                   file_name=f"{ticker}_forecast_{forecast_days}d.csv", mime="text/csv")
