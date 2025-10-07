# 🧠 Stock Price Forecasting – Process Documentation

## 1. Dataset Details
- **Source:** Yahoo Finance via `yfinance` API
- **Ticker Used:** AAPL (Apple Inc.)
- **Date Range:** 2015–2025
- **Features Used:** Date, Close price

## 2. Data Cleaning & Preparation
- Filled missing dates using forward fill.
- Converted Date to datetime format.
- Normalized closing prices using `MinMaxScaler`.
- Split data into training and testing sets for evaluation.

## 3. Feature Engineering
- Created 60-day look-back windows for LSTM.
- Used rolling averages for smoothing forecasts in visualization.

## 4. Model Development
### a. ARIMA
- Order chosen via small grid search.
- Model captures linear and autoregressive patterns.
- RMSE ≈ 66.49 (baseline performance)

### b. Prophet
- Tested in research (not deployed in Streamlit for simplicity).
- Tuned changepoint and seasonality priors.

### c. LSTM
- Deep learning sequential model with two LSTM layers and dropout.
- Captures nonlinear dependencies and long-term temporal relationships.
- RMSE ≈ 8.93 (best performance in experiments)

<img width="1681" height="743" alt="Screenshot 2025-10-07 151902" src="https://github.com/user-attachments/assets/5e160a3c-d005-47e4-b1f4-15d27436ed66" />


## 5. Evaluation
- Evaluation metric: **RMSE (Root Mean Squared Error)**
- LSTM outperformed traditional models (lower RMSE, better future trend capture).

## 6. Deployment
- **Frontend:** Streamlit app (`app.py`)
- **Backend API:** FastAPI (`api.py`)
- **Output:** Interactive forecasts, downloadable CSV, and JSON API endpoint.

