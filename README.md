# Stock Forecasting Dashboard

A full end-to-end **Stock Forecasting Project** using **LSTM** and **ARIMA**, built with **Streamlit** for visualization and **FastAPI** for backend integration.  
This project predicts future stock prices and allows users to interactively view forecasts and download results.

---

## Features

- Fetches live historical data from **Yahoo Finance**
- Implements **LSTM** for deep learning–based forecasting
- Includes **ARIMA** model for baseline comparison
- Visualizes actual vs predicted trends
- Offers **downloadable forecast CSV**
- Includes **FastAPI backend** for programmatic access

---

## 📂 Project Structure
```
├── app.py # Streamlit dashboard
├── api.py # FastAPI backend
├── lstm_model_aapl.h5 # Pretrained LSTM model
├── Process_Documentation.md #  process documentation
├── README.md # Project overview and instructions
├── requirements.txt # Required dependencies
├── stock_forecasting.ipynb # Full experimentation and model training

```

---

##  Models Used
- **ARIMA(5,1,0):** Classical statistical model
- **Prophet:** tested for trend/uncertainty (optional).
- **LSTM:** Deep learning model trained on historical closing prices

---


## SetUp Instructions

1. **Clone the repository**  
   ```bash
   git clone <https://github.com/ayu-collab/stock-forecast-dashboard.git>
   cd stock-forecast-dashboard
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
3️. Run FastAPI
   ```bash
   uvicorn api:app --reload
   ```
4. **Run the Streamlit app**  
   ```bash
   streamlit run app.py
   ```

4. **Open browser**  
   streamlit run app.py
   Then open the displayed URL (usually http://localhost:8501) in your browser.

---
##  Author

**Ayushma Devkota**  
Data Science Enthusiast | Machine Learning Learner  
devkotaaayushma08@gmail.com






