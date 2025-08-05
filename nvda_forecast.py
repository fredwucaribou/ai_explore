#!/usr/bin/env python3
"""Download NVDA prices and compare time-series forecasting models."""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objs as go
import plotly.offline as pyo


def download_data(start="2023-01-01"):
    """Download daily NVDA data starting from `start`."""
    df = yf.download("NVDA", start=start, progress=False)
    return df


def add_features(df):
    """Create lag and technical indicator features."""
    df = df.copy()
    df["Return"] = df["Adj Close"].pct_change()
    df["MA5"] = df["Adj Close"].rolling(window=5).mean()
    df["MA10"] = df["Adj Close"].rolling(window=10).mean()
    df["Lag1"] = df["Adj Close"].shift(1)
    df = df.dropna()
    return df


def split_train_test(df):
    """Split dataframe into pre-2024 training and 2024 YTD testing."""
    start_2024 = datetime(datetime.now().year, 1, 1)
    train = df[df.index < start_2024]
    test = df[df.index >= start_2024]
    return train, test


def forecast_sarimax(train, test, features):
    """Forecast using SARIMAX with exogenous features."""
    model = SARIMAX(train["Adj Close"], order=(5, 1, 0), exog=train[features])
    model_fit = model.fit(disp=False)
    forecast = model_fit.predict(start=test.index[0], end=test.index[-1], exog=test[features])
    return forecast


def forecast_random_forest(train, test, features):
    """Forecast using RandomForestRegressor."""
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(train[features], train["Adj Close"])
    preds = model.predict(test[features])
    return pd.Series(preds, index=test.index)


def plot_results(test, sarimax_pred, rf_pred):
    """Save an interactive Plotly chart comparing actual vs predictions."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test.index, y=test["Adj Close"], mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(x=sarimax_pred.index, y=sarimax_pred.values, mode="lines", name="SARIMAX"))
    fig.add_trace(go.Scatter(x=rf_pred.index, y=rf_pred.values, mode="lines", name="RandomForest"))
    fig.update_layout(title="NVDA 2024 YTD Forecast vs Actual", xaxis_title="Date", yaxis_title="Adj Close")
    pyo.plot(fig, filename="nvda_forecast.html", auto_open=False)


def main():
    df = download_data()
    df = add_features(df)
    train, test = split_train_test(df)
    features = ["Lag1", "Return", "MA5", "MA10"]
    sarimax_pred = forecast_sarimax(train, test, features)
    rf_pred = forecast_random_forest(train, test, features)
    mape_sarimax = (np.abs((test["Adj Close"] - sarimax_pred) / test["Adj Close"])).mean() * 100
    mape_rf = (np.abs((test["Adj Close"] - rf_pred) / test["Adj Close"])).mean() * 100
    print(f"SARIMAX MAPE: {mape_sarimax:.2f}%")
    print(f"RandomForest MAPE: {mape_rf:.2f}%")
    winner = "SARIMAX" if mape_sarimax < mape_rf else "RandomForest"
    print(f"Winner: {winner}")
    plot_results(test, sarimax_pred, rf_pred)


if __name__ == "__main__":
    main()
