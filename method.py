import pandas as pd

df = pd.read_csv(r"C:\Users\Lenovo\OneDrive\Desktop\timeserpro\fdata.csv")

def compute_kpis(df):
    total_sales = df['sales'].sum()
    yoy_growth = df['sales'].pct_change().mean() * 100
    cagr = ((df['sales'].iloc[-1] / df['sales'].iloc[0]) ** 
            (1/(len(df)-1)) - 1) * 100
    best_year = df.loc[df['sales'].idxmax()]
    worst_year = df.loc[df['sales'].idxmin()]

    return {
        "total_sales": total_sales,
        "avg_yoy_growth_%": yoy_growth,
        "cagr_%": cagr,
        "best_year": int(best_year['year']),
        "best_sales": best_year['sales'],
        "worst_year": int(worst_year['year']),
        "worst_sales": worst_year['sales']
    }

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

def regression_analysis(df, degree=1):
    X = df['year'].values.reshape(-1, 1)
    y = df['sales'].values

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression().fit(X_poly, y)
    y_pred = model.predict(X_poly)
    r2 = r2_score(y, y_pred)

    return {
        "coefficients": model.coef_.tolist(),
        "intercept": model.intercept_,
        "r2_score": r2,
        "predicted": y_pred.tolist()
    }

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

def arima_forecast(df, steps=5):
    model = ARIMA(df['sales'], order=(1,1,1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast.tolist()

def holt_forecast(df, steps=5):
    model = ExponentialSmoothing(df['sales'], trend="add")
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast.tolist()

import numpy as np

def fourier_analysis(df):
    sales = df['sales'].values
    fft_vals = np.fft.fft(sales)
    fft_freq = np.fft.fftfreq(len(sales), d=1)  
    
    return {
        "frequencies": fft_freq.tolist(),
        "spectrum": np.abs(fft_vals).tolist()
    }

import matplotlib.pyplot as plt

def plot_sales(df):
    plt.figure(figsize=(8,5))
    plt.plot(df['year'], df['sales'], marker='o')
    plt.title("Yearly Sales")
    plt.xlabel("Year")
    plt.ylabel("Sales")
    plt.grid(True)
    plt.show()

