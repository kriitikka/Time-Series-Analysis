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

def plot_regression(df, degree=1):
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    X = df['year'].values.reshape(-1, 1)
    y = df['sales'].values

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    y_pred = model.predict(X_poly)
    
    plt.scatter(df['year'], df['sales'], label="Actual")
    plt.plot(df['year'], y_pred, color="red", label="Regression Fit")
    plt.title("Regression Analysis")
    plt.xlabel("Year")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(True)
    plt.show()


last_year = df['year'].iloc[-1]

steps = 5   # forecast for 5 years
future_years = np.arange(last_year + 1, last_year + 1 + steps)


def plot_arima_forecast(df, steps=5):
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(df['sales'], order=(1,1,1))
    model_fit = model.fit()
    forecast_res = model_fit.get_forecast(steps=steps)
    forecast = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int()
    lower = conf_int.iloc[:,0]
    upper = conf_int.iloc[:,1]
    
    plt.plot(df['year'], df['sales'], label="History")
    plt.plot(future_years, forecast, 'r--', label="ARIMA Forecast")
    plt.fill_between(future_years, lower, upper, color='pink', alpha=0.3)  # CI
    plt.title("ARIMA Forecast")
    plt.xlabel("Year")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_holt_forecast(df, steps=5):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    
    model = ExponentialSmoothing(df['sales'], trend="add")
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    
    plt.bar(future_years, forecast, color='green', alpha=0.5, label="Holt Forecast")
    plt.title("Holt’s Linear Forecast")
    plt.xlabel("Year")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_fft_spectrum(df):
    sales = df['sales'].values
    fft_vals = np.fft.fft(sales)
    fft_freq = np.fft.fftfreq(len(sales), d=1)

    plt.bar(fft_freq, np.abs(fft_vals))
    plt.title("FFT Spectrum of Sales")
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
