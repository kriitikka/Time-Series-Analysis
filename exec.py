from method import compute_kpis, regression_analysis , arima_forecast, holt_forecast , fourier_analysis , plot_sales
import pandas as pd 

df = pd.read_csv(r"C:\Users\Lenovo\OneDrive\Desktop\timeserpro\fdata.csv")

# 2. Compute KPIs
kpis = compute_kpis(df)
print("KPIs:", kpis)

# 3. Regression
reg = regression_analysis(df, degree=1)
print("Regression R²:", reg['r2_score'])

# 4. Forecast
arima_pred = arima_forecast(df, steps=5)
holt_pred = holt_forecast(df, steps=5)
print("Forecast (ARIMA):", arima_pred)

# 5. Fourier
fft_results = fourier_analysis(df)
print("FFT Spectrum:", fft_results['spectrum'][:5])

# 6. Plot
plot_sales(df)
