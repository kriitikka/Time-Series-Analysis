Key Objectives
- Clean and preprocess yearly sales data.
- Extract KPIs (growth rate, YoY change, moving averages).
- Perform regression analysis to capture long-term trends.
- Build time series forecasting models:
  ARIMA , Holt’s Linear Trend
- Explore advanced decomposition and seasonal patterns using **Fourier Transform**.
- Visualize results with plots for better interpretation.
Tech Stack
Python:  pandas, numpy, matplotlib, seaborn, statsmodels, scipy

 Analysis Steps

### 1. KPI Analysis
- Calculate **Year-over-Year (YoY)** growth.
- Compute **CAGR** and rolling means.
- Visualize sales trends with line and bar plots.

### 2. Regression Analysis
- Fit linear/polynomial regression models to capture trend.
- Evaluate with R² and residual analysis.

### 3. ARIMA Forecast
- Perform stationarity tests (ADF).
- Apply **ARIMA(p,d,q)** for forecasting.
- Plot forecast vs actual sales.

### 4. Holt’s Linear Forecast
- Use **Holt’s linear trend method** for short-term prediction.
- Compare accuracy with ARIMA.

### 5. Fourier Analysis
- Apply **Fourier Transform** to identify seasonal/cyclic patterns.
- Extract dominant frequencies affecting sales.

### 6. FFT Spectrum
- Compute **Fast Fourier Transform (FFT)** of sales data.
- Plot frequency spectrum to analyze periodicities.
