# Time Series Exercise 6: Exponential Smoothing Methods

## Name: PREM PRASANTH J

## Roll No: 2305001028

## Department: CSE

## AIM

To implement and compare three exponential smoothing techniques for time-series forecasting:

1. Simple Exponential Smoothing
2. Holt's Linear Trend Method
3. Holt-Winters Seasonal Method

## PROGRAM

### Python Code

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing,
    Holt,
    ExponentialSmoothing
)

# 1. Simple Exponential Smoothing
base_kwh = [
    500, 510, 505, 515, 520, 518, 525,
    530, 528, 535, 540, 538, 545, 550
]

data1 = pd.Series(base_kwh)

model1 = SimpleExpSmoothing(data1)
fit1 = model1.fit(smoothing_level=0.3, optimized=False)
forecast1 = fit1.forecast(3)

print("1. Electricity Base Usage Forecast (SES):")
print(forecast1)

# 2. Holt's Linear Trend Method
trend_kwh = [
    400, 420, 435, 450, 470, 490, 510,
    530, 550, 570, 590, 610, 630, 650
]

data2 = pd.Series(trend_kwh)

model2 = Holt(data2)
fit2 = model2.fit()
forecast2 = fit2.forecast(3)

print("\n2. Electricity Growing Usage Forecast (Holt):")
print(forecast2)

# 3. Holt-Winters Seasonal Method
weekly_kwh = [
    500, 520, 510, 530, 580, 650, 640,
    510, 535, 520, 545, 595, 670, 660,
    525, 545, 535, 560, 610, 685, 675
]

data3 = pd.Series(weekly_kwh)

model3 = ExponentialSmoothing(
    data3,
    trend="add",
    seasonal="add",
    seasonal_periods=7
)

fit3 = model3.fit()
forecast3 = fit3.forecast(7)

print("\n3. Weekly Electricity Usage Forecast (Holt-Winters):")
print(forecast3)

# Plot the forecasts
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(data1, label="Actual Usage")
plt.plot(
    range(len(data1), len(data1) + 3),
    forecast1,
    marker='o',
    label="SES Forecast"
)
plt.title("Simple Exponential Smoothing")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(data2, label="Actual Usage")
plt.plot(
    range(len(data2), len(data2) + 3),
    forecast2,
    marker='o',
    label="Holt Forecast"
)
plt.title("Holt's Linear Trend Method")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(data3, label="Actual Usage")
plt.plot(
    range(len(data3), len(data3) + 7),
    forecast3,
    marker='o',
    label="Holt-Winters Forecast"
)
plt.title("Holt-Winters Seasonal Method")
plt.legend()

plt.tight_layout()
plt.show()
```

## OUTPUT

<img width="536" height="374" alt="image" src="https://github.com/user-attachments/assets/2cce7ae9-00e2-479b-852a-090b5960748b" />


## RESULT

Thus, the three exponential smoothing techniques—Simple Exponential Smoothing, Holt's Linear Trend Method, and Holt-Winters Seasonal Method—were successfully implemented and used to forecast electricity usage.
