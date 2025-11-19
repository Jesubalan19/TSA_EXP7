# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 6-10-25
### NAME : JESUBALAN A
### REG NO: 212223240060



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

data = pd.read_csv("TSLA.csv")

print(data.head())

target_col = 'Close'

if target_col not in data.columns:
    print(f"Column '{target_col}' not found! Available columns are: {list(data.columns)}")
else:
    result = adfuller(data[target_col].dropna())
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])

    x = int(0.8 * len(data))
    train_data = data.iloc[:x]
    test_data = data.iloc[x:]

    lag_order = 13

    model = AutoReg(train_data[target_col], lags=lag_order)
    model_fit = model.fit()

    plt.figure(figsize=(10, 6))
    plot_acf(data[target_col], lags=40, alpha=0.05)
    plt.title('Autocorrelation Function (ACF)')
    plt.show()

    plt.figure(figsize=(10, 6))
    plot_pacf(data[target_col], lags=40, alpha=0.05)
    plt.title('Partial Autocorrelation Function (PACF)')
    plt.show()

    predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1, dynamic=False)

    mse = mean_squared_error(test_data[target_col], predictions)
    print('Mean Squared Error (MSE):', mse)

    plt.figure(figsize=(12, 6))
    plt.plot(test_data[target_col], label='Test Data')
    plt.plot(predictions, label='Predictions', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel(target_col)
    plt.title('AR Model Predictions vs Test Data')
    plt.legend()
    plt.grid()
    plt.show()


```
### OUTPUT:


<img width="331" height="62" alt="Screenshot 2025-11-19 190754" src="https://github.com/user-attachments/assets/28ec88bc-2b4c-46c3-adb4-42d194c6d6e8" />


PACF 
<img width="737" height="563" alt="Screenshot 2025-11-19 190827" src="https://github.com/user-attachments/assets/79d4f77a-29d9-49a6-a3ed-e6eb2a8b6bf7" />

ACF
<img width="757" height="559" alt="Screenshot 2025-11-19 190856" src="https://github.com/user-attachments/assets/d48ecf93-bcac-4801-b29f-a65bd154fe56" />


FINIAL PREDICTION

<img width="1230" height="622" alt="Screenshot 2025-11-19 190928" src="https://github.com/user-attachments/assets/8289d3d5-dd9d-44b1-800c-423f5907e291" />



### RESULT:
Thus we have successfully implemented the auto regression function using python.
